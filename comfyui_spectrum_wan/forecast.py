from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple, TypeVar

import torch

_F = TypeVar("_F", bound=Callable[..., object])


def _disable_compile(fn: _F) -> _F:
    compiler_disable = getattr(getattr(torch, "compiler", None), "disable", None)
    if compiler_disable is not None:
        return compiler_disable(reason="Spectrum WAN forecaster should run eagerly")(fn)  # type: ignore[return-value]
    dynamo_disable = getattr(getattr(torch, "_dynamo", None), "disable", None)
    if dynamo_disable is not None:
        return dynamo_disable(fn)  # type: ignore[return-value]
    return fn


@dataclass
class _FitCache:
    coeff: Optional[torch.Tensor]
    solve_xt: Optional[torch.Tensor]
    feature_shape: torch.Size
    feature_dtype: torch.dtype
    total_steps: int


class ChebyshevFeatureForecaster:
    """
    Online Spectrum-style forecaster for WAN hidden features.

    The default cache mode preserves the original dense coefficient path.
    An optional exact low-VRAM mode avoids materializing the dense
    `(degree + 1, flat_feature_size)` coefficient tensor by caching only the
    small solver state and applying equivalent history weights chunk-by-chunk
    at prediction time.
    """

    def __init__(
        self,
        degree: int,
        ridge_lambda: float,
        blend_weight: float,
        history_size: int = 16,
        fit_chunk_size: int = 1_000_000,
        forecaster_cache_mode: str = "legacy_dense_coeff",
    ):
        self.degree = int(degree)
        self.ridge_lambda = float(ridge_lambda)
        self.blend_weight = float(blend_weight)
        self.history_size = int(history_size)
        self.fit_chunk_size = int(fit_chunk_size)
        self.forecaster_cache_mode = str(forecaster_cache_mode)
        if self.forecaster_cache_mode not in {"legacy_dense_coeff", "low_vram_exact"}:
            raise ValueError(
                f"Unsupported forecaster_cache_mode '{self.forecaster_cache_mode}'."
            )
        self.reset()

    def reset(self) -> None:
        self.history: List[Tuple[int, torch.Tensor]] = []
        self._feature_shape: Optional[torch.Size] = None
        self._feature_dtype: Optional[torch.dtype] = None
        self._feature_device: Optional[torch.device] = None
        self._flat_feature_size: int = 0
        self._fit_cache: Optional[_FitCache] = None

    def ready(self) -> bool:
        return len(self.history) >= 2

    def update(self, step_idx: int, feature: torch.Tensor) -> None:
        feat = feature.detach()
        if self._feature_shape is None:
            self._feature_shape = feat.shape
            self._feature_dtype = feat.dtype
            self._feature_device = feat.device
            self._flat_feature_size = int(feat.numel())
        elif feat.shape != self._feature_shape:
            self.reset()
            self._feature_shape = feat.shape
            self._feature_dtype = feat.dtype
            self._feature_device = feat.device
            self._flat_feature_size = int(feat.numel())

        self.history.append((int(step_idx), feat))
        if len(self.history) > self.history_size:
            self.history.pop(0)
        self._fit_cache = None

    def _tau(self, step_values: torch.Tensor, total_steps: int) -> torch.Tensor:
        # Uses actual schedule length rather than a fixed 50-step normalization.
        denom = max(int(total_steps) - 1, 1)
        return (step_values / float(denom)) * 2.0 - 1.0

    def _design(self, taus: torch.Tensor, degree: int) -> torch.Tensor:
        taus = taus.reshape(-1, 1).to(torch.float32)
        cols = [torch.ones((taus.shape[0], 1), device=taus.device, dtype=torch.float32)]
        if degree >= 1:
            cols.append(taus)
        for _ in range(2, degree + 1):
            cols.append(2.0 * taus * cols[-1] - cols[-2])
        return torch.cat(cols[: degree + 1], dim=1)

    def _history_chunk(self, start: int, end: int) -> torch.Tensor:
        return torch.stack(
            [feat.reshape(-1)[start:end].to(torch.float32) for _, feat in self.history],
            dim=0,
        )

    def _fit_if_needed(self, total_steps: int) -> None:
        if self._fit_cache is not None and self._fit_cache.total_steps == int(total_steps):
            return
        if not self.history:
            raise RuntimeError("Spectrum forecaster was asked to fit without history.")
        assert self._feature_shape is not None
        assert self._feature_dtype is not None
        assert self._feature_device is not None

        device = self._feature_device
        step_tensor = torch.tensor([s for s, _ in self.history], device=device, dtype=torch.float32)
        taus = self._tau(step_tensor, total_steps)
        x_mat = self._design(taus, self.degree)

        p = x_mat.shape[1]
        reg = self.ridge_lambda * torch.eye(p, device=device, dtype=torch.float32)
        lhs = x_mat.transpose(0, 1) @ x_mat + reg
        try:
            chol = torch.linalg.cholesky(lhs)
        except RuntimeError:
            jitter = float(lhs.diag().mean().item()) if lhs.numel() > 0 else 1.0
            jitter = max(jitter * 1e-6, 1e-6)
            chol = torch.linalg.cholesky(lhs + jitter * torch.eye(p, device=device, dtype=torch.float32))

        xt = x_mat.transpose(0, 1)
        coeff: Optional[torch.Tensor] = None
        solve_xt: Optional[torch.Tensor] = None

        if self.forecaster_cache_mode == "legacy_dense_coeff":
            flat_size = self._flat_feature_size
            coeff = torch.empty((p, flat_size), device=device, dtype=self._feature_dtype)
            for start in range(0, flat_size, self.fit_chunk_size):
                end = min(start + self.fit_chunk_size, self._flat_feature_size)
                h_chunk = self._history_chunk(start, end)
                xt_h_chunk = xt @ h_chunk
                c_chunk = torch.cholesky_solve(xt_h_chunk, chol).to(self._feature_dtype)
                coeff[:, start:end] = c_chunk
        else:
            solve_xt = torch.cholesky_solve(xt, chol)

        self._fit_cache = _FitCache(
            coeff=coeff,
            solve_xt=solve_xt,
            feature_shape=self._feature_shape,
            feature_dtype=self._feature_dtype,
            total_steps=int(total_steps),
        )

    def _predict_chebyshev_flat(self, step_idx: int, total_steps: int) -> torch.Tensor:
        assert self._fit_cache is not None
        coeff = self._fit_cache.coeff
        solve_xt = self._fit_cache.solve_xt
        if coeff is not None:
            cache_device = coeff.device
        else:
            assert solve_xt is not None
            cache_device = solve_xt.device
        tau_star = self._tau(
            torch.tensor([int(step_idx)], device=cache_device, dtype=torch.float32),
            total_steps,
        )
        x_star = self._design(tau_star, self.degree)

        if coeff is not None:
            flat_size = coeff.shape[1]
            out = torch.empty((flat_size,), device=coeff.device, dtype=self._fit_cache.feature_dtype)
            for start in range(0, flat_size, self.fit_chunk_size):
                end = min(start + self.fit_chunk_size, flat_size)
                pred_chunk = (x_star @ coeff[:, start:end].to(torch.float32)).reshape(-1)
                out[start:end] = pred_chunk.to(self._fit_cache.feature_dtype)
            return out

        assert solve_xt is not None
        weights = torch.matmul(x_star, solve_xt).reshape(-1)
        flat_size = self._flat_feature_size
        out = torch.empty((flat_size,), device=solve_xt.device, dtype=self._fit_cache.feature_dtype)
        for start in range(0, flat_size, self.fit_chunk_size):
            end = min(start + self.fit_chunk_size, flat_size)
            h_chunk = self._history_chunk(start, end)
            pred_chunk = torch.matmul(weights, h_chunk)
            out[start:end] = pred_chunk.to(self._fit_cache.feature_dtype)
        return out

    def _predict_linear_flat(self, step_idx: int) -> torch.Tensor:
        assert self.history
        _, last_feat = self.history[-1]
        if len(self.history) < 2:
            return last_feat.reshape(-1)

        prev_step, prev_feat = self.history[-2]
        last_step, last_feat = self.history[-1]
        prev_flat = prev_feat.reshape(-1)
        last_flat = last_feat.reshape(-1)

        dt = max(float(last_step - prev_step), 1.0)
        k = (float(step_idx) - float(last_step)) / dt
        return (last_flat.to(torch.float32) + k * (last_flat.to(torch.float32) - prev_flat.to(torch.float32))).to(last_feat.dtype)

    @_disable_compile
    def predict(self, step_idx: int, total_steps: int) -> torch.Tensor:
        if not self.history:
            raise RuntimeError("Spectrum forecaster was asked to predict without history.")
        if len(self.history) == 1:
            return self.history[-1][1]
        assert self._feature_shape is not None

        latest = self.history[-1][1]
        if self.blend_weight <= 0.0:
            return self._predict_linear_flat(step_idx).reshape(self._feature_shape).to(device=latest.device, dtype=latest.dtype)

        self._fit_if_needed(total_steps)

        if self.blend_weight >= 1.0:
            out_flat = self._predict_chebyshev_flat(step_idx, total_steps)
            if not torch.isfinite(out_flat).all():
                return latest
            return out_flat.reshape(self._feature_shape).to(device=latest.device, dtype=latest.dtype)

        lin_flat = self._predict_linear_flat(step_idx)
        cheb_flat = self._predict_chebyshev_flat(step_idx, total_steps)
        out_flat = ((1.0 - self.blend_weight) * lin_flat.to(torch.float32) + self.blend_weight * cheb_flat.to(torch.float32)).to(latest.dtype)
        if not torch.isfinite(out_flat).all():
            return latest
        return out_flat.reshape(self._feature_shape).to(device=latest.device, dtype=latest.dtype)
