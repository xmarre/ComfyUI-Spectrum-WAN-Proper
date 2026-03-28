from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from comfyui_spectrum_wan.forecast import ChebyshevFeatureForecaster


def _design(taus: torch.Tensor, degree: int) -> torch.Tensor:
    taus = taus.reshape(-1, 1).to(torch.float32)
    cols = [torch.ones((taus.shape[0], 1), device=taus.device, dtype=torch.float32)]
    if degree >= 1:
        cols.append(taus)
    for _ in range(2, degree + 1):
        cols.append(2.0 * taus * cols[-1] - cols[-2])
    return torch.cat(cols[: degree + 1], dim=1)


def _tau(step_values: torch.Tensor, total_steps: int) -> torch.Tensor:
    denom = max(int(total_steps) - 1, 1)
    return (step_values / float(denom)) * 2.0 - 1.0


def _dense_reference_prediction(
    history: list[tuple[int, torch.Tensor]],
    *,
    degree: int,
    ridge_lambda: float,
    fit_chunk_size: int,
    step_idx: int,
    total_steps: int,
) -> torch.Tensor:
    last_feat = history[-1][1]
    device = last_feat.device
    feature_dtype = last_feat.dtype
    flat_size = int(last_feat.numel())

    step_tensor = torch.tensor([s for s, _ in history], device=device, dtype=torch.float32)
    x_mat = _design(_tau(step_tensor, total_steps), degree)
    p = x_mat.shape[1]
    reg = ridge_lambda * torch.eye(p, device=device, dtype=torch.float32)
    lhs = x_mat.transpose(0, 1) @ x_mat + reg
    chol = torch.linalg.cholesky(lhs)
    xt = x_mat.transpose(0, 1)

    coeff = torch.empty((p, flat_size), device=device, dtype=feature_dtype)
    for start in range(0, flat_size, fit_chunk_size):
        end = min(start + fit_chunk_size, flat_size)
        h_chunk = torch.stack(
            [feat.reshape(-1)[start:end].to(torch.float32) for _, feat in history],
            dim=0,
        )
        xt_h_chunk = xt @ h_chunk
        coeff[:, start:end] = torch.cholesky_solve(xt_h_chunk, chol).to(feature_dtype)

    x_star = _design(
        _tau(torch.tensor([int(step_idx)], device=device, dtype=torch.float32), total_steps),
        degree,
    )
    out = torch.empty((flat_size,), device=device, dtype=feature_dtype)
    for start in range(0, flat_size, fit_chunk_size):
        end = min(start + fit_chunk_size, flat_size)
        pred_chunk = (x_star @ coeff[:, start:end].to(torch.float32)).reshape(-1)
        out[start:end] = pred_chunk.to(feature_dtype)
    return out.reshape(last_feat.shape)


def _linear_reference_prediction(history: list[tuple[int, torch.Tensor]], step_idx: int) -> torch.Tensor:
    prev_step, prev_feat = history[-2]
    last_step, last_feat = history[-1]
    dt = max(float(last_step - prev_step), 1.0)
    k = (float(step_idx) - float(last_step)) / dt
    return (
        last_feat.to(torch.float32)
        + k * (last_feat.to(torch.float32) - prev_feat.to(torch.float32))
    ).to(last_feat.dtype)


def test_low_vram_exact_prediction_matches_previous_dense_coefficient_path() -> None:
    torch.manual_seed(0)
    forecaster = ChebyshevFeatureForecaster(
        degree=4,
        ridge_lambda=0.1,
        blend_weight=1.0,
        history_size=8,
        fit_chunk_size=17,
        forecaster_cache_mode="low_vram_exact",
    )
    history: list[tuple[int, torch.Tensor]] = []
    for step_idx in range(6):
        feat = torch.randn((2, 3, 4, 5), dtype=torch.float16)
        history.append((step_idx, feat))
        forecaster.update(step_idx, feat)

    predicted = forecaster.predict(step_idx=7, total_steps=19)
    expected = _dense_reference_prediction(
        history,
        degree=4,
        ridge_lambda=0.1,
        fit_chunk_size=17,
        step_idx=7,
        total_steps=19,
    )

    assert predicted.shape == expected.shape
    assert torch.allclose(predicted, expected, atol=3e-3, rtol=1e-3)


def test_low_vram_blended_prediction_matches_previous_dense_path() -> None:
    torch.manual_seed(1)
    blend_weight = 0.35
    forecaster = ChebyshevFeatureForecaster(
        degree=3,
        ridge_lambda=0.2,
        blend_weight=blend_weight,
        history_size=8,
        fit_chunk_size=11,
        forecaster_cache_mode="low_vram_exact",
    )
    history: list[tuple[int, torch.Tensor]] = []
    for step_idx in range(5):
        feat = torch.randn((1, 5, 7), dtype=torch.float16)
        history.append((step_idx, feat))
        forecaster.update(step_idx, feat)

    predicted = forecaster.predict(step_idx=6, total_steps=17)
    cheb = _dense_reference_prediction(
        history,
        degree=3,
        ridge_lambda=0.2,
        fit_chunk_size=11,
        step_idx=6,
        total_steps=17,
    )
    lin = _linear_reference_prediction(history, step_idx=6)
    expected = ((1.0 - blend_weight) * lin.to(torch.float32) + blend_weight * cheb.to(torch.float32)).to(torch.float16)

    assert predicted.shape == expected.shape
    assert torch.allclose(predicted, expected, atol=3e-3, rtol=1e-3)


def test_legacy_dense_coeff_mode_preserves_original_cache_shape() -> None:
    forecaster = ChebyshevFeatureForecaster(
        degree=2,
        ridge_lambda=0.1,
        blend_weight=1.0,
        history_size=8,
        fit_chunk_size=13,
        forecaster_cache_mode="legacy_dense_coeff",
    )
    for step_idx in range(4):
        forecaster.update(step_idx, torch.randn((2, 3, 5), dtype=torch.float16))

    _ = forecaster.predict(step_idx=5, total_steps=11)

    cache = forecaster._fit_cache
    assert cache is not None
    assert cache.coeff is not None
    assert cache.solve_xt is None


def test_low_vram_exact_mode_keeps_small_solver_state_only() -> None:
    forecaster = ChebyshevFeatureForecaster(
        degree=2,
        ridge_lambda=0.1,
        blend_weight=1.0,
        history_size=8,
        fit_chunk_size=13,
        forecaster_cache_mode="low_vram_exact",
    )
    for step_idx in range(4):
        forecaster.update(step_idx, torch.randn((2, 3, 5), dtype=torch.float16))

    _ = forecaster.predict(step_idx=5, total_steps=11)

    cache = forecaster._fit_cache
    assert cache is not None
    assert cache.coeff is None
    assert cache.solve_xt is not None
    assert cache.solve_xt.shape == (3, 4)
