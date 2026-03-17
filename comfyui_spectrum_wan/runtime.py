from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch

from .config import SpectrumWanConfig
from .forecast import ChebyshevFeatureForecaster
from .handlers import WanBackendHandler, handler_metadata


@dataclass
class _StreamState:
    cfg: SpectrumWanConfig
    seen_sigmas: List[float] = field(default_factory=list)
    decisions_by_sigma: Dict[float, Dict[str, Any]] = field(default_factory=dict)
    curr_ws: float = 2.0
    num_consecutive_cached_steps: int = 0
    cycle_finished: bool = False
    forecasted_passes: int = 0
    actual_forward_count: int = 0
    forecaster: Optional[ChebyshevFeatureForecaster] = None

    def __post_init__(self) -> None:
        self.curr_ws = float(self.cfg.window_size)
        self.forecaster = ChebyshevFeatureForecaster(
            degree=self.cfg.degree,
            ridge_lambda=self.cfg.ridge_lambda,
            blend_weight=self.cfg.blend_weight,
            history_size=self.cfg.history_size,
            fit_chunk_size=self.cfg.fit_chunk_size,
        )

    def reset(self) -> None:
        self.seen_sigmas.clear()
        self.decisions_by_sigma.clear()
        self.curr_ws = float(self.cfg.window_size)
        self.num_consecutive_cached_steps = 0
        self.cycle_finished = False
        self.forecasted_passes = 0
        self.actual_forward_count = 0
        assert self.forecaster is not None
        self.forecaster.reset()


class SpectrumWanRuntime:
    def __init__(self, cfg: SpectrumWanConfig, handler: WanBackendHandler):
        self.cfg = cfg.validated()
        self.handler = handler
        self._last_schedule_signature = None
        self.streams: Dict[Tuple[str, Tuple[int, ...]], _StreamState] = {}
        self.run_id = 0
        self.last_info = {
            "enabled": self.cfg.enabled,
            "patched": False,
            "hook_target": None,
            "num_steps": 0,
            "last_sigma": None,
            "run_id": self.run_id,
            "config": asdict(self.cfg),
            "handler": handler_metadata(self.handler),
        }

    def _new_stream(self) -> _StreamState:
        return _StreamState(self.cfg)

    def reset_all(self) -> None:
        for stream in self.streams.values():
            stream.reset()
        self.last_info["run_id"] = self.run_id
        self.last_info["num_steps"] = self.last_info.get("num_steps", 0)
        self.last_info["handler"] = handler_metadata(self.handler)
        self.last_info["config"] = asdict(self.cfg)

    def update(self, cfg: SpectrumWanConfig, handler: WanBackendHandler) -> None:
        self.cfg = cfg.validated()
        self.handler = handler
        self.streams = {}
        self._last_schedule_signature = None
        self.run_id = 0
        self.last_info.update(
            {
                "enabled": self.cfg.enabled,
                "run_id": self.run_id,
                "config": asdict(self.cfg),
                "handler": handler_metadata(self.handler),
            }
        )

    def _schedule_signature(self, transformer_options: Dict[str, Any]):
        sample_sigmas = transformer_options.get("sample_sigmas", None)
        if sample_sigmas is None:
            return None
        try:
            vals = sample_sigmas.detach().float().cpu().flatten().tolist()
            return tuple(round(float(v), 8) for v in vals)
        except Exception:
            return None

    def _ensure_run_sync(self, transformer_options: Dict[str, Any]) -> None:
        sig = self._schedule_signature(transformer_options)
        if sig is None:
            return
        if self._last_schedule_signature is None:
            self._last_schedule_signature = sig
            self.last_info["num_steps"] = max(len(sig) - 1, 1)
            return
        if sig != self._last_schedule_signature:
            self.run_id += 1
            self._last_schedule_signature = sig
            self.last_info["num_steps"] = max(len(sig) - 1, 1)
            self.reset_all()

    def _stream_key(self, transformer_options: Dict[str, Any]) -> Tuple[str, Tuple[int, ...]]:
        cond_or_uncond = transformer_options.get("cond_or_uncond", None)
        if isinstance(cond_or_uncond, (list, tuple)):
            stream_subkey = tuple(int(x) for x in cond_or_uncond)
        else:
            stream_subkey = ()
        return (self.handler.stream_namespace(), stream_subkey)

    def _stream(self, transformer_options: Dict[str, Any]) -> _StreamState:
        key = self._stream_key(transformer_options)
        if key not in self.streams:
            self.streams[key] = self._new_stream()
        return self.streams[key]

    def num_steps(self) -> int:
        return max(int(self.last_info.get("num_steps", 0)), 1)

    def sigma_key(self, transformer_options: Dict[str, Any], timesteps: torch.Tensor) -> float:
        sigmas = transformer_options.get("sigmas", None)
        if sigmas is not None:
            try:
                return round(float(sigmas.detach().flatten()[0].item()), 8)
            except Exception:
                pass
        try:
            return round(float(timesteps.detach().flatten()[0].item()), 8)
        except Exception:
            return 0.0

    def begin_step(self, transformer_options: Dict[str, Any], timesteps: torch.Tensor) -> Dict[str, Any]:
        self._ensure_run_sync(transformer_options)
        stream = self._stream(transformer_options)
        sigma = self.sigma_key(transformer_options, timesteps)
        self.last_info["last_sigma"] = sigma

        if len(stream.seen_sigmas) >= self.num_steps() and not stream.cycle_finished:
            stream.cycle_finished = True
        if stream.cycle_finished:
            stream.reset()
        if stream.seen_sigmas and sigma == stream.seen_sigmas[0] and len(stream.seen_sigmas) > 1:
            stream.reset()

        if sigma in stream.decisions_by_sigma:
            return stream.decisions_by_sigma[sigma]

        step_idx = len(stream.seen_sigmas)
        stream.seen_sigmas.append(sigma)

        actual_forward = True
        if step_idx >= self.cfg.warmup_steps:
            actual_forward = ((stream.num_consecutive_cached_steps + 1) % max(1, int(torch.floor(torch.tensor(stream.curr_ws)).item()))) == 0

        assert stream.forecaster is not None
        if not stream.forecaster.ready():
            actual_forward = True

        if actual_forward:
            if step_idx >= self.cfg.warmup_steps:
                stream.curr_ws = round(stream.curr_ws + float(self.cfg.flex_window), 3)
            stream.num_consecutive_cached_steps = 0
            stream.actual_forward_count += 1
        else:
            stream.num_consecutive_cached_steps += 1
            stream.forecasted_passes += 1

        decision = {
            "sigma": sigma,
            "step_idx": step_idx,
            "actual_forward": actual_forward,
            "run_id": self.run_id,
            "phase_tag": self.handler.phase_tag,
            "stream_key": self._stream_key(transformer_options),
        }
        stream.decisions_by_sigma[sigma] = decision
        return decision

    def observe_feature(self, transformer_options: Dict[str, Any], step_idx: int, feature: torch.Tensor) -> None:
        stream = self._stream(transformer_options)
        assert stream.forecaster is not None
        stream.forecaster.update(step_idx, feature)

    def can_forecast(self, transformer_options: Dict[str, Any]) -> bool:
        stream = self._stream(transformer_options)
        assert stream.forecaster is not None
        return stream.forecaster.ready()

    def predict_feature(self, transformer_options: Dict[str, Any], step_idx: int) -> torch.Tensor:
        stream = self._stream(transformer_options)
        assert stream.forecaster is not None
        return stream.forecaster.predict(step_idx, self.num_steps())
