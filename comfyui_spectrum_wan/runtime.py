from __future__ import annotations

from collections import OrderedDict
from dataclasses import asdict, dataclass, field
from itertools import count
from typing import Any, Dict, List, Optional, Tuple

import torch

from .config import SpectrumWanConfig
from .forecast import ChebyshevFeatureForecaster
from .handlers import WanBackendHandler, handler_metadata

_HIGH_TO_LOW_DIRECTION = "wan22_high_noise->wan22_low_noise"
_RUN_TOKEN_KEY = "spectrum_wan_run_token"
_GLOBAL_STEP_OVERRIDE_KEY = "spectrum_wan_global_step_override"
_GLOBAL_STEP_KEY = "spectrum_wan_global_step"
_TRANSITION_HANDOFF_LIMIT = 16
_RUN_TOKEN_COUNTER = count(1)
_TRANSITION_HANDOFFS: "OrderedDict[Tuple[int, Tuple[int, ...], str], _PublishedTransitionHandoff]" = OrderedDict()


@dataclass
class _PublishedTransitionHandoff:
    run_token: int
    history: List[Tuple[int, torch.Tensor]]
    next_global_step: int
    total_steps_hint: int
    feature_shape: torch.Size
    feature_dtype: torch.dtype


@dataclass
class _BiasShiftPredictor:
    forecaster: ChebyshevFeatureForecaster
    low_phase_offset: int
    total_steps_hint: int
    feature_shape: torch.Size
    feature_dtype: torch.dtype
    bias_delta: Optional[torch.Tensor] = None

    @classmethod
    def from_handoff(
        cls,
        cfg: SpectrumWanConfig,
        handoff: _PublishedTransitionHandoff,
    ) -> Optional["_BiasShiftPredictor"]:
        forecaster = ChebyshevFeatureForecaster(
            degree=cfg.degree,
            ridge_lambda=cfg.ridge_lambda,
            blend_weight=cfg.blend_weight,
            history_size=cfg.history_size,
            fit_chunk_size=cfg.fit_chunk_size,
        )
        for global_step, feature in handoff.history:
            forecaster.update(global_step, feature)
        if not forecaster.ready():
            return None
        return cls(
            forecaster=forecaster,
            low_phase_offset=handoff.next_global_step,
            total_steps_hint=int(handoff.total_steps_hint),
            feature_shape=handoff.feature_shape,
            feature_dtype=handoff.feature_dtype,
        )

    def ready(self) -> bool:
        return self.bias_delta is not None

    def global_step(self, low_step_idx: int, explicit_global_step: Optional[int] = None) -> int:
        if explicit_global_step is not None:
            return int(explicit_global_step)
        return int(self.low_phase_offset + int(low_step_idx))

    def total_steps(self, local_total_steps: int) -> int:
        return max(int(self.total_steps_hint), int(self.low_phase_offset + local_total_steps), 1)

    def set_bias(self, actual_feature: torch.Tensor, low_step_idx: int, explicit_global_step: Optional[int], local_total_steps: int) -> bool:
        actual = actual_feature.detach()
        if actual.shape != self.feature_shape:
            return False
        if actual.dtype != self.feature_dtype and (not actual.dtype.is_floating_point or not self.feature_dtype.is_floating_point):
            return False

        global_step = self.global_step(low_step_idx, explicit_global_step)
        predicted = self.forecaster.predict(global_step, self.total_steps(local_total_steps))
        predicted = predicted.to(device=actual.device, dtype=actual.dtype)
        if predicted.shape != actual.shape or not torch.isfinite(predicted).all():
            return False

        self.feature_shape = actual.shape
        self.feature_dtype = actual.dtype
        self.bias_delta = actual - predicted
        return torch.isfinite(self.bias_delta).all().item()

    def predict(self, low_step_idx: int, explicit_global_step: Optional[int], local_total_steps: int) -> torch.Tensor:
        if self.bias_delta is None:
            raise RuntimeError("Bias-shift predictor is not initialized.")
        global_step = self.global_step(low_step_idx, explicit_global_step)
        predicted = self.forecaster.predict(global_step, self.total_steps(local_total_steps))
        predicted = predicted.to(device=self.bias_delta.device, dtype=self.feature_dtype)
        if predicted.shape != self.feature_shape:
            raise RuntimeError("Bias-shift predictor shape mismatch.")
        out = predicted + self.bias_delta
        if not torch.isfinite(out).all():
            raise RuntimeError("Bias-shift predictor produced non-finite values.")
        return out


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
    actual_history: List[Tuple[int, torch.Tensor]] = field(default_factory=list)
    bias_shift_predictor: Optional[_BiasShiftPredictor] = None
    bias_shift_attempted: bool = False
    run_token: Optional[int] = None

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
        self.actual_history.clear()
        self.bias_shift_predictor = None
        self.bias_shift_attempted = False
        self.run_token = None
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
        for key, stream in self.streams.items():
            if stream.run_token is not None:
                handoff_key = (int(stream.run_token), key[1], _HIGH_TO_LOW_DIRECTION)
                _TRANSITION_HANDOFFS.pop(handoff_key, None)
            stream.reset()
        self.last_info["run_id"] = self.run_id
        self.last_info["num_steps"] = self.last_info.get("num_steps", 0)
        self.last_info["handler"] = handler_metadata(self.handler)
        self.last_info["config"] = asdict(self.cfg)

    def update(self, cfg: SpectrumWanConfig, handler: WanBackendHandler) -> None:
        for key, stream in self.streams.items():
            if stream.run_token is not None:
                handoff_key = (int(stream.run_token), key[1], _HIGH_TO_LOW_DIRECTION)
                _TRANSITION_HANDOFFS.pop(handoff_key, None)
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

    def _resolve_run_token(self, transformer_options: Dict[str, Any]) -> Optional[int]:
        token = transformer_options.get(_RUN_TOKEN_KEY)
        if isinstance(token, int):
            return token
        if token is None:
            token = int(next(_RUN_TOKEN_COUNTER))
            transformer_options[_RUN_TOKEN_KEY] = token
            return token
        try:
            token = int(token)
        except Exception:
            return None
        transformer_options[_RUN_TOKEN_KEY] = token
        return token

    def _stream_subkey(self, transformer_options: Dict[str, Any]) -> Tuple[int, ...]:
        cond_or_uncond = transformer_options.get("cond_or_uncond", None)
        if isinstance(cond_or_uncond, (list, tuple)):
            return tuple(int(x) for x in cond_or_uncond)
        return ()

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
        return (self.handler.stream_namespace(), self._stream_subkey(transformer_options))

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

    def _bias_shift_enabled(self) -> bool:
        return self.cfg.transition_mode == "bias_shift" and self.handler.backend_id == "wan22_low_noise"

    def _should_publish_bias_shift_handoff(self) -> bool:
        return self.cfg.transition_mode == "bias_shift" and self.handler.backend_id == "wan22_high_noise"

    def _handoff_key(
        self,
        transformer_options: Dict[str, Any],
        stream: _StreamState,
    ) -> Optional[Tuple[int, Tuple[int, ...], str]]:
        run_token = stream.run_token
        if run_token is None:
            run_token = self._resolve_run_token(transformer_options)
            stream.run_token = run_token
        if run_token is None:
            return None
        return (run_token, self._stream_subkey(transformer_options), _HIGH_TO_LOW_DIRECTION)

    def _store_transition_handoff(
        self,
        key: Tuple[int, Tuple[int, ...], str],
        handoff: _PublishedTransitionHandoff,
    ) -> None:
        if key in _TRANSITION_HANDOFFS:
            _TRANSITION_HANDOFFS.pop(key)
        _TRANSITION_HANDOFFS[key] = handoff
        while len(_TRANSITION_HANDOFFS) > _TRANSITION_HANDOFF_LIMIT:
            _TRANSITION_HANDOFFS.popitem(last=False)

    def _publish_bias_shift_handoff(self, transformer_options: Dict[str, Any], stream: _StreamState) -> None:
        if len(stream.actual_history) < 2:
            return
        key = self._handoff_key(transformer_options, stream)
        if key is None:
            return

        latest_feature = stream.actual_history[-1][1]
        history = [
            (
                int(global_step),
                feature.detach().to(device="cpu"),
            )
            for global_step, feature in stream.actual_history
        ]
        handoff = _PublishedTransitionHandoff(
            run_token=key[0],
            history=history,
            next_global_step=int(stream.actual_history[-1][0]) + 1,
            total_steps_hint=max(int(stream.actual_history[-1][0]) + 1, self.num_steps()),
            feature_shape=latest_feature.shape,
            feature_dtype=latest_feature.dtype,
        )
        self._store_transition_handoff(key, handoff)

    def _consume_bias_shift_handoff(self, transformer_options: Dict[str, Any]) -> Optional[_PublishedTransitionHandoff]:
        stream = self._stream(transformer_options)
        key = self._handoff_key(transformer_options, stream)
        if key is None:
            return None
        return _TRANSITION_HANDOFFS.pop(key, None)

    def _global_step(
        self,
        transformer_options: Dict[str, Any],
        stream: _StreamState,
        step_idx: int,
    ) -> int:
        explicit_global_step = transformer_options.get(_GLOBAL_STEP_OVERRIDE_KEY)
        if explicit_global_step is not None:
            try:
                return int(explicit_global_step)
            except Exception:
                pass
        if stream.bias_shift_predictor is not None:
            return stream.bias_shift_predictor.global_step(step_idx)
        return int(step_idx)

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
        if stream.run_token is None:
            stream.run_token = self._resolve_run_token(transformer_options)

        actual_forward = True
        if self._bias_shift_enabled():
            if step_idx == 0 and not stream.bias_shift_attempted and stream.bias_shift_predictor is None:
                stream.bias_shift_attempted = True
                handoff = self._consume_bias_shift_handoff(transformer_options)
                if handoff is not None:
                    stream.bias_shift_predictor = _BiasShiftPredictor.from_handoff(self.cfg, handoff)

        global_step = self._global_step(transformer_options, stream, step_idx)
        transformer_options[_GLOBAL_STEP_KEY] = int(global_step)

        if step_idx >= self.cfg.warmup_steps:
            actual_forward = ((stream.num_consecutive_cached_steps + 1) % max(1, int(torch.floor(torch.tensor(stream.curr_ws)).item()))) == 0

        has_ready_transfer = stream.bias_shift_predictor is not None and stream.bias_shift_predictor.ready()
        if stream.bias_shift_predictor is not None and not stream.bias_shift_predictor.ready():
            actual_forward = True
        elif not has_ready_transfer:
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
            "global_step": global_step,
            "actual_forward": actual_forward,
            "run_id": self.run_id,
            "phase_tag": self.handler.phase_tag,
            "stream_key": self._stream_key(transformer_options),
        }
        stream.decisions_by_sigma[sigma] = decision
        return decision

    def observe_feature(
        self,
        transformer_options: Dict[str, Any],
        step_idx: int,
        feature: torch.Tensor,
        global_step: Optional[int] = None,
    ) -> None:
        stream = self._stream(transformer_options)
        assert stream.forecaster is not None
        stream.forecaster.update(step_idx, feature)
        feature_ref = feature.detach()
        step_for_history = int(step_idx if global_step is None else global_step)
        stream.actual_history.append((step_for_history, feature_ref))
        if len(stream.actual_history) > self.cfg.history_size:
            stream.actual_history.pop(0)

        if self._should_publish_bias_shift_handoff():
            self._publish_bias_shift_handoff(transformer_options, stream)

        if stream.bias_shift_predictor is not None:
            explicit_global_step = None if global_step is None else int(global_step)
            if not stream.bias_shift_predictor.set_bias(
                feature_ref,
                low_step_idx=int(step_idx),
                explicit_global_step=explicit_global_step,
                local_total_steps=self.num_steps(),
            ):
                stream.bias_shift_predictor = None

    def can_forecast(self, transformer_options: Dict[str, Any]) -> bool:
        stream = self._stream(transformer_options)
        if stream.bias_shift_predictor is not None and stream.bias_shift_predictor.ready():
            return True
        assert stream.forecaster is not None
        return stream.forecaster.ready()

    def predict_feature(
        self,
        transformer_options: Dict[str, Any],
        step_idx: int,
        global_step: Optional[int] = None,
    ) -> torch.Tensor:
        stream = self._stream(transformer_options)
        if stream.bias_shift_predictor is not None and stream.bias_shift_predictor.ready():
            try:
                return stream.bias_shift_predictor.predict(
                    low_step_idx=int(step_idx),
                    explicit_global_step=None if global_step is None else int(global_step),
                    local_total_steps=self.num_steps(),
                )
            except Exception:
                stream.bias_shift_predictor = None
                raise
        assert stream.forecaster is not None
        return stream.forecaster.predict(step_idx, self.num_steps())
