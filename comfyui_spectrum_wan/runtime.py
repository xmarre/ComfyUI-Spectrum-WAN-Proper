from __future__ import annotations

from collections import OrderedDict
import math
import sys
from dataclasses import asdict, dataclass, field
from itertools import count
from typing import Any, Dict, List, Optional, Set, Tuple

import torch

from .config import SpectrumWanConfig, bias_shift_backend_supported
from .forecast import ChebyshevFeatureForecaster, _disable_compile
from .handlers import WanBackendHandler, handler_metadata

_HIGH_TO_LOW_DIRECTION = "wan22_high_noise->wan22_low_noise"
_RUN_TOKEN_KEY = "spectrum_wan_run_token"
_GLOBAL_STEP_OVERRIDE_KEY = "spectrum_wan_global_step_override"
_GLOBAL_STEP_KEY = "spectrum_wan_global_step"
_ACTIVE_NUM_STEPS_KEY = "spectrum_wan_active_num_steps"
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
    degree: int
    ridge_lambda: float
    blend_weight: float
    history_size: int
    fit_chunk_size: int
    forecaster_cache_mode: str
    low_phase_offset: int
    total_steps_hint: int
    feature_shape: torch.Size
    feature_dtype: torch.dtype
    handoff_history: List[Tuple[int, torch.Tensor]] = field(default_factory=list)
    forecaster: Optional[ChebyshevFeatureForecaster] = None
    bias_delta: Optional[torch.Tensor] = None

    @classmethod
    def from_handoff(
        cls,
        cfg: SpectrumWanConfig,
        handoff: _PublishedTransitionHandoff,
    ) -> Optional["_BiasShiftPredictor"]:
        if len(handoff.history) < 2:
            return None
        return cls(
            degree=cfg.degree,
            ridge_lambda=cfg.ridge_lambda,
            blend_weight=cfg.blend_weight,
            history_size=cfg.history_size,
            fit_chunk_size=cfg.fit_chunk_size,
            forecaster_cache_mode=cfg.forecaster_cache_mode,
            low_phase_offset=handoff.next_global_step,
            total_steps_hint=int(handoff.total_steps_hint),
            feature_shape=handoff.feature_shape,
            feature_dtype=handoff.feature_dtype,
            handoff_history=list(handoff.history),
        )

    def _ensure_forecaster(self, target_device: torch.device) -> None:
        if self.forecaster is not None:
            return
        forecaster = ChebyshevFeatureForecaster(
            degree=self.degree,
            ridge_lambda=self.ridge_lambda,
            blend_weight=self.blend_weight,
            history_size=self.history_size,
            fit_chunk_size=self.fit_chunk_size,
            forecaster_cache_mode=self.forecaster_cache_mode,
        )
        for global_step, feature in self.handoff_history:
            forecaster.update(global_step, feature.to(device=target_device, non_blocking=True))
        self.handoff_history.clear()
        self.forecaster = forecaster

    def ready(self) -> bool:
        return self.bias_delta is not None

    def global_step(self, low_step_idx: int, explicit_global_step: Optional[int] = None) -> int:
        if explicit_global_step is not None:
            return int(explicit_global_step)
        return int(self.low_phase_offset + int(low_step_idx))

    def total_steps(self, local_total_steps: int, low_step_idx: int, explicit_global_step: Optional[int] = None) -> int:
        current_global = self.global_step(low_step_idx, explicit_global_step)
        remaining_local = max(int(local_total_steps) - int(low_step_idx), 1)
        return max(int(self.total_steps_hint), int(current_global + remaining_local), 1)

    def set_bias(self, actual_feature: torch.Tensor, low_step_idx: int, explicit_global_step: Optional[int], local_total_steps: int) -> bool:
        actual = actual_feature.detach()
        if actual.shape != self.feature_shape:
            return False
        if actual.dtype != self.feature_dtype and (not actual.dtype.is_floating_point or not self.feature_dtype.is_floating_point):
            return False

        self._ensure_forecaster(actual.device)
        assert self.forecaster is not None
        global_step = self.global_step(low_step_idx, explicit_global_step)
        predicted = self.forecaster.predict(
            global_step,
            self.total_steps(local_total_steps, low_step_idx, explicit_global_step),
        )
        predicted = predicted.to(device=actual.device, dtype=actual.dtype)
        if predicted.shape != actual.shape or not torch.isfinite(predicted).all():
            self.forecaster = None
            return False

        self.feature_shape = actual.shape
        self.feature_dtype = actual.dtype
        self.bias_delta = actual - predicted
        return torch.isfinite(self.bias_delta).all().item()

    def predict(self, low_step_idx: int, explicit_global_step: Optional[int], local_total_steps: int) -> torch.Tensor:
        if self.bias_delta is None:
            raise RuntimeError("Bias-shift predictor is not initialized.")
        assert self.forecaster is not None
        global_step = self.global_step(low_step_idx, explicit_global_step)
        predicted = self.forecaster.predict(
            global_step,
            self.total_steps(local_total_steps, low_step_idx, explicit_global_step),
        )
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
    last_processed_global_step: Optional[int] = None
    global_step_override_raw: Optional[int] = None
    global_step_override_anchor: Optional[int] = None
    bias_shift_global_anchor: Optional[int] = None
    bias_shift_global_anchor_step_idx: int = 0
    global_step_override_anchor_step_idx: int = 0
    observed_step_keys: Set[Tuple[int, int]] = field(default_factory=set)

    def __post_init__(self) -> None:
        self.curr_ws = float(self.cfg.window_size)
        self.forecaster = ChebyshevFeatureForecaster(
            degree=self.cfg.degree,
            ridge_lambda=self.cfg.ridge_lambda,
            blend_weight=self.cfg.blend_weight,
            history_size=self.cfg.history_size,
            fit_chunk_size=self.cfg.fit_chunk_size,
            forecaster_cache_mode=self.cfg.forecaster_cache_mode,
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
        self.last_processed_global_step = None
        self.global_step_override_raw = None
        self.global_step_override_anchor = None
        self.bias_shift_global_anchor = None
        self.bias_shift_global_anchor_step_idx = 0
        self.global_step_override_anchor_step_idx = 0
        self.observed_step_keys.clear()
        assert self.forecaster is not None
        self.forecaster.reset()


class SpectrumWanRuntime:
    def __init__(self, cfg: SpectrumWanConfig, handler: WanBackendHandler):
        self.cfg = cfg.validated()
        self.handler = handler
        self._validate_handler_transition_mode()
        self._last_schedule_signature = None
        self.streams: Dict[Tuple[str, Tuple[int, ...]], _StreamState] = {}
        self.run_id = 0
        self._orphaned_handoff_keys: Set[Tuple[int, Tuple[int, ...], str]] = set()
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

    def _debug_log(self, message: str) -> None:
        if not self.cfg.debug:
            return
        try:
            print(message, file=sys.stderr, flush=True)
        except Exception:
            pass

    def _new_stream(self) -> _StreamState:
        return _StreamState(self.cfg)

    def _cleanup_transition_handoffs(self) -> None:
        for handoff_key in self._orphaned_handoff_keys:
            _TRANSITION_HANDOFFS.pop(handoff_key, None)
        self._orphaned_handoff_keys.clear()

        for key, stream in self.streams.items():
            if stream.run_token is not None:
                handoff_key = (int(stream.run_token), key[1], _HIGH_TO_LOW_DIRECTION)
                _TRANSITION_HANDOFFS.pop(handoff_key, None)

    def _clear_transient_last_info(self) -> None:
        transient_keys = (
            "runtime_missing_attrs",
            "forecast_error",
            "schedule_signature_source",
            "schedule_signature_len",
            "schedule_signature_error",
            "sigma_key_source",
            "sigma_key_error",
            "no_forecast_warning",
        )
        for key in transient_keys:
            self.last_info.pop(key, None)
        self.last_info["last_sigma"] = None

    def reset_all(self) -> None:
        self._cleanup_transition_handoffs()
        for stream in self.streams.values():
            stream.reset()
        self._clear_transient_last_info()
        self.last_info["run_id"] = self.run_id
        self.last_info["num_steps"] = self.last_info.get("num_steps", 0)
        self.last_info["handler"] = handler_metadata(self.handler)
        self.last_info["config"] = asdict(self.cfg)

    def update(self, cfg: SpectrumWanConfig, handler: WanBackendHandler) -> None:
        self._cleanup_transition_handoffs()
        self.cfg = cfg.validated()
        self.handler = handler
        self._validate_handler_transition_mode()
        self.streams = {}
        self._last_schedule_signature = None
        self.run_id = 0
        self._clear_transient_last_info()
        self.last_info.update(
            {
                "enabled": self.cfg.enabled,
                "run_id": self.run_id,
                "config": asdict(self.cfg),
                "handler": handler_metadata(self.handler),
            }
        )

    def _validate_handler_transition_mode(self) -> None:
        if self.cfg.transition_mode == "bias_shift" and not bias_shift_backend_supported(self.handler.backend_id):
            raise ValueError(
                "transition_mode 'bias_shift' requires backend "
                "'wan22_high_noise' or 'wan22_low_noise' after handler resolution."
            )

    def _parse_metadata_int(self, value: Any, key: str) -> int:
        if isinstance(value, bool):
            raise ValueError(f"{key} must be an integer, got {value!r}.")
        if isinstance(value, int):
            return int(value)
        if isinstance(value, float):
            if value.is_integer():
                return int(value)
            raise ValueError(f"{key} must be an integer, got {value!r}.")
        if isinstance(value, str):
            text = value.strip()
            if text:
                check = text[1:] if text[0] in "+-" else text
                if check.isdigit():
                    return int(text)
        raise ValueError(f"{key} must be an integer, got {value!r}.")

    def _schedule_signature(self, transformer_options: Dict[str, Any]):
        sample_sigmas = transformer_options.get("sample_sigmas", None)
        if sample_sigmas is None:
            self.last_info["schedule_signature_source"] = "missing"
            self.last_info.pop("schedule_signature_len", None)
            self.last_info.pop("schedule_signature_error", None)
            transformer_options["spectrum_wan_schedule_known"] = False
            return None
        try:
            vals = sample_sigmas.detach().float().cpu().flatten().tolist()
            self.last_info["schedule_signature_source"] = "sample_sigmas"
            self.last_info["schedule_signature_len"] = len(vals)
            self.last_info.pop("schedule_signature_error", None)
            transformer_options["spectrum_wan_schedule_known"] = True
            return tuple(round(float(v), 8) for v in vals)
        except Exception as exc:
            self.last_info["schedule_signature_source"] = "error"
            self.last_info.pop("schedule_signature_len", None)
            self.last_info["schedule_signature_error"] = f"{type(exc).__name__}: {exc}"
            transformer_options["spectrum_wan_schedule_known"] = False
            self._debug_log(f"[Spectrum WAN] schedule_signature_error={type(exc).__name__}: {exc}")
            return None

    def _resolve_run_token(self, transformer_options: Dict[str, Any]) -> int:
        token = transformer_options.get(_RUN_TOKEN_KEY)
        if token is None:
            token = int(next(_RUN_TOKEN_COUNTER))
            transformer_options[_RUN_TOKEN_KEY] = token
            return token
        token = self._parse_metadata_int(token, _RUN_TOKEN_KEY)
        transformer_options[_RUN_TOKEN_KEY] = token
        return token

    def _resolve_global_step_override(self, transformer_options: Dict[str, Any]) -> Optional[int]:
        override = transformer_options.get(_GLOBAL_STEP_OVERRIDE_KEY)
        if override is None:
            return None
        return self._parse_metadata_int(override, _GLOBAL_STEP_OVERRIDE_KEY)

    def _stream_subkey(self, transformer_options: Dict[str, Any]) -> Tuple[int, ...]:
        cond_or_uncond = transformer_options.get("cond_or_uncond", None)
        if isinstance(cond_or_uncond, (list, tuple)):
            return tuple(int(x) for x in cond_or_uncond)
        return ()

    def _ensure_run_sync(self, transformer_options: Dict[str, Any]) -> None:
        sig = self._schedule_signature(transformer_options)
        if sig is None:
            transformer_options.pop(_ACTIVE_NUM_STEPS_KEY, None)
            self.last_info["num_steps"] = 0
            return
        num_steps = max(len(sig) - 1, 1)
        transformer_options[_ACTIVE_NUM_STEPS_KEY] = num_steps
        if self._last_schedule_signature is None:
            self._last_schedule_signature = sig
            self.last_info["num_steps"] = num_steps
            return
        if sig != self._last_schedule_signature:
            self.run_id += 1
            self._last_schedule_signature = sig
            self.last_info["num_steps"] = num_steps
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

    def _known_num_steps(self, transformer_options: Dict[str, Any]) -> Optional[int]:
        value = transformer_options.get(_ACTIVE_NUM_STEPS_KEY)
        if value is None:
            return None
        num_steps = self._parse_metadata_int(value, _ACTIVE_NUM_STEPS_KEY)
        return num_steps if num_steps > 0 else None

    def _stream_num_steps(
        self,
        stream: _StreamState,
        transformer_options: Dict[str, Any],
        step_idx: Optional[int] = None,
    ) -> int:
        known_num_steps = self._known_num_steps(transformer_options)
        if known_num_steps is not None:
            return known_num_steps

        observed_steps = len(stream.seen_sigmas)
        if step_idx is not None:
            observed_steps = max(observed_steps, int(step_idx) + 1)

        # Some ComfyUI sampler/model paths do not expose sample_sigmas in the
        # active transformer_options. In that case the previous fallback
        # num_steps()==1 made end_step() reset the stream after every model
        # call, so the forecaster never accumulated history. Keep a moving
        # lower-bound estimate instead. The exact end-of-run cleanup is then
        # handled by the next-cycle sigma reset in begin_step().
        return max(observed_steps + 1, 2)

    def _is_tail_actual_step(self, step_idx: int, known_num_steps: Optional[int]) -> bool:
        tail_actual_steps = int(self.cfg.tail_actual_steps)
        if tail_actual_steps <= 0 or known_num_steps is None:
            return False
        tail_start = max(0, int(known_num_steps) - tail_actual_steps)
        return int(step_idx) >= tail_start

    def end_step(self, transformer_options: Dict[str, Any], step_idx: int) -> None:
        known_num_steps = self._known_num_steps(transformer_options)
        if known_num_steps is None:
            return

        if int(step_idx) + 1 < known_num_steps:
            return

        final_num_steps = known_num_steps
        key = self._stream_key(transformer_options)
        stream = self.streams.get(key)
        if stream is None:
            return
        already_finished = bool(stream.cycle_finished)
        if not already_finished and self._should_publish_bias_shift_handoff() and stream.run_token is not None:
            handoff_key = (int(stream.run_token), key[1], _HIGH_TO_LOW_DIRECTION)
            self._orphaned_handoff_keys.add(handoff_key)
        no_forecast_warning = None
        if not already_finished and self.cfg.debug and stream.forecasted_passes <= 0:
            no_forecast_warning = (
                "Spectrum WAN: no forecasted steps occurred. "
                "warmup_steps/tail_actual_steps/window_size prevented acceleration."
            )

        # Do not immediately reset/pop the stream here. WAN 2.1 can perform a
        # duplicate model call at the completed final sigma. If we clear state
        # at end_step(), that duplicate call is misclassified as a new step 0.
        # Mark completion and let begin_step() reset only when the next call
        # moves away from the final sigma or starts a new cycle.
        stream.cycle_finished = True

        self._clear_transient_last_info()
        self.last_info["run_id"] = self.run_id
        self.last_info["num_steps"] = final_num_steps
        self.last_info["handler"] = handler_metadata(self.handler)
        self.last_info["config"] = asdict(self.cfg)
        if no_forecast_warning is not None:
            self.last_info["no_forecast_warning"] = no_forecast_warning
            self._debug_log(f"[Spectrum WAN] {no_forecast_warning}")

    def sigma_key(self, transformer_options: Dict[str, Any], timesteps: torch.Tensor) -> float:
        sigmas = transformer_options.get("sigmas", None)
        self.last_info.pop("sigma_key_source", None)
        self.last_info.pop("sigma_key_error", None)
        if sigmas is not None:
            try:
                out = round(float(sigmas.detach().flatten()[0].item()), 8)
                self.last_info["sigma_key_source"] = "sigmas"
                return out
            except Exception as exc:
                self.last_info["sigma_key_error"] = f"{type(exc).__name__}: {exc}"
                self._debug_log(f"[Spectrum WAN] sigma_key_error(sigmas)={type(exc).__name__}: {exc}")
        try:
            out = round(float(timesteps.detach().flatten()[0].item()), 8)
            self.last_info["sigma_key_source"] = "timesteps"
            return out
        except Exception as exc:
            self.last_info["sigma_key_error"] = f"{type(exc).__name__}: {exc}"
            self._debug_log(f"[Spectrum WAN] sigma_key_error(timesteps)={type(exc).__name__}: {exc}")
            return 0.0

    def _bias_shift_enabled(self) -> bool:
        return self.cfg.transition_mode == "bias_shift" and self.handler.backend_id == "wan22_low_noise"

    def _should_publish_bias_shift_handoff(self) -> bool:
        return self.cfg.transition_mode == "bias_shift" and self.handler.backend_id == "wan22_high_noise"

    def _handoff_key(
        self,
        transformer_options: Dict[str, Any],
        stream: _StreamState,
    ) -> Tuple[int, Tuple[int, ...], str]:
        run_token = stream.run_token
        if run_token is None:
            run_token = self._resolve_run_token(transformer_options)
            stream.run_token = run_token
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
        boundary_step = stream.last_processed_global_step
        if boundary_step is None:
            return
        key = self._handoff_key(transformer_options, stream)

        latest_feature = stream.actual_history[-1][1]
        history = [
            (
                int(global_step),
                feature,
            )
            for global_step, feature in stream.actual_history
        ]
        handoff = _PublishedTransitionHandoff(
            run_token=key[0],
            history=history,
            next_global_step=int(boundary_step) + 1,
            total_steps_hint=max(int(boundary_step) + 1, self.num_steps()),
            feature_shape=latest_feature.shape,
            feature_dtype=latest_feature.dtype,
        )
        self._store_transition_handoff(key, handoff)

    def _consume_bias_shift_handoff(self, transformer_options: Dict[str, Any]) -> Optional[_PublishedTransitionHandoff]:
        stream = self._stream(transformer_options)
        key = self._handoff_key(transformer_options, stream)
        return _TRANSITION_HANDOFFS.pop(key, None)

    def _global_step(
        self,
        transformer_options: Dict[str, Any],
        stream: _StreamState,
        step_idx: int,
    ) -> int:
        explicit_global_step = self._resolve_global_step_override(transformer_options)
        if explicit_global_step is not None:
            if stream.global_step_override_raw != explicit_global_step:
                stream.global_step_override_raw = explicit_global_step
                stream.global_step_override_anchor = explicit_global_step
                stream.global_step_override_anchor_step_idx = int(step_idx)
            assert stream.global_step_override_anchor is not None
            return int(stream.global_step_override_anchor + (int(step_idx) - int(stream.global_step_override_anchor_step_idx)))
        stream.global_step_override_raw = None
        stream.global_step_override_anchor = None
        stream.global_step_override_anchor_step_idx = 0
        if stream.bias_shift_predictor is not None:
            anchored_global_step = int(stream.bias_shift_predictor.global_step(step_idx))
            stream.bias_shift_global_anchor = anchored_global_step
            stream.bias_shift_global_anchor_step_idx = int(step_idx)
            return anchored_global_step
        if stream.bias_shift_global_anchor is not None:
            return int(
                stream.bias_shift_global_anchor
                + (int(step_idx) - int(stream.bias_shift_global_anchor_step_idx))
            )
        return int(step_idx)

    @_disable_compile
    def begin_step(self, transformer_options: Dict[str, Any], timesteps: torch.Tensor) -> Dict[str, Any]:
        self._ensure_run_sync(transformer_options)
        stream = self._stream(transformer_options)
        sigma = self.sigma_key(transformer_options, timesteps)
        self.last_info["last_sigma"] = sigma

        known_num_steps = self._known_num_steps(transformer_options)
        if known_num_steps is not None and len(stream.seen_sigmas) >= known_num_steps:
            stream.cycle_finished = True
        if stream.cycle_finished:
            # Some WAN/Comfy sampler paths issue a duplicate model call at the
            # final sigma after the sampler has already produced the last
            # visible step. Reuse the completed stream's final decision for
            # that duplicate call instead of resetting and treating it as a
            # new step-0 run. A new cycle still resets below when the sigma
            # changes away from the completed final sigma.
            if stream.seen_sigmas and sigma == stream.seen_sigmas[-1] and sigma in stream.decisions_by_sigma:
                return stream.decisions_by_sigma[sigma]
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
                    stream.bias_shift_global_anchor = int(handoff.next_global_step)
                    stream.bias_shift_global_anchor_step_idx = int(step_idx)
                    stream.bias_shift_predictor = _BiasShiftPredictor.from_handoff(self.cfg, handoff)

        global_step = self._global_step(transformer_options, stream, step_idx)
        total_steps = self._stream_num_steps(stream, transformer_options, step_idx)
        stream.last_processed_global_step = int(global_step)
        transformer_options[_GLOBAL_STEP_KEY] = int(global_step)

        tail_actual_only = self._is_tail_actual_step(step_idx, known_num_steps)
        schedule_unknown = known_num_steps is None

        if schedule_unknown:
            actual_forward = True
        elif not tail_actual_only and step_idx >= self.cfg.warmup_steps:
            actual_forward = (
                (stream.num_consecutive_cached_steps + 1)
                % max(1, math.floor(stream.curr_ws))
            ) == 0

        has_ready_transfer = stream.bias_shift_predictor is not None and stream.bias_shift_predictor.ready()
        if tail_actual_only:
            actual_forward = True
        elif stream.bias_shift_predictor is not None and not stream.bias_shift_predictor.ready():
            actual_forward = True
        elif not has_ready_transfer:
            assert stream.forecaster is not None
            if not stream.forecaster.ready():
                actual_forward = True

        decision = {
            "sigma": sigma,
            "step_idx": step_idx,
            "global_step": global_step,
            "actual_forward": actual_forward,
            "actual_forward_requested": actual_forward,
            "run_id": self.run_id,
            "phase_tag": self.handler.phase_tag,
            "stream_key": self._stream_key(transformer_options),
            "schedule_known": known_num_steps is not None,
            "outcome_finalized": False,
        }
        forecast_ready = has_ready_transfer or (
            stream.bias_shift_predictor is None
            and stream.forecaster is not None
            and stream.forecaster.ready()
        )
        self._debug_log(
            "[Spectrum WAN] "
            f"run_id={self.run_id} "
            f"backend={self.handler.backend_id} "
            f"phase={self.handler.phase_tag} "
            f"step={step_idx} "
            f"global_step={global_step} "
            f"num_steps={total_steps} "
            f"sigma={sigma:.8f} "
            f"actual_forward={actual_forward} "
            f"curr_ws={stream.curr_ws:.3f} "
            f"schedule_known={known_num_steps is not None} "
            f"forecast_ready={forecast_ready}"
        )
        stream.decisions_by_sigma[sigma] = decision
        return decision

    def finalize_step(
        self,
        transformer_options: Dict[str, Any],
        decision: Dict[str, Any],
        *,
        actual_taken: bool,
        forecast_taken: bool,
    ) -> None:
        if decision.get("outcome_finalized"):
            return
        if bool(actual_taken) == bool(forecast_taken):
            raise ValueError("Exactly one of actual_taken or forecast_taken must be true.")

        stream = self._stream(transformer_options)
        step_idx = int(decision["step_idx"])
        requested_actual = bool(decision.get("actual_forward_requested", decision.get("actual_forward", True)))
        if actual_taken:
            if requested_actual and step_idx >= self.cfg.warmup_steps:
                stream.curr_ws = round(stream.curr_ws + float(self.cfg.flex_window), 3)
            stream.num_consecutive_cached_steps = 0
            stream.actual_forward_count += 1
            decision["actual_forward"] = True
            if not requested_actual:
                decision["actual_fallback"] = True
        else:
            stream.num_consecutive_cached_steps += 1
            stream.forecasted_passes += 1
            decision["actual_forward"] = False
            if self._should_publish_bias_shift_handoff():
                self._publish_bias_shift_handoff(transformer_options, stream)

        decision["forecast_taken"] = bool(forecast_taken)
        decision["actual_taken"] = bool(actual_taken)
        decision["outcome_finalized"] = True

    def observe_feature(
        self,
        transformer_options: Dict[str, Any],
        step_idx: int,
        feature: torch.Tensor,
        global_step: Optional[int] = None,
    ) -> None:
        stream = self._stream(transformer_options)
        observed_key = (int(self.run_id), int(step_idx))
        if observed_key in stream.observed_step_keys:
            return
        stream.observed_step_keys.add(observed_key)
        assert stream.forecaster is not None
        stream.forecaster.update(step_idx, feature)
        feature_ref = feature.detach()
        if self._should_publish_bias_shift_handoff():
            step_for_history = int(step_idx if global_step is None else global_step)
            stream.actual_history.append((step_for_history, feature_ref.to(device="cpu")))
            if len(stream.actual_history) > self.cfg.history_size:
                stream.actual_history.pop(0)
            self._publish_bias_shift_handoff(transformer_options, stream)

        if stream.bias_shift_predictor is not None:
            explicit_global_step = None if global_step is None else int(global_step)
            if not stream.bias_shift_predictor.set_bias(
                feature_ref,
                low_step_idx=int(step_idx),
                explicit_global_step=explicit_global_step,
                local_total_steps=self._stream_num_steps(stream, transformer_options, int(step_idx)),
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
                    local_total_steps=self._stream_num_steps(stream, transformer_options, int(step_idx)),
                )
            except Exception:
                stream.bias_shift_predictor = None
                raise
        assert stream.forecaster is not None
        return stream.forecaster.predict(step_idx, self._stream_num_steps(stream, transformer_options, int(step_idx)))
