from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from comfyui_spectrum_wan.config import SpectrumWanConfig
from comfyui_spectrum_wan.handlers import resolve_handler
from comfyui_spectrum_wan.runtime import SpectrumWanRuntime


class DummyModel:
    model_name = "wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors"


class _DetachFailure:
    def __init__(self, message: str) -> None:
        self.message = message

    def detach(self):
        raise RuntimeError(self.message)


def _runtime(debug: bool) -> SpectrumWanRuntime:
    cfg = SpectrumWanConfig(
        backend="auto",
        blend_weight=1.0,
        degree=4,
        ridge_lambda=0.1,
        window_size=2.0,
        flex_window=0.75,
        warmup_steps=2,
        history_size=8,
        debug=debug,
    ).validated()
    handler = resolve_handler(cfg.backend, DummyModel())
    return SpectrumWanRuntime(cfg, handler)


def test_begin_step_debug_logs_to_stderr(capsys) -> None:
    runtime = _runtime(debug=True)
    sample_sigmas = torch.linspace(1.0, 0.0, 6)
    transformer_options = {"sample_sigmas": sample_sigmas, "cond_or_uncond": [0, 1]}

    runtime.begin_step(transformer_options, torch.tensor([sample_sigmas[0]]))

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "[Spectrum WAN]" in captured.err
    assert "backend=wan22_high_noise" in captured.err
    assert "phase=wan22_high_noise" in captured.err
    assert "step=0" in captured.err
    assert "global_step=0" in captured.err
    assert "num_steps=5" in captured.err
    assert "actual_forward=True" in captured.err


def test_begin_step_debug_false_stays_silent(capsys) -> None:
    runtime = _runtime(debug=False)
    sample_sigmas = torch.linspace(1.0, 0.0, 6)
    transformer_options = {"sample_sigmas": sample_sigmas, "cond_or_uncond": [0, 1]}

    runtime.begin_step(transformer_options, torch.tensor([sample_sigmas[0]]))

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_begin_step_debug_prefers_unready_bias_shift_over_ready_local_forecaster(capsys) -> None:
    runtime = _runtime(debug=True)
    runtime.cfg.warmup_steps = 0
    sample_sigmas = torch.linspace(1.0, 0.0, 6)
    transformer_options = {"sample_sigmas": sample_sigmas, "cond_or_uncond": [0, 1]}

    stream = runtime._stream(transformer_options)
    assert stream.forecaster is not None
    stream.forecaster.update(0, torch.ones((1, 2, 2)))
    stream.forecaster.update(1, torch.ones((1, 2, 2)) * 2)
    assert stream.forecaster.ready()

    stream.bias_shift_predictor = SimpleNamespace(
        ready=lambda: False,
        global_step=lambda step_idx: step_idx,
    )

    runtime.begin_step(transformer_options, torch.tensor([sample_sigmas[0]]))

    captured = capsys.readouterr()
    assert "actual_forward=True" in captured.err
    assert "forecast_ready=False" in captured.err


def test_schedule_signature_failure_sets_last_info_and_logs(capsys) -> None:
    runtime = _runtime(debug=True)

    assert runtime._schedule_signature({"sample_sigmas": _DetachFailure("bad sigmas")}) is None

    captured = capsys.readouterr()
    assert "schedule_signature_error=RuntimeError: bad sigmas" in captured.err
    assert runtime.last_info["schedule_signature_source"] == "error"
    assert runtime.last_info["schedule_signature_error"] == "RuntimeError: bad sigmas"


def test_sigma_key_fallback_logs_sigmas_failure_and_uses_timesteps(capsys) -> None:
    runtime = _runtime(debug=True)

    sigma = runtime.sigma_key({"sigmas": _DetachFailure("bad sigma key")}, torch.tensor([0.25]))

    captured = capsys.readouterr()
    assert sigma == 0.25
    assert "sigma_key_error(sigmas)=RuntimeError: bad sigma key" in captured.err
    assert runtime.last_info["sigma_key_source"] == "timesteps"
    assert runtime.last_info["sigma_key_error"] == "RuntimeError: bad sigma key"


def test_sigma_key_total_failure_logs_timesteps_error_and_returns_zero(capsys) -> None:
    runtime = _runtime(debug=True)

    sigma = runtime.sigma_key({"sigmas": _DetachFailure("bad sigma key")}, _DetachFailure("bad timesteps"))

    captured = capsys.readouterr()
    assert sigma == 0.0
    assert "sigma_key_error(timesteps)=RuntimeError: bad timesteps" in captured.err
    assert runtime.last_info["sigma_key_error"] == "RuntimeError: bad timesteps"


def test_reset_all_clears_transient_last_info_keys() -> None:
    runtime = _runtime(debug=False)
    runtime.last_info.update(
        {
            "runtime_missing_attrs": ["blocks"],
            "forecast_error": "RuntimeError: boom",
            "schedule_signature_source": "sample_sigmas",
            "schedule_signature_len": 5,
            "schedule_signature_error": "RuntimeError: bad sigmas",
            "sigma_key_source": "timesteps",
            "sigma_key_error": "RuntimeError: bad timesteps",
            "last_sigma": 0.25,
            "patched": True,
            "hook_target": "model.diffusion_model.forward_orig",
            "num_steps": 4,
        }
    )

    runtime.reset_all()

    assert "runtime_missing_attrs" not in runtime.last_info
    assert "forecast_error" not in runtime.last_info
    assert "schedule_signature_source" not in runtime.last_info
    assert "schedule_signature_len" not in runtime.last_info
    assert "schedule_signature_error" not in runtime.last_info
    assert "sigma_key_source" not in runtime.last_info
    assert "sigma_key_error" not in runtime.last_info
    assert runtime.last_info["last_sigma"] is None
    assert runtime.last_info["patched"] is True
    assert runtime.last_info["hook_target"] == "model.diffusion_model.forward_orig"
    assert runtime.last_info["num_steps"] == 4
