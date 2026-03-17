from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from comfyui_spectrum_wan.config import SpectrumWanConfig
from comfyui_spectrum_wan.handlers import resolve_handler
from comfyui_spectrum_wan.runtime import SpectrumWanRuntime


class DummyHighModel:
    model_name = "wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors"


class DummyLowModel:
    model_name = "wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors"


def _cfg(backend: str, transition_mode: str) -> SpectrumWanConfig:
    return SpectrumWanConfig(
        backend=backend,
        transition_mode=transition_mode,
        blend_weight=1.0,
        degree=2,
        ridge_lambda=0.1,
        window_size=2.0,
        flex_window=0.75,
        warmup_steps=0,
        history_size=8,
    ).validated()


def _make_runtime(backend: str, transition_mode: str) -> SpectrumWanRuntime:
    dummy_model = DummyHighModel() if backend == "wan22_high_noise" else DummyLowModel()
    return SpectrumWanRuntime(_cfg(backend, transition_mode), resolve_handler(backend, dummy_model))


def _publish_high_handoff(runtime: SpectrumWanRuntime, sigmas: torch.Tensor, run_token: int, stream_sig: list[int]) -> None:
    opts = {
        "sample_sigmas": sigmas,
        "cond_or_uncond": stream_sig,
        "spectrum_wan_run_token": run_token,
    }
    for i in range(sigmas.numel() - 1):
        decision = runtime.begin_step(opts, torch.tensor([sigmas[i]]))
        assert decision["actual_forward"]
        assert decision["global_step"] == i
        feature = torch.full((1, 4, 8), float(i + 1), dtype=torch.float32)
        runtime.observe_feature(
            opts,
            decision["step_idx"],
            feature,
            global_step=decision["global_step"],
        )


def _test_split_schedule_bias_shift() -> None:
    high_runtime = _make_runtime("wan22_high_noise", "bias_shift")
    low_runtime = _make_runtime("wan22_low_noise", "bias_shift")

    high_sigmas = torch.tensor([1.0, 0.8, 0.6], dtype=torch.float32)
    low_sigmas = torch.tensor([0.5, 0.35, 0.2, 0.1, 0.0], dtype=torch.float32)
    run_token = 101
    stream_sig = [0, 1]

    _publish_high_handoff(high_runtime, high_sigmas, run_token, stream_sig)

    low_opts = {
        "sample_sigmas": low_sigmas,
        "cond_or_uncond": stream_sig,
        "spectrum_wan_run_token": run_token,
    }

    low_start = low_runtime.begin_step(low_opts, torch.tensor([low_sigmas[0]]))
    assert low_start["actual_forward"], "bias_shift must force the first low-noise step actual"
    assert low_start["global_step"] == 2, "low-noise phase must continue from the published high-phase boundary"
    low_first_feature = torch.full((1, 4, 8), 5.0, dtype=torch.float32)
    low_runtime.observe_feature(
        low_opts,
        low_start["step_idx"],
        low_first_feature,
        global_step=low_start["global_step"],
    )
    assert low_runtime.can_forecast(low_opts)

    low_followup = low_runtime.begin_step(low_opts, torch.tensor([low_sigmas[1]]))
    assert not low_followup["actual_forward"], "bias_shift should supply the predictor on forecast-eligible low-noise steps"
    assert low_followup["global_step"] == 3
    bias_prediction = low_runtime.predict_feature(
        low_opts,
        low_followup["step_idx"],
        global_step=low_followup["global_step"],
    )
    assert bias_prediction.shape == low_first_feature.shape
    assert torch.isfinite(bias_prediction).all()

    low_refresh = low_runtime.begin_step(low_opts, torch.tensor([low_sigmas[2]]))
    assert low_refresh["actual_forward"], "bias_shift should preserve the normal scheduler cadence after initialization"


def _test_run_token_mismatch_falls_back() -> None:
    high_runtime = _make_runtime("wan22_high_noise", "bias_shift")
    low_runtime = _make_runtime("wan22_low_noise", "bias_shift")

    high_sigmas = torch.tensor([1.0, 0.8, 0.6], dtype=torch.float32)
    low_sigmas = torch.tensor([0.5, 0.35, 0.2, 0.1, 0.0], dtype=torch.float32)
    stream_sig = [3, 3]

    _publish_high_handoff(high_runtime, high_sigmas, run_token=201, stream_sig=stream_sig)

    low_opts = {
        "sample_sigmas": low_sigmas,
        "cond_or_uncond": stream_sig,
        "spectrum_wan_run_token": 202,
    }
    low_start = low_runtime.begin_step(low_opts, torch.tensor([low_sigmas[0]]))
    assert low_start["actual_forward"]
    low_runtime.observe_feature(
        low_opts,
        low_start["step_idx"],
        torch.full((1, 4, 8), 7.0, dtype=torch.float32),
        global_step=low_start["global_step"],
    )
    low_followup = low_runtime.begin_step(low_opts, torch.tensor([low_sigmas[1]]))
    assert low_followup["actual_forward"], "a different run token must not consume another run's published handoff"


def main() -> None:
    _test_split_schedule_bias_shift()
    _test_run_token_mismatch_falls_back()
    print("ok")


if __name__ == "__main__":
    main()
