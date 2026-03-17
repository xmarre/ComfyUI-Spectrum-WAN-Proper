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


class DummyTi2vModel:
    model_name = "wan2.2_ti2v_5b_fp16.safetensors"


def _expect_value_error(fn, expected_fragment: str) -> None:
    try:
        fn()
    except ValueError as exc:
        assert expected_fragment in str(exc), str(exc)
        return
    raise AssertionError("Expected ValueError")


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
        assert runtime._stream(opts).actual_history[-1][1].device.type == "cpu"


def test_split_schedule_bias_shift() -> None:
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
    predictor = low_runtime._stream(low_opts).bias_shift_predictor
    assert predictor is not None
    assert predictor.forecaster is None
    assert predictor.handoff_history
    low_first_feature = torch.full((1, 4, 8), 5.0, dtype=torch.float32)
    low_runtime.observe_feature(
        low_opts,
        low_start["step_idx"],
        low_first_feature,
        global_step=low_start["global_step"],
    )
    assert predictor.forecaster is not None
    assert not predictor.handoff_history
    assert predictor.forecaster.history[0][1].device == low_first_feature.device
    assert not low_runtime._stream(low_opts).actual_history
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


def test_run_token_mismatch_falls_back() -> None:
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


def test_forecasted_high_tail_updates_handoff_boundary() -> None:
    high_runtime = _make_runtime("wan22_high_noise", "bias_shift")
    low_runtime = _make_runtime("wan22_low_noise", "bias_shift")

    high_sigmas = torch.tensor([1.0, 0.8, 0.6, 0.4], dtype=torch.float32)
    low_sigmas = torch.tensor([0.3, 0.1, 0.0], dtype=torch.float32)
    opts = {
        "sample_sigmas": high_sigmas,
        "cond_or_uncond": [1, 0],
        "spectrum_wan_run_token": 301,
    }

    saw_forecast = False
    for i in range(high_sigmas.numel() - 1):
        decision = high_runtime.begin_step(opts, torch.tensor([high_sigmas[i]]))
        if decision["actual_forward"]:
            high_runtime.observe_feature(
                opts,
                decision["step_idx"],
                torch.full((1, 4, 8), float(i + 1), dtype=torch.float32),
                global_step=decision["global_step"],
            )
        else:
            saw_forecast = True

    assert saw_forecast, "test requires a forecasted high-noise tail"

    low_opts = {
        "sample_sigmas": low_sigmas,
        "cond_or_uncond": [1, 0],
        "spectrum_wan_run_token": 301,
    }
    low_start = low_runtime.begin_step(low_opts, torch.tensor([low_sigmas[0]]))
    assert low_start["global_step"] == 3, "low-noise phase must anchor to the processed high-phase boundary"


def test_global_step_override_advances_and_reanchors() -> None:
    high_runtime = _make_runtime("wan22_high_noise", "bias_shift")
    low_runtime = _make_runtime("wan22_low_noise", "bias_shift")

    high_sigmas = torch.tensor([1.0, 0.8, 0.6], dtype=torch.float32)
    low_sigmas = torch.tensor([0.5, 0.35, 0.2, 0.0], dtype=torch.float32)
    run_token = 401

    _publish_high_handoff(high_runtime, high_sigmas, run_token, [2, 2])

    low_opts = {
        "sample_sigmas": low_sigmas,
        "cond_or_uncond": [2, 2],
        "spectrum_wan_run_token": run_token,
        "spectrum_wan_global_step_override": 10,
    }

    low_start = low_runtime.begin_step(low_opts, torch.tensor([low_sigmas[0]]))
    assert low_start["global_step"] == 10
    low_runtime.observe_feature(
        low_opts,
        low_start["step_idx"],
        torch.full((1, 4, 8), 9.0, dtype=torch.float32),
        global_step=low_start["global_step"],
    )
    predictor = low_runtime._stream(low_opts).bias_shift_predictor
    assert predictor is not None
    assert predictor.total_steps(low_runtime.num_steps(), low_start["step_idx"], low_start["global_step"]) == 13

    low_followup = low_runtime.begin_step(low_opts, torch.tensor([low_sigmas[1]]))
    assert low_followup["global_step"] == 11
    assert predictor.total_steps(low_runtime.num_steps(), low_followup["step_idx"], low_followup["global_step"]) == 13

    low_opts["spectrum_wan_global_step_override"] = 20
    low_reanchored = low_runtime.begin_step(low_opts, torch.tensor([low_sigmas[2]]))
    assert low_reanchored["global_step"] == 20
    assert predictor.total_steps(low_runtime.num_steps(), low_reanchored["step_idx"], low_reanchored["global_step"]) == 21

    failing_high_runtime = _make_runtime("wan22_high_noise", "bias_shift")
    failing_low_runtime = _make_runtime("wan22_low_noise", "bias_shift")
    failing_run_token = 402

    _publish_high_handoff(failing_high_runtime, high_sigmas, failing_run_token, [2, 2])

    failing_opts = {
        "sample_sigmas": low_sigmas,
        "cond_or_uncond": [2, 2],
        "spectrum_wan_run_token": failing_run_token,
        "spectrum_wan_global_step_override": 30,
    }
    failing_start = failing_low_runtime.begin_step(failing_opts, torch.tensor([low_sigmas[0]]))
    assert failing_start["global_step"] == 30

    failing_stream = failing_low_runtime._stream(failing_opts)
    assert failing_stream.bias_shift_predictor is not None

    failing_low_runtime.observe_feature(
        failing_opts,
        failing_start["step_idx"],
        torch.full((1, 5, 8), 9.0, dtype=torch.float32),
        global_step=failing_start["global_step"],
    )

    assert failing_stream.bias_shift_predictor is None

    failing_followup = failing_low_runtime.begin_step(failing_opts, torch.tensor([low_sigmas[1]]))
    assert failing_followup["global_step"] == 31
    assert failing_followup["actual_forward"]


def test_unsupported_bias_shift_backends_raise() -> None:
    _expect_value_error(
        lambda: SpectrumWanConfig(backend="wan21", transition_mode="bias_shift").validated(),
        "requires backend",
    )

    cfg = SpectrumWanConfig(backend="auto", transition_mode="bias_shift").validated()
    handler = resolve_handler(cfg.backend, DummyTi2vModel())
    assert handler.backend_id == "wan22_ti2v_5b"
    _expect_value_error(
        lambda: SpectrumWanRuntime(cfg, handler),
        "after handler resolution",
    )


def test_malformed_metadata_raises() -> None:
    run_token_runtime = _make_runtime("wan22_high_noise", "separate_fit")
    _expect_value_error(
        lambda: run_token_runtime.begin_step(
            {
                "sample_sigmas": torch.tensor([1.0, 0.0], dtype=torch.float32),
                "cond_or_uncond": [0],
                "spectrum_wan_run_token": "bad-token",
            },
            torch.tensor([1.0], dtype=torch.float32),
        ),
        "spectrum_wan_run_token",
    )

    global_step_runtime = _make_runtime("wan22_high_noise", "separate_fit")
    _expect_value_error(
        lambda: global_step_runtime.begin_step(
            {
                "sample_sigmas": torch.tensor([1.0, 0.0], dtype=torch.float32),
                "cond_or_uncond": [0],
                "spectrum_wan_global_step_override": "bad-step",
            },
            torch.tensor([1.0], dtype=torch.float32),
        ),
        "spectrum_wan_global_step_override",
    )

    bool_metadata_runtime = _make_runtime("wan22_high_noise", "separate_fit")
    _expect_value_error(
        lambda: bool_metadata_runtime.begin_step(
            {
                "sample_sigmas": torch.tensor([1.0, 0.0], dtype=torch.float32),
                "cond_or_uncond": [0],
                "spectrum_wan_run_token": True,
            },
            torch.tensor([1.0], dtype=torch.float32),
        ),
        "spectrum_wan_run_token",
    )

    float_metadata_runtime = _make_runtime("wan22_high_noise", "separate_fit")
    _expect_value_error(
        lambda: float_metadata_runtime.begin_step(
            {
                "sample_sigmas": torch.tensor([1.0, 0.0], dtype=torch.float32),
                "cond_or_uncond": [0],
                "spectrum_wan_global_step_override": 10.5,
            },
            torch.tensor([1.0], dtype=torch.float32),
        ),
        "spectrum_wan_global_step_override",
    )


def main() -> None:
    test_split_schedule_bias_shift()
    test_run_token_mismatch_falls_back()
    test_forecasted_high_tail_updates_handoff_boundary()
    test_global_step_override_advances_and_reanchors()
    test_unsupported_bias_shift_backends_raise()
    test_malformed_metadata_raises()
    print("ok")


if __name__ == "__main__":
    main()
