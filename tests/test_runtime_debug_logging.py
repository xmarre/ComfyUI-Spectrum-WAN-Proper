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


class DummyModel:
    model_name = "wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors"


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
    assert "actual_forward=True" in captured.err


def test_begin_step_debug_false_stays_silent(capsys) -> None:
    runtime = _runtime(debug=False)
    sample_sigmas = torch.linspace(1.0, 0.0, 6)
    transformer_options = {"sample_sigmas": sample_sigmas, "cond_or_uncond": [0, 1]}

    runtime.begin_step(transformer_options, torch.tensor([sample_sigmas[0]]))

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
