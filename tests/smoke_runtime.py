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


def main() -> None:
    cfg = SpectrumWanConfig(
        backend="auto",
        blend_weight=1.0,
        degree=4,
        ridge_lambda=0.1,
        window_size=2.0,
        flex_window=0.75,
        warmup_steps=2,
        history_size=8,
    ).validated()
    handler = resolve_handler(cfg.backend, DummyModel())
    runtime = SpectrumWanRuntime(cfg, handler)

    sample_sigmas = torch.linspace(1.0, 0.0, 6)
    transformer_options = {"sample_sigmas": sample_sigmas, "cond_or_uncond": [0, 1]}

    saw_forecast = False
    for i in range(5):
        decision = runtime.begin_step(transformer_options, torch.tensor([sample_sigmas[i]]))
        if decision["actual_forward"]:
            feat = torch.randn(1, 32, 64, dtype=torch.float16)
            runtime.observe_feature(transformer_options, i, feat)
        else:
            pred = runtime.predict_feature(transformer_options, i)
            assert pred.shape == (1, 32, 64)
            assert torch.isfinite(pred).all()
            saw_forecast = True

    assert saw_forecast
    assert handler.backend_id == "wan22_high_noise"
    print("ok")


if __name__ == "__main__":
    main()
