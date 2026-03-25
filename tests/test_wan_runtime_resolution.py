from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from comfyui_spectrum_wan.config import SpectrumWanConfig
from comfyui_spectrum_wan.wan import WanSpectrumPatcher


class DummyInner:
    def __init__(self) -> None:
        self.blocks = []
        self.ref_conv = None
        self.original_calls = 0

    def forward_orig(self, *args, **kwargs):
        self.original_calls += 1
        return "orig"

    def patch_embedding(self, x):
        return x

    def condition_embedder(self, t, context, clip_fea):
        batch = t.shape[0]
        return (
            torch.zeros((batch, 1), dtype=t.dtype),
            torch.zeros((batch, 6), dtype=t.dtype),
            torch.zeros((batch, 1, 1), dtype=t.dtype),
            None,
        )

    def head(self, x, temb):
        return x

    def unpatchify(self, x, grid_sizes):
        return x


class DummyOuter:
    def __init__(self, inner: DummyInner) -> None:
        self.diffusion_model = inner


class DummyModel:
    model_name = "wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors"

    def __init__(self) -> None:
        self.model = DummyOuter(DummyInner())
        self.model_options = None

    def clone(self):
        return self


def _cfg() -> SpectrumWanConfig:
    return SpectrumWanConfig(
        backend="auto",
        blend_weight=1.0,
        degree=4,
        ridge_lambda=0.1,
        window_size=2.0,
        flex_window=0.75,
        warmup_steps=2,
        history_size=8,
        debug=True,
    ).validated()


def test_patched_wan_resolves_runtime_from_inner_model_when_call_options_are_fresh() -> None:
    patched = WanSpectrumPatcher.patch(DummyModel(), _cfg())
    inner = patched.model.diffusion_model

    sample_sigmas = torch.tensor([1.0, 0.5, 0.0], dtype=torch.float32)
    out = inner.forward_orig(
        torch.ones((1, 1, 2, 2), dtype=torch.float32),
        torch.tensor([sample_sigmas[0]], dtype=torch.float32),
        torch.zeros((1, 1, 1), dtype=torch.float32),
        transformer_options={"sample_sigmas": sample_sigmas, "cond_or_uncond": [0, 1]},
    )

    assert torch.is_tensor(out)
    assert inner.original_calls == 0
    assert inner._spectrum_wan_runtime.last_info["last_sigma"] == 1.0
