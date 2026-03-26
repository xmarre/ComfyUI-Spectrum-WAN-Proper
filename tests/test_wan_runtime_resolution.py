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
from comfyui_spectrum_wan.wan import _RUNTIME_KEY, WanSpectrumPatcher


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


class DummyInnerWithForward(DummyInner):
    def __init__(self) -> None:
        super().__init__()
        self.live_calls = 0
        self.seen_time_dim_concat = None
        self.seen_transformer_options = None

    def _forward(
        self,
        x,
        timestep,
        context,
        clip_fea=None,
        time_dim_concat=None,
        transformer_options=None,
        **kwargs,
    ):
        self.live_calls += 1
        self.seen_time_dim_concat = time_dim_concat
        self.seen_transformer_options = transformer_options
        return "live"


class DummyModelWithForward(DummyModel):
    def __init__(self) -> None:
        self.model = DummyOuter(DummyInnerWithForward())
        self.model_options = None


def test_patcher_wraps__forward_as_runtime_passthrough_when_available() -> None:
    patched = WanSpectrumPatcher.patch(DummyModelWithForward(), _cfg())
    inner = patched.model.diffusion_model

    sample_sigmas = torch.tensor([1.0, 0.5, 0.0], dtype=torch.float32)
    transformer_options = {"sample_sigmas": sample_sigmas, "cond_or_uncond": [0, 1]}
    time_dim_concat = torch.ones((1, 1, 1), dtype=torch.float32)
    out = inner._forward(
        torch.ones((1, 1, 2, 2), dtype=torch.float32),
        torch.tensor([sample_sigmas[0]], dtype=torch.float32),
        torch.zeros((1, 1, 1), dtype=torch.float32),
        time_dim_concat=time_dim_concat,
        transformer_options=transformer_options,
    )

    assert out == "live"
    assert inner.live_calls == 1
    assert inner.original_calls == 0
    assert inner.seen_time_dim_concat is time_dim_concat
    assert inner.seen_transformer_options is transformer_options
    assert inner.seen_transformer_options[_RUNTIME_KEY] is inner._spectrum_wan_runtime
    assert inner._spectrum_wan__forward_wrapped is True
    assert inner._spectrum_wan_wrapped_attr == "forward_orig"
    assert inner._spectrum_wan_runtime.last_info["hook_target"] == "model.diffusion_model.forward_orig"


def test_patcher__forward_passthrough_overwrites_stale_runtime() -> None:
    patched = WanSpectrumPatcher.patch(DummyModelWithForward(), _cfg())
    inner = patched.model.diffusion_model

    stale_runtime = SpectrumWanRuntime(_cfg(), resolve_handler("auto", DummyModel()))
    sample_sigmas = torch.tensor([1.0, 0.5, 0.0], dtype=torch.float32)
    transformer_options = {
        _RUNTIME_KEY: stale_runtime,
        "sample_sigmas": sample_sigmas,
        "cond_or_uncond": [0, 1],
    }

    inner._forward(
        torch.ones((1, 1, 2, 2), dtype=torch.float32),
        torch.tensor([sample_sigmas[0]], dtype=torch.float32),
        torch.zeros((1, 1, 1), dtype=torch.float32),
        transformer_options=transformer_options,
    )

    assert inner.seen_transformer_options is transformer_options
    assert inner.seen_transformer_options[_RUNTIME_KEY] is inner._spectrum_wan_runtime
    assert inner.seen_transformer_options[_RUNTIME_KEY] is not stale_runtime


class DummyInnerLegacyForward(DummyInner):
    def __init__(self) -> None:
        super().__init__()
        self.forward_calls = 0

    def forward(self, *args, **kwargs):
        self.forward_calls += 1
        return "legacy-forward"


class DummyModelLegacyForward(DummyModel):
    def __init__(self) -> None:
        self.model = DummyOuter(DummyInnerLegacyForward())
        self.model_options = None


def test_patcher_prefers_forward_orig_over_legacy_forward_when_no__forward() -> None:
    patched = WanSpectrumPatcher.patch(DummyModelLegacyForward(), _cfg())
    inner = patched.model.diffusion_model

    assert inner._spectrum_wan_wrapped_attr == "forward_orig"
    assert inner._spectrum_wan_runtime.last_info["hook_target"] == "model.diffusion_model.forward_orig"
