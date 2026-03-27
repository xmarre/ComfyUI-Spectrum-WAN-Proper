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
import comfyui_spectrum_wan.wan as wan
from comfyui_spectrum_wan.wan import _RUNTIME_KEY, _run_spectrum_forward, WanSpectrumPatcher


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


class NewApiInner(DummyInner):
    def __init__(self) -> None:
        super().__init__()
        self.condition_embedder = None
        self.freq_dim = 8
        self.img_emb = None

    def time_embedding(self, x):
        return torch.zeros((x.shape[0], 1), dtype=x.dtype, device=x.device)

    def time_projection(self, e):
        return torch.zeros((e.shape[0], 6), dtype=e.dtype, device=e.device)

    def text_embedding(self, context):
        return context


class DummyOuter:
    def __init__(self, inner: DummyInner) -> None:
        self.diffusion_model = inner

    def apply_model(
        self,
        x,
        timestep,
        context,
        clip_fea=None,
        time_dim_concat=None,
        transformer_options=None,
        **kwargs,
    ):
        if hasattr(self.diffusion_model, "_forward") and callable(getattr(self.diffusion_model, "_forward")):
            return self.diffusion_model._forward(
                x,
                timestep,
                context,
                clip_fea=clip_fea,
                time_dim_concat=time_dim_concat,
                transformer_options=transformer_options,
                **kwargs,
            )
        if callable(self.diffusion_model):
            return self.diffusion_model(
                x,
                timestep,
                context,
                clip_fea=clip_fea,
                transformer_options=transformer_options,
                **kwargs,
            )
        return self.diffusion_model.forward_orig(
            x,
            timestep,
            context,
            clip_fea=clip_fea,
            transformer_options=transformer_options,
            **kwargs,
        )


class NonWanInner:
    pass


class LegacyOnlyInner(DummyInner):
    pass


class ForwardOrigOnlyInner:
    def __init__(self) -> None:
        self.calls = 0

    def forward_orig(self, *args, **kwargs):
        self.calls += 1
        return "orig-only"


class ForwardOrigProxyWrapper:
    def __init__(self, inner) -> None:
        self.model = inner

    def forward_orig(self, *args, **kwargs):
        return self.model.forward_orig(*args, **kwargs)


class DummyInnerWrapper:
    def __init__(self, inner) -> None:
        self.model = inner
        self.forward_calls = 0
        self.seen_transformer_options = None

    def __call__(self, x, timestep, context, clip_fea=None, transformer_options=None, **kwargs):
        self.forward_calls += 1
        self.seen_transformer_options = transformer_options
        return self.model.forward_orig(
            x,
            timestep,
            context,
            clip_fea=clip_fea,
            transformer_options=transformer_options,
            **kwargs,
        )


class DummyModel:
    model_name = "wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors"

    def __init__(self) -> None:
        self.model = DummyOuter(DummyInner())
        self.model_options = None

    def clone(self):
        return self


class DummyModelForwardOrigOnly(DummyModel):
    def __init__(self) -> None:
        self.model = DummyOuter(ForwardOrigOnlyInner())
        self.model_options = None


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


def test_patcher_rebinds_runtime_when_live_inner_changes_before_apply_model() -> None:
    patched = WanSpectrumPatcher.patch(DummyModel(), _cfg())
    runtime = patched.model_options["transformer_options"][_RUNTIME_KEY]

    live_inner = DummyInner()
    patched.model.diffusion_model = live_inner

    sample_sigmas = torch.tensor([1.0, 0.5, 0.0], dtype=torch.float32)
    out = patched.model.apply_model(
        torch.ones((1, 1, 2, 2), dtype=torch.float32),
        torch.tensor([sample_sigmas[0]], dtype=torch.float32),
        torch.zeros((1, 1, 1), dtype=torch.float32),
        transformer_options={"sample_sigmas": sample_sigmas, "cond_or_uncond": [0, 1]},
    )

    assert torch.is_tensor(out)
    assert live_inner.original_calls == 0
    assert live_inner._spectrum_wan_runtime is runtime
    assert patched.model._spectrum_wan_bound_inner_id == id(live_inner)
    assert runtime.last_info["live_inner_id"] == id(live_inner)
    assert runtime.last_info["live_inner_type"] == "DummyInner"


def test_patcher_rebinds__forward_passthrough_when_live_inner_changes_before_apply_model() -> None:
    patched = WanSpectrumPatcher.patch(DummyModelWithForward(), _cfg())
    runtime = patched.model_options["transformer_options"][_RUNTIME_KEY]

    live_inner = DummyInnerWithForward()
    patched.model.diffusion_model = live_inner

    sample_sigmas = torch.tensor([1.0, 0.5, 0.0], dtype=torch.float32)
    transformer_options = {"sample_sigmas": sample_sigmas, "cond_or_uncond": [0, 1]}
    out = patched.model.apply_model(
        torch.ones((1, 1, 2, 2), dtype=torch.float32),
        torch.tensor([sample_sigmas[0]], dtype=torch.float32),
        torch.zeros((1, 1, 1), dtype=torch.float32),
        transformer_options=transformer_options,
    )

    assert out == "live"
    assert live_inner.live_calls == 1
    assert live_inner.seen_transformer_options is transformer_options
    assert live_inner.seen_transformer_options[_RUNTIME_KEY] is runtime
    assert live_inner._spectrum_wan_runtime is runtime
    assert live_inner._spectrum_wan__forward_wrapped is True


def test_patcher_rebinds_runtime_when_live_inner_is_wrapped_one_level_deeper() -> None:
    patched = WanSpectrumPatcher.patch(DummyModel(), _cfg())
    runtime = patched.model_options["transformer_options"][_RUNTIME_KEY]

    live_inner = DummyInner()
    wrapped_live_inner = DummyInnerWrapper(live_inner)
    patched.model.diffusion_model = wrapped_live_inner

    sample_sigmas = torch.tensor([1.0, 0.5, 0.0], dtype=torch.float32)
    out = patched.model.apply_model(
        torch.ones((1, 1, 2, 2), dtype=torch.float32),
        torch.tensor([sample_sigmas[0]], dtype=torch.float32),
        torch.zeros((1, 1, 1), dtype=torch.float32),
        transformer_options={"sample_sigmas": sample_sigmas, "cond_or_uncond": [0, 1]},
    )

    assert torch.is_tensor(out)
    assert wrapped_live_inner.forward_calls == 1
    assert wrapped_live_inner.seen_transformer_options[_RUNTIME_KEY] is runtime
    assert live_inner._spectrum_wan_runtime is runtime
    assert patched.model._spectrum_wan_bound_inner_id == id(live_inner)
    assert runtime.last_info["live_inner_type"] == "DummyInner"
    assert runtime.last_info["hook_target"] == "model.diffusion_model.model.forward_orig"


def test_patcher_binds_live_inner_when_forward_orig_exists_even_if_runtime_attrs_are_missing() -> None:
    patched = WanSpectrumPatcher.patch(DummyModelForwardOrigOnly(), _cfg())
    runtime = patched.model_options["transformer_options"][_RUNTIME_KEY]

    inner = patched.model.diffusion_model
    assert inner._spectrum_wan_runtime is runtime
    assert runtime.last_info["patched"] is True
    assert runtime.last_info["hook_target"] == "model.diffusion_model.forward_orig"
    assert "blocks" in runtime.last_info["runtime_missing_attrs"]


def test_runtime_missing_attrs_accepts_legacy_conditioning_api() -> None:
    inner = LegacyOnlyInner()

    missing = wan._spectrum_runtime_missing_attrs(inner)

    assert "condition_embedder|time_embedding" not in missing
    assert "condition_embedder|time_projection" not in missing
    assert "condition_embedder|text_embedding" not in missing


def test_runtime_missing_attrs_accepts_new_conditioning_api() -> None:
    inner = NewApiInner()

    missing = wan._spectrum_runtime_missing_attrs(inner)

    assert "condition_embedder|time_embedding" not in missing
    assert "condition_embedder|time_projection" not in missing
    assert "condition_embedder|text_embedding" not in missing


def test_locate_prefers_full_new_api_wan_descendant_over_forward_orig_only_proxy() -> None:
    patched = WanSpectrumPatcher.patch(DummyModel(), _cfg())
    runtime = patched.model_options["transformer_options"][_RUNTIME_KEY]

    live_inner = NewApiInner()
    patched.model.diffusion_model = ForwardOrigProxyWrapper(live_inner)
    original = wan._upstream_sinusoidal_embedding_1d
    wan._upstream_sinusoidal_embedding_1d = (
        lambda dim, t: torch.zeros((t.shape[0], dim), dtype=torch.float32, device=t.device)
    )
    try:
        patched.model.apply_model(
            torch.ones((1, 1, 2, 2), dtype=torch.float32),
            torch.tensor([1.0], dtype=torch.float32),
            torch.zeros((1, 1, 1), dtype=torch.float32),
            transformer_options={"sample_sigmas": torch.tensor([1.0, 0.5, 0.0]), "cond_or_uncond": [0, 1]},
        )
    finally:
        wan._upstream_sinusoidal_embedding_1d = original

    assert live_inner.original_calls == 0
    assert patched.model._spectrum_wan_bound_inner_id == id(live_inner)
    assert runtime.last_info["live_inner_type"] == "NewApiInner"


def test_locate_prefers_full_wan_descendant_over_forward_orig_only_proxy() -> None:
    patched = WanSpectrumPatcher.patch(DummyModel(), _cfg())
    runtime = patched.model_options["transformer_options"][_RUNTIME_KEY]

    live_inner = DummyInner()
    patched.model.diffusion_model = ForwardOrigProxyWrapper(live_inner)
    patched.model.apply_model(
        torch.ones((1, 1, 2, 2), dtype=torch.float32),
        torch.tensor([1.0], dtype=torch.float32),
        torch.zeros((1, 1, 1), dtype=torch.float32),
        transformer_options={"sample_sigmas": torch.tensor([1.0, 0.5, 0.0]), "cond_or_uncond": [0, 1]},
    )

    assert patched.model._spectrum_wan_bound_inner_id == id(live_inner)
    assert runtime.last_info["live_inner_type"] == "DummyInner"


def test_forward_orig_wrapper_falls_back_and_logs_when_runtime_attrs_are_missing(capsys) -> None:
    patched = WanSpectrumPatcher.patch(DummyModelForwardOrigOnly(), _cfg())
    inner = patched.model.diffusion_model

    out = inner.forward_orig(
        torch.tensor([1.0]),
        torch.tensor([1.0]),
        torch.tensor([1.0]),
        transformer_options={},
    )
    captured = capsys.readouterr()

    assert out == "orig-only"
    assert "runtime path unavailable" in captured.err


def test_patcher_clears_live_binding_state_when_current_inner_is_not_wan_like() -> None:
    patched = WanSpectrumPatcher.patch(DummyModel(), _cfg())
    runtime = patched.model_options["transformer_options"][_RUNTIME_KEY]

    patched.model.diffusion_model = NonWanInner()

    try:
        patched.model.apply_model(
            torch.ones((1, 1, 2, 2), dtype=torch.float32),
            torch.tensor([1.0], dtype=torch.float32),
            torch.zeros((1, 1, 1), dtype=torch.float32),
            transformer_options={},
        )
    except AttributeError:
        pass

    assert patched.model._spectrum_wan_bound_inner_id is None
    assert runtime.last_info["patched"] is False
    assert runtime.last_info["hook_target"] == "model.diffusion_model"
    assert "live_inner_id" not in runtime.last_info
    assert "live_inner_type" not in runtime.last_info


def test_successful_rebind_clears_stale_live_inner_root_type() -> None:
    patched = WanSpectrumPatcher.patch(DummyModel(), _cfg())
    runtime = patched.model_options["transformer_options"][_RUNTIME_KEY]

    patched.model.diffusion_model = NonWanInner()
    try:
        patched.model.apply_model(
            torch.ones((1, 1, 2, 2), dtype=torch.float32),
            torch.tensor([1.0], dtype=torch.float32),
            torch.zeros((1, 1, 1), dtype=torch.float32),
            transformer_options={},
        )
    except AttributeError:
        pass

    assert runtime.last_info["live_inner_root_type"] == "NonWanInner"

    live_inner = DummyInner()
    patched.model.diffusion_model = DummyInnerWrapper(live_inner)
    patched.model.apply_model(
        torch.ones((1, 1, 2, 2), dtype=torch.float32),
        torch.tensor([1.0], dtype=torch.float32),
        torch.zeros((1, 1, 1), dtype=torch.float32),
        transformer_options={"sample_sigmas": torch.tensor([1.0, 0.5, 0.0]), "cond_or_uncond": [0, 1]},
    )

    assert "live_inner_root_type" not in runtime.last_info


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


def test_run_spectrum_forward_records_forecast_errors_and_falls_back(capsys) -> None:
    inner = DummyInner()

    class RuntimeStub:
        def __init__(self) -> None:
            self.last_info = {}
            self.observed = []

        def begin_step(self, transformer_options, timesteps):
            return {"step_idx": 2, "actual_forward": False, "global_step": 7}

        def can_forecast(self, transformer_options):
            return True

        def predict_feature(self, transformer_options, step_idx, global_step=None):
            raise RuntimeError("forecast broke")

        def _debug_log(self, message: str) -> None:
            print(message, file=sys.stderr, flush=True)

        def observe_feature(self, transformer_options, step_idx, feature, global_step=None):
            self.observed.append((step_idx, global_step, tuple(feature.shape)))

    runtime = RuntimeStub()
    out = _run_spectrum_forward(
        inner,
        runtime,
        torch.ones((1, 1, 2, 2), dtype=torch.float32),
        torch.tensor([1.0], dtype=torch.float32),
        torch.zeros((1, 1, 1), dtype=torch.float32),
        transformer_options={},
    )

    captured = capsys.readouterr()
    assert torch.is_tensor(out)
    assert runtime.observed == [(2, 7, (1, 4, 1))]
    assert runtime.last_info["forecast_error"] == "RuntimeError: forecast broke"
    assert "forecast_error step=2 global_step=7 RuntimeError: forecast broke" in captured.err


def test_run_spectrum_forward_clears_stale_forecast_error_before_non_forecast_step() -> None:
    inner = DummyInner()

    class RuntimeStub:
        def __init__(self) -> None:
            self.last_info = {"forecast_error": "stale"}
            self.observed = []

        def begin_step(self, transformer_options, timesteps):
            return {"step_idx": 3, "actual_forward": True, "global_step": 8}

        def can_forecast(self, transformer_options):
            raise AssertionError("can_forecast should not run when actual_forward is True")

        def _debug_log(self, message: str) -> None:
            pass

        def observe_feature(self, transformer_options, step_idx, feature, global_step=None):
            self.observed.append((step_idx, global_step, tuple(feature.shape)))

    runtime = RuntimeStub()
    out = _run_spectrum_forward(
        inner,
        runtime,
        torch.ones((1, 1, 2, 2), dtype=torch.float32),
        torch.tensor([1.0], dtype=torch.float32),
        torch.zeros((1, 1, 1), dtype=torch.float32),
        transformer_options={},
    )

    assert torch.is_tensor(out)
    assert "forecast_error" not in runtime.last_info
    assert runtime.observed == [(3, 8, (1, 4, 1))]
