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
        return torch.zeros((*x.shape[:-1], 1), dtype=x.dtype, device=x.device)

    def time_projection(self, e):
        return torch.zeros((*e.shape[:-1], 6), dtype=e.dtype, device=e.device)

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


def test_direct_forward_orig_requires_runtime_in_current_transformer_options() -> None:
    patched = WanSpectrumPatcher.patch(DummyModel(), _cfg())
    inner = patched.model.diffusion_model

    sample_sigmas = torch.tensor([1.0, 0.5, 0.0], dtype=torch.float32)
    runtime = patched.model_options["transformer_options"][_RUNTIME_KEY]
    out = inner.forward_orig(
        torch.ones((1, 1, 2, 2), dtype=torch.float32),
        torch.tensor([sample_sigmas[0]], dtype=torch.float32),
        torch.zeros((1, 1, 1), dtype=torch.float32),
        transformer_options={
            _RUNTIME_KEY: runtime,
            "sample_sigmas": sample_sigmas,
            "cond_or_uncond": [0, 1],
        },
    )

    assert torch.is_tensor(out)
    assert inner.original_calls == 0
    assert inner._spectrum_wan_runtime.last_info["last_sigma"] == 1.0


def test_patched_wan_does_not_leak_runtime_into_later_bypassed_calls() -> None:
    model = DummyModel()
    patched = WanSpectrumPatcher.patch(model, _cfg())
    inner = patched.model.diffusion_model
    runtime = patched.model_options["transformer_options"][_RUNTIME_KEY]
    sample_sigmas = torch.tensor([1.0, 0.5, 0.0], dtype=torch.float32)

    first_out = inner.forward_orig(
        torch.ones((1, 1, 2, 2), dtype=torch.float32),
        torch.tensor([sample_sigmas[0]], dtype=torch.float32),
        torch.zeros((1, 1, 1), dtype=torch.float32),
        transformer_options={
            _RUNTIME_KEY: runtime,
            "sample_sigmas": sample_sigmas,
            "cond_or_uncond": [0, 1],
        },
    )

    bypass_out = model.model.diffusion_model.forward_orig(
        torch.ones((1, 1, 2, 2), dtype=torch.float32),
        torch.tensor([sample_sigmas[1]], dtype=torch.float32),
        torch.zeros((1, 1, 1), dtype=torch.float32),
        transformer_options={"sample_sigmas": sample_sigmas, "cond_or_uncond": [0, 1]},
    )

    assert torch.is_tensor(first_out)
    assert bypass_out == "orig"
    assert inner.original_calls == 1


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
    transformer_options = {
        _RUNTIME_KEY: patched.model_options["transformer_options"][_RUNTIME_KEY],
        "sample_sigmas": sample_sigmas,
        "cond_or_uncond": [0, 1],
    }
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
    assert inner.seen_transformer_options[_RUNTIME_KEY] is patched.model_options["transformer_options"][_RUNTIME_KEY]
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


def test_run_spectrum_forward_new_api_preserves_multitoken_timestep_shape() -> None:
    class CaptureBlock:
        def __init__(self) -> None:
            self.seen_e_shape = None

        def __call__(
            self,
            x,
            e,
            freqs=None,
            context=None,
            context_img_len=None,
            transformer_options=None,
        ):
            self.seen_e_shape = tuple(e.shape)
            return x

    class RuntimeStub:
        def __init__(self) -> None:
            self.last_info = {}

        def begin_step(self, transformer_options, timesteps):
            return {"step_idx": 0, "actual_forward": True, "global_step": 0}

        def can_forecast(self, transformer_options):
            raise AssertionError("can_forecast should not run when actual_forward is True")

        def _debug_log(self, message: str) -> None:
            pass

        def observe_feature(self, transformer_options, step_idx, feature, global_step=None):
            pass

    inner = NewApiInner()
    block = CaptureBlock()
    inner.blocks = [block]
    runtime = RuntimeStub()

    original = wan._upstream_sinusoidal_embedding_1d
    wan._upstream_sinusoidal_embedding_1d = (
        lambda dim, t: torch.zeros((t.shape[0], dim), dtype=torch.float32, device=t.device)
    )
    try:
        _run_spectrum_forward(
            inner,
            runtime,
            torch.ones((1, 1, 2, 2), dtype=torch.float32),
            torch.tensor([[1.0, 0.5]], dtype=torch.float32),
            torch.zeros((1, 1, 1), dtype=torch.float32),
            transformer_options={},
        )
    finally:
        wan._upstream_sinusoidal_embedding_1d = original

    assert block.seen_e_shape == (1, 2, 6, 1)


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
    runtime = patched.model_options["transformer_options"][_RUNTIME_KEY]

    out = inner.forward_orig(
        torch.tensor([1.0]),
        torch.tensor([1.0]),
        torch.tensor([1.0]),
        transformer_options={_RUNTIME_KEY: runtime},
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


def test_patcher__forward_passthrough_preserves_current_transformer_runtime() -> None:
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
    assert inner.seen_transformer_options[_RUNTIME_KEY] is stale_runtime


def test_patcher__forward_passthrough_injects_bound_runtime_when_missing() -> None:
    patched = WanSpectrumPatcher.patch(DummyModelWithForward(), _cfg())
    inner = patched.model.diffusion_model
    current_runtime = patched.model_options["transformer_options"][_RUNTIME_KEY]

    transformer_options = {"cond_or_uncond": [0, 1]}

    inner._forward(
        torch.ones((1, 1, 2, 2), dtype=torch.float32),
        torch.tensor([1.0], dtype=torch.float32),
        torch.zeros((1, 1, 1), dtype=torch.float32),
        transformer_options=transformer_options,
    )

    assert inner.seen_transformer_options is transformer_options
    assert inner.seen_transformer_options[_RUNTIME_KEY] is current_runtime


def test_runtime_accumulates_history_when_sample_sigmas_are_missing() -> None:
    cfg = SpectrumWanConfig(
        backend="wan21",
        warmup_steps=0,
        window_size=2.0,
        flex_window=0.75,
    ).validated()
    runtime = SpectrumWanRuntime(cfg, resolve_handler("wan21", DummyModel()))
    transformer_options = {"cond_or_uncond": [0, 1]}

    first = runtime.begin_step(transformer_options, torch.tensor([1.0], dtype=torch.float32))
    assert first["step_idx"] == 0
    assert first["actual_forward"] is True
    runtime.observe_feature(transformer_options, first["step_idx"], torch.ones((1, 2), dtype=torch.float32))
    runtime.end_step(transformer_options, first["step_idx"])

    second = runtime.begin_step(transformer_options, torch.tensor([0.8], dtype=torch.float32))
    assert second["step_idx"] == 1
    assert second["actual_forward"] is True
    runtime.observe_feature(transformer_options, second["step_idx"], torch.full((1, 2), 2.0, dtype=torch.float32))
    runtime.end_step(transformer_options, second["step_idx"])

    third = runtime.begin_step(transformer_options, torch.tensor([0.6], dtype=torch.float32))
    assert third["step_idx"] == 2
    assert third["actual_forward"] is False


def test_runtime_missing_sample_sigmas_does_not_reuse_previous_schedule_length() -> None:
    cfg = SpectrumWanConfig(
        backend="wan21",
        warmup_steps=0,
        window_size=2.0,
        flex_window=0.75,
    ).validated()
    runtime = SpectrumWanRuntime(cfg, resolve_handler("wan21", DummyModel()))

    scheduled_options = {
        "sample_sigmas": torch.tensor([1.0, 0.5, 0.0], dtype=torch.float32),
        "cond_or_uncond": [0, 1],
    }
    first_scheduled = runtime.begin_step(scheduled_options, torch.tensor([1.0], dtype=torch.float32))
    runtime.observe_feature(
        scheduled_options,
        first_scheduled["step_idx"],
        torch.ones((1, 2), dtype=torch.float32),
    )
    runtime.end_step(scheduled_options, first_scheduled["step_idx"])
    second_scheduled = runtime.begin_step(scheduled_options, torch.tensor([0.5], dtype=torch.float32))
    runtime.observe_feature(
        scheduled_options,
        second_scheduled["step_idx"],
        torch.full((1, 2), 2.0, dtype=torch.float32),
    )
    runtime.end_step(scheduled_options, second_scheduled["step_idx"])
    assert runtime.num_steps() == 2

    unscheduled_options = scheduled_options
    unscheduled_options.pop("sample_sigmas")
    first = runtime.begin_step(unscheduled_options, torch.tensor([1.0], dtype=torch.float32))
    assert first["step_idx"] == 0
    assert "spectrum_wan_active_num_steps" not in unscheduled_options
    assert runtime.last_info["num_steps"] == 0
    runtime.observe_feature(unscheduled_options, first["step_idx"], torch.ones((1, 2), dtype=torch.float32))
    runtime.end_step(unscheduled_options, first["step_idx"])

    second = runtime.begin_step(unscheduled_options, torch.tensor([0.8], dtype=torch.float32))
    assert second["step_idx"] == 1
    runtime.observe_feature(unscheduled_options, second["step_idx"], torch.full((1, 2), 2.0, dtype=torch.float32))
    runtime.end_step(unscheduled_options, second["step_idx"])

    third = runtime.begin_step(unscheduled_options, torch.tensor([0.6], dtype=torch.float32))
    assert third["step_idx"] == 2
    assert third["actual_forward"] is False


def test_runtime_reuses_completed_final_sigma_decision_before_next_cycle_reset() -> None:
    cfg = SpectrumWanConfig(
        backend="wan21",
        warmup_steps=0,
        window_size=2.0,
        flex_window=0.75,
    ).validated()
    runtime = SpectrumWanRuntime(cfg, resolve_handler("wan21", DummyModel()))
    sample_sigmas = torch.tensor([1.0, 0.5, 0.18181819], dtype=torch.float32)
    transformer_options = {
        "sample_sigmas": sample_sigmas,
        "cond_or_uncond": [0, 1],
    }

    for step in range(len(sample_sigmas)):
        decision = runtime.begin_step(transformer_options, sample_sigmas[step : step + 1])
        assert decision["step_idx"] == step
        if decision["actual_forward"]:
            runtime.observe_feature(
                transformer_options,
                decision["step_idx"],
                torch.full((1, 2), float(step + 1), dtype=torch.float32),
            )
        runtime.end_step(transformer_options, decision["step_idx"])

    final_duplicate = runtime.begin_step(
        transformer_options,
        sample_sigmas[-1:],
    )
    assert final_duplicate is decision
    assert final_duplicate["step_idx"] == 2

    next_cycle = runtime.begin_step(transformer_options, torch.tensor([1.0], dtype=torch.float32))
    assert next_cycle["step_idx"] == 0


def test_patcher_apply_model_overwrites_stale_runtime_with_current_outer_runtime() -> None:
    patched = WanSpectrumPatcher.patch(DummyModelWithForward(), _cfg())
    current_runtime = patched.model_options["transformer_options"][_RUNTIME_KEY]
    inner = patched.model.diffusion_model

    stale_runtime = SpectrumWanRuntime(_cfg(), resolve_handler("auto", DummyModel()))
    sample_sigmas = torch.tensor([1.0, 0.5, 0.0], dtype=torch.float32)
    transformer_options = {
        _RUNTIME_KEY: stale_runtime,
        "sample_sigmas": sample_sigmas,
        "cond_or_uncond": [0, 1],
    }

    out = patched.model.apply_model(
        torch.ones((1, 1, 2, 2), dtype=torch.float32),
        torch.tensor([sample_sigmas[0]], dtype=torch.float32),
        torch.zeros((1, 1, 1), dtype=torch.float32),
        transformer_options=transformer_options,
    )

    assert out == "live"
    assert inner.seen_transformer_options is transformer_options
    assert inner.seen_transformer_options[_RUNTIME_KEY] is current_runtime
    assert inner.seen_transformer_options[_RUNTIME_KEY] is not stale_runtime


def test_apply_model_installs_diffusion_wrapper_into_live_transformer_options() -> None:
    patched = WanSpectrumPatcher.patch(DummyModelWithForward(), _cfg())
    runtime = patched.model_options["transformer_options"][_RUNTIME_KEY]
    transformer_options = {"cond_or_uncond": [0, 1]}

    patched.model.apply_model(
        torch.ones((1, 1, 2, 2), dtype=torch.float32),
        torch.tensor([1.0], dtype=torch.float32),
        torch.zeros((1, 1, 1), dtype=torch.float32),
        transformer_options=transformer_options,
    )

    wrapper_slot = transformer_options["wrappers"]["diffusion_model"]["spectrum_wan_runtime"]
    assert transformer_options[_RUNTIME_KEY] is runtime
    assert wan._spectrum_wan_diffusion_model_wrapper in wrapper_slot
    assert runtime.last_info["live_diffusion_wrapper_installed"] is True


def test_apply_model_installs_live_diffusion_wrapper_once() -> None:
    patched = WanSpectrumPatcher.patch(DummyModelWithForward(), _cfg())
    transformer_options = {"cond_or_uncond": [0, 1]}

    for _ in range(2):
        patched.model.apply_model(
            torch.ones((1, 1, 2, 2), dtype=torch.float32),
            torch.tensor([1.0], dtype=torch.float32),
            torch.zeros((1, 1, 1), dtype=torch.float32),
            transformer_options=transformer_options,
        )

    wrapper_slot = transformer_options["wrappers"]["diffusion_model"]["spectrum_wan_runtime"]
    assert wrapper_slot.count(wan._spectrum_wan_diffusion_model_wrapper) == 1


def test_apply_model_finds_positional_transformer_options() -> None:
    patched = WanSpectrumPatcher.patch(DummyModelWithForward(), _cfg())
    runtime = patched.model_options["transformer_options"][_RUNTIME_KEY]
    transformer_options = {"cond_or_uncond": [0, 1]}

    patched.model.apply_model(
        torch.ones((1, 1, 2, 2), dtype=torch.float32),
        torch.tensor([1.0], dtype=torch.float32),
        torch.zeros((1, 1, 1), dtype=torch.float32),
        None,
        None,
        transformer_options,
    )

    wrapper_slot = transformer_options["wrappers"]["diffusion_model"]["spectrum_wan_runtime"]
    assert transformer_options[_RUNTIME_KEY] is runtime
    assert wan._spectrum_wan_diffusion_model_wrapper in wrapper_slot


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


class DummyInnerForDiffusionWrapper(DummyInner):
    patch_size = (1, 1, 1)

    def rope_encode(self, t, h, w, device=None, dtype=None, transformer_options=None):
        return torch.zeros((1,), device=device, dtype=dtype)

    def unpatchify(self, x, grid_sizes):
        return torch.zeros((1, 1, *grid_sizes), dtype=x.dtype, device=x.device)


class PaddedDummyInnerForDiffusionWrapper(DummyInnerForDiffusionWrapper):
    patch_size = (2, 2, 2)

    def __init__(self) -> None:
        super().__init__()
        self.rope_args = None

    def rope_encode(self, t, h, w, device=None, dtype=None, transformer_options=None):
        self.rope_args = (t, h, w)
        return torch.zeros((1,), device=device, dtype=dtype)


class DummyModelForDiffusionWrapper(DummyModel):
    def __init__(self) -> None:
        self.model = DummyOuter(DummyInnerForDiffusionWrapper())
        self.model_options = None


class DummyDiffusionExecutor:
    def __init__(self, class_obj) -> None:
        self.class_obj = class_obj
        self.calls = 0

    def __call__(self, *args, **kwargs):
        self.calls += 1
        return torch.full((1, 1, 1, 1, 1), -1.0)


def test_patcher_installs_diffusion_model_wrapper() -> None:
    patched = WanSpectrumPatcher.patch(DummyModelForDiffusionWrapper(), _cfg())
    wrapper_slot = patched.model_options["transformer_options"]["wrappers"]["diffusion_model"]["spectrum_wan_runtime"]

    assert wan._spectrum_wan_diffusion_model_wrapper in wrapper_slot


def test_diffusion_model_wrapper_runs_spectrum_without_calling_captured_forward_executor() -> None:
    patched = WanSpectrumPatcher.patch(DummyModelForDiffusionWrapper(), _cfg())
    inner = patched.model.diffusion_model
    runtime = patched.model_options["transformer_options"][_RUNTIME_KEY]
    executor = DummyDiffusionExecutor(inner)
    sample_sigmas = torch.tensor([1.0, 0.5, 0.0], dtype=torch.float32)
    transformer_options = {
        _RUNTIME_KEY: runtime,
        "sample_sigmas": sample_sigmas,
        "cond_or_uncond": [0, 1],
    }

    out = wan._spectrum_wan_diffusion_model_wrapper(
        executor,
        torch.ones((1, 1, 2, 2, 2), dtype=torch.float32),
        torch.tensor([sample_sigmas[0]], dtype=torch.float32),
        torch.zeros((1, 1, 1), dtype=torch.float32),
        None,
        None,
        transformer_options,
    )

    assert executor.calls == 0
    assert torch.is_tensor(out)
    assert tuple(out.shape) == (1, 1, 2, 2, 2)
    assert inner.original_calls == 0
    assert runtime.last_info["last_sigma"] == 1.0


def test_diffusion_model_wrapper_falls_back_for_non_wan_executor_target() -> None:
    cfg = _cfg()
    runtime = SpectrumWanRuntime(cfg, resolve_handler("auto", DummyModelForDiffusionWrapper()))
    executor = DummyDiffusionExecutor(NonWanInner())

    out = wan._spectrum_wan_diffusion_model_wrapper(
        executor,
        torch.ones((1, 1, 2, 2, 2), dtype=torch.float32),
        torch.tensor([1.0], dtype=torch.float32),
        torch.zeros((1, 1, 1), dtype=torch.float32),
        None,
        None,
        {_RUNTIME_KEY: runtime},
    )

    assert executor.calls == 1
    assert torch.equal(out, torch.full((1, 1, 1, 1, 1), -1.0))


class DummySampling:
    def calculate_input(self, sigma, x):
        return x

    def timestep(self, timestep):
        return timestep

    def calculate_denoised(self, sigma, model_output, x):
        return model_output


class DummyBaseOuterForApplyDirect:
    def __init__(self, inner) -> None:
        self.diffusion_model = inner
        self.model_sampling = DummySampling()
        self.current_patcher = None

    def get_dtype_inference(self):
        return torch.float32

    def process_timestep(self, timestep, **kwargs):
        return timestep


class DummyApplyExecutor:
    def __init__(self, class_obj) -> None:
        self.class_obj = class_obj
        self.calls = 0

    def __call__(self, *args, **kwargs):
        self.calls += 1
        return torch.full((1, 1, 1, 1, 1), -2.0)


def test_apply_model_wrapper_without_runtime_uses_executor_path() -> None:
    outer = DummyBaseOuterForApplyDirect(DummyInnerForDiffusionWrapper())
    executor = DummyApplyExecutor(outer)

    out = wan._spectrum_wan_apply_model_wrapper(
        executor,
        torch.ones((1, 1, 2, 2, 2), dtype=torch.float32),
        torch.tensor([1.0], dtype=torch.float32),
        None,
        torch.zeros((1, 1, 1), dtype=torch.float32),
        None,
        {"cond_or_uncond": [0, 1]},
    )

    assert executor.calls == 1
    assert torch.equal(out, torch.full((1, 1, 1, 1, 1), -2.0))


def test_apply_model_wrapper_can_run_spectrum_direct_when_diffusion_wrapper_is_bypassed() -> None:
    cfg = _cfg()
    inner = DummyInnerForDiffusionWrapper()
    outer = DummyBaseOuterForApplyDirect(inner)
    runtime = SpectrumWanRuntime(cfg, resolve_handler("wan21", DummyModelForDiffusionWrapper()))
    outer._spectrum_wan_runtime = runtime
    executor = DummyApplyExecutor(outer)
    sample_sigmas = torch.tensor([1.0, 0.5, 0.0], dtype=torch.float32)
    transformer_options = {
        "sample_sigmas": sample_sigmas,
        "cond_or_uncond": [0, 1],
    }

    out = wan._spectrum_wan_apply_model_wrapper(
        executor,
        torch.ones((1, 1, 2, 2, 2), dtype=torch.float32),
        torch.tensor([sample_sigmas[0]], dtype=torch.float32),
        None,
        torch.zeros((1, 1, 1), dtype=torch.float32),
        None,
        transformer_options,
    )

    assert executor.calls == 0
    assert torch.is_tensor(out)
    assert tuple(out.shape) == (1, 1, 2, 2, 2)
    assert inner.original_calls == 0
    assert runtime.last_info["last_sigma"] == 1.0
    assert transformer_options[_RUNTIME_KEY] is runtime


def test_apply_model_wrapper_direct_path_uses_padded_rope_sizes_and_crops_output() -> None:
    cfg = _cfg()
    inner = PaddedDummyInnerForDiffusionWrapper()
    outer = DummyBaseOuterForApplyDirect(inner)
    runtime = SpectrumWanRuntime(cfg, resolve_handler("wan21", DummyModelForDiffusionWrapper()))
    outer._spectrum_wan_runtime = runtime
    executor = DummyApplyExecutor(outer)
    sample_sigmas = torch.tensor([1.0, 0.5, 0.0], dtype=torch.float32)
    transformer_options = {
        "sample_sigmas": sample_sigmas,
        "cond_or_uncond": [0, 1],
    }

    out = wan._spectrum_wan_apply_model_wrapper(
        executor,
        torch.ones((1, 1, 3, 3, 3), dtype=torch.float32),
        torch.tensor([sample_sigmas[0]], dtype=torch.float32),
        None,
        torch.zeros((1, 1, 1), dtype=torch.float32),
        None,
        transformer_options,
    )

    assert executor.calls == 0
    assert torch.is_tensor(out)
    assert tuple(out.shape) == (1, 1, 3, 3, 3)
    assert inner.rope_args == (3, 4, 4)
    assert runtime.last_info["last_sigma"] == 1.0
    assert transformer_options[_RUNTIME_KEY] is runtime


def test_apply_model_wrapper_keeps_wan22_on_executor_path() -> None:
    cfg = _cfg()
    inner = DummyInnerForDiffusionWrapper()
    outer = DummyBaseOuterForApplyDirect(inner)
    runtime = SpectrumWanRuntime(cfg, resolve_handler("wan22_high_noise", DummyModelForDiffusionWrapper()))
    outer._spectrum_wan_runtime = runtime
    executor = DummyApplyExecutor(outer)
    transformer_options = {
        "sample_sigmas": torch.tensor([1.0, 0.5, 0.0], dtype=torch.float32),
        "cond_or_uncond": [0, 1],
    }

    out = wan._spectrum_wan_apply_model_wrapper(
        executor,
        torch.ones((1, 1, 2, 2, 2), dtype=torch.float32),
        torch.tensor([1.0], dtype=torch.float32),
        None,
        torch.zeros((1, 1, 1), dtype=torch.float32),
        None,
        transformer_options,
    )

    assert executor.calls == 1
    assert torch.equal(out, torch.full((1, 1, 1, 1, 1), -2.0))
    assert "last_sigma" not in runtime.last_info


def test_apply_model_wrapper_falls_back_when_wan21_direct_inner_is_unsupported() -> None:
    cfg = _cfg()
    inner = ForwardOrigOnlyInner()
    outer = DummyBaseOuterForApplyDirect(inner)
    runtime = SpectrumWanRuntime(cfg, resolve_handler("wan21", DummyModelForDiffusionWrapper()))
    outer._spectrum_wan_runtime = runtime
    executor = DummyApplyExecutor(outer)
    transformer_options = {
        "sample_sigmas": torch.tensor([1.0, 0.5, 0.0], dtype=torch.float32),
        "cond_or_uncond": [0, 1],
    }

    out = wan._spectrum_wan_apply_model_wrapper(
        executor,
        torch.ones((1, 1, 2, 2, 2), dtype=torch.float32),
        torch.tensor([1.0], dtype=torch.float32),
        None,
        torch.zeros((1, 1, 1), dtype=torch.float32),
        None,
        transformer_options,
    )

    assert executor.calls == 1
    assert torch.equal(out, torch.full((1, 1, 1, 1, 1), -2.0))
    assert "runtime_missing_attrs" in runtime.last_info
    assert "last_sigma" not in runtime.last_info


class PackLatentsTupleModule:
    @staticmethod
    def pack_latents(model_output):
        return torch.full((1, 1), 3.0), ["shape"]


class PackLatentsTensorModule:
    @staticmethod
    def pack_latents(model_output):
        return torch.full((1, 1), 4.0)


class PackLatentsUnsizedModule:
    @staticmethod
    def pack_latents(model_output):
        return torch.full((1, 1), 5.0)


class UnsizedLatentOutput:
    def __len__(self):
        raise TypeError("unsized")


def test_try_pack_latents_accepts_tuple_pack_latents_return() -> None:
    original = wan._comfy_latent_utils
    wan._comfy_latent_utils = lambda: [PackLatentsTupleModule]
    try:
        out = wan._try_pack_latents([torch.ones((1, 1)), torch.ones((1, 1))], True)
    finally:
        wan._comfy_latent_utils = original

    assert torch.equal(out, torch.full((1, 1), 3.0))


def test_try_pack_latents_accepts_tensor_pack_latents_return() -> None:
    original = wan._comfy_latent_utils
    wan._comfy_latent_utils = lambda: [PackLatentsTensorModule]
    try:
        out = wan._try_pack_latents([torch.ones((1, 1)), torch.ones((1, 1))], True)
    finally:
        wan._comfy_latent_utils = original

    assert torch.equal(out, torch.full((1, 1), 4.0))


def test_try_pack_latents_leaves_unsized_output_unchanged() -> None:
    original = wan._comfy_latent_utils
    wan._comfy_latent_utils = lambda: [PackLatentsUnsizedModule]
    model_output = UnsizedLatentOutput()
    try:
        out = wan._try_pack_latents(model_output, True)
    finally:
        wan._comfy_latent_utils = original

    assert out is model_output


def test_cast_to_device_preserves_non_floating_dtypes() -> None:
    bool_tensor = torch.tensor([True, False], dtype=torch.bool)
    int16_tensor = torch.tensor([1, 2], dtype=torch.int16)

    bool_out = wan._cast_to_device(bool_tensor, torch.device("cpu"), torch.float32)
    int16_out = wan._cast_to_device(int16_tensor, torch.device("cpu"), torch.float32)

    assert bool_out.dtype == torch.bool
    assert int16_out.dtype == torch.int16
