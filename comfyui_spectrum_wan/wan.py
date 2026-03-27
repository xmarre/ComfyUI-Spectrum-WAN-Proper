from __future__ import annotations

import sys
from dataclasses import asdict
from typing import Any, Dict, Optional, Tuple

import torch

from .config import SpectrumWanConfig
from .handlers import resolve_handler
from .runtime import SpectrumWanRuntime

_RUNTIME_KEY = "spectrum_wan_runtime"
_CFG_KEY = "spectrum_wan_cfg"
_ENABLED_KEY = "spectrum_wan_enabled"
_BACKEND_KEY = "spectrum_wan_backend"


def _clone_model(model: Any) -> Any:
    return model.clone() if hasattr(model, "clone") else model


def _ensure_model_options(model: Any) -> Dict[str, Any]:
    if not hasattr(model, "model_options") or model.model_options is None:
        model.model_options = {}
    return model.model_options


def _ensure_transformer_options(model: Any) -> Dict[str, Any]:
    opts = _ensure_model_options(model)
    if "transformer_options" not in opts or opts["transformer_options"] is None:
        opts["transformer_options"] = {}
    return opts["transformer_options"]


def _locate_inner_model(model: Any) -> Tuple[Optional[Any], Optional[str]]:
    outer = getattr(model, "model", None)
    if outer is not None and hasattr(outer, "diffusion_model"):
        return outer.diffusion_model, "model.diffusion_model"
    if hasattr(model, "diffusion_model"):
        return model.diffusion_model, "diffusion_model"
    return None, None


def _bind_runtime_to_inner(
    inner: Any,
    runtime: SpectrumWanRuntime,
    inner_name: Optional[str] = None,
) -> bool:
    if inner is None or not _looks_like_wan(inner):
        return False

    inner._spectrum_wan_runtime = runtime
    if hasattr(inner, "_forward") and callable(getattr(inner, "_forward")):
        _wrap_wan__forward_passthrough(inner)
    _wrap_wan_forward_orig(inner)

    runtime.last_info["patched"] = True
    runtime.last_info["hook_target"] = f"{inner_name}.forward_orig" if inner_name else "forward_orig"
    runtime.last_info["live_inner_type"] = type(inner).__name__
    runtime.last_info["live_inner_id"] = id(inner)
    return True


def _wrap_outer_apply_model(outer: Any, runtime: SpectrumWanRuntime) -> None:
    if outer is None or not hasattr(outer, "apply_model") or not callable(getattr(outer, "apply_model")):
        return

    outer._spectrum_wan_runtime = runtime

    if getattr(outer, "_spectrum_wan_apply_model_wrapped", False):
        return

    original_apply_model = outer.apply_model

    def wrapped_apply_model(*args, **kwargs):
        current_runtime = getattr(outer, "_spectrum_wan_runtime", None)
        if isinstance(current_runtime, SpectrumWanRuntime):
            current_inner = getattr(outer, "diffusion_model", None)
            previously_bound_id = getattr(outer, "_spectrum_wan_bound_inner_id", None)
            bound = _bind_runtime_to_inner(current_inner, current_runtime, "model.diffusion_model")
            current_inner_id = id(current_inner) if bound else None
            outer._spectrum_wan_bound_inner_id = current_inner_id

            if bound and previously_bound_id != current_inner_id:
                current_runtime._debug_log(
                    "[Spectrum WAN] rebound live inner "
                    f"type={type(current_inner).__name__} "
                    f"id={current_inner_id}"
                )
            elif not bound:
                current_runtime.last_info["patched"] = False
                current_runtime.last_info["hook_target"] = "model.diffusion_model"
                current_runtime.last_info.pop("live_inner_type", None)
                current_runtime.last_info.pop("live_inner_id", None)
                current_runtime._debug_log(
                    "[Spectrum WAN] live inner is not WAN-like; skipping runtime bind"
                )

        return original_apply_model(*args, **kwargs)

    outer.apply_model = wrapped_apply_model
    outer._spectrum_wan_apply_model_wrapped = True


def _looks_like_wan(inner: Any) -> bool:
    required = (
        "forward_orig",
        "blocks",
        "head",
        "patch_embedding",
        "condition_embedder",
        "unpatchify",
    )
    return all(hasattr(inner, name) for name in required)


def _resolve_runtime(
    transformer_options: Optional[Dict[str, Any]],
    inner: Optional[Any] = None,
) -> Optional[SpectrumWanRuntime]:
    if isinstance(transformer_options, dict):
        runtime = transformer_options.get(_RUNTIME_KEY)
        if isinstance(runtime, SpectrumWanRuntime):
            return runtime
    if inner is not None:
        runtime = getattr(inner, "_spectrum_wan_runtime", None)
        if isinstance(runtime, SpectrumWanRuntime):
            return runtime
    return None


def _run_spectrum_forward(
    inner: Any,
    runtime: SpectrumWanRuntime,
    x,
    t,
    context,
    clip_fea=None,
    freqs=None,
    transformer_options=None,
    **kwargs,
):
    if transformer_options is None:
        transformer_options = {}

    decision = runtime.begin_step(transformer_options, t)
    step_idx = decision["step_idx"]
    actual_forward = decision["actual_forward"]

    # Original WAN preprocessing.
    x = inner.patch_embedding(x.float()).to(x.dtype)
    grid_sizes = x.shape[2:]
    transformer_options["grid_sizes"] = grid_sizes
    x = x.flatten(2).transpose(1, 2)

    temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image = inner.condition_embedder(
        t, context, clip_fea
    )
    timestep_proj = timestep_proj.unflatten(1, (6, -1))

    if encoder_hidden_states_image is not None:
        encoder_hidden_states = torch.concat([encoder_hidden_states_image, encoder_hidden_states], dim=1)

    full_ref = None
    if getattr(inner, "ref_conv", None) is not None:
        full_ref = kwargs.get("reference_latent", None)
        if full_ref is not None:
            full_ref = inner.ref_conv(full_ref).flatten(2).transpose(1, 2)
            x = torch.concat((full_ref, x), dim=1)

    runtime.last_info.pop("forecast_error", None)

    if not actual_forward and runtime.can_forecast(transformer_options):
        try:
            predicted_x = runtime.predict_feature(
                transformer_options,
                step_idx,
                global_step=decision.get("global_step"),
            )
        except Exception as exc:
            runtime.last_info["forecast_error"] = f"{type(exc).__name__}: {exc}"
            runtime._debug_log(
                f"[Spectrum WAN] forecast_error step={step_idx} "
                f"global_step={decision.get('global_step')} {type(exc).__name__}: {exc}"
            )
            predicted_x = None

        if predicted_x is not None:
            predicted_x = predicted_x.to(device=x.device, dtype=x.dtype)
            if torch.isfinite(predicted_x).all():
                x = predicted_x
                x = inner.head(x, temb)
                if full_ref is not None:
                    x = x[:, full_ref.shape[1]:]
                return inner.unpatchify(x, grid_sizes)

    patches_replace = transformer_options.get("patches_replace", {})
    blocks_replace = patches_replace.get("dit", {})
    transformer_options["total_blocks"] = len(inner.blocks)
    transformer_options["block_type"] = "double"
    context_img_len = clip_fea.shape[-2] if clip_fea is not None else None

    for i, block in enumerate(inner.blocks):
        transformer_options["block_index"] = i
        if ("double_block", i) in blocks_replace:
            def block_wrap(args):
                out = {}
                out["img"] = block(
                    args["img"],
                    context=args["txt"],
                    e=args["vec"],
                    freqs=args["pe"],
                    context_img_len=context_img_len,
                    transformer_options=args["transformer_options"],
                )
                return out

            out = blocks_replace[("double_block", i)](
                {"img": x, "txt": encoder_hidden_states, "vec": timestep_proj, "pe": freqs, "transformer_options": transformer_options},
                {"original_block": block_wrap},
            )
            x = out["img"]
        else:
            x = block(
                x,
                e=timestep_proj,
                freqs=freqs,
                context=encoder_hidden_states,
                context_img_len=context_img_len,
                transformer_options=transformer_options,
            )

    runtime.observe_feature(
        transformer_options,
        step_idx,
        x,
        global_step=decision.get("global_step"),
    )

    x = inner.head(x, temb)
    if full_ref is not None:
        x = x[:, full_ref.shape[1]:]
    return inner.unpatchify(x, grid_sizes)


def _wrap_wan_forward_orig(inner: Any) -> None:
    if getattr(inner, "_spectrum_wan_wrapped", False):
        return

    original_forward_orig = inner.forward_orig

    def wrapped_forward_orig(
        x,
        t,
        context,
        clip_fea=None,
        freqs=None,
        transformer_options=None,
        **kwargs,
    ):
        runtime = _resolve_runtime(transformer_options, inner)
        if runtime is None or not runtime.cfg.enabled:
            if transformer_options is None:
                return original_forward_orig(x, t, context, clip_fea=clip_fea, freqs=freqs, **kwargs)
            return original_forward_orig(
                x,
                t,
                context,
                clip_fea=clip_fea,
                freqs=freqs,
                transformer_options=transformer_options,
                **kwargs,
            )

        out = _run_spectrum_forward(
            inner,
            runtime,
            x,
            t,
            context,
            clip_fea=clip_fea,
            freqs=freqs,
            transformer_options=transformer_options,
            **kwargs,
        )
        return out

    inner._spectrum_wan_original_forward_orig = original_forward_orig
    inner.forward_orig = wrapped_forward_orig
    inner._spectrum_wan_wrapped = True
    inner._spectrum_wan_wrapped_attr = "forward_orig"


def _wrap_wan__forward_passthrough(inner: Any) -> None:
    if getattr(inner, "_spectrum_wan__forward_wrapped", False):
        return

    original__forward = inner._forward

    def wrapped__forward(
        x,
        timestep,
        context,
        clip_fea=None,
        time_dim_concat=None,
        transformer_options=None,
        **kwargs,
    ):
        if transformer_options is None:
            transformer_options = {}
        inner_runtime = getattr(inner, "_spectrum_wan_runtime", None)
        if isinstance(inner_runtime, SpectrumWanRuntime):
            transformer_options[_RUNTIME_KEY] = inner_runtime
        return original__forward(
            x,
            timestep,
            context,
            clip_fea=clip_fea,
            time_dim_concat=time_dim_concat,
            transformer_options=transformer_options,
            **kwargs,
        )

    inner._spectrum_wan_original__forward = original__forward
    inner._forward = wrapped__forward
    inner._spectrum_wan__forward_wrapped = True


class WanSpectrumPatcher:
    @staticmethod
    def patch(model: Any, cfg: SpectrumWanConfig) -> Any:
        cfg = cfg.validated()
        patched = _clone_model(model)
        handler = resolve_handler(cfg.backend, patched)

        tr_opts = _ensure_transformer_options(patched)
        runtime = tr_opts.get(_RUNTIME_KEY)
        if isinstance(runtime, SpectrumWanRuntime):
            runtime.update(cfg, handler)
        else:
            runtime = SpectrumWanRuntime(cfg, handler)

        tr_opts[_CFG_KEY] = cfg
        tr_opts[_RUNTIME_KEY] = runtime
        tr_opts[_ENABLED_KEY] = cfg.enabled
        tr_opts[_BACKEND_KEY] = handler.backend_id
        tr_opts["spectrum_wan_cfg_dict"] = asdict(cfg)
        tr_opts["spectrum_wan_handler"] = {
            "backend_id": handler.backend_id,
            "phase_tag": handler.phase_tag,
            "is_moe_expert": handler.is_moe_expert,
        }

        outer = getattr(patched, "model", None)
        _wrap_outer_apply_model(outer, runtime)

        inner, inner_name = _locate_inner_model(patched)
        if _bind_runtime_to_inner(inner, runtime, inner_name):
            if cfg.debug:
                print(f"[Spectrum WAN] patched hook_target={runtime.last_info['hook_target']}", file=sys.stderr, flush=True)
        else:
            runtime.last_info["hook_target"] = inner_name
            runtime.last_info["patched"] = False

        return patched
