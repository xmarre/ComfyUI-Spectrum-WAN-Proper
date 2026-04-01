from __future__ import annotations

import sys
from dataclasses import asdict
from typing import Any, Dict, Optional, Tuple

import torch

try:
    from comfy.ldm.wan.model import sinusoidal_embedding_1d as _upstream_sinusoidal_embedding_1d
except ModuleNotFoundError as exc:  # pragma: no cover - ComfyUI is not available in unit tests.
    if exc.name and exc.name.startswith("comfy"):
        _upstream_sinusoidal_embedding_1d = None
    else:
        raise

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


def _iter_wrapper_children(obj: Any):
    for attr in ("model", "diffusion_model", "inner_model", "module", "wrapped_model", "_orig_mod"):
        child = getattr(obj, attr, None)
        if child is not None and child is not obj:
            yield attr, child


def _has_bindable_forward_orig(inner: Any) -> bool:
    return callable(getattr(inner, "forward_orig", None))


def _has_legacy_conditioning(inner: Any) -> bool:
    return callable(getattr(inner, "condition_embedder", None))


def _has_split_conditioning(inner: Any) -> bool:
    return (
        hasattr(inner, "freq_dim")
        and
        callable(getattr(inner, "time_embedding", None))
        and callable(getattr(inner, "time_projection", None))
        and callable(getattr(inner, "text_embedding", None))
    )


def _sinusoidal_embedding_1d(dim: int, timesteps):
    func = _upstream_sinusoidal_embedding_1d
    if func is None:
        try:
            from comfy.ldm.wan.model import sinusoidal_embedding_1d as func
        except ModuleNotFoundError as exc:
            if exc.name and exc.name.startswith("comfy"):
                raise RuntimeError("comfy.ldm.wan.model is unavailable for sinusoidal embedding fallback") from exc
            raise
    return func(dim, timesteps)


def _spectrum_runtime_missing_attrs(inner: Any) -> Tuple[str, ...]:
    base_required = (
        "blocks",
        "head",
        "patch_embedding",
        "unpatchify",
    )
    missing = [name for name in base_required if not hasattr(inner, name)]
    if not (_has_legacy_conditioning(inner) or _has_split_conditioning(inner)):
        missing.extend(
            [
                "condition_embedder|time_embedding",
                "condition_embedder|time_projection",
                "condition_embedder|text_embedding",
            ]
        )
    return tuple(missing)


def _bind_runtime_to_inner(
    inner: Any,
    runtime: SpectrumWanRuntime,
    inner_name: Optional[str] = None,
) -> bool:
    if inner is None or not _has_bindable_forward_orig(inner):
        return False

    inner._spectrum_wan_runtime = runtime
    missing = _spectrum_runtime_missing_attrs(inner)
    if not missing and hasattr(inner, "_forward") and callable(getattr(inner, "_forward")):
        _wrap_wan__forward_passthrough(inner)
    _wrap_wan_forward_orig(inner)

    runtime.last_info["patched"] = True
    runtime.last_info["hook_target"] = f"{inner_name}.forward_orig" if inner_name else "forward_orig"
    runtime.last_info["live_inner_type"] = type(inner).__name__
    runtime.last_info["live_inner_id"] = id(inner)
    runtime.last_info["runtime_missing_attrs"] = list(missing)
    runtime.last_info.pop("live_inner_root_type", None)
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
            transformer_options = kwargs.get("transformer_options")
            if transformer_options is None:
                transformer_options = {}
                kwargs["transformer_options"] = transformer_options
            if isinstance(transformer_options, dict):
                transformer_options[_RUNTIME_KEY] = current_runtime

            current_root = getattr(outer, "diffusion_model", None)
            current_inner, current_inner_name = _locate_wan_like_descendant(current_root, "model.diffusion_model")
            previously_bound_id = getattr(outer, "_spectrum_wan_bound_inner_id", None)
            bound = _bind_runtime_to_inner(current_inner, current_runtime, current_inner_name)
            current_inner_id = id(current_inner) if bound else None
            outer._spectrum_wan_bound_inner_id = current_inner_id

            if bound and previously_bound_id != current_inner_id:
                current_runtime._debug_log(
                    "[Spectrum WAN] rebound live inner "
                    f"path={current_inner_name} "
                    f"type={type(current_inner).__name__} "
                    f"id={current_inner_id}"
                )
            elif not bound:
                current_runtime.last_info["patched"] = False
                current_runtime.last_info["hook_target"] = current_inner_name or "model.diffusion_model"
                current_runtime.last_info["live_inner_root_type"] = (
                    type(current_root).__name__ if current_root is not None else None
                )
                current_runtime.last_info.pop("runtime_missing_attrs", None)
                current_runtime.last_info.pop("live_inner_type", None)
                current_runtime.last_info.pop("live_inner_id", None)
                current_runtime._debug_log(
                    "[Spectrum WAN] no WAN-like live inner under "
                    f"root_type={current_runtime.last_info['live_inner_root_type']}"
                )

        return original_apply_model(*args, **kwargs)

    outer.apply_model = wrapped_apply_model
    outer._spectrum_wan_apply_model_wrapped = True


def _looks_like_wan(inner: Any) -> bool:
    return _has_bindable_forward_orig(inner) and not _spectrum_runtime_missing_attrs(inner)


def _locate_wan_like_descendant(root: Any, root_name: str) -> Tuple[Optional[Any], Optional[str]]:
    if root is None:
        return None, root_name

    queue = [(root, root_name)]
    seen = set()
    fallback = None

    while queue:
        obj, name = queue.pop(0)
        obj_id = id(obj)
        if obj_id in seen:
            continue
        seen.add(obj_id)

        if _looks_like_wan(obj):
            return obj, name
        if fallback is None and _has_bindable_forward_orig(obj):
            fallback = (obj, name)

        for attr, child in _iter_wrapper_children(obj):
            queue.append((child, f"{name}.{attr}"))

    if fallback is not None:
        return fallback
    return None, root_name


def _locate_inner_model(model: Any) -> Tuple[Optional[Any], Optional[str]]:
    outer = getattr(model, "model", None)
    if outer is not None and hasattr(outer, "diffusion_model"):
        return _locate_wan_like_descendant(outer.diffusion_model, "model.diffusion_model")
    if hasattr(model, "diffusion_model"):
        return _locate_wan_like_descendant(model.diffusion_model, "diffusion_model")
    return None, None


def _resolve_runtime(
    transformer_options: Optional[Dict[str, Any]],
) -> Optional[SpectrumWanRuntime]:
    if isinstance(transformer_options, dict):
        runtime = transformer_options.get(_RUNTIME_KEY)
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

    transformer_options[_RUNTIME_KEY] = runtime
    decision = runtime.begin_step(transformer_options, t)
    step_idx = decision["step_idx"]
    actual_forward = decision["actual_forward"]

    # Original WAN preprocessing.
    x = inner.patch_embedding(x.float()).to(x.dtype)
    grid_sizes = x.shape[2:]
    transformer_options["grid_sizes"] = grid_sizes
    x = x.flatten(2).transpose(1, 2)
    context_img_len = None

    if _has_legacy_conditioning(inner):
        temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image = inner.condition_embedder(
            t, context, clip_fea
        )
        timestep_proj = timestep_proj.unflatten(1, (6, -1))

        if encoder_hidden_states_image is not None:
            encoder_hidden_states = torch.concat([encoder_hidden_states_image, encoder_hidden_states], dim=1)

        head_emb = temb
        context_img_len = clip_fea.shape[-2] if clip_fea is not None else None
    else:
        timestep_shape = t.shape
        head_emb = inner.time_embedding(
            _sinusoidal_embedding_1d(inner.freq_dim, t.reshape(-1)).to(dtype=x[0].dtype, device=t.device)
        )
        head_emb = head_emb.reshape(*timestep_shape, head_emb.shape[-1])
        time_proj = inner.time_projection(head_emb)
        timestep_proj = time_proj.unflatten(time_proj.ndim - 1, (6, -1))
        encoder_hidden_states = inner.text_embedding(context)
        encoder_hidden_states_image = None

        if clip_fea is not None and callable(getattr(inner, "img_emb", None)):
            encoder_hidden_states_image = inner.img_emb(clip_fea)
            encoder_hidden_states = torch.concat([encoder_hidden_states_image, encoder_hidden_states], dim=1)
            context_img_len = encoder_hidden_states_image.shape[1]

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
                x = inner.head(x, head_emb)
                if full_ref is not None:
                    x = x[:, full_ref.shape[1]:]
                out = inner.unpatchify(x, grid_sizes)
                runtime.end_step(transformer_options, step_idx)
                return out

    patches_replace = transformer_options.get("patches_replace", {})
    blocks_replace = patches_replace.get("dit", {})
    transformer_options["total_blocks"] = len(inner.blocks)
    transformer_options["block_type"] = "double"

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

    x = inner.head(x, head_emb)
    if full_ref is not None:
        x = x[:, full_ref.shape[1]:]
    out = inner.unpatchify(x, grid_sizes)
    runtime.end_step(transformer_options, step_idx)
    return out


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
        runtime = _resolve_runtime(transformer_options)
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

        missing = _spectrum_runtime_missing_attrs(inner)
        if missing:
            runtime.last_info["runtime_missing_attrs"] = list(missing)
            runtime._debug_log(
                "[Spectrum WAN] runtime path unavailable "
                f"inner_type={type(inner).__name__} "
                f"missing_attrs={','.join(missing)}"
            )
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
