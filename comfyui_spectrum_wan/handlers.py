from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional


@dataclass(frozen=True)
class WanBackendHandler:
    backend_id: str
    phase_tag: str
    is_moe_expert: bool = False

    def stream_namespace(self) -> str:
        return self.phase_tag


_HANDLERS: Dict[str, WanBackendHandler] = {
    "wan21": WanBackendHandler("wan21", "wan21"),
    "wan22_ti2v_5b": WanBackendHandler("wan22_ti2v_5b", "wan22_ti2v_5b"),
    "wan22_high_noise": WanBackendHandler("wan22_high_noise", "wan22_high_noise", is_moe_expert=True),
    "wan22_low_noise": WanBackendHandler("wan22_low_noise", "wan22_low_noise", is_moe_expert=True),
}


def _candidate_strings(model: Any) -> Iterable[str]:
    keys = (
        "model_name",
        "diffusion_model_name",
        "filename",
        "model_file",
        "ckpt_name",
        "title",
        "name",
    )
    seen = []
    for obj in (model, getattr(model, "model", None), getattr(getattr(model, "model", None), "diffusion_model", None)):
        if obj is None:
            continue
        for key in keys:
            value = getattr(obj, key, None)
            if isinstance(value, str) and value not in seen:
                seen.append(value)
                yield value.lower()


def resolve_handler(backend: str, model: Any) -> WanBackendHandler:
    if backend != "auto":
        return _HANDLERS[backend]

    joined = " | ".join(_candidate_strings(model))
    if "wan2.2" in joined or "wan22" in joined:
        if "high_noise" in joined:
            return _HANDLERS["wan22_high_noise"]
        if "low_noise" in joined:
            return _HANDLERS["wan22_low_noise"]
        if "ti2v" in joined or "5b" in joined:
            return _HANDLERS["wan22_ti2v_5b"]
    return _HANDLERS["wan21"]


def auto_backend_warning(backend: str, model: Any, handler: WanBackendHandler) -> Optional[str]:
    if backend != "auto":
        return None

    joined = " | ".join(_candidate_strings(model))
    if not joined:
        return (
            f"Spectrum WAN auto backend resolved to {handler.backend_id} without model metadata. "
            "Set backend explicitly if using Wan 2.2."
        )
    if handler.backend_id == "wan21":
        return (
            "Spectrum WAN auto backend resolved to wan21 from ambiguous metadata. "
            "Set backend explicitly if using Wan 2.2."
        )
    if handler.backend_id == "wan22_ti2v_5b" and "high_noise" not in joined and "low_noise" not in joined:
        return (
            "Spectrum WAN auto backend resolved to wan22_ti2v_5b without high/low expert metadata. "
            "Set backend explicitly for Wan 2.2 14B expert workflows."
        )
    return None


def handler_metadata(handler: WanBackendHandler) -> Dict[str, object]:
    return {
        "backend_id": handler.backend_id,
        "phase_tag": handler.phase_tag,
        "is_moe_expert": handler.is_moe_expert,
    }
