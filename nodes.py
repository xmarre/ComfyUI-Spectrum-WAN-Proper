from __future__ import annotations

from .comfyui_spectrum_wan.config import SpectrumWanConfig
from .comfyui_spectrum_wan.wan import WanSpectrumPatcher


class SpectrumApplyWAN:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "backend": (
                    [
                        "auto",
                        "wan21",
                        "wan22_ti2v_5b",
                        "wan22_high_noise",
                        "wan22_low_noise",
                    ],
                    {"default": "auto"},
                ),
                "transition_mode": (
                    [
                        "separate_fit",
                        "bias_shift",
                    ],
                    {"default": "separate_fit"},
                ),
                "enabled": ("BOOLEAN", {"default": True}),
                "blend_weight": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "degree": ("INT", {"default": 4, "min": 1, "max": 16, "step": 1}),
                "ridge_lambda": ("FLOAT", {"default": 0.10, "min": 0.0, "max": 10.0, "step": 0.01}),
                "window_size": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 32.0, "step": 0.05}),
                "flex_window": ("FLOAT", {"default": 0.75, "min": 0.0, "max": 16.0, "step": 0.05}),
                "warmup_steps": ("INT", {"default": 5, "min": 0, "max": 64, "step": 1}),
                "history_size": ("INT", {"default": 16, "min": 2, "max": 128, "step": 1}),
                "debug": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "apply"
    CATEGORY = "sampling/spectrum"

    def apply(
        self,
        model,
        backend,
        transition_mode,
        enabled,
        blend_weight,
        degree,
        ridge_lambda,
        window_size,
        flex_window,
        warmup_steps,
        history_size,
        debug,
    ):
        if not enabled:
            return (model,)

        cfg = SpectrumWanConfig(
            backend=backend,
            transition_mode=transition_mode,
            enabled=enabled,
            blend_weight=blend_weight,
            degree=degree,
            ridge_lambda=ridge_lambda,
            window_size=window_size,
            flex_window=flex_window,
            warmup_steps=warmup_steps,
            history_size=history_size,
            debug=debug,
        ).validated()
        return (WanSpectrumPatcher.patch(model, cfg),)


NODE_CLASS_MAPPINGS = {
    "SpectrumApplyWAN": SpectrumApplyWAN,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SpectrumApplyWAN": "Spectrum Apply WAN",
}
