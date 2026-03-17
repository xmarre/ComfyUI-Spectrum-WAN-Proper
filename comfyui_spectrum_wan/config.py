from __future__ import annotations

from dataclasses import dataclass


_VALID_BACKENDS = {
    "auto",
    "wan21",
    "wan22_ti2v_5b",
    "wan22_high_noise",
    "wan22_low_noise",
}


@dataclass
class SpectrumWanConfig:
    backend: str = "auto"
    enabled: bool = True
    blend_weight: float = 1.0
    degree: int = 4
    ridge_lambda: float = 0.10
    window_size: float = 2.0
    flex_window: float = 0.75
    warmup_steps: int = 5
    history_size: int = 16
    fit_chunk_size: int = 1_000_000
    debug: bool = False

    def validated(self) -> "SpectrumWanConfig":
        if self.backend not in _VALID_BACKENDS:
            raise ValueError(f"Unsupported backend '{self.backend}'.")
        if not (0.0 <= float(self.blend_weight) <= 1.0):
            raise ValueError("blend_weight must be in [0, 1].")
        if int(self.degree) < 1:
            raise ValueError("degree must be >= 1.")
        if float(self.ridge_lambda) < 0.0:
            raise ValueError("ridge_lambda must be >= 0.")
        if float(self.window_size) < 1.0:
            raise ValueError("window_size must be >= 1.")
        if float(self.flex_window) < 0.0:
            raise ValueError("flex_window must be >= 0.")
        if int(self.warmup_steps) < 0:
            raise ValueError("warmup_steps must be >= 0.")
        if int(self.history_size) < 2:
            raise ValueError("history_size must be >= 2.")
        if int(self.fit_chunk_size) < 1:
            raise ValueError("fit_chunk_size must be >= 1.")
        return self
