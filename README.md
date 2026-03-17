# ComfyUI Spectrum WAN Proper

Native ComfyUI custom node repo that ports [Spectrum](https://hanjq17.github.io/Spectrum/) from [Adaptive Spectral Feature Forecasting for Diffusion Sampling Acceleration](https://arxiv.org/abs/2603.01623) to the **WAN video backend** with **backend-specific handlers** for:

- **Wan 2.1**
- **Wan 2.2 TI2V 5B**
- **Wan 2.2 14B High-Noise expert**
- **Wan 2.2 14B Low-Noise expert**

This is intentionally one WAN-focused repository instead of a generic "video Spectrum for everything" package.

## What this repo does

Spectrum is a **training-free diffusion acceleration** method that caches and forecasts denoiser features over time instead of running the expensive network at every sampling step.

For the native ComfyUI WAN backend, this repo:

1. patches the internal **`WanModel.forward_orig(...)`** path,
2. caches the **final hidden transformer feature** after the last WAN block,
3. fits an online **Chebyshev + ridge regression** forecaster,
4. predicts the final hidden feature on skipped steps,
5. runs the normal WAN **head + unpatchify** on the predicted feature.

That is the WAN analogue of the official [Spectrum](https://hanjq17.github.io/Spectrum/) design: forecast the **final block output**, not arbitrary module activations or the outer model wrapper output.

## Why WAN gets backend-specific handlers

Wan 2.1 and Wan 2.2 are close enough to live in one repo, but not close enough to be treated as one undifferentiated backend.

### Wan 2.1 / Wan 2.2 5B

These are handled as **single-model WAN paths**.

### Wan 2.2 14B

Wan 2.2 14B uses **separate high-noise and low-noise expert models** in the native ComfyUI workflows.

This repo therefore treats them as **separate Spectrum targets**:

- one patched high-noise expert model
- one patched low-noise expert model

This is deliberate. It avoids pretending the two experts share one stationary feature trajectory when ComfyUI actually loads and samples them as distinct diffusion models.

## Installation

Clone or copy this repository into your ComfyUI `custom_nodes` directory:

```bash
git clone https://github.com/xmarre/ComfyUI-Spectrum-WAN-Proper ComfyUI/custom_nodes/ComfyUI-Spectrum-WAN-Proper
```

Then restart ComfyUI.

No additional dependencies are required beyond normal ComfyUI requirements.

## Node

### `Spectrum Apply WAN`

**Inputs**

- `model` — ComfyUI `MODEL`
- `backend` — backend handler selection
  - `auto`
  - `wan21`
  - `wan22_ti2v_5b`
  - `wan22_high_noise`
  - `wan22_low_noise`
- `enabled` — enable/disable Spectrum patching
- `blend_weight` — `w` in the official implementation; blend between local linear extrapolation and Chebyshev prediction
- `degree` — Chebyshev polynomial degree `M`
- `ridge_lambda` — ridge regularization `λ`
- `window_size` — initial practical scheduling window
- `flex_window` — amount added to the window after each actual forward
- `warmup_steps` — number of initial actual forwards
- `history_size` — number of actual WAN features to retain for fitting
- `debug` — stored in runtime metadata only

**Output**

- patched `MODEL`

## Recommended defaults

These defaults match the official Spectrum WAN setup closely enough for native ComfyUI usage:

```text
blend_weight = 1.0
degree = 4
ridge_lambda = 0.10
window_size = 2.0
flex_window = 0.75
warmup_steps = 5
history_size = 16
```

### Why `history_size = 16` instead of 100?

The official reference implementation uses a large history cap, but the recommended adaptive WAN settings in the paper consume only **14** or **10** actual network evaluations in the important regimes. A history cap of `16` therefore preserves all actual points in those standard settings while materially reducing memory pressure for WAN video features.

That is an explicit practical approximation in this repo.

## Usage

## Wan 2.1

Place the node directly after the WAN diffusion model loader:

```text
Load Diffusion Model (Wan 2.1)
  -> Spectrum Apply WAN (backend = wan21)
  -> sampler
```

## Wan 2.2 TI2V 5B

```text
Load Diffusion Model (Wan 2.2 TI2V 5B)
  -> Spectrum Apply WAN (backend = wan22_ti2v_5b)
  -> sampler
```

## Wan 2.2 14B T2V / I2V

Apply the node **separately to each expert**:

```text
Load Diffusion Model (Wan 2.2 high-noise 14B)
  -> Spectrum Apply WAN (backend = wan22_high_noise)

Load Diffusion Model (Wan 2.2 low-noise 14B)
  -> Spectrum Apply WAN (backend = wan22_low_noise)
```

Use those patched models in place of the original high-noise / low-noise expert models inside the existing ComfyUI Wan 2.2 workflow.

## Implementation notes

### Native hook point

The patch targets the internal WAN path **after the final transformer block loop and before the WAN head / unpatchify**.

This is the correct Spectrum target for WAN because the paper’s practical recommendation is to forecast the **final block output** rather than maintaining per-block caches.

### Patch scope

The patch keeps ComfyUI’s existing WAN features intact:

- `patches_replace` block replacement support
- I2V image context handling
- reference latent path via `ref_conv`
- normal WAN head and unpatchify behavior

### Schedule tracking

The runtime tracks runs from `transformer_options["sample_sigmas"]` and `transformer_options["sigmas"]` when available.

It also separates stream state by `cond_or_uncond` signature when ComfyUI provides one, so it does not accidentally reuse one runtime state across mismatched conditional/unconditional call patterns.

### Fit memory handling

WAN hidden features are very large. To keep the implementation practical, the ridge solve is implemented in **feature-dimension chunks**, which avoids the worst transient float32 allocations from a naive `(K, F)` full-matrix solve.

The final coefficient tensor is still large, because that is intrinsic to Spectrum’s feature-level forecasting approach.

## Assumptions, caveats, and limitations

### Targeted backend scope

This repo targets the native ComfyUI WAN backend path, i.e. `WanModel`-style diffusion backbones.

It is not a wrapper for LTX, Hunyuan, Cosmos, or third-party WAN wrappers.

### Wan 2.2 expert handling

For Wan 2.2 14B, this repo does **not** attempt to create one merged forecaster across both experts.

Instead, each expert model gets its own Spectrum runtime and its own cached feature history. That matches the native ComfyUI workflow structure and is the safer design.

### Internal-block custom patches on skipped steps

Any custom node that depends on **executing internal WAN blocks on every step** will naturally not see those block-level effects on skipped steps. Spectrum avoids those full block executions by design.

### Memory pressure

Even with final-block-only caching, WAN video features are large. `history_size`, frame count, resolution, and hidden width all affect VRAM pressure.

### Practical approximation versus the paper

This repo is faithful to the core method, but makes two explicit practical adaptations:

1. `history_size = 16` by default instead of keeping a much larger history cap.
2. The adaptive schedule follows the **practical official implementation behavior** used by public Spectrum code (`window_size` + `flex_window`) rather than exposing the paper’s triangular-step formula directly as the user-facing control surface.

### Future ComfyUI changes

This repo reproduces the current internal WAN `forward_orig(...)` structure closely. If ComfyUI changes the internal WAN model implementation later, this node may need a maintenance update.

## Repo structure

```text
ComfyUI-Spectrum-WAN-Proper/
├── __init__.py
├── nodes.py
├── pyproject.toml
├── LICENSE
├── README.md
├── comfyui_spectrum_wan/
│   ├── __init__.py
│   ├── config.py
│   ├── handlers.py
│   ├── forecast.py
│   ├── runtime.py
│   └── wan.py
└── tests/
    └── smoke_runtime.py
```

## Transition modes

`Spectrum Apply WAN` supports two expert-transition modes:

- `separate_fit` keeps the default per-expert reset / re-fit behavior.
- `bias_shift` is an experimental Wan 2.2 high-noise to low-noise handoff that forces the first low-noise step actual, computes a 1-step bias correction, refreshes that bias on later actual low-noise refresh steps, and uses the transferred high-noise predictor on forecast-eligible low-noise steps.

For Wan 2.2 expert workflows, set `transition_mode = bias_shift` on both expert nodes if you want the experimental handoff. The high-noise expert publishes the transfer state and the low-noise expert consumes it. If the handoff is missing or incompatible, the low-noise expert falls back to the normal per-expert fit path. `spectrum_wan_run_token` must stay consistent across both phases, and `spectrum_wan_global_step_override` is only needed if the low phase is not contiguous from the published high-phase boundary.

## Smoke test

A lightweight non-ComfyUI test is included:

```bash
PYTHONPATH=. python tests/smoke_runtime.py
```

Expected output:

```text
ok
```

## References and credits

- [Adaptive Spectral Feature Forecasting for Diffusion Sampling Acceleration](https://arxiv.org/abs/2603.01623)
- [Spectrum project page](https://hanjq17.github.io/Spectrum/)
- [Official Spectrum code](https://github.com/hanjq17/Spectrum)

This repository is an unofficial ComfyUI adaptation of the original Spectrum method for native WAN backends. Credit for the underlying method, paper, and reference implementation goes to the original Spectrum authors.
