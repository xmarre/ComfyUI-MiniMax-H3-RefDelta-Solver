# ComfyUI MiniMax-H3 RefDelta Solver

A dedicated ER-SDE-derived sampler for the
[MiniMax-H3 Pruned Ref-Delta Fused rank-1024 checkpoint](https://huggingface.co/xmarre/MiniMax-H3-Pruned-Ref-Delta-Fused-r1024-ComfyUI).

The target checkpoint is conceptually FL2VA plus a rank-1024 approximation of the
Ref2VA-from-FL2VA parameter delta, with exact/non-matrix delta pieces where applicable,
and the production checkpoint also contains INT8 ConvRot approximation effects. Genuine
Ref2VA is useful for same-state diagnostics of the imported delta. It is **not** a quality
oracle and is **not** used to drive the production scheduler.

## Production recommendation

Use:

- **MiniMax H3 RefDelta Stability Sampler**;
- ComfyUI **BasicScheduler** with `scheduler = beta`;
- the normal MiniMax-H3 `ModelSamplingAV` path;
- Spectrum MiniMax H3 v0.2.20+ when Spectrum forecasting is enabled.

The custom RefDelta scheduler is retained only for saved-workflow compatibility and
scheduler research. The bundled `r1024_provisional` profile is neutral and should not be
interpreted as a calibrated production schedule.

## Install

Clone this repository into `ComfyUI/custom_nodes` and restart ComfyUI:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/xmarre/ComfyUI-MiniMax-H3-RefDelta-Solver.git
```

The custom node has no runtime dependencies beyond current ComfyUI.

## Nodes

### MiniMax H3 RefDelta Stability Sampler

This is the recommended production node. Its production defaults are built directly into
the node while every production-relevant knob remains editable. There is no separate
sampler profile or calibration file to select. Diagnostic capture controls and legacy
controller-mode selection are kept out of this node so ordinary workflows do not mix
production tuning with research capture state.

Current empirically validated defaults:

```text
adaptive_order = true
risk_sensitivity = 1.00

stochastic_adaptation_strength = 0.50
minimum_stochastic_multiplier = 0.50

trajectory_correction = true
video_correction_strength = 0.15
audio_correction_strength = 0.05
correction_bound = 0.50
endpoint_fidelity_fraction = 0.15

s_noise = 1.00
max_stage = 3

video_stochastic_strength_scale = 1.00
audio_stochastic_strength_scale = 1.00
static_video_stochastic_adaptation_strength = 0.25
video_stability_restore_strength = 1.00

video_stability_motion_low = 0.15
video_stability_motion_high = 0.60
video_stability_diffusion_low = 0.05
video_stability_diffusion_high = 0.50
video_stability_diffusion_weight = 0.20
video_stability_normalization_floor = 0.10
video_stability_gamma = 1.00

video_stability_spatial_radius = 2
video_stability_temporal_radius = 2
video_stability_ema = 0.70
video_stability_start_fraction = 0.10
video_stability_full_fraction = 0.30
stochastic_gate_slew_limit = 0.00
```

These values are the current production candidate for the rank-1024 INT8 ConvRot checkpoint,
not a claim that one setting is universally optimal for every prompt/reference stack. The
node keeps the knobs exposed specifically so tuning does not require code changes.

Same-seed validation of the streamlined Stability Sampler against the advanced node produced
cell-for-cell identical CSV telemetry and byte-for-byte identical JSONL telemetry when both
were configured with these values. The streamlined node therefore changes workflow UX, not
the sampler/controller path.

#### What the stability controller does

The dynamic video policy is driven by the actual-only video risk. The audio stream has its
own scalar risk/gate. Video additionally receives a smooth `[B,1,T,H,W]` latent-space
stability map built from:

- channel-RMS motion between neighboring latent time slices;
- channel-RMS change from the previous **actual** denoised prediction;
- configurable temporal/diffusion thresholds and diffusion weighting;
- gamma shaping, spatial/temporal smoothing, actual-only EMA, and a progress ramp.

Static regions interpolate toward `static_video_stochastic_adaptation_strength`; moving or
uncertain regions remain closer to the dynamic `stochastic_adaptation_strength` policy.
The controller does not decode pixels, use optical flow, or add model evaluations/NFE.

The static target is deliberately a separate policy, not a global stochastic floor. In real
artifact testing this distinction mattered: globally raising the minimum multiplier could
introduce its own jitter, while selective low-motion restoration could suppress fine-detail
background shimmer without forcing the whole video onto the same stochastic behavior.

`endpoint_fidelity_fraction` remains independent and fades the final adapted gate toward
native multiplier `1` near the endpoint. Optional `stochastic_gate_slew_limit` acts before
endpoint restoration; `0` disables it.

#### Main tuning controls

| Control | Effect |
| --- | --- |
| `stochastic_adaptation_strength` | Dynamic-region adaptation strength |
| `minimum_stochastic_multiplier` | Global lower bound on the stochastic multiplier |
| `static_video_stochastic_adaptation_strength` | Static-region target policy |
| `video_stability_motion_low/high` | Which latent regions count as temporally stable |
| `video_stability_diffusion_low/high` | Whether those regions are also settled across actual denoising steps |
| `video_stability_diffusion_weight` | Importance of actual-to-actual stability |
| `video_stability_gamma` | Selectivity of restoration confidence |
| spatial/temporal radii + EMA | Smoothness/persistence of the stability map |
| start/full fractions | When stability restoration becomes active |
| `stochastic_gate_slew_limit` | Optional per-step gate-change bound |

Practical tuning order:

1. Keep `stochastic_adaptation_strength`, the stochastic floor, and static target separate.
2. If static detail still shimmers, inspect telemetry/maps before widening thresholds.
3. If the map identifies the correct regions but restoration is too weak, tune the static
   target rather than globally raising the floor.
4. If composition/motion changes too much, make restoration more selective with thresholds
   or gamma before applying a stronger policy everywhere.
5. Keep audio tuning independent through `audio_stochastic_strength_scale`.

### [Advanced/Diagnostic] MiniMax H3 RefDelta Sampler

This is the original node ID retained for saved workflows, manual controller experiments,
and diagnostic capture. Its defaults intentionally preserve released behavior:

- `stochastic_control_mode = legacy_global`;
- `stochastic_adaptation_strength = 0.50`;
- `minimum_stochastic_multiplier = 0.25`;
- `trajectory_correction = false`.

It additionally exposes:

- `legacy_global`, `streamwise`, and `spatiotemporal_stability` controller selection;
- `calibration_capture` / `calibration_id` for disk-backed diagnostic capture;
- the same detailed stability controls as the production node.

The legacy defaults are **not** the current recommended rank-1024 production defaults; they
exist so older saved workflows do not silently change behavior after upgrading.

For a strict stock ER-SDE baseline:

```text
adaptive_order = false
stochastic_adaptation_strength = 0
trajectory_correction = false
debug_telemetry = false
calibration_capture = false
```

That path delegates directly to current ComfyUI `sample_er_sde`.

### MiniMax H3 RefDelta Comparison Replay

This is the preferred same-state diagnostic path. The historical `calibration_*` names are
kept for workflow/file compatibility, but the data is now treated explicitly as
**diagnostic comparison data**, not scheduler calibration.

A complete comparison uses three executions of the same workflow while loading one full H3
model at a time.

#### Pass 1 — fused diagnostic capture

Use the advanced/diagnostic sampler with the fused model and enable:

```text
calibration_capture = true
calibration_id = int8-convrot-r1024-test-01
```

For a clean vector-field comparison, use stock beta, Spectrum off, and optionally disable
sampler adaptations:

```text
adaptive_order = false
stochastic_adaptation_strength = 0
trajectory_correction = false
```

The capture writes exact packed sampler states, fused x0 values, sigmas,
actual/forecast classification, scalar telemetry, and the final pre-inverse-scaling sampler
state under:

```text
ComfyUI/output/refdelta_calibration/<calibration_id>/
```

#### Pass 2 — FL2VA

Switch the MODEL loader to FL2VA, replace the sampler with
`MiniMax H3 RefDelta Comparison Replay`, use the same `calibration_id`, and set:

```text
comparison_label = fl2va
```

Only captured actual steps are evaluated at the exact captured state/sigma.

#### Pass 3 — Ref2VA

Switch the MODEL loader to genuine Ref2VA and set:

```text
comparison_label = ref2va
```

The completed FL2VA pass is required. Replay measures the true `Ref2VA - FL2VA` field
against the fused `fused - FL2VA` field, including video/audio cosine, magnitude,
projection, residual, and orthogonal components.

Comparison replay always returns the captured fused final sampler state, preserving the
original Continuum trajectory across diagnostic passes. Spectrum is rejected on comparison
passes. FL2VA/Ref2VA metrics are explanatory and never enter production stochastic control
or scheduler density automatically.

### [Legacy/Research] MiniMax H3 RefDelta Scheduler

The custom profile-density scheduler is deprecated for ordinary production workflows.
Existing saved workflows continue to work, and the scheduler/profile code remains available
for explicit research.

For production, use ComfyUI:

```text
BasicScheduler
scheduler = beta
```

The bundled `r1024_provisional` profile is a neutral beta-prior compatibility profile.
There is no production calibration file to select from the FL2VA/Ref2VA replay workflow.

`tools/build_profile.py` is therefore research-only. It:

- strips `comparison_*` and `ref_delta_*` fields before scheduler binning;
- deduplicates replay copies of the same production telemetry;
- emits neutral difficulty by default;
- requires `--experimental-stability-density` for any non-neutral density;
- marks generated profiles as non-production research output.

Example research use:

```bash
python tools/build_profile.py \
  ComfyUI/output/refdelta_telemetry/*.csv \
  --output /tmp/r1024_scheduler_research.json \
  --id r1024_scheduler_research \
  --experimental-stability-density
```

Do not use Ref2VA distance as a scheduler error metric.

## Risk/evidence model

RefDelta keeps two histories:

- solver history: every value returned to ER-SDE, including Spectrum forecasts;
- evidence history: actual model evaluations only.

Adaptive ER stage order uses deterministic trajectory risk. Stochastic adaptation uses
combined trajectory/stochastic evidence. Stochastic pressure is measured from the **native
pre-gate** ER-SDE increment relative to real denoised movement, preventing the controller
from reducing its own future evidence.

Spectrum forecasts may consume the latest actual-only risk, stochastic pressure, and cached
stability map, but they never become motion/diffusion/EMA/x0 evidence. Spectrum receives the
exact final stochastic increment actually added to the state.

## Telemetry and stability maps

When `debug_telemetry` is enabled, scalar JSONL/CSV telemetry is written under:

```text
ComfyUI/output/refdelta_telemetry/
```

It includes trajectory/stochastic risk, native/applied stochastic values, controller mode,
dynamic/static/audio targets, applied video-gate distribution, restore-mask distribution,
motion/diffusion summaries, progress/source state, and slew activity.

When both `debug_telemetry` and `debug_stability_maps` are enabled, nonterminal actual steps
also write compressed NPZ maps containing:

- temporal motion ratio;
- diffusion-change ratio when available;
- final restore mask;
- final applied video gate;
- seed, step, sigma, video shape, and controller config metadata.

Forecasts never emit evidence maps.

## Spectrum compatibility

The reviewed baseline is
[ComfyUI-Spectrum-MiniMax-H3 v0.2.20+](https://github.com/xmarre/ComfyUI-Spectrum-MiniMax-H3).

Important invariants:

- forecasts stay in solver history but out of RefDelta evidence history;
- the immediate forecast after an actual step consumes that actual step's latest stochastic
  evidence;
- forecast gaps preserve the last valid actual native stochastic/movement ratios;
- trajectory correction may target a forecast but uses only actual anchors/derivatives;
- spatiotemporal stability evidence updates only on actual model calls;
- Spectrum receives the exact post-controller stochastic increment;
- missing/stale/mismatched bridge state fails explicitly.

## Static checkpoint context

The published rank-1024 sidecar shows a mixed approximation rather than a uniform rank
limit: exact vector/bias pieces coexist with rank-limited matrix adapters. Overall retained
energy is dominated by exact/small tensors, while actually compressed matrices retain much
less of the original delta. This motivates trajectory-aware handling, but it does not make
Ref2VA distance a quality metric or scheduler oracle.

## Development

```bash
python -m pip install pytest ruff
python -m pytest -q
python -m ruff check .
python -m compileall -q comfyui_refdelta_solver tests tools
```

CI also checks native ER-SDE fixture parity across pinned ComfyUI revisions and builds the
installable wheel.

## License

GPL-3.0-or-later. The ER-SDE implementation is derived from GPL-licensed ComfyUI sampling
code; see [LICENSE](LICENSE).