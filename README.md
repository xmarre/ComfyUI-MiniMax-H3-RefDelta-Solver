# ComfyUI MiniMax-H3 RefDelta Solver

A dedicated RefDelta-aware sampler family for the
[MiniMax-H3 Pruned Ref-Delta Fused rank-1024 checkpoint](https://huggingface.co/xmarre/MiniMax-H3-Pruned-Ref-Delta-Fused-r1024-ComfyUI).

The Stability Sampler can now use **ER-SDE, SEEDS-2, SEEDS-3, SA-Solver PEC, or SA-Solver PECE**
from one `base_sampler` dropdown. ER-SDE remains the saved-workflow and production
default until the additional backends complete the same real-media validation gate.

The target checkpoint is conceptually FL2VA plus a rank-1024 approximation of the
Ref2VA-from-FL2VA parameter delta, with exact/non-matrix delta pieces where applicable,
and the production checkpoint also contains INT8 ConvRot approximation effects. Genuine
Ref2VA is useful for same-state diagnostics of the imported delta. It is **not** a quality
oracle and is **not** used to drive the production scheduler.

## Production recommendation

Use:

- **MiniMax H3 RefDelta Stability Sampler**;
- `base_sampler = er_sde` for the currently validated production default, or select
  `seeds_2`, `seeds_3`, `sa_solver`, or `sa_solver_pece` for the corresponding enhanced backend;
- ComfyUI **BasicScheduler** with `scheduler = beta`;
- the normal MiniMax-H3 `ModelSamplingAV` path;
- Spectrum MiniMax H3 v0.2.20+ for the existing ER-SDE interop; RefDelta SEEDS/SA-Solver interoperability requires the companion Spectrum multi-backend interop change (PR #91 or a release containing it). Active RefDelta SA-Solver PECE additionally requires the updated #91 active-PECE composition contract.

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

#### Base sampler family

`base_sampler` selects the numerical solver underneath the shared RefDelta trajectory
and stochastic-stability policy:

| Value | Native solver geometry retained | RefDelta integration |
| --- | --- | --- |
| `er_sde` | ComfyUI ER-SDE stages/noise | Existing full RefDelta path, including adaptive ER order |
| `seeds_2` | Native two-stage SEEDS and correlated stochastic decomposition | Outer-trajectory correction plus one frozen stochastic gate reused across both correlated noise segments |
| `seeds_3` | Native three-stage SEEDS and correlated stochastic decomposition | Outer-trajectory correction plus one frozen stochastic gate reused across all three correlated noise segments |
| `sa_solver` | Native SA-Solver PEC Adams predictor/corrector | RefDelta correction on each PEC endpoint evaluation plus gated native predictor noise |
| `sa_solver_pece` | Native SA-Solver PECE predicted/corrected topology | P0 and exact corrected C_i calls own persistent RefDelta evidence; later predicted P_i calls stay ephemeral current-corrector inputs; native predictor noise is gated from the persistent endpoint |

The SEEDS internal denoiser stages are intentionally not rewritten by RefDelta. Their
multi-stage stochastic construction depends on correlated noise over overlapping intervals;
the RefDelta gate is resolved once per outer interval and reused for every native segment so
stage count cannot advance the controller or alter the solver's correlation structure.

`sa_solver` means the native PEC path (`use_pece = false`). `sa_solver_pece` uses the
same current ComfyUI defaults—`predictor_order = 3`, `corrector_order = 4`, and
`simple_order_2 = false`—with `use_pece = true`.

Active PECE is not treated as a two-stage fixed-stride sampler. Native ComfyUI calls
`P0, P1, C1, P2, C2, ...`: the predicted and corrected evaluations at an outer coordinate
share sigma but use different latent states. RefDelta mirrors native endpoint replacement:

```text
persistent RefDelta evidence = P0, C1, C2, C3, ...
ephemeral PECE inputs        = P1, P2, P3, ...
```

Only persistent endpoint calls update RefDelta trajectory derivatives, spatiotemporal
stability evidence, and the stochastic gate used for the following predictor noise. This
avoids double-counting the same outer coordinate and prevents a Spectrum-predicted P_i from
entering persistent RefDelta evidence. Trajectory correction is likewise applied to the
persistent endpoint value, not twice to both P_i and C_i.

`s_noise` is shared by all five backends. `max_stage` and `adaptive_order` are
ER-SDE-specific numerical-order controls; SEEDS has its own fixed native stage geometry and
SA-Solver has its native Adams orders. Calibration capture remains ER-SDE-only because its
saved-state format is defined on the original ER-SDE outer trajectory.

With RefDelta controls disabled, each new backend delegates directly to the corresponding
native ComfyUI sampler. The instrumented zero-adaptation path is also covered by native
same-input/same-noise parity fixtures so the wrapper cannot silently perturb solver behavior.

The additional backends are newly implemented and automated-parity tested. They are not
described here as media-validated production replacements for ER-SDE yet.

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

### MiniMax H3 Uniform Flow Scheduler [Experimental]

This separate scheduler node grew out of the strongest scheduler lead seen in real
MiniMax-H3 testing: ComfyUI `ddim_uniform` produced many quick cuts, coherent action,
and unusually high scene/action variety. The released laboratory keeps that exact control
but now defaults to the best setting from the completed 19-step MiniMax-H3 + ER-SDE
media A/B pass:

```text
steps = 19
mode = phase_offset_uniform
phase = 0.50
denoise = 1.00
```

The phase sweep was materially informative. `0.60` and `0.65` reintroduced visible
flashing; `0.55` removed the flashing but produced weaker action than `0.50`; `0.45`
was worse again. `0.50` gave the best observed balance of coherent action, stable
visuals, useful shot variety, and audio behavior in the tested workflow. Small phase
changes also altered spoken intonation, confirming that placement changes the shared AV
trajectory rather than acting as a video-only quality knob.

Other scheduler families remain available as research controls. The real-media pass found
`uniform_linspace` solid; neutral `power_uniform` and neutral
`piecewise_structure_refinement` matched it as expected. `uniform_refinement_tail`
and `trailing_refined` could produce very good results, but the refinement-tail family
was more inventive/unstable across repeated tests. `av_arc_length` produced strong,
varied shots but also an artifacted shot in the tested pass. Default
`asymmetric_beta(0.6, 0.6)` was unstable, with rapid shot repetition and doubled audio.
These observations are empirical results from the tested H3 workflow, not universal
rankings for every prompt, checkpoint, or step count.

Connect the node's `SIGMAS` output to the same custom sampling path and continue using
**ER-SDE**. “DDIM” here describes time-point spacing; this node does not use a DDIM sampler
or replace ER-SDE. The current experimental recommendation is the default
`phase_offset_uniform / 0.50 / 19 steps` preset. The conservative production fallback
remains the RefDelta Stability Sampler with stock `BasicScheduler(beta)` until broader
cross-prompt and cross-seed validation justifies changing the global recommendation.

`legacy_ddim_uniform` delegates to ComfyUI's own scheduler and reproduces the full
`BasicScheduler` behavior, including table index 1, integer floor stride, reversal,
occasional extra point, and final tail slice. This unusual placement remains the exact
high-variety control. Exact parity is tested against the pinned ComfyUI revisions in CI.

All other modes choose one descending trajectory on H3's shared base flow coordinate `u`
and map it through the loaded model's video shift. H3's `ModelSamplingAV` and model code
remain authoritative for the corresponding audio time. The node reads `shift` and
`audio_shift` from the loaded model; it never constructs independent audio/video clocks.

Available modes:

| Mode | Definition / experiment |
| --- | --- |
| `legacy_ddim_uniform` | Exact current ComfyUI control, including integer table-stride aliasing |
| `uniform_linspace` | Inclusive `u=1..0` linspace with `steps+1` points |
| `phase_offset_uniform` | Nonterminal `u_i = 1 - (i + phase) / steps`, then exact zero; released default is `phase=0.50` |
| `power_uniform` | `u(x)=(1-x)^power`; `power=1` is exactly `uniform_linspace` |
| `uniform_refinement_tail` | Uniform body from `u=1` to explicit `tail_start`, then a power-refined tail using `tail_steps` |
| `trailing_refined` | Uses Diffusers-style final training-index normalization `(N-1)/N`, joins at `tail_start`, and explicitly refines to zero |
| `asymmetric_beta` | Continuous beta quantiles in shared base time; deliberately distinct from stock rounded-table `beta` |
| `av_arc_length` | Blends uniform placement with equal joint video/audio shifted-flow arc length |
| `piecewise_structure_refinement` | Continuous two-segment progress warp with independent structure/detail powers |
| `curvature_profile` | Immutable offline production-telemetry density; bundled `h3_uniform_neutral` is neutral |

Controls are marked advanced where appropriate and affect only their named modes. In
particular, `phase`, `power`, tail controls, beta parameters, arc controls, piecewise
controls, `profile`, and `profile_path` are inert outside their respective mode.
`auto_tail_steps=true` selects `min(5, effective_steps - 1)` after denoise expansion;
disable it to use the explicit `tail_steps` value. An explicit `tail_steps=0` remains the
exact uniform-baseline identity.

Continuous modes whose definition starts at `u=1` deliberately retain that endpoint.
Current ER-SDE applies ComfyUI's `offset_first_sigma_for_snr` safety adjustment to such a
first point for a `CONST` flow model; subsequent points are unchanged. Phase-offset,
positive-tail `trailing_refined`, and `legacy_ddim_uniform` intentionally start below one.

`denoise < 1` follows `BasicScheduler` semantics: the node constructs the corresponding
longer schedule and returns its exact tail. Outputs are deterministic CPU float32 tensors,
strictly decreasing, finite, and terminated by exact zero. Invalid H3 metadata, duplicate
points, impossible tails, and legacy step counts beyond the table's unique capacity fail
explicitly.

Schedule curves and implied audio times can be exported without running the model:

```bash
python tools/inspect_h3_schedule.py --mode phase_offset_uniform --steps 19 --phase 0.50 --format csv
```

The curvature mode uses a version-2 `shared-base-time-progress-density` schema so legacy
version-1 beta-prior profiles cannot be silently reinterpreted. Build a profile from
production trajectory telemetry and pass its path through the advanced `profile_path`
control:

```bash
python tools/build_profile.py \
  ComfyUI/output/refdelta_telemetry/*.csv \
  --output /tmp/h3_shared_flow_research.json \
  --id h3_shared_flow_research \
  --experimental-stability-density \
  --shared-flow-density \
  --video-shift 12.0
```

The builder removes `comparison_*` and `ref_delta_*` fields before binning and marks the
result as non-production. Experimental v2 density accepts only rows explicitly marked
`actual_model_evaluation=true`; missing or forecast-only telemetry fails closed rather than
redistributing cached Spectrum observations into forecast sigma bins. `--video-shift` must
match the source telemetry run; it is stored in the profile and used to invert video sigma
back to shared base time. Existing version-1 beta-prior profiles and the legacy scheduler
retain their original semantics.

The released default reflects real final-media A/B testing, but scheduler quality remains
prompt-, seed-, checkpoint-, and step-count-dependent. Keep `legacy_ddim_uniform`,
`uniform_linspace`, and stock `beta` available as controls when validating another
workflow.

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
- preserves version-1 beta-prior output by default and emits version-2 shared-base-time
  density only with `--shared-flow-density`.

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
