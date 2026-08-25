# Unreleased

## Spatiotemporal stochastic stability

- Added `legacy_global`, `streamwise`, and `spatiotemporal_stability` stochastic-control modes. `legacy_global` remains the default and preserves released packed-scalar behavior.
- Added independent actual-only video/audio stochastic targets and strength scales. Audio never consumes the video stability map.
- Added a channel-reduced H3 latent controller combining normalized neighbor-frame motion with change from the preceding actual x0. It applies a configurable dynamic-policy-to-static-policy interpolation without pixel decode, extra model calls, optical flow, or additional NFE.
- Added replicate-safe separable temporal/spatial smoothing, actual-only EMA, diffusion-progress ramp, normalization floor, gamma response, and an optional per-cell/scalar gate slew limit. Endpoint fidelity is applied after slew limiting.
- Extended packed `StreamLayout` with validated `(C,T,H,W)` reconstruction and repacking helpers. Incompatible latent shapes reset cached stability and gate state instead of reusing stale maps.
- Preserved Spectrum's actual-only evidence invariant: forecasts reuse cached stability/risk, never update motion/diffusion/EMA anchors, and receive the exact final gated increment added to the sampler state. Native pre-gate stochasticity remains the pressure-evidence source.
- Added scalar telemetry for targets, applied-gate and restore distributions, motion/diffusion summaries, progress/source state, and slew activity. Optional actual-only compressed NPZ map export provides the channel-reduced motion, diffusion, restore, and final gate maps with seed/step/sigma/config metadata.
- Added regression coverage for legacy equivalence, stream independence, static/moving separation, diffusion suppression, collapsed policies, finite bounds, smoothing edges, actual-only EMA, forecast contamination, shape resets, endpoint ordering, slew behavior, exact Spectrum publication, telemetry, debug-map export, invalid layouts, and native delegation with inactive adaptation.

## Labeled comparison calibration

- Generalized the disk-backed replay system into labeled comparison passes. One fused INT8 ConvRot capture can now be replayed separately through FL2VA and genuine Ref2VA while loading one full H3 model per execution.
- `MiniMax H3 RefDelta Sampler` can now capture each exact packed sampler state, fused x0, sigma schedule, actual/forecast classification, baseline telemetry, and final pre-inverse-scaling sampler tensor under an explicit `calibration_id`.
- Capture tensors are persisted step-by-step as safetensors rather than retained for the complete run in CPU/GPU memory. Interrupted captures are marked incomplete and rejected by replay.
- Added `MiniMax H3 RefDelta Comparison Replay` with a safe `comparison_label`. `fl2va` stores the base-model x0 pass; `ref2va` requires that completed pass and computes disk-backed cross-model delta decomposition.
- Comparison passes are schema-versioned and capture-fingerprinted. Separate labels never overwrite each other, completed labels are immutable, and interrupted labels can be replaced.
- Replay evaluates only captured actual steps on exact fused states/sigmas and returns the captured fused final pre-inverse-scaling tensor. Continuum therefore builds every later chunk from the original fused trajectory.
- Added separate video/audio metrics for fused-vs-FL2VA, fused-vs-Ref2VA, Ref2VA-from-FL2VA versus fused-from-FL2VA direction, magnitude, projection, residual, and orthogonal components. Undefined zero-delta normalizations emit null with an explicit defined flag.
- Ref2VA is documented and encoded as a comparison model for the imported reference delta, not as a quality oracle. Legacy Reference Replay/Guider nodes remain registered for saved workflows.
- Calibration capture keeps Spectrum actual/forecast metadata so replay never evaluates a genuine reference model against a forecast result. Spectrum-off capture remains the recommended first calibration dataset.

## Scheduler profile tooling

- Removed Ref2VA-relative “model error” and error-slope inputs from scheduler density construction.
- `tools/build_profile.py` now emits a neutral provisional profile by default while aggregating production stability and labeled comparison diagnostics into metadata.
- Replayed copies of captured production telemetry are deduplicated before stability binning, preventing capture + FL2VA + Ref2VA inputs from triple-counting one fused trajectory; metadata reports both raw and unique counts.
- An explicit `--experimental-stability-density` mode can use production trajectory risk, curvature, extrapolation error, optional stochastic pressure, and instability slope. Comparison metrics remain explanatory and never enter density.

## Risk and telemetry

- Split deterministic `trajectory_risk` from stochastic pressure. Adaptive ER order now responds only to trajectory evidence; stochastic adaptation uses combined evidence.
- Measure stochastic pressure from the native pre-adaptation ER-SDE increment, preventing the previous gate from reducing its own future evidence.
- Replace the hard-clamped stochastic/movement signal with the smooth bounded mapping `ratio / (1 + ratio)`.
- Treat zero/subnormal denoised movement as undefined stochastic evidence, eliminating the artificial first-step epsilon spike and frozen-stream risk floor.
- Use `torch.finfo(dtype).tiny` rather than numerical epsilon for the undefined-movement threshold so small legitimate BF16 movement remains measurable.
- Record native and actually-applied stochastic RMS/ratios separately while continuing to publish the exact post-gate increment to Spectrum.
- Preserve the last valid actual-model native stochastic/movement evidence across Spectrum forecast gaps. Forecasts neither invent new pressure evidence nor erase the previous actual measurement; the next actual evaluation replaces or clears it normally.
- Refresh the cached Spectrum forecast controller view immediately after an actual step measures its native stochastic/movement ratio. The current actual step keeps the risk/pressure that really controlled it, while the very next forecast consumes the newly available actual-only stochastic evidence instead of remaining one anchor stale.
- Allow bounded trajectory correction to be applied to a Spectrum forecast using only the most recent actual-model raw anchor and actual-only derivative evidence. Forecast values remain excluded from `evidence_history` and can never become future correction/risk anchors.
- Keep no-adaptation sampling numerically unchanged unless telemetry or calibration capture is explicitly enabled; the strict no-instrumentation baseline still delegates directly to native ComfyUI ER-SDE.

# MiniMax H3 RefDelta Solver v0.2.0

v0.2.0 adds explicit compatibility with Spectrum MiniMax H3 v0.2.20+ while preserving RefDelta's actual-anchor evidence rules and adaptive stochastic path.

## Solver

- ER-SDE-3 mathematics follows current ComfyUI and retains native MiniMax-H3 `ModelSamplingAV` semantics.
- Raw denoised anchors are differentiated using actual nonuniform ER solver coordinates.
- Bounded trajectory-risk signals smoothly gate stage 2, stage 3, and stochastic adaptation.
- Video and audio are reduced separately while remaining one packed H3 integration state.
- Optional bounded x0 trajectory correction is disabled by default.
- With all adaptations and telemetry disabled, the node delegates directly to native ComfyUI ER-SDE.

## Scheduler and diagnostics

- The scheduler weights a continuous beta(0.6, 0.6) prior with versioned rank-profile difficulty.
- The bundled rank-1024 profile is explicitly provisional and neutral pending matched same-state measurements.
- An optional dual-model guider compares fused and genuine Ref2VA x0/velocity on the same latent and conditioning.
- Scalar JSONL/CSV telemetry and offline profile/AdaLN-analysis tools are included.

## Compatibility

- Solver history and RefDelta evidence history are separate. Spectrum forecasts remain valid ER-SDE solver values but never become risk or trajectory-correction anchors.
- Spectrum forecast gaps preserve the last valid actual native stochastic/movement evidence instead of clearing it; only a later actual model evaluation may replace or clear that evidence.
- A forecast immediately after an actual step now consumes the stochastic/movement evidence measured on that actual step; cached forecast state is refreshed without rewriting the current actual step's telemetry or committing any forecast value to evidence history.
- Bounded trajectory correction may be applied to a forecast from the latest actual-only anchor and derivatives without committing the forecast to evidence history.
- Spectrum classifies actual, forecast, and offline-replay source steps through a structural bridge with no hard runtime dependency from RefDelta to Spectrum.
- RefDelta publishes the exact post-adaptation stochastic increment for Spectrum compensation and replay ownership.
- Native-equivalence mode retains direct delegation to current ComfyUI ER-SDE.
- Reference diagnostic results under Spectrum are matched to actual calls by sigma, avoiding stale sampler-ordinal failures after forecast steps.
- Invalid or mismatched bridge state fails explicitly instead of silently contaminating history or stochastic ownership.
- The native fixture matrix verifies that marked forecasts stay out of RefDelta evidence commits, preserve actual-only evidence across forecast gaps, and consume the stochastic evidence measured by the immediately preceding actual step.

The implementation is experimental until representative rank-1024 GPU calibration and held-out media validation are completed.
