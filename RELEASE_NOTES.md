# Unreleased

## Production sampler and preset

- Added `MiniMax H3 RefDelta Stability Sampler` as the recommended production node for the rank-1024 INT8 ConvRot fused checkpoint.
- Added the named `r1024_int8_convrot_stability_v1` production preset while keeping every production-relevant control editable in the node.
- The preset uses the current real-generation candidate: spatiotemporal stability control, dynamic stochastic strength `0.60`, minimum multiplier `0.50`, static-video strength `0.25`, motion band `0.15..0.60`, diffusion band `0.05..0.50`, diffusion weight `0.25`, gamma `0.75`, spatial/temporal radii `2`, EMA `0.70`, and the existing bounded trajectory-correction settings.
- Preserved the original `MiniMaxH3RefDeltaSampler` node ID and dataclass defaults for saved-workflow compatibility. It is now labeled `[Advanced/Diagnostic]` instead of silently adopting the new production preset.
- Kept diagnostic capture controls out of the recommended production node so normal generation settings are not mixed with replay/capture state.

## Spatiotemporal stochastic stability

- Added `legacy_global`, `streamwise`, and `spatiotemporal_stability` stochastic-control modes. `legacy_global` preserves released packed-scalar behavior for compatibility and A/B baselines.
- Added independent actual-only video/audio stochastic targets and strength scales. Audio never consumes the video stability map.
- Added a channel-reduced `[B,1,T,H,W]` H3 latent controller combining normalized neighboring-frame motion with change from the preceding actual x0. It requires no pixel decode, optical flow, extra model calls, or additional NFE.
- Added configurable dynamic-to-static policy interpolation, diffusion weighting, gamma, normalization floor, replicate-safe temporal/spatial smoothing, actual-only EMA, progress ramping, and an optional scalar/per-cell gate slew limit.
- Extended packed `StreamLayout` with validated `(C,T,H,W)` reconstruction/repacking while preserving real ComfyUI `[B,1,E]` packing and BF16 dtype.
- Incompatible latent shapes reset cached stability/gate state instead of reusing stale maps.
- Spectrum forecasts may consume cached actual-only stability and risk but never update motion, diffusion, EMA, x0 anchors, or stream-risk evidence.
- Spectrum receives the exact final stochastic increment actually added to the sampler state. Native pre-gate stochasticity remains the source of future stochastic-pressure evidence, avoiding controller self-feedback.
- Added scalar telemetry for controller targets, applied-gate/restore distributions, motion/diffusion summaries, progress/source state, and slew activity. Optional actual-only compressed NPZ debug maps expose temporal motion, diffusion change, restore confidence, and final video gate.

## Diagnostic same-state comparison replay

- Generalized disk-backed replay into labeled FL2VA and Ref2VA comparison passes while loading only one full H3 model per execution.
- Historical `calibration_capture` / `calibration_id` names and on-disk paths remain for saved-workflow compatibility, but the data is explicitly treated as **diagnostic comparison data**, not production scheduler calibration.
- Fused capture persists exact packed states, fused x0 values, sigmas, actual/forecast classification, telemetry, and the final pre-inverse-scaling sampler tensor.
- Added `MiniMax H3 RefDelta Comparison Replay`; `fl2va` stores the base-model same-state pass and `ref2va` combines it with genuine Ref2VA to decompose the imported reference delta.
- Comparison replay evaluates only captured actual steps and returns the captured fused final state, preserving the original Continuum trajectory across diagnostic executions.
- Added separate video/audio direction, magnitude, projection, residual, orthogonal-residual, x0, and velocity metrics. Zero/subnormal true deltas use explicit undefined/null normalized fields instead of NaN/Inf.
- Ref2VA is treated as a comparison model for the imported parameter delta, never as a quality oracle.
- Legacy Reference Replay and simultaneous Reference Guider nodes remain registered for saved workflows.

## Scheduler cleanup

- Production scheduling is now explicitly **ComfyUI `BasicScheduler` with `beta`**. There is no production scheduler calibration file derived from FL2VA/Ref2VA replay.
- Marked `MiniMax H3 RefDelta Scheduler` as `[Legacy/Research]` / deprecated for ordinary production workflows while retaining its node ID for saved-workflow compatibility.
- Reclassified the bundled `r1024_provisional` profile as a neutral compatibility profile; its metadata explicitly points production users to stock beta.
- `tools/build_profile.py` is research-only. It strips `comparison_*` and `ref_delta_*` fields before binning, deduplicates replay copies, emits neutral density by default, and requires `--experimental-stability-density` for any non-neutral research schedule.
- Generated research profiles explicitly record `production_use = false`; FL2VA/Ref2VA comparison diagnostics are neither embedded in scheduler profiles nor used as scheduler evidence.

## Risk, Spectrum, and regression hardening

- Split deterministic `trajectory_risk` from stochastic pressure. Adaptive ER order responds only to trajectory evidence; stochastic adaptation uses combined trajectory/stochastic evidence.
- Measure stochastic pressure from the native pre-adaptation ER-SDE increment and map stochastic/movement ratio smoothly as `ratio / (1 + ratio)`.
- Treat zero/subnormal denoised movement as undefined stochastic evidence using `torch.finfo(dtype).tiny`, preserving small legitimate BF16 movement.
- Preserve the last valid actual-model native stochastic/movement evidence across Spectrum forecast gaps; forecasts neither invent nor erase it.
- Refresh the cached forecast controller view immediately after an actual step measures new stochastic evidence, so the very next forecast consumes the freshest actual-only values.
- Bounded trajectory correction may target a Spectrum forecast but its raw anchor and derivatives remain actual-only; forecasts never enter `evidence_history`.
- Strengthened native-fixture regression tests to assert exact tensor identity of last-actual stochastic evidence after a forecast and exact last-actual correction anchors before and after the forecast gap.
- Preserve direct delegation to native ComfyUI ER-SDE in strict no-instrumentation/no-adaptation mode.

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
