# Unreleased

## Reference calibration

- Added a model-level `MiniMax H3 RefDelta Reference Diagnostic` that decorates the production MODEL instead of requiring the surrounding workflow to replace its guider.
- Added a structural model-option contract for compatible runtimes such as H3 Continuum. The integration has no hard import dependency and fails explicitly on malformed or nested contracts.
- Added a positive-only runtime guider mixin so Continuum can evaluate genuine Ref2VA with the exact per-chunk positive CONDITIONING and CFG=1 semantics used by its normal internal BasicGuider.
- Retained the previous explicit reference GUIDER as a deprecated saved-workflow compatibility node.
- Spectrum forecast calls remain reference-free; only actual fused evaluations produce matched genuine Ref2VA results.

## Risk and telemetry

- Split deterministic `trajectory_risk` from stochastic pressure. Adaptive ER order now responds only to trajectory evidence; stochastic adaptation uses combined evidence.
- Measure stochastic pressure from the native pre-adaptation ER-SDE increment, preventing the previous gate from reducing its own future evidence.
- Replace the hard-clamped stochastic/movement signal with the smooth bounded mapping `ratio / (1 + ratio)`.
- Treat zero/frozen denoised movement as undefined stochastic evidence, eliminating the artificial first-step epsilon spike and frozen-stream risk floor.
- Record native and actually-applied stochastic RMS/ratios separately while continuing to publish the exact post-gate increment to Spectrum.
- Keep no-adaptation sampling numerically unchanged: with stochastic adaptation disabled the native and applied stochastic increments are identical.

# MiniMax H3 RefDelta Solver v0.2.0

v0.2.0 adds explicit compatibility with Spectrum MiniMax H3 v0.2.18+ while preserving RefDelta's actual-anchor evidence rules and adaptive stochastic path.

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
- Spectrum classifies actual, forecast, and offline-replay source steps through a structural bridge with no hard runtime dependency from RefDelta to Spectrum.
- RefDelta publishes the exact post-adaptation stochastic increment for Spectrum compensation and replay ownership.
- Native-equivalence mode retains direct delegation to current ComfyUI ER-SDE.
- Reference diagnostic results under Spectrum are matched to actual calls by sigma, avoiding stale sampler-ordinal failures after forecast steps.
- Invalid or mismatched bridge state fails explicitly instead of silently contaminating history or stochastic ownership.
- The native fixture matrix verifies that marked forecasts stay out of RefDelta evidence commits.

The implementation is experimental until representative rank-1024 GPU calibration and held-out media validation are completed.
