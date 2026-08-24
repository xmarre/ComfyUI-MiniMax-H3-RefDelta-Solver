# Unreleased

## Reference calibration

- Added disk-backed two-pass same-state calibration instead of requiring fused and genuine Ref2VA MODELs in one live sampling graph.
- `MiniMax H3 RefDelta Sampler` can now capture each exact packed sampler state, fused x0, sigma schedule, actual/forecast classification, baseline telemetry, and final pre-inverse-scaling sampler tensor under an explicit `calibration_id`.
- Capture tensors are persisted step-by-step as safetensors rather than retained for the complete run in CPU/GPU memory. Interrupted captures are marked incomplete and rejected by replay.
- Added `MiniMax H3 RefDelta Reference Replay`. In a second execution of the same workflow, users switch the existing source MODEL to genuine Ref2VA and put Reference Replay in the same SAMPLER socket; no duplicate production model path is required.
- Reference Replay evaluates only captured actual steps on the exact original fused states/sigmas, merges same-state x0/velocity metrics into the baseline telemetry, and then returns the captured fused final sampler tensor. Continuum therefore builds later chunks from the original fused trajectory instead of drifting onto a newly generated Ref2VA trajectory.
- The replay path requires only one full H3 MODEL per execution. The previous simultaneous dual-model reference GUIDER remains registered as deprecated saved-workflow compatibility only.
- Calibration capture keeps Spectrum actual/forecast metadata so replay never evaluates a genuine reference model against a forecast result. Spectrum-off capture remains the recommended first calibration dataset.

## Risk and telemetry

- Split deterministic `trajectory_risk` from stochastic pressure. Adaptive ER order now responds only to trajectory evidence; stochastic adaptation uses combined evidence.
- Measure stochastic pressure from the native pre-adaptation ER-SDE increment, preventing the previous gate from reducing its own future evidence.
- Replace the hard-clamped stochastic/movement signal with the smooth bounded mapping `ratio / (1 + ratio)`.
- Treat zero/subnormal denoised movement as undefined stochastic evidence, eliminating the artificial first-step epsilon spike and frozen-stream risk floor.
- Use `torch.finfo(dtype).tiny` rather than numerical epsilon for the undefined-movement threshold so small legitimate BF16 movement remains measurable.
- Record native and actually-applied stochastic RMS/ratios separately while continuing to publish the exact post-gate increment to Spectrum.
- Keep no-adaptation sampling numerically unchanged unless telemetry or calibration capture is explicitly enabled; the strict no-instrumentation baseline still delegates directly to native ComfyUI ER-SDE.

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
