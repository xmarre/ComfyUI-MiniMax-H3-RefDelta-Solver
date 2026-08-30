# MiniMax H3 RefDelta Solver v0.5.0

## Selectable SEEDS and SA-Solver backends

- Expanded the RefDelta Stability Sampler `base_sampler` selector from ER-SDE-only operation to **ER-SDE, SEEDS-2, SEEDS-3, SA-Solver PEC, and SA-Solver PECE**. ER-SDE remains the saved-workflow and production default while the additional backends complete broader real-media validation.
- Delegated SEEDS and SA numerical equations to current ComfyUI rather than maintaining forked solver implementations. Native-equivalence mode remains a direct passthrough to the selected upstream sampler.
- Added shared RefDelta trajectory correction, spatiotemporal stability control, stochastic gating, telemetry, and fail-closed validation around the new native backends without adding model evaluations.

## SA-Solver PECE endpoint ownership

- Added a distinct `sa_solver_pece` backend using native ComfyUI `sample_sa_solver(..., use_pece=True)` with the reviewed predictor/corrector defaults.
- Modelled native PECE as `P0, P1/C1, P2/C2, ...` rather than a generic fixed-stride two-stage sampler: **P0 and exact corrected C_i evaluations own persistent RefDelta evidence**, while later predicted P_i evaluations remain ephemeral current-corrector inputs.
- Prevented same-sigma predicted/corrected states from being double-counted in RefDelta trajectory history, derivative evidence, or stochastic-control state.
- Persistent PECE endpoints must be actual model evaluations; a Spectrum-forecasted persistent endpoint fails closed rather than contaminating RefDelta evidence.
- Preserved native SA predictor-noise call topology and RNG/noise geometry for both PEC and PECE.

## SEEDS stochastic geometry

- Added RefDelta SEEDS-2 and SEEDS-3 wrappers while retaining the native internal stage structure and correlated stochastic decomposition.
- RefDelta resolves one stochastic gate per outer interval and reuses it for all native noise segments in that interval, avoiding controller-state advancement on internal SEEDS stages.
- Persistent trajectory/stability evidence remains actual-only and outer-coordinate-owned.

## Spectrum interoperability and compatibility

- Added the versioned RefDelta/Spectrum backend contract for SEEDS and SA-Solver composition. Spectrum classifies completed logical H3 calls as actual/forecast while RefDelta retains ownership of its reviewed evidence topology.
- RefDelta SEEDS/SA interoperability requires the companion Spectrum multi-backend integration from Spectrum PR #91 or a release containing it; standalone RefDelta use does not depend on Spectrum.
- Calibration capture remains ER-SDE-only because its persisted diagnostic format belongs to the original ER-SDE trajectory.
- Existing saved workflows continue to default to `base_sampler = er_sde`.

## Validation

- Added native same-input/same-noise parity coverage for SEEDS-2, SEEDS-3, SA-Solver PEC, and SA-Solver PECE.
- Added explicit PECE predicted/corrected endpoint-ownership tests, persistent-endpoint forecast rejection, selector/import coverage, and native SA noise-sampler call-topology checks.
- The reviewed multi-backend implementation passed the full pinned ComfyUI/Python CI matrix; the current SEEDS/SA-capable lane completed **292 tests**.

# MiniMax H3 RefDelta Solver v0.4.0

## Experimental H3 uniform-flow scheduler

- Added the separate stable node ID `MiniMaxH3UniformFlowScheduler`, displayed as `MiniMax H3 Uniform Flow Scheduler [Experimental]`; existing sampler and legacy scheduler node IDs remain unchanged.
- Promoted the completed real-media A/B winner to the experimental node defaults: **19 steps + `phase_offset_uniform` + `phase=0.50`**. This changes only the new experimental scheduler node.
- The 19-step phase sweep found `0.60` and `0.65` prone to visible flashing, `0.55` stable but weaker in action than `0.50`, and `0.45` worse. `0.50` gave the best observed balance of action, visual stability, shot variety, and audio behavior in the tested MiniMax-H3 + ER-SDE workflow.
- Preserved exact ComfyUI `ddim_uniform` / `BasicScheduler` parity as `legacy_ddim_uniform`, including integer table stride, phase placement, extra-point behavior, and denoise tail slicing.
- Added shared-base-time linspace, phase-offset, power, uniform refinement-tail, H3-aware trailing-refined, asymmetric beta, audiovisual arc-length, continuous piecewise structure/refinement, and safe offline curvature-profile modes.
- Real-media testing also found the refinement-tail family visually strong but less stable/inventive, `av_arc_length` promising but with an artifacted shot in the tested pass, and default `asymmetric_beta(0.6, 0.6)` unstable with rapid shot repetition and doubled audio. Neutral power/piecewise controls correctly matched uniform linspace.
- Added an explicit version-2 shared-base-time density schema and neutral control profile. Experimental v2 density fails closed unless telemetry contains explicitly actual model-evaluation rows and excludes Spectrum forecast rows before binning.
- Preserved one H3 audiovisual clock by mapping shared base time through the loaded model's video sampling API and leaving audio mapping to `ModelSamplingAV`/MiniMax-H3. Runtime video/audio shifts are validated; runtime scheduling does not hardcode the usual 12/3 shifts.
- Fixed packaged `curvature_profile` loading under ComfyUI's compatibility `importlib.resources` implementation by using one-child `Traversable.joinpath()` calls; added a regression that reproduces `CompatibilityFiles.SpecPath`.
- Added deterministic schedule inspection/export tooling plus regression coverage for pinned-ComfyUI parity, reduction identities, denoise tails, short-step defaults, non-default shifts, profile provenance, Spectrum forecast exclusion, numerical boundaries, immutability, packaging, and node registration.
- The conservative production fallback remains **MiniMax H3 RefDelta Stability Sampler + ComfyUI `BasicScheduler(beta)`** while the new phase-offset preset continues broader prompt/seed validation.

# MiniMax H3 RefDelta Solver v0.3.0

## Production sampler defaults

- Added `MiniMax H3 RefDelta Stability Sampler` as the recommended production node for the rank-1024 INT8 ConvRot fused checkpoint.
- Added a dedicated built-in production default set while keeping every production-relevant control editable directly in the node; there is no production profile/calibration file to select for the sampler.
- The validated current candidate uses spatiotemporal stability control, dynamic stochastic strength `0.50`, minimum multiplier `0.50`, static-video strength `0.25`, motion band `0.15..0.60`, diffusion band `0.05..0.50`, diffusion weight `0.20`, gamma `1.00`, spatial/temporal radii `2`, EMA `0.70`, and the existing bounded trajectory-correction settings.
- Same-seed validation against the advanced node produced cell-for-cell identical CSV telemetry and byte-for-byte identical JSONL telemetry when configured with the same production values, confirming that the streamlined node does not alter sampler/controller behavior.
- Preserved the original `MiniMaxH3RefDeltaSampler` node ID and dataclass defaults for saved-workflow compatibility. It is now labeled `[Advanced/Diagnostic]` instead of silently adopting the new production defaults.
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
