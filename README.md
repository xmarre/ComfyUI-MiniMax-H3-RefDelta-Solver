# ComfyUI MiniMax-H3 RefDelta Solver

A dedicated ER-SDE-derived sampler and beta-prior scheduler for the
[MiniMax-H3 Pruned Ref-Delta Fused rank-1024 checkpoint](https://huggingface.co/xmarre/MiniMax-H3-Pruned-Ref-Delta-Fused-r1024-ComfyUI).

This project targets a specific numerical problem: the fused checkpoint can make useful final predictions while its intermediate denoised trajectory differs from ordinary H3. The sampler measures that trajectory in ER-SDE's real solver coordinate and smoothly reduces history-dependent corrections when the local anchors become unreliable. It preserves ComfyUI's `ModelSamplingAV`, packed audio/video latent, model-output conversion, and H3 conditioning path.

> **Calibration status:** the sampler controls and bundled `r1024_provisional` scheduler profile are experimental. The profile deliberately has neutral error weights until matched same-state fused/Ref2VA telemetry is collected. Neutral means the scheduler remains a continuous-table counterpart of ComfyUI's beta(0.6, 0.6) prior. No unmeasured AdaLN heuristic is presented as a calibrated quality improvement.

## Install

Clone this repository into `ComfyUI/custom_nodes` and restart ComfyUI:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/xmarre/ComfyUI-MiniMax-H3-RefDelta-Solver.git
```

The custom node has no dependencies beyond current ComfyUI. It is designed for MiniMax-H3 packed audio/video sampling and fails explicitly when `ModelSamplingAV` is not present.

## Nodes

### MiniMax H3 RefDelta Sampler

Connect its `SAMPLER` output to `SamplerCustomAdvanced` (or another node accepting ComfyUI custom samplers).

The default path uses every real model evaluation and keeps all history local to one sampler invocation:

- nonuniform first and second divided differences in current ComfyUI's ER coordinate;
- dimensionless risk from curvature, derivative direction/magnitude change, realized extrapolation error, ER stage ratios, and stochastic displacement;
- smooth stage-2/stage-3 gates instead of discontinuous order switching;
- bounded stochastic adaptation that returns to native ER-SDE behavior near the endpoint;
- separate video/audio risk reductions;
- optional Taylor trajectory correction, **off by default**, bounded by recent raw x0 movement and never written back into history.

The advanced controls are intentionally bounded. Start with the defaults. `trajectory_correction` should remain off until diagnostic runs support it.

For a strict stock baseline, set:

```text
adaptive_order = false
stochastic_adaptation_strength = 0
trajectory_correction = false
debug_telemetry = false
```

That configuration delegates directly to ComfyUI's native `sample_er_sde`; it does not maintain a second approximation of the baseline.

### MiniMax H3 RefDelta Scheduler

Connect its `SIGMAS` output to the custom sampler path. The scheduler starts from a continuous form of ComfyUI's beta(0.6, 0.6) prior, then redistributes beta-step fractions according to a versioned rank profile. Production sampling only needs the profile JSON; it never needs the genuine reference model.

The included profile is `r1024_provisional`. It preserves the beta prior because its difficulty density is neutral. Replace it with a profile created from representative same-state diagnostic runs before calling the schedule rank-1024-calibrated.

### MiniMax H3 RefDelta Reference Diagnostic

This development guider evaluates the fused and genuine Ref2VA models sequentially on the exact same packed latent, timestep, CFG, prompt conditioning, and reference conditioning. Enable `debug_telemetry` on the sampler to record:

- video/audio x0 cosine and relative error;
- video/audio carried-solver velocity cosine and relative error.

The carried audio latent uses the video solver coordinate by ComfyUI design. Multiplying both compared audio velocities by the same physical audio-sigma conversion does not change these cosine or relative-error metrics.

This mode roughly doubles transformer evaluations, requires both models on the same load device, and is not part of production inference.

## Telemetry

When enabled, scalar-only `.jsonl` and `.csv` files are written under:

```text
ComfyUI/output/refdelta_telemetry/
```

Each actual model evaluation records the requested solver, sigma, effective audio sigma, latent/x0 movement, derivative/curvature, direction cosine, stage contribution, stochastic, adaptive-order, correction, and reference fields. Video and audio reductions are separate. Only raw x0 anchors and the derivative tensors required by the solver persist between steps; telemetry retains scalars, not latent snapshots.

Build a profile from multiple representative diagnostic files:

```bash
python tools/build_profile.py \
  ComfyUI/output/refdelta_telemetry/*.csv \
  --output comfyui_refdelta_solver/profiles/r1024_calibrated.json \
  --id r1024_calibrated
```

The builder bins normalized video-sigma progress and uses robust per-bin reference error, its local slope, and observed trajectory curvature. Inspect the result and validate it on held-out prompts/seeds before adding it as a default. A trajectory-only profile requires the explicit `--allow-no-reference` flag and remains experimental.

Useful analysis tools:

```bash
python tools/compare_runs.py run-a.csv run-b.csv
python tools/analyze_adaln_curve.py model.safetensors telemetry.csv
```

`analyze_adaln_curve.py` only reports correlation; AdaLN curvature is not used by the sampler or scheduler without measured evidence.

## What the static checkpoint analysis establishes

The published rank-1024 sidecar shows a mixed approximation, not a uniform rank limit:

- 267 vector/bias patches are exact;
- the 51 AdaLN projection matrices are effectively exact because their input width is only eight;
- 264 matrix adapters are rank-limited;
- overall weighted retained matrix-delta energy is about 99.43%, but that aggregate is dominated by exact/small tensors;
- among actually compressed matrices, median retained energy is about 46.11% and mean retained energy about 49.62%; the principal block groups are approximately 41.50% (`fc1`), 45.81% (`fc2`), 46.98% (`qkv`), and 59.65% (`out_proj`).

This supports investigating a mismatched local vector field. It does **not** establish when the mismatch occurs, whether AdaLN curvature predicts it, or that a particular correction improves output. Those require same-state runs.

## Required calibration run set

For a defensible production profile, collect at least:

1. several prompt/reference cases spanning motion, texture, speech/music, and quiet audio;
2. multiple seeds per case at the intended step count and CFG;
3. fused-only baseline telemetry plus same-state genuine Ref2VA telemetry;
4. held-out A/B runs for the resulting scheduler and sampler settings;
5. interruption/restart and changed latent-shape checks on the real GPU workflow.

Judge the final media as well as the telemetry. Intermediate sharpness alone is not an objective.

## Spectrum compatibility

RefDelta Solver v0.2.0 is compatible with
[ComfyUI-Spectrum-MiniMax-H3 v0.2.18+](https://github.com/xmarre/ComfyUI-Spectrum-MiniMax-H3).
The integration uses a fail-closed API-v1 contract:

- Spectrum labels each sampler result as an actual evaluation or forecast. RefDelta keeps every result in ER-SDE's solver history, but only actual model outputs enter its risk and trajectory-correction history.
- RefDelta publishes the exact stochastic tensor after its risk and endpoint gates. Spectrum owns that final tensor for skipped-state compensation and seeded offline replay, so noise is neither estimated from the native formula nor applied twice.
- Native-equivalence mode continues to delegate to ComfyUI's reviewed `sample_er_sde` and uses Spectrum's native ER-SDE tracking path.
- A missing, stale, or mismatched bridge fails explicitly; Spectrum rejects unreviewed RefDelta versions before forecasting.

The same-state reference diagnostic remains valid: forecast steps produce no reference result, while actual reference results are matched by sigma rather than sampler-loop ordinal.

## Development

```bash
python -m pip install pytest ruff
python -m pytest -q
python -m ruff check .
python -m compileall -q comfyui_refdelta_solver tests tools
```

CI also checks the instrumented no-adaptation path against current native ComfyUI ER-SDE fixtures and builds an installable wheel. Successful tests on `main` feed the release workflow; version changes can publish to the Comfy Registry when `REGISTRY_ACCESS_TOKEN` is configured.

## License

GPL-3.0-or-later. The ER-SDE implementation is derived from GPL-licensed ComfyUI sampling code; see [LICENSE](LICENSE).
