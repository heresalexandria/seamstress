# Reproduce the experimental color-refined candidate

This is a separate candidate from the [eight-join baseline](REPRODUCE-IYTYT.md). It adds pointwise residual color corrections after the baseline's existing grading. It keeps the same geometry, viewing crop, protected RGB LUTs, original source-frame mapping and full-render audio treatment. It does not add spatial warping, generated frames, adjacent-frame blending or retiming.

The candidate still requires playback review. Numerical agreement and passing tests do not certify invisible joins. **Geometry at frame 2888 remains unresolved.** The baseline recipe and movie remain available separately.

The candidate artifact names are:

- Saved plan: [plans/IYTYT-color-refined.json](../plans/IYTYT-color-refined.json).
- Calibration: [plans/IYTYT-color-refined-calibration.json](../plans/IYTYT-color-refined-calibration.json).
- Candidate movie: `output/IYTYT-source-conform-color-refined.mp4`.

[IYTYT-color-refined.json](IYTYT-color-refined.json) records the exact source, baseline, candidate, plan and calibration fingerprints. Both recipes are included in this repository as separate reproducible checkpoints.

Published recipe paths are relative to the repository root. Removing the original
machine-specific path prefixes changes the plan and calibration file hashes,
which the manifest now records alongside their historical
`before_path_normalization_sha256` values. The source and rendered-video hashes,
geometry, color models, timing, and audio remain unchanged. Historical research
reports can still refer to the earlier recipe hashes. This path normalization
does not change rendered pixels.

## Environment and source

Run commands from the repository root, using the base environment described in [the baseline installation instructions](REPRODUCE-IYTYT.md#1-set-up-the-project): Python 3.13.15, FFmpeg/ffprobe 7.0.2 and `requirements.lock` were used during development. Reuse the working `.venv` if it is already installed. No API key, neural weights or PyTorch is needed.

The source is the original `IYTYT.mp4`, with SHA-256:

```text
c64928f7c72aee57d4a536d1af711d354a0358109510b175407d011621e855ca
```

The CLI verifies this fingerprint. A fresh checkout needs a separate copy of the exact source movie, since media files are excluded from Git. Keep dependency and FFmpeg versions consistent when comparing results; identical encoded file bytes across machines are not guaranteed.

## Render from the saved plan

```sh
.venv/bin/seamstress conform IYTYT.mp4 \
  --plan plans/IYTYT-color-refined.json \
  --output output/IYTYT-source-conform-color-refined-reproduced.mp4 \
  --crf 14
```

This uses the candidate's CRF 14 quality setting with a new output name so the existing candidate can remain beside it. The renderer refuses existing output videos and matching `.repair.json` sidecars; choose another filename for repeated runs.

A full render retains all 3,347 source frames at exactly 24000/1001 fps, 1280×720, with approximately 139.598 seconds of video. It copies the original audio bitstream. The `.repair.json` sidecar records source and plan hashes, renderer/FFmpeg versions, source-frame mapping, geometry status and local-color model provenance.

## Verify and view the complete result

```sh
.venv/bin/seamstress verify IYTYT.mp4 \
  output/IYTYT-source-conform-color-refined-reproduced.mp4 \
  --plan plans/IYTYT-color-refined.json \
  --work-dir output/color-refined-reproduced-review
```

Use a new verification directory for another run. Open its `review.html` to inspect original and candidate playback, join loops and individual frames. `comparison.mp4`, `before-after.jpg` and `report.json` provide additional inspection material. Watch the complete color evolution at normal speed, including before and after each join; a smaller measured color difference alone does not establish a better-looking transition.

## Preview the first join

```sh
.venv/bin/seamstress conform IYTYT.mp4 \
  --plan plans/IYTYT-color-refined.json \
  --start-frame 289 --end-frame 433 \
  --output output/first-join-color-refined-reproduced.mp4 \
  --crf 14
```

This renders 144 frames, lasting 6.006 seconds, with the first join 3.003 seconds in. Original frame indices are zero based: the start is inclusive and the end is exclusive. Matrices and color models retain their original timeline indices. Preview audio is trimmed and re-encoded as AAC; the unchanged compressed audio is used only for full renders. `verify` expects a complete-timeline movie and rejects offset previews.

## Rebuild the plan from calibration

The saved calibration contains the baseline camera/grade measurements and the selected residual color models. It is sufficient to reconstruct the plan without refitting those models:

```sh
.venv/bin/seamstress design-conform IYTYT.mp4 \
  --calibration plans/IYTYT-color-refined-calibration.json \
  --output output/color-refined-rebuilt-plan.json

.venv/bin/seamstress conform IYTYT.mp4 \
  --plan output/color-refined-rebuilt-plan.json \
  --output output/IYTYT-source-conform-color-refined-rebuilt.mp4 \
  --crf 14
```

A real CLI rebuild was checked against the saved color-refined plan across all 3,347 frame matrices. The maximum matrix-element difference was `1.5543122344752192e-15`, below the `1e-12` comparison tolerance; the viewing matrix matched exactly. All nine protected LUT curves and all nine pairs of local-color models matched exactly. Both plans retained the baseline source metadata/hash, source-frame segment mapping, protected grading, crop and explicit geometry exclusion at frame 2888. This validates recipe reconstruction; playback review of an encoded result is still required.

Use the rebuilt plan and rebuilt movie together when verifying this route. `design-conform` validates the source fingerprint and reconstructs the calibrated geometry while preserving the stored color models and their supports. It does not estimate or approve treatments for a new video. The research model-assembly entrypoint is [research/assemble_color_refinement.py](../research/assemble_color_refinement.py); rendering and calibration reconstruction do not require rerunning model selection or fitting.

## Color treatment and limits

The residual models evaluate the current frame's RGB values and native output XY coordinates **after** the existing protected grade. The recipe mixes the original `headroom-v1` response with explicit `directional-gamut-v2` models. An omitted response means v1. V1 preserves each channel's 0/255 endpoints; v2 preserves true RGB black and white while allowing a saturated material's zero or 255 channel to move inward. Model response identifiers and hashes are recorded per side of each join in the render sidecar. See [the optional model schema](EXPERIMENTS.md#experimental-local-color-schema).

Corrections reach full strength at the two source frames adjacent to each join and ease to zero over their stored supports. Neither response resamples image geometry or mixes neighboring frames. Both bound RGB values; that bound does not by itself ensure monotone color response, faithful material colors or imperceptible temporal changes.

The baseline's constant crop remains 5.924% of total width and height, approximately 2.962% per side. Its eight supported geometric corrections remain the same, with frame 2888 excluded from geometry repair. Contradictory drawings, startup holds and any remaining visible color changes still require review; this candidate is not a completed seamless-repair claim.

The delivered encoded movie passed frame-count, frame-rate, dimensions, duration and copied-audio checks. All 96 engineering tests passed when this checkpoint was reviewed. In 59 measured material regions with at least 30 accepted pixels each, equal-region mean absolute signed RGB bias fell from 3.337 to 0.752 levels (77.5%); 54 of 59 regions improved. A stricter shared encoded-frame gate gives a similar 76.9% reduction across 54 regions. These are regional color measurements, not a percentage of perceptual seamlessness. Small residuals and some local regressions remain; see [the encoded review notes](../research/local-color-diagnosis/SELECTED-CANDIDATE.md).

For the existing delivered files, [baseline-comparison.html](../output/color-refined-review/baseline-comparison.html) loops the baseline and color-refined movie together. The standard `verify` command instead compares the original source with the candidate. Full measurement and media results are linked from the fingerprint manifest above.
