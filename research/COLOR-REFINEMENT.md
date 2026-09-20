# Color-only refinement of the committed conform baseline

Baseline: commit `f4ea566`, `plans/IYTYT-eight-joins.json`, and `output/IYTYT-source-conform-eight-joins.mp4`. The baseline reproduction instructions remain in `docs/REPRODUCE-IYTYT.md`.

The new candidate is rendered from the **original source**, with precisely the baseline geometry, crop, global grade, frame ordering, cadence and audio handling. It adds smooth, pointwise RGB/position corrections. Measurement flow estimates corresponding colors; no flow field, neighboring image, generated pose or crossfade enters the output renderer.

## Fitting and model selection

1. Decode the source and apply the fixed baseline framing and protected grading.
2. At each cut, estimate forward/backward color correspondences at 640×360. Exclude strong gradients, inconsistent motion, borders and large unmatched differences.
3. Fit only alternating spatial tiles of the adjacent frame pair. Balance coarse palette bins so small colored objects contribute alongside large backgrounds.
4. Fit separate outgoing/incoming color maps toward shared color targets. Validate on held-out spatial tiles and two additional, unused neighboring frame pairs.
5. Inspect independently selected native 1280×720 material interiors, plus color derivatives, gamut and nearby-frame behavior. Whole-frame averages alone concealed significant hair, clothing and shadow errors.
6. Render original frames at native resolution using a bounded color response, with a seven-second quintic ease on each side of the cut.

The final selection uses `directional-gamut-v2` at frames 361, 722, 1083, 1444, 1805, 2166 and 2888; a stronger `headroom-v1` model at 2527; and a finer directional model at 3240.

Directional models let a zero channel of a saturated material move inward, while preserving true RGB black and white. This removes the unnecessary constraint that forced some dark channels toward zero. The alternative stronger headroom fits at 1444 and 2166 were rejected because of color folding or excessive contrast compression.

At 2527, the outgoing table is genuinely black. Its incoming counterpart needs to become darker rather than both sides meeting at an impossible lifted-black midpoint. The stronger headroom fit is retained there after checking the actual material, valid-gamut derivatives and nearby frames. Its shadow compression is deliberate and specific to this reviewed join.

At 3240, one broad model neighborhood incorrectly combined the jacket and trousers despite different color resets. The selected model has 96 centers, RGB bandwidth 36 and normalized XY bandwidth 0.08, versus 64 / 48 / 0.2 elsewhere.

The choices at 1083, 1805 and 2888 are close. Directional models have slightly lower equal-region average bias, but individual regions can worsen modestly: the 1805 sky red residual is +1.66 levels versus +0.77 with the conservative model. These are explicit tradeoffs, not evidence of exact color identity.

## Reproduce model fitting

The saved plan/calibration is sufficient to reproduce rendering with the CLI; see `docs/REPRODUCE-COLOR-REFINEMENT.md`. Refitting is optional research work and uses the fixed source-specific baseline:

```sh
OPENBLAS_NUM_THREADS=2 .venv/bin/python research/directional_local_color.py \
  361 722 1083 1444 1805 2166 2888

OPENBLAS_NUM_THREADS=2 .venv/bin/python research/strong_local_color.py 2527

OPENBLAS_NUM_THREADS=2 .venv/bin/python research/directional_local_color.py 3240 \
  --centers 96 --color-scale 36 --position-scale 0.08 \
  --output-dir research/local-color-directional-fine

.venv/bin/python research/assemble_color_refinement.py \
  --plan-output output/refitted-color-plan.json \
  --calibration-output output/refitted-color-calibration.json
```

Fitting scripts replace their research model/report JSONs; use a separate checkout or `--output-dir` for directional variants when preserving an experiment. The assembly command refuses existing outputs and verifies the committed baseline plan hash. It records each selected model file/hash. OpenCV k-means uses fixed seeds, but exact refitted coefficients may depend on numerical libraries/platform. The saved calibration is the authoritative replay path.

These scripts measure **this source and this baseline**. They are not a claim of automatic, reviewed repair for arbitrary new movies.

## Evidence and remaining limits

- `local-color-diagnosis/RESPONSE-COMPARISON.md`: independent native material measurements and explicit regressions.
- `local-color-diagnosis/directional-fine-response/native-material-evaluation.json`: final jacket/trousers and full-outfit checks.
- `color-models/`: derivative, gamut and temporal sensitivity audits.
- `plans/IYTYT-color-refined.json`: final rendering recipe, including every selected model.
- `output/IYTYT-source-conform-color-refined.repair.json`: actual renderer, source/plan/model provenance and media treatment after rendering.

Central jacket red bias at 3240 falls from −18.49 to −1.53 RGB levels. The whole cyan outfit still averages −2.84 red, so the central ROI is not proof that every sleeve is matched. Other small residuals remain. These statistics describe pre-encode corresponding material interiors and are not perceptual scores.

The actual encoded result must also be checked, because RGB/YUV conversion and H.264 compression alter colors slightly. No numerical audit proves that all joins are invisible. The baseline's unresolved geometry at frame 2888 and original drawing/texture changes remain outside this color-only pass.
