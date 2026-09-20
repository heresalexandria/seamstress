# Residual color diagnosis on the locked source-conform baseline

The visible residual is primarily **color/material dependent**, with a smaller spatial component. A single exposure, per-channel affine adjustment, or broad illumination field is insufficient. Neighboring pink, purple, teal and pale paint shift in different directions, while brown hair and fur repeatedly become warmer. The preferred candidate is a smooth color-conditioned correction with a weak, regularized position dependency, measured from correspondence-consistent flat interiors. It changes RGB values only; it must not transport pixels or alter source drawings.

All input stills reproduce baseline commit `f4ea566`, using `plans/IYTYT-eight-joins.json` and its protected global LUTs. They precede video encoding. No geometry, production code, or full movie was changed by this investigation. Join 2888 still has its known unresolved geometry reset; only flat, correspondence-consistent regions are useful there.

## Matched material evidence

The numbers below are mean incoming-minus-outgoing RGB differences, in 8-bit units. These are signed regional biases, not unaligned pixel differences. Other surfaces in the same image can have the opposite bias, so a near-zero whole-frame average conceals a visible flash.

| First incoming frame | Measured local residual after baseline grading |
| --- | --- |
| 361 | Hair `(+7.4,+1.7,−8.7)`; upper fur `(+8.3,+5.0,−2.5)`; teal collar `(−4.1,−4.0,+2.5)`; pale hood `(−2.0,−0.2,−0.9)` |
| 722 | Pale car `(−9.5,−6.8,−11.0)`; adjacent pink `(+0.2,+7.0,−1.1)`; purple `(+4.0,+8.2,+1.5)`; teal `(−0.3,−3.4,−1.5)` |
| 1083 | Blue tracksuit `(−8.4,−3.0,+0.8)`; pale car `(−7.7,−4.0,−5.1)`; pink car `(+0.6,+7.2,−0.8)`; hair `(+9.6,+1.8,+0.6)` |
| 1444 | Pink bumper `(+4.1,+11.7,+0.9)`; pale trunk `(−7.3,−2.6,−3.6)`; blue tracksuit `(−4.9,−4.1,−3.9)`; gray truck near zero |
| 1805 | Green foliage loses about 6–7 green levels; blue sky changes little. Pale trunk `(−6.6,−3.2,−7.5)` |
| 2166 | Left/center/right sky lose about 6/7/9 red levels. Broad darkening is present, with different blue response in river and sky. Detailed rooftops and clouds also redraw, so not every difference is grade |
| 2527 | Cyan tracksuit `(−15.1,−3.9,+3.9)`; cream wall `(−1.1,−0.1,+0.4)`; couch `(−3.4,+0.3,−1.6)`; black table underside lifts `(+9.0,+6.0,+7.8)` |
| 2888 | Brown fur `(+6.4,+3.5,−2.2)` while blue sky `(−3.1,−0.7,−1.4)`; detailed buildings excluded because geometry also changes |
| 3240 | Blue tracksuit about `(−17.6,−9.9,−4.7)` in a small accepted patch; brown hair `(+5.1,+3.5,+0.1)`; stone and cream ceiling remain within about one level |

`visual-other-joins.json` contains native rectangle coordinates, counts, signed means/medians and residual variation for 50 regions. Regions with fewer than 30 accepted samples are explicitly marked as corroborating only. The 3240 tracksuit rectangle has 26 accepted samples; the independently clustered cyan family agrees in direction. The empty door-panel region is retained as an explicit measurement failure and is not evidence. `visual-361.json` additionally records 10 manually checked native flat rectangles and source hashes.

## First-join candidate validation

The ten manually chosen 361 rectangles were reevaluated using native samples, optical correspondence only for measurement, and a four-pixel contour exclusion. The candidate was fitted elsewhere; this script fits no model. Signed biases after `local-light-prior`, compared with baseline:

| Material | Baseline RGB bias | Candidate RGB bias |
| --- | --- | --- |
| Woman hair | `(+7.41,+1.73,−8.72)` | `(+0.28,−0.47,−1.41)` |
| Woman cheek | `(+1.95,+3.30,+1.89)` | `(−0.01,−0.33,+0.90)` |
| Sloth light cheek | `(+5.39,+2.32,−3.47)` | `(+0.93,+0.13,−1.56)` |
| Teal collar | `(−4.07,−4.01,+2.48)` | `(−1.66,−0.74,+0.57)` |
| Upper fur | `(+8.33,+5.04,−2.49)` | `(+1.75,+0.93,−1.11)` |
| Lower fur | `(+4.29,+4.03,−2.45)` | `(−2.33,+0.04,−0.71)` |
| Pale hood | `(−2.03,−0.17,−0.91)` | `(−0.57,−0.09,+0.14)` |

Mean absolute signed bias, averaging the ten regions equally, falls from **3.09 to 0.80 RGB levels**. The earlier color-only model gave 1.45 and earlier hybrid 1.30. This directly supports the intended visible improvement, beyond bulk pixel-error reduction. It is not proof of invisible transitions in motion.

The remaining weaknesses are the lower fur's red overcorrection (56 accepted native pixels) and a small sky-blue sign reversal (`+0.45` to `−1.41`). A 96-center `local-fine` trial improves the ten-region mean only from 0.802 to 0.760; it **worsens lower-fur red** from −2.33 to −2.56. It improves sloth cheek/collar while slightly worsening hair and windshield. The more regularized 96-center trial scores 0.829. There is no strong native material evidence to replace the 64-center default with either 96-center model.

## Actual all-nine fitted models: native material validation

`all-models-native-material-evaluation.json` evaluates the actual `research/local-color-fit/<cut>-model.json` files on all 60 named rectangles at native sampling density. Models and hashes are recorded; no refitting occurs. The principal corrections work, but several salient residuals remain:

| Material | Baseline mean RGB bias | Fitted-model mean RGB bias | Accepted native pixels |
| --- | --- | --- | --- |
| 3240 blue jacket | `(−18.49,−9.92,−4.54)` | `(−13.32,−1.92,−0.73)` | 194 |
| 2527 cyan outfit | `(−14.99,−4.00,+3.75)` | `(−1.81,+0.41,+0.10)` | 855 |
| 2527 black table underside | `(+9.00,+6.00,+7.78)` | `(+6.91,+4.70,+5.94)` | 3456 |
| 722 hair | `(+4.58,+2.17,−10.53)` | `(−0.43,−0.29,−5.85)` | 725 |
| 1444 hair | `(+3.94,+0.28,−10.78)` | `(−0.06,−1.13,−7.68)` | 839 |
| 2166 right sky | `(−8.89,−2.83,−2.93)` | `(−6.45,−0.43,+0.03)` | 10825 |

These should be addressed or explicitly retained as limitations before claiming color seams are removed. Small whole-region regressions occur in 3240 steps (+0.55 mean absolute signed bias, only 49 accepted pixels), 3240 ceiling (+0.36,817 pixels), 2888 water (+0.28,110 pixels), and 1444 truck (+0.14,5117 pixels). The remaining regressions are minor compared with the corrected flashes, but bulk error alone would hide them.

Several stubborn residuals are **mathematically imposed by the current amplitude bound**, not only sample balance. Per channel, with `h(c)=4(c/255)(1−c/255)`, the maximum available gap closure is `18*(h(left)+h(right))`. On the original matched flat samples, even infinitely strong coefficients cannot reduce the mean absolute residual below approximately:

- 3240 blue jacket: red 7.44 levels.
- 2527 black table: RGB `(6.54,4.35,5.52)` levels.
- 722 hair: blue 5.38 levels.
- 1444 hair: blue 4.81 levels.
- 2166 right sky: red 5.93 levels.

The actual fitted residual can be larger because of regularization and shared basis support. Exact preservation of zero and 255 does not inherently require such a low correction bound near those values. Raising sample weights alone cannot defeat this mathematical restriction. This diagnosis does not prescribe lifting true blacks or clipping any color; a revised bounded response would require its own native/material verification.

## Spatial dependence and limits

Different material deltas alone do not establish that position is necessary: a nonlinear color lookup can explain them. Two small flat fur patches at 361 contain the same exact outgoing RGB but different incoming medians; matched, coarsely color-binned samples also exhibit spatially varying deltas at all joins. These observations are suggestive, but texture, quantization and small correspondence errors can confound them. The stronger practical evidence is the separately held-out improvement of the smooth hybrid model, together with the named-region validation above. Treat position as a weak corrective prior, not as a freely varying illumination mask.

Near-zero and saturated colors need particular scrutiny. The black table underside at 2527 changes from true black to dark gray. Exact preservation of zero prevents lifting the black side to a symmetric midpoint; the nonzero side can still be darkened, but a symmetric protected model may retain residual bias. At 2166 many dark red/green values also approach zero. Do not trade protected contours or clipping for lower global error.

Flow is used only to compare corresponding colors. Fine window patterns, reflection streaks, contours, occlusions and the 2888 parallax mismatch are not color evidence and cannot be repaired by a color model. No generated or interpolated pixels appear in this investigation.

## Reproduction and files

1. `prepare_pairs.py` reconstructs the locked baseline native adjacent stills from source caches and the baseline plan; `baseline.json` records its identity.
2. `measure.py` measures all nine pairs at 640×360 correspondence resolution, then excludes any match within five native pixels of a per-channel edge. It rejects forward/backward inconsistency, large RGB differences and the image boundary. `all-joins-measurements.json` aggregates the per-join results; raw accepted samples are in each `matched-flat-samples.npz`.
3. `summarize_regions.py` measures visually selected native material rectangles and writes `visual-other-joins.json` plus native `material-regions-annotated.png` stills.
4. `evaluate_all_models.py` evaluates the actual fitted all-nine models on native matched material interiors and writes `all-models-native-material-evaluation.json`.
5. `evaluate_361_regions.py` evaluates the candidate model files recorded by SHA256 in `361/material-model-evaluation.json`. `361/material-*.png` shows original native crop pixels and the pointwise color-only result, enlarged fourfold without pixel blending.

Every join folder contains `left.png`, `right.png`, `baseline-adjacent.png`, `flat-color-residual-map.png`, `palette-evidence.png`, `measurements.json`, and a native annotated material image. The residual map displays accepted flat-color errors; it is diagnostic, not a proposed correction field. The all-join average absolute matched RGB residual ranges from 1.89 to 2.33 levels except 2166 at 4.38 levels; large material biases remain possible despite those small averages.
