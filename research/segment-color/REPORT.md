# Constant per-segment color experiment

This experiment preserves every original frame, pose, pixel location and cadence. It applies one RGB gain/bias transform for each original generated clip, for that clip's entire duration. There are no crossfades, tween frames, geometric warps or time-dependent color fades. The original `IYTYT.mp4` is unchanged. The short experimental previews are silent.

## Review files

- `361-comparison.mp4`: four seconds around frame 361, source left, color-only right.
- `722-comparison.mp4`: four seconds around frame 722, source left, color-only right.
- `361-color.mp4` and `722-color.mp4`: native 1280×720 color-only previews.
- Matching `*-source.mp4` files and native `*-cut.jpg` comparisons.

All six videos have 96 frames at 24000/1001, with explicit BT709 limited-range output. Previews cover zero-based frames 313–408 and 674–769. The original join remains at offset 48. A geometry jump or startup hold is intentionally still present: color-only processing cannot remove it.

## Measurement and model

For each of the nine confirmed cuts, match actual corresponding pixels between `(cut−1, cut)`, `(cut−2, cut+1)` and `(cut−3, cut+2)`. Forward/back flow is used only for sample correspondence, never for the output image. Reject inconsistent correspondence, borders, strong outline gradients and large initial appearance errors. Draw balanced samples from a 12×8 spatial grid so a large flat sky does not dominate the result. Fit a diagonal RGB affine map with robust Huber weights and a mild identity-gain prior. Fit on alternating spatial tiles and evaluate on the held-out tiles.

The local map is `left_rgb ≈ right_rgb * gain + bias` in decoded, nonlinear SDR RGB. This is an empirical grade match, not a physical exposure estimate. Applying a constant affine map preserves within-clip temporal color differences up to the constant channel scale, before clipping and quantization. It neither estimates nor removes intentional lighting changes inside a clip.

| Cut | Local RGB gain | Local RGB bias, byte units | Held-out mean absolute difference: original → corrected |
|---:|---|---|---:|
| 361 | 0.9123, 0.9332, 0.9757 | 8.110, 5.675, 3.096 | 4.37 → 2.49 |
| 722 | 0.8665, 0.9018, 0.8936 | 7.138, 6.571, 9.645 | 4.65 → 2.89 |
| 1083 | 0.9414, 0.9570, 0.9424 | 2.733, 1.896, 6.391 | 3.25 → 2.63 |
| 1444 | 0.8975, 0.9399, 0.9385 | 6.043, 3.524, 5.506 | 3.92 → 2.59 |
| 1805 | 0.9039, 0.9355, 0.9181 | 6.183, 4.293, 5.885 | 4.54 → 2.93 |
| 2166 | 0.9117, 0.9235, 0.9384 | 9.529, 9.750, 9.547 | 5.96 → 2.72 |
| 2527 | 0.9644, 0.9680, 0.9424 | 4.396, 3.705, 7.384 | 3.67 → 2.63 |
| 2888 | 0.9398, 0.9751, 0.9567 | 11.969, 8.083, 13.254 | 5.36 → 2.46 |
| 3240 | 0.9451, 0.9298, 0.9256 | 11.061, 11.086, 10.186 | 4.57 → 2.15 |

The three independent pair fits generally agree within 0.014 gain and 2.53 byte bias per channel. That is useful evidence of a grade reset near the cut, but not proof that the full image shares one grade. Residual redraws, changed texture and local lighting differences remain.

`estimator-controls.json` compares the exact cross-cut pair with two-frame pairs wholly inside neighboring clips, using both the regularized regression and symmetric total least squares. Around 361 and 722, within-clip gains remain approximately 0.998–1.000 while cross-cut gains differ by roughly 2–14%. A symmetric fit does not eliminate this effect. At 2166 the in-clip camera/lighting evolution is larger (roughly 1% gain over two frames), but the cut reset is still several times larger. This rules out ordinary regression attenuation as the main explanation.

## Cumulative drift: do not ship the propagated full-shot plan

Exact sequential propagation anchored to segment 0 follows:

```
G_next = G_previous * local_gain
B_next = B_previous + G_previous * local_bias
```

The previews use this rule: the first clip is unchanged, the second gets the 361 transform, and the third receives the cumulative 361+722 transform. `grades.json` records the hypothetical full-shot constant transforms; only the two short preview windows were rendered.

The result is unsuitable for a full render. By the last segment, cumulative gains are `[0.4715, 0.5740, 0.5545]` and biases `[45.45, 40.78, 53.21]`. This compresses contrast heavily and raises black outlines. Across seven sampled frames per segment, no channel clips, yet the last segment's 1st–99th percentile ranges collapse to approximately R 48–138, G 41–145 and B 53–155. Lack of clipping is therefore not a sufficient safety criterion. `dynamic-range.json` includes original/corrected percentile ranges and average color changes for every segment.

The consistent contrast increases at generation boundaries also imply an opposing longer-term evolution inside the generated clips, or differences in their scene statistics. Removing every boundary reset with exact constant transforms necessarily accumulates that evolution into the full shot. Choosing a different single reference grade merely moves the contrast loss/expansion between early and late segments; it does not remove the relative gain ratio between them.

## Recommended next color model

Use these correspondences as measurements, not an unconditional sequence of grading commands. Solve all segment grades jointly with an explicit overall contrast budget and black/white constraints. Minimize robust matched-color residuals plus a penalty for departing from each segment's original grade, and report the remaining seam residual. A bounded joint solve must trade some perfect local color matching for acceptable full-shot contrast: there is no constant-per-segment affine solution that simultaneously reproduces the unconstrained exact matches and avoids their measured cumulative drift.

If more than a partial match is required, a monotone tone curve anchored at black/white is a more appropriate next experiment than lifting cartoon ink with a free bias. It still needs cumulative-drift constraints and visual inspection. Any later model that deliberately changes the grade within a clip should be clearly distinguished from this constant-grade experiment and checked against actual lighting changes.

Reproduction from the repository root: `.venv/bin/python research/segment_color.py`, followed by `.venv/bin/python research/segment_color_diagnostics.py`. Preview paths must not already exist; the script refuses to overwrite media. These are experimental analysis scripts, not a promised seamless-repair CLI.
