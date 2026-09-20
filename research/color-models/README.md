# Global residual color models: research decision

Do not integrate the unrestricted global affine or polynomial models. They improve the colors seen at each cut, but extrapolate to omitted color families poorly. Prefer the independently audited, palette/position-local RBF candidate for a short temporal proof. That recommendation is conditional on each join receiving its own fit and review; the RBF audit below covers join 361 only.

No production file, geometry, source-frame selection, or video was changed. This study consumes the accepted encoded `output/IYTYT-source-conform-eight-joins.mp4`, including its existing protected diagonal LUTs. Native comparison stills are research only.

## Method

`global_models.py` fits residual corrections to a shared midpoint, separately on outgoing and incoming pixels. Thus neither side is arbitrarily the color reference. Flow finds correspondences only. Every proposed output pixel uses its own RGB; no output resampling, image blending, convolution, or generated image is involved.

For each of all 9 joins, use 3 cross-cut pairs (n−1,n), (n−2,n+1), (n−3,n+2), at 640×360. Bidirectional consistency, interior-gradient and RGB-residual gates remove obvious mismatches; samples are spatially balanced. Alternating tiles are held out. Fit robust regularized residual diagonal affine, full cross-channel affine, and headroom-protected cross-channel/degree 2 models. The protected form is `y_c=x_c+255*z_c*(1−z_c)*p(R,G,B)`, with a conservative sampled coefficient-scale bound. Full details and coefficients are in `results.json`.

Mean absolute RGB mismatch in held-out tiles, in 8-bit code values (lower is better):

| Join | Existing baseline | +Diagonal affine | +Cross affine | +Protected cross | +Protected quadratic |
|---:|---:|---:|---:|---:|---:|
| 361 | 2.494 | 2.422 | 2.057 | 2.122 | 1.853 |
| 722 | 2.804 | 2.785 | 2.319 | 2.391 | 2.223 |
| 1083 | 2.578 | 2.575 | 2.333 | 2.325 | 2.087 |
| 1444 | 2.538 | 2.521 | 2.199 | 2.276 | 2.117 |
| 1805 | 2.793 | 2.765 | 2.383 | 2.495 | 2.377 |
| 2166 | 3.519 | 3.006 | 2.787 | 3.315 | 3.226 |
| 2527 | 2.656 | 2.585 | 1.970 | 2.018 | 1.881 |
| 2888 | 2.593 | 2.520 | 1.852 | 1.889 | 1.856 |
| 3240 | 2.257 | 2.257 | 2.204 | 2.221 | 2.001 |

The quadratic gain is 8–29%. Native comparisons at 361,2166,2527,2888 show mild color changes with geometry and line locations identical by construction; the weak gain at 2166 does not resolve its city/sky difference. This proxy measures matched interior colors, not a perceptual guarantee of a seamless cut.

## Rejection evidence

- Unprotected cross-channel affine creates out-of-range values in0.09–2.40% of actual anchor channels; clipping would crush those values. Protected candidates had no sampled clipping and map exact black/white to themselves.
- Holding out a whole coarse color family exposes overgeneralization. Protected quadratic worsens unseen light neutrals by 42% and blue by 32% at 361; unseen blue by 26% at 2166 and23% at 3240. Every join has at least one >5% regression. Cross-channel linear models are generally worse under this test. A model that wins a spatial split can still misgrade a newly visible material later in the segment.
- At1805, independently fitting the three cross-cut pairs changes the predicted correction by up to 1.96 bytes at the 95th percentile; parallax/redrawing contaminates even interior correspondence. This supports conservative correction there.
- Within-clip controls induce95th-percentile changes of 0.05–0.42 byte for protected quadratic. They do not show a large intrinsic color drift, but are only two short controls per join.

No spatial blur is introduced by a pointwise transform; nevertheless its color derivative can change line contrast. Numerical gamut checks, held-out palettes, and native temporal review remain necessary.

## Independent RBF audit requested by root

Audited the existing saved model, without refitting: `research/local-color-probe/361-tuning-models.json`, key `local-light-prior`, 64 centers, feature scales [48,48,48,.2,.2], ridge.003. Implementation is `research/local_color_probe.py:apply_samples`; reproducible audit is `audit_rbf.py`, results `rbf-audit.json`.

The formula is `F_c=255*z_c + 4*L*z_c*(1−z_c)*tanh(raw_c/L)`, with L=18. Since `|tanh|≤1` and 4L=72<255, for every in-range input and finite coefficient set:

- `F_c≥z_c*(255−72)≥0`;
- `255−F_c≥(1−z_c)*(255−72)≥0`;
- exact black and white remain exact.

This is a range guarantee independent of the RBF fit, not merely a sampled observation. The code only computes RGB/XY features, RBF weights, channel corrections, and rounding; there is no sampling of neighboring output pixels.

Finite-difference color Jacobians were evaluated at 76,895 RGB/XY combinations per side (13³ RGB cube ×7×5 position grid), plus 57,600 sampled anchor colors per side. No sampled own-channel derivative or determinant is nonpositive. Minimum singular value is 0.814 over the broad cube and 0.857 over actual anchor colors; minimum broad determinant 0.732. This indicates no sampled color fold or local contrast inversion, but is not a proof over the continuous 5D domain. Exact black/white and range guarantees were also checked numerically.

Weakly supported grid colors (confidence<.1) receive ≤0.723 byte correction; their Jacobian minimum singular is 0.970. At actual anchors the full-pixel sampled maximum correction is 5.53 bytes, above the 4.56 bytes reported for matched interior observations; maximum 99th-percentile change 3.71 bytes. Antialiased ink and edge pixels were not in the fit but still receive the same pointwise mapping.

Temporal sensitivity: translating an unchanged anchor color by 16 native pixels horizontally and 8 vertically through this spatial field produces at most 0.403 byte change,99th-percentile≤0.126 byte. Doubling that displacement gives≤0.633 byte maximum on observed anchor colors. Quintic residual-weight fading over 168 frames contributes at most 0.062 byte/frame at these anchors. The broad artificial gamut has slightly larger extrema (0.520 byte and 0.081 byte/frame respectively). These are small enough to justify a short temporal proof, rather than prove its invisibility.

Confidence is NOT semantic protection: at least 95% of samples from baseline frames ±84 away still have confidence1, and the correction can reach 5.97 bytes there. A new object can share calibrated colors and position while requiring another correction. The RBF taper is continuous but only piecewise differentiable because of nearest-center and threshold operations; it avoids abrupt value jumps, not every derivative kink. Do not treat the taper as evidence that a long-support correction remains perceptually valid as objects move.

Recommended disposition: numerically defensible for the 361 short proof, keeping geometry and timing exactly fixed. Audit each other fitted join separately and inspect the support interval for moving skin, blue costume, sky and shadow behavior. Do not automatically promote the polynomial fallback.

## Artifacts

- `results.json`: all 9 numerical results, safety and material-family holdouts, fitted global coefficients.
- `comparison-{join}.jpg`: five 640px rows (baseline and four models).
- `native-{361,2166,2527,2888}.jpg`: native-resolution still pairs, baseline above protected quadratic.
- `observations-{join}.npz`: cached correspondence data, tile splits, controls and anchor stills.
- `rbf-audit.json`: saved-candidate color Jacobians, spatial sensitivity, confidence and range checks.

Run `.venv/bin/python research/color-models/global_models.py`, `.venv/bin/python research/color-models/native_review.py`, and `.venv/bin/python research/color-models/audit_rbf.py`. None renders or updates the full video.
