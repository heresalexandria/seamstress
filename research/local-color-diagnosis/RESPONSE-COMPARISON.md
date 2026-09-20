# Independent native material comparison of color responses

This comparison evaluates fitted research models without modifying production, the baseline plan, any geometry, or source drawings. Flow supplies measurement correspondences only. All signed RGB deltas below mean incoming minus outgoing, in 8-bit levels. Each evaluation records the exact model path and SHA256.

The stronger per-channel headroom response fixes several dark-color mismatches but is not universally acceptable. The parent audit reported compression/folds at 1444 and 2166; their stronger models remain rejected despite improved regional means. The directional response provides much better native color agreement at those joins. The 3240 outfit additionally needs a finer spatial/color basis to distinguish jacket from trousers.

## Per-cut selection from native material evidence

These are local-color preview recommendations. They do not override derivative, gamut, cross-frame or playback failures found by other audits.

| Cut | Equal-region mean absolute signed bias: v1 / strong / directional | Native recommendation |
| --- | --- | --- |
| 361 | 0.802 / 0.707 / 0.687 | Directional is a modest improvement: collar and hair agree better. Lower-fur red overcorrection around −2.3 remains. |
| 722 | 0.713 / 0.247 / 0.307 | Directional is acceptable and removes the hair-blue flash. Strong also fits these regions well, but is not required. |
| 1083 | 0.807 / 0.788 / 0.782 | Essentially tied. Retaining v1 is conservative; directional is also reasonable if its wider audit is better. Blue tracksuit retains a +3.1 blue bias in either candidate. |
| 1444 | 0.896 / 0.680 / 0.603 | Prefer directional. Hair blue improves from −7.68 to +0.08, with small changes in stable truck/paint. Reject strong based on reported derivative audit. |
| 1805 | 0.936 / 0.818 / 0.888 | Conservative preference for v1: directional improves mid-distance foliage but introduces more sky-red bias, +0.77 to +1.66. No decisive all-material improvement. |
| 2166 | 2.385 / 1.100 / 0.607 | Prefer directional. Three sky-red biases become +1.07,+0.42,−0.60 instead of −4.22,−4.88,−6.45. Reject strong based on reported derivative audit. Cloud/skyline texture redraw is still visible and is not a color error. |
| 2527 | 1.328 / 0.575 / 1.007 | Prefer strong if its derivative/gamut audit passes: it fixes the black table much better. Directional is good on the outfit but still leaves the table visibly gray. |
| 2888 | 0.705 / 0.685 / 0.691 | Essentially tied. Retain v1 conservatively: directional improves sky but slightly worsens fur and broad building gray. Geometry remains separately unresolved. |
| 3240 | 1.371 / 1.181 / 0.985 | Prefer the finer directional model, which scores 0.459 on the same six reliable original regions. Coarse variants retain a conspicuous jacket-red mismatch. Fine still has some outfit residual outside the central jacket/trouser ROIs. |

The averages give equal weight to each region with at least 30 accepted native pixels. They are not a perceptual score, and small averages can conceal a salient individual channel. Native before/after context crops were inspected; RGB corrections preserve the supplied drawing positions. No new contour displacement appears because the method never remaps pixels. This does not establish absence of temporal color instability.

## Main residuals and regressions

The 2527 black table improves from baseline RGB bias `(+9,+6,+7.78)` to v1 `(+6.91,+4.70,+5.94)`, strong `(+1.58,+0.99,+0.97)`, and directional `(+4.97,+3.08,+4.05)`. The directional model fits an arithmetic midpoint, but the outgoing table is true black and cannot be lifted under its exact-black protection. Its incoming result therefore remains near the impossible shared midpoint. A feasible-target rule for genuinely protected RGB black/white would address a different issue from allowing a zero channel of a saturated cyan material to move.

At 722, hair-blue bias changes from −10.53 baseline to −5.85 v1, +0.47 strong and +0.37 directional. At 1444 the corresponding values are −10.78,−7.68,−1.87,+0.08. These are clear native material improvements, not merely whole-frame error gains.

Directional's largest local regressions against v1 are 1805 sky (+0.475 mean absolute signed bias), 2888 fur (+0.423), and 3240 pale stone (+0.243). The underlying signed RGB values remain small: 1805 sky `(1.66,−0.42,−0.24)`, 2888 fur `(−0.83,−0.28,−1.44)`, and 3240 stone `(0.91,−0.21,−0.13)`. Those changes justify caution or retaining v1 at nearly tied cuts; they do not invalidate the clear hair/sky improvements elsewhere.

## Why 3240 needed a finer model

The exact calibration observations contain 52 samples in the diagnostic jacket rectangle, split 26 training/26 held out, with palette weight 3.14. Its baseline mean RGB is outgoing `(29.30,134.07,197.38)` and incoming `(9.45,124.50,193.59)`. The strong feasible target is `(16.66,129.29,195.49)` on both sides, but the model produces `(24.57,129.71,195.23)` and `(11.29,128.90,195.77)`.

The trousers have 136 samples, 96 training/40 held out, and a much smaller red reset: 33.30→24.90. One nearby RBF center at native `(722.6,432.3)`, RGB `(27.71,127.43,192.54)`, represents the entire cyan outfit. The next center is distant. A shared broad basis therefore compromises toward the more numerous trouser samples. The jacket is not missing from training data; it needs a distinct local correction.

The finer directional model uses 96 centers, RGB scale 36 and XY scale 0.08. Native validation:

| Region | Baseline | Coarse directional | Fine directional | Accepted pixels |
| --- | --- | --- | --- | --- |
| Jacket | `(−18.49,−9.92,−4.54)` | `(−7.01,−1.14,−0.23)` | `(−1.53,+0.44,+0.53)` | 194 |
| Trousers | `(−9.81,−7.26,−3.84)` | `(+1.42,+1.26,+0.30)` | `(−0.78,+0.58,−0.09)` | 509 |
| Full outfit cyan class | `(−14.41,−8.76,−4.09)` | `(−3.01,−0.09,+0.16)` | `(−2.84,−0.17,+0.11)` | 1000 |

Fine improves the central jacket, trousers, fur and steps. Hair, pale stone and ceiling change very little from coarse directional. The full cyan-class result shows why the central ROIs must not be treated as proof that every sleeve/other outfit patch is matched: some portions still have greater red residual. Fine is the best tested native candidate for this join, subject to the independent derivative and temporal checks.

## Files

- `strong-response/native-material-evaluation.json`: baseline/v1/strong across all nine joins; context PNGs alongside.
- `directional-response/native-material-evaluation.json`: adds the correct `directional_local_color.apply_samples` response; context PNGs alongside.
- `directional-fine-response/native-material-evaluation.json`: all original 3240 regions plus added trousers and outfit-cyan checks; `3240-characters-context.png` shows all five variants.
- `strong-response/3240-coverage.json`: exact calibration sample counts, weights, input/target/output RGB and nearby centers.
- `evaluate_strong_models.py`, `evaluate_directional_models.py`, `evaluate_directional_fine.py`, `inspect_3240_coverage.py`: reproduction scripts. They fit no model and write only diagnostic outputs.
