# Final encoded native color audit

**The measured color improvements survive encoding.** The completed candidate MP4 and baseline MP4 were decoded at the exact outgoing/incoming frame indices of all nine joins. Both contain 3347 frames at 24000/1001 fps and 1280×720. This audit reads the videos; it neither refits corrections nor changes production files.

Using the original frozen correspondence points, mean absolute signed RGB bias across 59 usable material rectangles falls from **3.337 to 0.752 levels**, a **77.5% reduction**. Fifty-four of 59 regions improve. Requiring both encoded videos to independently pass the stricter shared gate yields **3.407 to 0.788**, a **76.9% reduction**, across 54 regions; 51 improve. These are equal-region signed-bias measurements, not whole-frame pixel MAE or a guarantee of perceptual invisibility.

## Per-join results

The frozen comparison retains the same original points for all four versions. The candidate pre-encode column uses the actual rounded, pointwise-corrected image before its measurement blur, making this an encoding comparison rather than a comparison against the earlier unrounded sample model.

| Incoming frame | Regions | Baseline encoded bias | Candidate pre-encode bias | Candidate encoded bias |
| --- | ---: | ---: | ---: | ---: |
| 361 | 10 | 2.961 | 0.844 | 0.884 |
| 722 | 6 | 4.228 | 0.368 | 0.372 |
| 1083 | 6 | 3.462 | 0.747 | 0.806 |
| 1444 | 7 | 3.492 | 0.607 | 0.827 |
| 1805 | 6 | 2.601 | 0.808 | 0.813 |
| 2166 | 6 | 4.323 | 0.576 | 0.584 |
| 2527 | 7 | 3.217 | 0.626 | 0.620 |
| 2888 | 5 | 2.623 | 0.904 | 0.977 |
| 3240 | 6 | 3.255 | 0.646 | 0.847 |

Encoding adds a mean absolute regional signed-channel bias of **0.315 levels**, median **0.201**, and 95th percentile **0.949** on the frozen regions. The largest local codec effects occur at 3240. The shared encoded gate's jacket subset changes by −2.49 red levels relative to the corresponding pre-encode candidate; the broader frozen jacket changes by less but still noticeably in the statistics. Encoding therefore slightly weakens some very small residuals without reversing the principal improvement.

## Salient encoded material results

Signed RGB is incoming minus outgoing. These use the original frozen points; all values are 8-bit levels.

| Region | Baseline encoded | Candidate encoded | Frozen pixels |
| --- | --- | --- | ---: |
| 361 hair | `(+7.71,+1.71,−9.47)` | `(+1.81,−0.39,−0.37)` | 750 |
| 361 teal collar | `(−3.57,−3.92,+2.36)` | `(+0.27,−0.10,+0.29)` | 300 |
| 722 hair | `(+4.87,+1.42,−10.69)` | `(+0.66,+0.45,+0.04)` | 725 |
| 1444 hair | `(+4.24,+0.31,−10.12)` | `(+0.54,−1.92,−0.48)` | 839 |
| 2527 outfit | `(−15.11,−4.14,+3.74)` | `(+0.84,+0.12,+0.11)` | 855 |
| 2527 black table | `(+9.00,+6.00,+8.81)` | `(+1.00,+1.00,+0.79)` | 3456 |
| 3240 jacket | `(−20.66,−9.29,−4.54)` | `(−3.38,+0.49,+0.54)` | 194 |
| 3240 trousers | `(−10.02,−7.52,−4.30)` | `(−0.94,+0.98,+0.03)` | 509 |
| 3240 whole cyan outfit | `(−14.78,−8.87,−4.29)` | `(−3.52,−0.18,+0.00)` | 1000 |

The trousers and whole-outfit checks were added during the 3240 fine-model diagnosis and are excluded from the aggregate, preserving the original region set. The outfit's remaining red mismatch is real measurement evidence that this join's color is improved, not completely matched.

## Regressions and sample limitations

Two originally stable regions have a meaningful increase in their small signed bias:

- **1805 sky:** mean absolute signed bias 0.607→1.014, candidate RGB `(2.05,−0.88,+0.12)`. The shared encoded gate agrees, with red about +2.18.
- **2888 water:** 0.396→1.295, candidate RGB `(−1.15,−1.67,+1.07)`. The shared gate agrees. This is a small grade regression alongside the known separate parallax/geometry reset.

Neither regression approaches the size of the corrected hair, paint or outfit flashes. Both are retained explicitly rather than hidden by aggregate improvement. Their native crops are `1805-sky-regression.png` and `2888-water-regression.png`.

The stricter independent encoded-flow gate becomes unreliable or rejects nearly everything in several low-texture regions: 361 pale hood has 0 accepted joint samples, 2527 black table 0, and 3240 steps 0; 361 face cheek retains only 2. The original frozen counts are 4311,3456,49 and560 respectively. These regions **do not have independent joint-flow verification**. Their frozen correspondence statistics remain available, and native images were inspected. In a constant black patch, the table's change is also directly apparent without needing a unique motion estimate. The 3240 door panel has only 7 frozen samples and 6 joint samples and is excluded from all aggregate judgments. Low-retention cases are enumerated in `summary.json`.

The frozen and shared-gate comparisons give nearly the same aggregate improvement, so the result is not dependent on selecting only favorable newly accepted points. Per-region conclusions still need the counts above.

## Visual inspection and scope

Native context comparisons for all nine joins were inspected. The encoded candidate retains the cleaner hair/clothing/paint agreement seen before encoding; the black-table lift is substantially reduced. The 3240 jacket and trousers are much closer in color, with a small remaining red difference. No contour displacement is introduced by this color-only operation. Existing redraw, texture and parallax differences remain visible where present, particularly the 2166 cloud texture and the already unresolved 2888 geometry.

This is a native adjacent-frame color audit. It does not certify that every transition is invisible during playback, nor audit the entire temporal correction envelope, audio, or every frame of the movie.

Files: `native-material-evaluation.json` is the complete sample-level regional report with `completed: true`; `summary.json` contains aggregate metrics and low-support caveats. `*-encoded-context.png` shows baseline/candidate before and after encoding. Full native encoded pairs are saved as `*-baseline_encoded-{left,right}.png` and `*-candidate_encoded-{left,right}.png`. The report records video metadata, exact frame indices, plan hash and region-definition hashes.
