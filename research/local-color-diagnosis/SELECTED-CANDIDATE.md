# Selected candidate: native color evidence and encoded audit

The selected plan is `plans/IYTYT-color-refined.json`: directional response at every cut except the stronger headroom response at 2527 and finer directional model at 3240. `selected-candidate-preencode-summary.json` verifies that each selected plan model exactly matches the model used in the native material evaluation. It records the plan hash and all baseline/v1/selected regional measurements.

The selected models' equal-region mean absolute signed RGB biases are 0.687, 0.307, 0.782, 0.603, 0.888, 0.607, 0.575, 0.691 and 0.459 at the nine cuts respectively. These are pre-encode diagnostic summaries, not proof of invisible seams. The parent selected directional at 1083/1805/2888 despite nearly tied material evidence because wider validation improved and the observed local regressions were small.

Known residuals/regressions to retain during encoded/playback review:

- 361: lower-fur red remains about −2.31 levels; candidate sky-blue bias remains about −1.41.
- 1083: the sampled tracksuit retains about +3.14 blue levels.
- 1805: sky red grows from +0.77 under v1 to +1.66 under the selected response; mid-distance foliage retains −2.64 green, improved from −3.26.
- 2527: black table improves substantially but retains about `(1.58,0.99,0.97)` RGB levels. Exact outgoing black remains black.
- 2888: fur bias becomes about `(−0.83,−0.28,−1.44)`, slightly worse than v1; broad sky improves. Its separate geometry issue remains unresolved.
- 3240: central jacket and trousers improve strongly, with red biases −1.53 and −0.78. The broader cyan outfit class still averages −2.84 red, so other outfit portions are not fully matched.

The parent reports that the selected models passed the independent response audit. This native ROI study does not independently certify the temporal envelope, derivative/gamut behavior, audio, frame count, or codec metadata.

## Encoded audit completed

The finalized MP4 was audited successfully. Reproduce the audit with:

```sh
.venv/bin/python research/local-color-diagnosis/evaluate_encoded_candidate.py > research/local-color-diagnosis/encoded-candidate.log
```

The script compares the actual `output/IYTYT-source-conform-eight-joins.mp4` and `output/IYTYT-source-conform-color-refined.mp4`. It checks finalized metadata and frame count before decoding all 18 required frames in one pass per movie. It writes only research diagnostics under `encoded-candidate/`; it never fits, modifies or re-encodes either movie.

The material rectangles and pre-encode baseline flow remain frozen. Baseline and candidate are measured at identical corresponding native pixels. The primary comparison additionally requires both encoded versions to pass the same strict forward/backward flow, contour-distance and RGB-difference gates, plus agreement with the frozen flow. The report includes retained counts and both frozen-gate and shared-encoded-gate results so changed sample acceptance cannot conceal a regression.

Four versions are compared on each shared sample set: baseline pre-encode, exact rounded candidate pre-encode, actual baseline encoded and actual candidate encoded. This separates remaining color mismatch from encoding-induced bias. Exact pre-encode candidate anchors have already been cached with model/input hashes by `--prepare-only`; they preserve the baseline geometry and grading and apply only the selected pointwise color correction.

All nine joins are now measured and `encoded-candidate/native-material-evaluation.json` records `completed: true`. The actual encoded improvements survive: mean signed material bias falls from 3.337 to 0.752 levels across 59 frozen-reference regions (77.5% reduction); the independently shared encoded gate confirms 76.9% across 54 regions. Encoding adds about 0.315 levels of mean absolute signed-channel bias, with most effects below one level.

The 3240 encoded jacket and whole cyan outfit retain red biases −3.38 and −3.52. Small regressions remain in the 1805 sky and 2888 water. Several flat regions lose independent flow support entirely, including the 2527 black table; their frozen statistics and native inspection are explicitly separated from joint-flow evidence. The table still improves from RGB(9,6,8.81) to (1,1,0.79). Full findings, per-cut numbers, native visual observations and sample caveats are in `encoded-candidate/REPORT.md` and `encoded-candidate/summary.json`. This audit does not claim complete perceptual invisibility or temporal-envelope verification.
