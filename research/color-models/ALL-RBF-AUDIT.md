# All-nine saved RBF color safety audit

Read-only audit of `research/local-color-fit/all-models.json`. No fitting, attenuation, production edits, or video rendering. Native original anchor frames receive the accepted baseline geometry and LUT unchanged, then their RGB values are sampled at every fourth pixel in both axes. The color transform is pointwise RGB/XY; audit sampling does not alter output.

Each side was evaluated at 76,895 RGB/position combinations (13³ RGB cube ×7×5 spatial grid) and 57,600 native anchor RGB/position samples. Finite differences use ±0.5 byte. The conservative near-singular diagnostic threshold is minimum singular value <0.25; a negative own-channel derivative or nonpositive determinant also flags a failure. These are numerical audit thresholds, not perceptual visibility thresholds.

| Join | Grid min own-channel derivative | Grid min singular value | Grid min determinant | Native min singular value | Native max change (byte) | Native p99 change (byte) | Flags |
|---:|---:|---:|---:|---:|---:|---:|:---|
| 361 | 0.814 | 0.814 | 0.732 | 0.846 | 5.579 | 3.708 | None |
| 722 | 0.755 | 0.727 | 0.659 | 0.780 | 9.572 | 5.074 | None |
| 1083 | 0.836 | 0.810 | 0.702 | 0.861 | 5.206 | 3.605 | None |
| 1444 | 0.796 | 0.796 | 0.767 | 0.814 | 8.599 | 3.472 | None |
| 1805 | 0.839 | 0.817 | 0.765 | 0.862 | 5.744 | 3.337 | None |
| 2166 | 0.746 | 0.746 | 0.485 | 0.754 | 9.852 | 6.112 | None |
| 2527 | 0.747 | 0.747 | 0.429 | 0.758 | 8.483 | 6.096 | None |
| 2888 | 0.849 | 0.849 | 0.832 | 0.858 | 5.735 | 3.213 | None |
| 3240 | 0.817 | 0.796 | 0.743 | 0.848 | 5.983 | 2.320 | None |

All nine models pass. No attenuation is justified by this audit. Every saved model has finite coefficients and center data and limit 18; therefore the analytical no-clipping bound applies over the entire valid input gamut, independent of sampling. Exact black/white also pass all sampled positions. No sampled own-channel derivative is negative, and no sampled Jacobian is near singular.

Limits: derivative monotonicity is sampled, not a proof over continuous RGB/XY space. A positive Jacobian and valid gamut do not prove correct material grading or a seamless temporal transition. The confidence taper expresses feature proximity, not semantic identity. The earlier 361 temporal-sensitivity test remains in `rbf-audit.json`; it was not repeated here. Native maximum includes edge pixels outside the fit’s interior-pixel gate, so it can exceed the fitting report’s maximum.

Reproduce: `.venv/bin/python research/color-models/audit_all_rbf.py`. Full measurements, model hash, and all side-specific metrics are in `all-rbf-audit.json`.
