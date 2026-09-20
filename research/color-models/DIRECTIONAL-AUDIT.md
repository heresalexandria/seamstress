# Directional color-response audit: all nine plus fine 3240

All nine standard directional models, and the finer 3240 model, pass the requested numerical checks. No automatic attenuation is recommended. They remain candidates for native material/temporal review; this audit does not establish perceptual perfection.

The saved models were evaluated without fitting, modifying production code, or rendering a video. Original native source anchors receive the existing baseline geometry and LUT unchanged; RGB values are then sampled every fourth row and column, without resizing. Models and their SHA256 hashes are recorded in the JSON reports.

## Range and endpoint behavior

The response uses a smooth RGB black/white gate, a bounded requested offset, and direction-dependent available range `h`. A channel correction is `h*tanh(offset/h)`, with an explicit numerical guard for zero headroom. For finite parameters and valid RGB input, its magnitude cannot exceed `h`; output therefore remains in [0,255]. True RGB black/white are exact. A zero channel in another saturated color may rise, which resolves the older per-channel-headroom model’s asymmetric matching problem.

No saved candidate clips in the sampled grid or native palettes. Numerical derivatives use ±0.25 byte inside the gamut and one-sided differences at 0/255, never invalid input values.

The soft gamut bound becomes flat when a requested correction pushes an already saturated channel outward. This makes the full RGB Jacobian singular at some 0/255 channel boundaries; it is not a color fold. To distinguish boundary flatness from loss of interior color contrast, the audit also evaluates: (a) an RGB cube restricted to 8..247 in every channel, and (b) the Jacobian columns corresponding to native input channels in 8..247, while retaining all output channels. Negative own-channel derivatives and negative determinants remain checked over the entire gamut, not just this interior subset.

## Per-cut results

Each standard side has 171,955 broad RGB/XY samples (17³ RGB ×7×5 positions), 35,000 interior RGB/XY samples, 57,600 native anchor samples, and 57,600 native samples 84 frames away. No sampled negative own-channel derivative or negative determinant appears in any standard model or fine 3240. The following minima exclude the expected saturated-boundary plateau as described above.

| Join | Interior-grid min singular | Native active-color min singular | ±84-frame active min singular, full strength | 16×8px motion p99/max (byte) | Native max correction (byte) | Recommendation |
|---:|---:|---:|---:|:---|---:|:---|
| 361 | 0.568 | 0.681 | 0.668 | 0.148/0.278 | 7.26 | Accept for native review |
| 722 | 0.294 | 0.635 | 0.627 | 0.124/0.485 | 8.49 | Accept for native review |
| 1083 | 0.618 | 0.776 | 0.804 | 0.127/0.625 | 8.17 | Accept for native review |
| 1444 | 0.458 | 0.653 | 0.674 | 0.130/0.355 | 10.06 | Accept for native review |
| 1805 | 0.553 | 0.692 | 0.720 | 0.119/0.399 | 5.96 | Accept for native review |
| 2166 | 0.610 | 0.752 | 0.716 | 0.140/0.344 | 5.24 | Accept for native review |
| 2527 | 0.485 | 0.544 | 0.577 | 0.127/0.496 | 7.95 | Accept for native review |
| 2888 | 0.783 | 0.868 | 0.804 | 0.107/0.340 | 5.21 | Accept for native review |
| 3240 | 0.658 | 0.723 | 0.750 | 0.121/0.377 | 6.02 | Accept for native review |
| Fine 3240 | 0.511 | 0.643 | 0.646 | 0.349/0.943 | 10.07 | Accept for native review |

The actual ±84-frame correction is half strength under the 168-frame quintic support. Across the standard models, the minimum full 3×3 Jacobian singular value at those half-weight samples is 0.499; the table deliberately reports full-strength active-color values as a stricter extrapolation check. No reversal appears in the sampled maps.

Fine 3240 has more spatial sensitivity than the standard 3240 candidate: p99 change under a 16×8 native pixel translation is 0.350 versus 0.121 byte, maximum 0.943 versus 0.377 byte. Doubling displacement gives p99 ≤0.694 byte/max 1.637 bytes. At its sampled anchors, changing only the 168-frame quintic support weight contributes at most 0.113 byte/frame. These support proceeding to the visual proof, with attention to moving clothing or skin crossing the smaller spatial neighborhoods.

## Strong 2527 exception

The older strong per-channel-headroom 2527 model is separately defensible if native table/outfit evidence favors it. Valid-gamut rechecking finds no folds or negative own-channel derivatives, but strong shadow compression remains: minimum native singular 0.118; sample [11,1,0]→[1.85,.139,0]. That is a loss of very dark distinctions, so its acceptance depends on identifying those pixels as the lifted black table rather than assuming every low-error fit is valid. Root’s separate material review provides that evidence. Do not attenuate this correction indiscriminately just because its shadow gain is small. Full details and other older-model recommendations are in `STRONG-RBF-AUDIT.md`.

## Limits and reproducibility

This is a bounded numerical audit, not a proof of monotonicity over the continuous RGB/XY domain. It does not replace checking moving materials through the whole temporal support. RBF confidence indicates proximity in RGB/position, not semantic identity. Pointwise RGB changes cannot blur or move an outline, but can change its contrast.

- `directional-rbf-audit.json`: all-nine broad gamut, native anchors, ±84 frames, range/endpoints, motion and support-weight sensitivity.
- `directional-interior-audit.json`: all-nine plus fine 3240 interior/active-color metrics, including actual away palettes.
- `fine3240-broad-audit.json`: extra full-gamut check for the finer model.
- `strong2527-valid-gamut.json`: strong 2527 native recheck with valid endpoint differences.
- Scripts: `audit_directional.py`, `audit_directional_interior.py`, `audit_strong_rbf.py`, `audit_strong_temporal.py`.

No production files, geometry, timing, or full-video output were changed by these audits.
