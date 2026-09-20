# Strong per-channel-headroom model audit

The limit-63.75 model remains within gamut, but range safety does not establish useful color contrast. Prefer the directional variant for general use. Do not attenuate every model automatically: some measured shadow compression is the intended removal of a lifted black level.

| Join | Recommendation for strong variant | Concrete evidence |
|---:|:---|:---|
| 361 | Hold; prefer directional | No sampled fold, but broad-gamut minimum singular value 0.086 and native minimum 0.360. Low blue values are substantially compressed. |
| 722 | Hold; prefer directional | No sampled fold; broad minimum 0.190, native 0.319. A native [150,109,77] becomes [149.2,107.8,107.9], a 31-byte blue adjustment. |
| 1083 | Numerically acceptable for visual review | No fold; native minimum singular value 0.479. |
| 1444 | Reject at full strength | Right-side broad-gamut determinant reaches −0.137; an interior red/blue-coupled color mapping folds. Native minimum singular value is 0.133. The fold is at interior RGB values, so it is not an endpoint finite-difference artifact. |
| 1805 | Numerically acceptable for visual review | No fold; native minimum singular value 0.802, native maximum correction 6.87 bytes. |
| 2166 | Hold; prefer directional | No fold, but native minimum singular value 0.214, outgoing shadow median luminance gain 0.610, shadow p99 absolute change 7.81 bytes. This needs material-specific justification. |
| 2527 | Conditionally accept on verified table/outfit evidence | No sampled fold or negative own-channel derivative; gamut safe. The strong correction intentionally compresses lifted table shadows. Details below. |
| 2888 | Numerically acceptable for visual review | No fold; native minimum singular value 0.378. It is stronger than v1, but does not repair the separate geometry issue. |
| 3240 | Numerically acceptable for visual review | No fold; native minimum singular value 0.514. Prefer the directional fine-palette candidate if its material comparison is better. |

“Numerically acceptable” permits a native temporal proof; it does not certify invisible grading. No production changes or attenuation were applied.

## 2527: a material-specific exception

A fresh valid-gamut audit uses one-sided differences at RGB0/255 and ±0.25-byte differences inside the gamut. Native left/right minimum singular values are 0.478/0.118; determinants remain positive, with a right-side minimum of 0.00387. A representative minimum-contrast pixel at normalized position [0.1845,0.7900] maps [11,1,0] to [1.85,0.139,0]. This is very dark material near the table, not a generic midtone inversion.

Incoming shadow median luminance gain is 0.418, versus 0.832 for v1. This would normally warrant caution because it removes shadow distinctions, but root's independent table ROI evidence says this material should match a nearly black counterpart. On that evidence, it is defensible to retain the stronger 2527 correction instead of reducing it indiscriminately. Check the rest of the table/outline region in the temporal proof; matched color error alone would not establish this exception.

At ±84 frames, no sampled fold appears, and the actual half-weight map stays well conditioned. Translating an unchanged anchor color by 16 native pixels horizontally and 8 vertically through the correction field changes a channel by p99 0.257 byte and maximum 1.703 bytes. Quintic fading over 168 frames adds at most 0.283 byte per frame at the sampled anchors. Those are larger than directional v2, so retain native motion review as the final test.

## Why directional handling is preferable elsewhere

Preserving every channel endpoint prevents a saturated input such as cyan R=0 from rising toward a shared R=9. The outgoing R=18 then bears all correction, encouraging color compression. Preserving only true RGB black/white while allowing bounded channel offsets resolves that asymmetry. The newer directional audit evaluates that alternative separately.

Files: `strong-rbf-audit.json` (all nine), `strong-temporal-audit.json` (native ±84 frames), and `strong2527-valid-gamut.json` (critical 2527 valid-domain recheck). The initial strong-grid run used centered endpoint differences; valid-domain handling was added afterward. Interior folds remain valid evidence, and the 2527 recheck confirms its native conclusion. The current scripts use one-sided endpoint differences for reproducibility.
