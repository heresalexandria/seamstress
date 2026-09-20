# Seven-second symmetric grade tracks

This is a bounded photometric experiment after rejection of the morphing RIFE result. It synthesizes no poses or motion. No full video was rendered. The previous `*-color.mp4` previews show the **constant segment** experiment, not this new time-varying track. The later `722-slow-grade-comparison.mp4` shows the protected shared-midpoint model described below across its full support.

The slow track is suitable for a limited visual preview after protecting black and white. Its rate is comparable to ordinary measured grade evolution, and it avoids cumulative contrast collapse. Numerical measurements cannot certify that the resulting 14-second color adjustment is imperceptible; it still needs playback over the entire support, especially on a stable sky or wall.

## Explicit first-cut color map

At zero-based cut frame 361, the fitted map from the incoming clip's RGB to the outgoing clip's RGB is:

```
gain = [0.9122672957, 0.9332263830, 0.9757030604]
bias = [8.1099310257, 5.6751119819, 3.0955079542]
left_rgb ≈ gain * right_rgb + bias
```

This map is fitted to matched interior pixels around the actual cut. It is **not** a histogram match. The unprotected symmetric half-transform endpoints are:

```
left frame 360:
  gain = [1.04698135, 1.03515765, 1.01237443]
  bias = [-4.34291339, -2.98806039, -1.57654172]
right frame 361:
  gain = [0.95512685, 0.96603643, 0.98777683]
  bias = [4.14803317, 2.88657519, 1.55727138]
```

If the fitted local relation is exact, these transform both sides to the same intermediate grade. They leave all spatial coordinates and frame indices untouched.

## Track construction

Let the local affine map be `M(x) = g*x+b`, per channel. Its fractional power is:

```
M^p(x) = g^p*x + b*(g^p-1)/(g-1)
```

For `g≈1`, use bias `p*b`. Before the cut use exponent −0.5; after use +0.5. Taper the exponent to zero over 168 frames (7.007 seconds) using `1−smoothstep5(distance/168)`, where `smoothstep5(t)=6t^5−15t^4+10t^3`. The left peak is at `cut−1`, the right peak at `cut`. Values and first two temporal derivatives reach neutral smoothly at the outer edges; the derivative of the correction is also zero at each cut endpoint.

The apparent parameter jump between the two endpoint transforms is intentional: it counteracts the measured source grade jump. It does not interpolate images. Any unmatched local color difference remains and must be inspected.

Supports do not overlap for this source's confirmed cuts. Neighboring 15.06-second segments have approximately one second near neutral between the seven-second supports. A source with closer cuts must use a constrained joint track instead of blindly composing overlapping corrections. The final segment ends before its seven-second return finishes; keep the small remaining correction rather than accelerating the return. At the last source frame its gain is approximately `[0.9925,0.9904,0.9898]` with bias `[1.504,1.518,1.398]`.

## Rate compared with the original footage

Forty-two four-frame matched-pixel controls sampled every 24 frames through the first three source segments estimate ordinary within-clip grade changes, excluding every cut. Measurement noise, changing local lighting and imperfect correspondence contribute to these controls; they are a comparison, not a perceptual threshold.

Using RGB levels 16, 48, 96, 144, 192 and 240, the median of the control's largest channel/level change is 0.0768 byte values per frame; its 90th percentile is 0.1760. Typical median change across all channel/level probes is 0.0230 per frame, 90th percentile 0.0769.

| Cut | Peak added affine change, byte values/frame | Median channel/level change at that frame |
|---:|---:|---:|
| 361 | 0.0774 | 0.0194 |
| 722 | 0.1546 | 0.0405 |
| 1083 | 0.0662 | 0.0230 |
| 1444 | 0.1123 | 0.0265 |
| 1805 | 0.1016 | 0.0286 |
| 2166 | 0.0697 | 0.0311 |
| 2527 | 0.0377 | 0.0147 |
| 2888 | 0.0725 | 0.0343 |
| 3240 | 0.0593 | 0.0297 |

The strongest modeled change, at 722, is about 3.7 byte values/second in one bright color channel; most tones change more slowly. This supports testing the seven-second support. It does not prove invisibility: gradual contrast changes may be detectable in static regions even below the variation of moving content.

## Preserve cartoon ink and highlight detail

Unprotected affine half-transforms should not ship. They can drive positive source shadows below zero: up to 5.23% of the red-channel pixels at frame 2165 in sampled range checks. At 2888 they push 5.52% of blue-channel pixels above 255. Slow timing alone does not solve clipping.

The bounded tone-curve control uses monotone PCHIP interpolation through RGB input knots `[0,16,48,96,144,192,240,255]`. Interior knot outputs follow the current affine grade; endpoints remain exactly 0 and 255. Reject any non-increasing knot sequence. This keeps black outlines and white endpoints, keeps the middle tonal values close to the measured affine mapping, and avoids hard clipping. It changes color only, not shape. Before converting to a uint8 LUT, validate dense monotonicity and endpoints for every track sample.

On held-out pixels from the exact cut pair, protected symmetric tone curves give mean absolute error 2.41 at 361 versus raw 4.45, and 2.46 at 722 versus raw 4.46. At 2166, protection trades some fit accuracy for range safety (2.99 vs unprotected 2.68, still below raw 5.96). These values differ slightly from the three-pair aggregate because this control uses only the immediate cut pair.

Data and executable research are in `slow-grade-analysis.json`, `slow-grade-summary.json`, `slow-grade-tone-controls.json`, `research/slow_segment_grade.py` and `research/slow_grade_tone_controls.py`. `slow-grade-track.png` shows the added correction at RGB 96. A production implementation should retain explicit identity behavior, finite positive gain validation, no-overlap handling, monotone tone checks, and a recorded source hash.

The appropriate next validation is a 14–16-second source/color comparison around one strong cut, at full resolution, with geometry handled independently. Inspect steady backgrounds for contrast pumping and outlines for color changes; do not accept on cut residual alone.

## Current bounded model: one protected transfer and its inverse

`protected-midpoint-curves.json` is the recommended **experimental** data for a limited combined preview. It contains nine objects with `frame`, `support_before`, `support_after`, `left_lut` and `right_lut`. Both supports are 168 frames. Each LUT is 256×3 floating-point RGB, indexed by input channel value; linearly interpolate for fractional input values. This is the parent's shared-midpoint design, which improves on independently protecting the two half-transforms above.

First construct a single monotone right→left transfer `T` from the fitted gain/bias, using the protected PCHIP knots above. Then form:

```
left_lut(L)  = 0.5 * (L + inverse_T(L))
right_lut(R) = 0.5 * (R + T(R))
```

When `L=T(R)`, both maps yield exactly the same common midpoint. This averages scalar color values through a pointwise transfer function; it does not average neighboring frames or generate poses. For fractional frame support weight `w`, use the transfer `identity + w*(lut−identity)`.

For source frame `f<cut`, distance is `cut−1−f` and use the left LUT. For `f>=cut`, distance is `f−cut` and use the right LUT. Inside the support use `w=1−smoothstep5(distance/support)`; otherwise identity. Weight is exactly 1 at source frames `cut−1` and `cut`, exactly 0 at the outer support endpoints. This source-frame convention must remain explicit if a separate retiming operation changes output frame indices.

The saved LUTs are strictly monotone, preserve exact black/white, and introduce no clipping. Dense continuous inverse consistency is exact to floating-point precision; the 256-entry linearly interpolated LUT approximation differs by less than 0.020 byte values in the tested correspondence identity. No supports overlap in this source. Endpoint preservation and monotonicity hold throughout the temporal ease because it is a convex combination with identity.

| Cut | Held-out MAE, raw → protected midpoint | Peak added change, byte values/frame |
|---:|---:|---:|
| 361 | 4.45 → 2.48 | 0.081 |
| 722 | 4.46 → 2.48 | 0.151 |
| 1083 | 3.26 → 2.48 | 0.073 |
| 1444 | 3.71 → 2.32 | 0.115 |
| 1805 | 4.23 → 2.46 | 0.105 |
| 2166 | 5.96 → 3.19 | 0.073 |
| 2527 | 3.78 → 2.65 | 0.041 |
| 2888 | 4.95 → 2.44 | 0.077 |
| 3240 | 4.09 → 2.15 | 0.062 |

The exact values are in `protected-midpoint-evaluation.json`; `research/protected_midpoint_grade.py` reproduces them and the source/color comparison. Protection leaves more color residual at 2166, but prevents crushed shadows. This is an explicit tradeoff worth keeping. The model's slow temporal rate remains comparable with the measured normal-source controls; playback acceptance remains necessary.
