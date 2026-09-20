# All-join geometry feasibility

No full film was rendered. All methods sample one original frame per output frame. No neural interpolation, crossfade, or nonrigid deformation is used.

`cut-measurements.json` contains independently fitted incoming-to-outgoing affine matrices and local camera-rate estimates for all nine joins. `affine-crosscheck.json` compares the seven available matrices in the root diagnostic: they agree within0.02–0.67 native pixels RMS at four corners and center. The common vertical aspect impulse is real; disagreement between estimators does not explain the cumulative drift.

`summary.json` gives cumulative and bounded alternatives. `*-matrices.json` contains one exact source-to-output matrix per source frame, plus the constant view matrix needed to prevent any unobserved border pixels for that entire variant. These are research arrays, not approved production plans.

| Method | Final X scale | Final Y scale | Maximum anisotropy | Required total crop |
|---|---:|---:|---:|---:|
| Cumulative fixed affine |1.060|1.273|1.202|0.56%|
| Cumulative plus24-frame one-sided rate easing |1.323|1.590|1.202|32.18%|
| Symmetric half-affine, neutral in168 frames |1|1|1.017|5.31%|
| Same plus balanced±12-frame rate easing |1|1|1.017|12.73%|

The small global crop for cumulative fixed affine is misleading: its final image itself is enlarged and vertically distorted by20.2% relative to horizontal. The selected last-scene stills make this visible. Reject cumulative geometry as a general strategy.

The half-affine alternative divides the cut correction between the outgoing and incoming clips via affine matrix logarithms. It eases both sides back toward neutral over roughly seven seconds. Each cut adds only about1% aspect change to each side rather than accumulating all earlier cuts. The final segment has only107 frames, so its final neutral return is truncated to4.46 seconds; retaining some correction through the ending is another option.

The balanced camera variant uses the mean of measured incoming/outgoing velocities as its expected cut step. Over12 frames before and12 after, an integrated smoothstep creates a shared camera offset at the cut. The correction velocity cancels the native rate discontinuity; its integrated offset returns to zero afterward. This avoids the camera excursion caused by naively pinning both the cut correction offset and the support endpoints to zero while constraining their tangents. The separate `neutral168_tangent` experiment demonstrates that rejected approach and needs30.2% crop.

Balanced camera easing is feasible for the first join, but cannot be applied identically everywhere without excess crop. Its per-join no-fill crop bounds are:361 2.56%,722 2.48%,1083 1.84%,1444 1.89%,1805 1.78%,2166 5.92%,2527 2.41%,2888 12.73%,3240 5.16%. High zoom/pan rates at2888 create the worst camera-offset excursion. A narrower local rate-easing interval or a gently varying local view would reduce the global crop cost, with different acceleration/framing tradeoffs. Do not crop the entire movie12.73% just to accommodate this provisional rule.

Native proofs with original audio are `balanced-361-audio.mp4` and `balanced-2166-audio.mp4`; comparisons use an identically cropped source. Their constant preview crop is2.56% and5.92% respectively. No color correction is applied. `balanced-proofs.json` includes independently tracked two-frame camera curves from freshly decoded encoded proofs. The strongly varying original drawing/camera cadence before2166 is preserved, so single-step velocity does not become perfectly uniform.

Camera rates remain estimates. Parallax and animated foreground movement can bias a global camera model, especially722,1805,2527, and2888. The confidence rule declines rate correction when registration evidence is weak, but it is not a perceptual guarantee. Review the original and corrected short windows before any whole-film plan.

## Current export for integration

Use `eight-global-matrices.json`, produced by `../export_eight.py`. It provides all3347 `frame_matrices`, plus `view_matrix`, `geometry_mode: affine`, and `edge_extension_pixels: 0`. Source metadata is included; the root conform plan should supply the source fingerprint and color segments.

Join2888's entire global correction is identity. The layer diagnostic found that the compromise global affine shifts the foreground10–18px to favor the city, so that join is explicitly unresolved pending a separate source-only layer treatment. The remaining eight joins use the bounded half-affine policy; reliable large camera-rate differences at361,2166,2527,3240 receive balanced±12-frame easing. Other rate corrections were declined because their evidence was weak or the estimated difference small.

The minimal constant view enlargement is1.062968494, removing5.923834% of total width and height (37.91 horizontal pixels and21.33 vertical pixels per side). Inverse mapping all3347 output rectangles proves zero unavailable-source pixels with at least2.00031 native pixels of clearance for cubic support. Worst coverage is frame2165. The maximum affine singular-value ratio is1.012309, and no aspect error accumulates into the final scene.

`eight-global-camera-audit.json` records independently tracked source and transformed images. A single frame1795 flag proved to be unstable scene-depth consensus in that scene: only14.6% of tracked points support its similarity fit. `eight-global-paired-camera-audit.json` fixes the correspondence set and transports the same original landmarks through both transforms. It finds no newly introduced strong direction reversals in the eight treated joins (thresholds0.1% zoom/frame or1 native center pixel/frame). At1795 the actual correction-only zoom change is−0.00122%/frame; the large apparent velocity swing comes from source parallax/tracking ambiguity, not the tiny applied correction.

These checks establish coverage, frame identity, and bounded camera effects. They do not establish perceptual invisibility; original animation and parallax remain, and2888 is unresolved. No whole-film rendering was performed for this export.
