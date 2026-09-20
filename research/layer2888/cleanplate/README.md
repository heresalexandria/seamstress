# Static generated clean plate and bounded layer proof

**Final verdict: the two-second layer proof is rejected for integration. Frame2888 remains unresolved.** The clean plate is a useful asset, but pixel coverage and good still-image alignment did not establish temporal continuity. No production files or the accepted eight-join candidate were changed.

## One generation, saved reproducibly

The built-in `image_gen.imagegen` tool was called exactly once to remove the sloth, woman, jetpacks and flames from a640×512 crop of original frame2888. No API key was read and no character poses were generated. The generated plate was copied into the workspace:

- `generated-roi.png`: generated1402×1122 opaque clean background.
- `source-2888-roi.png`: original crop, native box[288,160,928,672].
- `prompt.txt`: exact request.
- `input-metadata.json`, `generation-metadata.json`: source/crop, generated path, dimensions and SHA256 hashes.
- `registration-report.json`: fixed affine registration against visible original background.

The generated visible architecture aligned unusually well:718 visible-background matches,691 initial affine inliers, final median reprojection error0.32px/p90 0.89px at source scale. The selected registration adjusts vertical scale by0.24%. Existing visible building lines, framing and perspective remained close; the hidden background is necessarily reconstructed rather than a recovered original.

## Static proof stages

The first constrained patch changed only3,189 exposed pixels in the earlier v2 diagnostic. It filled all those holes, with **zero changed pixels outside the hole mask**, but necessarily retained existing bad donor strips. `patched-2888.png` and the comparison crops show that limitation.

The subsequent recomposition removes all source-subject background contamination under independent foreground/background mappings and fills it from the same registered plate. Its flame matte uses the original flame colors against the estimated clean background; it does not generate a flame or blend different character poses. Opaque source nozzles and subject interiors are preserved. The native stills in `recomposed/` were coherent enough to justify a short temporal test. They did not certify a repair.

A masked background homography was also tried independently in `../background-homography/`. It preserved straight-line geometry mathematically but worsened actual registration: background RGB residual9.81→10.72 and contours>2px11.5%→16.1%. It was rejected; the temporal proof retained the fixed affine background map.

## The single temporal proof

- `proof/candidate.mp4`: native1280×720,48 frames at24000/1001fps,2.002seconds, matching-duration source audio.
- `proof/comparison.mp4`: original beside experimental layers, with the same1.06 view crop.
- `proof/render-report.json`: frame-by-frame coverage, foreground preservation, boundary mismatch and the explicit rejection.
- `proof/tracking.json`: original foreground and background tracking evidence.
- `../camera-ease.json`: separately measured background-only five-frame balanced camera easing. Its extra correction is exactly identity outside its support and retains at least4.55px outer source clearance under the1.06 view.
- Source range is[2864,2912), and the local layer correction returns to source geometry in frames2900–2910. Frames2910 and2911 are exact source under the common view crop.

No original poses were synthesized or crossfaded. Each output foreground comes from that same-time original source drawing. Source foreground cores were protected, and the view/camera/layer transforms were combined before resampling. The background removal guard was reduced to3px plus explicit flame/glow coverage. The proof reports **zero uncovered pixels across all48 frames and zero changed identified foreground-core pixels**. Those properties are necessary but insufficient.

## Direct visual failures

The proof fails independently of color grading:

1. **Frame2887:** a visible city/sky halo follows the region above the sloth's head and the woman's/sloth's extended arms. Its boundary moves against the existing scenery. See `proof/failure-2887-halo-head-and-arms.png` and the native frame.
2. **Frame2904:** duplicate pale contours remain below the woman's extended right forearm/hand, approximatelyx810–875,y355–385. See `proof/failure-2904-duplicate-right-hand.png`.
3. **Frame2904:** repeated pale boundary fragments appear below the woman's shoes and the sloth's feet. See the corresponding `failure-2904-duplicate-shoes.png` and `failure-2904-duplicate-sloth-feet.png`.
4. The single affine plate mapping does not follow all city depths. Its background feature p90 residual reaches about30px by2911. The boundary between reconstructed strips and the original city consequently changes from frame to frame. Boundary color-error measurements corroborate this, but the rejection is based on visible contour and city discontinuities, not an arbitrary score threshold.

### Grade confound

This research proof also eases its local incoming gain back toidentity during the final10 frames. That creates brightening absent from the intended production approach, which applies the already validated slow protected grade to the entire composite. Therefore the exit's brightness change is **not** evidence against the intended geometry treatment. The visible matte/parallax failures above remain independent of that confound, so no additional rerender was justified or performed.

## Stop decision

The generated plate solved static coverage; it did not solve moving matte boundaries or depth-dependent background alignment. Transporting a mostly rigid foreground mask and a single background affine is insufficient for this proof. The result is not an invisible transition and was not integrated. No additional generation, dense warp, inpainting experiment or full-video render was performed after the rejection.
