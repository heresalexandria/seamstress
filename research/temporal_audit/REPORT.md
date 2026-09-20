# Temporal audit of the encoded v2 output

This audit compares decoded original and `output/v2/repaired.mp4` frames across all nine joins. The original proxy receives v2's 3.6764% common crop so framing is comparable. Analysis uses full-picture 640×360 proxies, consecutive picture differences, edge energy, and sparse tracked motion over one, two, and four frames. Frame strips were inspected directly. This is not a normal-speed perceptual acceptance test.

The remaining defects involve **how the picture evolves**, not simply the residual between two registered frames. V2 retains many redraw/detail changes, and its fading geometry corrections can create a second camera movement after the original join. At 361, 3240, and 2888, the camera reverses scale direction during recovery. At 2166, vertical motion overshoots in the continuation. A bridge should begin with the original source, not inherit these warped temporal neighborhoods.

## Recommended original-source endpoints

The machine-readable selection is [`../../plans/bridge-windows.json`](../../plans/bridge-windows.json). `start` and `end` are preserved anchor frames; only the interior frames are replaced. All indices are zero based. See [`selected-endpoints.jpg`](selected-endpoints.jpg) for source context four frames before the start, at the start, on both sides of the cut, at the end, and four frames after the end.

| Join | Start → end | New interior frames | Primary residual / endpoint reason |
| ---: | ---: | ---: | --- |
| 361 | 349 → 365 | 15 | Forward magnification stops; face/car/sky detail changes remain. Earlier start permits deceleration without backing away. Preserve mouth action with interior references. |
| 722 | 714 → 726 | 11 | Foreground car and moving background require separate motion. Outfit, wheel, and silhouette details reset. End before the full head/body turn. |
| 1083 | 1075 → 1089 | 13 | Skater, bumper contact, hair and backdrop change independently. V2 still has local nonrigid disagreement; preserve hand contact and balance. |
| 1444 | 1436 → 1448 | 11 | Foreground proportions and sharpness change. End before the next leg push around 1452, and retain the intervening rise/crouch action. |
| 1805 | 1797 → 1811 | 13 | Strong forward-travel parallax and cadence change; v2 visibly bends poles/foreground geometry. Keep rigid streetscape shapes while layers move at different speeds. |
| 2166 | 2154 → 2170 | 15 | Aerial pullout slows by about 3× and vertical drift changes direction. Street-grid/sky detail and grade reset require temporal synthesis. |
| 2527 | 2519 → 2537 | 17 | Apartment furniture and subjects redraw; seven initial held intervals pause action. End ten frames after the cut to reach actual new motion. |
| 2888 | 2880 → 2892 | 11 | Flight, skyline and river have incompatible depths; grade and composition change. V2 introduces a late reverse zoom. Keep hand contact and two distinct body silhouettes. |
| 3240 | 3228 → 3244 | 15 | Push-in stops; columns, doorway, lions and stairs redraw. Earlier start avoids reverse zoom; end before the woman's natural turn around 3245. |

These are candidate synthesis windows, not proof that a two-endpoint interpolator can repair every interior frame. Character action is not always monotonic between the endpoints. The plan includes interior references to keep the crouch, mouth movement, hand contact, and walking action from being flattened into a simple morph.

## Why asymmetric windows matter

For frame 361, symmetric anchors 355→369 have only **+0.053% scale change per frame** on average, while the incoming camera rate is **+0.279% per frame** and outgoing rate is approximately zero. A cubic path matching that incoming speed must overshoot and reverse. Anchors **349→365** instead average **+0.157%**, with incoming **+0.301%** and nearly zero outgoing speed: a sensible deceleration is possible.

At 3240, symmetric anchors 3234→3248 average **+0.141%** while incoming speed is **+0.899%**. The selected **3228→3244** averages **+0.414%** with incoming **+0.898%**, permitting a smoother stop before the character turn. The across-window transform has weak support because the architecture itself changes, so these numbers constrain the broad camera trend rather than every pixel.

At 2166, selected anchors **2154→2170** provide a strongly supported aerial trend: scale rate goes from **−1.473%** to **−0.420%** per frame, averaging **−1.016%** across the window. Vertical center motion changes from **+4.47** to **−0.35 pixels/frame** at proxy resolution. A coherent bridge should decelerate scale and change vertical direction once. V2 instead adds a later downward/scale recovery wave.

For 722, 1805, and 2888, single-affine estimates mix different depth layers. Across-window inlier fractions are approximately **42%, 29%, and 28%** respectively. Treating those estimates as the camera would misdirect synthesis. The selected windows also use full-picture content and action timing.

## Measured v2 residuals

The table is diagnostic, not a ranking of perceived quality. Raw picture difference rises during legitimate fast motion; edge energy can change with scale, resampling, and detail. It is the discontinuity and its temporal context that matter.

| Join | Original cut picture MAE | V2 cut picture MAE | V2 edge-energy step | Additional v2 behavior |
| ---: | ---: | ---: | ---: | --- |
| 361 | 14.22 | 4.99 | +5.8% | Late scale rate becomes about −0.097%/frame after a positive incoming zoom. |
| 722 | 15.67 | 5.76 | +4.6% | Subject/background registration remains incompatible; local outline/contrast pop persists. |
| 1083 | 12.98 | 7.39 | −1.4% | Across-cut nonrigid tracked residual grows despite lower raw difference. |
| 1444 | 13.48 | 7.25 | +5.9% | Detail and foreground proportion reset remain. |
| 1805 | 13.13 | 15.58 | +2.2% | Real parallax remains large; field correction introduces bent poles and shape breathing. |
| 2166 | 16.59 | 18.17 | −0.6% | Post-cut vertical center velocity later reaches −1.705 px/frame, versus source −0.348. |
| 2527 | 12.77 | 4.75 | +4.1% | Initial-hold filling does not resolve furniture/subject appearance regeneration. |
| 2888 | 20.90 | 11.04 | +4.4% | Late scale becomes −0.581%/frame, versus approximately stationary source scale. |
| 3240 | 30.88 | 12.59 | +0.4% | Late reverse zoom and a second motion burst around +15 frames; stairs/door detail still changes. |

V2's lower registered error therefore cannot support a seamlessness claim. It often makes two pictures more similar while leaving an implausible trajectory or evolving shape.

## Synthesis implications

- Prefer a coherent run of at least six distinct temporal states over a single generated midpoint. Fill the selected window to its exact frame count while preserving both original endpoints and the rational timebase.
- Condition on source motion context before and after the anchors, plus the listed interior action references. A two-image morph cannot recover nonmonotonic gestures or resolve all occlusions.
- Preserve rigid lines and semantic relationships: straight streetlights, fixed building columns, hand contact, skateboard geometry, and separate limbs. Low residuals are not a substitute for this review.
- Grade and texture must evolve across the run. Do not leave the appearance change concentrated in the one frame where ownership switches between source images.
- Check the first and last generated transitions as carefully as the original seam. A bridge that fits the middle but enters at the wrong velocity simply moves the hitch.
- For 361, 2166, 2888, and 3240, compare camera trajectories beyond both endpoints and reject a new reverse zoom or delayed recovery wave.

Artifacts: `strip-<frame>.jpg`, `curves-<frame>.png`, full records in `curves.json`, numerical summary in `summary.json`, and alternate endpoint measurements in `bridge_candidates.json`. The script is `../temporal_audit.py` and reads only the original and rendered videos.
