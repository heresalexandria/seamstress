# Fixed affine feasibility, native source drawings

These are measurements of consecutive original frames, with one fixed affine transform and one global per-channel affine RGB grade applied to the incoming frame. No interpolated frames, blends, local warps, neural rendering, or redrawing are involved. The matrices map incoming pixels into the previous source frame's coordinate system. They are diagnostics, not a finished sequence conform: matching two endpoint images alone does not solve camera velocity or startup holds.

Each numbered directory contains the native original frames, corrected incoming frame, side-by-side regions, a hard-toggle blink, and JSON metrics. `../affine-all-joins.json` contains the six requested new checks plus the previously tested 361. The separate parent experiment covers 722 and 1444.

Canny contour distances are symmetric nearest-contour distances measured at 1280×720, with a 10-pixel outer guard. RGB MAE is in 8-bit channel levels. The neighbor column is the immediately preceding ordinary pair, also fit with an affine and grade; normal moving frames and held drawings have very different residuals.

| Incoming frame | Raw RGB MAE | Affine + grade MAE | Contours >2px | Prior ordinary pair >2px |
|---:|---:|---:|---:|---:|
| 1083 | 13.58 | 4.62 | 2.17% | 5.26% |
| 1805 | 13.60 | 4.59 | 2.81% | 37.51% |
| 2166 | 17.40 | 4.93 | 2.12% | 1.95% |
| 2527 | 13.75 | 4.25 | 1.13% | 1.16% |
| 2888 | 22.33 | 10.99 | 13.04% | 8.76% |
| 3240 | 31.79 | 4.67 | 0.94% | 0.93% |

## Regions actually inspected

- **1083, bridge side view:** woman face/hair, trouser outline, hands, skateboard; sloth arms, car outline and wheel; bridge cables. The fixed affine makes the major contours nearly coincide. Remaining differences are car paint/reflection texture and wheel spoke phase. The wheel phase is compatible with rotation and is not evidence of contradictory character geometry. Minor skateboard wheel/underside details also differ. The next moving ordinary pair has 13.24% contours >2px, well above the corrected seam's 2.17%. A fixed reframe and grade looks plausible here.
- **1805, street rear view:** both protagonists, rear bumper/lights/road marks; left building windows and bicycle/car; right shop, pedestrians, tree and fruitstall. Major geometry aligns closely. Residual is mainly brightness/texture/sharpness, especially glows and small distant detail. No obvious changed face/limb topology in the inspected regions. Natural source camera/background movement is much larger in adjacent ordinary pairs. A fixed reframe and grade looks plausible here.
- **2166, aerial city:** central road grid and park, left grid, entire distant skyline. Building roofs, road intersections, silhouettes, and lights largely retain their positions after affine. Remaining errors are small texture/contrast changes and tiny distant-light details. No obvious contradictory structural linework; corrected contour residual is about the preceding ordinary pair. A fixed reframe and grade looks plausible here.
- **2527, apartment:** woman's face, both hands and tracksuit contours; sloth face/feet/tablet/sofa; floor, shelving, lamp and plants. Major and facial contours closely agree. Remaining residual is mostly flat-color/shading/texture and slight edge sharpness. There is no convincing geometry problem requiring a new pose. This seam has an independently established prolonged source startup hold; registration does not cure that hold.
- **2888, daylight flight:** both characters and jetpacks, Empire State spire/facade, near city, right water/skyline. A single affine does not align near/character/distant layers simultaneously. The selected global fit aligns much of the city but leaves both characters vertically displaced by roughly 10–18px. This is a relative scene/camera-motion reset; the source characters retain recognizable poses and topology, so it should not be described as proven character redrawing. See the neighborhood feature analysis below.
- **3240, library:** woman and sloth outlines/hands/feet, door and inscription, ceiling/arch, right lion/column/stairs. Contours almost coincide after affine. Residual is primarily shading/color/sharpness rather than pose change. The measured 0.94% contour mismatch is effectively the same as the preceding ordinary pair's 0.93%. No morph reconstruction is justified by the endpoint geometry.

## 2888: relative background jump is real, not just bad grade

`2888/region_motion.json` uses native matched features in fixed spatial regions; these are regional statistics, not pixel warps. Positive y is downward. Region boxes can contain multiple depths, so medians are diagnostic rather than perfect object tracks.

| Pair | Characters median dx,dy | Empire State median dx,dy | Bottom city median dx,dy |
|---|---:|---:|---:|
| 2884→2885 | +0.44,+1.78 | −1.28,+2.41 | +2.13,+9.63 |
| 2885→2886 | ~0,+0.02 | −0.05,+0.03 | −0.01,−0.01 |
| 2886→2887 | +0.49,+2.00 | −2.24,+4.03 | +4.51,+17.50 |
| **2887→2888 seam** | **−0.36,−0.26** | **+1.60,−13.96** | **−0.04,−21.82** |
| 2888→2889 | ~0,~0 | ~0,−0.25 | ~0,−0.39 |
| 2889→2890 | −0.01,+0.10 | −0.01,−2.16 | ~0,−2.00 |

The background moves downward before the seam, jumps upward at the seam while the characters barely move, then proceeds upward at a much smaller rate. A single constant transform can trade character alignment against background alignment but cannot independently correct these layers. Calling every residual at this cut a redraw would overstate the evidence; the confirmed problem is contradictory relative layout/motion across the cut.

## Feasibility judgment

Five of the six new checks give strong evidence for source-preserving fixed affine reframing plus color correction. The consistent extra vertical scale is material: similarity-only diagnostics substantially overstated residual geometry. The large remaining exception is 2888's independently moving layers. These static findings do not certify seamless playback: original hold cadence, camera rates, grade consistency across whole segments, borders, and accumulated reframing still need temporal review.
