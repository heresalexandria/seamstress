# Independent diagnostics

The nine segment joins are confirmed by direct neighboring-frame inspection plus a full-frame motion/grade scan. Frame numbers are zero-based and name the first frame of the new segment. The cadence is 361 frames for the first eight clips, then 352 for the ninth clip.

| Frame | Time (seconds) | Similarity scale | Median B,G,R shift | Notes |
|---|---:|---:|---|---|
| 361 | 15.056708 | 0.99306 | [1.0, 1.0, -3.0] | Subtle foreground size contraction and per-channel color change; streetlight/background position mismatch. First new generation frame then nearly held frame. |
| 722 | 30.113417 | 0.99500 | [2.0, -1.0, -3.0] | Car and standing character slightly contract/reposition upward; sky and costume shift color. Background motion does not share the same foreground warp. |
| 1083 | 45.170125 | 0.99641 | [0.0, 1.0, 0.0] | Small contraction plus nonuniform redraw of car and skateboarder; slight grade and line/texture change; first new frame held. |
| 1444 | 60.226833 | 0.99089 | [1.0, 1.0, -1.0] | Foreground contracts/repositions while relative car/skateboarder alignment changes; small cool channel shift. |
| 1805 | 75.283542 | 0.99439 | [1.0, 0.0, -1.0] | City street wide shot: contraction breaks the preceding strong camera scale change. Geometry match is relatively coherent, color change mild. |
| 2166 | 90.340250 | 1.00070 | [-5.0, -8.0, -7.0] | Aerial city: global darkening and small crop/geometry reset interrupts pullout/tilt; new clip begins with much slower motion. |
| 2527 | 105.396958 | 0.99759 | [2.0, 2.0, 2.0] | Apartment: whole-frame lightening, shelf/furniture/background proportions change slightly, main subjects redraw; starts with multiple held frames. |
| 2888 | 120.453667 | 0.99092 | [-4.0, -6.0, -8.0] | Daylight flight: strong darkening/color shift and local skyline/building/character composition reset. Only 47% of tracked features support one global similarity, so one affine crop cannot reconcile it. |
| 3240 | 135.135000 | 0.98333 | [-2.0, -3.0, -3.0] | Library steps: strong geometric redraw and contraction despite preceding forward zoom, plus darker grade. Door/columns/lions require different local corrections. Starts with multiple held frames. |

Scale and channel shifts above are diagnostics, not final correction parameters. They compare adjacent frames, so they include genuine scene motion. Measurements are on 480×270 proxies; shifts are in 8-bit channel values. Color residual is measured after sparse optical-flow similarity alignment. One global transform leaves obvious local redraws.

Very large raw frame differences at 23.8–24.3s, 67–75s, 93.0s, 114.8–116.3s and 124–125s mostly come from fast camera moves and changing scene content. Thresholding raw difference alone is therefore not a reliable seam detector here. The actual seams are much stronger outliers in color residual after registration, and have the segment cadence.

The source frequently holds animation on multiple adjacent frames; do not classify every duplicate as a seam or smooth the entire film indiscriminately. Several actual seams also introduce held startup frames, which can produce a temporal hitch even once appearance matches.

Fresh approach suggestion: use separate low-frequency photometric correction, locally varying correspondence with edge-aware confidence, and temporal residual smoothing across the join. A single global transform cannot repair the changing relationships between foreground and background. A single generated inbetween could help true topology/occlusion failures, but it cannot by itself make the velocity and color neighborhoods continuous.
