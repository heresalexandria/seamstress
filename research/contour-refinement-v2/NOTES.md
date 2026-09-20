# Native library contour experiment

Source anchors: frames 3228 and 3244, native 1280×720. Source indices, audio timing, exact anchors, and camera path remain unchanged. The experiment uses the production camera bridge and an overridden RIFE renderer only; production modules were not edited.

Compared variants:

- baseline: RIFE's native bilinear endpoint warps and learned sigmoid mask.
- cubic: the same predicted gathers sampled once with cubic interpolation.
- refined: cubic gathers composed with a confidence-weighted DIS residual correction, capped at 3 px, using the opposite endpoint's learned blend weight. Residual fields are temporally smoothed and constrained to positive Jacobians, with a zero-slope temporal envelope at both anchors.
- masksharp: cubic gathers and sigmoid(2 × learned mask logits), preserving a continuous rather than binary source choice.
- refined_masksharp: both refinements.

The useful improvement is cubic resampling, with modest further benefit from sharpening the blend mask. At the midpoint, mean absolute Laplacian in the character contour crop is 8.55 baseline, 9.70 cubic, 9.72 refined, 10.25 masksharp, 10.21 refined_masksharp. DIS contributes essentially nothing beyond cubic sampling in this example. The first experimental half-residual method also introduced an extra endpoint hitch and should not be used.

The mask-weighted residual method removes that endpoint issue. The final temporal contour difference is 1.03 in baseline and every v2 variant. Mask sharpening adds at most 0.41 gray/color levels of mean absolute temporal change over cubic, with a median increase of 0.10; sampled temporal contact sheets show no abrupt contour-source jump. These statistics cannot prove invisibility during playback.

Visual inspection: cubic and masksharp make outlines more definite, especially hair and shoulders. Faint doubled/soft regions around changing hands and the sloth head remain. This experiment does not eliminate the underlying incompatible drawings or occlusions, and should not be represented as a complete deghosting solution.

Inspect midpoint-contours.jpg, temporal-contours.jpg, comparison.mp4, and the individual native midpoint PNGs. The comparison video currently shows baseline/refined/refined_masksharp; cubic.mp4 and masksharp.mp4 are separate full-frame clips.
