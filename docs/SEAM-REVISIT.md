# Revisiting the two remaining IYTYT joins

This documents the v0.2.3 review. The subsequent [framing review](FRAMING-REVIEW.md)
addresses automatic-export regressions at 1:00, 1:15 and 1:30 and rechecks the
remaining 2:00 limitation.

This review compares the original stitched video with the saved eight-join and
color-refined renders. It also checks the generalized engine separately: its
conservative automatic measurements do not reproduce every decision in those
manually reviewed recipes. The original videos and saved recipes are unchanged.

The source is 1280 × 720 at 24000/1001 fps. The joins commonly called “30 seconds”
and “120 seconds” start at original frames **722 (30.113 s)** and
**2888 (120.454 s)**. Measurements below use freshly decoded video; pixel
statistics support the diagnosis but do not certify an invisible transition.

## 30 seconds: the background changes speed

In the accepted color-refined render, the background's mean horizontal movement
falls from about **3.69 to 0.60 source pixels per frame**. The car remains nearly
stationary: approximately −0.066 to −0.022 pixels per frame. The outgoing
background also has an uneven held-frame cadence; the incoming background moves
more uniformly.

The remaining global seam alignment is already close to identity: scale 0.99994
and translation about −0.09, −0.31 pixels. Stronger whole-frame alignment cannot
reconcile the background's speed with the car's speed. Moving the whole picture
would also move the car. Some pose and drawing details change between the
original generated images.

A complete correction would need to isolate the background through time, retime
it independently, and reconstruct areas revealed behind moving subjects. That
requires reliable temporal masks and occlusion handling. It is not established
as a safe automatic correction from this flat composite. Crossfading or morphing
the poses would repeat the previously rejected artifacts.

There was a separate limitation in the generalized engine: its outgoing camera
rate is unreliable here because the car and scenery have different motion.
It therefore skipped the geometric correction that the reviewed custom recipe
applies. A strong match between two seam frames alone is insufficient to separate
a generation's recrop from actual camera motion.

The revised fallback measures several broad motion groups across independent
source intervals. It corrects a framing component only when those groups agree
on motion in that direction, and independent cross-cut pairs support the same
edit after accounting for ordinary movement. At this join, their disagreement is
almost entirely horizontal, allowing partial vertical crop/scale correction.
The uncertain horizontal component stays untouched and camera-rate easing stays
off. This remains one global affine correction, with no object masks or local
shape warping. The report and app explicitly flag the remaining framing
uncertainty. It improves the generalized result; it does not remove the
background slowdown in the accepted custom render.

## 120 seconds: different depths jump differently

Across the accepted color-refined seam, feature groups have these approximate
vertical displacements (positive means downward):

| Region | Seam displacement |
| --- | ---: |
| Foreground subjects | +2.36 pixels |
| Empire State Building | −14.59 pixels |
| Near city buildings | −22.59 pixels |

The city had been moving downward before the seam, then jumps upward around the
relatively stable subjects. One global transform cannot align all these depths.
The old layered and generated-background experiments produced halos, duplicate
contours or depth drift; they are not a supported repair. Checking earlier
outgoing frames also found no reliable duplicate overlap to trim. Dropping two
frames leaves a substantial depth mismatch and changes timing.

The generalized engine can still correct color without moving those layers.
Previously, it required a globally consistent affine match before trying any
color correction. This confused two questions: whether the pictures depict the
same scene, and whether one camera transform explains all their motion.

The revised analysis permits color-only fitting when independent, distributed
feature correspondences demonstrate scene continuity. Its local motion models
are evidence for matching surfaces; they are never used to warp output. Existing
protected tone and bounded local-color checks still have to pass, including
held-out observations, other cross-cut pairs, minority palettes and color
mapping safety checks. Unsafe geometry remains excluded and reported.

For this seam, the full automatic run reduces held-out matched-interior RGB error
from **4.83 to 1.34 levels** and regional bias from **4.42 to 0.47**. This recovers
color treatment comparable to the accepted custom result. It does not fix the
remaining architectural jump in that accepted result.

## Regression against the other joins

A fresh baseline analysis exactly reproduced the previous generalized plan.
Running the revised analysis on all nine original markers changes geometry only
around frame 722, and color only around frames 722 and 2888. The constant viewing
crop remains exactly **5.558980457%**. The other seven joins retain identical
matrices and color models; rendered samples on both sides of those joins are
pixel-identical before encoding. Frame count remains 3347 at 24000/1001 fps.

The automatically inferred plan now includes five full framing corrections,
one explicitly partial framing correction, nine protected tone curves and eight
local color corrections. Frames 1444, 1805 and 2888 still exclude geometry.
The reviewed eight-join and color-refined recipes remain available unchanged;
this automatic plan is not a replacement for their manual review decisions.

## Running the updated analysis

The normal app's **Analyze** action and the existing CLI stages use the same
calibrator. No special frame numbers or scene masks are built into the engine.
Reanalyze an existing project to get the new plan; previously generated exports
and saved custom recipes do not change automatically.

```sh
seamstress calibrate --project /path/to/video.seamstress
seamstress preview --project /path/to/video.seamstress
seamstress export --project /path/to/video.seamstress --output corrected-new.mp4
```

Review the new preview at normal speed. The analysis report records geometry
exclusions separately from the evidence and validation for color correction.
A color-only correction does not mean the join's movement has been repaired.
