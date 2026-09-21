# Framing continuity in the generalized pipeline

This review compares the generalized IYTYT export against both the original
source and the previously accepted color-refined recipe. The original source,
accepted recipes and accepted movies remain unchanged. All examples below refer
to original zero-based source frames at 24000/1001 fps.

## Framing at 1:00 and 1:15

The previous automatic export omitted geometry at frames 1444 (60.227 seconds)
and 1805 (75.284 seconds), although the accepted custom recipe corrected both.
The seam matches were strong; outgoing parallax made a single camera-rate
estimate unreliable. The omitted corrections left vertical scale changes of
approximately 2.5% and 1.75%.

At 1:00, the new fallback identifies a reprojected repeated endpoint using local
feature trajectories, spatial holdouts and independent temporal intervals.
It also requires broad stationary anchors on both sides: an animation hold
alone cannot establish a camera hold. Every intervening frame is checked to
avoid aliasing alternating camera motion into a false stationary reference.
This source has 123 qualifying anchors. The resulting correction restores the
measured global recrop with camera-rate easing disabled.

At 1:15, stationary evidence is too spatially concentrated to authorize the
same automatic inference. Attempts to infer a partial correction from averaged
affine motion produced false repairs on legitimate camera pans and were
excluded from the engine. Automatic calibration still flags this seam.

The [reviewed reproduction recipe](REPRODUCE-FRAMING-REVIEW.md) restores this
specific seam using its previously accepted framing measurements. This is an
explicit source-specific review decision, recorded in the calibration and plan;
it is not a hidden automatic exception. Color is refitted and validated in that
restored geometry. The generalized pipeline contains no IYTYT frame numbers.

## Camera cadence at 1:30

Frame 2166 (90.340 seconds) has a strong whole-frame seam match. The regression
was in the camera-rate estimate, not the geometric registration. Two-frame
motion estimates alternate between slower and faster steps in a four-frame
drawing cadence. Taking the median of three intervals favors one phase.

The previous automatic estimate was a log-scale rate of −0.017309 per frame and
vertical movement of 9.894 native pixels per frame. That produced stronger zoom
and vertical compensation than the accepted recipe's −0.012881 and 8.051.

The cadence check examines six two-frame intervals within the same generation.
It requires a repeatable alternating component, rejects material camera
acceleration, and checks five independently registered four-frame intervals
against the composed shorter measurements. Only then does it substitute a
slope measured over the complete trajectory. For this seam, the resulting
rate is −0.013108 and 8.170 pixels per frame. The ordinary estimator remains
unchanged where this additional evidence does not pass.

This changes a camera estimate. It does not change frame order, frame count,
animation timing or the drawings themselves.

## Why 2:00 remains a geometry exclusion

Frame 2888 (120.454 seconds) is unresolved in both the accepted custom recipe and
the generalized export. A fresh audit tested whether there was a smaller common
scale correction hiding beneath the incompatible layer movement.

Raw horizontal alignment suggests approximately 0.7–0.8% contraction across
several regions, but ordinary pre-seam movement makes that interpretation
ambiguous: the background is already zooming while the characters are nearly
stationary. Vertical scale estimates disagree substantially between depths.

Even applying the raw horizontal correction as a counterfactual barely changes
the full displacement: approximately 14.17 to 14.06 pixels on the Empire State
Building and 21.83 to 21.64 pixels in the near city. It cannot repair the visible
vertical jump. Matching local patches also does not establish a separate blur
cliff large enough to justify softening the frame.

No automatic geometry or blur correction is added there. Color correction
remains available under its existing independent checks. Repairing the depth
discontinuity would require separately handled scene layers and reliable
occlusion reconstruction; the current flat-video renderer does not provide
that reconstruction.

## Using the changes

The app now exposes [per-seam controls](SEAM-CONTROLS.md) for these automatic checks and for importing the reviewed 1:15 framing. Custom geometry remains a recorded review decision; the automatic evidence requirements are unchanged.

The app's **Analyze** action and the CLI's `calibrate` / `process` stages share
the same implementation. Reanalyze an existing project, then regenerate its
previews and export to a new filename. Old exports and frozen plans do not
change automatically. No example-specific frame numbers are used by the
camera estimator.

Metrics and successful integrity checks do not establish imperceptible seams.
Review the new movie at normal speed, including the movement before and after
each boundary.
