# Choose how each seam is corrected

Select a join in the timeline to open **This seam’s correction**. Settings belong
to that seam and are saved in the project. Existing projects keep the automatic
behavior unless you change a setting.

1. Choose the framing and color treatments for the selected join.
2. Click **Apply seam settings**. **Reset changes** discards only unapplied edits.
3. Run **Analyze & match**, then **Review previews**. Settings changes invalidate
   the previous correction plan and its previews; old exported files remain on disk.
4. Review both the seam loop and the full shot before exporting to a new filename.

The **Applied correction** readout describes the last analysis: which framing
method was used, whether cadence adjustment and camera-rate easing were applied,
and the color treatment. Its review notes explain skipped or partial corrections.
**Detection readout** is separate: a confidently located boundary is not evidence
that every kind of correction is safe there.

## Framing and color

| Control | Behavior |
| --- | --- |
| Framing · Automatic | Measure the global framing change and apply supported corrections. |
| Framing · Off | Leave this seam’s geometry uncorrected while retaining its color settings. |
| Framing · Custom | Use exact reviewed or edited framing measurements for this seam. |
| Color · Automatic | Fit protected tone curves and, when independently supported, local color correction. |
| Color · Tone only | Allow protected global tone curves; disable the spatial residual color model. |
| Color · Off | Apply no color correction at this seam. |

Turning one seam’s framing off does not remove the constant viewing crop needed
to cover source edges for other active seams. Similarly, each seam’s decision is
local even though the complete plan is rebuilt and validated together.

Expand **Automatic checks** to enable or disable individual framing tactics:

- **Partial framing recovery** looks for a common framing change when scene
  layers have different motion. It keeps unsupported directions unchanged.
- **Repeated endpoint recovery** recognizes a reframed repeated endpoint. It
  requires broad stationary evidence across intervening frames, not just a
  locally held drawing. This is the automatic tactic used for the reviewed
  IYTYT 1:00 seam.
- **Animation cadence check** measures camera speed over complete drawing cycles
  when short motion samples alternate. This corrected excess camera compensation
  at the IYTYT 1:30 seam without retiming animation.

All three are enabled by default and retain their evidence checks. Disabling a
check removes that tactic; it does not force another rejected correction. These
controls apply to **Automatic** framing and are inactive under **Custom** or **Off**.

**Camera-rate easing** is enabled by default. It reconciles supported differences
in camera speed around the boundary. Recovered endpoint/partial edits can still
leave easing unapplied because those measurements already separate the framing
change from ordinary motion. For custom framing, the checkbox chooses whether
to reconcile the supplied before/after rates.

## Keep or import reviewed framing

**Use analyzed framing** copies the latest accepted framing measurements into a
custom draft. Click **Apply seam settings** to keep them. The button is unavailable
when analysis has not supplied accepted measurements. This is useful when you
want to retain a framing choice while adjusting its color treatment or camera
rates.

**Import reviewed framing** opens a calibration JSON file and reads only the
selected seam’s geometry and camera-rate choice. A calibration is the
`source_conform_calibration` file produced by analysis; a rendering plan or an
analysis report is not interchangeable with it.

The import requires the same source SHA-256, dimensions, frame rate and frame
count, plus the exact incoming source-frame index. A seam excluded from the
calibration’s geometry cannot be imported as an accepted correction. A file for
another video or another boundary is refused. Imported matrices, rates and
provenance are copied into the project, so reopening it does not depend on the
calibration file remaining at its original path.

Importing saves the custom framing immediately and clears stale analysis. Run
**Analyze & match** to refit and independently validate color in the chosen
geometry; importing does not transplant old color curves. You can temporarily
choose **Automatic** or **Off** without losing saved custom measurements, then
return to **Custom**. Moving the seam to a different frame clears those bound
measurements. A custom marker returns to automatic framing; an off marker stays off. Rerunning **Find
seams** retains manual, disabled and customized markers at their exact source
frames while refreshing the remaining automatic suggestions.

For the reviewed IYTYT 1:15 correction, import the original `IYTYT.mp4`, select
frame **1805**, and choose
[`plans/IYTYT-framing-reviewed-calibration.json`](../plans/IYTYT-framing-reviewed-calibration.json).
The app restores that seam’s previously accepted affine transform and camera
rates. It makes an explicit custom decision; automatic endpoint inference still
has insufficient evidence at this boundary. See the
[framing findings](FRAMING-REVIEW.md) and
[complete reproduction recipe](REPRODUCE-FRAMING-REVIEW.md).

## Edit custom measurements

Choose **Custom**, then expand **Edit custom measurements**. It starts from your
saved custom framing, otherwise accepted analyzed measurements, otherwise an
identity transform and zero camera rates. The six affine values map the incoming
picture onto the outgoing picture in native source pixels:

```text
x′ = a·x + b·y + tx
y′ = c·x + d·y + ty
```

The scales `a = d = 1`, shears `b = c = 0`, and shifts `tx = ty = 0` form the
identity. The two scale values can differ, allowing the slight anisotropic
resizing used in the reviewed recipe. The renderer distributes the correction
across both sides of the join and gradually returns to the original framing;
these values do not simply move every later frame by the entered amount.

Each before/after camera rate has four values: log scale per frame, rotation in
radians per frame, and horizontal/vertical movement in pixels per frame. The
before rate still helps separate ordinary camera movement from the framing jump
when easing is off. Zero rates are appropriate only for a stationary camera or
a measured edit that already excludes ordinary motion.

Edited values are recorded as **Edited in Seamstress**. The backend checks finite
values, transform bounds, source identity and coverage. Custom framing is your
review decision, not an automatic claim that the scene supports it. Change a
small amount, apply, analyze, and compare playback before making another change.

## What these controls cannot reconstruct

The IYTYT 2:00 boundary moves foreground and background by different amounts.
A single affine transform cannot align both depth layers. Turning off evidence
checks or choosing custom geometry cannot supply missing occluded image detail.
Color may still be corrected independently, but the depth jump remains a review
point. The app does not morph, blend source frames, synthesize replacements or
reconstruct independent scene layers, and no setting guarantees imperceptible
seams.
