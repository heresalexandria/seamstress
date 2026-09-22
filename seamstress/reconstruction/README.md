# Native layer reconstruction

This is an opt-in repair provider for a selected difficult seam. It runs before
the existing conform plan's viewing crop and color corrections. It does not
replace or silently modify the default correction pipeline.

```python
from seamstress.reconstruction import propose, edit_bundle, render_bundle, summary

manifest = propose("shot.mp4", 2888, "work/proposal-1",
    baseline_plan="work/accepted.plan.json",
    options={"reachFrames": 16, "motionStrength": 0.7, "segmentation": "auto"})
print(summary(manifest))
revised = edit_bundle(manifest, "work/proposal-2", {
    "strokes": [{"frame": 2888, "layer_id": "foreground", "mode": "protect",
                 "points": [[610, 315], [615, 315]], "radius": 2}]
})
native_preview = render_bundle(revised, "work/native-preview")
```

Output directories must be new. A bundle is published only after `READY.json`
contains its final manifest hash. Interrupted jobs have no valid READY marker;
retry in a new directory. Every source, baseline, mask, plate and matte is hash
bound. Import can relocate the source only when its content fingerprint matches.
The frozen baseline is included as an asset. A different baseline is accepted
only if its viewing geometry and color context within this support are equal.

## Coordinates and API

All frame indices are zero-based **original source frames**. `frame` is the
incoming frame at the seam. `support.start` is inclusive; `support.end` is
exclusive. Coordinates and RGB arrays use the original decoded width and height,
not the app preview size. Source frame count and rational frame rate are retained.
SDR native odd/even dimensions are supported; HDR is explicitly rejected.
Proposal/edit/source-recovery windows are limited to 512 MiB of decoded RGB
before allocating frames. Oversized requests fail with the maximum suggested
reach at that source resolution. Authored import verification and native rendering
stream source frames instead of decoding the entire source into memory.

- `propose(source, frame, output_dir, *, baseline_plan=None, options=None,
  progress=None, cancelled=None) -> Path`
- `edit_bundle(manifest, output_dir, edits, *, progress=None, cancelled=None) -> Path`
- `prepare_backgrounds(manifest, output_dir, *, provider=None, progress=None,
  cancelled=None) -> Path`
- `import_bundle(manifest, output_dir, *, source, baseline_plan=None, ...) -> Path`
- `import_authored(source, frame, output_dir, *, support, frames, layers,
  baseline_plan=None, compositor=None, provenance=None, review_approved=False,
  progress=None, cancelled=None) -> Path`
- `load_bundle(manifest, *, verify=True) -> dict`
- `summary(manifest) -> dict`
- `apply_frame(manifest_or_loaded_bundle, n, original_rgb) -> uint8 RGB`
- `render_bundle(manifest, output_dir, *, progress=None, cancelled=None) -> Path`

Progress callbacks receive `{fraction, progress, message}`; cancellation callbacks
return a boolean. `apply_frame` returns exact original RGB outside support and
on identity frames. The native FFV1/Matroska render is lossless and includes the
corresponding source audio interval as FLAC. The app's candidate preview adds the
accepted conform view/color and creates its usual playback encode.

## Proposals and edits

The initial proposal groups mutually matched SIFT features into coherent
similarity motions. A compact independently moving group provides a foreground
seed; background groups provide proposed depth masks. The classical path uses
seeded GrabCut and local flow for **mask transport only**. `segmentation='auto'`
may use an already installed optional local model for the seed; `classic` always
uses CV; `neural` requires the installed model. No model is downloaded implicitly.
Recorded model provenance distinguishes neural seeding from subsequent CV tracking.

Expected adjacent-frame camera velocity is removed from the measured cut
discrepancy. The remaining correction is residualized against accepted baseline
geometry (`inverse(B_right) @ B_left @ velocity @ measured`) so an already
aligned seam is not corrected twice. The common viewing crop cancels from this
relation. A non-similarity residual from an affine baseline is refused; the
rigid-layer invariant is retained. A compact quintic gate modifies correction matrices, using
identical-frame holds as cadence evidence. It never blends original actor poses.
The first implementation keeps outgoing frames and applies the proposal on the
incoming side. Authored imports can carry independently designed bilateral paths.

Editable fields:

- `mask_keyframes: [{frame, layer_id, path}]`: native grayscale PNG, black to white.
- `strokes: [{frame, layer_id, mode, radius, points}]`: native coordinates and
  include/exclude/protect/emission modes. Protect creates an opaque source core.
  Emission proposes warm additive light and requires review; it is not a general
  transparency estimator for smoke, glass, hair or motion blur.
- `layer_matrices: [{frame, layer_id, matrix}]`: explicit 3×3 source-to-output
  similarities. Perspective, shear, reflection and nonuniform scale are rejected.
- `motionStrength: 0..1`, `reachFrames`: adjust an automatic proposal. Reach can
  shrink within the existing decoded interval; propose again to extend it.
- `review_approved: true`: records user review but cannot bypass blocking errors.

Local mask edits propagate to the nearest edited keyframe on the same side of
the generation boundary. Previously authored keyframes remain fixed references
when later edits add another frame. Corrected masks rebuild source-first donor recovery.
Low-confidence propagation remains visible in QA. Every revision is immutable.

## Native bundle format

`manifest.json` uses schema `seamstress.reconstruction/v1`:

```json
{
  "source": {"path": "absolute original", "sha256": "...", "width": 1280,
    "height": 720, "fps_fraction": "24000/1001", "frame_count": 3347},
  "baseline": {"path": "original plan", "sha256": "...", "asset": "baseline.plan.json"},
  "frame": 2888, "support": {"start": 2872, "end": 2916},
  "assets": {"relative/path.png": "sha256..."},
  "layers": [{"id": "background-0", "kind": "background", "order": 0,
    "mask_by_frame": {"2888": "masks/background-0-2888.png"},
    "matrices": {"2888": [[1,0,0],[0,1,0],[0,0,1]]}}],
  "frames": {"2888": {"source": "source/frame-2888.png",
    "plate": "plates/plate-2888.png", "matte": "mattes/matte-2888.npz",
    "source_rgb_sha256": "...", "foreground_source_frame": 2888}},
  "qa": {"status": "needs-review", "auto_eligible": false,
    "blocking_errors": [], "issues": [], "metrics": {}}
}
```

The example abbreviates arrays: every supported frame must have all assets and
matrices, at least one background layer, and exactly one foreground group. The
foreground NPZ contains native float arrays `alpha[H,W]` in 0..1,
`premultiplied[H,W,3]` and `emission[H,W,3]` in 0..255, plus the exact
`background_sha256`. Composition is `warp(P) + (1-warp(alpha))*B + warp(emission)`.
An optional per-frame `residual_strength` retains the same-frame reconstruction
residual near identity; it never references another actor drawing.

`import_authored` accepts absolute input file paths using these same row/layer
structures and copies them into the new bundle. Source PNGs must match decoded
original frames exactly. It preserves authored alpha/P/emission and explicit
motion without rerunning segmentation. Gap interpolation, if used, is explicit
in `compositor` and its affected pixel counts/radii are reported.

## Source-first backgrounds and optional generation

Background recovery registers neighboring original frames while excluding both
target and donor actors. It copies only visible donor pixels. Unrecovered pixels
use a labeled classical-inpaint placeholder and are exposed as `unknown` masks.
The current single-similarity donor registration can abstain on parallax; this
does not establish correct hidden geometry for arbitrary scenes.

`prepare_backgrounds` calls `provider(source_rgb, editable_mask, source_frame)`
once for the frame with the most remaining unknown pixels. The provider returns
`{image_path, request_id?, cache_hit?}` and must preserve source RGB outside the
mask. The engine registers this one background anchor into the remaining source
frames and changes only unknown areas. The workflow owns upload permission,
BYOK credentials, persistent request budgets and provider provenance. Generated
background always requires review; it is plausible synthesis, not source recovery.
Omitting the provider reruns source donor recovery with the current masks and
creates an immutable source-only revision without any network or model request.

## Acceptance gates

Malformed assets, source mismatches, non-rigid matrices, bad coverage and broad
disocclusion gaps cannot be waived by user review. Automatic eligibility is
narrow: one measured small motion, many mutually matched spatially distributed
features, tight geometric and photometric residuals, verified adjacent velocity,
no subject segmentation, no hidden reconstruction, and complete output coverage.
Changing baseline geometry or color within the window requires final-context
review even when native-frame registration is confident.
Articulated actors, inferred depth masks, generated content and uncertain tracking
require inspection. An automatic no-op can qualify when the source already
satisfies these measurements; low-texture identity fallback cannot.

These gates do not prove perceptual perfection. Inspect the whole active window,
especially thin ink, hands, occlusion edges, light effects and support ramps. The
successful 2:00 authored import is pixel-identical to its reviewed 96-frame native
reference; that equivalence does not make automatic segmentation equally reliable.
