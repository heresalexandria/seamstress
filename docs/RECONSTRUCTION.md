# Layer reconstruction

Use this optional workflow when objects at different depths jump by different amounts at a generation seam. Standard framing and color correction remain the default, and their accepted values stay in place. Reconstruction is applied to original, native-resolution frames first; the existing framing, crop and grade are then applied once.

Reconstruction keeps each foreground drawing on its original source frame. It estimates separate rigid motion for scene layers, recovers obscured background from registered nearby source frames, and eases layer corrections back to the source within a bounded window. Mask tracking uses optical flow as measurement, not a rendered morph between poses. Missing background can optionally come from one generated anchor propagated across the window. Frame count, frame rate and audio timing are unchanged.

## In the app

1. Import the video, review seam markers, and run **Analyze & match** as usual.
2. Select the difficult seam and expand **Layer reconstruction**. **Propose layers** creates a separate candidate. **Attempt automatic repair** also renders it and accepts only the narrow cases that pass conservative automatic checks. Other cases remain candidates.
3. Open **Edit layer masks**. Choose a layer and source frame. Paint Include/Exclude strokes, or use **Keep original detail** for opaque outlines and **Warm glow** for a foreground light. Save to propagate the keyframe locally. Inspect both sides of the generation boundary; propagation does not assume a pose survives that boundary unchanged.
4. Optional: **AI settings → Download local model** installs MobileSAM for point-prompted subject selection. In the mask editor, choose **Local model selection → Place selection points**, mark the subject with Include points and surroundings with Exclude points, then **Run local selection**. The resulting mask still needs review.
5. Use **Recover background · source only** after mask changes. If source frames do not reveal enough scenery, enable AI fill explicitly as described below.
6. Adjust **Reach & motion**, then **Build candidate preview**. Reach is in source frames on each side. Extending beyond the already decoded window requires a new proposal. Review the loop at normal speed and frame-step its edges, outlines and exposed scenery.
7. Check **I reviewed this reconstruction**, then **Accept reconstruction**. Blocking coverage/geometry errors cannot be overridden by this checkbox. New previews and exports include the accepted repair. Editing an accepted mask forks a new candidate; the accepted repair remains active until replaced.

**Reject candidate** discards the pending choice. **Revert reconstruction** removes the selected seam's accepted layer repair while retaining its underlying framing/color correction and repairs at other seams. Old files remain available in project history. Changing the marker or underlying correction at an accepted reconstructed seam requires reverting that reconstruction first.

The **Run workflow** dialog also offers a default-off, source-only reconstruction option. Uncertain candidates are saved for review and excluded from the export; a completed workflow does not mean every reconstruction was accepted.

## CLI

The app and CLI share project files, candidate/acceptance rules and rendering code. Frames are zero-based incoming seam frames; `--timecode` accepts seconds or `HH:MM:SS.mmm` (four-part labels use non-drop frame numbering).

```sh
# Existing workflow, with optional source-only reconstruction attempts.
seamstress process oner.mp4 --work-dir oner.seamstress \
  --reconstruct --output oner-corrected.mp4

# One marked seam, all reconstruction steps through preview / guarded acceptance.
seamstress reconstruct --project oner.seamstress --frame 2888 --stage auto

# Or choose each stage separately.
seamstress reconstruct --project oner.seamstress --frame 2888 \
  --stage propose --reach-frames 24 --motion-strength 0.8 --segmentation classic
seamstress reconstruct --project oner.seamstress --frame 2888 \
  --stage edit --edits masks.json
seamstress reconstruct --project oner.seamstress --frame 2888 --stage background
seamstress reconstruct --project oner.seamstress --frame 2888 --stage render
# Watch candidatePreviewPath from the returned project JSON before accepting.
seamstress reconstruct --project oner.seamstress --frame 2888 --stage accept --reviewed
seamstress export --project oner.seamstress --output oner-reviewed.mp4

# Restore only this seam's underlying correction.
seamstress reconstruct --project oner.seamstress --frame 2888 --stage revert
```

`reconstruct oner.mp4 --frame 2888 --work-dir new-project --base-plan accepted.plan.json` starts a new project from an existing accepted conform plan. Omit `--base-plan` to run normal analysis. `--all-seams --stage auto` attempts all enabled seams and skips seams with accepted reconstruction. `--output new.mp4` adds export, but refuses if the selected reconstruction still needs review. `process --reconstruct` instead exports the accepted plan and retains pending candidates in its project.

An example `masks.json` (coordinates/radii are native source pixels; use a layer ID reported in the candidate):

```json
{
  "strokes": [{
    "frame": 2888,
    "layer_id": "foreground",
    "mode": "include",
    "radius": 8,
    "points": [[640, 320], [652, 326]]
  }]
}
```

The edit stage also accepts `mask_keyframes` (`frame`, `layer_id`, local grayscale PNG `path`) and `layer_matrices` (`layer_id`, `frame`, a 3×3 translation/rotation/uniform-scale matrix). Affine shear and nonuniform scale are refused. The `segment` stage accepts an edits JSON containing `neuralPrompt: {frame, layerId, points: [{x, y, label: 1}]}`; label 0 is an exclusion point. All candidate changes invalidate its prior preview/review.

New proposals cap decoded source RGB at 512 MiB; high-resolution shots may need a smaller reach. The error reports a suitable upper bound. Automatic whole-shot runs retain per-seam abstention notes and continue when a local window or geometry cannot be supported.

## Optional local model

Desktop installers bundle the CPU inference runtime; weights are an explicit download, approximately 44.7 MB. CLI users install the optional runtime first:

```sh
python -m pip install -e '.[segmentation]'
seamstress setup-segmentation-model --download
```

Use `--source-dir downloaded-weights` for offline import. Both files must match the pinned SHA-256 digests. The desktop stores weights in its application data; the CLI uses its OS cache, overridable with `SEAMSTRESS_MODEL_DIR`. `--segmentation auto` uses an already installed model when motion grouping finds a subject, otherwise classical segmentation. `classic` avoids the model; `neural` requires it. No inference uploads video or images.

Weights come from [Acly's MobileSAM ONNX export](https://huggingface.co/Acly/MobileSAM), pinned to revision `0d3b403339b4674a82493d5e97964dd78089ddc8`. The underlying [MobileSAM project](https://github.com/ChaoningZhang/MobileSAM) is Apache-2.0; export packaging is MIT. The engine records provenance and includes the upstream license. Segmentation confidence does not certify a clean matte or an invisible seam.

## Optional OpenAI background fill

In **AI settings**, save your own OpenAI API key. It is encrypted using the OS-backed Electron credential store, owned by the main process, and never returned to the renderer or written into projects. The worker receives it through private stdin only for an explicitly enabled cloud job. An account with access to the configured image model and API billing is required; a ChatGPT subscription alone does not supply this key.

For CLI use, provide `OPENAI_API_KEY` in the process environment through your preferred secret manager. The CLI does not read `.env` automatically or accept the key in command arguments:

```sh
seamstress reconstruct --project oner.seamstress --frame 2888 \
  --stage background --allow-ai --max-ai-requests 1 --quality medium
```

The provider uses OpenAI Images edits (`gpt-image-2`). It uploads a selected source frame and editable mask to synthesize missing **background**, not character poses or tween frames. Local compositing enforces that pixels outside the mask stay unchanged. One anchor is registered into nearby frames; it is not independently regenerated every frame. Generated candidates always require review.

The cap counts request attempts, including rate-limit retries; it is not a dollar budget. Cancellation stops local work but cannot un-send a request already accepted by the provider. Requests and hash-checked cache results are recorded before applying anything. An ambiguous interrupted request leaves a pending receipt and blocks silent resubmission of the same request. Resolve its provider status before deliberately clearing that receipt; simply rerunning is not a free retry. No AI request is made just by saving a key or checking the upload option.

## Reproducible bundles and preservation

A bundle contains `manifest.json`, `READY.json`, hashed source frames, masks, mattes, plates, motion schedules, baseline context and diagnostics. Keep the entire directory together. Import copies and verifies these assets into a new project revision:

```sh
seamstress reconstruct --project oner.seamstress --frame 2888 \
  --stage import --bundle reviewed-bundle/manifest.json
```

Then render/review/accept as usual. Source content, dimensions, cadence and local baseline framing/grade must match. A relocated original source can be supplied through a new project if its bytes match. Bundle paths cannot escape their directory. Imported acceptance is never trusted as automatic project acceptance.

Candidates, accepted revisions and history are separate. Cancellation or a stale job cannot publish over newer settings. Reconstruction windows cannot cross another marker or overlap another accepted reconstruction. Other seam corrections remain unchanged. Default plans take the established rendering path without reconstruction. Export re-encodes the movie, so preservation outside repair windows means identical render values and pre-encoding pixels, not identical MP4 bytes.

The 2:00 IYTYT reference was imported into this format and all 96 reconstructed native frames matched the approved research compositor exactly. That is a replay regression, not a claim that automatic segmentation or generation recreates the authored solution for every shot. Fully hidden scenery, changed anatomy and substantial scene changes can need manual work; QA deliberately abstains when evidence is insufficient.
