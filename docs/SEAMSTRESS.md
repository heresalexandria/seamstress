# Seamstress: refine AI-generated oners

An **oner** is a continuous single shot. In an AI video workflow, you build one by extending the same shot across sequential generations and stitching them together. Small changes in framing, scale, composition, or color can reveal where one generation meets the next.

Seamstress works on that already-stitched video, helping you review and correct those continuity seams. It turns the single file into an editable project with source playback, suggested markers, measured correction plans, previews, and a full-resolution export. The separate generation clips are not required. It runs locally without an API key; the desktop app and CLI share the same processing engine and project files.

## Open the app

On this Mac, the packaged application is built in `app/release/mac-arm64/Seamstress.app`. Open it, then drop your stitched oner into the window. The app prepares a playback copy and automatically scans the whole timeline for candidate generation seams.

1. Select a seam marker to inspect it. Play the seam loop at normal speed; step with the left/right arrows to check the first incoming frame.
2. Drag a marker, edit its time or exact frame in the inspector, add a marker at the playhead, or remove/disable a false positive. Frames are zero-based. Time fields accept seconds, `MM:SS.mmm` or `HH:MM:SS.mmm`.
3. Choose **Analyze & match**, then **Review previews**. Or use **Run workflow**: choose quality and a destination, then run missing analysis/preview stages and export using the current markers. A new automatic detection pass is available through **Find seams**; it replaces the current suggestions.
4. Switch between original, corrected and comparison playback. Review the whole film as well as each seam. The preview is a smaller playback copy; export uses the original resolution.
5. **Export** opens the quality and destination controls. High quality is H.264 CRF 14. Lower CRF makes larger files. Original audio streams are copied, and every source frame keeps its original place and frame rate.

Projects save automatically. **Open** accepts a `project.json`; **Continue last session** reopens the last project. Development projects live in `.app-data/projects`; the packaged app uses its normal macOS application-support folder. The source video stays where you imported it, so keep it available. Projects store its SHA-256 and refuse a changed source.

Moving, enabling, disabling or deleting a seam invalidates the correction plan and corrected previews. Source playback and thumbnails remain usable. Analysis and detection reports are saved beside the project for inspection.

## One command

After the Python/FFmpeg installation in the main README:

```sh
seamstress process "/path/to/video.mp4" \
  --work-dir "/path/to/video.seamstress" \
  --output "/path/to/video-corrected.mp4"
```

This validates/imports the source, makes playback media, detects seams, calibrates framing and color corrections, renders whole-video and individual seam previews, exports at native resolution, and checks frame count, dimensions, fractional frame rate, stream start times and audio bitstreams. It writes progress to stderr and the final project JSON to stdout.

If omitted, the project directory is the input filename with `.seamstress` in place of its extension; the export goes into that project's artifacts. Existing project/output paths are refused. To continue an existing project:

```sh
seamstress resume --project "/path/to/video.seamstress" \
  --output "/path/to/another-corrected.mp4"
```

Resume reuses the current markers and any current analysis/preview artifacts. It does not replace manually reviewed seam positions. Use a new export filename.

## Run stages separately

```sh
# Import and detect. Hints are optional; every frame boundary is scanned.
seamstress detect "/path/to/video.mp4" --work-dir "/path/to/video.seamstress"

# Rerun detection on an existing project. This replaces its marker list.
seamstress detect --project "/path/to/video.seamstress" --sensitivity 0.6

# Replace markers with exact times OR exact incoming frame numbers.
seamstress mark --project "/path/to/video.seamstress" --timecodes "00:08,00:15.250,01:02"
# seamstress mark --project "/path/to/video.seamstress" --seams "192,366,1488"

# Measure only the enabled joins and save a source-specific plan.
seamstress calibrate --project "/path/to/video.seamstress"

# Make the whole-film playback copy and short seam previews.
seamstress preview --project "/path/to/video.seamstress" --preview-width 640

# Render at native resolution with original compressed audio.
seamstress export --project "/path/to/video.seamstress" \
  --output "/path/to/video-corrected.mp4" --crf 14

# Print all markers, metadata and artifact paths.
seamstress inspect --project "/path/to/video.seamstress"
```

You can bypass detection in the one-command workflow with `--timecodes "8,15,30"` or `--seams "192,360,720"`. Numbers in `--seams` always refer to the first incoming frame after a cut. Seconds/timecodes are rounded to the nearest source frame. CLI four-part `HH:MM:SS:FF` notation uses **non-drop** nominal frame counts; semicolon/drop-frame labels are not supported.

Detection defaults to soft interval hints of 10, 15 and 30 seconds. Set `--intervals "12,20"` to use other hints, or `--intervals ""` to disable hints. A seam at eight seconds can still be detected. Raising `--sensitivity` toward 1 produces more candidates; lowering it toward 0 is more selective. Suggestions and confidence values are heuristic, not probabilities or proof of a generation boundary.

For automation use `.venv/bin/python -m seamstress` in place of `seamstress`; exit code 0 means the requested stage completed. Ctrl-C cancels processing. Completed artifacts stay on disk; an incomplete stage is not published as a current project artifact. A completed export is not a certification of invisible seams.

## What the correction does

The detector scans an aspect-preserving low-resolution copy across the entire timeline. It looks for unusual changes in registered appearance, geometry, color, sharpness and timing relative to the surrounding motion. It then examines candidate neighborhoods more carefully. Common clip intervals are additional evidence, never instructions to place a seam where no change exists.

Calibration compares the source frames around each marked boundary, matches visual features, estimates a small global camera transform, and checks whether the match has enough reliable evidence across the image. Supported framing changes are distributed across both sides of the join and eased back to the original framing. A small constant viewing crop keeps exposed edges out of view. Camera-rate reconciliation is applied only where supported by the observations.

Color calibration first fits protected tone curves, then fits a bounded residual model that can adjust colors differently in different image regions. It checks observations withheld from fitting and adjacent frame pairs, rejecting unsupported corrections. Color changes ease in and out smoothly around the join. Black/white protection and gamut bounds limit unintended shifts.

Rendering uses exactly one original source drawing for each output frame. It changes that drawing's global framing and pointwise color. It does not crossfade, blend different frames, morph objects, synthesize tween frames or replace poses. Optical flow is used to measure matching color observations, not to deform the rendered footage.

Seams with insufficient evidence, large geometric changes, parallax or unrelated scenes can be left unresolved and listed in the report. Those are deliberate review points: a flat composite may not contain the image information needed to reconstruct a physically continuous shot.

## Current input/output scope

The current tested pipeline is for constant-frame-rate, 8-bit SDR video with even dimensions of at least 32 pixels, square pixels, and video timestamps starting at zero. Common MP4/MOV/MKV/WebM containers are accepted when the bundled decoder supports their codec. Rotation metadata is applied to display geometry. Variable/discontinuous frame timing, nonzero video start offsets, anamorphic pixels, HDR/high-bit-depth and unsupported geometry produce a clear import error rather than silent timing or HDR conversion. Multiple original audio streams are retained in final exports; if an audio codec cannot be muxed into MP4, export fails explicitly.

Native export is H.264, 8-bit 4:2:0 MP4; it is high quality, not lossless restoration. Playback proxies use AAC audio for browser compatibility. The generic defaults limit calibration to 128 marked seams and 250,000 frames. Long videos need time and disk space for the source proxy, previews and export; detection uses disk-backed scan data and the renderer streams frames.

Automatic calibration is new and conservative. It does not reproduce the bespoke IYTYT measurements byte-for-byte; the saved, reviewed IYTYT recipes remain available unchanged in their reproduction guides. Inspect a new video's results before relying on its export.

## Develop, test and package

Install the base Python dependencies first. Then, from the repository root:

```sh
cd app
npm ci
npm run dev
```

`npm run build` type-checks and builds the renderer. `npm start` opens that production renderer in Electron. All fonts and fabric artwork are local assets/CSS; processing and interface rendering need no API key or internet after installation.

```sh
# From repository root:
OPENBLAS_NUM_THREADS=2 .venv/bin/python -m unittest discover -s tests -q
cd app
npm test
npm run test:e2e
```

To build the standalone macOS app, install PyInstaller in the same Python environment, then:

```sh
# From repository root:
uv pip install --python .venv/bin/python pyinstaller==6.22.3
cd app
npm run package   # .app folder
npm run dist      # DMG and ZIP
```

The backend build bundles its Python runtime and FFmpeg/ffprobe dependencies, so the resulting app does not depend on this checkout or Homebrew. Builds are local and unsigned/not notarized; distributing a polished installer externally additionally requires your Apple signing/notarization identity. The packaged runtime includes dependency notices. See [validation notes](VALIDATION-0.2.md) for the tested cases and reproducible checks.

`analyze`, `repair`, `run` and `bridge` are retained historical experiments. The desktop app and the new `process` workflow use **source conform**, the approach developed after the morphing results were rejected.

Signed CI builds and automatic release publication are documented in [the release guide](RELEASES.md) and [signing setup](MACOS-SIGNING.md). The unsigned local packaging commands above are for development only.
