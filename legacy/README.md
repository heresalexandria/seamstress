# Seamstress

Seamstress makes the cut between two separately generated clips of the same
continuous shot feel seamless. It works in six stages: ingest/conform,
temporal-overlap detection and trim, pooled motion-compensated global color
matching, conditional spatial (ECC) alignment, motion-compensated residual
seam propagation, and final encode/verification.

The tool has one CLI entrypoint:

```sh
./seamstress.py PREVIOUS_CLIP NEXT_CLIP [options]
```

`PREVIOUS_CLIP` is the earlier clip in the timeline. `NEXT_CLIP` is the later
clip in the timeline.

## Requirements

Install the Python dependencies:

```sh
python -m pip install -r requirements.txt
```

Install `ffmpeg` and `ffprobe` separately and make sure both commands are on
`PATH`.

The two clips do not need matching frame rates or resolutions. Nominal fps
differences are conformed (the altered clip is retimestamped to the
unaltered clip's exact rational fps) with a warning; resolution mismatches
are lanczos-scaled to the unaltered clip's size with a warning, and only
refused if the aspect ratio differs by more than 0.5%. The corrected output
is encoded with `libx264`, `yuv420p`, and copied audio from the clip being
altered (trimmed to stay in sync) when audio is present.

## Quick Start

Adjust the next clip so its first frame matches the previous clip's final
frame, auto-detecting any duplicated head frames and trimming them:

```sh
./seamstress.py previous.mp4 next.mp4
```

This writes:

```text
next_seamless.mp4
next_seamless_report.json
next_seamless_diagnostics/
next_seamless_joined_preview.mp4
```

Preserve the next clip and adjust the previous clip's ending instead:

```sh
./seamstress.py previous.mp4 next.mp4 --alter previous
```

This writes:

```text
previous_seamless.mp4
previous_seamless_report.json
previous_seamless_diagnostics/
previous_seamless_joined_preview.mp4
```

Use explicit output paths when integrating with a larger pipeline:

```sh
./seamstress.py previous.mp4 next.mp4 \
  --output corrected-next.mp4 \
  --report corrected-next-report.json \
  --diagnostics-dir corrected-next-diagnostics \
  --joined corrected-next-joined-preview.mp4
```

## How It Works

1. **Ingest & conform** - probes both clips, reconciles fps/resolution
   mismatches, and carries color tags (bt709 etc.) through to the output.
2. **Temporal alignment** - cross-correlates high-pass features of clip1's
   tail against clip2's head to detect duplicated/overlapping frames and
   compute the correct trim (or falls back to a butt joint if no overlap is
   found).
3. **Global color match** - pools motion-compensated pixel correspondences
   from every overlapping frame pair (or a few flow-aligned frames when
   there is no overlap) and solves one constant affine + per-channel curve
   correction for the whole altered clip (never a per-frame correction).
4. **Spatial alignment** - conditionally applies a subpixel ECC affine warp
   when the estimated displacement is meaningful but not extreme.
5. **Seam finishing** - if a residual mismatch remains at the new boundary,
   propagates it forward (or backward, for `--alter previous`) over a short
   transition, warping the high-frequency detail band with the clip's own
   optical flow so it doesn't ghost under motion.
6. **Verification & outputs** - encodes the corrected clip, a joined preview
   (concatenation of both final clips, re-encoded once) for timeline QC, a
   seam z-score metric, diagnostics PNGs, and a JSON report.

## Choosing What To Alter

Use the default `--alter next` when the first clip is already approved and
the later clip can be re-rendered; Seamstress re-renders `next` to match
`previous`'s ending. Use `--alter previous` when the later clip must remain
untouched; Seamstress re-renders `previous`'s ending to match `next`'s start
instead. Temporal-overlap trimming is always computed from the physical
`previous`/`next` pair regardless of which clip is altered.

## CLI Reference

```text
usage: seamstress.py [-h] [--alter {next,previous}] [-o OUTPUT]
                      [--report REPORT] [--diagnostics-dir DIAGNOSTICS_DIR]
                      [--no-diagnostics] [--mask-margin MASK_MARGIN]
                      [--trim-percentile TRIM_PERCENTILE] [--trim TRIM]
                      [--fps-policy {conform,strict,resample}]
                      [--color-model {affine,affine+curves}]
                      [--warp {auto,always,never}]
                      [--transition-frames TRANSITION_FRAMES]
                      [--joined JOINED] [--no-joined] [--crf CRF]
                      [--preset PRESET]
                      PREVIOUS_CLIP NEXT_CLIP
```

Arguments:

- `PREVIOUS_CLIP`: earlier clip in the timeline.
- `NEXT_CLIP`: later clip in the timeline.

Options:

- `--alter {next,previous}`: choose which clip gets re-rendered. Default:
  `next`.
- `-o, --output PATH`: corrected clip path. Default: the altered input clip
  name with `_seamless` before the extension.
- `--report PATH`: JSON report path. Default: output name with
  `_report.json`.
- `--diagnostics-dir PATH`: diagnostic PNG directory. Default: output name
  with `_diagnostics`.
- `--no-diagnostics`: skip diagnostic PNG output.
- `--mask-margin PIXELS`: ignore this many pixels at each frame edge during
  alignment, color solving, and flow confidence checks. Default: `48`.
- `--trim-percentile VALUE`: residual percentile kept while solving the
  robust affine color matrix. Lower values reject more outliers. Default:
  `75.0`.
- `--trim auto|N`: `auto` detects head-overlap and trims duplicate frames
  from `next`; an integer forces the trim (`0` disables detection/trimming).
  Default: `auto`.
- `--fps-policy {conform,strict,resample}`: how to reconcile differing
  nominal frame rates. `conform` (default) retimestamps the altered clip to
  the unaltered clip's exact rational fps and warns if nominal rates differ
  by more than 1%. `strict` errors on any mismatch. `resample` is not
  implemented.
- `--color-model {affine,affine+curves}`: `affine` solves a single robust
  3x4 RGB affine transform. `affine+curves` (default) adds per-channel
  monotonic residual LUTs fit on top of the affine solve.
- `--warp {auto,always,never}`: whether to apply the Stage 4 ECC spatial
  warp. `auto` (default) applies it only when the ECC score is >= 0.5 and
  the implied corner displacement is between 0.2px and ~12px.
- `--transition-frames N`: frames used for the Stage 5 motion-compensated
  residual-propagation transition. Use `0` to disable. Default: `6`.
- `--joined PATH`: path for the joined timeline QC preview (concatenation of
  the final previous + next clips, re-encoded once). Default: `<output
  stem>_joined_preview.mp4`.
- `--no-joined`: do not write the joined preview.
- `--crf N`: `libx264` quality. Lower is higher quality and larger output.
  Default: `10`.
- `--preset NAME`: `libx264` speed/compression preset. Default: `slow`.
- `-h, --help`: show the full CLI help.

## Outputs

- The corrected clip (`--output`).
- A joined preview (`--joined`), the final previous clip concatenated with
  the final next clip and re-encoded once, for scrubbing the cut on a
  timeline.
- A JSON report (`--report`) recording input/output paths, the detected
  trim and alignment confidence, the affine and curve color transforms, the
  ECC warp decision, the seam residual and its z-score, and all tuning
  options used for the run.
- A diagnostics directory (`--diagnostics-dir`, unless `--no-diagnostics`)
  containing:
  - `01_reference_boundary.png`: boundary frame the altered clip must match
  - `02_source_boundary.png`: original boundary frame from the altered clip
  - `03_source_corrected.png`: boundary frame after color (and optional
    spatial) correction
  - `04_output_boundary_encoded.png`: decoded boundary frame from the
    output video
  - `05_encoded_absdiff_amplified.png`: amplified absolute difference image
  - `06_flow_confidence.png`: visualization of the seam-residual detail
    band/confidence mask (only written when a transition was needed)
  - `07_color_curves.png`: per-channel LUT curve plot (only written for
    `--color-model affine+curves`)

## Tuning

Start with the defaults. Increase `--mask-margin` when frame edges contain
generation artifacts, black borders, watermarks, or partial objects that
should not drive alignment or color solving.

Lower `--trim-percentile` when the clips contain localized changes at the
seam and the color solve is overfitting to those changing regions.

Use `--transition-frames 0` to inspect the pure affine/curve/warp correction
without the seam-residual transition.

If temporal-overlap detection misfires on an unusual pair, set `--trim`
explicitly (`--trim 0` forces a butt joint).

## Troubleshooting

If the tool reports an aspect-ratio mismatch, normalize the clips with
`ffmpeg` before running Seamstress; anything narrower than a 0.5% aspect
mismatch is scaled automatically.

If ECC alignment fails to converge, Seamstress falls back to identity (no
spatial warp) and relies on color correction plus the Stage 5 seam
transition; this is expected on pairs with too much motion for a single
affine warp to explain.

If the output path is the same as either input path, Seamstress exits
instead of overwriting source media. Write to a new path and replace files
manually only after reviewing the result.
