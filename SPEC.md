# Seamstress v2 — Technical Specification

Goal: given two clips that form one continuous shot (generated separately, e.g. Seedance 2
continuations), alter the **next** clip so that placing the two back-to-back on a timeline is
visually seamless at Hollywood-finishing quality. Must work on arbitrary clip pairs, not just
well-behaved ones.

The existing `seamstress.py` (v1) solves alignment + color from a SINGLE boundary frame pair and
rejects fps mismatches. v2 replaces it. Keep the same single-file CLI (`seamstress.py`,
positional `previous next`, `--alter next|previous` preserved), rewrite internals per the stages
below. Python 3, numpy + opencv (cv2) from `requirements.txt`, ffmpeg/ffprobe subprocesses for
all I/O. No new heavyweight dependencies (no torch). scipy is acceptable if added to
requirements.txt (useful for PCHIP curves).

## Known facts about the test pair (01.mp4, 02.mp4 in repo root)

- Both 1280x720 yuv420p bt709 h264. 01: 241 frames @ 241000/10041 (~24.0016) fps. 02: 240 frames
  @ 2410/101 (~23.8614) fps. Seedance emits slightly different rational fps per clip — v1's
  exact-fps check rejects its own primary use case. Never require exact fps equality.
- Verified by cross-correlation probe: 02[0] ≈ 01[last-1], 02[1] ≈ 01[last] (high-pass NCC 0.71
  and 0.41 vs ≈0.0 noise floor). Correct trim for this pair = drop first 2 frames of 02.
- Matched boundary pair shows ≈ −3 G / −2.5 B mean channel shift in 02 vs 01.
- Use this pair as the acceptance test (see §8).

## Stage 1 — Ingest & conform

- ffprobe both clips: size, rational fps, nb_frames, pix_fmt, color metadata, audio presence.
- fps policy (`--fps-policy`, default `conform`):
  - `conform`: treat both as frame sequences; output the altered clip retimestamped at the
    UNALTERED clip's exact rational fps (so the pair shares one timebase on the timeline). Warn
    if nominal rates differ by > 1%. Never blend/interpolate frames by default.
  - `strict`: old behavior, error on mismatch.
  - (Optional `resample` via ffmpeg minterpolate may be stubbed as not-implemented.)
- Resolution mismatch: lanczos-scale the altered clip to the reference clip's size, with a
  warning. Refuse only on aspect-ratio mismatch > 0.5%.
- Carry bt709 (or whatever the source declares) color tags through to the output encode
  (`-colorspace/-color_primaries/-color_trc` flags).
- All frame processing in RGB float32 0–255, decoded via ffmpeg rawvideo rgb24 pipes (as v1).

## Stage 2 — Temporal alignment (overlap / rewind detection)

Purpose: find integer frame offset between clip1's tail and clip2's head; trim clip2's head so
its first output frame is the first NEW frame (no rewind glitch).

- Decode tail N=25 frames of clip1 and head M=50 of clip2 downscaled to ~320px wide.
- Feature: per-frame high-pass grayscale (gray − gaussian blur σ≈2), zero-mean unit-var
  normalized. This makes matching robust to the color shift.
- Cost matrix C[i,j] = mean(feat1_i * feat2_j) (NCC). Aggregate along diagonals
  (constant j−i); best diagonal = candidate offset. Confidence = (best diagonal mean −
  median of others) / (std of others); require confidence ≥ ~4 AND best per-pair NCC ≥ ~0.25
  to accept overlap.
- If accepted: trim = index in clip2 of (frame matching clip1's last frame) + 1. Also expose
  `--trim N` manual override and `--trim 0` to disable. Report trimmed frame count and
  confidence in stdout + JSON report.
- If rejected: assume butt joint (trim 0), print that no overlap was detected. If the best
  alignment implies clip2 starts BEFORE clip1's tail even ends minus available frames (i.e.
  a gap — clip2's head matches nothing and motion suggests missing frames), we cannot
  synthesize frames; warn and let Stage 5 hide it.
- Audio: if clip2 has audio, trim it by trim/fps seconds so A/V stays in sync; otherwise ignore.

## Stage 3 — Global color match (the core fix)

Never solve color from a single frame pair. Build a pooled, motion-compensated correspondence
set:

- Frame pairs: if Stage 2 found overlap of K≥1 frames, use ALL overlapping pairs (same content
  rendered twice — gold correspondences). If no overlap, use the boundary pair plus up to 4
  additional pairs formed by flow-aligning clip2 frames 0..3 back onto clip1's last frame.
- For each pair: align clip2 frame onto clip1 frame with dense optical flow
  (`cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)`), computed on the high-pass
  gray features. Per-pixel confidence mask = forward-backward flow consistency (reject where
  ||fwd + warped bwd|| > ~1.5 px) AND away from frame borders (mask margin, default 48 px) AND
  not near sensor clip (all channels in [4, 251] on both sides).
- Pool confident pixels across all pairs (subsample to ≤ ~500k samples).
- Color model (`--color-model`, default `affine+curves`):
  1. `affine`: 3x4 RGB affine solved by iteratively trimmed least squares (keep v1's approach,
     trim percentile default 75).
  2. `affine+curves`: affine base, then residual per-channel smooth monotonic 1-D LUTs fitted on
     binned medians of (affine-corrected source → reference), monotonicity enforced, regularized
     toward identity where bins have < ~200 samples. PCHIP or isotonic + smoothing. 256-entry
     LUT per channel, applied with linear interp on float values.
- Apply ONE constant transform to every frame of clip2 — never per-frame/time-varying (temporal
  wobble is worse than residual offset).
- Guardrails: if the solved transform changes global mean by > 25 code values or curves are
  wildly non-identity at the extremes, warn loudly and clamp curve extrapolation.
- Dither: after all float-space correction, add triangular-PDF dither (amplitude 0.5 code value,
  fresh noise per frame) before quantizing to uint8, to prevent banding from the curves.

## Stage 4 — Spatial alignment (make it conditional)

Keep v1's ECC affine estimation on the (color-invariant, high-pass) boundary pair, but:

- Auto mode (default): apply the warp to clip2 only if ECC score ≥ 0.5 AND the implied max
  corner displacement is between 0.2 px and ~12 px. Below 0.2 px → skip (identity is fine);
  above ~12 px or low score → skip warping and rely on Stage 5 (a bad global warp across a
  whole clip is worse than a soft transition). `--warp always|never|auto`.
- When overlap frames exist (Stage 2), estimate the affine on the best-matching pair, not on
  frames one step apart in time.

## Stage 5 — Seam finishing (not a crossfade)

After trim + color + optional warp, measure residual at the (new) boundary: MAE between clip1's
last frame and flow-aligned corrected clip2 first frame.

- If residual is tiny (MAE < ~1.5): clean cut, do nothing.
- Otherwise: **motion-compensated residual propagation** over T frames (default 6,
  `--transition-frames`):
  - residual R0 = clip1_last − corrected_clip2_first (float, computed after flow alignment so
    detail edges line up; where flow confidence is low, keep only the low-frequency component).
  - Split R0 into low-frequency (gaussian σ≈8) and detail bands. The low band is ramped over T
    frames with a smoothstep decay. The detail band is PROPAGATED: warp it frame-to-frame using
    clip2's own optical flow (frame t → t+1) so the correction sticks to moving content instead
    of ghosting, with decay (e.g., ×(1 − smoothstep(t/T))) and confidence-masked falloff.
  - This replaces v1's static `seam_residual * strength` ramp (which ghosts under motion).
- For `--alter previous`, mirror the logic at the end of clip1.

## Stage 6 — Verification, outputs, report

- Encode: libx264, `--crf` default 10, preset slow, yuv420p, copy audio (with Stage 2 trim
  applied), `+faststart`, color tags carried through. Output default `<altered>_seamless.mp4`.
- `--joined PATH` (default: also write `<output stem>_joined_preview.mp4`): concatenation of
  final clip1 + corrected clip2 re-encoded once, for timeline QC.
- Seam metric (the acceptance signal): in the joined result, compute per-frame-step mean abs
  difference d(t) for ~24 steps on each side of the cut. Seam z-score =
  (d(cut) − median(d)) / MAD-based σ of the neighboring steps. Report it; z ≤ 2 is a pass,
  print a clear warning if above.
- Diagnostics dir (as v1): boundary pair PNGs before/after each stage, flow-confidence mask,
  amplified diff, plus a small PNG plotting the per-channel LUT curves. JSON report with all
  matrices, trim, confidence, seam z-score, settings.

## 7 — CLI summary (new/changed flags)

`--trim auto|N` (default auto), `--fps-policy conform|strict`, `--color-model
affine|affine+curves`, `--warp auto|always|never`, `--transition-frames N`,
`--joined PATH | --no-joined`, plus v1's surviving flags (`--alter`, `--output`, `--report`,
`--diagnostics-dir`, `--no-diagnostics`, `--mask-margin`, `--crf`, `--preset`,
`--trim-percentile`). Remove `--anchor-frames` (superseded by `--transition-frames`).

## 8 — Acceptance tests (must pass before done)

Run with the repo venv (`./venv/bin/python`). ffmpeg is on PATH.

1. `./venv/bin/python seamstress.py 01.mp4 02.mp4` succeeds (no fps rejection), auto-detects
   trim = 2, seam z-score ≤ 2 in the joined preview, output frame count = 02 frames − 2.
2. Boundary continuity: last output frame of the transition region shows no ghost doubling —
   verify by checking that d(t) has no spike AND no dip-below-neighbors (a near-zero d at the
   cut means a duplicated frame — also a failure).
3. `--alter previous` runs and produces the mirrored result.
4. `--trim 0` and `--warp never` run cleanly.
5. Synthetic fps test: `ffmpeg -i 02.mp4 -r 25 -c:v libx264 -crf 12 02_25fps.mp4` (scratch dir),
   then run against 01.mp4 — must conform with a warning, not reject.
6. Audio-bearing test: add a silent audio track to a copy of 02, confirm output audio duration
   shrank by trim/fps seconds.
7. v1-style diagnostics and JSON report are written and contain the new fields.
