# Reproduce the current IYTYT baseline

This is the process that produced `output/IYTYT-source-conform-eight-joins.mp4`. It uses the original source frames, global framing/aspect correction, and protected color curves. It does not generate poses, blend adjacent frames, or use the rejected neural-bridge renderer.

The user reviewed this baseline as much better and mostly seamless, with local color changes still visible at the joins. Those remaining color corrections have **not** been added to this checkpoint.

[IYTYT-baseline.json](IYTYT-baseline.json) records SHA-256 fingerprints of the existing source, rendered movie, saved plan, and calibration so this exact checkpoint can be identified later.

Published recipe paths are relative to the repository root. Removing the original
machine-specific path prefixes changes the plan and calibration file hashes,
which the manifest now records alongside their historical
`before_path_normalization_sha256` values. Source and rendered-video hashes,
calibration measurements, transforms, color models, frame timing, and audio are
unchanged. Historical research reports can still refer to the earlier recipe
hashes. This path normalization does not change rendered pixels.

## 1. Set up the project

Run all commands from the repository root. On macOS, install FFmpeg if needed:

```sh
brew install ffmpeg
```

The tested environment is Python **3.13.15**, FFmpeg/ffprobe **7.0.2**, and the versions in `requirements.lock`. Python 3.13 must be installed separately. Set up the local environment once:

```sh
python3.13 -m venv .venv
.venv/bin/python -m pip install -r requirements.lock
.venv/bin/python -m pip install --no-deps -e .
.venv/bin/seamstress --help
```

Alternatively, use `uv` (which can provision the Python runtime):

```sh
uv venv --python 3.13 .venv
uv pip install --python .venv/bin/python -r requirements.lock
uv pip install --python .venv/bin/python --no-deps -e .
```

An existing working `.venv` can be reused without reinstalling. The supplied local environment was created with `uv` and has no `pip` module; if reinstalling its packages, use the two `uv pip install` commands above. The render commands use the environment's executable directly, so shell activation is unnecessary. Neither `.env`, an OpenAI API key, PyTorch, nor downloaded neural weights is needed for this workflow.

Keep the original **`IYTYT.mp4` in the repository root**. Its SHA-256 must be:

```text
c64928f7c72aee57d4a536d1af711d354a0358109510b175407d011621e855ca
```

You can check it with `shasum -a 256 IYTYT.mp4`; the CLI also verifies it automatically. Source media, rendered videos, models, `.env`, and `.venv` are excluded from Git. A fresh checkout needs its own copy of this exact source video.

## 2. Render using the saved recipe

```sh
.venv/bin/seamstress conform IYTYT.mp4 \
  --plan plans/IYTYT-eight-joins.json \
  --output output/IYTYT-source-conform-eight-joins-reproduced.mp4 \
  --crf 14
```

The original render used this same source, plan, and quality setting, with the output name `output/IYTYT-source-conform-eight-joins.mp4`. The command above uses a different output name so it can run beside that existing movie. Changing only the output name does not change the treatment.

The renderer refuses an existing output video or its matching `.repair.json` sidecar. For another run, choose a new output name. It never overwrites the source. A successful run writes the video and a `.repair.json` sidecar with the source hash, plan hash, tool versions, frame mapping, crop, and unresolved-join status.

## 3. Verify the rendered movie

```sh
.venv/bin/seamstress verify IYTYT.mp4 \
  output/IYTYT-source-conform-eight-joins-reproduced.mp4 \
  --plan plans/IYTYT-eight-joins.json \
  --work-dir output/eight-joins-reproduced-review
```

Use a new verification directory on subsequent runs. It produces:

- `review.html`: synchronized original/candidate players with join selection and frame stepping. Open it in a browser.
- `comparison.mp4`: a side-by-side reel of the joins.
- `before-after.jpg`: frames immediately around the joins.
- `report.json`: frame count/rate, duration, dimensions, copied-audio verification, and local comparison measurements.

The full baseline retains **3,347 frames**, **24000/1001 fps**, **1280×720**, and approximately **139.598 seconds**, with the original audio bitstream copied unchanged. Verification of the existing baseline found zero video-duration difference. Metrics and successful tests do not establish invisible joins; inspect playback at normal speed.

## Optional: render only the first join

This uses the **same full-movie recipe**, rather than the older separate first-proof recipe:

```sh
.venv/bin/seamstress conform IYTYT.mp4 \
  --plan plans/IYTYT-eight-joins.json \
  --start-frame 289 --end-frame 433 \
  --output output/first-join-baseline-reproduced.mp4 \
  --crf 14
```

The preview contains 144 frames, lasts 6.006 seconds, and has the join 3.003 seconds in. Frame indices are zero based; the start is inclusive and the end exclusive. Preview audio is trimmed and re-encoded as AAC. Full renders alone copy the original audio bitstream. `verify` expects a full-length render and rejects these offset previews.

## Optional: rebuild the recipe from calibration

The saved plan is sufficient for rendering. To reproduce its construction as well:

```sh
.venv/bin/seamstress design-conform IYTYT.mp4 \
  --calibration plans/IYTYT-calibration.json \
  --output output/eight-joins-rebuilt-plan.json

.venv/bin/seamstress conform IYTYT.mp4 \
  --plan output/eight-joins-rebuilt-plan.json \
  --output output/eight-joins-rebuilt.mp4 \
  --crf 14
```

The calibration contains measured cut transforms, camera rates, grading curves, and the explicit geometry exclusion at frame 2888. The builder reconstructs the per-frame transforms and constant crop without running the exploratory research scripts. All 3,347 transforms match the saved baseline within floating-point precision; the viewing crop and grading curves are identical. A real CLI rebuild-and-render test produced the same decoded video for the six-second preview in the tested environment. File/container bytes and cross-platform encodes are not guaranteed identical.

Use `verify` with the rebuilt movie and rebuilt plan if verifying this alternative route. These measurements are specific to `IYTYT.mp4`; `design-conform` does not automatically calibrate an arbitrary new video. The `run`, `repair`, and `bridge` commands retain older experiments and are **not** the process used for this baseline.

## Treatment and known limits

- Geometry is corrected at frames 361, 722, 1083, 1444, 1805, 2166, 2527, and 3240. Protected grading is applied around all nine joins, including 2888.
- The constant crop removes 5.924% of the total width and height, approximately 2.962% per side. There are no exposed source borders in this plan.
- Geometry changes return toward the source framing over 168 frames on either side; selected camera-rate corrections use 12 frames per side. The final geometry return is shorter because the movie ends.
- Frame 2888, at 120.454 seconds, has no geometric repair. Its characters and city require different corrections. The rejected layer/inpainting experiments are absent from this movie.
- Original drawing changes, startup holds, and the remaining local color shifts are not fully resolved. No generated frames, new local-color fields, or retiming experiments are included.

To run the engineering tests:

```sh
.venv/bin/python -m unittest discover -s tests -v
```

This checkpoint passed 78 tests when it was reviewed. See [the experimental CLI reference](EXPERIMENTS.md) for the plan schema and [research/STATUS.md](../research/STATUS.md) for experimental findings. Run the command above for the current test suite.
