# Seamstress

Seamstress is a local Electron video studio and CLI for detecting, reviewing and correcting joins in stitched continuous shots. It preserves original drawings and timing while matching framing and color.

**Start here: [Desktop app and generalized CLI guide](SEAMSTRESS.md).** Drop a video into the app for automatic seam detection, edit markers on the timeline, compare correction previews, and export at native resolution. Or run the entire workflow with one command:

```sh
seamstress process input.mp4 --work-dir input.seamstress --output corrected.mp4
```

Use `detect`, `mark`, `calibrate`, `preview` and `export` for individual stages. The new workflow accepts arbitrary seam positions; 10/15/30-second intervals are soft detection hints. It makes conservative, reviewable corrections; it cannot guarantee an invisible repair when source drawings or scene content differ.

The sections below preserve the original experiments and reproduction records.

**Start here: [Reproduce `IYTYT-source-conform-eight-joins.mp4`](REPRODUCE-IYTYT.md).** This guide covers installation, the saved recipe, full rendering, a six-second preview, verification, and rebuilding the recipe from calibration. It uses `conform`; no API key or neural model is needed.

For the separate color-only experiment, see [Reproduce the color-refined candidate](REPRODUCE-COLOR-REFINEMENT.md). It retains the baseline framing and adds optional residual color models; it remains a candidate for viewing.

The [fresh 30s and 120s review](SEAM-REVISIT.md) explains the remaining background-motion discontinuities and the separate color-only improvement to automatic calibration. It preserves these saved recipes.

The current baseline is `output/IYTYT-source-conform-eight-joins.mp4`, rendered from [plans/IYTYT-eight-joins.json](../plans/IYTYT-eight-joins.json) at CRF 14. It preserves original drawings and timing with global framing correction and slow, protected color grading. The user finds this version much better and mostly seamless, but still sees local color shifts at the joins. **It is a checkpoint for further refinement, not a completed imperceptible-seam repair.** Geometry at frame 2888 remains explicitly unresolved. No new local-color treatment is included in this baseline.

The RIFE `bridge` render was rejected in viewing: its pose and scenery morphing is more distracting than the original cuts. Its commands and plans below are retained for reproducible experiments, **not as a recommended repair**.

The older field-warp `repair` mode remains available as an experiment. Its v2 render was rejected for visible transitions and camera/shape wobble; it is not the recommended result.

## Install

FFmpeg and ffprobe must be on `PATH`. Development used Python **3.13.15** and FFmpeg **7.0.2**. The base dependencies pinned in `requirements.lock` require Python 3.12 or newer. The optional `bridge` extra installs PyTorch.

```sh
# macOS, if FFmpeg is not installed:
brew install ffmpeg

python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.lock
python -m pip install --no-deps -e .
seamstress --help
```

Equivalent setup with `uv`:

```sh
uv venv --python 3.13 .venv
uv pip install --python .venv/bin/python -r requirements.lock
uv pip install --python .venv/bin/python --no-deps -e .
```

`conform`, `analyze` and the color experiments need only the base installation above. They do not need a neural model or PyTorch. To reproduce the rejected bridge experiment, install its optional runtime with `python -m pip install -e '.[bridge]'`. For the pinned macOS arm64 / Python 3.13 bridge environment, use the full runtime lockfile instead:

```sh
python -m pip install -r requirements-bridge.lock
python -m pip install --no-deps -e '.[bridge]'
```

With `uv`, replace the first installation command with `uv pip install --python .venv/bin/python -r requirements-bridge.lock`, then install the project with `uv pip install --python .venv/bin/python --no-deps -e '.[bridge]'`. The full lock includes the base dependencies and the tested PyTorch runtime. Other platforms may need platform-specific PyTorch wheels; the unpinned extra above allows the resolver to choose them. Keep dependency versions and the FFmpeg build consistent when comparing reproduction runs. Byte-identical output across machines is not guaranteed.

Every command also works as `.venv/bin/python -m seamstress ...`. Video processing is local and does not read `.env`, upload source frames, or require an OpenAI API key. Initial dependency setup downloads public software; the optional bridge model setup downloads its public weights.

## Source-drawing conform: under development

`conform` maps each output frame to one original source frame. It can change that frame's global framing and pointwise color, while preserving its original drawing, gesture and timing. It does not interpolate between frames, use optical-flow deformation, or generate replacement poses. A full render keeps the source frame count and exact fractional frame rate; original audio is copied without re-encoding.

Use an explicitly reviewed **schema-3** plan, rather than a schema-1 field-warp or schema-2 bridge plan:

```sh
seamstress conform input.mp4 \
  --plan path/to/reviewed-source-conform.json \
  --output output/source-conform-candidate.mp4 \
  --crf 14
```

The plan path above is a generic placeholder. For this exact source, use the saved eight-join plan in the reproduction guide. The older first-join proof below uses different framing support and is retained as a separate experiment.

Reproduce the approximately six-second **first-join-only proof**, spanning original frames 289–432:

```sh
seamstress conform IYTYT.mp4 \
  --plan plans/IYTYT-first-proof.json \
  --start-frame 289 --end-frame 433 \
  --output output/source-proof/first-join-cli.mp4 \
  --crf 14
```

Choose a new output filename if the proof already exists. [IYTYT-first-proof.json](../plans/IYTYT-first-proof.json) is an experimental plan for reviewing the first join; it is **not an accepted all-join repair plan**. The output contains 144 frames at 24000/1001 fps, lasting 6.006 seconds.

`--start-frame` is inclusive and `--end-frame` is exclusive, both on the original source timeline. Defaults select the full source. A preview still validates the complete source hash and plan, and applies matrices/grade curves using original frame indices. Preview audio is trimmed to the same interval and **re-encoded as AAC**; only a full render copies the original audio bitstream. The sidecar records the range, preview status and audio mode.

A separate full-length **eight-join experiment** is available. It applies global geometry corrections at eight joins and protected grading at all nine. The flying-scene geometry at frame 2888 (120.454 seconds) remains unresolved and intentionally untouched. This is not a completed or approved repair:

```sh
seamstress conform IYTYT.mp4 \
  --plan plans/IYTYT-eight-joins.json \
  --output output/IYTYT-source-conform-eight-joins.mp4 --crf 14

seamstress verify IYTYT.mp4 output/IYTYT-source-conform-eight-joins.mp4 \
  --plan plans/IYTYT-eight-joins.json \
  --work-dir output/conform-eight-verification
```

Use new output paths if these artifacts already exist. This plan distributes small affine corrections over seven seconds on either side of each supported join and reconciles camera rates over twelve frames each side. It avoids cumulative stretching and uses a constant 5.924% total viewing crop (about 2.962% per side). The encoded result retains all 3,347 frames, the exact duration, and the original audio bitstream. The [review player](../output/conform-eight-verification/review.html) and [comparison reel](../output/conform-eight-verification/comparison.mp4) are for perceptual evaluation; numerical checks do not establish invisible joins.

## Regenerate the experimental plan from calibration

The saved render can also be reproduced from its recorded geometry and color measurements:

```sh
seamstress design-conform IYTYT.mp4 \
  --calibration plans/IYTYT-calibration.json \
  --output output/regenerated-eight-joins.json

seamstress conform IYTYT.mp4 \
  --plan output/regenerated-eight-joins.json \
  --output output/regenerated-eight-joins.mp4 --crf 14
```

`design-conform` reconstructs all per-frame matrices, the constant viewing crop, protected tone curves, and unresolved-join metadata. It verifies the source fingerprint and refuses existing outputs. The calibrated defaults are 168 original frames of geometry support on each side and 12 frames of camera-rate support. Optional `--geometry-support` and `--rate-support` overrides change those geometry parameters only; they do not change the stored color supports. Overlapping geometry supports and a crop beyond the calibrated limit are rejected.

[IYTYT-calibration.json](../plans/IYTYT-calibration.json) records each cut's right-to-left affine matrix, incoming/outgoing motion rates, explicit rate-easing decision, protected RGB LUTs, and the exclusion reason for frame 2888. Rates use log scale, radians and native center-pixel displacement per source frame. Measurements and choices are specific to the source fingerprint. The builder **does not estimate or approve repairs for a new video**. Automatic analysis remains provisional, and different footage needs its own reviewed calibration.

The builder uses the actual source dimensions/frame count and proves a two-pixel source margin for the interpolation footprint. The geometry return at the final cut is shortened to 106 frames because that segment ends early; this differs from the color track, whose remaining small correction persists to the movie's end. `design_report` records the actual supports, crop and coverage. `generator` records the algorithm/version and calibration SHA-256. New render sidecars also record the exact plan SHA-256, renderer version and FFmpeg version.

A parity test compares all 3,347 regenerated transforms with the saved eight-join plan to a floating-point tolerance; the view matrix and all nine grade curves also match. This makes the current experiment reproducible without running the exploratory scripts or trusting stale measurement caches. It does not resolve frame 2888 or certify the other transitions as invisible.

A real CLI smoke test regenerated the plan and rendered original frames 289–432. Its decoded-video hash matched the same preview rendered from the earlier saved plan exactly in the tested environment. This verifies the rebuild-and-render path for that excerpt; it is not a guarantee of byte-identical encodes across machines.

## Conform plan format

A conform plan has `schema_version: 3`, `method: "source_conform"`, `source` metadata matching width, height, frame count and `fps_fraction`, and the source file's SHA-256 in `source_sha256`.

| Field | Meaning |
| --- | --- |
| `segments` | Contiguous `{start,end,matrix,gain,bias}` objects covering every source frame exactly once. Indices are zero based; `end` is exclusive. RGB gains/biases have three values. |
| `matrix` | A 3×3 source-to-output transform. Default geometry permits uniform scale, rotation and translation. |
| `geometry_mode` | Default `"similarity"`; explicit `"affine"` also permits a bounded aspect correction, with largest/smallest stretch ratio at most 1.2. |
| `frame_matrices` | Optional complete list of one 3×3 transform per source frame, overriding segment matrices. |
| `view_matrix` | Optional common similarity transform, applied after the segment or per-frame transform. It can supply a consistent crop/view where coverage requires it. |
| `edge_extension_pixels` | Explicit tolerance for replicated edge pixels, from 0 to 4. Default 0; exposed image area beyond this tolerance is rejected. |
| `grade_curves` | Optional protected left/right RGB LUTs around cuts. Use identity segment gains `[1,1,1]` and biases `[0,0,0]` with this option. |
| `local_color_curves` | Optional experimental color/position models applied after geometry and existing grading; schema below. Omitted or empty leaves the baseline image processing unchanged. |

Protected `grade_curves` contain `{frame,support_before,support_after,left_lut,right_lut}`. Each LUT is 256×3, finite, monotone, bounded to 0–255, and preserves black and white. The left curve reaches full strength at source frame `frame−1`, the right at `frame`; each eases to identity at its outer support endpoint. Supports must be ordered and disjoint. This modifies the pointwise color transfer, not the mixture of adjacent images.

The experimental [protected midpoint curves](../research/segment-color/protected-midpoint-curves.json) use seven-second supports with a shared grade at each join and a slow return to neutral. They avoid the cumulative contrast collapse observed when matching each whole segment to the preceding corrected segment. [Color measurements and limitations](../research/segment-color/SLOW-GRADE.md) explain the construction and range checks. Slow color variation still needs playback review for visible pumping; these measurements are not an accepted full-movie repair.

The renderer validates source identity, plan coverage, transform bounds and LUTs before processing. It refuses existing videos, existing repair sidecars and output aliases of the original source. The `.repair.json` sidecar records zero synthesized frames, identity or contiguous-original frame mapping, frame count, global magnification, exposed-edge tolerance, grade-curve count and clipping from segment gains/biases. Encoding is lossy H.264, so exact identity before encoding does not imply byte-identical decoded pixels afterward.

Conform alone cannot recover contradictory drawings or remove excessive held poses while keeping one original frame per output frame. Review the actual encoded camera motion, crop, proportions and color evolution before treating a proof as a repair.

### Experimental local-color schema

Each `local_color_curves` entry has `{frame,support_before,support_after,left,right}`. Frame indices refer to the original source, including during previews. `frame` is the first frame after the cut; positive integer supports use the same quintic envelope as protected grading: full strength at `frame−1` and `frame`, zero at `frame−1−support_before` and `frame+support_after`. Supports must be ordered, disjoint and begin within the source. A final support may extend beyond the source end.

Both `left` and `right` contain:

| Model field | Required value |
| --- | --- |
| `mode` | `"hybrid"`. |
| `response` | Optional `"headroom-v1"` (default) or explicit `"directional-gamut-v2"`. Unknown responses are rejected. |
| `feature_scale` | Five positive finite scales in RGBXY order, each from `1e-12` to `1e15`. RGB uses 0–255 values after the existing grade; XY uses native output coordinates divided by `width−1` and `height−1`. |
| `centers` | `K×5` finite values, where `1 ≤ K ≤ 512`; centers are already divided by `feature_scale`. Absolute values must not exceed `1e15`. |
| `coefficients` | `K×3` finite RGB correction coefficients, with absolute values at most `1e12`. |
| `limit` | Finite correction bound from `1e-12` through `63.75` RGB levels. |

The model uses normalized Gaussian neighborhood weights with a confidence taper outside calibrated colors/positions. The default response's channel correction is `weight × limit × tanh(raw/limit) × 4(c/255)(1−c/255)`. It preserves each channel's 0 and 255 endpoints. The opt-in `directional-gamut-v2` response instead protects true RGB black/white with a smooth color-level gate, then bounds each signed correction using the available distance to that channel's destination endpoint. A zero channel in a saturated color may increase, or a 255 channel may decrease. Both responses keep RGB in range at the supported limit; existing models with no `response` retain their original behavior. Range safety does **not** establish monotonic color response or perceptual quality. These options require source-specific model review; no automatic fitter or repair-quality claim is provided.

This changes only the current frame's RGB values, without spatial resampling or adjacent-frame blending. `design-conform` passes optional models through from calibration; geometry-support overrides do not change their color supports. The render sidecar records `local_color_curve_count`, the algorithm, each side's response, and hashes of the curves and individual models. The saved eight-join baseline and its reproduction guide do not include this option.

## Optional model for the rejected bridge experiment

```sh
.venv/bin/python -m seamstress setup-model
```

This downloads the official RIFE 4.25 checkpoint into `models/rife425/flownet.pkl` and verifies its SHA-256 before making it available. An existing correct checkpoint is reused; a conflicting file is refused. To choose another location, use `setup-model --output PATH` and pass that same path as `bridge --weights PATH`.

Pinned checkpoint SHA-256:

```text
6615790efd627772917205db291f51cd392528a157ecbb2ecaeec3bff8eb6de2
```

The model and inference code come from [hzwer/Practical-RIFE](https://github.com/hzwer/Practical-RIFE), specifically its [official RIFE 4.25 checkpoint](https://drive.google.com/file/d/1ZKjcbmt1hypiFprJPIKW0Tt0lr_2i7bg/view). The upstream project states that the linked models share its MIT license. Vendored code retains **Copyright (c) 2021 hzwer** and the full [MIT license](../seamstress/_vendor/rife/LICENSE); the model URL and checksum are recorded in [provenance.json](../seamstress/_vendor/rife/provenance.json).

## Reproduce the rejected motion-bridge experiment

The source is `IYTYT.mp4`: 1280 × 720, 3347 frames, at exactly 24000/1001 fps. Its video duration is about 139.598 seconds. Use the saved schema-2 bridge plan:

```sh
.venv/bin/python -m seamstress bridge IYTYT.mp4 \
  --plan plans/IYTYT-bridges.json \
  --output output/IYTYT-bridged.mp4 \
  --device auto \
  --crf 14

.venv/bin/python -m seamstress verify IYTYT.mp4 \
  output/IYTYT-bridged.mp4 \
  --plan plans/IYTYT-bridges.json \
  --work-dir output/bridge-verification
```

If the result already exists, use a new output filename and verification directory. Existing videos and repair sidecars are refused. The renderer checks the source fingerprint and checkpoint fingerprint before inference. The original source is preserved.

`--device auto` chooses CUDA when available, then Apple MPS, then CPU. Set `--device cpu`, `--device mps`, or `--device cuda` to choose explicitly. CPU inference is supported but can take substantially longer. `--crf` accepts 0–51; lower values increase encoding quality and file size.

The video is re-encoded as H.264 with explicit limited-range BT.709 conversion and metadata. The original audio is copied without re-encoding. Only selected bridge interiors receive synthesized images; there is **no global crop**. Original endpoints and all other source images are retained before encoding, so the final lossy video is not pixel-identical to the source outside the bridges.

## Bridge locations and timing

Frame indices are **zero based**. `frame` names the first frame after the original edit. `bridge_start` and `bridge_end` are **inclusive, preserved source endpoints**; only `bridge_start + 1` through `bridge_end - 1` are synthesized.

| Join frame | Time in seconds | Preserved endpoints | Synthesized frames |
| ---: | ---: | ---: | ---: |
| 361 | 15.056708 | 349 → 365 | 15 |
| 722 | 30.113417 | 714 → 726 | 11 |
| 1083 | 45.170125 | 1073 → 1087 | 13 |
| 1444 | 60.226833 | 1436 → 1448 | 11 |
| 1805 | 75.283542 | 1797 → 1811 | 13 |
| 2166 | 90.340250 | 2158 → 2174 | 15 |
| 2527 | 105.396958 | 2519 → 2537 | 17 |
| 2888 | 120.453667 | 2880 → 2892 | 11 |
| 3240 | 135.135000 | 3228 → 3244 | 15 |

The last edit differs from the earlier clip spacing. A nominal 15-second interval is only a detection hint.

The windows are deliberately asymmetric. For example, beginning earlier at the library permits the incoming push-in to slow before the stationary continuation, while ending at 3244 preserves the woman's subsequent turn. The apartment window reaches beyond its excessive startup hold. Endpoint-selection evidence and limitations are in [the temporal audit](../research/temporal_audit/REPORT.md); [bridge-windows.json](../plans/bridge-windows.json) records the measured candidates and content constraints.

## Plans and synthesis

The historical bridge plan has `schema_version: 2`, `method: "rife_bridge"`, complete source metadata, and the original source's SHA-256. Each seam provides:

| Field | Purpose |
| --- | --- |
| `frame`, `time` | Original join index and time. |
| `bridge_start`, `bridge_end` | Preserved source endpoints bounding replacement frames. |
| `start_slope`, `end_slope` | Nonnegative endpoint derivatives of normalized interpolation progress. The monotone cubic limiter prevents the progress parameter from reversing. |
| `camera_control` | Request separate estimation/control of global camera motion, rather than assigning one timing curve to both camera and local appearance. |
| `rationale` | Why these endpoints and constraints were selected. |

A monotone progress parameter alone does not guarantee natural camera movement. The endpoints must also admit a plausible path, and foreground/background parallax must not be mistaken for one rigid camera transform. A camera estimate can be unreliable in a scene dominated by moving characters or several depth layers.

RIFE synthesizes each intermediate image from endpoint evidence; it does not understand dialogue, anatomy, or scene semantics. A two-endpoint bridge can remove an intentional intermediate gesture. Keep bridge windows short, preserve original endpoint action, and review hand contact, faces, limbs, rigid architecture, and occlusion boundaries.

## Inspect the encoded result

`verify` writes:

- `review.html`: a local synchronized original/candidate player with join selection, looping, frame stepping, and playback speed controls.
- `comparison.mp4`: side-by-side seam neighborhoods for inspection at normal speed.
- `before-after.jpg`: the frames directly around each original join.
- `report.json`: frame-count/rate/dimension/duration checks, audio presence, and per-join measurements.

For conform results, numeric comparisons apply the plan's common viewing transform at native resolution before equal downsampling. The visual player shows the uncropped original beside the candidate so the viewing crop remains apparent. Plan status and unresolved geometry joins are displayed explicitly. `verify` accepts complete-timeline renders; it rejects preview sidecars because preview time zero has a source offset.

Rendering writes a matching `.repair.json` sidecar containing bridge indices, synthesis counts, endpoint checks, model identity, and output information.

Watch the full window around each join, including where synthesis starts and stops. Check camera velocity, repeated or frozen poses, doubled outlines, texture changes, lighting, and the rigid shape of buildings or props. A lower registered error can coexist with a visible hitch or distorted drawing. Numerical improvement does not establish an imperceptible transition.

## Other CLI commands and the earlier experiment

`analyze` remains useful for finding and inspecting candidate joins:

```sh
seamstress analyze input.mp4 --work-dir output/inspection --interval 15

seamstress analyze IYTYT.mp4 --work-dir output/explicit \
  --seams 361,722,1083,1444,1805,2166,2527,2888,3240
```

It writes `boundaries.jpg`, a schema-1 `plan.json`, and `scan.json` for automatic detection. Automatic candidates need inspection. A schema-1 analysis plan is **not** a schema-2 bridge plan: use its source metadata, fingerprint, and reviewed joins to author explicit bridge endpoints and timing in a schema-2 plan.

The older `repair` pipeline split dense residual displacement and grading corrections across each seam, transported those fields through surrounding frames, and faded them back to the source. It also supported startup-hold interpolation, field-fold prevention, and an experimental texture transition. Its global crop hid out-of-bounds sampling. On this source, the method left visible redraws and introduced delayed zoom reversals or local shape wobble, so that render was rejected.

Its plan is retained at `plans/IYTYT.json`, and the earlier v2 artifacts remain under `output/v2/` for comparison. To reproduce that **experimental** method in a new location:

```sh
seamstress repair IYTYT.mp4 --plan plans/IYTYT.json \
  --output output/legacy-experiment.mp4 --crf 14
```

`run` currently invokes `analyze` → this older `repair` → `verify`. Both that workflow and `bridge` are rejected experiments on this source, retained for comparison and reproducibility.

## Tests and limits

```sh
.venv/bin/python -m unittest discover -s tests -v
```

The test suite covers frame/coordinate conventions, motion and grading behavior, plan validation, source/output protection, fractional frame rate, SDR color roundtrips, and relevant synthesis invariants. Conform integration tests use a tiny real video/audio fixture through the CLI, verify full-render audio packet hashes and timing, check preview audio alignment and original-frame indexing, check identity source frames before encoding, and load the protected grading artifacts. Integration and normal-speed visual review remain necessary on the actual encoded movie.

The pipeline processes 8-bit SDR RGB. It is not an HDR, variable-frame-rate, or arbitrary-color-space mastering workflow. Neural interpolation cannot guarantee recovery of missing or contradictory content in the flattened source. Quality must be judged on the generated motion and the actual viewing experience, not on the model name or an error score.
