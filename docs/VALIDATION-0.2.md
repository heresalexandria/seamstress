# Seamstress 0.2 validation

Tested locally on macOS arm64, September 20, 2026. These checks establish implementation behavior and media integrity, not universal perceptual seamlessness.

## Processing engine

`OPENBLAS_NUM_THREADS=2 .venv/bin/python -m unittest discover -s tests -q` passes **139 tests**. Coverage includes existing source-conform/local-color regressions plus new whole-timeline detection, general calibration, saved-project orchestration, and actual FFmpeg integrations.

The new cases include arbitrary eight-second boundaries, subtle regularly spaced changes, no-cut pans/fades/held drawings, portrait footage, unreliable/scene-change geometry, short source handles, no-seam identity output, explicit marker replacement, reanalysis invalidating downstream artifacts, concurrent stale-plan rejection, cancellation, native frame count and fractional rate, multiple copied audio streams, video at a nonzero stream index, delayed audio, exact seam-preview counts/rates, VFR/anamorphic/high-bit-depth rejection, and nonzero video-start rejection.

Export verification checks frame count, dimensions, fractional frame rate, stream start times, and compressed audio hashes. Its report explicitly does not certify visual perfection.

## Original movie

The default detector finds all nine known IYTYT continuation boundaries at exact first-incoming-frame indices: **361, 722, 1083, 1444, 1805, 2166, 2527, 2888, 3240**. No additional suggestions were returned at default sensitivity in this run. Synthetic tests independently cover arbitrary timing; the implementation does not encode these source-specific positions.

Automatic calibration generated a new review candidate with five accepted geometry matches, eight protected tone curves, seven regional color models, and a constant 5.56% total viewing crop. It conservatively declined geometry at frames 722, 1444, 1805 and 2888, recording the evidence/reasons. This new calibration is not a replacement for the previously reviewed, manually calibrated IYTYT recipe.

The complete 3,347-frame movie was rendered as a 480-pixel-wide corrected preview, with nine individual seam previews in `output/seamstress-app-validation.seamstress`. The accepted baseline/color-refined outputs and their reproducible plans remain separate.

## Desktop interface

`npm run build` passes TypeScript checking and Vite production compilation. `npm test` passes three Node tests for restricted asset/media access and HTTP byte-range playback semantics.

The real Electron smoke test uses the actual preload, Python worker and encoded media. It replaces only native file dialogs with deterministic choices. It exercises import/detection; manual marker create/edit/toggle/delete; seeking and frame stepping; project reopening; analysis, full/seam previews and native export; synchronized original/corrected playback; and comparison controls. An additional manual automation check cancels a real import and keeps the previous project usable. Default smoke input is generated locally with FFmpeg; no ignored video fixture or fake processing results are required.

Run from `app/`:

```sh
npm run test:e2e
SEAMSTRESS_TEST_APP="$PWD/release/mac-arm64/Seamstress.app/Contents/MacOS/Seamstress" npm run test:e2e
```

The packaged smoke passed the complete workflow using a separate local user-data directory. Runtime inspection confirmed `app.isPackaged` and a live worker executable inside the app’s `Contents/Resources/backend`; evidence is saved in `output/app-ui-smoke/packaged-runtime.json`. Original cartoon UI captures are in `output/app-ui-smoke/empty.png` and `output/app-ui-smoke/workflow-cartoon.png`.

## Standalone bundle

The macOS arm64 backend contains its Python runtime, FFmpeg/ffprobe and 92 FFmpeg dynamic-library dependencies. Its audit checked 297 Mach-O binaries for unresolved/non-system absolute dependency paths. It passed H.264 encode/decode with expected pixel-content checks, isolated imports, relocation, and project opening with Homebrew/Python removed from `PATH`.

The build fingerprints source code, Python/package versions and FFmpeg binaries/dependencies. Source changes during a build cause failure; a subsequent unchanged build reuses the validated bundle. The local release is not Developer ID signed or notarized.
