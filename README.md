# Seamstress

A local video studio for making stitched continuous shots feel continuous. Drop in a video, review detected joins, match framing and color, and export a high-quality corrected movie.

**[Download for Apple silicon](https://github.com/heresalexandria/seamstress/releases/latest/download/Seamstress-mac-arm64.dmg)** · **[Download for Intel Mac](https://github.com/heresalexandria/seamstress/releases/latest/download/Seamstress-mac-x64.dmg)** · [Release notes](https://github.com/heresalexandria/seamstress/releases/latest)

Release installers are built by GitHub Actions, signed with Developer ID, notarized by Apple, and verified before publication. Open the DMG and drag **Seamstress** into Applications. The app includes its processing engine and FFmpeg; no Python installation or API key is needed. Download links resolve after the first signed release is published.

## In the app

1. Drop a video onto Seamstress. It scans the entire timeline for candidate seams; common 10-, 15-, and 30-second spacing helps detection without restricting marker positions.
2. Review, add, remove, or drag markers. Enter a timecode for precise placement.
3. Run analysis and correction, then compare the original and corrected seam previews or the full movie.
4. Export at the original resolution and frame rate, with original audio preserved.

Seamstress applies gradual, bounded framing and color corrections to original frames. The desktop workflow never morphs drawings, crossfades poses, or generates replacement frames. It skips corrections that lack reliable evidence. A redraw, changed gesture, or intentional edit may still be visible; inspect previews before exporting.

The installed app checks GitHub Releases for updates. Click the version button or **Seamstress → Check for Updates…** to download an update and restart when ready. Processing must finish or be cancelled before installation. See [how updates work](docs/updates.md).

## Command line

Install Python 3.12 or newer and FFmpeg, then:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.lock
python -m pip install --no-deps -e .
seamstress process input.mp4 --work-dir input.seamstress --output corrected.mp4
```

One command runs detection, analysis, correction previews, and export. The same stages can run separately with `detect`, `mark`, `calibrate`, `preview`, and `export`. Projects save markers, analysis, correction plans, and revision-specific artifacts for repeatable review.

**[Desktop and CLI guide](docs/SEAMSTRESS.md)** includes individual commands, seam timecodes, supported inputs, and limitations. The current engine supports constant-frame-rate, square-pixel, 8-bit SDR video; it is not an HDR mastering workflow.

## Develop and release

```sh
# After the Python setup above:
cd app
npm ci
npm run dev
```

```sh
.venv/bin/python -m unittest discover -s tests -v
npm --prefix app test
npm --prefix app run build
npm --prefix app run test:e2e
```

[Release guide](docs/RELEASES.md) · [Signing and notarization](docs/MACOS-SIGNING.md) · [Updater design](docs/updates.md)

CI tests pull requests and builds native packages. A merged PR with exactly one `major`, `minor`, or `patch` label starts the signed release pipeline; `no-release` skips publication. Both Mac architectures must pass signing, notarization, package verification, and the actual app workflow before installers and update metadata are published together.

## Earlier implementations and reproduction

The previous GitHub implementation is preserved unchanged in [legacy/](legacy/README.md), with its original history retained. It is separate from the current desktop app and CLI.

The accepted original-frame experiment remains reproducible: [IYTYT eight-join guide](docs/REPRODUCE-IYTYT.md), [color refinement](docs/REPRODUCE-COLOR-REFINEMENT.md), and [research history](docs/EXPERIMENTS.md). Source videos and generated outputs are local files and are not included in this repository.
