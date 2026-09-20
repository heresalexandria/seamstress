# Reproduce the reviewed framing candidate

This recipe produces `output/IYTYT-framing-reviewed.mp4`. It combines automatic
framing recovery at 1:00 and cadence-aware camera estimation at 1:30 with the
previously accepted source-specific framing at 1:15. The depth jump at 2:00
remains unresolved. It is a candidate for playback review, not a claim of
perfect continuity. See the [measured findings](FRAMING-REVIEW.md).

Use the existing environment or follow the [baseline setup](REPRODUCE-IYTYT.md#1-set-up-the-project).
Run from the repository root with the original `IYTYT.mp4`; its SHA-256 must be
`c64928f7c72aee57d4a536d1af711d354a0358109510b175407d011621e855ca`.

```sh
.venv/bin/seamstress conform IYTYT.mp4 \
  --plan plans/IYTYT-framing-reviewed.json \
  --output output/IYTYT-framing-reviewed-reproduced.mp4 \
  --crf 14
```

Choose a new output name; existing videos and sidecars are protected from
overwrite. Rendering requires no API key or generated images. There is one
original drawing per output frame, no morphs, no temporal blending, and no
retiming. Full exports copy the original audio bitstream. The recipe retains
3347 frames at 24000/1001 fps and 1280×720, with a constant 5.4133% crop of each
total dimension to keep source edges covered.

To reproduce plan construction as well:

```sh
.venv/bin/seamstress design-conform IYTYT.mp4 \
  --calibration plans/IYTYT-framing-reviewed-calibration.json \
  --output output/framing-reviewed-rebuilt-plan.json
.venv/bin/seamstress conform IYTYT.mp4 \
  --plan output/framing-reviewed-rebuilt-plan.json \
  --output output/framing-reviewed-rebuilt.mp4 \
  --crf 14
```

The stored `review_decisions` identifies frame 1805's accepted calibration and
its fingerprint. The generic plan builder reconstructs all rendering matrices,
color models and viewing crop from this calibration; descriptive plan notes and
generator metadata can differ. Source paths are relative for portability.

To check a full export:

```sh
.venv/bin/seamstress verify IYTYT.mp4 \
  output/IYTYT-framing-reviewed-reproduced.mp4 \
  --plan plans/IYTYT-framing-reviewed.json \
  --work-dir output/framing-reviewed-reproduced-review
```

For an eight-second local preview, add `--start-frame 1348 --end-frame 1540`
to the first render command and choose a new filename. This previews the 1:00
join at 4.004 seconds. Equivalent ranges are 1709–1901 for 1:15, 2070–2262 for
1:30, and 2792–2984 for 2:00. End frames are exclusive. Preview audio is trimmed
and re-encoded; `verify` expects a complete video.

Running ordinary automatic analysis on another video uses only the generalized
checks. It does not adopt this movie's reviewed 1:15 decision. Review notes
identify joins where automatic framing remains unreliable.
