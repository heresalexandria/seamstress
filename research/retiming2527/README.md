# Discrete drawing retiming at2527 — do not integrate automatically

The source contains a real eight-frame held drawing, frames2527–2534 (334ms). This is more than a camera stop: both character regions remain held for all seven adjacent intervals, whereas the woman's preceding12 frames contain only two held intervals and substantial articulated motion. After2534 the source has consistent two-frame drawing exposures.

The native proof `comparison-audio.mp4` compares the original to exact original-frame selections. It contains120 frames (5.005s), covering output2503–2622. No geometry transform, optical flow, image blend, synthesized pose, audio retiming, or production change is applied. `retimed-audio.mp4` is the candidate alone. Original audio remains aligned to the original output timeline.

`source-map.json` skips source2527–2532, retaining the final two pictures of the same initial held drawing. It reduces the initial exposure from8 frames to2. A smooth3-second reduction of the source-time offset repays the six omitted frames through six exact repetitions. Those repetitions occur at output2548,2559,2567,2576,2584,2595. The first six advancing poses are preserved; none are interpolated. The source map is monotonic, has the same length as the original, and is identity again from2595 onward.

The cost is concrete: six normally two-frame exposures become three-frame exposures. This replaces one334ms stop with six125ms exposures in an otherwise regular83ms cadence. It also advances gestures and mouth drawings by up to250.25ms against the unchanged soundtrack. The still comparison visibly confirms earlier pointing and earlier tablet/sloth movement; those are genuine original poses, but their timing has changed. Retiming does not repair the furniture/body redraw, color shift, or lost camera continuation.

## Decision

Reject automatic integration under the current seamlessness requirement. The initial hold is objectively shortened, but the proof creates new cadence irregularity and temporary audio/action misalignment. That tradeoff could be preferred after normal-speed listening/viewing, but the hold-duration metric alone is not enough to accept it. The root task will review the available proof; no production frame mapping was changed.

Failure criteria for any further discrete retiming candidate:

- Skipping a distinct original action state, introducing backward source motion, or adding a new jump between different drawings.
- New exposures longer than the local normal two-frame cadence that create visible judder; the present proof adds six three-frame exposures.
- Moving a sung/spoken mouth shape, gesture, or beat relative to the soundtrack; this proof permits up to250ms displacement and has no synchronization validation.
- Merely moving the camera stop or appearance discontinuity to another point.
- Treating a lower picture-difference score or shorter hold as perceptual acceptance.

At3240, existing source evidence shows a three-frame initial drawing (3240–3242), followed by normal two-frame exposures from3243. That is only one excess frame (~42ms). Removing it while preserving duration would add an excess exposure elsewhere, while the much larger push-in-to-stop camera change remains. Reject automatic retiming there as well; no3240 retiming proof was generated.

Reproduce the2527 experiment with:

```sh
.venv/bin/python research/retiming2527/audit_and_proof.py
```

`region-cadence.json` contains whole-frame, woman, sloth, and furniture change evidence. `source-map.json` contains every selected source index and the measured original/candidate exposure runs. Compression-noise-tolerant hold detection was cross-checked against the startup contact sheet; it is not a substitute for listening to the proof.
