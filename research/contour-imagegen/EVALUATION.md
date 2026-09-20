# One-call targeted contour restoration test

Built-in `image_gen.imagegen` was used once. No API key was read or used. No generated pixels were integrated into the video.

The edit target is the 384×384 crop at `(448,192)–(832,576)` of `research/rife/native-checked/seam-3240-middle.png` (intermediate source-frame position 3236). Original frames 3228 and 3244 supplied identity and clean-line references. The exact submitted prompt is `prompt.txt`; all three input crops are retained here.

The generator returned a 1254×1254 square image. It is saved unmodified as `generated-roi.png`; `generated-native-roi.png` is its Lanczos reduction to the original crop resolution for inspection. `comparison.png` shows target versus this reduction.

Visual result: the faint overlapping head/hand outlines are substantially cleaner. Overall identity, pose and crop composition remain close. However, it also sharpens/redraws the entire clothing silhouette, fur markings, shoes and parts of the architecture. The result therefore violates the requested locality and unchanged-pixel constraints. Hands are resolved into a particular drawing rather than only removing faint duplicate pixels, and the stronger line weight could create a temporal pop.

At the original crop size, mean absolute RGB difference is 6.27/255 over the full image and 4.86/255 in selected untouched architectural strips. Even excluding generous bounding boxes around all requested head/hand edit regions, 22.1% of remaining pixels change by more than 10 levels in at least one channel. This includes scaling/resampling differences as well as generative changes; it is not a pure measure of geometry drift.

Assessment: unsuitable as a direct frame or complete-ROI replacement. It may serve as a visual reference for carefully masked local contour patches, but those patches would still require registration, line-weight matching, and adjacent-frame visual testing. No temporal usability has been demonstrated. The unchanged RIFE bridge remains the more faithful automatic output for this test.
