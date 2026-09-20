# Rejected RIFE bridge candidate: encoded temporal audit

The user rejected `output/IYTYT-bridged.mp4`: the synthesized transitions visibly morph at every join and are worse than the original hard cuts. This candidate is not an accepted repair. Anchor variants and further interpolation experiments were stopped. Lower registered error must not be presented as evidence of imperceptible seams.

This audit concerns the first encoded candidate with the windows recorded in `measurements.json`, before any proposed fixes. It decodes native 1280×720 RGB, then measures equal 640×360 proxies over each inclusive anchor span plus eight original frames at both ends. `native-*.jpg` images retain native resolution, original on the left and encoded bridge on the right. `edges-*.jpg` show source/bridge pairs around entry, original join and exit; `curves-*.png` plot registered residual, two-frame sparse similarity estimates, sharpness and brightness. A held interval labelled N means N−1→N.

| Join | Inclusive anchors | Registered error at cut, original → bridge | Remaining evidence |
|---:|---:|---:|---|
| 361 | 349–365 | 5.97 → 1.45 | Broad framing decelerates smoothly; generated mouth progression differs from source. New consecutive near-holds end at 364/365. |
| 722 | 714–726 | 5.43 → 2.49 | Native review reveals missing/ghosted objects despite reduced error. Multiple independent depth layers invalidate a single global-motion quality score. |
| 1083 | 1073–1087 | 4.42 → 2.21 | No large boundary spike, but interpolated line detail softens to 92% of source edge energy. |
| 1444 | 1436–1448 | 4.77 → 2.02 | Added entry/exit near-holds at 1437/1448; entry registered change falls from source 3.25 to 0.30. Minimum edge energy is 87% of source. |
| 1805 | 1797–1811 | 5.22 → 4.10 | Most severe quality loss: missing/ghosted scene objects, minimum edge energy 75%, entry near-hold at 1798 (0.19 vs source 4.71), exit near-hold at 1811 (0.89 vs 3.17). |
| 2166 | 2158–2174 | 6.67 → 2.71 | The previous geometry experiment's late camera wobble is reduced, but synthesized appearance remains perceptible and rejected. |
| 2527 | 2519–2537 | 4.86 → 2.03 | Doubled hands/forearms and flattened gesture at the join in native review; newly synthesized action replaces original poses. |
| 2888 | 2880–2892 | 6.03 → 3.27 | No large boundary spike; original graded change is spread through synthesized imagery. This does not establish fidelity of characters, exhaust or architecture. |
| 3240 | 3228–3244 | 5.92 → 3.78 | Broad push-in stops smoothly and the later original turn remains. Synthesized transition is still perceptually rejected. |

No boundary passed the deliberately conservative *new strong registered-residual spike* threshold. That finding only means the largest one-frame photometric discontinuities were reduced. Optical flow can align deformation and ghosting, and low change can indicate a stall. The added holds and missing objects are direct counterexamples to treating these numbers as acceptance criteria.

Similarity camera rates are diagnostics rather than ground truth: sparse trackers can follow the car, foreground character or a different depth layer from frame to frame. Commanded camera curves are in `commanded-camera.json`. The largest difference from the earlier rejected geometry method is smoother broad framing at 2166 and 3240, not acceptable seamless output.

The next experiment should preserve every original pose and pixel location. Constant grade correction per original segment can address photometric mismatch without introducing synthetic motion; it cannot itself fix a scale/crop reset or startup hold. No further interpolation or anchor refinement is recommended following the user's rejection.
