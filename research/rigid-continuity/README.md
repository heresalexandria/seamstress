# First join: original pixels, persistent geometry

Preferred proof: `original-fixed-affine-24-with-audio.mp4`. It contains original source frames313–408 at1280×720 and24000/1001 fps, plus the matching original audio. The cut is at local frame48. This is an experiment, not a claim of an invisible seam.

Every output frame samples exactly one source frame at its original index. No crossfade, neural interpolation, generated pose, dense deformation, or temporal retiming is used. The optional affine transform is constant in aspect/shear throughout the incoming segment. A similarity camera adjustment changes scale/rotation/translation smoothly for24 frames, then the complete correction remains fixed through frame721. Later original camera motion is retained under that constant transform.

## Reproduce

From the repository root:

```sh
.venv/bin/python research/rigid-continuity/first_join.py --duration 24 --geometry fixed-affine
.venv/bin/python research/rigid-continuity/landmarks.py
.venv/bin/python research/rigid-continuity/audit.py
```

The main script writes the silent native proof, side-by-side comparison, contact sheet, and exact source-to-output matrix for every frame0–721. To add the aligned source audio:

```sh
ffmpeg -v error -nostdin -y -i research/rigid-continuity/original-fixed-affine-24.mp4 -ss 13.054708333333334 -i IYTYT.mp4 -map 0:v:0 -map 1:a:0 -t 4.004 -c:v copy -c:a aac -b:a 192k -movflags +faststart research/rigid-continuity/original-fixed-affine-24-with-audio.mp4
```

## Evidence and limitations

The car/windshield has an anisotropic edit impulse. Independent local template matches to16 named static landmarks show that12 agree with an affine transformation: forward horizontal scale0.99520 and vertical scale0.97479. A similarity fit leaves the top windshield corners5.6–6.2 native pixels away while the bottom corners align within1 pixel. The affine fit reduces the top-corner errors to0.09–0.41 pixels. Moving background lamps are explicit outliers and are not forced to match. See `landmark-evidence.json` and `landmark-evidence.jpg`.

Production registration selects a similar affine correction, approximately1.0056 horizontally and1.0249 vertically. The proof uses that fixed affine, adds the predicted one-frame incoming camera step, then reduces the camera-rate difference with an integrated smoothstep over24 frames. The measured preceding zoom is about0.277% per frame; the incoming car framing is near stationary. The correction is never returned to identity. At frame385 it stops changing and is held through721. The original segment remains visually near fixed car framing until roughly541, then pulls back and pans; those original later movements remain.

Fresh encoded-video tracking measures the cut's horizontal/vertical scale step changing from−0.472%/−2.447% to+0.371%/+0.170%. The corrected cut therefore follows the preceding positive zoom instead of shrinking vertically. This is evidence about camera geometry, not overall perceptual success. See `camera-audit.json` and `camera-rate-audit.png`.

The global safety crop removes0.556% of total width/height:3.56 horizontal pixels and2.00 vertical pixels per side. Every transformed output corner remains at least2 source pixels inside the source boundary. No reflected, extended, or invented border pixels are used. The final incoming correction scales horizontally about1.04258 and vertically about1.06256 before that small global crop. Its singular-value ratio remains exactly1.019172 throughout the incoming segment; this verifies that its anisotropy does not animate.

The original illustration changes, background parallax, texture/color change, and motion cadence remain present. No color correction is applied in these files. These source differences cannot all be removed by a single global transform. The first proof should be watched before applying the method to the remaining joins.
