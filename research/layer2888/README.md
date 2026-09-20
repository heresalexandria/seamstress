# Frame2888: source-only layer proof — rejected for output

The bounded experiment separates woman/sloth/jetpacks/flames from the city using hand-annotated seed polygons, RGB foreground seeds and OpenCV GrabCut. It transforms each original incoming foreground with one similarity and its original background with a separate affine. It fills only pixels visible in masked neighboring original frames; it does not synthesize pixels or blend character poses. **Magenta is deliberately exposed missing background, not a proposed finishing effect.**

## Result

This prototype is not complete or suitable for integration. Splitting the layers removes the forced movement of the characters caused by a global affine, but the matte/clean-plate problem is material. Source pixels behind the legs and flame edges are not adequately reconstructed from the tested neighbors. Background donors can also disagree across city depths. No production modules were changed.

- `foreground-isolation.png`: incoming-frame mask on neutral gray. Main outlines are identified well; silver packs and partly transparent flames retain uncertain background-colored edge slivers. A binary GrabCut mask is not a complete alpha matte.
- `v2/result-2888.png`: best bounded affine prototype with exact inverse previous-frame donor. **3,189 uncovered interior pixels** remain, particularly under arms, between legs, feet and flames.
- `v2/result-2889.png`, `v2/result-2896.png`: **5,230** and **5,934** uncovered interior pixels. These reveal that the donor solution is not temporally stable. Outer-frame holes are reported separately.
- `v2/cut-blink.mp4`: hard toggle between original outgoing drawing and diagnostic incoming composite, including exposed holes. This is a comparison, not a repaired video.
- `v2/report.json`: matrices, registration evidence, donor contributions and hole counts.

Foreground incoming→outgoing similarity:

```
[ 1.01125595,  0.00037237, -6.60278891]
[-0.00037237,  1.01125595, -6.03751842]
[ 0,           0,           1         ]
```

Foreground 311/369 feature inliers, median residual0.93px, p90 3.01px.

Background incoming→outgoing affine:

```
[ 1.00731900,  0.00015832, -4.43289795]
[-0.00845351,  1.01571791, 13.65857395]
[ 0,           0,           1         ]
```

Background 868/1786 inliers, median residual2.51px, p90 9.33px. A single city plane is too limited for all observed depth changes. The earlier unrestricted-donor attempt (`result-2888.png`) also showed copied flame contamination and building-strip misalignment; v2 rejects the poorest donors and enforces the previous-frame inverse mapping exactly at the cut.

## Additional adjacent-cut dense background test — also rejected

`dense_cut_test.py` uses source2887→2888 only, DIS forward/backward consistency, and normalized smooth residual fields for the **background alone**. The incoming foreground keeps its own fixed similarity. It samples a single original drawing for each layer, with no two-pose blending and no neural synthesis. Previously visible background from outgoing2887 fills available holes exactly.

- Raw background flow extrapolation folds: Jacobian minimum−19.81. Heavy smoothing is required.
- A24px additional Gaussian smoothing finally gives Jacobian range0.737–1.139, passing the bounded no-fold test.
- Straight source-building lines nevertheless acquire up to1.72px p95 deviation from their best-fit straight line, with a maximum of2.92px on one long roofline. Empire State vertical facade lines have roughly1.5–1.7px p95 curvature. Passing a Jacobian test does not preserve straight architecture.
- **4,612 uncovered pixels** remain around foreground outlines/flames, shown explicitly in `dense/result-2888.png`.
- The native result visually contains matte gaps and residual copied flame/background-edge contamination. No temporal sequence was produced because those defects already invalidate this proof.

The extra test was stopped at these artifacts. It demonstrates that per-pixel background registration can reduce some relative layer discrepancy, but it does not establish an invisible repair. Further source-only work would require a better matte and trustworthy depth-specific clean-plate registration; this experiment does not justify presenting frame2888 as solved.

## Later static clean-plate experiment

A later single built-in image-generation call produced a faithful static city clean plate and resolved the missing pixels. The bounded48-frame source-layer proof still failed on moving matte/plate boundaries and depth-dependent parallax, and was rejected for integration. Its final report and exact visual defects are documented in `cleanplate/README.md`; the raw generated asset remains available for future work. Frame2888 is still unresolved.
