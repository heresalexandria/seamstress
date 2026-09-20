"""Measure and visually inspect conservative motion-aligned appearance transfer."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from seamstress.analysis import pair_metrics
from seamstress.media import read_frames, VideoWriter
from seamstress.repair import corrected_window
from seamstress.texture import aligned_appearance, texture_transition


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("IYTYT.mp4"))
    parser.add_argument("--plan", type=Path, default=Path("output/v1/plan.json"))
    parser.add_argument("--out", type=Path, default=Path("research/texture"))
    parser.add_argument("--seams", type=int, nargs="+", default=[361, 2888, 3240])
    parser.add_argument("--window", type=int, default=8)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    plan = json.loads(args.plan.read_text())
    config = dict(plan["config"])
    reports = []
    for seam in args.seams:
        n = args.window
        raw = list(read_frames(args.source, seam - n, 2*n + 2, size=(640, 360)))
        geometric, geometry_report = corrected_window(raw, n, config)
        changed, texture_report = texture_transition(geometric, n, config)
        original_metrics = pair_metrics(raw[n-1], raw[n])
        geometry_metrics = pair_metrics(geometric[n-1], geometric[n])
        changed_metrics = pair_metrics(changed[n-1], changed[n])
        result = {"frame": seam, "original": original_metrics, "geometry": geometry_metrics,
                  "texture": changed_metrics, "texture_report": texture_report,
                  "geometry_report": geometry_report,
                  "relative_texture_residual_reduction": 1-changed_metrics["registered_mae"]/geometry_metrics["registered_mae"]}
        reports.append(result)
        # One row per method; before and after anchors, then aligned pair and
        # amplified difference show whether lower residual came from blur.
        sheet = Image.new("RGB", (640*4, 390*3), "#131820")
        draw = ImageDraw.Draw(sheet)
        for row, (label, frames) in enumerate((("original", raw), ("geometry", geometric),
                                               ("matched appearance", changed))):
            left, right = frames[n-1:n+1]
            aligned, confidence, _ = aligned_appearance(left, right, config)
            difference = np.abs(left.astype(float)-aligned.astype(float))
            diff = np.clip(difference*4, 0, 255).astype(np.uint8)
            for col, (tag, image) in enumerate((("left", left), ("right", right),
                                                ("aligned right", aligned), ("difference x4", diff))):
                sheet.paste(Image.fromarray(image), (col*640, row*390))
                draw.text((col*640+8, row*390+365), f"{seam} | {label} | {tag}", fill="white")
        sheet.save(args.out/f"seam-{seam}.jpg", quality=95)
        # Zoom crops show true pixel contours at 2x without smoothing.
        crop_sheet = Image.new("RGB", (480*4, 340*2), "#131820")
        draw = ImageDraw.Draw(crop_sheet)
        regions = [(200, 100, 440, 260), (40, 35, 280, 195)]
        for row, box in enumerate(regions):
            for col, (tag, image) in enumerate((("geometry left", geometric[n-1]),
                                               ("geometry right", geometric[n]),
                                               ("appearance left", changed[n-1]),
                                               ("appearance right", changed[n]))):
                crop = Image.fromarray(image).crop(box).resize((480,320), Image.Resampling.NEAREST)
                crop_sheet.paste(crop, (col*480,row*340))
                draw.text((col*480+6,row*340+324),tag,fill="white")
        crop_sheet.save(args.out/f"seam-{seam}-crops.png")
        with VideoWriter(args.out/f"seam-{seam}-comparison.mp4", 1280, 360,
                         plan["source"]["fps_fraction"], crf=16, preset="fast") as writer:
            for _ in range(3):
                for a, b in zip(geometric, changed):
                    writer.write(np.concatenate([a,b], axis=1))
        print(f"{seam}: geometry residual={geometry_metrics['registered_mae']:.3f}; "
              f"texture={changed_metrics['registered_mae']:.3f}; "
              f"gain={result['relative_texture_residual_reduction']:.1%}; "
              f"sharpness floor={texture_report['minimum_sharpness_ratio']:.3f}; "
              f"alpha={[round(r['mean_alpha'],3) for r in texture_report['anchors']]}", flush=True)
    (args.out/"report.json").write_text(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
