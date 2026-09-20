"""Audit dense seam correction fields without changing or rendering the video.

Usage: .venv/bin/python research/evaluate_fields.py --plan output/v1/plan.json
Produces reproducible geometry/uncertainty measurements for every enabled seam.
Negative gather-map Jacobians indicate folds, not merely low optical-flow confidence.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from seamstress.media import probe, read_frames
from seamstress.repair import build_fields, flow, grid, sample


def quantiles(array):
    return {label: float(np.percentile(array, percentile)) for label, percentile in
            (("min", 0), ("p0_1", 0.1), ("p1", 1), ("p5", 5),
             ("p50", 50), ("p95", 95), ("p99", 99), ("max", 100))}


def jacobian_report(displacement):
    dy, dx = np.gradient(displacement, axis=(0, 1))
    # q(x,y)=(x,y)+d(x,y) is the gather transform used by cv2.remap.
    a, b, c, d = 1 + dx[..., 0], dy[..., 0], dx[..., 1], 1 + dy[..., 1]
    determinant = a * d - b * c
    frobenius_squared = a*a + b*b + c*c + d*d
    discriminant = np.maximum(0, frobenius_squared**2 - 4 * determinant**2)
    singular_min = np.sqrt(np.maximum(0, (frobenius_squared - np.sqrt(discriminant))/2))
    singular_max = np.sqrt(np.maximum(0, (frobenius_squared + np.sqrt(discriminant))/2))
    # Check the entire fade, not just weight=1. A positive endpoint determinant
    # can still hide an intermediate fold if both eigenvalues become negative.
    trace = dx[..., 0] + dy[..., 1]
    det_gradient = dx[..., 0] * dy[..., 1] - dy[..., 0] * dx[..., 1]
    optimum_weight = np.clip(-trace / np.where(abs(det_gradient) > 1e-9,
                                              2*det_gradient, 1e-9), 0, 1)
    minimum = np.minimum(1, determinant)
    stationary = 1 + trace * optimum_weight + det_gradient * optimum_weight**2
    minimum = np.where(det_gradient > 0, np.minimum(minimum, stationary), minimum)
    coords = grid(displacement.shape) + displacement
    height, width = displacement.shape[:2]
    outside = (coords[..., 0] < 0) | (coords[..., 0] > width - 1) | (
        coords[..., 1] < 0) | (coords[..., 1] > height - 1)
    return {
        "determinant": quantiles(determinant),
        "fold_fraction": float(np.mean(determinant <= 0)),
        "severe_compression_fraction": float(np.mean(determinant < 0.25)),
        "fade_min_determinant": float(minimum.min()),
        "fade_fold_fraction": float(np.mean(minimum <= 0)),
        "singular_min": quantiles(singular_min),
        "singular_max": quantiles(singular_max),
        "out_of_bounds_fraction": float(outside.mean()),
    }


def evaluate(source, seams, config, size):
    results = []
    for seam in seams:
        frame = seam["frame"]
        # Same four-frame velocity estimator as the repair, with enough context
        # on both sides. Anchor fields do not depend on the larger fade window.
        frames = read_frames(source, frame - 5, 11, size=size)
        cut = 5
        fields = build_fields(frames, cut, config)
        a, b = frames[cut - 1], frames[cut]
        forward, backward = fields["forward"], fields["backward"]
        fb = np.linalg.norm(forward + sample(backward, forward), axis=2)
        bf = np.linalg.norm(backward + sample(forward, backward), axis=2)
        coords = grid(forward.shape) + forward
        height, width = a.shape[:2]
        inside = ((coords[..., 0] >= 0) & (coords[..., 0] < width - 1) &
                  (coords[..., 1] >= 0) & (coords[..., 1] < height - 1))
        gray = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY).astype(np.float32)
        grad = np.hypot(cv2.Sobel(gray, cv2.CV_32F, 1, 0),
                        cv2.Sobel(gray, cv2.CV_32F, 0, 1)) / 8
        edges = grad > 5
        transported_bf = sample(bf, forward)
        # A valid mask has to compare evidence at corresponding positions.
        valid_common = inside & (fb < 2.5) & (transported_bf < 2.5)
        invalid_native = (fb >= 2.5) | (bf >= 2.5)
        invalid_common = ~valid_common
        va = -flow(a, frames[0]) / 4
        vb = flow(b, frames[9]) / 4
        vb_left = sample(vb, forward)
        velocity_mismatch = np.linalg.norm(va - vb_left, axis=2)
        result = {
            "frame": frame,
            "left": jacobian_report(fields["left_map"] * config.get("geometry_strength", 1)),
            "right": jacobian_report(fields["right_map"] * config.get("geometry_strength", 1)),
            "uncertainty": {
                "fb_error": quantiles(fb),
                "bf_error": quantiles(bf),
                "out_of_bounds_fraction": float((~inside).mean()),
                "invalid_common_fraction": float(invalid_common.mean()),
                "invalid_edge_fraction": float(invalid_common[edges].mean()) if edges.any() else None,
                "mask_coordinate_disagreement_fraction": float(np.mean(invalid_native != invalid_common)),
            },
            "velocity_mismatch_px_per_frame": quantiles(velocity_mismatch[valid_common]),
            "left_speed_px_per_frame": quantiles(np.linalg.norm(va, axis=2)[valid_common]),
            "right_speed_px_per_frame": quantiles(np.linalg.norm(vb_left, axis=2)[valid_common]),
            "build_diagnostics": fields["diagnostics"],
        }
        results.append(result)
        print(f"{frame}: Jmin left={result['left']['determinant']['min']:.3f}, "
              f"right={result['right']['determinant']['min']:.3f}; "
              f"folds={result['left']['fold_fraction']:.3%}/{result['right']['fold_fraction']:.3%}; "
              f"uncertain={invalid_common.mean():.1%}; "
              f"velocity p50={result['velocity_mismatch_px_per_frame']['p50']:.3f}", flush=True)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=Path("output/v1/plan.json"))
    parser.add_argument("--output", type=Path, default=Path("research/diagnostics/field_audit.json"))
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    source = Path(plan["source"]["path"])
    meta = probe(source)
    config = plan["config"]
    width = min(config.get("flow_width", 640), meta["width"])
    size = (width, round(meta["height"] * width / meta["width"]))
    report = {
        "source": str(source), "plan": str(args.plan), "analysis_size": list(size),
        "description": "Anchor gather-map Jacobians, coordinate-correct flow uncertainty, and uncorrected velocity mismatch",
        "seams": evaluate(source, [s for s in plan["seams"] if s.get("enabled", True)], config, size),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
