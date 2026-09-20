"""Experimental motion-aligned appearance transition with contour rejection.

This module deliberately makes no claim of reconstructing changed drawings.
Opposite-side appearance is admitted only where bidirectional flow and local
contours agree. Unmatched outlines keep their original pixels, preventing an
ordinary crossfade from creating two visible copies of a line.
"""
from __future__ import annotations

import cv2
import numpy as np

from .repair import flow, grid, sample, smoothstep


def _gray_float(rgb):
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)


def _gradient(gray):
    return np.stack((cv2.Sobel(gray, cv2.CV_32F, 1, 0) / 8,
                     cv2.Sobel(gray, cv2.CV_32F, 0, 1) / 8), axis=-1)


def _sharpness(rgb):
    gray = _gray_float(rgb)
    return float(np.mean(np.abs(cv2.Laplacian(gray, cv2.CV_32F))))


def aligned_appearance(current, reference, config=None):
    """Return reference warped to current, and a conservative confidence mask.

    The RGB arrays must have identical dimensions. Confidence is in current
    coordinates and includes geometric bounds, bidirectional consistency,
    local correlation, and contour agreement. The report measures admitted
    appearance; it does not certify perceptual quality.
    """
    config = config or {}
    if current.shape != reference.shape:
        raise ValueError("Appearance frames must have matching RGB dimensions")
    forward, backward = flow(current, reference), flow(reference, current)
    aligned = sample(reference, forward, cv2.INTER_CUBIC)
    back_error = np.linalg.norm(forward + sample(backward, forward), axis=2)
    coords = grid(current.shape) + forward
    h, w = current.shape[:2]
    inside = ((coords[..., 0] >= 2) & (coords[..., 0] < w - 3) &
              (coords[..., 1] >= 2) & (coords[..., 1] < h - 3))
    geometry = np.exp(-np.square(back_error / config.get("texture_flow_tolerance", 1.25)))
    a, b = _gray_float(current), _gray_float(aligned)
    ga, gb = _gradient(a), _gradient(b)
    ma, mb = np.linalg.norm(ga, axis=-1), np.linalg.norm(gb, axis=-1)
    edge_threshold = config.get("texture_edge_threshold", 7.0)
    ea, eb = ma > edge_threshold, mb > edge_threshold
    da = cv2.distanceTransform((~ea).astype(np.uint8), cv2.DIST_L2, 5)
    db = cv2.distanceTransform((~eb).astype(np.uint8), cv2.DIST_L2, 5)
    edge_distance = config.get("texture_edge_distance", 0.9)
    unmatched = (ea & (db > edge_distance)) | (eb & (da > edge_distance))
    cosine = np.sum(ga * gb, axis=-1) / np.maximum(ma * mb, 1e-4)
    conflicting = (ma > 4) & (mb > 4) & (cosine < 0.55)
    rejected_edges = cv2.dilate((unmatched | conflicting).astype(np.uint8),
                                np.ones((3, 3), np.uint8)).astype(bool)
    # Local NCC rejects changed texture while tolerating compatible grade drift.
    mean_a = cv2.boxFilter(a, -1, (7, 7))
    mean_b = cv2.boxFilter(b, -1, (7, 7))
    var_a = np.maximum(0, cv2.boxFilter(a*a, -1, (7, 7)) - mean_a*mean_a)
    var_b = np.maximum(0, cv2.boxFilter(b*b, -1, (7, 7)) - mean_b*mean_b)
    covariance = cv2.boxFilter(a*b, -1, (7, 7)) - mean_a*mean_b
    correlation = covariance / np.maximum(np.sqrt(var_a*var_b), 1e-4)
    correlation_gate = np.clip((correlation - 0.35) / 0.55, 0, 1)
    # Homogeneous paint areas lack gradient evidence but can safely share a
    # bounded colour adjustment when geometric consistency remains strong.
    flat_both = (var_a < 12) & (var_b < 12)
    correlation_gate[flat_both] = 1
    low_error = np.mean(np.abs(cv2.GaussianBlur(current.astype(np.float32), (0, 0), 2) -
                               cv2.GaussianBlur(aligned.astype(np.float32), (0, 0), 2)), axis=-1)
    appearance = np.exp(-np.square(low_error / config.get("texture_color_tolerance", 45)))
    confidence = geometry * correlation_gate * appearance * inside
    confidence[rejected_edges] = 0
    # An eroded thresholded support prevents low-confidence mixtures along a
    # disocclusion, where a soft blend itself could expose a ghost contour.
    support = (confidence > config.get("texture_min_confidence", 0.35)).astype(np.uint8)
    support = cv2.erode(support, np.ones((3, 3), np.uint8))
    confidence *= support
    confidence = confidence.astype(np.float32)
    report = {
        "accepted_fraction": float(np.mean(confidence > 0)),
        "mean_confidence": float(confidence.mean()),
        "rejected_edge_fraction": float(rejected_edges.mean()),
        "flow_consistent_fraction": float(np.mean((back_error < 1.25) & inside)),
        "aligned_mae": float(np.mean(np.abs(current.astype(float) - aligned.astype(float)))),
    }
    return aligned, confidence, report


def texture_transition(frames, cut, config=None):
    """Blend matched appearance across a corrected RGB window.

    Args:
        frames: RGB uint8 frame sequence, preferably after geometry correction.
        cut: Index of the first frame on the right side.
        config: Optional controls, including ``texture_strength`` (default .5).

    Returns:
        (list of RGB uint8 frames, JSON-serializable diagnostics).

    Each current frame keeps its own coordinate geometry and time sample;
    opposite-anchor appearance is independently motion-aligned to that frame.
    The first and last window frames remain unchanged.
    """
    config = config or {}
    if cut < 1 or cut >= len(frames):
        raise ValueError("cut must separate two nonempty frame sequences")
    strength = float(config.get("texture_strength", 0.5))
    if not 0 <= strength <= 0.5:
        raise ValueError("texture_strength must be between 0 and 0.5")
    output, reports = [], []
    for i, frame in enumerate(frames):
        if i < cut:
            reference = frames[cut]
            envelope = float(smoothstep(i / max(cut - 1, 1)))
        else:
            reference = frames[cut - 1]
            envelope = float(smoothstep((len(frames) - 1 - i) / max(len(frames) - cut - 1, 1)))
        weight = strength * envelope
        if weight == 0:
            output.append(frame.copy())
            reports.append({"index": i, "weight": 0.0, "sharpness_ratio": 1.0})
            continue
        aligned, confidence, report = aligned_appearance(frame, reference, config)
        alpha = (weight * confidence)[..., None]
        changed = np.clip(frame.astype(np.float32) * (1 - alpha) +
                          aligned.astype(np.float32) * alpha, 0, 255).round().astype(np.uint8)
        output.append(changed)
        report.update(index=i, weight=weight, mean_alpha=float(alpha.mean()),
                      sharpness_ratio=_sharpness(changed) / max(_sharpness(frame), 1e-5))
        reports.append(report)
    anchor_reports = [reports[cut - 1], reports[cut]]
    return output, {
        "method": "bidirectional motion-aligned appearance with contour rejection",
        "experimental": True, "requires_visual_review": True,
        "texture_strength": strength, "anchors": anchor_reports,
        "minimum_sharpness_ratio": min(r["sharpness_ratio"] for r in reports),
        "frames": reports,
    }
