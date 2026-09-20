"""Measured correspondence between adjacent frames of an animated shot.

Every returned matrix maps *right/source* pixel coordinates into the
*left/reference* frame. Colour transforms likewise map right RGB into left
RGB, in the 0..255 domain. No image synthesis takes place in this module.

Registration is intentionally allowed to fail: a moving character, a blank
background, or a changed drawing does not establish a reliable camera move.
"""

from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np


def _rgb8(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("Expected an H×W×3 RGB image")
    if not np.all(np.isfinite(array)):
        raise ValueError("Image contains nonfinite values")
    if array.dtype == np.uint8:
        return np.ascontiguousarray(array)
    if np.issubdtype(array.dtype, np.floating) and array.size and array.max() <= 1.0:
        array = array * 255.0
    return np.ascontiguousarray(np.clip(np.rint(array), 0, 255).astype(np.uint8))


def _gray(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def _gradient(gray: np.ndarray) -> np.ndarray:
    smooth = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), 0.8)
    return cv2.magnitude(cv2.Sobel(smooth, cv2.CV_32F, 1, 0, ksize=3),
                         cv2.Sobel(smooth, cv2.CV_32F, 0, 1, ksize=3)) / 8.0


def _coverage(points: np.ndarray, shape: tuple[int, int]) -> float:
    """Occupied 4×4 cells; duplicate points cannot inflate support."""
    if len(points) == 0:
        return 0.0
    height, width = shape
    cells = np.clip(np.floor(points / [width, height] * 4), 0, 3).astype(int)
    return len(np.unique(cells[:, 1] * 4 + cells[:, 0])) / 16.0


def _warp(image: np.ndarray, matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    height, width = image.shape[:2]
    warped = cv2.warpAffine(image, matrix[:2].astype(np.float32), (width, height),
                            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    valid = cv2.warpAffine(np.ones((height, width), np.uint8),
                          matrix[:2].astype(np.float32), (width, height),
                          flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
    valid = cv2.erode(valid, np.ones((5, 5), np.uint8)).astype(bool)
    return warped, valid


def _fit_color(left: np.ndarray, right: np.ndarray, valid: np.ndarray) -> dict[str, Any]:
    """Robust affine RGB fit with equalised intensity-bin sampling.

    Huber-like IRLS prevents small drawing/motion differences from pulling the
    global colour correction toward the subject's new position.
    """
    y = left[valid].astype(np.float64)
    x = right[valid].astype(np.float64)
    if len(x) < 32:
        return {"gain": [1.0] * 3, "bias": [0.0] * 3, "confidence": 0.0,
                "residual_mae": None, "sample_count": int(len(x))}
    # Deterministic reduction preserves reproducibility between CLI runs.
    stride = max(1, math.ceil(len(x) / 40000))
    x, y = x[::stride], y[::stride]
    gain, bias = [], []
    for channel in range(3):
        xc, yc = x[:, channel], y[:, channel]
        spans = np.percentile(xc, [5, 95])
        if spans[1] - spans[0] < 12:
            a, b = 1.0, float(np.median(yc - xc))
        else:
            bins = np.clip((xc / 16).astype(int), 0, 15)
            counts = np.bincount(bins, minlength=16)
            base_weights = 1.0 / np.sqrt(np.maximum(counts[bins], 1))
            base_weights /= np.mean(base_weights)
            a, b = 1.0, float(np.median(yc - xc))
            design = np.column_stack((xc, np.ones_like(xc)))
            for _ in range(8):
                residual = yc - (a * xc + b)
                center = np.median(residual)
                sigma = max(1.5, 1.4826 * float(np.median(np.abs(residual - center))))
                weights = base_weights * np.minimum(1.0, 1.5 * sigma / np.maximum(np.abs(residual), 1e-6))
                root_weights = np.sqrt(weights)
                solution, *_ = np.linalg.lstsq(design * root_weights[:, None],
                                               yc * root_weights, rcond=None)
                a, b = float(solution[0]), float(solution[1])
            a = float(np.clip(a, 0.5, 2.0))
            b = float(np.clip(b, -96, 96))
        gain.append(a)
        bias.append(b)
    error = np.mean(np.abs(y - (x * gain + bias)), axis=1)
    median = float(np.median(error))
    support = float(np.mean(error < max(5.0, median * 2.5)))
    return {"gain": gain, "bias": bias,
            "confidence": float(np.clip(support * math.exp(-median / 20.0), 0, 1)),
            "residual_mae": float(np.mean(error)), "sample_count": int(len(x))}


def _measure(left: np.ndarray, warped: np.ndarray, valid: np.ndarray,
             color: dict[str, Any]) -> dict[str, float]:
    if valid.sum() < 32:
        return {"mae": 255.0, "corrected_mae": 255.0, "trimmed_corrected_mae": 255.0,
                "gradient_mae": 255.0, "gradient_ncc": 0.0, "overlap": 0.0}
    raw = np.abs(left.astype(np.float32) - warped.astype(np.float32)).mean(axis=2)[valid]
    corrected = np.clip(warped.astype(np.float32) * color["gain"] + color["bias"], 0, 255)
    errors = np.abs(left.astype(np.float32) - corrected).mean(axis=2)[valid]
    ga, gb = _gradient(_gray(left)), _gradient(_gray(warped))
    # Flat areas carry little evidence about camera motion.
    active = valid & ((ga + gb) > 4.0)
    if active.sum() >= 32:
        va, vb = ga[active].astype(float), gb[active].astype(float)
        denom = np.linalg.norm(va - va.mean()) * np.linalg.norm(vb - vb.mean())
        ncc = float(np.dot(va - va.mean(), vb - vb.mean()) / max(denom, 1e-9))
        gradient_error = float(np.mean(np.abs(va - vb)))
    else:
        ncc, gradient_error = 0.0, 0.0
    trimmed = errors[errors <= np.percentile(errors, 85)]
    return {"mae": float(np.mean(raw)), "corrected_mae": float(np.mean(errors)),
            "trimmed_corrected_mae": float(np.mean(trimmed)),
            "gradient_mae": gradient_error, "gradient_ncc": ncc,
            "overlap": float(np.mean(valid))}


def _plausible(matrix: np.ndarray, width: int, height: int) -> bool:
    if not np.all(np.isfinite(matrix)) or np.linalg.det(matrix[:2, :2]) <= 0:
        return False
    singular = np.linalg.svd(matrix[:2, :2], compute_uv=False)
    angle = abs(math.degrees(math.atan2(matrix[1, 0], matrix[0, 0])))
    return bool(0.72 < min(singular) and max(singular) < 1.4 and
                max(singular) / min(singular) < 1.08 and angle < 18 and
                abs(matrix[0, 2]) < width * 0.4 and abs(matrix[1, 2]) < height * 0.4)


def register_pair(left_rgb: np.ndarray, right_rgb: np.ndarray,
                  max_width: int = 640) -> dict[str, Any]:
    """Estimate camera geometry and colour from two RGB frames.

    ``matrix`` is a nested-list 3×3 transform in original image coordinates,
    suitable for ``cv2.warpAffine(right, matrix[:2], ...)`` without
    ``WARP_INVERSE_MAP``. Geometry is reliable only when ``reliable`` is true;
    callers should preserve/review low-confidence seams instead of assuming
    a failed estimate is evidence that no correction is needed.

    ``confidence`` measures registration evidence, not perceptual seamlessness.
    Residual metrics use 0..255 RGB/gradient units.
    """
    left, right = _rgb8(left_rgb), _rgb8(right_rgb)
    if left.shape != right.shape:
        raise ValueError("Both frames must have the same dimensions")
    if max_width < 64:
        raise ValueError("max_width must be at least 64")
    original_height, original_width = left.shape[:2]
    scale = min(1.0, float(max_width) / original_width)
    if scale < 1:
        shape = (int(round(original_width * scale)), int(round(original_height * scale)))
        left = cv2.resize(left, shape, interpolation=cv2.INTER_AREA)
        right = cv2.resize(right, shape, interpolation=cv2.INTER_AREA)
    height, width = left.shape[:2]
    # Actual x/y scales include integer resize rounding.
    downscale = np.diag([width / original_width, height / original_height, 1.0])
    lg, rg = _gray(left), _gray(right)
    texture_fraction = float(np.mean(_gradient(lg) > 2.0))
    identity = np.eye(3, dtype=np.float64)
    base_warp, base_valid = _warp(right, identity)
    base_color = _fit_color(left, base_warp, base_valid)
    base_metrics = _measure(left, base_warp, base_valid, base_color)
    candidates: list[dict[str, Any]] = []
    diagnostics: dict[str, Any] = {"keypoints_left": 0, "keypoints_right": 0,
                                  "matches": 0, "inliers": 0,
                                  "coverage": 0.0, "texture_fraction": texture_fraction,
                                  "warnings": []}
    feature_matrix = None
    source_points = target_points = np.empty((0, 2), np.float32)
    inlier_mask = np.zeros(0, bool)
    if texture_fraction >= 0.005:
        sift = cv2.SIFT_create(nfeatures=3500, contrastThreshold=0.015, edgeThreshold=12)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lk, ld = sift.detectAndCompute(clahe.apply(lg), None)
        rk, rd = sift.detectAndCompute(clahe.apply(rg), None)
        diagnostics.update(keypoints_left=len(lk), keypoints_right=len(rk))
        if ld is not None and rd is not None and len(ld) >= 4 and len(rd) >= 4:
            matcher = cv2.BFMatcher(cv2.NORM_L2)
            matches_lr = matcher.knnMatch(ld, rd, k=2)
            matches_rl = matcher.knnMatch(rd, ld, k=2)
            reverse = {m.queryIdx: m.trainIdx for pair in matches_rl if len(pair) == 2
                       for m, n in [pair] if m.distance < 0.80 * n.distance}
            matches = [m for pair in matches_lr if len(pair) == 2
                       for m, n in [pair] if m.distance < 0.80 * n.distance
                       and reverse.get(m.trainIdx) == m.queryIdx]
            # Large textured characters must not overwhelm sparse background.
            cells: dict[tuple[int, int], int] = {}
            balanced = []
            for match in sorted(matches, key=lambda m: m.distance):
                px, py = lk[match.queryIdx].pt
                cell = (min(5, int(px / width * 6)), min(3, int(py / height * 4)))
                if cells.get(cell, 0) < 16:
                    balanced.append(match)
                    cells[cell] = cells.get(cell, 0) + 1
            diagnostics["matches"] = len(balanced)
            if len(balanced) >= 4:
                target_points = np.float32([lk[m.queryIdx].pt for m in balanced])
                source_points = np.float32([rk[m.trainIdx].pt for m in balanced])
                affine, mask = cv2.estimateAffinePartial2D(
                    source_points, target_points, method=cv2.RANSAC,
                    ransacReprojThreshold=2.25, maxIters=6000, confidence=0.999,
                    refineIters=30)
                if affine is not None and mask is not None:
                    inlier_mask = mask[:, 0].astype(bool)
                    feature_matrix = np.vstack([affine, [0.0, 0.0, 1.0]])
                    coverage = _coverage(target_points[inlier_mask], (height, width))
                    predicted = source_points @ affine[:, :2].T + affine[:, 2]
                    errors = np.linalg.norm(predicted - target_points, axis=1)
                    diagnostics.update(inliers=int(inlier_mask.sum()), coverage=coverage,
                                       inlier_ratio=float(inlier_mask.mean()),
                                       reprojection_median=float(np.median(errors[inlier_mask])))
                    if _plausible(feature_matrix, width, height):
                        candidates.append({"name": "sift_similarity", "matrix": feature_matrix})
                    else:
                        diagnostics["warnings"].append("Feature transform exceeds supported camera-change bounds")
                        feature_matrix = None
    if texture_fraction < 0.005:
        diagnostics["warnings"].append("Insufficient image texture to determine camera geometry")
    # ECC is a candidate refinement, not an unconditional replacement for the
    # robust spatially distributed feature model. Low-frequency luminance helps
    # with line-art changes and small generation grain differences.
    if texture_fraction >= 0.015:
        template = cv2.GaussianBlur(lg.astype(np.float32) / 255.0, (0, 0), 1.3)
        moving = cv2.GaussianBlur(rg.astype(np.float32) / 255.0, (0, 0), 1.3)
        seed = feature_matrix if feature_matrix is not None else identity
        try:
            inverse_seed = np.linalg.inv(seed)[:2].astype(np.float32)
            correlation, inverse_warp = cv2.findTransformECC(
                template, moving, inverse_seed, cv2.MOTION_AFFINE,
                (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 80, 1e-5),
                None, 5)
            refined = np.linalg.inv(np.vstack([inverse_warp, [0.0, 0.0, 1.0]]))
            if _plausible(refined, width, height):
                feature_disagreement = 0.0
                if feature_matrix is not None and inlier_mask.any():
                    pts = source_points[inlier_mask]
                    one = pts @ seed[:2, :2].T + seed[:2, 2]
                    two = pts @ refined[:2, :2].T + refined[:2, 2]
                    feature_disagreement = float(np.median(np.linalg.norm(one - two, axis=1)))
                # ECC that moves distributed matched points >2.5px is usually
                # following a changed foreground shape, not the camera.
                if feature_disagreement <= 2.5:
                    candidates.append({"name": "ecc_affine", "matrix": refined,
                                       "ecc_correlation": float(correlation),
                                       "feature_disagreement": feature_disagreement})
        except (cv2.error, np.linalg.LinAlgError):
            diagnostics["warnings"].append("ECC refinement did not converge")

    best = {"name": "identity", "matrix": identity, "color": base_color,
            "metrics": base_metrics, "score": base_metrics["trimmed_corrected_mae"] +
            0.4 * base_metrics["gradient_mae"]}
    for candidate in candidates:
        warped, valid = _warp(right, candidate["matrix"])
        color = _fit_color(left, warped, valid)
        # Compare each candidate against identity on the *same* valid pixels,
        # so it cannot win merely by cropping away changed content.
        fair = valid & base_valid
        metrics = _measure(left, warped, fair, color)
        baseline = _measure(left, base_warp, fair, base_color)
        score = metrics["trimmed_corrected_mae"] + 0.4 * metrics["gradient_mae"]
        base_score = baseline["trimmed_corrected_mae"] + 0.4 * baseline["gradient_mae"]
        improvement = (base_score - score) / max(base_score, 0.5)
        candidate.update(color=color, metrics=metrics, score=score,
                         improvement=improvement)
        if improvement > 0.005 and score < best["score"]:
            best = candidate
    matches = diagnostics["matches"]
    inliers = diagnostics["inliers"]
    coverage = diagnostics["coverage"]
    inlier_ratio = inliers / max(matches, 1)
    feature_confidence = (min(1.0, inliers / 30.0) * min(1.0, coverage / 0.55) *
                          min(1.0, inlier_ratio / 0.70))
    similarity = max(0.0, best["metrics"]["gradient_ncc"])
    if feature_matrix is not None:
        confidence = feature_confidence * (0.55 + 0.45 * similarity)
    else:
        # Global ECC alone cannot distinguish foreground motion from camera
        # motion. Keep it explicitly below the automatic-repair threshold.
        confidence = min(0.45, similarity * min(1.0, texture_fraction / 0.2) * 0.5)
    if best["name"] == "identity" and feature_matrix is not None:
        # If features strongly disagree with identity but its residual won,
        # neither provides adequate evidence of a global camera transform.
        corners = np.array([[0, 0], [width, 0], [0, height], [width, height]], float)
        displacement = np.linalg.norm(corners @ feature_matrix[:2, :2].T +
                                      feature_matrix[:2, 2] - corners, axis=1)
        if np.median(displacement) > 2.0:
            confidence *= 0.5
            diagnostics["warnings"].append("Feature geometry and whole-image evidence disagree")
    reliable = bool(confidence >= 0.55 and inliers >= 10 and coverage >= 0.3125
                    and best["metrics"]["overlap"] >= 0.70)
    if not reliable:
        diagnostics["warnings"].append("Geometry requires review; automatic acceptance threshold not met")
    full_matrix = np.linalg.inv(downscale) @ best["matrix"] @ downscale
    diagnostics["candidates"] = [{"method": c["name"],
                                  "improvement": c.get("improvement", 0.0),
                                  "gradient_ncc": c.get("metrics", {}).get("gradient_ncc"),
                                  "ecc_correlation": c.get("ecc_correlation")}
                                 for c in candidates]
    return {"matrix": full_matrix.tolist(), "confidence": float(np.clip(confidence, 0, 1)),
            "reliable": reliable, "method": best["name"], "color": best["color"],
            "metrics": {"before": base_metrics, "after": best["metrics"],
                        "improvement": float(best.get("improvement", 0.0))},
            "diagnostics": diagnostics}


def dense_correspondence(left_rgb: np.ndarray, right_rgb: np.ndarray,
                         max_width: int = 960) -> dict[str, np.ndarray | float]:
    """Bidirectional DIS flow with forward/backward and photometric checks.

    Returned ``forward[y,x]`` samples right at ``(x,y)+forward[y,x]``.
    ``backward`` samples left from right coordinates. ``valid`` and
    ``confidence`` are in left coordinates. Flows are restored to input
    resolution; invalid/occluded regions must not be blindly synthesized.
    """
    left, right = _rgb8(left_rgb), _rgb8(right_rgb)
    if left.shape != right.shape:
        raise ValueError("Both frames must have the same dimensions")
    if max_width < 64:
        raise ValueError("max_width must be at least 64")
    original_height, original_width = left.shape[:2]
    factor = min(1.0, max_width / original_width)
    width, height = round(original_width * factor), round(original_height * factor)
    if factor < 1:
        left = cv2.resize(left, (width, height), interpolation=cv2.INTER_AREA)
        right = cv2.resize(right, (width, height), interpolation=cv2.INTER_AREA)
    else:
        height, width = left.shape[:2]
    lg, rg = _gray(left), _gray(right)
    estimator = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    estimator.setUseSpatialPropagation(True)
    estimator.setVariationalRefinementIterations(10)
    forward = estimator.calc(lg, rg, None)
    backward = estimator.calc(rg, lg, None)
    yy, xx = np.mgrid[:height, :width].astype(np.float32)
    sx, sy = xx + forward[..., 0], yy + forward[..., 1]
    reverse = cv2.remap(backward, sx, sy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    consistency_error = np.linalg.norm(forward + reverse, axis=2)
    flow_size = np.linalg.norm(forward, axis=2) + np.linalg.norm(reverse, axis=2)
    tolerance = 0.6 + 0.05 * flow_size
    inside = (sx >= 1) & (sx < width - 2) & (sy >= 1) & (sy < height - 2)
    reconstructed = cv2.remap(right, sx, sy, cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REFLECT101)
    # Account for global colour grading before judging flow correspondences.
    color = _fit_color(left, reconstructed, inside & (consistency_error < tolerance))
    corrected = reconstructed.astype(np.float32) * color["gain"] + color["bias"]
    photo_error = np.mean(np.abs(left.astype(np.float32) - corrected), axis=2)
    confidence = (np.exp(-np.square(consistency_error / np.maximum(tolerance, 0.1))) *
                  np.exp(-photo_error / 15.0) * inside).astype(np.float32)
    valid = inside & (consistency_error < tolerance) & (photo_error < 25.0)
    if factor < 1:
        resize = (original_width, original_height)
        for flow in (forward, backward):
            flow[..., 0] *= original_width / width
            flow[..., 1] *= original_height / height
        forward = cv2.resize(forward, resize, interpolation=cv2.INTER_LINEAR)
        backward = cv2.resize(backward, resize, interpolation=cv2.INTER_LINEAR)
        confidence = cv2.resize(confidence, resize, interpolation=cv2.INTER_LINEAR)
        consistency_error = cv2.resize(consistency_error / factor, resize,
                                       interpolation=cv2.INTER_LINEAR)
        valid = cv2.resize(valid.astype(np.uint8), resize,
                           interpolation=cv2.INTER_NEAREST).astype(bool)
    return {"forward": forward, "backward": backward, "valid": valid,
            "confidence": confidence, "consistency_error": consistency_error,
            "valid_fraction": float(valid.mean())}
