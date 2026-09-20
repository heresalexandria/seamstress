"""RIFE appearance interpolation with a separately constrained camera path.

The model interpolates in a stabilized coordinate system. An explicit cubic
Hermite similarity transform then matches the measured camera velocities at
the two source endpoints. All image synthesis runs at input resolution.
"""
from __future__ import annotations

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt

from .registration import register_pair
from .repair import robust_color, flow, sample


def _dense_camera_registration(left, right):
    """Independent fallback from spatially balanced consistent dense matches."""
    height, width = left.shape[:2]
    size = (min(width, 640), round(height * min(width, 640) / width))
    a, b = (cv2.resize(f, size, interpolation=cv2.INTER_AREA) for f in (left, right))
    forward, backward = flow(a, b), flow(b, a)
    fb = np.linalg.norm(forward + sample(backward, forward), axis=-1)
    yy, xx = np.mgrid[4:size[1]-4:8, 4:size[0]-4:8]
    source = np.stack((xx.ravel(), yy.ravel()), axis=1).astype(np.float32)
    displacement = forward[yy, xx].reshape(-1, 2)
    target = source + displacement
    valid = ((fb[yy, xx].ravel() < 1.5) & (target[:, 0] > 2) &
             (target[:, 0] < size[0]-3) & (target[:, 1] > 2) & (target[:, 1] < size[1]-3))
    # Each fixed grid location contributes once; highly textured characters
    # cannot acquire extra votes through repeated keypoints.
    outside_subject = ((source[:, 0] < size[0]*.22) | (source[:, 0] > size[0]*.78) |
                       (source[:, 1] < size[1]*.30) | (source[:, 1] > size[1]*.90))
    candidates = []
    for label, mask in (("full_grid", valid), ("outer_background_grid", valid & outside_subject)):
        points, corresponding = source[mask], target[mask]
        if len(points) < 30:
            continue
        affine, inliers = cv2.estimateAffinePartial2D(points, corresponding, method=cv2.RANSAC,
                          ransacReprojThreshold=2, maxIters=6000, confidence=.999, refineIters=30)
        if affine is None:
            continue
        inliers = inliers[:, 0].astype(bool)
        cells = np.clip((points[inliers] / size * 4).astype(int), 0, 3)
        coverage = len(np.unique(cells[:, 1]*4+cells[:, 0])) / 16
        residual = np.linalg.norm(points @ affine[:, :2].T + affine[:, 2] - corresponding, axis=1)
        error = float(np.median(residual[inliers]))
        ratio = float(inliers.mean())
        confidence = min(1., coverage/.625) * min(1., ratio/.65) * np.exp(-error/4)
        scale = np.sqrt(np.linalg.det(affine[:, :2]))
        if not .75 < scale < 1.4:
            continue
        down = np.diag([size[0]/width, size[1]/height, 1])
        full = np.linalg.inv(down) @ np.vstack((affine, [0, 0, 1])) @ down
        candidates.append({"matrix": np.linalg.inv(full).tolist(), "confidence": float(confidence),
                           "method": "dense_"+label, "inlier_ratio": ratio, "coverage": coverage,
                           "median_fit_residual_analysis_pixels": error,
                           "consistent_grid_points": int(len(points)), "inliers": int(inliers.sum())})
    if not candidates:
        return None
    return max(candidates, key=lambda item: item["confidence"])


def _camera_registration(left, right):
    registration = register_pair(left, right)
    if registration["confidence"] < .25:
        dense = _dense_camera_registration(left, right)
        if dense is not None and dense["confidence"] > registration["confidence"]:
            dense["initial_feature_confidence"] = registration["confidence"]
            return dense
    return registration


def _similarity(matrix, center):
    matrix = np.asarray(matrix, dtype=np.float64)
    original_center = matrix[:2, :2] @ center + matrix[:2, 2]
    u, singular, vh = np.linalg.svd(matrix[:2, :2])
    rotation = u @ vh
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vh
    result = matrix.copy()
    result[:2, :2] = rotation * singular.mean()
    result[:2, 2] = original_center - result[:2, :2] @ center
    return result


def _parameters(matrix, center):
    a = matrix[:2, :2]
    return np.array([np.log(np.sqrt(np.linalg.det(a))), np.arctan2(a[1, 0], a[0, 0]),
                     *(a @ center + matrix[:2, 2] - center)])


def _transform(parameters, center):
    scale, angle = np.exp(parameters[0]), parameters[1]
    a = scale * np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    matrix = np.eye(3)
    matrix[:2, :2] = a
    matrix[:2, 2] = center + parameters[2:] - a @ center
    return matrix


def synthesize_camera_bridge(model, left, right, pre_frames, post_frames, progress):
    """Return native-resolution bridge frames and diagnostics.

    ``model.synthesize(left_rgb, right_rgb, progress)`` must return one RGB
    uint8 frame for every progress value, including both unchanged endpoints.
    ``pre_frames`` contains chronological source frames immediately before
    left; ``post_frames`` contains chronological frames immediately after
    right. Both exclude the endpoints. Up to four handles per side are used.
    ``progress`` controls learned appearance interpolation. Camera movement is
    evaluated on the actual uniformly spaced source-frame times instead.

    Camera paths exposing substantial unseen areas are rejected. At most a
    four-pixel source-edge extension is allowed for tiny corner gaps, and its
    extent is reported. No reflected border or global crop is introduced.
    """
    left, right = np.asarray(left), np.asarray(right)
    if left.shape != right.shape or left.ndim != 3 or left.shape[2] != 3:
        raise ValueError("Camera bridge endpoints must have matching RGB dimensions")
    if left.dtype != np.uint8 or right.dtype != np.uint8:
        raise ValueError("Camera bridge endpoints must be RGB uint8")
    progress = np.asarray(progress, dtype=float)
    if (progress.ndim != 1 or len(progress) < 3 or not np.all(np.isfinite(progress)) or
            progress[0] != 0 or progress[-1] != 1 or np.any(np.diff(progress) <= 0)):
        raise ValueError("Progress must increase strictly from zero to one with at least three samples")
    before = list(pre_frames)[-4:]
    after = list(post_frames)[:4]
    if not before or not after or any(np.asarray(f).shape != left.shape for f in before + after):
        raise ValueError("Provide matching source-frame handles on both sides of the bridge")
    height, width = left.shape[:2]
    center = np.array([(width - 1) / 2, (height - 1) / 2])
    endpoint_registration = _camera_registration(left, right)
    previous_registration = _camera_registration(left, before[0])
    following_registration = _camera_registration(right, after[-1])
    if endpoint_registration["confidence"] < 0.25:
        raise ValueError("Endpoint camera registration is unreliable; choose better source anchors")
    final = _similarity(np.linalg.inv(np.array(endpoint_registration["matrix"])), center)
    previous = _similarity(np.linalg.inv(np.array(previous_registration["matrix"])), center)
    following = _similarity(np.linalg.inv(np.array(following_registration["matrix"])), center) @ final
    p1 = _parameters(final, center)
    v0 = -_parameters(previous, center) / len(before)
    v1 = (_parameters(following, center) - p1) / len(after)
    # Camera rates with no matching evidence must not manufacture acceleration.
    warnings = []
    if previous_registration["confidence"] < 0.25:
        v0 = p1 / (len(progress) - 1)
        warnings.append("Incoming rate lacked evidence; used endpoint-average camera rate")
    if following_registration["confidence"] < 0.25:
        v1 = p1 / (len(progress) - 1)
        warnings.append("Outgoing rate lacked evidence; used endpoint-average camera rate")
    corners = np.array([[0, 0], [width - 1, 0], [0, height - 1], [width - 1, height - 1]], float)
    inverse_final = np.linalg.inv(final)
    right_corners = corners @ inverse_final[:2, :2].T + inverse_final[:2, 2]
    inverse_previous, inverse_following = np.linalg.inv(previous), np.linalg.inv(following)
    previous_corners = corners @ inverse_previous[:2, :2].T + inverse_previous[:2, 2]
    following_corners = corners @ inverse_following[:2, :2].T + inverse_following[:2, 2]
    all_corners = np.concatenate((corners, right_corners, previous_corners, following_corners))
    minimum = np.floor(all_corners.min(axis=0)) - 8
    maximum = np.ceil(all_corners.max(axis=0)) + 8
    canvas_size = tuple((maximum - minimum + 1).astype(int))
    if canvas_size[0] * canvas_size[1] > 2.5 * width * height:
        raise ValueError("Bridge requires excessive common-view canvas; inspect camera registration")
    offset = np.eye(3)
    offset[:2, 2] = -minimum

    def common_warp(image, matrix, interpolation=cv2.INTER_CUBIC):
        return cv2.warpAffine(image, matrix[:2].astype(np.float32), canvas_size,
                              flags=interpolation, borderMode=cv2.BORDER_CONSTANT)

    left_common = common_warp(left, offset).astype(np.float32)
    right_common = common_warp(right, offset @ inverse_final).astype(np.float32)
    ones = np.ones((height, width), dtype=np.uint8)
    valid_left = common_warp(ones, offset, cv2.INTER_NEAREST) > 0
    valid_right = common_warp(ones, offset @ inverse_final, cv2.INTER_NEAREST) > 0
    gain, bias = robust_color(right_common, left_common, valid_left & valid_right)
    right_common = np.clip(right_common * gain + bias, 0, 255)
    left_common[~valid_left & valid_right] = right_common[~valid_left & valid_right]
    right_common[~valid_right & valid_left] = left_common[~valid_right & valid_left]
    valid_union = valid_left | valid_right
    endpoint_valid_union = valid_union.copy()
    # Real handle frames supply corners newly exposed by a curved camera path.
    # Their transforms were already estimated for the endpoint velocities.
    # This retains the full camera trajectory without reflecting image edges.
    for context, matrix, context_gain, context_bias in (
        (before[0], inverse_previous, np.ones(3), np.zeros(3)),
        (after[-1], inverse_following, gain, bias),
    ):
        context_common = common_warp(context, offset @ matrix).astype(np.float32)
        context_common = np.clip(context_common * context_gain + context_bias, 0, 255)
        valid_context = common_warp(ones, offset @ matrix, cv2.INTER_NEAREST) > 0
        added = ~valid_union & valid_context
        left_common[added] = context_common[added]
        right_common[added] = context_common[added]
        valid_union |= valid_context
    # Extend only model context with the nearest real source edge. Output
    # coverage below separately bounds visible extrapolation to four pixels
    # and no more than 0.1% of the native frame area.
    nearest = distance_transform_edt(~valid_union, return_distances=False, return_indices=True)
    left_common[~valid_union] = left_common[tuple(nearest[:, ~valid_union])]
    right_common[~valid_union] = right_common[tuple(nearest[:, ~valid_union])]
    inverse_offset = np.linalg.inv(offset)
    span = len(progress) - 1
    camera_matrices, camera_parameters, coverage, edge_extensions = [], [], [], []
    # Validate coverage before calling the model, avoiding expensive synthesis
    # of a camera path that cannot retain the full native source frame.
    original_v0, original_v1 = v0.copy(), v1.copy()
    average_rate = p1 / span
    last_failure = None
    for rate_retention in (1., .875, .75, .625, .5, .375, .25, .125, 0.):
        v0 = average_rate + rate_retention * (original_v0 - average_rate)
        v1 = average_rate + rate_retention * (original_v1 - average_rate)
        camera_matrices, camera_parameters, coverage, edge_extensions = [], [], [], []
        failed = False
        for i in range(1, span):
            u = i / span
            p = ((-2*u**3 + 3*u*u) * p1 + (u**3 - 2*u*u + u) * span * v0 +
                 (u**3 - u*u) * span * v1)
            matrix = _transform(p, center) @ inverse_offset
            valid = cv2.warpAffine(valid_union.astype(np.uint8), matrix[:2].astype(np.float32),
                                   (width, height), flags=cv2.INTER_NEAREST,
                                   borderMode=cv2.BORDER_CONSTANT) > 0
            missing = float(np.mean(~valid))
            if missing > 0:
                depth = cv2.distanceTransform((~valid).astype(np.uint8), cv2.DIST_L2, 5)
                maximum_depth = float(depth.max())
                if maximum_depth > 4.01 or missing > .001:
                    last_failure = (missing, maximum_depth, i)
                    failed = True
                    break
            else:
                maximum_depth = 0.
            camera_matrices.append(matrix)
            camera_parameters.append(p.tolist())
            coverage.append(1 - missing)
            edge_extensions.append({"fraction": missing, "maximum_depth_pixels": maximum_depth})
        if not failed:
            break
    if failed:
        missing, maximum_depth, index = last_failure
        raise ValueError(f"Camera path exposes {missing:.4%} unseen border pixels (depth {maximum_depth:.2f}px) at bridge sample {index}; adjust anchors")
    if rate_retention < 1:
        warnings.append(f"Camera velocity deviation from endpoint-average rate reduced to {rate_retention:.1%} to retain observed source coverage")
    synthesized = model.synthesize(left_common.round().astype(np.uint8),
                                   right_common.round().astype(np.uint8), progress)
    if len(synthesized) != len(progress):
        raise RuntimeError("Interpolator returned the wrong number of bridge frames")
    output = [left.copy()]
    for i, (frame, matrix) in enumerate(zip(synthesized[1:-1], camera_matrices), 1):
        u = i / span
        image = cv2.warpAffine(frame, matrix[:2].astype(np.float32), (width, height),
                              flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT).astype(np.float32)
        # Smooth grade velocity is zero at both untouched source endpoints.
        grade_progress = u*u*u*(10 + u*(-15 + 6*u))
        image = image * (1 - grade_progress + grade_progress/gain) - grade_progress*bias/gain
        output.append(np.clip(image, 0, 255).round().astype(np.uint8))
    output.append(right.copy())
    report = {
        "method": "RIFE appearance in common view with Hermite similarity camera",
        "native_resolution": [width, height], "canvas_size": [int(v) for v in canvas_size],
        "endpoints_exact": bool(np.array_equal(output[0], left) and np.array_equal(output[-1], right)),
        "endpoint_confidence": endpoint_registration["confidence"],
        "incoming_confidence": previous_registration["confidence"],
        "outgoing_confidence": following_registration["confidence"],
        "endpoint_camera_parameters": p1.tolist(),
        "incoming_camera_rate": v0.tolist(), "outgoing_camera_rate": v1.tolist(),
        "measured_incoming_camera_rate": original_v0.tolist(), "measured_outgoing_camera_rate": original_v1.tolist(),
        "camera_rate_retention": rate_retention,
        "camera_parameters": camera_parameters, "source_coverage_min": min(coverage),
        "additional_real_context_canvas_fraction": float(np.mean(valid_union & ~endpoint_valid_union)),
        "source_edge_extensions": edge_extensions,
        "endpoint_registration_method": endpoint_registration.get("method"),
        "endpoint_dense_fit": {k: v for k, v in endpoint_registration.items() if k != "matrix"}
            if endpoint_registration.get("method", "").startswith("dense_") else None,
        "color_gain": gain.tolist(), "color_bias": bias.tolist(), "warnings": warnings,
        "requires_visual_review": True,
    }
    return output, report
