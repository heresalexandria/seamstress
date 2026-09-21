"""Automatic, conservative calibration of the original-frame conform pipeline.

Camera geometry is global affine/similarity only. Optical flow selects color
observations; it never warps output. A calibration is a reviewable candidate,
not a claim that every seam can be repaired from a flattened video.
"""
from __future__ import annotations

from collections import deque
import copy
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Callable

import cv2
import numpy as np
from scipy.interpolate import PchipInterpolator

from . import __version__
from .conform import validate_conform_plan
from .corrections import normalize_correction
from .camera_rate import recover_cadence_rate
from .design import ALGORITHM, build_conform_plan
from .local_color import apply_samples as apply_local_samples
from .media import iter_frames, probe
from .motion_evidence import recover_partial_edit
from .framing_evidence import recover_endpoint_edit
from .registration import _fit_color, _measure, _warp


class CalibrationCancelled(InterruptedError):
    """Raised before publication when a cancellation callback returns true."""


DEFAULT_OPTIONS = {
    'analysis_max_size': 640, 'max_samples': 12000, 'min_samples': 240,
    'local_centers': 64, 'geometry_support_seconds': 7.,
    'rate_support_frames': 12, 'camera_handles': 6,
    'max_view_crop_fraction': .08, 'min_feature_matches': 12,
    'min_inlier_fraction': .65, 'min_spatial_coverage': .3125,
    'max_scale_change': .10, 'max_anisotropy': 1.045,
    'max_rotation_degrees': 3., 'max_center_shift_fraction': .07,
    'max_reprojection_p90': 4., 'min_gradient_correlation': .60,
    'color_min_improvement': .025, 'local_min_improvement': .025,
    'local_limit': 18., 'local_ridge': .003, 'local_iterations': 12,
    'local_color_scale': 48., 'local_position_scale': .20,
    'max_tone_change': 24., 'enable_local_color': True,
    'max_seams': 128, 'max_frame_count': 250000,
}


def _options(options=None):
    if options is None:
        return dict(DEFAULT_OPTIONS)
    if not isinstance(options, dict):
        raise ValueError('calibration options must be an object')
    unknown = set(options)-set(DEFAULT_OPTIONS)
    if unknown:
        raise ValueError('Unknown calibration options: '+', '.join(sorted(unknown)))
    result = {**DEFAULT_OPTIONS, **options}
    integers = {'analysis_max_size': (64, 1920), 'max_samples': (80, 50000),
                'min_samples': (24, 10000), 'local_centers': (2, 128),
                'rate_support_frames': (1, 240), 'camera_handles': (2, 12),
                'min_feature_matches': (6, 200), 'local_iterations': (1, 30),
                'max_seams': (1, 1024), 'max_frame_count': (2, 1000000)}
    for key, (lo, hi) in integers.items():
        value = result[key]
        if type(value) is not int or not lo <= value <= hi:
            raise ValueError(f'{key} must be an integer in [{lo}, {hi}]')
    ranges = {'geometry_support_seconds': (.05, 60), 'max_view_crop_fraction': (0, .25),
              'min_inlier_fraction': (.3, 1), 'min_spatial_coverage': (.125, 1),
              'max_scale_change': (.001, .25), 'max_anisotropy': (1, 1.15),
              'max_rotation_degrees': (.1, 15), 'max_center_shift_fraction': (.001, .25),
              'max_reprojection_p90': (.5, 12), 'min_gradient_correlation': (.1, .99),
              'color_min_improvement': (0, .5), 'local_min_improvement': (0, .5),
              'local_limit': (1, 30), 'local_ridge': (.0001, 1),
              'local_color_scale': (16, 128), 'local_position_scale': (.1, 1),
              'max_tone_change': (1, 48)}
    for key, (lo, hi) in ranges.items():
        v = result[key]
        if type(v) not in (float, int) or not np.isfinite(v) or not lo <= v <= hi:
            raise ValueError(f'{key} must be a finite number in [{lo}, {hi}]')
    if type(result['enable_local_color']) is not bool:
        raise ValueError('enable_local_color must be boolean')
    if result['min_samples'] > result['max_samples']:
        raise ValueError('min_samples must not exceed max_samples')
    return result


def _check(cancelled):
    if cancelled is not None and cancelled():
        raise CalibrationCancelled('Calibration cancelled; no output bundle was published')


def _event(progress, stage, completed, total, message, seam=None):
    if progress is not None:
        progress({'stage': stage, 'completed': int(completed), 'total': int(total),
                  'fraction': float(completed/max(total, 1)), 'seam': seam, 'message': message})


def _size(width, height, maximum):
    factor = min(1., maximum/max(width, height))
    return max(32, round(width*factor)), max(32, round(height*factor))


def _coverage(points, width, height):
    if len(points) == 0:
        return 0.
    cell = np.clip((points/[width, height]*4).astype(int), 0, 3)
    return len(np.unique(cell[:, 1]*4+cell[:, 0]))/16


def calibrate_pair(left, right, options=None):
    """Return measured right-to-left global geometry in input-pixel units.

    ``accepted`` is an evidence/bounds decision, not a seamlessness score.
    ``scene_consistent`` can remain true when a known large transform is
    excluded: color calibration may still have valid matched observations.
    """
    opts = _options(options)
    left, right = np.asarray(left), np.asarray(right)
    if left.shape != right.shape or left.ndim != 3 or left.shape[2] != 3 or left.dtype != np.uint8 or right.dtype != np.uint8:
        raise ValueError('calibrate_pair needs equally sized uint8 RGB images')
    h0, w0 = left.shape[:2]
    if min(h0, w0) < 32:
        raise ValueError('Calibration requires image dimensions of at least32 pixels')
    w, h = _size(w0, h0, opts['analysis_max_size'])
    if (w, h) != (w0, h0):
        left, right = [cv2.resize(x, (w, h), interpolation=cv2.INTER_AREA) for x in (left, right)]
    base = {'accepted': False, 'scene_consistent': False, 'matrix': np.eye(3).tolist(),
            'confidence': 0., 'reason': '', 'method': 'none', 'diagnostics': {}}
    raw_mae = float(abs(left.astype(float)-right).mean())
    if raw_mae < .12:
        return {**base, 'accepted': True, 'scene_consistent': True, 'confidence': 1.,
                'reason': 'Identical pictures; no geometric correction needed', 'method': 'identity',
                'diagnostics': {'raw_mae': raw_mae, 'coverage': 1., 'inlier_fraction': 1., 'identity': True}}
    sift = cv2.SIFT_create(nfeatures=4500, contrastThreshold=.010, edgeThreshold=14)
    gray = [cv2.cvtColor(x, cv2.COLOR_RGB2GRAY) for x in (left, right)]
    ka, da = sift.detectAndCompute(gray[0], None)
    kb, db = sift.detectAndCompute(gray[1], None)
    if da is None or db is None or min(len(da), len(db)) < 4:
        return {**base, 'reason': 'Insufficient texture for global camera evidence'}
    matcher = cv2.BFMatcher()
    lr, rl = matcher.knnMatch(da, db, k=2), matcher.knnMatch(db, da, k=2)
    reverse = {m.queryIdx: m.trainIdx for pair in rl if len(pair) == 2
               for m, n in [pair] if m.distance < .78*n.distance}
    matches = [m for pair in lr if len(pair) == 2 for m, n in [pair]
               if m.distance < .78*n.distance and reverse.get(m.trainIdx) == m.queryIdx]
    balanced, cells = [], {}
    for match in sorted(matches, key=lambda x: x.distance):
        x, y = ka[match.queryIdx].pt
        cell = (min(5, int(x/w*6)), min(3, int(y/h*4)))
        if cells.get(cell, 0) < 24:
            balanced.append(match); cells[cell] = cells.get(cell, 0)+1
    if len(balanced) < opts['min_feature_matches']:
        return {**base, 'reason': 'Too few mutually consistent feature matches',
                'diagnostics': {'matches': len(balanced), 'raw_mae': raw_mae}}
    target = np.float32([ka[m.queryIdx].pt for m in balanced])
    source = np.float32([kb[m.trainIdx].pt for m in balanced])
    candidates = []
    for name, estimator in [('similarity', cv2.estimateAffinePartial2D), ('affine', cv2.estimateAffine2D)]:
        transform, inside = estimator(source, target, method=cv2.RANSAC,
                                       ransacReprojThreshold=1.8, maxIters=4000,
                                       confidence=.999, refineIters=20)
        if transform is None or inside is None:
            continue
        inside = inside[:, 0].astype(bool)
        errors = np.linalg.norm(source@transform[:, :2].T+transform[:, 2]-target, axis=1)
        candidates.append((name, np.vstack([transform, [0, 0, 1]]), inside, errors))
    if not candidates:
        return {**base, 'reason': 'RANSAC camera model did not converge'}
    chosen = candidates[0]
    if len(candidates) == 2:
        sim, aff = candidates
        if np.median(aff[3])+.05 < .85*np.median(sim[3]):
            chosen = aff
    name, matrix, inside, errors = chosen
    coverage = _coverage(target[inside], w, h)
    fraction = float(inside.mean())
    p90 = float(np.percentile(errors, 90))
    warped, valid = _warp(right, matrix)
    color = _fit_color(left, warped, valid)
    quality = _measure(left, warped, valid, color)
    scene = (int(inside.sum()) >= opts['min_feature_matches'] and
             fraction >= opts['min_inlier_fraction'] and coverage >= opts['min_spatial_coverage'] and
             p90 <= opts['max_reprojection_p90'] and
             quality['gradient_ncc'] >= opts['min_gradient_correlation'] and quality['overlap'] >= .75)
    singular = np.linalg.svd(matrix[:2, :2], compute_uv=False)
    center = np.array([(w-1)/2, (h-1)/2])
    shift = matrix[:2, :2]@center+matrix[:2, 2]-center
    angle = abs(math.degrees(math.atan2(matrix[1, 0], matrix[0, 0])))
    bounds = (np.linalg.det(matrix[:2, :2]) > 0 and
              np.max(abs(singular-1)) <= opts['max_scale_change'] and
              singular.max()/max(singular.min(), 1e-9) <= opts['max_anisotropy'] and
              angle <= opts['max_rotation_degrees'] and
              np.linalg.norm(shift/[w, h]) <= opts['max_center_shift_fraction'])
    confidence = float(min(1., inside.sum()/30)*min(1., coverage/.55)*fraction*max(0., quality['gradient_ncc']))
    reason = ('Distributed global camera fit accepted' if scene and bounds else
              'Large camera/shape correction exceeds conservative limits' if scene else
              'Global camera evidence is unreliable: scene change, parallax, or foreground motion')
    scale = np.diag([w/w0, h/h0, 1.])
    full = np.linalg.inv(scale)@matrix@scale
    return {**base, 'accepted': bool(scene and bounds), 'scene_consistent': bool(scene),
            'matrix': full.tolist(), 'confidence': confidence, 'reason': reason, 'method': name,
            'diagnostics': {'matches': len(balanced), 'inliers': int(inside.sum()),
                            'inlier_fraction': fraction, 'coverage': coverage,
                            'reprojection_p90_analysis_pixels': p90, 'analysis_size': [w, h],
                            'singular_values': singular.tolist(), 'center_shift_fraction': float(np.linalg.norm(shift/[w, h])),
                            'rotation_degrees': angle, 'quality': quality, 'raw_mae': raw_mae}}


def _parameters(matrix, center):
    a = matrix[:2, :2]
    u, singular, vt = np.linalg.svd(a)
    rotation = u@vt
    return np.array([np.log(np.sqrt(abs(np.linalg.det(a)))),
                     np.arctan2(rotation[1, 0], rotation[0, 0]),
                     *(a@center+matrix[:2, 2]-center)])


def _camera_rate(frames, indices, upscale, center, opts, cancelled):
    rates, evidence = [], []
    # Two-frame intervals retain the cartoon's existing on-twos cadence.
    for a, b in zip(indices[::2], indices[2::2]):
        _check(cancelled)
        fit = calibrate_pair(frames[a], frames[b], opts)
        evidence.append({'frames': [a, b], 'accepted': fit['accepted'], 'confidence': fit['confidence']})
        if fit['accepted']:
            backward = upscale@np.asarray(fit['matrix'])@np.linalg.inv(upscale)
            rates.append(_parameters(np.linalg.inv(backward), center)/(b-a))
    if not rates:
        return np.zeros(4), False, {'reason': 'No reliable within-segment camera interval', 'pairs': evidence}
    rates = np.array(rates)
    median = np.median(rates, axis=0)
    scales = np.array([.002, .001, max(1., center[0]*.004), max(1., center[1]*.004)])
    disagreement = float(np.max(np.median(abs(rates-median), axis=0)/scales))
    reliable = len(rates) >= 2 and disagreement < 1.5
    return median, reliable, {'pairs': evidence, 'accepted_intervals': len(rates),
                              'normalized_rate_disagreement': disagreement}


def _supports(cuts, count, requested):
    eligible, exclusions = [], {}
    for i, cut in enumerate(cuts):
        before = cut-1-(cuts[i-1] if i else 0)
        after = (cuts[i+1]-1 if i+1 < len(cuts) else count-1)-cut
        if min(before, after) < 2:
            exclusions[cut] = 'Too few independent source handles near an endpoint or adjacent seam'
        else:
            eligible.append(cut)
    support = requested
    for a, b in zip(eligible, eligible[1:]):
        support = min(support, max(1, (b-a-2)//2))
    return max(1, support), exclusions


def _identity_plan(calibration, metadata):
    return {'schema_version': 3, 'method': 'source_conform', 'geometry_mode': 'affine',
            'source': metadata.copy(), 'source_sha256': calibration['source_sha256'],
            'seams': [], 'segments': [{'start': 0, 'end': metadata['frame_count'],
                                      'matrix': np.eye(3).tolist(), 'gain': [1, 1, 1], 'bias': [0, 0, 0]}],
            'view_matrix': np.eye(3).tolist(), 'edge_extension_pixels': 0,
            'grade_curves': [], 'local_color_curves': [], 'unresolved_seams': [],
            'status': 'Identity candidate: no enabled seams; source frames and timing preserved',
            'design_report': {'global_geometry_corrections': [], 'constant_view_crop_fraction_total_dimension': 0.}}


def _assemble(calibration, metadata):
    plan = build_conform_plan(calibration, metadata) if calibration['cuts'] else _identity_plan(calibration, metadata)
    if 'correction_settings' in calibration:
        plan['correction_settings'] = copy.deepcopy(calibration['correction_settings'])
        intentional = {item['frame'] for item in calibration['correction_settings']
                       if item['correction']['geometry'] == 'off'}
        plan['unresolved_seams'] = [frame for frame in plan['unresolved_seams'] if frame not in intentional]
    return plan


def _color_scene_evidence(left, right, cancelled=None):
    """Establish local scene continuity without authorizing output geometry.

    This deliberately stricter fallback is only used when the existing global
    camera fit cannot establish scene continuity. Up to three bounded affine
    models explain independent local correspondences, as can occur at different
    scene depths. They are evidence only: none is used to move output pixels.
    Training and held-out matches are distinct, spatially balanced SIFT points;
    matching one small object in an otherwise different scene is insufficient.
    """
    _check(cancelled)
    h, w = left.shape[:2]
    report = {'accepted': False, 'method': 'held-out-local-correspondence',
              'analysis_size': [w, h], 'matches': 0, 'models': [],
              'reason': 'Too few independent local matches to establish scene continuity'}
    gray = [cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) for image in (left, right)]
    detector = cv2.SIFT_create(nfeatures=4500, contrastThreshold=.010, edgeThreshold=14)
    ka, da = detector.detectAndCompute(gray[0], None)
    kb, db = detector.detectAndCompute(gray[1], None)
    if da is None or db is None or min(len(da), len(db)) < 2:
        return report
    matcher = cv2.BFMatcher()
    lr, rl = matcher.knnMatch(da, db, k=2), matcher.knnMatch(db, da, k=2)
    reverse = {m.queryIdx: m.trainIdx for pair in rl if len(pair) == 2
               for m, n in [pair] if m.distance < .72*n.distance}
    matches = [m for pair in lr if len(pair) == 2 for m, n in [pair]
               if m.distance < .72*n.distance and reverse.get(m.trainIdx) == m.queryIdx]
    points, cells, seen_left, seen_right = [], {}, set(), set()
    for index, match in enumerate(sorted(matches, key=lambda item: item.distance)):
        if index % 128 == 0:
            _check(cancelled)
        a, b = np.array(ka[match.queryIdx].pt), np.array(kb[match.trainIdx].pt)
        if (np.linalg.norm(b-a) > math.hypot(w, h)*.10 or min(*a, *b) < 8 or
                max(a[0], b[0]) >= w-8 or max(a[1], b[1]) >= h-8):
            continue
        cell = (int(a[0]/w*6), int(a[1]/h*4))
        # Multiple SIFT orientations at one corner must not make that same
        # physical observation count as both training and held-out evidence.
        key_a, key_b = tuple(np.round(a/2).astype(int)), tuple(np.round(b/2).astype(int))
        if cells.get(cell, 0) >= 20 or key_a in seen_left or key_b in seen_right:
            continue
        patches = [cv2.getRectSubPix(image, (13, 13), tuple(p.astype(float))).astype(float).ravel()
                   for image, p in zip(gray, (a, b))]
        patches = [patch-patch.mean() for patch in patches]
        norm = np.linalg.norm(patches[0])*np.linalg.norm(patches[1])
        if norm < 100 or patches[0]@patches[1]/norm < .70:
            continue
        points.append((a, b)); cells[cell] = cells.get(cell, 0)+1
        seen_left.add(key_a); seen_right.add(key_b)
    report['matches'] = len(points)
    if len(points) < 48:
        return report
    target = np.float32([point[0] for point in points])
    source = np.float32([point[1] for point in points])
    train = np.zeros(len(points), bool)
    train[np.random.default_rng(681).permutation(len(points))[:len(points)//2]] = True
    remaining = train.copy(); errors = []; center = np.array([w/2, h/2])
    for _ in range(3):
        _check(cancelled)
        if remaining.sum() < 12:
            break
        transform, inside = cv2.estimateAffine2D(source[remaining], target[remaining],
                                                method=cv2.RANSAC, ransacReprojThreshold=1.8,
                                                maxIters=4000, confidence=.999, refineIters=20)
        if transform is None or inside is None or inside.sum() < 10:
            break
        singular = np.linalg.svd(transform[:, :2], compute_uv=False)
        shift = transform[:, :2]@center+transform[:, 2]-center
        angle = abs(math.degrees(math.atan2(transform[1, 0], transform[0, 0])))
        if (np.linalg.det(transform[:, :2]) <= 0 or np.max(abs(singular-1)) > .12 or
                singular.max()/max(singular.min(), 1e-9) > 1.12 or angle > 8 or
                np.linalg.norm(shift/[w, h]) > .10):
            break
        error = np.linalg.norm(source@transform[:, :2].T+transform[:, 2]-target, axis=1)
        errors.append(error); report['models'].append(transform.tolist())
        remaining &= error >= 2.
    if not errors:
        report['reason'] = 'No bounded local motion model has independent support'
        return report
    support = np.min(errors, axis=0) < 2.
    heldout = support & ~train

    def distributed_coverage(points, minimum=3):
        bins = np.clip((points/[w, h]*4).astype(int), 0, 3)
        counts = np.bincount(bins[:, 1]*4+bins[:, 0], minlength=16)
        return float(np.mean(counts >= minimum))

    coverage = min(distributed_coverage(points[support]) for points in (target, source))
    hull = min(float(cv2.contourArea(cv2.convexHull(points[support]))/(w*h))
               for points in (target, source))
    heldout_coverage = min(distributed_coverage(points[heldout], 1) for points in (target, source))
    heldout_hull = (min(float(cv2.contourArea(cv2.convexHull(points[heldout]))/(w*h))
                       for points in (target, source)) if heldout.sum() >= 3 else 0.)
    fraction = float(support[~train].mean())
    accepted = bool(fraction >= .80 and heldout.sum() >= 24 and coverage >= .5 and hull >= .4 and
                    heldout_coverage >= .5 and heldout_hull >= .4)
    report.update(accepted=accepted, heldout_fraction=fraction, heldout_matches=int(heldout.sum()),
                  distributed_coverage=coverage, convex_hull_fraction=hull,
                  heldout_coverage=heldout_coverage, heldout_convex_hull_fraction=heldout_hull,
                  reason=('Distributed local correspondence establishes scene continuity for color only'
                          if accepted else 'Local matches do not establish broad independent scene continuity'))
    return report


def _flow_observations(left, right, opts):
    h, w = left.shape[:2]
    gray = [cv2.cvtColor(x, cv2.COLOR_RGB2GRAY) for x in (left, right)]
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    dis.setFinestScale(0); dis.setVariationalRefinementIterations(5)
    forward = dis.calc(gray[0], gray[1], None)
    backward = dis.calc(gray[1], gray[0], None)
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    rx, ry = xx+forward[..., 0], yy+forward[..., 1]
    inverse = cv2.remap(backward, rx, ry, cv2.INTER_LINEAR)
    a = cv2.GaussianBlur(left.astype(np.float32), (0, 0), .65)
    b = cv2.remap(cv2.GaussianBlur(right.astype(np.float32), (0, 0), .65), rx, ry, cv2.INTER_LINEAR)
    gradients = []
    for frame in (a, b):
        dx, dy = cv2.Sobel(frame, cv2.CV_32F, 1, 0), cv2.Sobel(frame, cv2.CV_32F, 0, 1)
        gradients.append(np.sqrt(dx*dx+dy*dy).max(axis=2))
    valid = (np.linalg.norm(forward+inverse, axis=2) < .7) & (np.maximum(*gradients) < 24)
    valid &= (abs(a-b).max(axis=2) < 48) & (rx > 4) & (rx < w-5) & (ry > 4) & (ry < h-5)
    valid[:4] = False; valid[-4:] = False; valid[:, :4] = False; valid[:, -4:] = False
    rows, cols = min(8, max(2, h//12)), min(12, max(2, w//12))
    cap = max(2, opts['max_samples']//(rows*cols))
    samples = []; rng = np.random.default_rng(9183)
    for row in range(rows):
        for col in range(cols):
            y0, y1, x0, x1 = row*h//rows, (row+1)*h//rows, col*w//cols, (col+1)*w//cols
            y, x = np.nonzero(valid[y0:y1, x0:x1]); y += y0; x += x0
            if len(y) < 4:
                continue
            keep = rng.choice(len(y), min(cap, len(y)), replace=False); y, x = y[keep], x[keep]
            samples.append((a[y, x], b[y, x], np.stack([xx[y, x]/(w-1), yy[y, x]/(h-1)], -1),
                            np.stack([rx[y, x]/(w-1), ry[y, x]/(h-1)], -1),
                            np.full(len(x), (row+col)%2 == 0), np.full(len(x), row*cols+col)))
    if not samples:
        raise ValueError('No reliable matched interior colors')
    result = [np.concatenate([s[i] for s in samples]) for i in range(6)]
    if min(int(result[4].sum()), int((~result[4]).sum())) < opts['min_samples']//2:
        raise ValueError('Insufficient spatially held-out color samples')
    return (*result, {'sample_count': len(result[0]), 'valid_fraction': float(valid.mean()),
                      'tile_coverage': len(samples)/(rows*cols)})


def _error(a, b, positions):
    residual = a-b
    tiles = np.minimum((positions*[12, 8]).astype(int), [11, 7])
    keys = tiles[:, 1]*12+tiles[:, 0]
    biases = [np.median(residual[keys == key], axis=0) for key in np.unique(keys) if (keys == key).sum() >= 3]
    return {'mae': float(abs(residual).mean()), 'regional_bias': float(abs(np.array(biases)).mean()) if biases else float(abs(residual).mean())}


def _robust_grade(right, left):
    gain, bias = [], []
    for channel in range(3):
        x, y = right[:, channel]/255, left[:, channel]/255
        valid = (x > .02) & (x < .98) & (y > .02) & (y < .98)
        x, y = x[valid], y[valid]
        if len(x) < 24 or np.std(x) < .035:
            gain.append(1.); bias.append(float(np.median(y-x)*255) if len(x) else 0.); continue
        design = np.column_stack([x, np.ones(len(x))]); coef = np.array([1., 0.])
        ridge = np.diag([.002, .00002])*len(x)
        for _ in range(8):
            e = y-design@coef
            weights = np.minimum(1., .02/np.maximum(abs(e), 1e-8))
            coef = np.linalg.solve(design.T@(design*weights[:, None])+ridge,
                                   design.T@(y*weights)+ridge@np.array([1., 0.]))
        gain.append(float(coef[0])); bias.append(float(coef[1]*255))
    return np.array(gain), np.array(bias)


def _protected_luts(gain, bias):
    knots = np.array([0., 16., 48., 96., 144., 192., 240., 255.])
    target = knots[:, None]*gain+bias; target[0] = 0; target[-1] = 255
    if np.any(np.diff(target, axis=0) <= 0):
        raise ValueError('Global grade cannot preserve monotone protected endpoints')
    dense = np.linspace(0, 255, 8193); levels = np.arange(256.)
    left, right = [], []
    for c in range(3):
        transfer = PchipInterpolator(knots, target[:, c])
        mapped = transfer(dense)
        if np.any(np.diff(mapped) <= 0):
            raise ValueError('Global tone transfer is not invertible')
        inverse = PchipInterpolator(mapped, dense)
        left.append(.5*(levels+inverse(levels))); right.append(.5*(levels+transfer(levels)))
    tables = [np.clip(np.stack(values, axis=1), 0., 255.) for values in (left, right)]
    # PCHIP inverse evaluation can return255+6e-14 at its last knot. The
    # renderer deliberately enforces exact gamut bounds, so restore the
    # analytically exact endpoints before validation/publication.
    for table in tables:
        table[0] = 0.; table[-1] = 255.
        if not np.isfinite(table).all() or np.any(np.diff(table, axis=0) < 0):
            raise ValueError('Protected global LUT is not finite and monotone')
    return tuple(tables)


def _lut(samples, table):
    return np.stack([np.interp(samples[:, c], np.arange(256), table[:, c]) for c in range(3)], -1)


def _basis(rgb, xy, centers, scale):
    values = np.concatenate([rgb, xy], axis=1).astype(np.float32)/scale
    distance = np.maximum(np.sum(values*values, axis=1)[:, None]+np.sum(centers*centers, axis=1)[None]-2*values@centers.T, 0)
    nearest = distance.min(axis=1)
    basis = np.exp(-.5*(distance-nearest[:, None])); basis /= np.maximum(basis.sum(axis=1)[:, None], 1e-9)
    return basis*np.exp(-np.maximum(nearest-3, 0)/3)[:, None]


def _response(rgb, raw, limit, derivative=False):
    c = np.asarray(rgb, float)
    gate = (1-np.exp(-np.mean(c*c, axis=1, keepdims=True)/64))*(1-np.exp(-np.mean((255-c)**2, axis=1, keepdims=True)/64))
    t = np.tanh(raw/limit); offset = limit*t*gate
    available = np.where(offset >= 0, 255-c, c)
    z = offset/np.maximum(available, 1e-12)
    correction = available*np.tanh(z)
    if derivative:
        return correction, (1-np.tanh(z)**2)*gate*(1-t*t)*(available > 0)
    return correction


def _fit_local(rgb, xy, target, centers, weights, opts, cancelled):
    scale = np.array([opts['local_color_scale']]*3+[opts['local_position_scale']]*2, np.float32)
    basis = _basis(rgb, xy, centers, scale).astype(float)
    ridge = np.eye(len(centers))*(len(rgb)/len(centers)*opts['local_ridge'])
    delta = target-rgb
    co = np.linalg.solve(basis.T@(basis*weights[:, None])+ridge, basis.T@(weights[:, None]*delta))
    def loss(coef):
        e = delta-_response(rgb, basis@coef, opts['local_limit']); a = abs(e)
        return float(np.sum(weights[:, None]*np.where(a < 2, .5*e*e, 2*(a-1)))+.5*np.sum(coef*(ridge@coef)))
    for _ in range(opts['local_iterations']):
        _check(cancelled)
        prediction, derivative = _response(rgb, basis@co, opts['local_limit'], True)
        error = delta-prediction; steps = []
        for c in range(3):
            w = weights*np.minimum(1., 2/np.maximum(abs(error[:, c]), 1e-6))
            jac = basis*derivative[:, c, None]
            steps.append(np.linalg.solve(jac.T@(jac*w[:, None])+ridge, jac.T@(w*error[:, c])-ridge@co[:, c]))
        step = np.stack(steps, axis=1); prior = loss(co); rate = 1.
        while rate > 1/128 and loss(co+rate*step) > prior:
            rate *= .5
        co += rate*step
        if np.max(abs(rate*step)) < .002:
            break
    return {'mode': 'hybrid', 'response': 'directional-gamut-v2', 'feature_scale': scale.tolist(),
            'centers': centers.tolist(), 'coefficients': co.tolist(), 'limit': opts['local_limit']}


def _local_safety(model, rgb, xy, cancelled):
    # Include actual observed colors and a color/spatial grid. Boundary plateaus
    # are valid soft limiting; reject inversions and near-singular interiors.
    levels = np.array([0, 8, 32, 64, 128, 192, 247, 255], float)
    cube = np.stack(np.meshgrid(levels, levels, levels), -1).reshape(-1, 3)
    positions = np.array([[x, y] for x in (0., .5, 1.) for y in (0., .5, 1.)])
    c = np.concatenate([rgb[::max(1, len(rgb)//2000)], np.tile(cube, (len(positions), 1))])
    p = np.concatenate([xy[::max(1, len(rgb)//2000)], np.repeat(positions, len(cube), axis=0)])
    out = apply_local_samples(c, p, model); jac = []
    for channel in range(3):
        _check(cancelled)
        plus, minus = c.copy(), c.copy()
        plus[:, channel] = np.minimum(255, c[:, channel]+.5)
        minus[:, channel] = np.maximum(0, c[:, channel]-.5)
        jac.append((apply_local_samples(plus, p, model)-apply_local_samples(minus, p, model))/(plus[:, channel]-minus[:, channel])[:, None])
    jac = np.stack(jac, -1); diagonal = np.diagonal(jac, axis1=1, axis2=2)
    determinant = np.linalg.det(jac); interior = ((c >= 8) & (c <= 247)).all(axis=1)
    minimum = float(np.linalg.svd(jac[interior], compute_uv=False)[:, -1].min()) if interior.any() else 1.
    peak = float(abs(out-c).max())
    accepted = bool(diagonal.min() >= -1e-4 and determinant.min() >= -1e-4 and minimum >= .25 and out.min() >= -1e-5 and out.max() <= 255.00001)
    return {'accepted': accepted, 'minimum_own_channel_derivative': float(diagonal.min()),
            'minimum_color_determinant': float(determinant.min()), 'minimum_interior_singular': minimum,
            'maximum_sampled_change': peak, 'sample_count': len(c)}


def _color_fit(pairs, opts, cancelled):
    a, b, ap, bp, train, _, coverage = pairs[0]
    report = {'observations': coverage, 'global_accepted': False, 'local_accepted': False, 'reasons': []}
    if coverage['tile_coverage'] < .3 or coverage['valid_fraction'] < .12:
        report['reasons'].append('Insufficient distributed matched color coverage'); return None, None, report
    raw = _error(a[~train], b[~train], ap[~train]); report['before'] = raw
    if raw['mae'] < .65 and raw['regional_bias'] < .4:
        report['reasons'].append('Residual color difference is below conservative correction threshold'); return None, None, report
    left_lut = right_lut = np.repeat(np.arange(256.)[:, None], 3, axis=1)
    tone = None
    try:
        gain, bias = _robust_grade(b[train], a[train])
        if np.any((gain < .75) | (gain > 1.33)) or np.max(abs(bias)) > 30:
            raise ValueError('Estimated global grade exceeds conservative gain/bias limits')
        l, r = _protected_luts(gain, bias)
        if max(abs(l-left_lut).max(), abs(r-right_lut).max()) > opts['max_tone_change']:
            raise ValueError('Protected global LUT exceeds configured change limit')
        scores = []
        for k, (aa, bb, ax, _, tr, _, _) in enumerate(pairs):
            use = ~tr if k == 0 else np.ones(len(aa), bool)
            before = _error(aa[use], bb[use], ax[use]); after = _error(_lut(aa[use], l), _lut(bb[use], r), ax[use])
            scores.append({'before': before, 'after': after})
        improvement = 1-scores[0]['after']['mae']/max(raw['mae'], 1e-8)
        accept = improvement >= opts['color_min_improvement'] and all(s['after']['mae'] <= s['before']['mae']*1.02+.05 and s['after']['regional_bias'] <= s['before']['regional_bias']*1.05+.08 for s in scores)
        report['global_validation'] = scores
        if accept:
            left_lut, right_lut = l, r; tone = (l, r); report['global_accepted'] = True
        else:
            report['reasons'].append('Protected global LUT failed held-out or independent-pair improvement')
    except (ValueError, np.linalg.LinAlgError) as exc:
        report['reasons'].append(str(exc))
    if not opts['enable_local_color']:
        report['reasons'].append('Local correction disabled by options'); return tone, None, report
    mapped = [(_lut(aa, left_lut), _lut(bb, right_lut), ax, bx, tr) for aa, bb, ax, bx, tr, _, _ in pairs]
    a, b, ap, bp, train = mapped[0]; residual = _error(a[~train], b[~train], ap[~train])
    if residual['mae'] < .65 and residual['regional_bias'] < .4:
        report['reasons'].append('Protected global LUT leaves no material local residual'); return tone, None, report
    target = (a+b)/2; features = np.concatenate([target[train], (ap[train]+bp[train])/2], axis=1).astype(np.float32)
    scale = np.array([opts['local_color_scale']]*3+[opts['local_position_scale']]*2)
    features /= scale
    bins = (target[train]/24).astype(int); _, inv, counts = np.unique(bins, axis=0, return_inverse=True, return_counts=True)
    weights = np.clip(np.sqrt(np.median(counts)/counts[inv]), .25, 8); weights /= weights.mean()
    centers_count = min(opts['local_centers'], max(2, len(features)//12))
    rng = np.random.default_rng(221); chosen = rng.choice(len(features), len(features), replace=True, p=weights/weights.sum())
    cv2.setRNGSeed(19)
    _, _, centers = cv2.kmeans(features[chosen], centers_count, None, (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 40, .03), 2, cv2.KMEANS_PP_CENTERS)
    _check(cancelled)
    left = _fit_local(a[train], ap[train], target[train], centers, weights, opts, cancelled)
    right = _fit_local(b[train], bp[train], target[train], centers, weights, opts, cancelled)
    checks = []
    for k, (aa, bb, ax, bx, tr) in enumerate(mapped):
        _check(cancelled)
        use = ~tr if k == 0 else np.ones(len(aa), bool)
        one, two = apply_local_samples(aa[use], ax[use], left), apply_local_samples(bb[use], bx[use], right)
        before, after = _error(aa[use], bb[use], ax[use]), _error(one, two, ax[use])
        # Coarse material/color bins protect against a majority sky/wall hiding
        # a worse minority palette in the same held-out observations.
        bins = ((aa[use]+bb[use])/2/48).astype(int); keys = bins[:, 0]*36+bins[:, 1]*6+bins[:, 2]
        regressions = []
        for key in np.unique(keys):
            subset = keys == key
            if subset.sum() < 30:
                continue
            prior = float(abs(aa[use][subset]-bb[use][subset]).mean()); new = float(abs(one[subset]-two[subset]).mean())
            if new > prior*1.15+.25:
                regressions.append({'palette_bin': int(key), 'before': prior, 'after': new})
        checks.append({'before': before, 'after': after, 'palette_regressions': regressions})
    safety = [_local_safety(m, rgb, xy, cancelled) for m, rgb, xy in [(left, a, ap), (right, b, bp)]]
    accept = (checks[0]['after']['mae'] <= residual['mae']*(1-opts['local_min_improvement']) and
              all(s['after']['mae'] <= s['before']['mae']*1.02+.05 and s['after']['regional_bias'] <= s['before']['regional_bias']*1.05+.08 and not s['palette_regressions'] for s in checks) and all(s['accepted'] for s in safety))
    report.update(local_validation=checks, local_safety=safety, local_accepted=bool(accept))
    if not accept:
        report['reasons'].append('Local model failed held-out palette, cross-pair, or Jacobian safeguards')
    return tone, (left, right) if accept else None, report


def _write_bundle(payloads, cancelled):
    _check(cancelled)
    published = []
    folder = next(iter(payloads)).parent
    with tempfile.TemporaryDirectory(prefix='.seamstress-publish-', dir=folder) as temp:
        staged = []
        for i, (destination, payload) in enumerate(payloads.items()):
            file = Path(temp)/f'{i}.json'; file.write_text(json.dumps(payload, indent=2, allow_nan=False))
            staged.append((file, destination))
        _check(cancelled)
        try:
            for source, destination in staged:
                os.link(source, destination); published.append(destination)
        except BaseException:
            for path in published:
                path.unlink(missing_ok=True)
            raise


def calibrate_video(source: Path, seams: list[int], output: Path, *,
                    progress: Callable | None = None, cancelled: Callable | None = None,
                    options: dict | None = None, seam_settings: dict | None = None,
                    analysis_boundaries: list[int] | None = None,
                    plan_transform: Callable | None = None,
                    geometry_support_frames: int | None = None) -> dict:
    """Measure a CFR SDR source and publish new calibration, plan, and report.

    ``output`` names the calibration JSON. Sibling ``.plan.json`` and
    ``.report.json`` files are also created, never overwritten. ``progress``
    receives dictionaries(stage, completed,total,fraction,seam,message).
    ``cancelled()`` is checked between decoded frames and fitting iterations;
    cancellation raises CalibrationCancelled before any output publication.
    Callers must validate constant frame rate and unrotated native dimensions.
    ``seam_settings`` maps enabled incoming frame indices to correction choices;
    omitted seams retain the established automatic defaults.
    ``analysis_boundaries`` can include additional joins that limit generation
    handles without being analyzed. ``plan_transform`` is an internal assembly
    hook used to fit a selected seam within a frozen correction plan.
    ``geometry_support_frames`` supplies that workflow's exact whole-frame
    return support without rounding it through the public seconds option.
    """
    opts = _options(options); _check(cancelled)
    source, output = Path(source).expanduser().resolve(), Path(output).expanduser().resolve()
    plan_path, report_path = output.with_suffix('.plan.json'), output.with_suffix('.report.json')
    paths = [output, plan_path, report_path]
    if len(set(paths)) != 3 or source in paths or any(p.exists() for p in paths):
        raise ValueError('Choose new calibration, plan, and report paths distinct from the source')
    metadata = probe(source)
    if metadata.get('color_transfer') in ('smpte2084', 'arib-std-b67'):
        raise ValueError('HDR needs a separate high-bit-depth calibration pipeline')
    count, width, height = metadata['frame_count'], metadata['width'], metadata['height']
    if count < 1 or count > opts['max_frame_count'] or min(width, height) < 32:
        raise ValueError('Source frame count or dimensions exceed supported calibration bounds')
    if not isinstance(seams, list) or any(type(n) is not int or not 0 < n < count for n in seams) or seams != sorted(set(seams)):
        raise ValueError('seams must be sorted unique zero-based incoming frame indices within the source')
    if len(seams) > opts['max_seams']:
        raise ValueError('Too many seams for the configured bounded calibration run')
    boundaries=seams if analysis_boundaries is None else analysis_boundaries
    if (not isinstance(boundaries,list) or len(boundaries)>10000 or
            any(type(n) is not int or not 0<n<count for n in boundaries) or
            boundaries!=sorted(set(boundaries)) or not set(seams)<=set(boundaries)):
        raise ValueError('analysis_boundaries must be sorted unique source frame indices including every analyzed seam')
    if plan_transform is not None and not callable(plan_transform):
        raise ValueError('plan_transform must be callable')
    if seam_settings is None:
        seam_settings = {}
    if (not isinstance(seam_settings, dict) or
            any(type(frame) is not int or frame not in seams for frame in seam_settings)):
        raise ValueError('seam_settings must map enabled seam frame indices to correction objects')
    settings = {frame: normalize_correction(seam_settings.get(frame), metadata=metadata, frame=frame)
                for frame in seams}
    _event(progress, 'fingerprint', 0, 1, 'Fingerprinting the source')
    digest = hashlib.sha256()
    with source.open('rb') as handle:
        for block in iter(lambda: handle.read(4*1024*1024), b''):
            _check(cancelled); digest.update(block)
    settings = {frame: normalize_correction(policy, metadata=metadata,
                                           source_sha256=digest.hexdigest(), frame=frame)
                for frame, policy in settings.items()}
    output.parent.mkdir(parents=True, exist_ok=True)
    fps = float(Fraction(metadata['fps_fraction']))
    if geometry_support_frames is not None and (type(geometry_support_frames) is not int or
            not 1 <= geometry_support_frames <= max(1, round(60*fps))):
        raise ValueError('geometry_support_frames must be a positive whole-frame support of at most 60 seconds')
    requested = (geometry_support_frames if geometry_support_frames is not None else
                 max(1, round(opts['geometry_support_seconds']*fps)))
    support, handle_exclusions = _supports(boundaries, count, requested)
    if boundaries != seams:
        # Unanalyzed markers bound the selected generation's handles, but an
        # unrelated pair elsewhere must not shorten this correction's return.
        support = requested
        positions = {cut: i for i, cut in enumerate(boundaries)}
        for cut in seams:
            if cut in handle_exclusions:
                continue
            index = positions[cut]
            neighbors = boundaries[max(0, index-1):index]+boundaries[index+1:index+2]
            for neighbor in neighbors:
                support = min(support, max(1, (abs(neighbor-cut)-2)//2))
        handle_exclusions = {cut: reason for cut, reason in handle_exclusions.items() if cut in seams}
    rate_support = min(opts['rate_support_frames'], support)
    calibration = {'schema_version': 1, 'method': 'source_conform_calibration', 'algorithm': ALGORITHM,
                   'source': metadata.copy(), 'source_sha256': digest.hexdigest(),
                   'parameters': {'geometry_support': support, 'rate_support': rate_support,
                                  'source_margin_pixels': 0., 'max_view_crop_fraction_total_dimension': opts['max_view_crop_fraction']},
                   'cuts': [], 'excluded_geometry': [], 'grade_curves': [], 'local_color_curves': [],
                   'correction_settings': [{'frame': frame, 'correction': policy} for frame, policy in settings.items()],
                   'status': ('REVIEWED SETTINGS CANDIDATE: includes explicit manual framing; visual review required'
                              if any(policy['geometry'] == 'manual' for policy in settings.values()) else
                              'AUTOMATIC CANDIDATE: source-specific measured corrections; visual review required'),
                   'generator': {'name': 'seamstress.calibrate', 'version': __version__, 'options': opts}}
    if geometry_support_frames is not None:
        calibration['generator']['geometry_support_frames'] = geometry_support_frames
    report = {'schema_version': 1, 'source_sha256': digest.hexdigest(), 'source': metadata.copy(),
              'options': opts, 'correction_settings': copy.deepcopy(calibration['correction_settings']),
              'seams': [], 'limitations': [
                  'Global affine geometry cannot resolve depth-dependent parallax or changed poses.',
                  'Color samples use correspondence only; output never uses dense warps or mixed source frames.',
                  'Jacobians and palette safeguards are sampled, not a guarantee of imperceptible transitions.',
                  'A close seam pair can shorten the globally shared neutral-return support.',
                  'Constant frame rate and native display orientation must be validated by the caller.']}
    def assembled():
        candidate=_assemble(calibration,metadata)
        if plan_transform is not None:
            candidate=plan_transform(candidate,calibration,metadata)
            validate_conform_plan(candidate,metadata)
        return candidate
    if not seams:
        plan = assembled()
        report['summary'] = {'seam_count': 0, 'geometry_accepted': 0, 'protected_tone_curves': 0, 'local_color_curves': 0, 'constant_crop_fraction': 0., 'geometry_exclusions': []}
        validate_conform_plan(plan, metadata)
        _write_bundle({output: calibration, plan_path: plan, report_path: report}, cancelled)
        _event(progress, 'complete', 1, 1, 'Identity calibration created; no enabled seams')
        return {'calibration_path': str(output), 'plan_path': str(plan_path), 'report_path': str(report_path),
                'calibration': calibration, 'report': report, 'unresolved_seams': []}
    aw, ah = _size(width, height, opts['analysis_max_size'])
    down = np.diag([aw/width, ah/height, 1.]); upscale = np.linalg.inv(down)
    center = np.array([(width-1)/2, (height-1)/2]); handles = opts['camera_handles']
    # Keep the ordinary estimator's requested handles unchanged. The additional
    # pictures only test for phase aliasing across complete animation cycles.
    analysis_handles = max(handles, 12)
    windows = []
    positions={cut:i for i,cut in enumerate(boundaries)}
    for cut in seams:
        i=positions[cut]
        start = max(0, cut-analysis_handles-1, boundaries[i-1] if i else 0)
        end = min(count-1, cut+analysis_handles, boundaries[i+1]-1 if i+1 < len(boundaries) else count-1)
        windows.append((cut, start, end))
    with tempfile.TemporaryDirectory(prefix='.seamstress-calibration-', dir=output.parent) as temporary:
        temporary = Path(temporary); rolling = deque(maxlen=2*analysis_handles+2); pending = 0
        iterator = iter_frames(source, 0, windows[-1][2]+1, (aw, ah))
        try:
            for index, frame in enumerate(iterator):
                _check(cancelled); rolling.append((index, frame))
                if index % max(1, round(fps)) == 0:
                    _event(progress, 'decode', index, windows[-1][2]+1, 'Reading source frames')
                while pending < len(windows) and index == windows[pending][2]:
                    cut, start, end = windows[pending]
                    policy = settings[cut]
                    frames = {n: frame for n, frame in rolling if start <= n <= end}
                    _event(progress, 'geometry', pending, len(seams), 'Measuring global geometry and camera rates', cut)
                    fit = calibrate_pair(frames[cut-1], frames[cut], opts); _check(cancelled)
                    native = upscale@np.asarray(fit['matrix'])@down
                    pre, pre_ok, pre_report = _camera_rate(frames, list(range(max(start, cut-handles-1), cut)), upscale, center, opts, cancelled)
                    post, post_ok, post_report = _camera_rate(frames, list(range(cut, min(end, cut+handles)+1)), upscale, center, opts, cancelled)
                    for side, reliable, rate_report in (('pre', pre_ok, pre_report), ('post', post_ok, post_report)):
                        if reliable and policy['geometry'] == 'auto' and policy['cadence']:
                            cadence = recover_cadence_rate(
                                frames, cut, side, pair_fit=lambda left, right: calibrate_pair(left, right, opts),
                                upscale=upscale, center=center, cancelled=cancelled)
                            rate_report['cadence_recovery'] = cadence
                            if cadence['accepted']:
                                if side == 'pre':
                                    pre = np.asarray(cadence['rate'])
                                else:
                                    post = np.asarray(cadence['rate'])
                    ease = bool(policy['rate_easing'] and pre_ok and post_ok and np.linalg.norm((pre-post)/[.002, .001, max(1., width*.002), max(1., height*.002)]) > .15)
                    reason = handle_exclusions.get(cut)
                    partial = endpoint = None
                    if policy['geometry'] == 'off':
                        reason = 'Geometry disabled by the per-seam correction policy'
                    elif policy['geometry'] == 'manual':
                        if reason:
                            raise ValueError(f'Manual geometry at frame {cut}: {reason}')
                    elif not fit['accepted']:
                        reason = fit['reason']
                    elif not pre_ok:
                        if reason is None and policy['partial_recovery']:
                            partial = recover_partial_edit(
                                frames, cut, fit, pair_fit=lambda left, right: calibrate_pair(left, right, opts),
                                upscale=upscale, options=opts, cancelled=cancelled)
                        if reason is None and policy['endpoint_recovery'] and (not partial or not partial['accepted']):
                            endpoint = recover_endpoint_edit(
                                frames, cut, fit, upscale=upscale, options=opts, cancelled=cancelled)
                        if not (partial and partial['accepted']) and not (endpoint and endpoint['accepted']):
                            reason = reason or 'Incoming edit cannot be separated from camera motion: outgoing rate is unreliable'
                    if reason:
                        calibration['excluded_geometry'].append({'frame': cut, 'reason': reason})
                    recovery = partial if partial and partial['accepted'] else endpoint
                    recovered = recovery is not None and recovery['accepted'] and reason is None
                    # Recovered edits already separate recrop from ordinary
                    # motion. Zero assembly rates prevent applying it twice.
                    record = {'frame': cut, 'right_to_left_matrix': (
                        np.eye(3) if reason else np.asarray(recovery['edit_matrix']) if recovered else native).tolist(),
                        'pre_rate': (np.zeros(4) if recovered else pre).tolist(),
                        'post_rate': (np.zeros(4) if recovered else post).tolist(), 'ease_rate': False if recovered else ease,
                        'correction': copy.deepcopy(policy)}
                    if policy['geometry'] == 'manual':
                        for key in ('right_to_left_matrix', 'pre_rate', 'post_rate', 'ease_rate'):
                            record[key] = copy.deepcopy(policy['manual'][key])
                        record['ease_rate'] = bool(record['ease_rate'] and policy['rate_easing'])
                        calibration.setdefault('review_decisions', []).append({
                            'frame': cut, 'treatment': 'Explicit manual global framing and camera rates',
                            'automatic_approval': False, 'correction': copy.deepcopy(policy),
                            'provenance': copy.deepcopy(policy['manual'].get('provenance'))})
                    elif policy['geometry'] == 'off':
                        calibration.setdefault('review_decisions', []).append({
                            'frame': cut, 'treatment': 'Geometry intentionally disabled',
                            'automatic_approval': False, 'correction': copy.deepcopy(policy)})
                    calibration['cuts'].append(record)
                    report['seams'].append({'frame': cut, 'geometry': fit, 'measured_native_right_to_left_matrix': native.tolist(), 'geometry_excluded_reason': reason,
                                            'correction': copy.deepcopy(policy),
                                            'geometry_status': ('off' if policy['geometry'] == 'off' else 'unresolved' if reason else 'manual' if policy['geometry'] == 'manual' else 'automatic'),
                                            'pre_rate': pre_report, 'post_rate': post_report,
                                            'rate_easing': bool(record['ease_rate'] and not reason), 'color': {'status': 'pending'}})
                    if partial is not None:
                        partial['original_native_pre_rate'] = pre.tolist()
                        partial['original_native_post_rate'] = post.tolist()
                        report['seams'][-1]['partial_geometry'] = partial
                    if endpoint is not None:
                        report['seams'][-1]['framing_recovery'] = endpoint
                    keep = [n for n in range(max(start, cut-3), min(end, cut+2)+1)]
                    np.savez_compressed(temporary/f'{cut}.npz', indices=np.array(keep), frames=np.stack([frames[n] for n in keep]))
                    pending += 1
        finally:
            iterator.close()
        if pending != len(seams):
            raise ValueError('Source ended before all requested seam windows could be decoded')
        # A measured transform can be locally plausible but require too much
        # constant crop. Exclude the largest accepted impulse until the actual
        # all-frame source-coverage calculation succeeds; never silently fill.
        while True:
            _check(cancelled)
            try:
                plan = assembled(); break
            except ValueError as exc:
                if 'crop' not in str(exc) and 'covers' not in str(exc):
                    raise
                excluded = {x['frame'] for x in calibration['excluded_geometry']}
                remaining = [r for r in calibration['cuts'] if r['frame'] not in excluded]
                if not remaining:
                    raise
                worst = max(remaining, key=lambda r: np.linalg.norm(_parameters(np.asarray(r['right_to_left_matrix']), center)/[.03, .02, width*.03, height*.03]))
                if settings[worst['frame']]['geometry'] == 'manual':
                    raise ValueError(f'Manual geometry at frame {worst["frame"]} exceeds the source coverage/crop budget: {exc}') from exc
                reason = 'Excluded to satisfy actual constant-crop source coverage: '+str(exc)
                calibration['excluded_geometry'].append({'frame': worst['frame'], 'reason': reason})
                seam_report = next(r for r in report['seams'] if r['frame'] == worst['frame'])
                seam_report['geometry_excluded_reason'] = reason
                seam_report['geometry_status'] = 'unresolved'
                seam_report['rate_easing'] = False
        _event(progress, 'color', 0, len(seams), 'Fitting protected tone and bounded local color')
        for i, seam_report in enumerate(report['seams']):
            _check(cancelled); cut = seam_report['frame']
            policy = settings[cut]
            _event(progress, 'color', i, len(seams), 'Validating source-specific color corrections', cut)
            if policy['color'] == 'off':
                seam_report['color'] = {'status': 'off', 'reason': 'Color disabled by the per-seam correction policy'}
                continue
            if cut in handle_exclusions:
                seam_report['color'] = {'status': 'excluded', 'reason': handle_exclusions[cut]}; continue
            with np.load(temporary/f'{cut}.npz') as stored:
                originals = {int(n): image for n, image in zip(stored['indices'], stored['frames'])}
                if seam_report['geometry']['scene_consistent']:
                    # Keep the established path and its numerical operations
                    # unchanged. The fallback cannot loosen camera/rate gates.
                    continuity = {'accepted': True, 'method': 'global-registration',
                                  'reason': 'Existing global registration establishes scene continuity'}
                else:
                    evidence = []
                    for offset in range(3):
                        a, b = cut-1-offset, cut+offset
                        if a in originals and b in originals:
                            evidence.append({'frames': [a, b],
                                             **_color_scene_evidence(originals[a], originals[b], cancelled)})
                    continuity = {'accepted': len(evidence) >= 2 and all(pair['accepted'] for pair in evidence),
                                  'method': 'independent-local-correspondence', 'pairs': evidence,
                                  'reason': 'Every independent cross-cut pair must establish broad local scene continuity'}
                seam_report['scene_continuity'] = continuity
                if not continuity['accepted']:
                    seam_report['color'] = {'status': 'excluded', 'reason': 'Insufficient structural evidence that these pictures continue the same scene'}
                    continue
                frames = {}
                for n, image in zip(stored['indices'], stored['frames']):
                    native_matrix = np.asarray(plan['view_matrix'])@np.asarray(plan['frame_matrices'][int(n)])
                    matrix = down@native_matrix@upscale
                    frames[int(n)] = cv2.warpAffine(image, matrix[:2], (aw, ah), flags=cv2.INTER_CUBIC)
            pairs = []
            try:
                for offset in range(3):
                    a, b = cut-1-offset, cut+offset
                    if a in frames and b in frames:
                        _check(cancelled); pairs.append(_flow_observations(frames[a], frames[b], opts))
                if len(pairs) < 2:
                    raise ValueError('No independent cross-cut pair for color validation')
                color_options = {**opts, 'enable_local_color': False} if policy['color'] == 'tone' else opts
                tone, local, color_report = _color_fit(pairs, color_options, cancelled)
                before, after = min(support, cut-1), min(support, count-1-cut)
                if tone is not None:
                    calibration['grade_curves'].append({'frame': cut, 'support_before': before, 'support_after': after,
                                                         'left_lut': tone[0].tolist(), 'right_lut': tone[1].tolist()})
                if local is not None:
                    calibration['local_color_curves'].append({'frame': cut, 'support_before': before, 'support_after': after,
                                                               'left': local[0], 'right': local[1]})
                color_report['status'] = 'accepted' if tone is not None or local is not None else 'unchanged'
                seam_report['color'] = color_report
            except (ValueError, cv2.error, np.linalg.LinAlgError) as exc:
                seam_report['color'] = {'status': 'excluded', 'reason': str(exc)}
        plan = assembled()
    report['summary'] = {'seam_count': len(seams), 'geometry_accepted': len(seams)-len(calibration['excluded_geometry']),
                         'protected_tone_curves': len(calibration['grade_curves']),
                         'local_color_curves': len(calibration['local_color_curves']),
                         'constant_crop_fraction': plan['design_report']['constant_view_crop_fraction_total_dimension'],
                         'partial_geometry_frames': [item['frame'] for item in report['seams']
                                                     if item.get('partial_geometry', {}).get('accepted') and not item['geometry_excluded_reason']],
                         'endpoint_geometry_frames': [item['frame'] for item in report['seams']
                                                      if item.get('framing_recovery', {}).get('accepted') and not item['geometry_excluded_reason']],
                         'cadence_adjusted_frames': [item['frame'] for item in report['seams']
                                                     if not item['geometry_excluded_reason']
                                                     and item['correction']['geometry'] == 'auto'
                                                     and not item.get('partial_geometry', {}).get('accepted')
                                                     and not item.get('framing_recovery', {}).get('accepted') and (
                                                         item['pre_rate'].get('cadence_recovery', {}).get('accepted') or
                                                         (item['rate_easing'] and item['post_rate'].get('cadence_recovery', {}).get('accepted')))],
                         'manual_geometry_frames': [item['frame'] for item in report['seams'] if item['geometry_status'] == 'manual'],
                         'geometry_disabled_frames': [item['frame'] for item in report['seams'] if item['geometry_status'] == 'off'],
                         'color_disabled_frames': [item['frame'] for item in report['seams'] if item['color']['status'] == 'off'],
                         'geometry_exclusions': calibration['excluded_geometry']}
    plan['generator'] = {'name': 'seamstress.calibrate', 'version': __version__, 'algorithm': ALGORITHM}
    if 'review_decisions' in calibration:
        report['review_decisions'] = copy.deepcopy(calibration['review_decisions'])
    validate_conform_plan(plan, metadata)
    _check(cancelled); _event(progress, 'publish', 0, 1, 'Publishing calibration, plan, and evidence report')
    _write_bundle({output: calibration, plan_path: plan, report_path: report}, cancelled)
    _event(progress, 'complete', len(seams), len(seams), 'Calibration candidate ready for visual review')
    return {'calibration_path': str(output), 'plan_path': str(plan_path), 'report_path': str(report_path),
            'calibration': calibration, 'report': report, 'unresolved_seams': plan['unresolved_seams']}
