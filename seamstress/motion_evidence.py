"""Recover only geometrical edit components that mixed scene motion identifies.

Correspondences are evidence, never rendering instructions. The returned edit
is one global affine transform. It deliberately leaves the ambiguous motion
direction untouched and must not be passed through camera-rate compensation a
second time.
"""
from __future__ import annotations

import math

import cv2
import numpy as np
from scipy.linalg import expm, logm


def _coverage(points, width, height):
    if not len(points):
        return 0.
    cells = np.clip((points/[width, height]*4).astype(int), 0, 3)
    return len(np.unique(cells[:, 1]*4+cells[:, 0]))/16


def _matches(left, right):
    sift = cv2.SIFT_create(nfeatures=4500, contrastThreshold=.010, edgeThreshold=14)
    pairs = [sift.detectAndCompute(cv2.cvtColor(x, cv2.COLOR_RGB2GRAY), None)
             for x in (left, right)]
    (ka, da), (kb, db) = pairs
    if da is None or db is None or min(len(da), len(db)) < 12:
        return np.empty((0, 2)), np.empty((0, 2))
    matcher = cv2.BFMatcher()
    lr, rl = matcher.knnMatch(da, db, k=2), matcher.knnMatch(db, da, k=2)
    reverse = {m.queryIdx: m.trainIdx for pair in rl if len(pair) == 2
               for m, n in [pair] if m.distance < .78*n.distance}
    matches = [m for pair in lr if len(pair) == 2 for m, n in [pair]
               if m.distance < .78*n.distance and reverse.get(m.trainIdx) == m.queryIdx]
    h, w = left.shape[:2]
    balanced, cells = [], {}
    seen_left, seen_right = set(), set()
    for match in sorted(matches, key=lambda x: x.distance):
        x, y = ka[match.queryIdx].pt
        target = kb[match.trainIdx].pt
        # SIFT can describe several orientations of one physical point. They
        # cannot count as independent support for a spatial motion layer.
        left_key = tuple(round(v*4) for v in (x, y))
        right_key = tuple(round(v*4) for v in target)
        if left_key in seen_left or right_key in seen_right:
            continue
        cell = (min(5, int(x/w*6)), min(3, int(y/h*4)))
        if cells.get(cell, 0) < 24:
            balanced.append(match)
            cells[cell] = cells.get(cell, 0)+1
            seen_left.add(left_key)
            seen_right.add(right_key)
    return (np.float32([ka[m.queryIdx].pt for m in balanced]).reshape(-1, 2),
            np.float32([kb[m.trainIdx].pt for m in balanced]).reshape(-1, 2))


def _motion_models(left, right):
    """Peel coherent similarities without forcing foreground/background together."""
    p, q = _matches(left, right)
    h, w = left.shape[:2]
    pixel = max(w, h)/640
    remain = np.ones(len(p), bool)
    result = []
    for _ in range(4):
        if remain.sum() < 12:
            break
        affine, _ = cv2.estimateAffinePartial2D(
            p[remain], q[remain], method=cv2.RANSAC,
            ransacReprojThreshold=.45*pixel, maxIters=10000,
            confidence=.999, refineIters=20)
        if affine is None:
            break
        error = np.linalg.norm(p@affine[:, :2].T+affine[:, 2]-q, axis=1)
        inside = (error < .65*pixel) & remain
        remain &= ~inside
        if inside.sum() < 12:
            continue
        points = p[inside]
        fraction = float(inside.mean())
        coverage = _coverage(points, w, h)
        span = np.ptp(points, axis=0)/[w, h]
        if fraction >= .2 and coverage >= .3125 and span.min() >= .2:
            result.append({'matrix': np.vstack([affine, [0, 0, 1.]]).tolist(),
                           'fraction': fraction, 'coverage': coverage,
                           'span': span.tolist(), 'matches': int(inside.sum()),
                           'residual_p90': float(np.percentile(error[inside], 90))})
    return result


def _power(matrix, power):
    value = expm(logm(np.asarray(matrix, float))*power)
    if np.max(abs(np.imag(value))) > 1e-8 or not np.isfinite(value).all():
        raise ValueError('Motion model has no stable real affine power')
    return np.real(value)


def _parameters(matrix, center):
    a = matrix[:2, :2]
    return np.array([math.log(math.sqrt(np.linalg.det(a))),
                     math.atan2(a[1, 0], a[0, 0]),
                     *(a@center+matrix[:2, 2]-center)])


def _matrix(parameters, center):
    scale, angle = math.exp(parameters[0]), parameters[1]
    result = np.eye(3)
    result[:2, :2] = scale*np.array([[math.cos(angle), -math.sin(angle)],
                                    [math.sin(angle), math.cos(angle)]])
    result[:2, 2] = center+parameters[2:]-result[:2, :2]@center
    return result


def _infer_edit(registration, groups, cross, post_motion, shape, options=None):
    """Pure inference in analysis pixels; group matrices advance two frames.

    ``cross`` contains (right-to-left registration, pre steps, post steps).
    F_pre**pre_steps @ registration @ F_post**post_steps maps the first
    incoming frame to the predicted unedited picture at that same instant.
    """
    h, w = shape
    pixel = max(w, h)/640
    bounds = {'max_scale_change': .10, 'max_anisotropy': 1.045,
              'max_rotation_degrees': 3., 'max_center_shift_fraction': .07}
    if options is not None:
        bounds.update({key: options[key] for key in bounds if key in options})
    reject = lambda reason, **data: {'accepted': False, 'reason': reason, **data}
    if len(groups) != 3 or any(len(group) < 2 or sum(x['fraction'] for x in group) < .70
                                for group in groups):
        return reject('Insufficient independent, spatially broad motion layers')
    center = np.array([(w-1)/2, (h-1)/2])
    points = np.array([[x, y, 1.] for x in (0., center[0], w-1.)
                       for y in (0., center[1], h-1.)])
    motions = [_power(x['matrix'], .5) for group in groups for x in group]
    parameters = np.array([_parameters(m, center) for m in motions])
    common = _matrix(np.median(parameters, axis=0), center)
    fields = np.array([(points@m.T-points)[:, :2] for m in motions])
    differences = fields-np.median(fields, axis=0)
    covariance = differences.reshape(-1, 2).T@differences.reshape(-1, 2)
    values, axes = np.linalg.eigh(covariance)
    direction = axes[:, -1]
    dominance = float(values[-1]/max(values.sum(), 1e-12))
    spread = float(np.max(np.linalg.norm(differences, axis=2)))
    diagnostics = {'ambiguity_direction': direction.tolist(),
                   'ambiguity_direction_fraction': dominance,
                   'motion_spread_analysis_pixels_per_frame': spread}
    if dominance < .98 or spread < .75*pixel:
        return reject('Motion uncertainty is not a substantial single direction', diagnostics=diagnostics)
    projector = np.eye(2)-np.outer(direction, direction)
    reliable = fields@projector
    reliable_spread = float(np.max(np.linalg.norm(reliable-np.median(reliable, axis=0), axis=2)))
    diagnostics['perpendicular_rate_spread_analysis_pixels'] = reliable_spread
    if reliable_spread > .15*pixel:
        return reject('Motion layers disagree in the proposed correction direction', diagnostics=diagnostics)
    # Compensate observed normal motion before removing the ambiguous direction.
    # This avoids mistaking an ordinary shared camera pan for a generation edit.
    compensated = common@np.asarray(registration, float)
    edit = np.eye(3)
    edit[:2] += projector@(compensated-np.eye(3))[:2]
    field = (points@edit.T-points)[:, :2]
    signal = float(np.max(np.linalg.norm(field, axis=1)))
    independent = []
    for matrix, pre_steps, post_steps in cross:
        candidate = _power(common, pre_steps)@matrix@_power(post_motion, post_steps)
        candidate_field = (points@candidate.T-points)[:, :2]@projector
        independent.append(float(np.max(np.linalg.norm(candidate_field-field, axis=1))))
    consistency = max(independent, default=float('inf'))
    diagnostics.update(cross_pair_disagreement_analysis_pixels=independent,
                       correction_peak_analysis_pixels=signal,
                       common_pre_motion=common.tolist(),
                       common_pre_parameters=np.median(parameters, axis=0).tolist(),
                       post_motion=np.asarray(post_motion).tolist(), projector=projector.tolist())
    if len(independent) < 3 or consistency > .75*pixel or signal < max(2*pixel, 5*consistency):
        return reject('Projected edit is not independently stable above motion uncertainty', diagnostics=diagnostics)
    singular = np.linalg.svd(edit[:2, :2], compute_uv=False)
    shift = edit[:2, :2]@center+edit[:2, 2]-center
    angle = abs(math.degrees(math.atan2(edit[1, 0], edit[0, 0])))
    if (np.linalg.det(edit[:2, :2]) <= 0 or np.max(abs(singular-1)) > bounds['max_scale_change'] or
            singular.max()/singular.min() > bounds['max_anisotropy'] or
            angle > bounds['max_rotation_degrees'] or
            np.linalg.norm(shift/[w, h]) > bounds['max_center_shift_fraction']):
        return reject('Projected edit exceeds conservative global affine bounds', diagnostics=diagnostics)
    supported = np.median(parameters, axis=0)
    supported[2:] = projector@supported[2:]
    return {'accepted': True, 'reason': 'Only the independently supported framing component is corrected',
            'edit_matrix': edit.tolist(), 'supported_rate': supported.tolist(),
            'diagnostics': diagnostics,
            'limitation': 'Mixed scene motion leaves framing in one direction unresolved; rate easing is disabled.'}


def recover_partial_edit(frames, cut, seam_fit, *, pair_fit, upscale=None, options=None, cancelled=None):
    """Conservative fallback for an accepted seam with an unreliable pre-rate.

    ``frames`` maps absolute source indices to RGB analysis pictures.
    ``pair_fit(left, right)`` returns ordinary global registration evidence.
    ``upscale`` maps those analysis coordinates to native source coordinates.
    Call only after the normal camera-rate path fails. An accepted edit already
    includes measured motion compensation: assemble it with zero pre/post rates
    and ``ease_rate=False``. Keep raw registration/rates in the evidence report.
    """
    reject = lambda reason, **data: {'accepted': False, 'reason': reason, **data}
    if not seam_fit.get('accepted') or not seam_fit.get('scene_consistent'):
        return reject('Original global seam registration is unreliable')
    needed = range(cut-7, cut+7)
    if any(index not in frames for index in needed):
        return reject('Partial framing needs three independent motion intervals on each side')
    h, w = frames[cut].shape[:2]
    pixel = max(w, h)/640
    evidence = seam_fit.get('diagnostics', {})
    if (min(w, h) < 128 or evidence.get('inlier_fraction', 0) < .90 or
            evidence.get('coverage', 0) < .75 or
            evidence.get('reprojection_p90_analysis_pixels', float('inf')) > 1.75*pixel or
            evidence.get('quality', {}).get('gradient_ncc', 0) < .85):
        return reject('Partial framing requires unusually strong full-frame seam alignment')
    def check():
        if cancelled is not None and cancelled():
            raise InterruptedError('Partial framing analysis cancelled')
    try:
        groups = []
        for a in (cut-7, cut-5, cut-3):
            check()
            groups.append(_motion_models(frames[a], frames[a+2]))
        if any(len(group) < 2 or sum(x['fraction'] for x in group) < .70 for group in groups):
            return reject('Insufficient independent, spatially broad motion layers', motion_layers=groups)
        post = []
        center = np.array([(w-1)/2, (h-1)/2])
        for a in (cut, cut+2, cut+4):
            check()
            fit = pair_fit(frames[a], frames[a+2])
            if fit.get('accepted'):
                post.append(_parameters(_power(np.linalg.inv(fit['matrix']), .5), center))
        if len(post) < 2:
            return reject('Independent incoming motion cannot be validated', motion_layers=groups)
        post_motion = _matrix(np.median(post, axis=0), center)
        cross = []
        cross_diagnostics = []
        for a, b in ((cut-3, cut), (cut-2, cut+1), (cut-1, cut+2)):
            check()
            fit = pair_fit(frames[a], frames[b])
            # Cross pairs can have mixed motion. Require distributed matching
            # evidence, then validate just their independently supported axis.
            diag = fit.get('diagnostics', {})
            if diag.get('matches', 0) < 50 or diag.get('coverage', 0) < .5:
                return reject('Independent cross-pair evidence is too sparse', motion_layers=groups)
            cross.append((np.asarray(fit['matrix']), cut-a, b-cut))
            cross_diagnostics.append({'frames': [a, b], 'matrix': fit['matrix']})
        result = _infer_edit(np.asarray(seam_fit['matrix']), groups, cross, post_motion, (h, w), options)
        result['motion_layers'] = groups
        result['cross_pairs'] = cross_diagnostics
        if result['accepted']:
            native = np.eye(3) if upscale is None else np.asarray(upscale, float)
            if native.shape != (3, 3) or not np.isfinite(native).all() or abs(np.linalg.det(native)) < 1e-12:
                return reject('Invalid analysis-to-native coordinate mapping')
            result['edit_matrix_analysis'] = result['edit_matrix']
            result['edit_matrix'] = (native@np.asarray(result['edit_matrix'])@np.linalg.inv(native)).tolist()
            result['supported_rate_analysis'] = result['supported_rate']
            rate = np.array(result['supported_rate'])
            rate[2:] = native[:2, :2]@rate[2:]
            result['supported_rate'] = rate.tolist()
        return result
    except (ValueError, np.linalg.LinAlgError, cv2.error) as exc:
        return reject('Partial framing evidence could not be validated: '+str(exc))
