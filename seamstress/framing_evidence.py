"""Identify reprojected continuation endpoints using local motion as evidence.

A global camera rate is not meaningful in many moving, layered scenes. A
repeated endpoint can nevertheless be identified: after accounting for one
global recrop, independently moving features have almost no temporal advance.
Local tracks identify that advance; they never become output image warps.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

from .motion_evidence import _coverage, _matches


def _phase_fit(source, target, velocity):
    """Fit affine(source) = target + phase * velocity, separating parallax.

    Removing the affine component of velocity makes phase identifiable only
    from independent scene motion. Pure affine camera motion is rank deficient.
    """
    basis = np.column_stack((source, np.ones(len(source))))
    residual_velocity = velocity-basis@np.linalg.lstsq(basis, velocity, rcond=None)[0]
    residual_target = target-basis@np.linalg.lstsq(basis, target, rcond=None)[0]
    information = float(np.sum(residual_velocity**2))
    if len(source) < 12 or information < 1e-12:
        return None
    phase = -float(np.sum(residual_velocity*residual_target))/information
    matrix = np.linalg.lstsq(basis, target+phase*velocity, rcond=None)[0].T
    error = basis@matrix.T-target-phase*velocity
    variance = float(np.sum(error**2))/max(1, 2*len(source)-7)
    return {'phase': phase, 'phase_standard_error': math.sqrt(variance/information),
            'non_affine_motion_rms': math.sqrt(information/(2*len(source))),
            'matrix': matrix, 'residual_p90': float(np.percentile(np.linalg.norm(error, axis=1), 90))}


def _infer_endpoint(left, right, pre_velocities, post_velocities, registration, shape, options=None,
                    *, stationary_velocities=None):
    """Pure validation of a repeated endpoint; velocities are pixels/frame.

    Inputs already passed forward/backward tracking and broad seam registration.
    Separate spatial folds and six temporal intervals must independently agree.
    """
    reject = lambda reason, **data: {'accepted': False, 'reason': reason, **data}
    h, w = shape
    pixel = max(w, h)/640
    left, right = np.asarray(left, float), np.asarray(right, float)
    registration = np.asarray(registration, float)
    if len(left) < 60 or left.shape != right.shape or len(pre_velocities) != 3 or len(post_velocities) != 3:
        return reject('Repeated endpoint needs at least60 independent tracks and three intervals per side')
    # Reject isolated tracking/matching failures using the independently measured
    # full-frame registration. A substantial discarded fraction fails closed.
    error = np.linalg.norm(right@registration[:2, :2].T+registration[:2, 2]-left, axis=1)
    keep = error <= 1.25*pixel
    if keep.mean() < .9 or keep.sum() < 60:
        return reject('Tracked features disagree with the full-frame seam registration')
    left, right = left[keep], right[keep]
    coverage = _coverage(left, w, h)
    span = np.ptp(left, axis=0)/[w, h]
    if coverage < .5 or span.min() < .5:
        return reject('Repeated-endpoint tracks lack independent spatial support')
    diagnostics = {'tracks': len(left), 'retained_fraction': float(keep.mean()),
                   'coverage': coverage, 'span': span.tolist(), 'intervals': []}
    # Checkerboard cells distribute both folds through the image, unlike a
    # split by feature strength (which can concentrate one fold on a subject).
    cells = np.floor(left/[w, h]*12).astype(int)
    folds = (cells[:, 0]+cells[:, 1]) % 2
    if min(np.sum(folds == 0), np.sum(folds == 1)) < 20:
        return reject('Repeated-endpoint spatial holdout is too small', diagnostics=diagnostics)
    for side, velocities in [('outgoing', pre_velocities), ('incoming', post_velocities)]:
        for number, velocity in enumerate(velocities):
            velocity = np.asarray(velocity, float)[keep]
            if side == 'incoming':
                velocity = velocity@registration[:2, :2].T
            fits = []
            for fold in (0, 1):
                train, test = folds == fold, folds != fold
                fitted = _phase_fit(right[train], left[train], velocity[train])
                if fitted is None:
                    return reject('Local motion cannot distinguish recrop from ordinary camera motion', diagnostics=diagnostics)
                matrix = fitted.pop('matrix')
                predicted = right[test]@matrix[:, :2].T+matrix[:, 2]
                heldout = np.linalg.norm(predicted-left[test]-fitted['phase']*velocity[test], axis=1)
                fitted['heldout_p90'] = float(np.percentile(heldout, 90))
                fits.append(fitted)
            diagnostics['intervals'].append({'side': side, 'interval': number, 'folds': fits})
            if any(fit['non_affine_motion_rms'] < .15*pixel for fit in fits):
                return reject('Local motion cannot distinguish recrop from ordinary camera motion', diagnostics=diagnostics)
            if any(abs(fit['phase']) > .3 or abs(fit['phase'])+2*fit['phase_standard_error'] > .5 for fit in fits):
                return reject('Tracked scene motion does not identify a repeated endpoint', diagnostics=diagnostics)
            if any(fit['heldout_p90'] > 1.25*pixel for fit in fits):
                return reject('Repeated-endpoint affine evidence fails spatial holdout', diagnostics=diagnostics)
            if abs(fits[0]['phase']-fits[1]['phase']) > .25:
                return reject('Independent scene regions disagree about endpoint timing', diagnostics=diagnostics)
    # A hold in local animation does NOT prove that the camera also held. For
    # example, a drawing animated on twos can sit under a continuously panning
    # camera. Require broad anchors observed nearly stationary on BOTH sides;
    # phase evidence alone must never authorize removal of affine camera motion.
    velocities = (np.asarray(stationary_velocities)[:, keep] if stationary_velocities is not None else
                  np.concatenate((np.asarray(pre_velocities)[:, keep],
                                  np.asarray(post_velocities)[:, keep]@registration[:2, :2].T)))
    stationary = np.max(np.linalg.norm(velocities, axis=2), axis=0) <= .25*pixel
    anchors = left[stationary]
    anchor_coverage = _coverage(anchors, w, h)
    anchor_span = np.ptp(anchors, axis=0)/[w, h] if len(anchors) else np.zeros(2)
    anchor_hull = (cv2.contourArea(cv2.convexHull(anchors.astype(np.float32)))/(w*h)
                   if len(anchors) >= 3 else 0.)
    diagnostics['stationary_anchors'] = {'count': len(anchors), 'coverage': anchor_coverage,
                                         'span': anchor_span.tolist(), 'hull_fraction': anchor_hull,
                                         'checked_intervals': len(velocities)}
    if (len(anchors) < 40 or stationary.mean() < .35 or anchor_coverage < .5 or
            anchor_span.min() < .6 or anchor_hull < .3):
        return reject('Local animation hold does not establish a stationary camera reference', diagnostics=diagnostics)
    corners = np.array([[0., 0.], [w-1., 0.], [0., h-1.], [w-1., h-1.]])
    signal = float(np.max(np.linalg.norm(corners@registration[:2, :2].T+registration[:2, 2]-corners, axis=1)))
    diagnostics['recrop_peak_analysis_pixels'] = signal
    if signal < 2*pixel:
        return reject('Repeated endpoint has no substantial framing reset', diagnostics=diagnostics)
    return {'accepted': True, 'reason': 'Independent scene motion identifies an affine-reprojected repeated endpoint',
            'edit_matrix': registration.tolist(), 'diagnostics': diagnostics,
            'limitation': 'Only the framing reset is repaired; original endpoint holds and depth-dependent motion remain.'}


def _recover_endpoint_edit(frames, cut, seam_fit, *, upscale=None, options=None, cancelled=None):
    """Recover a full recrop only for strongly identified repeated endpoints.

    Call after ordinary global camera-rate validation fails. Accepted output is
    already the edit matrix: assemble with zero rates and ease_rate=False.
    """
    reject = lambda reason, **data: {'accepted': False, 'reason': reason, **data}
    if not seam_fit.get('accepted') or not seam_fit.get('scene_consistent'):
        return reject('Original global seam registration is unreliable')
    if any(index not in frames for index in range(cut-7, cut+7)):
        return reject('Repeated endpoint needs six motion handles on each side')
    h, w = frames[cut].shape[:2]
    pixel = max(w, h)/640
    evidence = seam_fit.get('diagnostics', {})
    if (min(h, w) < 128 or evidence.get('inlier_fraction', 0) < .95 or
            evidence.get('coverage', 0) < .875 or
            evidence.get('reprojection_p90_analysis_pixels', float('inf')) > 1.1*pixel or
            evidence.get('quality', {}).get('gradient_ncc', 0) < .93):
        return reject('Repeated endpoint requires unusually strong full-frame seam alignment')
    registration = np.asarray(seam_fit['matrix'], float)
    bounds = {'max_scale_change': .1, 'max_anisotropy': 1.045,
              'max_rotation_degrees': 3., 'max_center_shift_fraction': .07}
    if options:
        bounds.update({key: options[key] for key in bounds if key in options})
    singular = np.linalg.svd(registration[:2, :2], compute_uv=False)
    center = np.array([(w-1)/2, (h-1)/2])
    shift = registration[:2, :2]@center+registration[:2, 2]-center
    if (np.linalg.det(registration[:2, :2]) <= 0 or
            np.max(abs(singular-1)) > bounds['max_scale_change'] or
            singular.max()/singular.min() > bounds['max_anisotropy'] or
            abs(math.degrees(math.atan2(registration[1, 0], registration[0, 0]))) > bounds['max_rotation_degrees'] or
            np.linalg.norm(shift/[w, h]) > bounds['max_center_shift_fraction']):
        return reject('Repeated-endpoint recrop exceeds conservative global affine bounds')
    def check():
        if cancelled is not None and cancelled():
            raise InterruptedError('Repeated-endpoint analysis cancelled')
    check()
    left, right = _matches(frames[cut-1], frames[cut])
    if len(left) < 60:
        return reject('Too few independent boundary correspondences')
    gray = {index: cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) for index, frame in frames.items()}
    valid = np.ones(len(left), bool)
    trajectories = []
    frame_velocities = []
    tracking = {'winSize': (21, 21), 'maxLevel': 3,
                'criteria': (cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT, 40, .001)}
    for anchor, points, sign in [(cut-1, left, -1), (cut, right, 1)]:
        positions = [points]
        # Camera stationarity needs EVERY frame. Checking only on-twos handles
        # would alias an alternating camera movement into a false static anchor.
        for distance in range(1, 7):
            check()
            tracked, status, error = cv2.calcOpticalFlowPyrLK(gray[anchor], gray[anchor+sign*distance], points, None, **tracking)
            if tracked is None or status is None:
                return reject('Endpoint trajectories cannot be tracked')
            reverse, back_status, _ = cv2.calcOpticalFlowPyrLK(gray[anchor+sign*distance], gray[anchor], tracked, None, **tracking)
            if reverse is None or back_status is None:
                return reject('Endpoint trajectories cannot be validated backwards')
            valid &= status[:, 0].astype(bool)&back_status[:, 0].astype(bool)
            valid &= (np.linalg.norm(points-reverse, axis=1) < .5*pixel)&(error[:, 0] < 20)
            valid &= np.isfinite(tracked).all(axis=1)&np.isfinite(reverse).all(axis=1)
            positions.append(tracked)
        trajectories.append(np.diff(positions[::2], axis=0)/(2*sign))
        frame_velocities.append(np.diff(positions, axis=0)/sign)
    stationary_velocities = np.concatenate((frame_velocities[0],
                                            frame_velocities[1]@registration[:2, :2].T))[:, valid]
    result = _infer_endpoint(left[valid], right[valid], trajectories[0][:, valid],
                             trajectories[1][:, valid], registration, (h, w), options,
                             stationary_velocities=stationary_velocities)
    if result['accepted']:
        native = np.eye(3) if upscale is None else np.asarray(upscale, float)
        if native.shape != (3, 3) or not np.isfinite(native).all() or abs(np.linalg.det(native)) < 1e-12:
            return reject('Invalid analysis-to-native coordinate mapping')
        result['edit_matrix_analysis'] = result['edit_matrix']
        result['edit_matrix'] = (native@np.asarray(result['edit_matrix'])@np.linalg.inv(native)).tolist()
    return result


def recover_endpoint_edit(frames, cut, seam_fit, *, upscale=None, options=None, cancelled=None):
    """Return bounded recrop evidence, rejecting numerical failures safely.

    An accepted matrix needs zero assembly rates and no rate easing. Cancellation
    remains an InterruptedError so the caller can stop the whole analysis.
    """
    try:
        return _recover_endpoint_edit(frames, cut, seam_fit, upscale=upscale,
                                      options=options, cancelled=cancelled)
    except (cv2.error, np.linalg.LinAlgError, ValueError, FloatingPointError, OverflowError) as exc:
        return {'accepted': False, 'reason': 'Repeated-endpoint evidence could not be validated',
                'diagnostic_error': type(exc).__name__}
