"""Recover camera-rate estimates aliased by a proven four-frame drawing cadence.

This changes an estimate only. It never resamples time, changes a source frame,
or relaxes a registration's acceptance criteria.
"""
from __future__ import annotations

import cv2
import numpy as np


def _parameters(matrix, center):
    linear = matrix[:2, :2]
    u, _, vt = np.linalg.svd(linear)
    rotation = u@vt
    return np.array([np.log(np.sqrt(np.linalg.det(linear))),
                     np.arctan2(rotation[1, 0], rotation[0, 0]),
                     *(linear@center+matrix[:2, 2]-center)])


def _grid(center):
    return np.array([(x, y) for y in (-center[1], 0., center[1])
                     for x in (-center[0], 0., center[0])])


def _rate_rms(parameters, grid):
    p = np.asarray(parameters)
    x = p[..., 0, None]*grid[:, 0]-p[..., 1, None]*grid[:, 1]+p[..., 2, None]
    y = p[..., 1, None]*grid[:, 0]+p[..., 0, None]*grid[:, 1]+p[..., 3, None]
    return np.sqrt(np.mean(x*x+y*y, axis=-1))


def recover_cadence_rate(frames, cut, side, *, pair_fit, upscale, center, cancelled=None):
    """Return a native rate only when six intervals prove an alternating cadence.

    ``pair_fit`` has calibrate_pair's right-to-left matrix contract. The caller
    retains its original rate when ``accepted`` is false, and must already have
    reliable ordinary motion evidence for this side of the seam.
    """
    if side not in ('pre', 'post'):
        raise ValueError('Cadence side must be pre or post')
    start = cut-13 if side == 'pre' else cut
    indices = list(range(start, start+13, 2))
    diagnostics = {'side': side, 'source_indices': indices}

    def reject(reason):
        return {'accepted': False, 'reason': reason, 'diagnostics': diagnostics}

    def check():
        if cancelled and cancelled():
            raise InterruptedError('Camera cadence analysis cancelled')

    check()
    if any(index not in frames for index in indices):
        return reject('Insufficient within-segment frames for cadence evidence')
    center = np.asarray(center, dtype=float)
    upscale = np.asarray(upscale, dtype=float)
    downscale = np.linalg.inv(upscale)
    grid = _grid(center)
    pixel = max(.5, float(np.min(2*center+1))*.001)
    matrices, rates, intervals = [], [], []

    def measure(a, b):
        check()
        try:
            fit = pair_fit(frames[a], frames[b])
            if not fit.get('accepted') or fit.get('confidence', 0.) < .75:
                return None
            matrix = upscale@np.asarray(fit['matrix'], dtype=float)@downscale
            if (matrix.shape != (3, 3) or not np.isfinite(matrix).all()
                    or not np.allclose(matrix[2], [0., 0., 1.])
                    or np.linalg.det(matrix[:2, :2]) <= 0):
                return None
            np.linalg.inv(matrix)
        except (ValueError, np.linalg.LinAlgError, cv2.error):
            return None
        return matrix

    for a, b in zip(indices, indices[1:]):
        backward = measure(a, b)
        if backward is None:
            return reject('Cadence needs six reliable, distributed motion intervals')
        matrices.append(backward)
        rates.append(_parameters(np.linalg.inv(backward), center)/(b-a))
        intervals.append([a, b])
    rates = np.asarray(rates)
    diagnostics['intervals'] = intervals
    diagnostics['interval_native_rates'] = rates.tolist()
    # Remove ordinary acceleration before asking whether the residual repeats
    # fast/slow on all three four-frame cycles. A median of three two-frame
    # velocities can otherwise select the fast phase twice and bias the rate.
    time = np.arange(6, dtype=float)-2.5
    phase = (-1.)**np.arange(6)
    design = np.column_stack([np.ones(6), time, phase])
    coefficients = np.linalg.lstsq(design, rates, rcond=None)[0]
    alternating = coefficients[2]
    amplitude = float(_rate_rms(alternating, grid))
    residual = _rate_rms(rates-design@coefficients, grid)
    detrended = _rate_rms(rates-design[:, :2]@coefficients[:2], grid)
    explained = 1-float(np.sum(residual**2))/max(float(np.sum(detrended**2)), 1e-12)
    # Second differences remove a linear trend independently of the fitted
    # intercept/slope. Their phase-adjusted vectors must agree in every triple.
    phase_vectors = np.array([phase[i]*(rates[i]-2*rates[i+1]+rates[i+2])/4 for i in range(4)])
    phase_error = _rate_rms(phase_vectors-alternating, grid)
    drift = float(_rate_rms(coefficients[1]*5, grid))
    diagnostics.update({'alternating_native_pixels_per_frame': amplitude,
                        'alternating_explained_fraction': explained,
                        'maximum_model_residual_native_pixels_per_frame': float(residual.max()),
                        'maximum_phase_disagreement_native_pixels_per_frame': float(phase_error.max()),
                        'trend_drift_native_pixels_per_frame': drift})
    if amplitude < pixel or explained < .9 or residual.max() > .25*amplitude or phase_error.max() > .25*amplitude:
        return reject('No strong repeatable cadence after ordinary acceleration is removed')
    # A long-window slope is not a seam-side derivative under strong camera
    # acceleration. Only recover when cadence dominates the measured drift;
    # retain the caller's shorter estimate for an accelerating camera.
    if drift > .5*amplitude:
        return reject('Camera acceleration is too large for a full-window cadence rate')
    # Verify every overlapping four-frame registration against its composition
    # of two shorter measurements. Alternating fitting errors must not become a
    # camera-rate correction simply because their velocity sequence oscillates.
    points = np.column_stack([grid+center, np.ones(len(grid))]).T
    errors = []
    for i, (a, b) in enumerate(zip(indices, indices[2:])):
        direct = measure(a, b)
        if direct is None:
            return reject('Independent four-frame camera evidence is unreliable')
        composed = matrices[i]@matrices[i+1]
        difference = ((direct-composed)@points)[:2]
        errors.append(float(np.sqrt(np.mean(np.sum(difference**2, axis=0)))))
    diagnostics['four_frame_composition_errors_native_pixels'] = errors
    if max(errors)/4 > .5*pixel:
        return reject('Four-frame motion disagrees with the alternating short intervals')
    # Compose the accepted transforms, express the trajectory relative to the
    # seam-side anchor, then fit its slope. This accounts for zoom/translation
    # composition and balances cadence phase across the complete 12-frame span.
    trajectory = [np.eye(3)]
    for backward in matrices:
        check()
        trajectory.append(np.linalg.inv(backward)@trajectory[-1])
    reference = trajectory[-1] if side == 'pre' else trajectory[0]
    inverse_reference = np.linalg.inv(reference)
    positions = np.array([_parameters(matrix@inverse_reference, center) for matrix in trajectory])
    times = np.asarray(indices, dtype=float)
    times -= times.mean()
    fit = np.linalg.lstsq(np.column_stack([np.ones(7), times]), positions, rcond=None)[0]
    rate = fit[1]
    diagnostics.update({'native_rate': rate.tolist(), 'trajectory_native_parameters': positions.tolist()})
    check()
    return {'accepted': True, 'rate': rate.tolist(),
            'reason': 'Repeated cadence and independent four-frame motion support a full-trajectory rate',
            'diagnostics': diagnostics}
