"""Opt-in interpolation of excess startup holds without changing source time.

This is deliberately not a video-wide cadence normalizer. Only a contiguous
near-duplicate run immediately after a known seam can be filled. The first
normal held interval and every original moving frame remain at their indices.
"""
from __future__ import annotations

import cv2
import numpy as np


def _difference(a: np.ndarray, b: np.ndarray) -> dict:
    def proxy(frame):
        height, width = frame.shape[:2]
        if width > 640:
            frame = cv2.resize(frame, (640, round(height * 640 / width)),
                               interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        return cv2.GaussianBlur(gray, (5, 5), 1.1).astype(np.float32)

    difference = np.abs(proxy(a) - proxy(b))
    return {
        'mean': float(difference.mean()),
        'p99': float(np.percentile(difference, 99)),
        'changed_fraction': float(np.mean(difference > 5)),
    }


def analyze_startup_hold(frames, cut: int, max_hold: int = 12) -> dict:
    """Describe near-duplicate startup intervals using compression-tolerant metrics.

    A one-frame hold is always preserved. Longer normal holds observed later in
    this same window are preserved too. Eligibility is a suggestion for an
    explicitly enabled repair, not proof that a pause was unintentional.
    """
    if not 0 <= cut < len(frames):
        raise ValueError('cut must name a frame in the supplied window')
    if max_hold < 1:
        raise ValueError('max_hold must be positive')
    metrics = [_difference(frames[i - 1], frames[i])
               for i in range(cut + 1, len(frames))]
    # Quantized compression noise can change the mean while linework is static.
    # Requiring all three conditions also preserves small localized animation.
    holds = [m['p99'] <= 6 and m['mean'] <= 1.25 and
             m['changed_fraction'] <= .016 for m in metrics]
    startup = 0
    while startup < len(holds) and holds[startup]:
        startup += 1
    first_motion = cut + startup + 1
    complete_runs = []
    run = 0
    for held in holds[startup + 1:]:
        if held:
            run += 1
        elif run:
            complete_runs.append(run)
            run = 0
    # Ignore a run cut off by the right edge of the window.
    normal = max(1, int(np.ceil(np.percentile(complete_runs, 75)))) if complete_runs else 1
    has_anchor = first_motion < len(frames) and startup <= max_hold
    excess = max(0, startup - normal) if has_anchor else 0
    reason = ('excess startup holds' if excess else
              'no moving anchor within hold limit' if not has_anchor and startup else
              'startup follows normal hold cadence')
    confidence = 'high' if len(complete_runs) >= 3 else 'medium' if complete_runs else 'low'
    return {
        'eligible': bool(excess), 'reason': reason,
        'startup_held_intervals': startup,
        'normal_held_intervals': normal,
        'excess_held_intervals': excess,
        'first_motion_offset': first_motion - cut if has_anchor else None,
        'baseline_hold_runs': complete_runs,
        'cadence_confidence': confidence,
        'startup_difference_metrics': metrics[:min(startup + 1, max_hold + 1)],
    }


def _inside(displacement):
    height, width = displacement.shape[:2]
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    x = x + displacement[..., 0]
    y = y + displacement[..., 1]
    return (x >= 0) & (x <= width - 1) & (y >= 0) & (y <= height - 1)


def _tween(a, b, forward, backward, alpha, confidence_a, confidence_b):
    from .repair import sample

    # Invert y = x + alpha * flow(x), rather than treating forward flow as an
    # exact inverse map. Repeated fixed-point sampling helps curved boundaries.
    gather_a = -alpha * forward
    gather_b = -(1 - alpha) * backward
    for _ in range(3):
        gather_a = -alpha * sample(forward, gather_a)
        gather_b = -(1 - alpha) * sample(backward, gather_b)
    warped_a = sample(a, gather_a, cv2.INTER_CUBIC).astype(np.float32)
    warped_b = sample(b, gather_b, cv2.INTER_CUBIC).astype(np.float32)
    ca = sample(confidence_a, gather_a) * _inside(gather_a)
    cb = sample(confidence_b, gather_b) * _inside(gather_b)
    wa, wb = (1 - alpha) * ca, alpha * cb
    total = wa + wb
    uncertain = (ca < .35) | (cb < .35)
    disagreement = np.max(np.abs(warped_a - warped_b), axis=2) > 28
    # At occlusions or unresolved outline disagreement, choose the better
    # supported side instead of averaging two incompatible edges.
    choose_a = wa >= wb
    fallback = np.where(choose_a[..., None], warped_a, warped_b)
    blend = ((warped_a * wa[..., None] + warped_b * wb[..., None]) /
             np.maximum(total[..., None], 1e-6))
    select = (total < .1) | (disagreement & uncertain)
    blend[select] = fallback[select]
    return np.clip(np.rint(blend), 0, 255).astype(np.uint8), float(np.mean(uncertain | select))


def fill_startup_hold(frames, cut: int, max_hold: int = 12):
    """Fill only excess initial duplicate samples, preserving count and endpoints.

    ``frames`` contains RGB uint8 arrays. The return value is ``(frames, report)``.
    All untouched arrays are returned unchanged. Enable this selectively: a
    freeze can be intentional, and optical-flow interpolation may fail where
    appearance changes without a reliable correspondence.
    """
    report = analyze_startup_hold(frames, cut, max_hold=max_hold)
    output = list(frames)
    report.update({'applied': False, 'replaced_offsets': [],
                   'uncertain_fraction': 0., 'source_time_shift': 0.})
    if not report['eligible']:
        return output, report
    from .repair import flow, sample

    normal = report['normal_held_intervals']
    first_motion = cut + report['first_motion_offset']
    anchor_index = cut + normal
    a, b = frames[anchor_index], frames[first_motion]
    forward, backward = flow(a, b), flow(b, a)
    error_a = np.linalg.norm(forward + sample(backward, forward), axis=2)
    error_b = np.linalg.norm(backward + sample(forward, backward), axis=2)
    confidence_a = np.exp(-np.square(error_a / 2)).astype(np.float32)
    confidence_b = np.exp(-np.square(error_b / 2)).astype(np.float32)
    uncertainties = []
    for i in range(anchor_index + 1, first_motion):
        alpha = (i - anchor_index) / (first_motion - anchor_index)
        output[i], uncertainty = _tween(a, b, forward, backward, alpha,
                                       confidence_a, confidence_b)
        uncertainties.append(uncertainty)
        report['replaced_offsets'].append(i - cut)
    report.update({
        'applied': bool(uncertainties),
        'uncertain_fraction': max(uncertainties, default=0.),
        'flow_consistent_fraction': float(np.mean((error_a < 2) & (error_b < 2))),
        'method': 'bidirectional flow interpolation of excess held samples',
        'requires_visual_review': bool(uncertainties),
    })
    return output, report
