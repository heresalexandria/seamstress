"""Whole-timeline, read-only edit suggestions with explicit review uncertainty.

An inexpensive translation-compensated scan nominates arbitrary-time peaks.
Optional periodic hints add search neighborhoods, never boundaries by themselves.
Dense flow is measured only at nominated pairs and a few neighboring controls.
Frames live in a temporary disk-backed array; only compact per-frame signals
accumulate in memory, never the full decoded pixel sequence.
"""
from __future__ import annotations

from contextlib import closing
from pathlib import Path
import math
import tempfile
import time
from collections.abc import Mapping

import cv2
import numpy as np
from scipy.ndimage import percentile_filter
from scipy.signal import find_peaks

from .media import iter_frames, probe


class DetectionCancelled(InterruptedError):
    """Raised after cancellation, with decoder and temporary files closed."""


def _check_cancelled(cancelled):
    if cancelled is not None and cancelled():
        raise DetectionCancelled('Seam detection cancelled.')


def _options(value):
    if value is not None and not isinstance(value, Mapping):
        raise ValueError('Detection options must be a mapping.')
    supplied = dict(value or {})
    allowed = {'interval_hints', 'sensitivity', 'min_spacing', 'scan_width'}
    unknown = set(supplied)-allowed
    if unknown:
        raise ValueError('Unknown detection options: '+', '.join(map(str, sorted(unknown, key=str))))
    result = {'interval_hints': [10., 15., 30.], 'sensitivity': .5,
              'min_spacing': 1., 'scan_width': 320}
    result.update(supplied)
    for name, lo, hi in [('sensitivity', 0., 1.), ('min_spacing', 0., 60.)]:
        x = result[name]
        if isinstance(x, bool) or not isinstance(x, (float, int)) or not math.isfinite(x) or not lo <= x <= hi:
            raise ValueError(f'{name} must be between {lo} and {hi}.')
        result[name] = float(x)
    width = result['scan_width']
    if isinstance(width, bool) or not isinstance(width, int) or not 64 <= width <= 960:
        raise ValueError('scan_width must be an integer between 64 and 960.')
    hints = result['interval_hints']
    if hints is None:
        hints = []
    if not isinstance(hints, (list, tuple)):
        raise ValueError('interval_hints must be a list of positive seconds.')
    if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or x <= 0 for x in hints):
        raise ValueError('interval_hints must contain positive finite seconds.')
    result['interval_hints'] = sorted(set(float(x) for x in hints))
    return result


def _proxy_size(width, height, longest):
    scale = min(1., longest/max(width, height))
    return max(1, round(width*scale)), max(1, round(height*scale))


def _fast_pair(a, b):
    """Translation compensation and appearance cues at a very small resolution."""
    x, y = a.astype(np.float32), b.astype(np.float32)
    ga, gb = [cv2.cvtColor(v, cv2.COLOR_RGB2GRAY) for v in (x, y)]
    shift, response = cv2.phaseCorrelate(ga, gb)
    dx, dy = shift
    h, w = ga.shape
    if not np.isfinite([dx, dy, response]).all() or response < .12 or abs(dx) > w*.2 or abs(dy) > h*.2:
        dx = dy = 0.
    registered = cv2.warpAffine(y, np.float32([[1, 0, -dx], [0, 1, -dy]]), (w, h),
                                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    margin_x, margin_y = min(w//4, math.ceil(abs(dx))+2), min(h//4, math.ceil(abs(dy))+2)
    delta = registered[margin_y:h-margin_y, margin_x:w-margin_x]-x[margin_y:h-margin_y, margin_x:w-margin_x]
    histogram = 0.
    for c in range(3):
        ha = np.bincount((a[..., c]//16).ravel(), minlength=16)/a.shape[0]/a.shape[1]
        hb = np.bincount((b[..., c]//16).ravel(), minlength=16)/b.shape[0]/b.shape[1]
        histogram += abs(ha-hb).sum()/6
    sharp_a = float(np.mean(abs(cv2.Laplacian(ga, cv2.CV_32F))))
    sharp_b = float(np.mean(abs(cv2.Laplacian(gb, cv2.CV_32F))))
    return [float(abs(y-x).mean()), float(abs(delta).mean()),
            float(abs(delta.mean((0, 1))).mean()), histogram,
            abs(sharp_b-sharp_a)/max(1., (sharp_a+sharp_b)/2), dx, dy]


def _novelty(signals, fps):
    # Upper-quartile context tolerates normal animation on twos and short holds.
    window = max(9, round(fps*1.5) | 1)
    features = signals[:, :5]
    typical = percentile_filter(features, percentile=75, size=(window, 1), mode='nearest')
    spread = percentile_filter(abs(features-typical), percentile=75, size=(window, 1), mode='nearest')
    floors = np.array([.30, .25, .20, .009, .055])
    excess = np.maximum(0., (features-typical)/(spread+floors))
    return np.minimum(excess, 30.)@np.array([.20, 1., .65, .65, .25])


def _pair_evidence(a, b):
    """Registration for measurement only; no output frame is transformed."""
    ga, gb = [cv2.cvtColor(im, cv2.COLOR_RGB2GRAY) for im in (a, b)]
    estimator = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
    estimator.setFinestScale(0)
    estimator.setGradientDescentIterations(16)
    estimator.setVariationalRefinementIterations(3)
    field = estimator.calc(ga, gb, None)
    h, w = ga.shape
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    rx, ry = xx+field[..., 0], yy+field[..., 1]
    valid = (rx > 3) & (rx < w-4) & (ry > 3) & (ry < h-4)
    valid[:3] = False
    valid[-3:] = False
    valid[:, :3] = False
    valid[:, -3:] = False
    left = cv2.GaussianBlur(a.astype(np.float32), (0, 0), .65)
    right = cv2.remap(cv2.GaussianBlur(b.astype(np.float32), (0, 0), .65), rx, ry, cv2.INTER_LINEAR)
    delta = right-left
    values = abs(delta[valid]).mean(1)
    if not len(values):
        return {'residual': 255., 'photometric': 255., 'structure': 255.,
                'registered_mae': 255., 'motion': 0., 'camera': [0.]*4}
    # Trim disocclusions, but retain enough support for broad redraw/scene cuts.
    keep = valid & (abs(delta).mean(2) <= np.percentile(values, 90))
    residual = float(abs(delta[keep]).mean())
    gradient = cv2.magnitude(cv2.Sobel(ga, cv2.CV_32F, 1, 0), cv2.Sobel(ga, cv2.CV_32F, 0, 1))
    flat = keep & (gradient < 28)
    if flat.sum() < 64:
        flat = keep
    tile_bias = []
    for y0 in range(0, h, max(1, h//4)):
        for x0 in range(0, w, max(1, w//4)):
            mask = flat[y0:y0+h//4, x0:x0+w//4]
            if mask.sum() >= 12:
                tile_bias.append(float(abs(np.mean(delta[y0:y0+h//4, x0:x0+w//4][mask], axis=0)).mean()))
    photometric = float(np.percentile(tile_bias, 75)) if tile_bias else residual
    structure = float(np.mean(abs(delta[keep]-np.median(delta[keep], axis=0))))
    magnitude = np.linalg.norm(field[valid], axis=1)
    motion = float(np.percentile(magnitude, 90))
    # A robust affine approximation summarizes global camera change, never repairs it.
    step = max(2, min(w, h)//16)
    coords = np.column_stack([xx[::step, ::step].ravel()-w/2, yy[::step, ::step].ravel()-h/2,
                              np.ones(xx[::step, ::step].size)])
    vectors = field[::step, ::step].reshape(-1, 2)
    weights = np.ones(len(vectors))
    coefficients = np.zeros((3, 2))
    for _ in range(3):
        coefficients = np.linalg.lstsq(coords*weights[:, None], vectors*weights[:, None], rcond=None)[0]
        error = np.linalg.norm(coords@coefficients-vectors, axis=1)
        weights = np.minimum(1., 1.5/np.maximum(error, .001))
    diagonal = math.hypot(w, h)
    camera = [*coefficients[2], (coefficients[0, 0]+coefficients[1, 1])*diagonal/2,
              (coefficients[0, 1]-coefficients[1, 0])*diagonal/2]
    normalizer = 1+.22*motion*(320/w)
    coherence = float(np.mean(np.linalg.norm(coords@coefficients-vectors, axis=1) < 1.5))
    return {'residual': residual/normalizer, 'photometric': photometric,
            'structure': structure/normalizer, 'registered_mae': residual,
            'motion': motion, 'camera': [float(x) for x in camera], 'camera_coherence': coherence}


def detect_seams(source: Path, *, progress=None, cancelled=None, options=None) -> dict:
    """Suggest editable first-incoming-frame boundaries across the whole source.

    ``progress`` receives dictionaries with stage/fraction/message and optional
    frame/total_frames. ``cancelled()`` is checked during decoding and analysis.
    Confidence is a heuristic review ranking, not a calibrated probability.
    ``scan_width`` bounds the longest proxy dimension and preserves aspect ratio.
    """
    start = time.monotonic()
    opts = _options(options)
    _check_cancelled(cancelled)
    source = Path(source).expanduser().resolve()
    metadata = probe(source)
    fps = metadata['fps']
    size = _proxy_size(metadata['width'], metadata['height'], opts['scan_width'])
    if min(size) < 16:
        raise ValueError('Source aspect ratio needs a larger scan_width to retain at least 16 pixels on its short side.')
    small_size = _proxy_size(*size, min(max(size), max(96, math.ceil(12*max(size)/min(size)))))

    def emit(stage, fraction, message, **extra):
        _check_cancelled(cancelled)
        if progress is not None:
            progress({'stage': stage, 'fraction': float(fraction), 'message': message, **extra})

    emit('scan', 0., 'Scanning the entire timeline for visual changes.', frame=0,
         total_frames=metadata['frame_count'])
    signals = []
    with tempfile.TemporaryDirectory(prefix='seamstress-detection-') as directory:
        cache = Path(directory)/'proxy.rgb'
        previous = None
        with cache.open('wb') as output, closing(iter_frames(source, size=size)) as frames:
            for i, frame in enumerate(frames):
                _check_cancelled(cancelled)
                output.write(frame.tobytes())
                small = cv2.resize(frame, small_size, interpolation=cv2.INTER_AREA)
                signals.append([0.]*7 if previous is None else _fast_pair(previous, small))
                previous = small
                if i % max(1, round(fps)) == 0:
                    emit('scan', .45*min(1., i/max(1, metadata['frame_count'])),
                         'Measuring appearance and camera motion.', frame=i,
                         total_frames=metadata['frame_count'])
        count = len(signals)
        probed_count = metadata['frame_count']
        metadata = {**metadata, 'frame_count': count, 'frame_count_estimated': False}
        diagnostics = {'method': 'whole-timeline-motion-normalized-v1', 'options': opts,
                       'proxy_size': list(size), 'decoded_frame_count': count,
                       'probed_frame_count': probed_count, 'temporary_proxy_bytes': cache.stat().st_size,
                       'confidence_interpretation': 'Heuristic review ranking; not a calibrated probability.',
                       'advisories': []}
        if count < 3:
            emit('complete', 1., 'Video has too few frames to identify interior boundaries.')
            return {'seams': [], 'metadata': metadata, 'diagnostics': diagnostics}
        data = np.asarray(signals, dtype=np.float64)
        novelty = _novelty(data, fps)
        min_peak = 2.1-(opts['sensitivity']-.5)*1.5
        peaks, _ = find_peaks(novelty, height=min_peak, distance=max(1, round(fps*.12)))
        # Keep arbitrary-time peaks independently from periodic search hints.
        budget = max(24, math.ceil(count/fps*2.5))
        ranked = sorted(peaks.tolist(), key=lambda n: (-novelty[n], n))[:budget]
        candidates = set(ranked)
        hard = np.flatnonzero((data[:, 0] > 28) & (data[:, 3] > .12))
        candidates.update(hard.tolist())
        periodic = set()
        half_window = max(2, round(fps*.65))
        for interval in opts['interval_hints']:
            # Bound very dense user hints: a prior should not turn every frame
            # into an expensive test. The normal whole-timeline scan remains.
            if interval < 2:
                diagnostics['advisories'].append(f'Interval hint {interval:g}s is too dense; using visual peaks only for that hint.')
                continue
            for t in np.arange(interval, count/fps, interval):
                anchor = round(t*fps)
                periodic.update(range(max(1, anchor-half_window), min(count, anchor+half_window+1)))
        candidates.update(periodic)
        candidates = {n for n in candidates if 1 <= n < count}
        offsets = sorted({max(1, round(fps*.17)), max(2, round(fps*.25))})
        needed = set(candidates)
        for n in candidates:
            needed.update(n+sign*d for d in offsets for sign in [-1, 1] if 1 <= n+sign*d < count)
        # The decoder is closed before mapping. Explicitly close the memmap on
        # every exit path so temporary storage is removable on Windows too.
        proxies = np.memmap(cache, mode='r', dtype=np.uint8, shape=(count, size[1], size[0], 3))
        evidence = {}
        try:
            for index, n in enumerate(sorted(needed)):
                _check_cancelled(cancelled)
                evidence[n] = _pair_evidence(proxies[n-1], proxies[n])
                if index % 8 == 0:
                    emit('candidates', .45+.48*index/max(1, len(needed)),
                         'Checking candidate changes against ordinary neighboring motion.',
                         frame=n, total_frames=count)
        finally:
            proxies._mmap.close()
            del proxies
        scored = []
        threshold = 2.5-(opts['sensitivity']-.5)*2.2
        sorted_needed = np.array(sorted(needed))
        for n in sorted(candidates):
            _check_cancelled(cancelled)
            current = evidence[n]
            lo = int(np.searchsorted(sorted_needed, max(1, n-round(fps*.38))))
            hi = int(np.searchsorted(sorted_needed, min(count-1, n+round(fps*.38)), side='right'))
            controls = [evidence[j] for j in sorted_needed[lo:hi] if abs(j-n) >= 2]
            if not controls:
                continue
            gains = {}
            for metric, floor in [('residual', .45), ('photometric', .65), ('structure', .45)]:
                ordinary = float(np.percentile([v[metric] for v in controls], 75))
                gains[metric] = max(0., (current[metric]-ordinary)/(ordinary+floor))
            camera = np.asarray([v['camera'] for v in controls])
            camera_deviation = float(np.linalg.norm(np.asarray(current['camera'])-np.median(camera, axis=0)))
            ordinary_motion = float(np.percentile([v['motion'] for v in controls], 75))
            geometry = max(0., camera_deviation/(ordinary_motion+1.)-1.)
            score = (1.5*gains['residual']+1.1*gains['photometric']+.35*gains['structure']
                     +.30*min(geometry, 5.)+.10*min(novelty[n], 6.))
            unrelated = data[n, 0] > 28 and data[n, 3] > .12 and current['registered_mae'] > 10
            # Large ordinary motion creates disocclusions and less trustworthy
            # flow residuals. Require stronger evidence than for a held camera.
            motion_uncertainty = max(1., current['motion']/max(4., max(size)*.025))
            score /= math.sqrt(motion_uncertainty)
            rigid_jump = (geometry > 2. and current.get('camera_coherence', 0.) > .85
                          and camera_deviation > max(1.5, max(size)*.006))
            if rigid_jump:
                score = max(score, threshold+.2+min(geometry-2., 3.)*.35)
            if unrelated:
                score = max(score, 5.+min(current['registered_mae']/20, 4.))
            visual_evidence = current['residual'] > .6 or current['photometric'] > .8 or rigid_jump
            # Timing alone has no score. A prior is only a small tie breaker
            # after an independently measurable appearance discontinuity.
            prior = n in periodic and score >= threshold*.8 and visual_evidence
            if prior:
                score += .08
            if score < threshold or not visual_evidence:
                continue
            reasons = []
            if gains['photometric'] > .35:
                reasons.append('Matched surfaces change color more than neighboring frames.')
            if gains['residual'] > .35:
                reasons.append('Motion-compensated appearance has an isolated discontinuity.')
            if geometry > .6:
                reasons.append('Camera framing or scale changes abruptly relative to nearby motion.')
            if unrelated:
                reasons.append('Large appearance and structural change suggests a different scene.')
            advisories = ['Review this suggestion at normal playback speed.']
            classification = 'scene_cut' if unrelated else 'continuation'
            if unrelated:
                advisories.append('This may be an intentional or unrelated scene cut; seamless correction may be inappropriate.')
            if rigid_jump and gains['photometric'] < .35 and gains['residual'] < .35:
                advisories.append('Geometry-only change: an intentional camera acceleration can produce similar evidence.')
            if n < 5 or n >= count-5:
                advisories.append('Near the video endpoint; little context is available for correction.')
            confidence = min(.98, max(.35, .48+.13*(score-threshold)))
            if motion_uncertainty > 1.5 and not unrelated:
                confidence = min(confidence, .58)
                advisories.append('Fast motion or disocclusion makes this match less reliable.')
            if confidence < .60:
                classification = 'uncertain' if not unrelated else classification
                advisories.append('Low-confidence candidate: motion, occlusion, or lighting could explain the change.')
            scored.append({'frame': int(n), 'time': float(n/fps), 'confidence': float(confidence),
                           'score': float(score), 'reasons': reasons or ['An isolated visual change exceeds its local motion baseline.'],
                           'enabled': True, 'origin': 'detected', 'classification': classification,
                           'review_advisories': advisories, 'metrics': {**current, 'appearance_excess': gains,
                           'camera_deviation': camera_deviation, 'fast_novelty': float(novelty[n]),
                           'motion_uncertainty': motion_uncertainty, 'rigid_camera_jump': bool(rigid_jump),
                           'periodic_hint_support': bool(prior)}})
        spacing = max(1, round(opts['min_spacing']*fps))
        selected = []
        for item in sorted(scored, key=lambda v: (-v['score'], v['frame'])):
            if all(abs(item['frame']-other['frame']) >= spacing for other in selected):
                selected.append(item)
        selected.sort(key=lambda v: v['frame'])
        diagnostics.update({'visual_peak_count': len(ranked), 'periodic_search_pair_count': len(periodic),
                            'candidate_pair_count': len(candidates), 'registered_pair_count': len(evidence),
                            'pre_suppression_proposals': len(scored), 'score_threshold': threshold,
                            'elapsed_seconds': time.monotonic()-start,
                            'scan_signals': {'raw_mae': data[:, 0].tolist(), 'translation_residual': data[:, 1].tolist(),
                                             'color_step': data[:, 2].tolist(), 'histogram_distance': data[:, 3].tolist(),
                                             'novelty': novelty.tolist()}})
    emit('complete', 1., f'Found {len(selected)} editable seam suggestions.', frame=count, total_frames=count)
    return {'seams': selected, 'metadata': metadata, 'diagnostics': diagnostics}
