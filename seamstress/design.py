"""Rebuild an experimental conform plan from explicit reviewed calibration.

This module does not estimate new cuts, camera motion, or color. Calibration
contains those measurements and the decision to exclude unreliable geometry.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
from fractions import Fraction

import numpy as np
from scipy.linalg import expm, logm
from scipy.optimize import brentq

from . import __version__
from .conform import _similarity, validate_conform_plan
from .media import probe
from .repair import fingerprint


ALGORITHM = "balanced-affine-neutral-return-v1"


def _positive_int(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f'{name} must be a positive integer')
    return value


def _matrix(parameters, center):
    scale, angle = np.exp(parameters[0]), parameters[1]
    result = np.eye(3)
    result[:2, :2] = scale*np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    result[:2, 2] = center+parameters[2:]-result[:2, :2]@center
    return result


def build_conform_plan(calibration, metadata, geometry_support=None, rate_support=None):
    """Pure matrix/LUT assembly; returned plan still requires visual review."""
    if not isinstance(calibration, dict) or calibration.get('schema_version') != 1 or calibration.get('method') != 'source_conform_calibration':
        raise ValueError('Expected schema_version 1 and method source_conform_calibration')
    if calibration.get('algorithm') != ALGORITHM:
        raise ValueError(f'Calibration algorithm must be {ALGORITHM}')
    source = calibration.get('source', {})
    if not isinstance(source, dict):
        raise ValueError('Calibration needs source metadata')
    for key in ('width', 'height', 'frame_count', 'fps_fraction'):
        if source.get(key) != metadata[key]:
            raise ValueError(f'Calibration source {key} differs from input')
    defaults = calibration.get('parameters', {})
    if not isinstance(defaults, dict):
        raise ValueError('Calibration parameters must be an object')
    geometry_support = _positive_int(geometry_support if geometry_support is not None else defaults.get('geometry_support', 168), 'geometry_support')
    rate_support = _positive_int(rate_support if rate_support is not None else defaults.get('rate_support', 12), 'rate_support')
    if rate_support > geometry_support:
        raise ValueError('rate_support must not exceed geometry_support')
    margin = defaults.get('source_margin_pixels', 2.)
    max_crop = defaults.get('max_view_crop_fraction_total_dimension', .08)
    if type(margin) not in (int, float) or not np.isfinite(margin) or not 0 <= margin <= 4:
        raise ValueError('source_margin_pixels must be finite and between zero and four')
    if type(max_crop) not in (int, float) or not np.isfinite(max_crop) or not 0 <= max_crop < .5:
        raise ValueError('max_view_crop_fraction_total_dimension must be finite in [0,0.5)')
    cuts = calibration.get('cuts')
    if not isinstance(cuts, list) or not cuts:
        raise ValueError('Calibration must contain reviewed cuts')
    count, width, height = metadata['frame_count'], metadata['width'], metadata['height']
    center = np.array([(width-1)/2, (height-1)/2])
    exclusions = calibration.get('excluded_geometry', [])
    if not isinstance(exclusions, list) or any(not isinstance(e, dict) or type(e.get('frame')) is not int or not isinstance(e.get('reason'), str) or not e['reason'].strip() for e in exclusions):
        raise ValueError('excluded_geometry requires frame and nonempty reason objects')
    excluded = {e['frame'] for e in exclusions}
    if len(excluded) != len(exclusions):
        raise ValueError('Excluded geometry frames must be unique')
    matrices = np.repeat(np.eye(3)[None], count, axis=0)
    previous_cut, previous_support_end = 0, -1
    supported, duration_records = [], []
    for record in cuts:
        if not isinstance(record, dict):
            raise ValueError('Each calibrated cut must be an object')
        cut = record.get('frame')
        if type(cut) is not int or not previous_cut < cut < count:
            raise ValueError('Calibration cuts must be sorted unique source frame indices')
        previous_cut = cut
        registration = _similarity(record.get('right_to_left_matrix'), f'cut {cut} registration', allow_affine=True)
        pre, post = np.asarray(record.get('pre_rate'), dtype=float), np.asarray(record.get('post_rate'), dtype=float)
        if pre.shape != (4,) or post.shape != (4,) or not np.isfinite(pre).all() or not np.isfinite(post).all():
            raise ValueError('Pre/post rates must contain four finite native-unit values')
        ease = record.get('ease_rate')
        if type(ease) is not bool:
            raise ValueError('ease_rate must be an explicit boolean review decision')
        if cut in excluded:
            continue
        before, after = min(geometry_support, cut-1), min(geometry_support, count-1-cut)
        if min(before, after) < 1:
            raise ValueError('Geometry corrections require a source frame on each side of the cut anchors')
        if cut-1-before <= previous_support_end:
            raise ValueError('Geometry supports overlap; reduce geometry_support or review the cuts')
        previous_support_end = cut+after
        delta = (pre-post) if ease else np.zeros(4)
        balanced = _matrix((pre+post)/2 if ease else pre, center)@registration
        logarithm = logm(balanced)
        if np.max(np.abs(np.imag(logarithm))) > 1e-9:
            raise ValueError(f'Cut {cut} does not admit a real supported affine interpolation')
        logarithm = np.real(logarithm)
        for side, anchor, duration in ((-1, cut-1, before), (1, cut, after)):
            for distance in range(duration+1):
                fraction = distance/duration
                weight = 1-(3*fraction*fraction-2*fraction**3)
                rate_fraction = np.clip(1-(distance+.5)/rate_support, 0, 1)
                offset = -delta*(rate_support/2)*(rate_fraction**3-.5*rate_fraction**4)
                matrices[anchor+side*distance] = _matrix(offset, center)@expm(side*.5*logarithm*weight)
        supported.append(cut)
        duration_records.append({'frame': cut, 'before_frames': before, 'after_frames': after})
    cut_indices = {r['frame'] for r in cuts}
    if not excluded <= cut_indices:
        raise ValueError('Every geometry exclusion must name a calibrated cut')
    corners = np.array([[0,0,1], [width-1,0,1], [0,height-1,1], [width-1,height-1,1]], dtype=float).T
    inverse = np.linalg.inv(matrices)
    def clearance(enlargement):
        view = _matrix(np.array([np.log(enlargement), 0, 0, 0]), center)
        points = inverse@(np.linalg.inv(view)@corners)
        return min(points[:,0].min()-margin, points[:,1].min()-margin,
                   width-1-margin-points[:,0].max(), height-1-margin-points[:,1].max())
    if clearance(1) >= 0:
        enlargement = 1.
    elif clearance(4) < 0:
        raise ValueError('No supported constant viewing crop covers all calibrated transforms')
    else:
        enlargement = float(brentq(clearance, 1, 4)+1e-6)
    crop = 1-1/enlargement
    if crop > max_crop:
        raise ValueError(f'Required total-dimension view crop {crop:.6f} exceeds calibrated limit {max_crop:.6f}')
    view = _matrix(np.array([np.log(enlargement), 0, 0, 0]), center)
    points = np.linalg.inv(view@matrices)@corners
    minimum = np.minimum.reduce([points[:,0].min(axis=1), points[:,1].min(axis=1),
                                 width-1-points[:,0].max(axis=1), height-1-points[:,1].max(axis=1)])
    # Recheck all generated matrices and protected LUTs using the renderer's
    # authoritative schema before publishing any plan.
    plan = {'schema_version': 3, 'method': 'source_conform', 'geometry_mode': 'affine',
            'source': metadata.copy(), 'source_sha256': calibration.get('source_sha256'),
            'seams': [{'frame': r['frame'], 'time': r['frame']/float(Fraction(metadata['fps_fraction']))} for r in cuts],
            'segments': [{'start': 0, 'end': count, 'matrix': np.eye(3).tolist(), 'gain': [1,1,1], 'bias': [0,0,0]}],
            'frame_matrices': matrices.tolist(), 'view_matrix': view.tolist(), 'edge_extension_pixels': 0,
            'grade_curves': calibration.get('grade_curves', []), 'unresolved_seams': sorted(excluded),
            'status': 'EXPERIMENTAL CALIBRATED CANDIDATE; perceptual review pending; not certified seamless',
            'notes': ['One original drawing per frame; no image interpolation, temporal crossfade, or generated poses.',
                      'Camera and color measurements come from explicit calibration; this command does not automatically validate them.',
                      'Geometry return support is shortened at source endpoints; exact durations are recorded in design_report.'],
            'design_report': {'algorithm': ALGORITHM, 'geometry_support': geometry_support, 'rate_support': rate_support,
                              'source_margin_pixels': margin, 'max_view_crop_fraction_total_dimension': max_crop,
                              'global_geometry_corrections': supported, 'excluded_geometry': exclusions,
                              'actual_geometry_supports': duration_records, 'constant_view_enlargement': enlargement,
                              'constant_view_crop_fraction_total_dimension': crop,
                              'minimum_native_source_clearance_pixels': float(minimum.min()),
                              'minimum_clearance_frame': int(minimum.argmin()),
                              'maximum_affine_singular_ratio': float(np.linalg.cond(matrices[:,:2,:2]).max())}}
    if 'local_color_curves' in calibration:
        plan['local_color_curves'] = calibration['local_color_curves']
    validate_conform_plan(plan, metadata)
    return plan


def design_conform(input, calibration, output, geometry_support=None, rate_support=None):
    input, calibration, output = map(Path, (input, calibration, output))
    if output.exists() or output.resolve() in (input.resolve(), calibration.resolve()):
        raise ValueError('Choose a new plan output distinct from source and calibration')
    data = calibration.read_bytes()
    measurements = json.loads(data)
    metadata = probe(input)
    if not isinstance(measurements, dict) or fingerprint(input) != measurements.get('source_sha256'):
        raise ValueError('Source fingerprint differs from calibration')
    plan = build_conform_plan(measurements, metadata, geometry_support, rate_support)
    plan['generator'] = {'name': 'seamstress.design-conform', 'version': __version__, 'algorithm': ALGORITHM,
                         'calibration_sha256': hashlib.sha256(data).hexdigest()}
    payload = json.dumps(plan, indent=2, allow_nan=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='seamstress-design-', dir=output.parent) as temporary:
        staged = Path(temporary)/'plan.json'; staged.write_text(payload)
        os.link(staged, output)
    return {'plan_path': str(output.resolve()), 'source_sha256': plan['source_sha256'],
            'plan_sha256': hashlib.sha256(payload.encode()).hexdigest(), 'status': plan['status'],
            'unresolved_seams': plan['unresolved_seams'], **plan['design_report']}
