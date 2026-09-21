"""Validated, portable choices for correcting an individual source seam.

Manual measurements are global affine geometry and camera rates in native
pixels. They do not authorize output warps, frame blending, or relaxed crop
safety. Retaining them in automatic/off mode permits reversible UI toggles.
"""
from __future__ import annotations

import re
import math

import numpy as np

from .conform import _similarity


DEFAULT_CORRECTION = {
    'geometry': 'auto', 'partial_recovery': True, 'endpoint_recovery': True,
    'cadence': True, 'rate_easing': True, 'color': 'auto',
}


def _numbers(value, shape, name):
    def numeric(item):
        if isinstance(item, (list, tuple)):
            return all(numeric(part) for part in item)
        try:
            return type(item) in (int, float) and math.isfinite(item)
        except (TypeError, OverflowError):
            return False
    if not isinstance(value, (list, tuple)) or not numeric(value):
        raise ValueError(f'{name} must contain only finite numbers')
    try:
        result = np.asarray(value, dtype=float)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError(f'{name} has an invalid shape') from exc
    if result.shape != shape:
        raise ValueError(f'{name} must have shape {shape}')
    return result


def _hash(value, name):
    if not isinstance(value, str) or not re.fullmatch('[0-9a-f]{64}', value):
        raise ValueError(f'{name} must be a lowercase SHA-256 digest')
    return value


def normalize_correction(value=None, *, metadata=None, source_sha256=None, frame=None):
    """Return explicit defaults and a detached, strictly validated manual recipe.

When source metadata, fingerprint, or seam frame are available, additionally
check native-unit bounds and imported measurement provenance before analysis.
"""
    if value is None:
        value = {}
    if not isinstance(value, dict):
        raise ValueError('correction must be an object')
    unknown = set(value)-set(DEFAULT_CORRECTION)-{'manual'}
    if unknown:
        raise ValueError('Unknown correction fields: '+', '.join(sorted(map(str, unknown))))
    result = {**DEFAULT_CORRECTION, **value}
    for field in ('partial_recovery', 'endpoint_recovery', 'cadence', 'rate_easing'):
        if type(result[field]) is not bool:
            raise ValueError(f'correction {field} must be boolean')
    for field, choices in [('geometry', ('auto', 'off', 'manual')), ('color', ('auto', 'tone', 'off'))]:
        if not isinstance(result[field], str) or result[field] not in choices:
            raise ValueError(f'correction {field} must be one of '+', '.join(choices))
    if 'manual' not in result:
        if result['geometry'] == 'manual':
            raise ValueError('manual geometry requires manual measurements')
        return result
    manual = result['manual']
    required = {'right_to_left_matrix', 'pre_rate', 'post_rate', 'ease_rate'}
    if not isinstance(manual, dict) or not required <= set(manual) or set(manual)-required-{'provenance'}:
        raise ValueError('manual requires matrix, pre_rate, post_rate, ease_rate and optional provenance only')
    matrix = _numbers(manual['right_to_left_matrix'], (3, 3), 'manual right_to_left_matrix')
    _similarity(matrix, 'manual right_to_left_matrix', allow_affine=True)
    dimensions = np.array([100000., 100000.])
    if metadata is not None:
        if not isinstance(metadata, dict) or any(type(metadata.get(k)) is not int or metadata[k] < 1
                                                 for k in ('width', 'height')):
            raise ValueError('correction metadata requires positive integer width and height')
        dimensions = np.array([metadata['width'], metadata['height']], dtype=float)
    center = (dimensions-1)/2 if metadata is not None else np.zeros(2)
    if np.any(abs(matrix[:2, :2]@center+matrix[:2, 2]-center) > dimensions*2):
        raise ValueError('manual matrix translation exceeds native-dimension bounds')
    cleaned = {'right_to_left_matrix': matrix.tolist()}
    for field in ('pre_rate', 'post_rate'):
        rate = _numbers(manual[field], (4,), 'manual '+field)
        if np.any(abs(rate) > np.r_[.1, .1, dimensions*.25]):
            raise ValueError(f'manual {field} exceeds bounded native camera rates')
        cleaned[field] = rate.tolist()
    if type(manual['ease_rate']) is not bool:
        raise ValueError('manual ease_rate must be boolean')
    cleaned['ease_rate'] = manual['ease_rate']
    if 'provenance' in manual:
        provenance = manual['provenance']
        keys = {'kind', 'label', 'source_sha256', 'frame'}
        if not isinstance(provenance, dict) or not keys <= set(provenance) or set(provenance)-keys-{'calibration_sha256'}:
            raise ValueError('manual provenance requires kind, label, source_sha256, frame and optional calibration_sha256 only')
        for field in ('kind', 'label'):
            if not isinstance(provenance[field], str) or not provenance[field].strip() or len(provenance[field]) > 256:
                raise ValueError(f'manual provenance {field} must be nonempty text of at most 256 characters')
        _hash(provenance['source_sha256'], 'manual provenance source_sha256')
        if 'calibration_sha256' in provenance:
            _hash(provenance['calibration_sha256'], 'manual provenance calibration_sha256')
        if type(provenance['frame']) is not int or provenance['frame'] < 1:
            raise ValueError('manual provenance frame must be a positive integer')
        if frame is not None and provenance['frame'] != frame:
            raise ValueError('manual provenance frame differs from the selected seam')
        if source_sha256 is not None and provenance['source_sha256'] != source_sha256:
            raise ValueError('manual provenance source fingerprint differs from the input')
        cleaned['provenance'] = dict(provenance)
    result['manual'] = cleaned
    return result
