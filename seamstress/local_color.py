"""Optional color/position RBF correction of an individual source drawing.

RGB is measured after the conform renderer's geometry and protected grade.
XY denotes native output pixel coordinates normalized to [0, 1]. Centers are
already divided by feature_scale, in RGBXY order. No neighboring frame or
spatial resampling is involved. Models require separate perceptual review.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

import numpy as np


ALGORITHM = 'hybrid-rbf-headroom-v1'
V1_RESPONSE = 'headroom-v1'
V2_RESPONSE = 'directional-gamut-v2'
MAX_CENTERS = 512
MAX_BASIS_ENTRIES = 2_097_152


@dataclass(frozen=True)
class _Model:
    scale: np.ndarray
    centers: np.ndarray
    coefficients: np.ndarray
    limit: float
    response: str


def _array(value, name):
    try:
        raw = np.asarray(value)
        if raw.dtype.kind not in 'iuf':
            raise ValueError('Expected real numeric values')
        result = raw.astype(np.float64)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError(f'{name} must contain finite numbers') from exc
    if not np.isfinite(result).all():
        raise ValueError(f'{name} must contain finite numbers')
    return result


def _compile_model(model, name='local color model'):
    if isinstance(model, _Model):
        return model
    if not isinstance(model, dict) or model.get('mode') != 'hybrid':
        raise ValueError(f'{name}.mode must be hybrid')
    response = model.get('response', V1_RESPONSE)
    if not isinstance(response, str) or response not in (V1_RESPONSE, V2_RESPONSE):
        raise ValueError(f'{name}.response must be {V1_RESPONSE} or {V2_RESPONSE}')
    scale = _array(model.get('feature_scale'), f'{name}.feature_scale')
    # These limits prevent squared float32 feature distances from overflowing;
    # they are far outside useful photographic color/position bandwidths.
    if scale.shape != (5,) or np.any(scale <= 0) or np.any(scale > 1e15) or np.any(scale < 1e-12):
        raise ValueError(f'{name}.feature_scale needs five positive finite scales in [1e-12, 1e15]')
    centers = _array(model.get('centers'), f'{name}.centers')
    if centers.ndim != 2 or centers.shape[1] != 5 or not 1 <= len(centers) <= MAX_CENTERS or np.any(abs(centers) > 1e15):
        raise ValueError(f'{name}.centers must be 1 to {MAX_CENTERS} rows of five finite scaled features within 1e15')
    coefficients = _array(model.get('coefficients'), f'{name}.coefficients')
    if coefficients.shape != (len(centers), 3) or np.any(abs(coefficients) > 1e12):
        raise ValueError(f'{name}.coefficients must have one finite RGB row per center, within 1e12')
    limit = model.get('limit')
    # |correction| <= limit*4*c/255*(1-c/255). limit <= 255/4
    # therefore keeps every c in [0,255] in range without hard clipping.
    if type(limit) not in (int, float) or not 1e-12 <= limit <= 255/4:
        raise ValueError(f'{name}.limit must be finite and between 1e-12 and 63.75')
    # v2's fitting prototype evaluates coefficients and the three-channel
    # response in float64. Retaining that precision makes final quantization
    # agree even for outputs close to half a channel level. v1 is unchanged.
    coefficient_type = np.float64 if response == V2_RESPONSE else np.float32
    return _Model(scale.astype(np.float32), centers.astype(np.float32),
                  coefficients.astype(coefficient_type), float(limit), response)


def validate_local_color_curves(curves, frame_count):
    """Validate original-frame supports and bounded hybrid color models."""
    if not isinstance(curves, list):
        raise ValueError('local_color_curves must be a list')
    previous_end = -1
    for i, curve in enumerate(curves):
        if not isinstance(curve, dict):
            raise ValueError('Each local color curve must be an object')
        cut, before, after = (curve.get(key) for key in ('frame', 'support_before', 'support_after'))
        if any(type(v) is not int for v in (cut, before, after)) or not 0 < cut < frame_count or min(before, after) < 1:
            raise ValueError('Local color curves need a valid cut and positive integer support lengths')
        if cut-1-before <= previous_end:
            raise ValueError('Local color curve supports must be ordered and cannot overlap or begin before source')
        previous_end = cut+after
        for side in ('left', 'right'):
            _compile_model(curve.get(side), f'local_color_curves[{i}].{side}')


def prepare_local_color_curves(curves, frame_count):
    """Validate once and reuse small compiled model arrays throughout a render."""
    validate_local_color_curves(curves, frame_count)
    return [{**curve, 'left': _compile_model(curve['left']), 'right': _compile_model(curve['right'])}
            for curve in curves]


def local_color_at(frame, curves):
    """Return model/strength at an original frame, or None at zero support.

    Both anchors (cut-1 and cut) have strength 1. Each outer endpoint has
    strength 0. The quintic envelope matches conform.tone_lut_at exactly.
    """
    for curve in curves:
        cut = curve['frame']
        distance = ((cut-1-frame)/curve['support_before'] if frame < cut
                    else (frame-cut)/curve['support_after'])
        if 0 <= distance < 1:
            t = 1-distance
            return curve['left' if frame < cut else 'right'], t*t*t*(10+t*(-15+6*t))
    return None


def _apply_samples(rgb, xy, model, weight):
    c = rgb.astype(np.float32)
    values = np.concatenate([c, xy], axis=1).astype(np.float32)/model.scale
    centers = model.centers
    distances = np.maximum(np.sum(values*values, axis=1)[:, None]
                           + np.sum(centers*centers, axis=1)[None, :]
                           - 2*values@centers.T, 0)
    nearest = distances.min(axis=1)
    basis = np.exp(-.5*(distances-nearest[:, None]))
    basis /= np.maximum(basis.sum(axis=1)[:, None], 1e-9)
    confidence = np.exp(-np.maximum(nearest-3, 0)/3)
    basis *= confidence[:, None]
    raw = basis@model.coefficients
    if model.response == V2_RESPONSE:
        c = c.astype(np.float64)
        low = np.mean(c*c, axis=1, keepdims=True)
        high = np.mean((255-c)**2, axis=1, keepdims=True)
        gate = (1-np.exp(-low/64))*(1-np.exp(-high/64))
        offset = model.limit*np.tanh(raw/model.limit)*gate
        available = np.where(offset >= 0, 255-c, c)
        correction = available*np.tanh(offset/np.maximum(available, 1e-12))
    else:
        correction = model.limit*np.tanh(raw/model.limit)*4*(c/255)*(1-c/255)
    return c+weight*correction


def _weight(value):
    if type(value) not in (int, float) or not 0 <= value <= 1:
        raise ValueError('Local color weight must be finite and in [0, 1]')
    return value


def apply_samples(rgb, xy, model, weight=1.):
    """Evaluate unrounded RGB samples for diagnostics, using normalized XY."""
    weight = _weight(weight)
    c, positions = _array(rgb, 'rgb'), _array(xy, 'xy')
    if c.ndim != 2 or c.shape[1] != 3 or positions.shape != (len(c), 2):
        raise ValueError('rgb and xy must have shapes N by 3 and N by 2')
    if np.any(c < 0) or np.any(c > 255) or np.any(positions < 0) or np.any(positions > 1):
        raise ValueError('RGB must be in [0,255] and native normalized XY in [0,1]')
    compiled = _compile_model(model)
    result_type = np.float64 if compiled.response == V2_RESPONSE else np.float32
    if weight == 0 or not len(c):
        return c.astype(result_type)
    # Diagnostic calls can also contain millions of pixels. Bound their basis
    # matrices just like full-frame rendering, preserving response precision.
    size = min(32768, MAX_BASIS_ENTRIES//len(compiled.centers))
    output = np.empty(c.shape, dtype=result_type)
    for start in range(0, len(c), size):
        output[start:start+size] = _apply_samples(c[start:start+size], positions[start:start+size], compiled, weight)
    return output


def apply_image(image, model, weight=1., chunk_size=32768):
    """Apply color only, with bounded temporary storage and one final rounding."""
    weight = _weight(weight)
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) < 1:
        raise ValueError('Local color image must be a nonempty uint8 RGB array')
    if type(chunk_size) is not int or chunk_size < 1:
        raise ValueError('chunk_size must be a positive integer')
    if weight == 0:
        return image
    compiled = _compile_model(model)
    if not np.any(compiled.coefficients):
        return image
    height, width = image.shape[:2]
    rgb = image.reshape(-1, 3)
    output = np.empty_like(rgb)
    size = min(chunk_size, MAX_BASIS_ENTRIES//len(compiled.centers))
    for start in range(0, len(rgb), size):
        indices = np.arange(start, min(start+size, len(rgb)))
        # Coordinates remain relative to the complete native output image,
        # including in previews and at chunk boundaries.
        xy = np.column_stack((indices % width, indices//width))/[max(width-1, 1), max(height-1, 1)]
        corrected = _apply_samples(rgb[start:start+size], xy, compiled, weight)
        output[start:start+size] = np.rint(np.clip(corrected, 0, 255)).astype(np.uint8)
    return output.reshape(image.shape)


def apply_local_color_at(image, frame, curves):
    active = local_color_at(frame, curves)
    return image if active is None else apply_image(image, *active)


def local_color_provenance(curves):
    """Fingerprint the exact original JSON model payloads, before compilation."""
    def digest(value):
        data = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
        return hashlib.sha256(data).hexdigest()
    responses = {curve[side].get('response', V1_RESPONSE) for curve in curves for side in ('left', 'right')}
    algorithm = (ALGORITHM if responses <= {V1_RESPONSE} else
                 'hybrid-rbf-directional-gamut-v2' if responses == {V2_RESPONSE} else
                 'hybrid-rbf-mixed-responses-v2')
    return {'algorithm': algorithm, 'feature_order': 'RGBXY',
            'coordinate_system': 'native output XY normalized by width-1 and height-1',
            'application_order': 'after geometry, segment grade, and protected grade',
            'curves_sha256': digest(curves),
            'models': [{'frame': curve['frame'],
                        'left_response': curve['left'].get('response', V1_RESPONSE),
                        'right_response': curve['right'].get('response', V1_RESPONSE),
                        'left_sha256': digest(curve['left']),
                        'right_sha256': digest(curve['right'])} for curve in curves]}
