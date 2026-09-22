"""Opt-in native-frame reconstruction, isolated from the established renderer.

Accepted conform values are applied exactly once, after reconstruction. Plans
without reconstruction entries never import the reconstruction engine.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from .media import iter_frames


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def context_signature(recipe, start, end):
    """Bind a repair to only the original grading/framing it actually consumes."""
    segments = recipe['segments']
    matrices = recipe.get('frame_matrices')
    frame_context = []
    index = 0
    for number in range(start, end):
        while number >= segments[index]['end']:
            index += 1
        segment = segments[index]
        frame_context.append([matrices[number] if matrices is not None else segment['matrix'],
                              segment['gain'], segment['bias']])
    def intersecting(curves):
        return [row for row in curves
                if row['frame'] - 1 - row['support_before'] < end
                and row['frame'] + row['support_after'] >= start]
    return digest_json({
        'source': recipe['source_sha256'], 'range': [start, end],
        'view': recipe.get('view_matrix', np.eye(3).tolist()),
        'edge': recipe.get('edge_extension_pixels', 0), 'frames': frame_context,
        'grade': intersecting(recipe.get('grade_curves', [])),
        'local': intersecting(recipe.get('local_color_curves', [])),
    })


def validate_entries(recipe, metadata):
    entries = recipe.get('reconstructions', [])
    if not isinstance(entries, list) or len(entries) > 10000:
        raise ValueError('reconstructions must be a bounded list')
    previous = -1
    seen = set()
    for row in entries:
        if not isinstance(row, dict):
            raise ValueError('Reconstruction entries must be objects')
        frame, start, end = (row.get(key) for key in ('frame', 'start', 'end'))
        if (any(type(n) is not int for n in (frame, start, end))
                or not 0 <= start < frame < end <= metadata['frame_count']):
            raise ValueError('Reconstruction must span one valid incoming seam frame')
        if frame in seen or start < previous:
            raise ValueError('Reconstruction windows must be unique, ordered and nonoverlapping')
        previous = end
        seen.add(frame)
        if not isinstance(row.get('manifest'), str) or not Path(row['manifest']).is_absolute():
            raise ValueError('Reconstruction manifest must have an absolute local path')
        for key in ('sha256', 'contextSha256'):
            value = row.get(key)
            if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
                raise ValueError(f'Reconstruction {key} must be a SHA-256 digest')
        if row['contextSha256'] != context_signature(recipe, start, end):
            raise ValueError('Reconstruction framing or grade context changed; rebuild this seam candidate')
    return entries


def entry_for_manifest(manifest, recipe):
    from .reconstruction import load_bundle
    path = Path(manifest).resolve()
    bundle = load_bundle(path)
    start, end = bundle['support']['start'], bundle['support']['end']
    return {'frame': bundle['frame'], 'start': start, 'end': end,
            'manifest': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'contextSha256': context_signature(recipe, start, end)}


class FrameReconstruction:
    """Stream native frames only where a scaled preview needs reconstruction."""
    def __init__(self, source, recipe, metadata):
        from .reconstruction import load_bundle
        self.source = source
        self.rows = validate_entries(recipe, metadata)
        self.bundles = []
        self.native = None
        self.native_row = None
        self.native_number = None
        for row in self.rows:
            path = Path(row['manifest'])
            if hashlib.sha256(path.read_bytes()).hexdigest() != row['sha256']:
                raise ValueError('Accepted reconstruction manifest changed')
            bundle = load_bundle(path)
            original = bundle['source']
            if original['sha256'] != recipe['source_sha256']:
                raise ValueError('Reconstruction belongs to a different source video')
            if any(original[key] != metadata[key] for key in ('width', 'height', 'fps_fraction', 'frame_count')):
                raise ValueError('Reconstruction source dimensions or timing differ')
            if (bundle['frame'] != row['frame'] or bundle['support'] != {'start': row['start'], 'end': row['end']}):
                raise ValueError('Reconstruction support differs from the accepted plan')
            self.bundles.append(bundle)

    def apply(self, number, rgb, resize=None):
        from .reconstruction import apply_frame
        for index, row in enumerate(self.rows):
            if row['start'] <= number < row['end']:
                native = rgb
                if resize:
                    if self.native_row != index or self.native_number != number:
                        self.close()
                        self.native = iter_frames(self.source, number, row['end'] - number)
                        self.native_row = index
                    native = next(self.native)
                    self.native_number = number + 1
                result = apply_frame(self.bundles[index], number, native)
                if result.shape != native.shape or result.dtype != np.uint8:
                    raise ValueError('Reconstruction returned invalid native RGB pixels')
                return cv2.resize(result, resize, interpolation=cv2.INTER_AREA) if resize else result
        self.close()
        return rgb

    def close(self):
        if self.native is not None:
            self.native.close()
        self.native = self.native_row = self.native_number = None
