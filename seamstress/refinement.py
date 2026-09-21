"""Refine one source seam while retaining a frozen plan everywhere else.

A refinement never chooses a new viewing crop or rebuilds other seams. The
baseline's render values are copied exactly outside an explicit source window.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np

from .conform import validate_conform_plan
from .media import probe
from .repair import fingerprint


class RefinementError(RuntimeError):
    """A requested local change cannot preserve its accepted context."""


def _frame(item):
    return item.get('frame') if isinstance(item, dict) else item


def _hash_bytes(data):
    return hashlib.sha256(data).hexdigest()


def load_baseline(path, metadata, source_sha256):
    path = Path(path).expanduser().resolve()
    data = path.read_bytes()
    plan = json.loads(data)
    validate_conform_plan(plan, metadata)
    if plan['source_sha256'] != source_sha256:
        raise ValueError('The refinement baseline belongs to a different source video')
    # Color calibration works on the original source in the final geometry.
    # Constant legacy segment grades cannot be silently refitted around.
    if any(segment['gain'] != [1, 1, 1] or segment['bias'] != [0, 0, 0]
           for segment in plan['segments']):
        raise RefinementError('Refinement requires a source-conform baseline with neutral segment grades')
    if 'frame_matrices' not in plan:
        if any(not np.array_equal(segment['matrix'], np.eye(3)) for segment in plan['segments']):
            raise RefinementError('Refinement requires explicit per-frame geometry or an identity baseline')
        plan['frame_matrices'] = np.repeat(np.eye(3)[None], metadata['frame_count'], axis=0).tolist()
    return plan, _hash_bytes(data)


def _curve_window(row):
    return row['frame']-1-row['support_before'], row['frame']+row['support_after']+1


def refinement_window(baseline, frame, metadata, boundaries, support_frames=None):
    """Select a complete target support without touching a neighbor's support."""
    count = metadata['frame_count']
    if type(frame) is not int or not 0 < frame < count:
        raise ValueError('Choose an incoming source frame inside the video')
    boundaries = sorted(set([frame, *boundaries, *[_frame(row) for row in baseline.get('seams', [])]]))
    if any(type(value) is not int or not 0 < value < count for value in boundaries):
        raise ValueError('Analysis boundaries must be source frame indices inside the video')
    support_rows = baseline.get('design_report', {}).get('actual_geometry_supports', [])
    target_windows = [(row['frame']-1-row['before_frames'], row['frame']+row['after_frames']+1)
                      for row in support_rows if row['frame'] == frame]
    for key in ('grade_curves', 'local_color_curves'):
        target_windows.extend(_curve_window(row) for row in baseline.get(key, []) if row['frame'] == frame)
    required = max([1, *[max(frame-1-start, end-1-frame) for start, end in target_windows]])
    if support_frames is None:
        support = max(required, round(7*metadata['fps'])) if not target_windows else required
        index = boundaries.index(frame)
        for neighbor in boundaries[max(0, index-1):index]+boundaries[index+1:index+2]:
            support = min(support, max(1, (abs(neighbor-frame)-2)//2))
    else:
        if type(support_frames) is not int or not 1 <= support_frames <= max(1, round(60*metadata['fps'])):
            raise ValueError('supportFrames must be a positive whole-frame support of at most 60 seconds')
        support = support_frames
    if support < required:
        raise RefinementError('The requested window would leave part of the previous seam correction outside it; use its full support')
    start, end = max(0, frame-1-support), min(count, frame+support+1)
    if min(frame-1-start, end-1-frame) < 1:
        raise RefinementError('Refinement needs at least one return frame on each side of the seam')
    for neighbor in boundaries:
        if neighbor != frame and start <= neighbor <= end:
            raise RefinementError('The refinement window reaches another marked boundary; shorten its support')
    occupied = [(row['frame'], row['frame']-1-row['before_frames'], row['frame']+row['after_frames']+1)
                for row in support_rows if row['frame'] != frame]
    for key in ('grade_curves', 'local_color_curves'):
        occupied.extend((row['frame'], *_curve_window(row)) for row in baseline.get(key, []) if row['frame'] != frame)
    if any(start < other_end and other_start < end for _, other_start, other_end in occupied):
        raise RefinementError('The refinement window overlaps a preserved neighbor correction; shorten its support')
    old_geometry = [(row['frame']-1-row['before_frames'], row['frame']+row['after_frames']+1)
                    for row in support_rows if row['frame'] == frame]
    for number in range(start, end):
        if (not np.array_equal(baseline['frame_matrices'][number], np.eye(3)) and
                not any(lo <= number < hi for lo, hi in old_geometry)):
            raise RefinementError('The refinement window contains geometry not attributed to this seam; preserve its existing measurements or use a complete analyzed baseline')
    return start, end, support, boundaries


def _replace(rows, candidates, frame):
    return sorted([copy.deepcopy(row) for row in rows if _frame(row) != frame] +
                  [copy.deepcopy(row) for row in candidates if _frame(row) == frame], key=_frame)


def _outside_signature(plan, start, end, frame):
    payload = {key: plan.get(key) for key in ('segments', 'view_matrix', 'edge_extension_pixels', 'geometry_mode')}
    payload['frame_matrices'] = plan['frame_matrices'][:start]+plan['frame_matrices'][end:]
    for key in ('grade_curves', 'local_color_curves'):
        payload[key] = [row for row in plan.get(key, []) if row['frame'] != frame]
    return _hash_bytes(json.dumps(payload, sort_keys=True, separators=(',', ':'), allow_nan=False).encode())


def merge_refinement_plan(baseline, candidate, metadata, *, frame, start, end, baseline_sha256):
    """Combine a target-only candidate with an immutable accepted render plan."""
    validate_conform_plan(baseline, metadata)
    validate_conform_plan(candidate, metadata)
    if (any(type(value) is not int for value in (frame, start, end)) or
            not 0 <= start <= frame-1 < frame < end <= metadata['frame_count']):
        raise ValueError('Refinement requires an explicit in-range window spanning the selected seam')
    if candidate['source_sha256'] != baseline['source_sha256']:
        raise RefinementError('Target measurements belong to a different baseline source')
    if [_frame(row) for row in candidate.get('seams', [])] != [frame]:
        raise RefinementError('A refinement candidate must contain only the selected seam')
    matrices = candidate.get('frame_matrices')
    if matrices is None:
        raise RefinementError('Target calibration did not produce explicit per-frame geometry')
    if any(not np.array_equal(matrix, np.eye(3)) for matrix in matrices[:start]+matrices[end:]):
        raise RefinementError('Target geometry exceeds the declared refinement window')
    for key in ('grade_curves', 'local_color_curves'):
        for row in candidate.get(key, []):
            lo, hi = _curve_window(row)
            if row['frame'] != frame or lo < start or hi > end:
                raise RefinementError('Target color exceeds the declared refinement window')
    result = copy.deepcopy(baseline)
    result['frame_matrices'][start:end] = copy.deepcopy(matrices[start:end])
    for key in ('grade_curves', 'local_color_curves', 'seams', 'unresolved_seams', 'review_decisions', 'correction_settings'):
        if key in baseline or key in candidate:
            result[key] = _replace(baseline.get(key, []), candidate.get(key, []), frame)
    view = np.asarray(result.get('view_matrix', np.eye(3)))
    width, height = metadata['width'], metadata['height']
    corners = np.array([[0, 0, 1], [width-1, 0, 1], [0, height-1, 1], [width-1, height-1, 1]]).T
    source = np.linalg.inv(view@np.asarray(result['frame_matrices'][start:end]))@corners
    clearance = np.minimum.reduce([source[:, 0].min(axis=1), source[:, 1].min(axis=1),
                                   width-1-source[:, 0].max(axis=1), height-1-source[:, 1].max(axis=1)])
    if float(clearance.min()) < -float(baseline.get('edge_extension_pixels', 0))-.000001:
        raise RefinementError('This seam correction exposes source edges with the preserved view. Reduce its adjustment; refinement cannot change the global viewing crop')
    old_design = result.setdefault('design_report', {})
    new_design = candidate.get('design_report', {})
    for key in ('global_geometry_corrections', 'excluded_geometry', 'actual_geometry_supports'):
        old_design[key] = _replace(old_design.get(key, []), new_design.get(key, []), frame)
    all_source = np.linalg.inv(view@np.asarray(result['frame_matrices']))@corners
    all_clearance = np.minimum.reduce([all_source[:, 0].min(axis=1), all_source[:, 1].min(axis=1),
        width-1-all_source[:, 0].max(axis=1), height-1-all_source[:, 1].max(axis=1)])
    old_design['minimum_native_source_clearance_pixels'] = float(all_clearance.min())
    old_design['minimum_clearance_frame'] = int(all_clearance.argmin())
    old_design['maximum_affine_singular_ratio'] = float(np.linalg.cond(np.asarray(result['frame_matrices'])[:, :2, :2]).max())
    outside_hash = _outside_signature(baseline, start, end, frame)
    if _outside_signature(result, start, end, frame) != outside_hash:
        raise RefinementError('Refinement changed render values outside its declared window')
    result['refinement'] = {'baseline_plan_sha256': baseline_sha256, 'frame': frame,
                            'start_frame': start, 'end_frame': end, 'view_pinned': True,
                            'outside_window_identical': True, 'outside_render_sha256': outside_hash,
                            'minimum_window_source_clearance_pixels': float(clearance.min())}
    result['status'] = 'SINGLE-SEAM REFINEMENT CANDIDATE: unchanged render values outside the declared source window; review required'
    validate_conform_plan(result, metadata)
    return result


def refine_video(source, frame, baseline_plan, output, *, correction=None, analysis_boundaries=(),
                 support_frames=None, options=None, progress=None, cancelled=None):
    """Measure only one seam, fitting color in the final pinned geometry."""
    from .calibration import calibrate_video
    source = Path(source).expanduser().resolve()
    metadata = probe(source)
    digest = fingerprint(source)
    baseline_path = Path(baseline_plan).expanduser().resolve()
    baseline, baseline_sha = load_baseline(baseline_path, metadata, digest)
    start, end, support, boundaries = refinement_window(baseline, frame, metadata, analysis_boundaries, support_frames)
    def transform(candidate, calibration, actual_metadata):
        if _hash_bytes(baseline_path.read_bytes()) != baseline_sha:
            raise RefinementError('The baseline plan changed during refinement')
        calibration['refinement_context'] = {'baseline_plan': str(baseline_path), 'baseline_plan_sha256': baseline_sha,
            'frame': frame, 'start_frame': start, 'end_frame': end, 'view_pinned': True}
        return merge_refinement_plan(baseline, candidate, actual_metadata, frame=frame, start=start, end=end, baseline_sha256=baseline_sha)
    choices = dict(options or {})
    result = calibrate_video(source, [frame], Path(output), options=choices,
                             seam_settings={frame: correction} if correction is not None else {},
                             analysis_boundaries=boundaries, plan_transform=transform,
                             geometry_support_frames=support,
                             progress=progress, cancelled=cancelled)
    if _hash_bytes(baseline_path.read_bytes()) != baseline_sha:
        raise RefinementError('The baseline plan changed during refinement')
    result['refinement'] = {'baseline_plan_sha256': baseline_sha, 'frame': frame,
                            'start_frame': start, 'end_frame': end, 'view_pinned': True,
                            'outside_window_identical': True}
    return result
