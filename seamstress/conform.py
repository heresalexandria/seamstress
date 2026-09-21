"""Preserve source drawings while correcting segment framing and grading.

There is exactly one source frame for every output frame. The default path
only adjusts geometry/color. Explicit reconstruction entries may supply
reviewed source-native layers before those same geometry/color operations.
"""
from __future__ import annotations

import json
import hashlib
from fractions import Fraction
import os
from pathlib import Path
import re
import sys
import tempfile
import time
from contextlib import ExitStack

import cv2
import numpy as np

from .media import VideoWriter, iter_frames, mux_audio, probe, _run, _tool
from .repair import fingerprint
from . import __version__
from .local_color import (validate_local_color_curves, prepare_local_color_curves,
                          apply_local_color_at, local_color_provenance)


def _similarity(value, name, allow_affine=False):
    try:
        matrix = np.asarray(value, dtype=np.float64)
    except (ValueError, TypeError) as exc:
        raise ValueError(f'{name} must be a finite 3 by 3 similarity matrix') from exc
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError(f'{name} must be a finite 3 by 3 similarity matrix')
    linear = matrix[:2, :2]
    scale2 = np.linalg.det(linear)
    if not np.allclose(matrix[2], [0, 0, 1], atol=1e-8) or scale2 <= 0:
        raise ValueError(f'{name} must preserve orientation and cannot contain perspective')
    if not allow_affine and not np.allclose(linear.T @ linear, np.eye(2) * scale2, atol=1e-7):
        raise ValueError(f'{name} permits only uniform scale, rotation, and translation')
    if allow_affine:
        singular = np.linalg.svd(linear, compute_uv=False)
        if singular.max()/singular.min() > 1.2:
            raise ValueError(f'{name} exceeds the 20% aspect distortion limit')
    if not .5 <= np.sqrt(scale2) <= 2:
        raise ValueError(f'{name} magnification must be between 0.5 and 2')
    return matrix


def validate_conform_plan(plan, metadata):
    if not isinstance(plan, dict) or plan.get('schema_version') != 3 or plan.get('method') != 'source_conform':
        raise ValueError('Expected schema_version 3 and method source_conform')
    source = plan.get('source')
    if not isinstance(source, dict):
        raise ValueError('Plan needs source metadata')
    for key in ('width', 'height', 'frame_count', 'fps_fraction'):
        if source.get(key) != metadata[key]:
            raise ValueError(f'Source {key} differs from conform plan')
    digest = plan.get('source_sha256')
    if not isinstance(digest, str) or not re.fullmatch('[0-9a-f]{64}', digest):
        raise ValueError('Plan needs source SHA-256')
    geometry = plan.get('geometry_mode', 'similarity')
    if geometry not in ('similarity', 'affine'):
        raise ValueError('geometry_mode must be similarity or affine')
    segments = plan.get('segments')
    if not isinstance(segments, list) or not segments:
        raise ValueError('Plan needs contiguous segments covering the video')
    expected = 0
    for i, segment in enumerate(segments):
        if not isinstance(segment, dict):
            raise ValueError('Each segment must be an object')
        start, end = segment.get('start'), segment.get('end')
        if type(start) is not int or type(end) is not int or start != expected or not start < end <= metadata['frame_count']:
            raise ValueError('Segments must cover each source frame once, in order; end is exclusive')
        _similarity(segment.get('matrix'), f'segments[{i}].matrix', geometry == 'affine')
        for name in ('gain', 'bias'):
            values = np.asarray(segment.get(name), dtype=float)
            if values.shape != (3,) or not np.isfinite(values).all():
                raise ValueError(f'segments[{i}].{name} must contain three finite numbers')
            if name == 'gain' and (np.any(values <= 0) or np.any(values > 4)):
                raise ValueError('Color gains must be positive and no greater than four')
            if name == 'bias' and np.any(np.abs(values) > 128):
                raise ValueError('Color biases must stay within 128 levels')
        expected = end
    if expected != metadata['frame_count']:
        raise ValueError('Segments do not cover the complete source')
    matrices = plan.get('frame_matrices')
    if matrices is not None:
        if not isinstance(matrices, list) or len(matrices) != expected:
            raise ValueError('frame_matrices must contain exactly one matrix per source frame')
        for i, matrix in enumerate(matrices):
            _similarity(matrix, f'frame_matrices[{i}]', geometry == 'affine')
    _similarity(plan.get('view_matrix', np.eye(3)), 'view_matrix')
    extension = plan.get('edge_extension_pixels', 0)
    if type(extension) not in (int, float) or not np.isfinite(extension) or not 0 <= extension <= 4:
        raise ValueError('edge_extension_pixels must be between zero and four')
    curves = plan.get('grade_curves', [])
    if not isinstance(curves, list):
        raise ValueError('grade_curves must be a list')
    previous_end = -1
    for curve in curves:
        if not isinstance(curve, dict):
            raise ValueError('Each grade curve must be an object')
        cut, before, after = (curve.get(key) for key in ('frame', 'support_before', 'support_after'))
        if any(type(v) is not int for v in (cut, before, after)) or not 0 < cut < expected or min(before, after) < 1:
            raise ValueError('Grade curves need a valid cut and positive integer support lengths')
        if cut - 1 - before <= previous_end:
            raise ValueError('Grade curve supports must be ordered and cannot overlap')
        previous_end = cut + after
        for key in ('left_lut', 'right_lut'):
            table = np.asarray(curve.get(key), dtype=float)
            if (table.shape != (256, 3) or not np.isfinite(table).all() or
                    np.any(table < 0) or np.any(table > 255) or np.any(np.diff(table, axis=0) < -1e-8) or
                    not np.allclose(table[0], 0) or not np.allclose(table[-1], 255)):
                raise ValueError(f'{key} must be a monotone 256 by 3 LUT preserving black and white')
    if curves and any(not np.allclose(s['gain'], 1) or not np.allclose(s['bias'], 0) for s in segments):
        raise ValueError('Use either segment gains/biases or protected grade curves, not both')
    validate_local_color_curves(plan.get('local_color_curves', []), expected)
    if 'reconstructions' in plan:
        from .reconstruction_render import validate_entries
        validate_entries(plan, metadata)


def tone_lut_at(frame, curves):
    """Return a bounded tone map; temporal weights never mix source images."""
    identity = np.repeat(np.arange(256, dtype=np.float32)[:, None], 3, axis=1)
    for curve in curves:
        cut = curve['frame']
        distance = (cut-1-frame)/curve['support_before'] if frame < cut else (frame-cut)/curve['support_after']
        if 0 <= distance < 1:
            t = 1-distance
            weight = t*t*t*(10+t*(-15+6*t))
            table = np.asarray(curve['left_lut' if frame < cut else 'right_lut'], dtype=np.float32)
            return identity + weight*(table-identity)
    return None


def conform_frame(frame, matrix, gain, bias, edge_extension_pixels=0):
    """Sample one original drawing once, then apply its constant segment grade."""
    height, width = frame.shape[:2]
    corners = np.array([[0, 0, 1], [width-1, 0, 1], [0, height-1, 1], [width-1, height-1, 1]])
    source_corners = corners @ np.linalg.inv(matrix).T
    outside = max(0., float(-source_corners[:, :2].min()),
                  float(source_corners[:, 0].max()-(width-1)),
                  float(source_corners[:, 1].max()-(height-1)))
    if outside > edge_extension_pixels + .01:
        raise ValueError(f'Camera correction exposes {outside:.2f} pixels beyond source; adjust view_matrix')
    image = frame if np.allclose(matrix, np.eye(3), atol=1e-10) else cv2.warpAffine(
        frame, matrix[:2], (width, height), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    graded = image.astype(np.float32) * gain + bias
    clipped = int(np.count_nonzero((graded < 0) | (graded > 255)))
    return np.clip(graded, 0, 255).round().astype(np.uint8), outside, clipped


def render_conform(input, plan, output, crf=14, start_frame=0, end_frame=None, *,
                   progress=None, cancelled=None, preview_width=None):
    input, plan, output = map(Path, (input, plan, output))
    if output.resolve() == input.resolve() or output.exists() or output.with_suffix('.repair.json').exists():
        raise ValueError('Choose a new output path distinct from source and existing sidecars')
    metadata = probe(input)
    end_frame = metadata['frame_count'] if end_frame is None else end_frame
    if (type(start_frame) is not int or type(end_frame) is not int or
            not 0 <= start_frame < end_frame <= metadata['frame_count']):
        raise ValueError('Preview range must satisfy 0 <= start_frame < end_frame <= source frame count')
    preview = start_frame != 0 or end_frame != metadata['frame_count'] or preview_width is not None
    width, height = metadata['width'], metadata['height']
    resize = None
    scale = np.eye(3)
    if preview_width is not None:
        if type(preview_width) is not int or preview_width < 32:
            raise ValueError('Preview width must be an integer of at least 32 pixels')
        width = min(width, preview_width) // 2 * 2
        height = max(2, round(metadata['height'] * width / metadata['width'] / 2) * 2)
        resize = (width, height)
        sx, sy = width / metadata['width'], height / metadata['height']
        scale = np.array([[sx, 0, (sx-1)/2], [0, sy, (sy-1)/2], [0, 0, 1.]])
    inverse_scale = np.linalg.inv(scale)
    plan_bytes = plan.read_bytes()
    recipe = json.loads(plan_bytes)
    validate_conform_plan(recipe, metadata)
    if fingerprint(input) != recipe['source_sha256']:
        raise ValueError('Source fingerprint differs from conform plan')
    ffmpeg_version = _run([_tool('ffmpeg'), '-version']).stdout.decode('utf-8', errors='replace').splitlines()[0]
    view = np.array(recipe.get('view_matrix', np.eye(3)), dtype=float)
    segments = recipe['segments']
    matrices = recipe.get('frame_matrices')
    local_curves = prepare_local_color_curves(recipe.get('local_color_curves', []), metadata['frame_count'])
    local_provenance = local_color_provenance(recipe['local_color_curves']) if local_curves else None
    output.parent.mkdir(parents=True, exist_ok=True)
    index = 0
    clipped = 0
    maximum_extension = 0.
    magnifications = []
    render_started = time.monotonic()
    total_frames = end_frame-start_frame
    with tempfile.TemporaryDirectory(prefix='seamstress-conform-', dir=output.parent) as temp, ExitStack() as cleanup:
        temp = Path(temp)
        reconstruction = None
        if recipe.get('reconstructions'):
            from .reconstruction_render import FrameReconstruction
            reconstruction = FrameReconstruction(input, recipe, metadata)
            cleanup.callback(reconstruction.close)
        with VideoWriter(temp/'video.mp4', width, height, metadata['fps_fraction'], crf=crf, preset='fast' if preview else 'slow',
                         color_tags={k:metadata.get(k) for k in ('color_space','color_transfer','color_primaries','color_range')
                                     if not (k=='color_space' and metadata.get(k)=='gbr')}) as writer:
            frames = iter_frames(input, start_frame, end_frame-start_frame, size=resize) if resize else iter_frames(input, start_frame, end_frame-start_frame)
            for number, frame in enumerate(frames, start_frame):
                if cancelled and cancelled():
                    raise InterruptedError('Rendering cancelled')
                if number >= metadata['frame_count']:
                    raise RuntimeError('Source decoded more frames than its metadata')
                if reconstruction is not None:
                    frame = reconstruction.apply(number, frame, resize)
                while number >= segments[index]['end']:
                    index += 1
                segment = segments[index]
                matrix = scale @ view @ np.array(matrices[number] if matrices is not None else segment['matrix']) @ inverse_scale
                result, extension, clipping = conform_frame(frame, matrix, np.array(segment['gain']), np.array(segment['bias']), recipe.get('edge_extension_pixels', 0) * max(scale[0, 0], scale[1, 1]) + (1 if resize else 0))
                table = tone_lut_at(number, recipe.get('grade_curves', []))
                if table is not None:
                    result = cv2.LUT(result, table[:, None, :]).round().astype(np.uint8)
                if local_curves:
                    result = apply_local_color_at(result, number, local_curves)
                writer.write(result)
                clipped += clipping
                maximum_extension = max(maximum_extension, extension)
                magnifications.append(float(np.sqrt(np.linalg.det(matrix[:2, :2]))))
                if number == segment['start']:
                    print(f'Conforming original segment {index+1}/{len(segments)} at frame {number}', file=sys.stderr, flush=True)
                completed = number-start_frame+1
                if progress and (completed % 12 == 0 or completed == total_frames):
                    progress({'stage': 'preview' if preview else 'export', 'fraction': completed/total_frames,
                              'message': f'Rendering {completed:,} / {total_frames:,} original frames'})
                if completed % 240 == 0 or completed == total_frames:
                    elapsed = time.monotonic()-render_started
                    print(f'Conforming frames: {completed}/{total_frames} completed in {elapsed:.1f}s', file=sys.stderr, flush=True)
        if writer.frames_written != end_frame-start_frame:
            raise RuntimeError('Source decoded fewer frames than its metadata')
        if preview and metadata['has_audio']:
            _run([_tool('ffmpeg'), '-v', 'error', '-nostdin', '-i', str(temp/'video.mp4'),
                  '-i', str(input), '-map', '0:v:0', '-map', '1:a:0', '-c:v', 'copy',
                  '-af', f'atrim=start={start_frame/metadata["fps"]:.12f}:end={end_frame/metadata["fps"]:.12f},asetpts=PTS-{start_frame/metadata["fps"]:.12f}/TB',
                  # atrim bounds the audio. -shortest can discard the encoded
                  # video tail when AAC packets end before the final picture.
                  '-c:a', 'aac', '-b:a', '192k', '-movflags', '+faststart', str(temp/'complete.mp4')])
        else:
            mux_audio(temp/'video.mp4', input, temp/'complete.mp4')
        if cancelled and cancelled():
            raise InterruptedError('Rendering cancelled')
        rendered = probe(temp/'complete.mp4')
        if (rendered['frame_count_estimated'] or rendered['frame_count'] != total_frames or
                Fraction(rendered['fps_fraction']) != Fraction(metadata['fps_fraction']) or
                (rendered['width'], rendered['height']) != (width, height)):
            raise RuntimeError('Muxed output does not preserve the requested frame count, rate, and dimensions')
        if cancelled and cancelled():
            raise InterruptedError('Rendering cancelled')
        os.link(temp/'complete.mp4', output)
    report = {
        'output': str(output.resolve()), 'source_sha256': recipe['source_sha256'],
        'plan_sha256': hashlib.sha256(plan_bytes).hexdigest(),
        'renderer': 'seamstress.conform', 'renderer_version': __version__,
        'ffmpeg_version': ffmpeg_version,
        'method': 'One original drawing per frame; segment grading and global framing correction',
        'geometry_mode': recipe.get('geometry_mode', 'similarity'),
        'protected_grade_curve_count': len(recipe.get('grade_curves', [])),
        'local_color_curve_count': len(local_curves),
        'view_matrix': view.tolist(),
        'constant_view_magnification': float(np.sqrt(np.linalg.det(view[:2, :2]))),
        'plan_status': recipe.get('status', 'Unreviewed'),
        'unresolved_seams': recipe.get('unresolved_seams', []),
        'synthesized_frames': 0, 'frame_mapping': 'contiguous_original_frames' if preview else 'identity', 'frame_count': rendered['frame_count'],
        'output_fps_fraction': rendered['fps_fraction'], 'muxed_video_verified': True,
        'source_start_frame': start_frame, 'source_end_frame_exclusive': end_frame,
        'preview': preview, 'output_width': width, 'output_height': height, 'audio_mode': 'trimmed and encoded as AAC' if preview and metadata['has_audio'] else 'copied',
        'minimum_magnification': min(magnifications), 'maximum_magnification': max(magnifications),
        'maximum_source_edge_extension_pixels': maximum_extension,
        'clipped_rgb_channel_fraction': clipped/(len(magnifications)*width*height*3),
        'requires_visual_review': True,
    }
    if local_curves:
        report['local_color_model_provenance'] = local_provenance
    if recipe.get('reconstructions'):
        report['reconstructions'] = recipe['reconstructions']
        report['method'] = 'Original frame cadence; explicit native layer reconstruction followed by preserved framing and grading'
    output.with_suffix('.repair.json').write_text(json.dumps(report, indent=2))
    return report
