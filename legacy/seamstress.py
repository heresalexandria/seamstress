#!/usr/bin/env python3
"""
Make the cut between two timeline-adjacent clips visually seamless.

The CLI accepts the previous clip and the next clip as positional arguments.
By default it alters the next clip so its first frame matches the previous
clip's final frame. Use --alter previous when you need to preserve the next
clip and adjust the previous clip's ending instead.

Pipeline (see SPEC.md for the full technical spec):
  1. Ingest & conform      - ffprobe both clips, reconcile fps/resolution.
  2. Temporal alignment    - detect head/tail frame overlap, compute trim.
  3. Global color match    - pooled, motion-compensated affine+curves fit.
  4. Spatial alignment     - conditional ECC affine warp.
  5. Seam finishing        - motion-compensated residual propagation.
  6. Verification & output - encode, joined preview, seam metric, reports.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, Optional

cv2 = None
np = None


def load_python_dependencies() -> None:
    global cv2, np
    try:
        import cv2 as cv2_module
        import numpy as np_module
    except ImportError as exc:
        raise RuntimeError(
            "missing Python dependency; install requirements with "
            "`python -m pip install -r requirements.txt`"
        ) from exc
    cv2 = cv2_module
    np = np_module


# --------------------------------------------------------------------------
# Constants (spec defaults)
# --------------------------------------------------------------------------

TAIL_N = 25
HEAD_M = 50
FEATURE_WIDTH = 320
HIGHPASS_SIGMA = 2.0
CONFIDENCE_THRESHOLD = 4.0
PEAK_NCC_THRESHOLD = 0.25
FLOW_MAX_WIDTH = 960
FB_CONSISTENCY_PX = 1.5
SATURATION_LOW = 4
SATURATION_HIGH = 251
MAX_COLOR_SAMPLES = 500_000
CURVE_MIN_BIN_COUNT = 200
CURVE_EXTREME_CLAMP = 40.0
MEAN_SHIFT_WARN = 25.0
SEAM_CLEAN_MAE = 1.5
LOW_FREQ_SIGMA = 8.0
SEAM_PEAK_WEIGHT = 0.5
LOW_FREQ_CONF_EPS = 0.25
MIN_KEPT_RATIO = 0.02
ECC_WIDTH = 640
FPS_WARN_RATIO = 0.01
ASPECT_REFUSE_RATIO = 0.005


# --------------------------------------------------------------------------
# Data model
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class VideoInfo:
    width: int
    height: int
    fps_num: int
    fps_den: int
    fps: float
    frames: int
    duration: float
    pix_fmt: str
    color_space: str
    color_transfer: str
    color_primaries: str
    has_audio: bool
    audio_duration: float

    @property
    def size_arg(self) -> str:
        return f"{self.width}x{self.height}"

    @property
    def rate_arg(self) -> str:
        return f"{self.fps_num}/{self.fps_den}"


@dataclass
class AlignmentResult:
    trim: int
    confidence: float
    peak_ncc: float
    accepted: bool
    best_offset: int
    tail_n: int
    head_m: int
    overlap_k: int
    pairs: list  # list of (prev_global_idx, next_idx, ncc)
    reason: str = ""


@dataclass
class SeamPlan:
    previous: Path
    next_clip: Path
    source: Path
    reference: Path
    source_boundary: str
    reference_boundary: str
    anchor_side: str
    altered_label: str
    reference_label: str


@dataclass
class ColorModel:
    affine: "object"  # np.ndarray 4x3
    luts: "object" = None  # np.ndarray 3x256 or None
    mean_shift: "object" = None
    warnings: list = field(default_factory=list)


@dataclass
class SeamData:
    active: bool
    mae0: float
    low_freq: "object" = None
    detail0: "object" = None
    confidence_ratio: float = 0.0


# --------------------------------------------------------------------------
# ffprobe
# --------------------------------------------------------------------------

def run_json(cmd: list[str]) -> dict:
    return json.loads(subprocess.check_output(cmd, text=True))


def ffprobe_video(path: Path) -> VideoInfo:
    data = run_json(
        [
            "ffprobe", "-v", "error", "-show_entries",
            "stream=index,codec_type,width,height,r_frame_rate,nb_frames,"
            "duration,pix_fmt,color_space,color_transfer,color_primaries",
            "-of", "json", str(path),
        ]
    )
    streams = data.get("streams", [])
    vstreams = [s for s in streams if s.get("codec_type") == "video"]
    astreams = [s for s in streams if s.get("codec_type") == "audio"]
    if not vstreams:
        raise RuntimeError(f"no video stream found in {path}")
    vstream = vstreams[0]

    fps_num, fps_den = (int(part) for part in vstream["r_frame_rate"].split("/"))
    fps = fps_num / fps_den if fps_den else 0.0
    duration = float(vstream.get("duration") or 0.0)
    frames = int(vstream.get("nb_frames") or 0)
    if not frames and fps and duration:
        frames = int(round(duration * fps))

    def tag(key: str) -> str:
        value = vstream.get(key)
        if not value or value in ("unknown", "N/A"):
            return "bt709"
        return value

    has_audio = len(astreams) > 0
    audio_duration = float(astreams[0].get("duration") or 0.0) if has_audio else 0.0

    return VideoInfo(
        width=int(vstream["width"]),
        height=int(vstream["height"]),
        fps_num=fps_num,
        fps_den=fps_den,
        fps=fps,
        frames=frames,
        duration=duration,
        pix_fmt=vstream.get("pix_fmt") or "yuv420p",
        color_space=tag("color_space"),
        color_transfer=tag("color_transfer"),
        color_primaries=tag("color_primaries"),
        has_audio=has_audio,
        audio_duration=audio_duration,
    )


def probe_frame_count(path: Path, info: VideoInfo) -> int:
    """Best-effort exact frame count; nb_frames can be absent/unreliable."""
    if info.frames and info.frames > 0:
        return info.frames
    tw, th = scaled_size(info.width, info.height, 64)
    count = sum(1 for _ in iter_frames(path, tw, th, vf=f"scale={tw}:{th}"))
    return count


# --------------------------------------------------------------------------
# Frame decoding (ffmpeg rawvideo pipes)
# --------------------------------------------------------------------------

def scaled_size(width: int, height: int, target_width: int) -> tuple[int, int]:
    target_width = max(2, int(round(target_width / 2)) * 2)
    scale = target_width / width
    target_height = max(2, int(round(height * scale / 2)) * 2)
    return target_width, target_height


def iter_frames(path: Path, width: int, height: int, vf: Optional[str] = None) -> Iterator["np.ndarray"]:
    """Decode a clip to RGB24 raw frames, streaming, from the start.

    Frame skipping must be done by discarding decoded frames from this
    stream (see skip_frames()) rather than via ffmpeg -ss, which is
    timestamp-based and imprecise for our exact-frame-count needs.
    """
    cmd = ["ffmpeg", "-v", "error", "-i", str(path)]
    if vf:
        cmd += ["-vf", vf]
    cmd += ["-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=10 ** 8)
    frame_bytes = width * height * 3
    assert proc.stdout is not None
    try:
        while True:
            buf = proc.stdout.read(frame_bytes)
            if len(buf) < frame_bytes:
                break
            yield np.frombuffer(buf, np.uint8).reshape(height, width, 3).copy()
    finally:
        try:
            proc.stdout.close()
        except Exception:
            pass
        if proc.poll() is None:
            proc.terminate()
        try:
            proc.wait(timeout=5)
        except Exception:
            proc.kill()
            proc.wait()


def skip_frames(gen: Iterator["np.ndarray"], n: int) -> Iterator["np.ndarray"]:
    for _ in range(n):
        next(gen, None)
    return gen


def fetch_frames(path: Path, width: int, height: int, count: int, skip: int = 0,
                  vf: Optional[str] = None) -> list:
    gen = iter_frames(path, width, height, vf=vf)
    skip_frames(gen, skip)
    return list(itertools.islice(gen, count))


# --------------------------------------------------------------------------
# Feature extraction
# --------------------------------------------------------------------------

def highpass_u8(rgb: "np.ndarray", sigma: float = HIGHPASS_SIGMA) -> "np.ndarray":
    rgb_u8 = rgb if rgb.dtype == np.uint8 else np.clip(rgb, 0, 255).astype(np.uint8)
    gray = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2GRAY).astype(np.float32)
    blur = cv2.GaussianBlur(gray, (0, 0), sigma)
    hp = gray - blur
    return np.clip(hp + 128.0, 0, 255).astype(np.uint8)


def highpass_f01(rgb: "np.ndarray", sigma: float = HIGHPASS_SIGMA) -> "np.ndarray":
    rgb_u8 = rgb if rgb.dtype == np.uint8 else np.clip(rgb, 0, 255).astype(np.uint8)
    gray = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    blur = cv2.GaussianBlur(gray, (0, 0), 3.0)
    return np.clip(gray - blur + 0.5, 0.0, 1.0).astype(np.float32)


def normalize_feat(hp_u8: "np.ndarray") -> "np.ndarray":
    f = hp_u8.astype(np.float32)
    m = f.mean()
    s = f.std()
    if s < 1e-6:
        s = 1e-6
    return (f - m) / s


def make_mask(height: int, width: int, margin: int) -> "np.ndarray":
    mask = np.zeros((height, width), np.uint8)
    margin = max(0, min(margin, min(height, width) // 3))
    mask[margin: height - margin, margin: width - margin] = 255
    return mask


# --------------------------------------------------------------------------
# Stage 2 - temporal alignment
# --------------------------------------------------------------------------

def temporal_align(previous: Path, previous_info: VideoInfo, next_clip: Path,
                    next_info: VideoInfo, total_prev: int) -> AlignmentResult:
    tw1, th1 = scaled_size(previous_info.width, previous_info.height, FEATURE_WIDTH)
    tw2, th2 = scaled_size(next_info.width, next_info.height, FEATURE_WIDTH)

    prev_gen = iter_frames(previous, tw1, th1, vf=f"scale={tw1}:{th1}:flags=lanczos")
    prev_small: list = []
    for frame in prev_gen:
        prev_small.append(frame)
        if len(prev_small) > TAIL_N:
            prev_small.pop(0)

    next_gen = iter_frames(next_clip, tw2, th2, vf=f"scale={tw2}:{th2}:flags=lanczos")
    next_small = list(itertools.islice(next_gen, HEAD_M))

    if len(prev_small) < 2 or len(next_small) < 2:
        return AlignmentResult(0, 0.0, 0.0, False, 0, len(prev_small), len(next_small), 0, [],
                                reason="insufficient frames decoded for temporal alignment")

    feat1 = np.stack([normalize_feat(highpass_u8(f)).reshape(-1) for f in prev_small])
    feat2 = np.stack([normalize_feat(highpass_u8(f)).reshape(-1) for f in next_small])
    N = feat1.shape[0]
    M = feat2.shape[0]
    C = (feat1 @ feat2.T) / feat1.shape[1]

    diag_means: dict[int, float] = {}
    diag_max: dict[int, float] = {}
    for D in range(-(N - 1), M):
        idx_i = np.arange(max(0, -D), min(N, M - D))
        if len(idx_i) < 2:
            continue
        idx_j = idx_i + D
        vals = C[idx_i, idx_j]
        diag_means[D] = float(vals.mean())
        diag_max[D] = float(vals.max())

    if not diag_means:
        return AlignmentResult(0, 0.0, 0.0, False, 0, N, M, 0, [],
                                reason="no valid diagonals to compare")

    best_D = max(diag_means, key=diag_means.get)
    best_mean = diag_means[best_D]
    others = [v for d, v in diag_means.items() if d != best_D]
    median_others = float(np.median(others)) if others else 0.0
    std_others = float(np.std(others)) if others else 1e-6
    std_others = max(std_others, 1e-6)
    confidence = (best_mean - median_others) / std_others
    peak_ncc = diag_max[best_D]

    j_star = (N - 1) + best_D
    in_range = 0 <= j_star < M
    accepted = confidence >= CONFIDENCE_THRESHOLD and peak_ncc >= PEAK_NCC_THRESHOLD and in_range

    trim = 0
    pairs: list = []
    if accepted:
        trim = j_star + 1
        k = trim
        for kk in range(k):
            i_local = kk - best_D
            prev_global = total_prev - N + i_local
            pairs.append((prev_global, kk, float(C[i_local, kk])))

    reason = "" if accepted else (
        "overlap match implies out-of-window offset" if not in_range else
        f"confidence {confidence:.2f} or peak NCC {peak_ncc:.3f} below threshold"
    )

    return AlignmentResult(
        trim=trim, confidence=confidence, peak_ncc=peak_ncc, accepted=accepted,
        best_offset=best_D, tail_n=N, head_m=M, overlap_k=(trim if accepted else 0),
        pairs=pairs, reason=reason,
    )


# --------------------------------------------------------------------------
# Optical flow helpers
# --------------------------------------------------------------------------

def compute_flow(gray0_u8: "np.ndarray", gray1_u8: "np.ndarray",
                  max_width: int = FLOW_MAX_WIDTH) -> "np.ndarray":
    h, w = gray0_u8.shape
    if w > max_width:
        sw, sh = scaled_size(w, h, max_width)
        g0 = cv2.resize(gray0_u8, (sw, sh), interpolation=cv2.INTER_AREA)
        g1 = cv2.resize(gray1_u8, (sw, sh), interpolation=cv2.INTER_AREA)
    else:
        g0, g1 = gray0_u8, gray1_u8
        sw, sh = w, h
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    flow = dis.calc(g0, g1, None)
    if (sw, sh) != (w, h):
        flow_up = cv2.resize(flow, (w, h), interpolation=cv2.INTER_LINEAR)
        flow_up[..., 0] *= (w / sw)
        flow_up[..., 1] *= (h / sh)
        return flow_up
    return flow


def warp_by_flow(img: "np.ndarray", flow: "np.ndarray") -> "np.ndarray":
    h, w = flow.shape[:2]
    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
    map_x = (grid_x + flow[..., 0]).astype(np.float32)
    map_y = (grid_y + flow[..., 1]).astype(np.float32)
    return cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                      borderMode=cv2.BORDER_REPLICATE)


def fb_consistency_mask(fwd: "np.ndarray", bwd: "np.ndarray",
                         threshold: float = FB_CONSISTENCY_PX) -> "np.ndarray":
    h, w = fwd.shape[:2]
    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
    sample_x = (grid_x + fwd[..., 0]).astype(np.float32)
    sample_y = (grid_y + fwd[..., 1]).astype(np.float32)
    bwd_at = cv2.remap(bwd, sample_x, sample_y, interpolation=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_REPLICATE)
    err = np.sqrt((fwd[..., 0] + bwd_at[..., 0]) ** 2 + (fwd[..., 1] + bwd_at[..., 1]) ** 2)
    return err <= threshold


def border_mask(height: int, width: int, margin: int) -> "np.ndarray":
    m = max(0, min(margin, min(height, width) // 3))
    mask = np.zeros((height, width), bool)
    mask[m: height - m, m: width - m] = True
    return mask


def unsaturated_mask(a: "np.ndarray", b: "np.ndarray") -> "np.ndarray":
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    ok_a = (a >= SATURATION_LOW).all(axis=2) & (a <= SATURATION_HIGH).all(axis=2)
    ok_b = (b >= SATURATION_LOW).all(axis=2) & (b <= SATURATION_HIGH).all(axis=2)
    return ok_a & ok_b


# --------------------------------------------------------------------------
# Stage 3 - global color match
# --------------------------------------------------------------------------

def align_pair_for_color(ref_frame: "np.ndarray", src_frame: "np.ndarray", margin: int):
    """Motion-compensate src onto ref's grid; return (warped_src, ref, mask)."""
    g_ref = highpass_u8(ref_frame)
    g_src = highpass_u8(src_frame)
    fwd = compute_flow(g_ref, g_src)   # ref -> src displacement
    bwd = compute_flow(g_src, g_ref)   # src -> ref displacement
    warped_src = warp_by_flow(src_frame.astype(np.float32), fwd)
    fb_mask = fb_consistency_mask(fwd, bwd)
    b_mask = border_mask(ref_frame.shape[0], ref_frame.shape[1], margin)
    sat_mask = unsaturated_mask(ref_frame, warped_src)
    mask = fb_mask & b_mask & sat_mask
    return warped_src, ref_frame.astype(np.float32), mask, fb_mask


def build_correspondence_pool(pairs: list[tuple], margin: int, max_samples: int = MAX_COLOR_SAMPLES):
    """pairs: list of (ref_frame, src_frame) full-res RGB uint8 arrays."""
    src_chunks = []
    ref_chunks = []
    total_pixels = 0
    kept_pixels = 0
    for ref_frame, src_frame in pairs:
        warped_src, ref_f, mask, _ = align_pair_for_color(ref_frame, src_frame, margin)
        total_pixels += mask.size
        kept_pixels += int(mask.sum())
        if mask.sum() < 100:
            continue
        src_chunks.append(warped_src[mask])
        ref_chunks.append(ref_f[mask])
    if not src_chunks:
        raise RuntimeError("no confident pixel correspondences found for color solve")
    src = np.concatenate(src_chunks, axis=0)
    ref = np.concatenate(ref_chunks, axis=0)
    if len(src) > max_samples:
        idx = np.random.choice(len(src), max_samples, replace=False)
        src, ref = src[idx], ref[idx]
    ratio = kept_pixels / max(1, total_pixels)
    return src, ref, ratio


def solve_affine_trimmed(src: "np.ndarray", ref: "np.ndarray", trim_percentile: float) -> "np.ndarray":
    src = src.astype(np.float64)
    ref = ref.astype(np.float64)
    n = src.shape[0]
    src_b = np.concatenate([src, np.ones((n, 1))], axis=1)
    selected = np.ones(n, dtype=bool)
    matrix = np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0], [0, 0, 0]])
    for _ in range(5):
        matrix = np.linalg.lstsq(src_b[selected], ref[selected], rcond=None)[0]
        predicted = src_b @ matrix
        residual = np.sqrt(np.mean((predicted - ref) ** 2, axis=1))
        threshold = np.percentile(residual, trim_percentile)
        new_selected = residual <= threshold
        if new_selected.sum() < 1000:
            break
        selected = new_selected
    return matrix


def isotonic_regression(y: "np.ndarray", weights: "np.ndarray") -> "np.ndarray":
    """Pool-adjacent-violators, weighted, non-decreasing regression."""
    n = len(y)
    blocks = []  # [value, weight, count]
    for i in range(n):
        v, w = float(y[i]), float(weights[i])
        blocks.append([v, w, 1])
        while len(blocks) >= 2 and blocks[-2][0] > blocks[-1][0]:
            v2, w2, c2 = blocks.pop()
            v1, w1, c1 = blocks.pop()
            nv = (v1 * w1 + v2 * w2) / (w1 + w2)
            blocks.append([nv, w1 + w2, c1 + c2])
    out = np.empty(n)
    idx = 0
    for v, _w, c in blocks:
        out[idx: idx + c] = v
        idx += c
    return out


def gaussian_smooth_1d(x: "np.ndarray", sigma: float) -> "np.ndarray":
    radius = max(1, int(round(sigma * 3)))
    k = np.arange(-radius, radius + 1)
    kernel = np.exp(-(k ** 2) / (2.0 * sigma ** 2))
    kernel /= kernel.sum()
    padded = np.pad(x, radius, mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def fit_channel_curves(src: "np.ndarray", ref: "np.ndarray", affine_matrix: "np.ndarray",
                        min_bin_count: int = CURVE_MIN_BIN_COUNT) -> "np.ndarray":
    n = src.shape[0]
    src_b = np.concatenate([src.astype(np.float64), np.ones((n, 1))], axis=1)
    pred = src_b @ affine_matrix  # affine-corrected source, in ref space

    identity = np.arange(256, dtype=np.float64)
    luts = np.zeros((3, 256), dtype=np.float64)
    for c in range(3):
        x = np.clip(pred[:, c], 0, 255)
        y = ref[:, c]
        bins = np.clip(np.round(x).astype(np.int64), 0, 255)

        order = np.argsort(bins)
        bins_sorted = bins[order]
        y_sorted = y[order]
        unique_bins, start_idx, counts = np.unique(bins_sorted, return_index=True, return_counts=True)

        medians = np.full(256, np.nan)
        bin_counts = np.zeros(256, dtype=np.int64)
        for ub, s, cnt in zip(unique_bins, start_idx, counts):
            medians[ub] = np.median(y_sorted[s: s + cnt])
            bin_counts[ub] = cnt

        target = identity.copy()
        have = ~np.isnan(medians)
        target[have] = medians[have]
        weight = np.clip(bin_counts / float(min_bin_count), 0.0, 1.0)
        blended = weight * target + (1.0 - weight) * identity

        pav_weight = bin_counts.astype(np.float64) + 1.0
        mono = isotonic_regression(blended, pav_weight)
        smooth = gaussian_smooth_1d(mono, sigma=4.0)

        # Guardrail: clamp wild extrapolation, then re-monotonize.
        deviation = smooth - identity
        if np.max(np.abs(deviation)) > CURVE_EXTREME_CLAMP:
            smooth = identity + np.clip(deviation, -CURVE_EXTREME_CLAMP, CURVE_EXTREME_CLAMP)
            smooth = isotonic_regression(smooth, np.ones(256))
            smooth = gaussian_smooth_1d(smooth, sigma=4.0)

        luts[c] = smooth
    return luts


def apply_color(frame: "np.ndarray", affine_matrix: "np.ndarray",
                 luts: Optional["np.ndarray"] = None) -> "np.ndarray":
    flat = frame.reshape(-1, 3).astype(np.float64)
    flat_b = np.concatenate([flat, np.ones((flat.shape[0], 1))], axis=1)
    corrected = flat_b @ affine_matrix
    if luts is not None:
        xp = np.arange(256, dtype=np.float64)
        for c in range(3):
            corrected[:, c] = np.interp(np.clip(corrected[:, c], 0, 255), xp, luts[c])
    return corrected.reshape(frame.shape[0], frame.shape[1], 3).astype(np.float32)


def add_triangular_dither(frame: "np.ndarray") -> "np.ndarray":
    u1 = np.random.random(frame.shape).astype(np.float32)
    u2 = np.random.random(frame.shape).astype(np.float32)
    tri = (u1 + u2 - 1.0) * 0.5
    return frame + tri


# --------------------------------------------------------------------------
# Stage 4 - spatial alignment (conditional ECC warp)
# --------------------------------------------------------------------------

def downsample_for_ecc(frame: "np.ndarray", max_width: int) -> tuple:
    height, width = frame.shape[:2]
    if width <= max_width:
        return frame, 1.0
    scale = max_width / width
    size = (int(round(width * scale)), int(round(height * scale)))
    return cv2.resize(frame, size, interpolation=cv2.INTER_AREA), scale


def scale_affine(matrix: "np.ndarray", factor: float) -> "np.ndarray":
    scaled = matrix.copy()
    scaled[0, 2] *= factor
    scaled[1, 2] *= factor
    return scaled


def estimate_affine_ecc(reference_frame: "np.ndarray", source_frame: "np.ndarray",
                         mask_margin: int, ecc_width: int = ECC_WIDTH) -> tuple:
    ref_small, scale = downsample_for_ecc(reference_frame, ecc_width)
    src_small, _ = downsample_for_ecc(source_frame, ecc_width)
    small_mask = make_mask(ref_small.shape[0], ref_small.shape[1], round(mask_margin * scale))
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 1500, 1e-8)
    initial = np.eye(2, 3, dtype=np.float32)
    try:
        cc, matrix = cv2.findTransformECC(
            highpass_f01(ref_small), highpass_f01(src_small), initial,
            cv2.MOTION_AFFINE, criteria, small_mask, 5,
        )
    except cv2.error:
        return np.eye(2, 3, dtype=np.float32), 0.0

    matrix = scale_affine(matrix, 1.0 / scale).astype(np.float32)
    if scale == 1.0:
        return matrix, float(cc)

    full_mask = make_mask(reference_frame.shape[0], reference_frame.shape[1], mask_margin)
    full_criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 500, 1e-9)
    try:
        cc, matrix = cv2.findTransformECC(
            highpass_f01(reference_frame), highpass_f01(source_frame), matrix,
            cv2.MOTION_AFFINE, full_criteria, full_mask, 5,
        )
    except cv2.error:
        pass
    return matrix.astype(np.float32), float(cc)


def warp_frame(frame: "np.ndarray", matrix: "np.ndarray", width: int, height: int) -> "np.ndarray":
    return cv2.warpAffine(
        frame, matrix, (width, height),
        flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_REPLICATE,
    )


def max_corner_displacement(matrix: "np.ndarray", width: int, height: int) -> float:
    corners = np.array([[0, 0], [width - 1, 0], [0, height - 1], [width - 1, height - 1]], dtype=np.float64)
    a = matrix[:, :2].astype(np.float64)
    t = matrix[:, 2].astype(np.float64)
    transformed = corners @ a.T + t
    disp = np.linalg.norm(transformed - corners, axis=1)
    return float(disp.max())


# --------------------------------------------------------------------------
# Stage 5 - seam finishing
# --------------------------------------------------------------------------

def smoothstep(x: float) -> float:
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


def seam_decay(t: int, total: int) -> float:
    """Residual weight at transition step t.

    Peak weight is capped below 1.0 (SEAM_PEAK_WEIGHT) so no frame is ever
    fully overwritten by the propagated residual -- overwriting the first
    altered frame paints real motion difference onto it and causes a visible
    motion hesitation at the cut.
    """
    if total <= 1:
        return SEAM_PEAK_WEIGHT if t == 0 else 0.0
    return SEAM_PEAK_WEIGHT * (1.0 - smoothstep(t / (total - 1)))


def compute_seam_data(ref_boundary: "np.ndarray", corrected_boundary: "np.ndarray",
                       margin: int) -> SeamData:
    mae0 = float(np.mean(np.abs(ref_boundary.astype(np.float32) - corrected_boundary.astype(np.float32))))
    if mae0 < SEAM_CLEAN_MAE:
        return SeamData(active=False, mae0=mae0)

    warped_ref, corrected_f, mask, fb_mask = align_pair_for_color(corrected_boundary, ref_boundary, margin)
    # warped_ref is ref_boundary motion-compensated onto corrected_boundary's grid
    r0 = warped_ref - corrected_f
    conf2d = fb_mask.astype(np.float32)
    # Confidence-weighted low-frequency band (normalized convolution): where
    # flow tracking failed, the residual is real motion difference, not color
    # drift -- those regions must contribute nothing rather than their raw
    # difference. The epsilon in the denominator makes the band decay to zero
    # where local confidence density is low.
    conf_blur = cv2.GaussianBlur(conf2d, (0, 0), LOW_FREQ_SIGMA)
    low_freq = np.stack(
        [cv2.GaussianBlur(r0[..., c] * conf2d, (0, 0), LOW_FREQ_SIGMA) /
         (conf_blur + LOW_FREQ_CONF_EPS) for c in range(3)],
        axis=-1,
    )
    detail = r0 - low_freq
    conf = conf2d[..., None]
    detail0 = detail * conf
    return SeamData(active=True, mae0=mae0, low_freq=low_freq.astype(np.float32),
                     detail0=detail0.astype(np.float32), confidence_ratio=float(fb_mask.mean()))


def apply_seam_to_window(frames_chrono: list, low_freq, detail0, total: int, direction: str) -> list:
    n = len(frames_chrono)
    total = min(total, n)  # never spread the decay curve wider than the frames we have
    out = list(frames_chrono)
    seq = list(range(n)) if direction == "forward" else list(range(n - 1, -1, -1))
    current_detail = detail0
    prev_idx = None
    for k, idx in enumerate(seq):
        if k > 0:
            flow = compute_flow(highpass_u8(frames_chrono[prev_idx]), highpass_u8(frames_chrono[idx]))
            current_detail = warp_by_flow(current_detail, flow)
        w = seam_decay(k, total)
        if w > 0:
            out[idx] = frames_chrono[idx] + low_freq * w + current_detail * w
        prev_idx = idx
    return out


# --------------------------------------------------------------------------
# Stage 6 - encode, joined preview, seam metric, diagnostics, report
# --------------------------------------------------------------------------

def write_frame(stdin, frame_float: "np.ndarray") -> None:
    dithered = add_triangular_dither(frame_float)
    u8 = np.clip(np.round(dithered), 0, 255).astype(np.uint8)
    stdin.write(u8.tobytes())


def build_encoder_cmd(output: Path, width: int, height: int, rate_arg: str, crf: int, preset: str,
                       color_tags: tuple, audio_path: Optional[Path], audio_seek: float,
                       has_audio: bool) -> list[str]:
    cmd = [
        "ffmpeg", "-y", "-v", "warning",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", rate_arg, "-i", "-",
    ]
    maps = ["-map", "0:v:0"]
    if has_audio and audio_path is not None:
        if audio_seek > 0:
            cmd += ["-ss", f"{audio_seek:.6f}"]
        cmd += ["-i", str(audio_path)]
        maps += ["-map", "1:a:0"]
    cmd += maps
    cmd += ["-c:v", "libx264", "-preset", preset, "-crf", str(crf), "-pix_fmt", "yuv420p"]
    if color_tags:
        cmd += ["-colorspace", color_tags[0], "-color_primaries", color_tags[1], "-color_trc", color_tags[2]]
    if has_audio and audio_path is not None:
        cmd += ["-c:a", "copy"]
    cmd += ["-movflags", "+faststart", str(output)]
    return cmd


def render_altered_clip(source_path: Path, output: Path, decode_width: int, decode_height: int,
                         decode_vf: Optional[str], skip: int, out_width: int, out_height: int,
                         rate_arg: str, affine_matrix, apply_warp: bool, color_matrix, luts,
                         seam: SeamData, anchor_side: str, transition_frames: int,
                         crf: int, preset: str, color_tags: tuple, audio_path: Optional[Path],
                         audio_seek: float, has_audio: bool) -> int:

    def pipeline(raw: "np.ndarray") -> "np.ndarray":
        f = raw
        if apply_warp:
            f = warp_frame(f, affine_matrix, out_width, out_height)
        return apply_color(f, color_matrix, luts)

    encoder = subprocess.Popen(
        build_encoder_cmd(output, out_width, out_height, rate_arg, crf, preset, color_tags,
                           audio_path, audio_seek, has_audio),
        stdin=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    assert encoder.stdin is not None

    gen = iter_frames(source_path, decode_width, decode_height, vf=decode_vf)
    skip_frames(gen, skip)

    frame_count = 0
    do_transition = seam.active and transition_frames > 0
    T = transition_frames

    try:
        if do_transition and anchor_side == "start":
            buf = []
            for raw in gen:
                buf.append(pipeline(raw))
                frame_count += 1
                if len(buf) >= T:
                    break
            processed = apply_seam_to_window(buf, seam.low_freq, seam.detail0, T, "forward") if buf else buf
            for f in processed:
                write_frame(encoder.stdin, f)
            for raw in gen:
                write_frame(encoder.stdin, pipeline(raw))
                frame_count += 1
        elif do_transition and anchor_side == "end":
            trailing: list = []
            for raw in gen:
                corrected = pipeline(raw)
                trailing.append(corrected)
                frame_count += 1
                if len(trailing) > T:
                    write_frame(encoder.stdin, trailing.pop(0))
            processed = apply_seam_to_window(trailing, seam.low_freq, seam.detail0, T, "backward") if trailing else trailing
            for f in processed:
                write_frame(encoder.stdin, f)
        else:
            for raw in gen:
                write_frame(encoder.stdin, pipeline(raw))
                frame_count += 1
    except BrokenPipeError as exc:
        raise RuntimeError("encoder closed before all frames were written") from exc
    finally:
        try:
            encoder.stdin.close()
        except Exception:
            pass

    stderr = encoder.stderr.read().decode("utf-8", errors="replace") if encoder.stderr else ""
    status = encoder.wait()
    if status:
        raise RuntimeError(f"encoder failed with exit code {status}\n{stderr}")
    return frame_count


def build_joined_preview(prev_path: Path, prev_skip: int, prev_vf: Optional[str],
                          next_path: Path, next_skip: int, next_vf: Optional[str],
                          width: int, height: int, rate_arg: str, crf: int, preset: str,
                          color_tags: tuple, joined_output: Path) -> dict:
    encoder = subprocess.Popen(
        build_encoder_cmd(joined_output, width, height, rate_arg, crf, preset, color_tags,
                           None, 0.0, False),
        stdin=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    assert encoder.stdin is not None

    d_values: list[float] = []
    prev_frame = None
    cut_step: Optional[int] = None
    step_idx = 0
    prev_count = 0

    try:
        gen1 = iter_frames(prev_path, width, height, vf=prev_vf)
        skip_frames(gen1, prev_skip)
        for frame in gen1:
            if prev_frame is not None:
                d_values.append(float(np.mean(np.abs(frame.astype(np.float32) - prev_frame.astype(np.float32)))))
                step_idx += 1
            write_frame(encoder.stdin, frame.astype(np.float32))
            prev_frame = frame
            prev_count += 1

        gen2 = iter_frames(next_path, width, height, vf=next_vf)
        skip_frames(gen2, next_skip)
        first_next = True
        for frame in gen2:
            if prev_frame is not None:
                d_values.append(float(np.mean(np.abs(frame.astype(np.float32) - prev_frame.astype(np.float32)))))
                if first_next:
                    cut_step = step_idx
                    first_next = False
                step_idx += 1
            write_frame(encoder.stdin, frame.astype(np.float32))
            prev_frame = frame
    except BrokenPipeError as exc:
        raise RuntimeError("joined-preview encoder closed early") from exc
    finally:
        try:
            encoder.stdin.close()
        except Exception:
            pass

    stderr = encoder.stderr.read().decode("utf-8", errors="replace") if encoder.stderr else ""
    status = encoder.wait()
    if status:
        raise RuntimeError(f"joined preview encoder failed with exit code {status}\n{stderr}")

    return {"d_values": d_values, "cut_step": cut_step, "prev_frame_count": prev_count}


def seam_zscore(d_values: list[float], cut_step: Optional[int], window: int = 24) -> dict:
    if cut_step is None or cut_step >= len(d_values):
        return {"z": None, "d_cut": None, "reason": "no cut step available"}
    lo = max(0, cut_step - window)
    hi = min(len(d_values), cut_step + window + 1)
    neighborhood = [d_values[i] for i in range(lo, hi) if i != cut_step]
    if len(neighborhood) < 4:
        return {"z": None, "d_cut": d_values[cut_step], "reason": "not enough neighboring steps"}
    d_cut = d_values[cut_step]
    median = float(np.median(neighborhood))
    mad = float(np.median(np.abs(np.array(neighborhood) - median)))
    sigma = mad * 1.4826
    if sigma < 1e-6:
        sigma = float(np.std(neighborhood)) or 1e-6
    z = (d_cut - median) / sigma
    return {"z": float(z), "d_cut": float(d_cut), "median_neighbors": median, "sigma": sigma}


def write_png(path: Path, rgb: "np.ndarray") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(np.clip(rgb, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR))


def draw_lut_plot(luts: "np.ndarray", path: Path) -> None:
    size = 512
    canvas = np.full((size, size, 3), 255, np.uint8)
    for i in range(0, 256, 32):
        x = int(i / 255 * (size - 1))
        cv2.line(canvas, (x, 0), (x, size - 1), (220, 220, 220), 1)
        cv2.line(canvas, (0, size - 1 - x), (size - 1, size - 1 - x), (220, 220, 220), 1)
    cv2.line(canvas, (0, size - 1), (size - 1, 0), (180, 180, 180), 1)
    colors = [(220, 60, 60), (60, 160, 60), (60, 60, 220)]  # R, G, B in BGR order for cv2
    for c in range(3):
        pts = []
        for i in range(256):
            x = int(i / 255 * (size - 1))
            y = int(size - 1 - np.clip(luts[c, i], 0, 255) / 255 * (size - 1))
            pts.append((x, y))
        cv2.polylines(canvas, [np.array(pts, np.int32)], False, colors[c], 2)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), canvas)


def frame_metrics(reference: "np.ndarray", candidate: "np.ndarray") -> dict:
    diff = reference.astype(np.float32) - candidate.astype(np.float32)
    return {"mae": float(np.mean(np.abs(diff))), "rmse": float(math.sqrt(float(np.mean(diff * diff))))}


# --------------------------------------------------------------------------
# Plan / CLI plumbing
# --------------------------------------------------------------------------

def default_output_path(previous: Path, next_clip: Path, alter: str) -> Path:
    source = next_clip if alter == "next" else previous
    return source.with_name(f"{source.stem}_seamless{source.suffix}")


def default_report_path(output: Path) -> Path:
    return output.with_name(f"{output.stem}_report.json")


def default_diagnostics_dir(output: Path) -> Path:
    return output.with_name(f"{output.stem}_diagnostics")


def default_joined_path(output: Path) -> Path:
    return output.with_name(f"{output.stem}_joined_preview.mp4")


def build_plan(previous: Path, next_clip: Path, alter: str) -> SeamPlan:
    if alter == "next":
        return SeamPlan(
            previous=previous, next_clip=next_clip, source=next_clip, reference=previous,
            source_boundary="first", reference_boundary="last", anchor_side="start",
            altered_label="next clip start", reference_label="previous clip end",
        )
    if alter == "previous":
        return SeamPlan(
            previous=previous, next_clip=next_clip, source=previous, reference=next_clip,
            source_boundary="last", reference_boundary="first", anchor_side="end",
            altered_label="previous clip end", reference_label="next clip start",
        )
    raise ValueError(f"unknown alter mode: {alter}")


def parse_trim_arg(value: str):
    if value == "auto":
        return "auto"
    try:
        return int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--trim must be 'auto' or an integer") from exc


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="seamstress.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Make the transition between two separately generated clips of the "
            "same continuous shot visually seamless."
        ),
        epilog="""\
Typical use:
  ./seamstress.py previous.mp4 next.mp4

Preserve the next clip and alter the previous clip's ending instead:
  ./seamstress.py previous.mp4 next.mp4 --alter previous

Write explicit outputs:
  ./seamstress.py previous.mp4 next.mp4 \\
    --output next_fixed.mp4 \\
    --report next_fixed_report.json \\
    --diagnostics-dir next_fixed_diagnostics

What the tool does:
  1. Ingests both clips and conforms fps/resolution mismatches.
  2. Detects head/tail overlap and trims duplicate frames.
  3. Solves a pooled, motion-compensated global color match (affine+curves).
  4. Conditionally applies a subpixel ECC affine warp.
  5. Propagates a motion-compensated residual across a short transition.
  6. Encodes the result, a joined preview, and a diagnostics/report bundle.
""",
    )
    parser.add_argument("previous", type=Path, help="Earlier clip in the timeline.")
    parser.add_argument("next", type=Path, help="Later clip in the timeline.")
    parser.add_argument("--alter", choices=("next", "previous"), default="next",
                         help="Which clip to re-render. Default: next.")
    parser.add_argument("-o", "--output", type=Path, help="Path for the corrected clip.")
    parser.add_argument("--report", type=Path, help="Path for the JSON report.")
    parser.add_argument("--diagnostics-dir", type=Path, help="Directory for diagnostic PNGs.")
    parser.add_argument("--no-diagnostics", action="store_true", help="Do not write diagnostic PNGs.")
    parser.add_argument("--mask-margin", type=int, default=48,
                         help="Pixels to ignore at each frame edge during alignment/color solving. Default: 48.")
    parser.add_argument("--trim-percentile", type=float, default=75.0,
                         help="Residual percentile kept while solving the robust affine color matrix. Default: 75.0.")
    parser.add_argument("--trim", type=parse_trim_arg, default="auto",
                         help="'auto' to detect head overlap, or an integer frame count. Use 0 to disable. Default: auto.")
    parser.add_argument("--fps-policy", choices=("conform", "strict", "resample"), default="conform",
                         help="How to reconcile differing nominal frame rates. Default: conform.")
    parser.add_argument("--color-model", choices=("affine", "affine+curves"), default="affine+curves",
                         help="Color correction model. Default: affine+curves.")
    parser.add_argument("--warp", choices=("auto", "always", "never"), default="auto",
                         help="Whether to apply the Stage 4 spatial warp. Default: auto.")
    parser.add_argument("--transition-frames", type=int, default=6,
                         help="Frames used for the Stage 5 residual-propagation transition. Use 0 to disable. Default: 6.")
    parser.add_argument("--joined", type=Path, default=None,
                         help="Path for the joined timeline QC preview. Default: <output stem>_joined_preview.mp4.")
    parser.add_argument("--no-joined", action="store_true", help="Do not write the joined preview.")
    parser.add_argument("--crf", type=int, default=10, help="libx264 CRF for the corrected clip. Default: 10.")
    parser.add_argument("--preset", default="slow", help="libx264 preset. Default: slow.")
    return parser.parse_args(argv)


def resolved_path(path: Path) -> Path:
    return path.expanduser().resolve()


def run(args: argparse.Namespace) -> int:
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        raise RuntimeError("ffmpeg and ffprobe must be on PATH")

    load_python_dependencies()
    warnings: list[str] = []

    previous = resolved_path(args.previous)
    next_clip = resolved_path(args.next)
    for path in (previous, next_clip):
        if not path.is_file():
            raise RuntimeError(f"not found: {path}")

    output = resolved_path(args.output) if args.output else default_output_path(previous, next_clip, args.alter)
    if output in (previous, next_clip):
        raise RuntimeError("output must be different from both input paths")
    report_path = resolved_path(args.report) if args.report else default_report_path(output)
    diagnostics_dir = resolved_path(args.diagnostics_dir) if args.diagnostics_dir else default_diagnostics_dir(output)
    joined_path = None
    if not args.no_joined:
        joined_path = resolved_path(args.joined) if args.joined else default_joined_path(output)

    plan = build_plan(previous, next_clip, args.alter)

    print(f"previous clip: {plan.previous.name}")
    print(f"next clip:     {plan.next_clip.name}")
    print(f"altering:      {plan.source.name} ({plan.altered_label})")
    print(f"reference:     {plan.reference.name} ({plan.reference_label})")

    # ---------------- Stage 1: ingest & conform ----------------
    previous_info = ffprobe_video(previous)
    next_info = ffprobe_video(next_clip)
    source_info = next_info if plan.source == next_clip else previous_info
    reference_info = previous_info if plan.reference == previous else next_info

    print(f"video:         {source_info.size_arg} @ {source_info.rate_arg} fps "
          f"-> reference {reference_info.size_arg} @ {reference_info.rate_arg} fps")

    if args.fps_policy == "resample":
        raise RuntimeError("--fps-policy resample is not implemented")

    fps_diff_ratio = abs(source_info.fps - reference_info.fps) / max(reference_info.fps, 1e-6)
    if args.fps_policy == "strict":
        if fps_diff_ratio > 1e-6:
            raise RuntimeError("clips must have the same frame rate (--fps-policy strict)")
    else:
        if fps_diff_ratio > FPS_WARN_RATIO:
            msg = (f"warning: nominal fps differ by {fps_diff_ratio * 100:.2f}% "
                   f"({source_info.rate_arg} vs {reference_info.rate_arg}); "
                   "conforming altered clip to the reference clip's timebase")
            print(msg)
            warnings.append(msg)

    out_width, out_height = reference_info.width, reference_info.height
    need_scale = (source_info.width, source_info.height) != (out_width, out_height)
    conform_vf = None
    if need_scale:
        ar_source = source_info.width / source_info.height
        ar_ref = out_width / out_height
        ar_diff = abs(ar_source - ar_ref) / ar_ref
        if ar_diff > ASPECT_REFUSE_RATIO:
            raise RuntimeError(
                f"aspect ratio mismatch too large ({ar_diff * 100:.2f}%); refusing to scale"
            )
        msg = (f"warning: resolution mismatch ({source_info.size_arg} vs {reference_info.size_arg}); "
               "lanczos-scaling altered clip to reference size")
        print(msg)
        warnings.append(msg)
        conform_vf = f"scale={out_width}:{out_height}:flags=lanczos"

    previous_is_source = plan.source == previous
    next_is_source = plan.source == next_clip

    def vf_for(is_source_file: bool) -> Optional[str]:
        return conform_vf if is_source_file else None

    color_tags_source = (source_info.color_space, source_info.color_primaries, source_info.color_transfer)

    total_prev = probe_frame_count(previous, previous_info)
    total_next = probe_frame_count(next_clip, next_info)
    print(f"frame counts:  previous={total_prev}, next={total_next}")

    # ---------------- Stage 2: temporal alignment ----------------
    alignment = temporal_align(previous, previous_info, next_clip, next_info, total_prev)
    if args.trim == "auto":
        if alignment.accepted:
            trim = alignment.trim
            print(f"overlap detected: trim={trim} frames, confidence={alignment.confidence:.2f}, "
                  f"peak NCC={alignment.peak_ncc:.3f}")
        else:
            trim = 0
            print(f"no overlap detected ({alignment.reason}); assuming butt joint (trim=0)")
    else:
        trim = max(0, int(args.trim))
        print(f"manual trim override: trim={trim} "
              f"(auto-detected trim={alignment.trim if alignment.accepted else 0}, "
              f"confidence={alignment.confidence:.2f})")

    if trim >= total_next:
        raise RuntimeError(f"trim ({trim}) would remove all frames of the next clip ({total_next})")

    k_overlap = trim if (args.trim == "auto" and alignment.accepted) else (trim if trim > 0 else 0)
    if k_overlap and k_overlap != alignment.overlap_k:
        # manual override: assume the same simple 1:1 tail<->head mapping
        pairs = [(total_prev - k_overlap + i, i, 0.0) for i in range(k_overlap)]
    else:
        pairs = alignment.pairs

    audio_seek = 0.0
    if next_is_source and trim > 0:
        audio_seek = trim * next_info.fps_den / next_info.fps_num

    # ---------------- fetch full-res boundary frames ----------------
    need_prev_tail = max(k_overlap, 1)
    prev_tail_full = fetch_frames(previous, out_width, out_height, need_prev_tail,
                                   skip=total_prev - need_prev_tail, vf=vf_for(previous_is_source))
    prev_last_global = total_prev - 1
    prev_tail_base = total_prev - need_prev_tail

    if k_overlap > 0:
        need_next_head = max(k_overlap, trim + 1)
    else:
        need_next_head = max(5, trim + 1)
    next_head_full = fetch_frames(next_clip, out_width, out_height, need_next_head, skip=0,
                                   vf=vf_for(next_is_source))

    def prev_frame_at(global_idx: int) -> "np.ndarray":
        local = global_idx - prev_tail_base
        if 0 <= local < len(prev_tail_full):
            return prev_tail_full[local]
        return fetch_frames(previous, out_width, out_height, 1, skip=global_idx,
                             vf=vf_for(previous_is_source))[0]

    def next_frame_at(idx: int) -> "np.ndarray":
        if idx < len(next_head_full):
            return next_head_full[idx]
        return fetch_frames(next_clip, out_width, out_height, 1, skip=idx,
                             vf=vf_for(next_is_source))[0]

    # ---------------- Stage 3: global color match ----------------
    color_pairs_frames = []  # (ref_frame, src_frame) in plan.reference/plan.source roles
    if k_overlap > 0:
        for kk in range(k_overlap):
            prev_g = total_prev - k_overlap + kk
            pf = prev_frame_at(prev_g)
            nf = next_frame_at(kk)
            if plan.reference == previous:
                color_pairs_frames.append((pf, nf))
            else:
                color_pairs_frames.append((nf, pf))
    else:
        prev_last = prev_frame_at(prev_last_global)
        extra_count = min(5, total_next)
        for kk in range(extra_count):
            nf = next_frame_at(kk)
            if plan.reference == previous:
                color_pairs_frames.append((prev_last, nf))
            else:
                color_pairs_frames.append((nf, prev_last))

    src_pool, ref_pool, kept_ratio = build_correspondence_pool(color_pairs_frames, args.mask_margin)
    print(f"color correspondence: {len(src_pool)} pixels pooled from {len(color_pairs_frames)} pair(s), "
          f"{kept_ratio * 100:.1f}% kept")

    affine_matrix = solve_affine_trimmed(src_pool, ref_pool, args.trim_percentile)
    luts = None
    if args.color_model == "affine+curves":
        luts = fit_channel_curves(src_pool, ref_pool, affine_matrix)

    mean_shift = ref_pool.mean(axis=0) - src_pool.mean(axis=0)
    color_fallback_reason = None
    if kept_ratio < MIN_KEPT_RATIO:
        color_fallback_reason = (
            f"only {kept_ratio * 100:.2f}% of pixels produced confident correspondences "
            f"(< {MIN_KEPT_RATIO * 100:.0f}%) -- the clips likely share no matching content"
        )
    elif np.any(np.abs(mean_shift) > MEAN_SHIFT_WARN):
        color_fallback_reason = (
            f"solved color transform implies a global mean shift of {mean_shift.tolist()} "
            f"(> {MEAN_SHIFT_WARN} code values on at least one channel)"
        )

    if color_fallback_reason is not None:
        msg = ("WARNING: rejecting solved color transform and falling back to IDENTITY "
               f"(no color change): {color_fallback_reason}")
        print(msg)
        warnings.append(msg)
        affine_matrix = np.vstack([np.eye(3), np.zeros(3)]).astype(np.float64)
        luts = None

    print("affine color matrix (input RGB + bias -> output RGB):")
    print(affine_matrix)

    # ---------------- Stage 4: spatial alignment ----------------
    if k_overlap > 0:
        best_pair = max(pairs, key=lambda p: p[2]) if pairs else (total_prev - 1, 0, 0.0)
        prev_idx_for_ecc, next_idx_for_ecc = best_pair[0], best_pair[1]
    else:
        prev_idx_for_ecc, next_idx_for_ecc = prev_last_global, 0

    prev_ecc_frame = prev_frame_at(prev_idx_for_ecc)
    next_ecc_frame = next_frame_at(next_idx_for_ecc)
    if plan.reference == previous:
        ecc_ref_frame, ecc_src_frame = prev_ecc_frame, next_ecc_frame
    else:
        ecc_ref_frame, ecc_src_frame = next_ecc_frame, prev_ecc_frame

    ecc_matrix, ecc_score = estimate_affine_ecc(ecc_ref_frame, ecc_src_frame, args.mask_margin)
    corner_disp = max_corner_displacement(ecc_matrix, out_width, out_height)
    print(f"ECC score: {ecc_score:.4f}, max corner displacement: {corner_disp:.3f}px")

    if args.warp == "always":
        apply_warp = True
    elif args.warp == "never":
        apply_warp = False
    else:
        apply_warp = ecc_score >= 0.5 and 0.2 <= corner_disp <= 12.0
    print(f"spatial warp: {'applied' if apply_warp else 'skipped'} (mode={args.warp})")
    affine_for_render = ecc_matrix if apply_warp else np.eye(2, 3, dtype=np.float32)

    # ---------------- Stage 5: seam finishing ----------------
    def pipeline_frame(raw: "np.ndarray") -> "np.ndarray":
        f = raw
        if apply_warp:
            f = warp_frame(f, affine_for_render, out_width, out_height)
        return apply_color(f, affine_matrix, luts)

    if plan.source == next_clip:
        source_boundary_raw = next_frame_at(trim)
        reference_boundary_raw = prev_frame_at(prev_last_global)
    else:
        source_boundary_raw = prev_frame_at(prev_last_global)
        reference_boundary_raw = next_frame_at(trim)

    corrected_boundary = pipeline_frame(source_boundary_raw)
    seam = compute_seam_data(reference_boundary_raw.astype(np.float32), corrected_boundary, args.mask_margin)
    if seam.active:
        print(f"seam residual: MAE {seam.mae0:.3f} (>= {SEAM_CLEAN_MAE}); "
              f"propagating over {args.transition_frames} frames")
    else:
        print(f"seam residual: MAE {seam.mae0:.3f} (< {SEAM_CLEAN_MAE}); clean cut, no transition needed")

    raw_metrics = frame_metrics(reference_boundary_raw, source_boundary_raw)
    corrected_metrics = frame_metrics(reference_boundary_raw, corrected_boundary)
    print(f"raw boundary:       MAE {raw_metrics['mae']:.4f}")
    print(f"corrected boundary: MAE {corrected_metrics['mae']:.4f}")

    # ---------------- render altered clip ----------------
    output.parent.mkdir(parents=True, exist_ok=True)
    render_skip = trim if plan.source == next_clip else 0
    render_audio_path = plan.source
    render_has_audio = source_info.has_audio

    frames_written = render_altered_clip(
        source_path=plan.source, output=output,
        decode_width=source_info.width, decode_height=source_info.height, decode_vf=conform_vf,
        skip=render_skip, out_width=out_width, out_height=out_height, rate_arg=reference_info.rate_arg,
        affine_matrix=affine_for_render, apply_warp=apply_warp, color_matrix=affine_matrix, luts=luts,
        seam=seam, anchor_side=plan.anchor_side, transition_frames=max(0, args.transition_frames),
        crf=args.crf, preset=args.preset, color_tags=color_tags_source,
        audio_path=render_audio_path, audio_seek=audio_seek, has_audio=render_has_audio,
    )
    print(f"rendered {frames_written} frames -> {output}")

    output_info = ffprobe_video(output)
    output_boundary = fetch_frames(
        output, output_info.width, output_info.height, 1,
        skip=(0 if plan.source == next_clip else output_info.frames - 1),
    )
    output_boundary = output_boundary[0] if output_boundary else corrected_boundary.astype(np.uint8)
    encoded_metrics = frame_metrics(reference_boundary_raw, output_boundary)
    print(f"encoded output boundary: MAE {encoded_metrics['mae']:.4f}")

    # ---------------- joined preview + seam metric ----------------
    seam_report = {"z": None}
    joined_result = None
    if joined_path is not None:
        if plan.source == previous:
            joined_prev_path, joined_prev_skip, joined_prev_vf = output, 0, None
        else:
            joined_prev_path, joined_prev_skip, joined_prev_vf = previous, 0, None
        if plan.source == next_clip:
            joined_next_path, joined_next_skip, joined_next_vf = output, 0, None
        else:
            joined_next_path, joined_next_skip, joined_next_vf = next_clip, trim, vf_for(next_is_source)

        joined_result = build_joined_preview(
            joined_prev_path, joined_prev_skip, joined_prev_vf,
            joined_next_path, joined_next_skip, joined_next_vf,
            out_width, out_height, reference_info.rate_arg, args.crf, args.preset,
            color_tags_source, joined_path,
        )
        seam_report = seam_zscore(joined_result["d_values"], joined_result["cut_step"])
        print(f"joined preview: {joined_path}")
        if seam_report["z"] is not None:
            status = "PASS" if abs(seam_report["z"]) <= 2.0 else "WARNING: seam z-score outside +/-2"
            print(f"seam z-score: {seam_report['z']:.3f} ({status})")
        else:
            print(f"seam z-score: unavailable ({seam_report.get('reason')})")

    # ---------------- diagnostics ----------------
    if not args.no_diagnostics:
        diagnostics_dir.mkdir(parents=True, exist_ok=True)
        write_png(diagnostics_dir / "01_reference_boundary.png", reference_boundary_raw)
        write_png(diagnostics_dir / "02_source_boundary.png", source_boundary_raw)
        write_png(diagnostics_dir / "03_source_corrected.png", corrected_boundary)
        write_png(diagnostics_dir / "04_output_boundary_encoded.png", output_boundary)
        absdiff = np.abs(reference_boundary_raw.astype(np.int16) - output_boundary.astype(np.int16))
        write_png(diagnostics_dir / "05_encoded_absdiff_amplified.png", np.clip(absdiff * 6, 0, 255))
        if seam.active:
            conf_vis = np.clip(seam.detail0.mean(axis=2) * 4 + 128, 0, 255)
            write_png(diagnostics_dir / "06_flow_confidence.png",
                      np.stack([conf_vis] * 3, axis=-1))
        if luts is not None:
            draw_lut_plot(luts, diagnostics_dir / "07_color_curves.png")
        print(f"diagnostics: {diagnostics_dir}")

    # ---------------- report ----------------
    report_data = {
        "previous": str(previous),
        "next": str(next_clip),
        "alter": args.alter,
        "source": str(plan.source),
        "reference": str(plan.reference),
        "output": str(output),
        "joined_preview": str(joined_path) if joined_path else None,
        "frames_written": frames_written,
        "fps_policy": args.fps_policy,
        "resolution_conform": need_scale,
        "trim": trim,
        "trim_mode": "auto" if args.trim == "auto" else "manual",
        "alignment": {
            "accepted": alignment.accepted,
            "detected_trim": alignment.trim,
            "confidence": alignment.confidence,
            "peak_ncc": alignment.peak_ncc,
            "best_offset": alignment.best_offset,
            "tail_n": alignment.tail_n,
            "head_m": alignment.head_m,
            "reason": alignment.reason,
        },
        "color_model": args.color_model,
        "affine_matrix": affine_matrix.tolist(),
        "luts": luts.tolist() if luts is not None else None,
        "mean_shift": mean_shift.tolist(),
        "color_fallback": color_fallback_reason,
        "warp": {
            "mode": args.warp,
            "applied": bool(apply_warp),
            "ecc_score": ecc_score,
            "max_corner_displacement": corner_disp,
            "matrix": ecc_matrix.tolist(),
        },
        "seam": {
            "active": seam.active,
            "mae0": seam.mae0,
            "transition_frames": args.transition_frames,
            "confidence_ratio": seam.confidence_ratio,
            "zscore": seam_report,
        },
        "raw_boundary": raw_metrics,
        "corrected_boundary": corrected_metrics,
        "encoded_output_boundary": encoded_metrics,
        "mask_margin": args.mask_margin,
        "trim_percentile": args.trim_percentile,
        "crf": args.crf,
        "preset": args.preset,
        "warnings": warnings,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report_data, indent=2), encoding="utf-8")
    print(f"report: {report_path}")
    return 0


def main(argv: list[str]) -> int:
    try:
        args = parse_args(argv)
        return run(args)
    except subprocess.CalledProcessError as exc:
        print(f"error: command failed with exit code {exc.returncode}: {' '.join(exc.cmd)}", file=sys.stderr)
        return 1
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
