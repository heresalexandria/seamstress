"""Frame-exact RGB video IO backed by FFmpeg; original media is never modified."""

from __future__ import annotations

import json
import math
import shutil
import subprocess
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterator

import numpy as np

_PROBE_CACHE: dict[tuple, dict[str, Any]] = {}


def _tool(name: str) -> str:
    executable = shutil.which(name)
    if executable is None:
        raise RuntimeError(f"{name} is required; install FFmpeg and add it to PATH")
    return executable


def _run(arguments: list[str]) -> subprocess.CompletedProcess[bytes]:
    result = subprocess.run(arguments, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if result.returncode:
        error = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"{Path(arguments[0]).name} failed: {error}")
    return result


def probe(path: Path | str) -> dict[str, Any]:
    """Return video geometry, rate, duration, frame count, and audio presence."""
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    stat = path.stat()
    cache_key = (str(path), stat.st_size, stat.st_mtime_ns)
    if cache_key in _PROBE_CACHE:
        return _PROBE_CACHE[cache_key].copy()
    result = _run([
        _tool("ffprobe"), "-v", "error", "-show_streams", "-show_format",
        "-of", "json", str(path),
    ])
    data = json.loads(result.stdout)
    video = next((stream for stream in data.get("streams", []) if stream.get("codec_type") == "video" and not stream.get("disposition", {}).get("attached_pic")), None)
    if video is None:
        raise ValueError(f"no video stream found in {path}")
    fps_fraction = video.get("avg_frame_rate", "0/0")
    if fps_fraction in ("0/0", "0", "N/A"):
        fps_fraction = video.get("r_frame_rate", "0/0")
    try:
        fps = float(Fraction(fps_fraction))
    except (ValueError, ZeroDivisionError) as exc:
        raise ValueError(f"cannot determine frame rate for {path}") from exc
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError(f"invalid frame rate for {path}: {fps_fraction}")
    duration = float(video.get("duration") or data.get("format", {}).get("duration") or 0)
    frame_count = video.get("nb_frames")
    estimated = frame_count in (None, "N/A", "0")
    if estimated:
        counted = json.loads(_run([_tool('ffprobe'), '-v', 'error', '-count_frames',
            '-select_streams', str(video.get('index', 0)), '-show_entries', 'stream=nb_read_frames',
            '-of', 'json', str(path)]).stdout)
        actual = next(iter(counted.get('streams', [])), {}).get('nb_read_frames')
        if actual not in (None, 'N/A', '0'):
            frame_count, estimated = int(actual), False
        else:
            frame_count = round(duration * fps)
    else:
        frame_count = int(frame_count)
    rotation = next((float(s.get('rotation', 0)) for s in video.get('side_data_list', []) if 'rotation' in s),
                    float(video.get('tags', {}).get('rotate', 0)))
    width, height = int(video['width']), int(video['height'])
    if round(rotation) % 180 == 90:
        width, height = height, width
    metadata = {
        "path": str(path), "width": width, "height": height,
        "fps": fps, "fps_fraction": fps_fraction, "frame_count": frame_count,
        "frame_count_estimated": estimated, "duration": duration,
        "has_audio": any(stream.get("codec_type") == "audio" for stream in data.get("streams", [])),
        "codec_name": video.get("codec_name"), "pix_fmt": video.get("pix_fmt"),
        "color_space": video.get("color_space"), "color_transfer": video.get("color_transfer"),
        "color_primaries": video.get("color_primaries"), "color_range": video.get("color_range"),
        "rotation": rotation, "video_stream_index": int(video.get('index', 0)),
        "sample_aspect_ratio": video.get("sample_aspect_ratio", "1:1"),
        "video_start_time": float(video.get("start_time") or 0),
        "audio_start_times": [float(stream.get("start_time") or 0) for stream in data.get("streams", [])
                              if stream.get("codec_type") == "audio"],
    }
    _PROBE_CACHE[cache_key] = metadata.copy()
    return metadata


def iter_frames(
    path: Path | str, start: int = 0, count: int | None = None,
    size: tuple[int, int] | None = None,
) -> Iterator[np.ndarray]:
    """Yield RGB uint8 frames; start/count are exact zero-based frame positions.

    ``size`` is (width, height). Decode/trim, rather than time seeking, preserves
    precise indices even around codec keyframes. Close the generator if stopping early.
    """
    if start < 0 or (count is not None and count < 0):
        raise ValueError("start and count must be nonnegative")
    if count == 0:
        return
    metadata = probe(path)
    width, height = size or (metadata["width"], metadata["height"])
    if width <= 0 or height <= 0:
        raise ValueError("frame dimensions must be positive")
    filters = [f"trim=start_frame={start}" + (f":end_frame={start + count}" if count is not None else "")]
    if size is not None:
        filters.append(f"scale={width}:{height}:flags=lanczos+accurate_rnd+full_chroma_int")
    args = [
        _tool("ffmpeg"), "-v", "error", "-nostdin", "-i", str(Path(path).expanduser().resolve()),
        "-map", f"0:{metadata.get('video_stream_index', 0)}", "-an", "-sn", "-dn", "-vf", ",".join(filters),
        "-sws_flags", "accurate_rnd+full_chroma_int",
        "-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1",
    ]
    if count is not None:
        args[-1:-1] = ["-frames:v", str(count)]
    process = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert process.stdout is not None and process.stderr is not None
    complete = False
    try:
        frame_bytes = width * height * 3
        while True:
            data = bytearray()
            while len(data) < frame_bytes:
                chunk = process.stdout.read(frame_bytes - len(data))
                if not chunk:
                    break
                data.extend(chunk)
            if not data:
                break
            if len(data) != frame_bytes:
                raise RuntimeError(f"FFmpeg returned a partial frame ({len(data)} / {frame_bytes} bytes)")
            yield np.frombuffer(data, dtype=np.uint8).reshape(height, width, 3)
        error = process.stderr.read().decode("utf-8", errors="replace").strip()
        status = process.wait()
        complete = True
        if status:
            raise RuntimeError(f"FFmpeg frame decode failed: {error}")
    finally:
        if not complete and process.poll() is None:
            process.terminate()
        process.stdout.close()
        process.stderr.close()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def read_frames(
    path: Path | str, start: int, count: int, size: tuple[int, int] | None = None,
) -> np.ndarray:
    """Read exactly count RGB frames, raising if the video ends prematurely."""
    if count <= 0:
        raise ValueError("count must be positive")
    frames = list(iter_frames(path, start, count, size))
    if len(frames) != count:
        raise ValueError(f"requested {count} frames at index {start}, received {len(frames)}")
    return np.stack(frames)


def read_frame(path: Path | str, index: int, size: tuple[int, int] | None = None) -> np.ndarray:
    return read_frames(path, index, 1, size)[0]


class VideoWriter:
    """Stream RGB arrays to a new H.264 video; audio is added separately."""

    def __init__(
        self, output: Path | str, width: int, height: int, fps: float | str,
        crf: int = 16, preset: str = "slow",
        color_tags: dict[str, str | None] | None = None,
    ) -> None:
        self.output = Path(output).expanduser().resolve()
        if self.output.exists():
            raise FileExistsError(f"refusing to overwrite existing output: {self.output}")
        if width <= 0 or height <= 0 or width % 2 or height % 2:
            raise ValueError("H.264 yuv420p output requires positive even width and height")
        if not 0 <= crf <= 51:
            raise ValueError("CRF must be between 0 and 51")
        if float(Fraction(str(fps))) <= 0:
            raise ValueError("fps must be positive")
        # Raw RGB is full-range. Make its YUV conversion agree with the stream's
        # matrix/range tags; tags alone would misdescribe FFmpeg's default matrix.
        tags = {"color_space": "bt709", "color_transfer": "bt709",
                "color_primaries": "bt709", "color_range": "tv"}
        tags.update({key: value for key, value in (color_tags or {}).items()
                     if key in tags and value not in (None, "unknown", "unspecified")})
        matrices = {"bt709", "bt470bg", "smpte170m", "smpte240m", "fcc"}
        if tags["color_space"] not in matrices:
            raise ValueError(f"unsupported SDR output color matrix: {tags['color_space']}")
        if tags["color_range"] not in ("tv", "pc"):
            raise ValueError("output color_range must be tv or pc")
        if tags["color_transfer"] in ("smpte2084", "arib-std-b67"):
            raise ValueError("HDR transfer functions require a separate high-bit-depth pipeline")
        self.width, self.height = width, height
        self.frames_written = 0
        self._closed = False
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.process = subprocess.Popen([
            _tool("ffmpeg"), "-v", "error", "-nostdin", "-n", "-f", "rawvideo",
            "-pix_fmt", "rgb24", "-video_size", f"{width}x{height}", "-framerate", str(fps),
            "-i", "pipe:0", "-an", "-vf",
            f"scale=in_range=pc:out_range={tags['color_range']}:out_color_matrix={tags['color_space']}:flags=accurate_rnd+full_chroma_int",
            "-c:v", "libx264", "-preset", preset,
            "-crf", str(crf), "-pix_fmt", "yuv420p",
            "-color_primaries", tags["color_primaries"], "-color_trc", tags["color_transfer"],
            "-colorspace", tags["color_space"], "-color_range", tags["color_range"],
            "-movflags", "+faststart", str(self.output),
        ], stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

    def write(self, frame: np.ndarray) -> None:
        if self._closed:
            raise RuntimeError("video writer is closed")
        if frame.shape != (self.height, self.width, 3) or frame.dtype != np.uint8:
            raise ValueError(f"expected uint8 RGB frame shaped {(self.height, self.width, 3)}, got {frame.dtype} {frame.shape}")
        assert self.process.stdin is not None
        try:
            self.process.stdin.write(np.ascontiguousarray(frame).tobytes())
        except BrokenPipeError as exc:
            assert self.process.stderr is not None
            message = self.process.stderr.read().decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"FFmpeg encode failed: {message}") from exc
        self.frames_written += 1

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        assert self.process.stdin is not None and self.process.stderr is not None
        try:
            self.process.stdin.close()
        except BrokenPipeError:
            pass
        error = self.process.stderr.read().decode("utf-8", errors="replace").strip()
        status = self.process.wait()
        self.process.stderr.close()
        if status:
            raise RuntimeError(f"FFmpeg encode failed: {error}")

    def __enter__(self) -> "VideoWriter":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if exc_type is None:
            self.close()
        else:
            self._closed = True
            if self.process.poll() is None:
                self.process.terminate()
            if self.process.stdin is not None:
                try:
                    self.process.stdin.close()
                except BrokenPipeError:
                    pass
            self.process.wait()
            if self.process.stderr is not None:
                self.process.stderr.close()


def mux_audio(video: Path | str, source: Path | str, output: Path | str) -> Path:
    """Copy rendered video and original audio into a new file, without truncation.

    Audio is stream-copied, retaining its original quality and timing. Incompatible
    audio/container combinations fail explicitly instead of silently transcoding.
    """
    video, source, output = (Path(path).expanduser().resolve() for path in (video, source, output))
    if output in (video, source):
        raise ValueError("mux output must differ from both inputs")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    _run([
        _tool("ffmpeg"), "-v", "error", "-nostdin", "-n", "-i", str(video), "-i", str(source),
        "-map", "0:v:0", "-map", "1:a?", "-map_metadata", "1", "-map_chapters", "1",
        "-c", "copy", "-movflags", "+faststart", str(output),
    ])
    return output
