"""Validate editable repair plans before allocating frames or writing output."""

from __future__ import annotations

import math
import re
from fractions import Fraction
from typing import Any


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return value


def _integer(value: Any, name: str, minimum: int, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if value < minimum or (maximum is not None and value > maximum):
        upper = f" and at most {maximum}" if maximum is not None else ""
        raise ValueError(f"{name} must be at least {minimum}{upper}")
    return value


def _number(
    value: Any, name: str, minimum: float = 0, maximum: float | None = None,
    *, exclusive_minimum: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    if value < minimum or (exclusive_minimum and value == minimum) or (maximum is not None and value > maximum):
        lower = "greater than" if exclusive_minimum else "at least"
        upper = f" and at most {maximum}" if maximum is not None else ""
        raise ValueError(f"{name} must be {lower} {minimum}{upper}")
    return float(value)


def _metadata(value: Any, name: str) -> dict[str, Any]:
    metadata = _mapping(value, name)
    for field in ("width", "height", "frame_count"):
        _integer(metadata.get(field), f"{name}.{field}", 1)
    fps = _number(metadata.get("fps"), f"{name}.fps", exclusive_minimum=True)
    if "duration" in metadata:
        _number(metadata["duration"], f"{name}.duration", exclusive_minimum=True)
    if "fps_fraction" in metadata:
        try:
            rate = float(Fraction(metadata["fps_fraction"]))
        except (ValueError, TypeError, ZeroDivisionError, OverflowError) as exc:
            raise ValueError(f"{name}.fps_fraction must be a positive rational frame rate") from exc
        if rate <= 0 or not math.isclose(rate, fps, rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError(f"{name}.fps_fraction must agree with {name}.fps")
    return metadata


def validate_plan(plan: Any, metadata: Any) -> None:
    """Raise ValueError for an unsafe/incompatible v1 plan; never mutate it.

    ``metadata`` is a fresh ``media.probe`` result. SHA-256 format is checked
    here; the renderer separately computes and compares the actual fingerprint.
    Windows are half-open ``[frame - before, frame + after)`` after clipping.
    At least five usable frames are required on each side of every seam.
    """
    plan = _mapping(plan, "plan")
    version = _integer(plan.get("schema_version"), "schema_version", 1)
    if version != 1:
        raise ValueError(f"unsupported schema_version {version}; expected 1")
    source = _metadata(plan.get("source"), "source")
    metadata = _metadata(metadata, "metadata")
    for field in ("width", "height", "frame_count"):
        if source[field] != metadata[field]:
            raise ValueError(f"source.{field} differs from the input video; analyze again")
    if not math.isclose(source["fps"], metadata["fps"], rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError("source.fps differs from the input video; analyze again")
    fingerprint = plan.get("source_sha256")
    if not isinstance(fingerprint, str) or re.fullmatch(r"[0-9a-f]{64}", fingerprint) is None:
        raise ValueError("source_sha256 must be a 64-character lowercase SHA-256 digest")

    config = _mapping(plan.get("config"), "config")
    known = {
        "flow_width", "max_displacement", "color_spatial_sigma", "local_color_strength",
        "geometry_strength", "color_strength", "crop_fraction", "max_crop_fraction",
        "velocity_strength",
        "texture_strength", "texture_window",
    }
    unknown = set(config) - known
    if unknown:
        raise ValueError(f"unknown config field(s): {', '.join(sorted(map(str, unknown)))}")
    flow_width = _integer(
        config.get("flow_width", min(640, source["width"])),
        "config.flow_width", 64, source["width"],
    )
    # DIS works at arbitrary analysis widths. The full-resolution H.264 output
    # retains the source geometry and has its own even-dimension constraint.
    _number(config.get("max_displacement", 0.045), "config.max_displacement", 0, 0.2)
    _number(
        config.get("color_spatial_sigma", 24), "config.color_spatial_sigma",
        0, flow_width, exclusive_minimum=True,
    )
    for field, default in (
        ("local_color_strength", 0.8), ("geometry_strength", 1),
        ("color_strength", 1), ("velocity_strength", 1),
    ):
        _number(config.get(field, default), f"config.{field}", 0, 1)
    maximum_crop = _number(config.get("max_crop_fraction", 0.04), "config.max_crop_fraction", 0, 0.2)
    _number(config.get("crop_fraction", 0), "config.crop_fraction", 0, maximum_crop)
    _number(config.get("texture_strength", 0), "config.texture_strength", 0, 0.5)
    _integer(config.get("texture_window", 10), "config.texture_window", 2, source["frame_count"])

    seams = plan.get("seams")
    if not isinstance(seams, list):
        raise ValueError("seams must be a list")
    frame_count = source["frame_count"]
    previous_frame = 0
    previous_enabled_end = 0
    for index, record in enumerate(seams):
        name = f"seams[{index}]"
        record = _mapping(record, name)
        frame = _integer(record.get("frame"), f"{name}.frame", 1, frame_count - 1)
        if frame <= previous_frame:
            raise ValueError("seams must have unique frame indices in increasing order")
        previous_frame = frame
        before = _integer(record.get("before", 12), f"{name}.before", 5, frame_count)
        after = _integer(record.get("after", 18), f"{name}.after", 5, frame_count)
        for field in ("enabled", "fill_startup_hold"):
            if field in record and not isinstance(record[field], bool):
                raise ValueError(f"{name}.{field} must be a boolean")
        if "time" in record:
            _number(record["time"], f"{name}.time")
        start, end = max(0, frame - before), min(frame_count, frame + after)
        if frame - start < 5 or end - frame < 5:
            raise ValueError(f"{name} requires at least five available frames on each side")
        if record.get("enabled", True):
            if start < previous_enabled_end:
                raise ValueError(f"{name} repair window overlaps the preceding enabled window")
            previous_enabled_end = end
