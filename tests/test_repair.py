"""Behavioral checks for repair geometry, grading, temporal support, and safety.

Run with ``python -m unittest discover -s tests -v``. Synthetic flow provides
known correspondences so remap arithmetic is tested independently of estimation.
"""

from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from seamstress import repair as repair_module


def texture() -> np.ndarray:
    rng = np.random.default_rng(218)
    image = rng.integers(35, 185, (80, 128, 3), dtype=np.uint8)
    return cv2.GaussianBlur(image, (3, 3), 0.5)


def translated(image: np.ndarray, dx: float) -> np.ndarray:
    return cv2.warpAffine(
        image, np.float32([[1, 0, dx], [0, 1, 0]]),
        (image.shape[1], image.shape[0]), borderMode=cv2.BORDER_REFLECT_101,
    )


class RepairGeometryTests(unittest.TestCase):
    def test_identical_sequence_remains_pixel_identical(self) -> None:
        image = texture()
        frames = [image.copy() for _ in range(10)]
        result, report = repair_module.corrected_window(frames, 5, {})
        self.assertEqual(len(result), len(frames))
        for frame in result:
            np.testing.assert_array_equal(frame, image)
        self.assertEqual(report["required_crop_fraction"], 0)

    def test_excess_translation_is_removed_but_animation_motion_remains(self) -> None:
        image = texture()
        positions = [-4, -2, 0, 10, 12, 14]
        frames = [translated(image, x) for x in positions]
        position_by_id = {id(frame): position for frame, position in zip(frames, positions)}

        def known_flow(a: np.ndarray, b: np.ndarray) -> np.ndarray:
            result = np.zeros((*a.shape[:2], 2), np.float32)
            result[..., 0] = position_by_id[id(b)] - position_by_id[id(a)]
            return result

        with patch.object(repair_module, "flow", side_effect=known_flow):
            result, report = repair_module.corrected_window(
                frames, 3, {"max_displacement": 0.2, "color_strength": 0},
            )
        # The edit jumps ten pixels while the animation advances two. Splitting
        # the excess eight pixels puts the seam anchors at +4 and +6 pixels.
        interior = np.s_[8:-8, 20:-20]
        np.testing.assert_array_equal(result[2][interior], translated(image, 4)[interior])
        np.testing.assert_array_equal(result[3][interior], translated(image, 6)[interior])
        np.testing.assert_array_equal(result[0], frames[0])
        np.testing.assert_array_equal(result[-1], frames[-1])
        self.assertAlmostEqual(report["expected_motion_p50"], 2.0, places=5)
        self.assertGreaterEqual(report["required_crop_fraction"], 4 / 128)
        self.assertLessEqual(report["required_crop_fraction"], 4 / 128 + 2 / min(image.shape[:2]))

    def test_remap_direction_is_gather_not_forward_scatter(self) -> None:
        image = texture()
        displacement = np.zeros((*image.shape[:2], 2), np.float32)
        displacement[..., 0] = -3
        result = repair_module.sample(image, displacement)
        np.testing.assert_array_equal(result[:, 3:], image[:, :-3])

    def test_displacement_vectors_transform_with_scene_scale(self) -> None:
        y, x = np.mgrid[:80, :128]
        displacement = np.full((80, 128, 2), 6, np.float32)
        motion = np.stack((0.2 * x, 0.2 * y), axis=-1).astype(np.float32)
        pulled = repair_module.transport_vector(displacement, motion)
        # Current-to-anchor magnification is 1.2, so six anchor pixels represent
        # five current-frame pixels. Ignore the derivative's reflected edges.
        np.testing.assert_allclose(pulled[10:-10, 10:-10], 5, atol=2e-5)

    def test_velocity_tangents_match_two_to_six_pixel_motion(self) -> None:
        y, x = np.mgrid[:80, :256]
        mono = np.rint(20 + 200 * np.exp(-((x - 96) ** 2 + (y - 40) ** 2) / 100)).astype(np.uint8)
        image = np.repeat(mono[..., None], 3, axis=2)
        positions = list(range(-20, 1, 2)) + list(range(10, 71, 6))
        frames = [translated(image, position) for position in positions]
        positions_by_id = {id(frame): position for frame, position in zip(frames, positions)}

        def known_flow(a: np.ndarray, b: np.ndarray) -> np.ndarray:
            field = np.zeros((*a.shape[:2], 2), np.float32)
            field[..., 0] = positions_by_id[id(b)] - positions_by_id[id(a)]
            return field

        with patch.object(repair_module, "flow", side_effect=known_flow):
            corrected, _ = repair_module.corrected_window(
                frames, 11, {"max_displacement": 0.2, "color_strength": 0, "velocity_strength": 1},
            )

        def center(frame: np.ndarray) -> float:
            mass = np.maximum(frame[..., 0].astype(float) - 20, 0)
            return float(np.sum(mass * x) / np.sum(mass))

        centers = [center(frame) for frame in corrected]
        for before, after in ((9, 10), (10, 11), (11, 12)):
            self.assertAlmostEqual(centers[after] - centers[before], 4, delta=0.1)
        np.testing.assert_array_equal(corrected[0], frames[0])
        np.testing.assert_array_equal(corrected[-1], frames[-1])

    def test_smooth_fade_has_stationary_endpoints(self) -> None:
        fade = repair_module.smoothstep
        self.assertEqual(float(fade(0)), 0)
        self.assertEqual(float(fade(1)), 1)
        self.assertEqual(float(fade(-1)), 0)
        self.assertEqual(float(fade(2)), 1)
        epsilon = 1e-4
        self.assertLess(float(fade(epsilon) / epsilon), 1e-6)
        self.assertLess(float((1 - fade(1 - epsilon)) / epsilon), 1e-6)

    def test_symmetric_color_shift_meets_at_common_grade(self) -> None:
        a = texture()
        b = a + np.array([12, 8, 4], np.uint8)
        frames = [a.copy() for _ in range(4)] + [b.copy() for _ in range(4)]
        zero = np.zeros((*a.shape[:2], 2), np.float32)
        with patch.object(repair_module, "flow", return_value=zero):
            result, _ = repair_module.corrected_window(frames, 4, {})
        expected = a.astype(np.int16) + np.array([6, 4, 2])
        self.assertLessEqual(np.max(abs(result[3].astype(np.int16) - expected)), 1)
        self.assertLessEqual(np.max(abs(result[4].astype(np.int16) - expected)), 1)
        np.testing.assert_array_equal(result[0], a)
        np.testing.assert_array_equal(result[-1], b)

    def test_crop_excludes_overscan_and_preserves_output_dimensions(self) -> None:
        frame = np.full((80, 128, 3), 150, np.uint8)
        frame[:8] = 0
        frame[-8:] = 0
        frame[:, :13] = 0
        frame[:, -13:] = 0
        result = repair_module.crop_frame(frame, 0.1)
        self.assertEqual(result.shape, frame.shape)
        np.testing.assert_array_equal(result, np.full_like(frame, 150))
        np.testing.assert_array_equal(repair_module.crop_frame(frame, 0), frame)

    def test_folded_map_is_regularized_to_positive_jacobian(self) -> None:
        y, x = np.mgrid[:80, :128]
        mapping = np.zeros((80, 128, 2), np.float32)
        mapping[..., 0] = 14 * np.sin(x / 3)
        corrected, reported_minimum, softened = repair_module.safe_mapping(mapping)
        dy_dx = np.gradient(corrected[..., 1], axis=1)
        dx_dy = np.gradient(corrected[..., 0], axis=0)
        dx_dx = np.gradient(corrected[..., 0], axis=1)
        dy_dy = np.gradient(corrected[..., 1], axis=0)
        determinant = (1 + dx_dx) * (1 + dy_dy) - dx_dy * dy_dx
        self.assertTrue(softened)
        self.assertTrue(np.isfinite(corrected).all())
        self.assertGreaterEqual(float(determinant.min()), 0.25)
        self.assertAlmostEqual(reported_minimum, float(determinant.min()))

    def test_invalid_crop_fractions_fail_before_resizing(self) -> None:
        frame = texture()
        for fraction in (-0.01, 0.5, 1, float("nan"), float("inf")):
            with self.subTest(fraction=fraction):
                with self.assertRaisesRegex(ValueError, "Crop fraction"):
                    repair_module.crop_frame(frame, fraction)


class RepairSafetyTests(unittest.TestCase):
    def test_source_cannot_be_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source.mp4"
            source.write_bytes(b"original bytes")
            with self.assertRaisesRegex(ValueError, "distinct"):
                repair_module.repair(source, Path(tmp) / "missing-plan.json", source)
            self.assertEqual(source.read_bytes(), b"original bytes")

    def test_existing_output_is_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "output.mp4"
            output.write_bytes(b"existing output")
            with self.assertRaisesRegex(ValueError, "new path"):
                repair_module.repair(root / "source.mp4", root / "missing-plan.json", output)
            self.assertEqual(output.read_bytes(), b"existing output")

    def test_plan_for_changed_source_is_rejected_before_render(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.mp4"
            source.write_bytes(b"source changed after analysis")
            plan = root / "plan.json"
            metadata = {"width": 128, "height": 80, "fps": 24, "frame_count": 20}
            plan.write_text(json.dumps({
                "schema_version": 1, "source_sha256": "0" * 64, "source": metadata,
                "config": {"flow_width": 128}, "seams": [],
            }))
            with patch("seamstress.media.probe", return_value=metadata):
                with self.assertRaisesRegex(ValueError, "fingerprint differs"):
                    repair_module.repair(source, plan, root / "output.mp4")
            self.assertFalse((root / "output.mp4").exists())


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg is required")
class RenderIntegrityTests(unittest.TestCase):
    def test_sdr_color_roundtrip_and_fractional_frame_rate(self) -> None:
        from seamstress.media import VideoWriter, probe, read_frame

        colors = np.array([
            [230, 25, 30], [25, 200, 45], [20, 50, 225], [220, 180, 35],
            [35, 180, 200], [180, 50, 180], [12, 12, 12], [230, 230, 230],
        ], np.uint8)
        image = np.repeat(np.repeat(colors[None, :, :], 64, axis=0), 32, axis=1)
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "color-roundtrip.mp4"
            with VideoWriter(output, 256, 64, "24000/1001", crf=0, preset="ultrafast") as writer:
                for _ in range(5):
                    writer.write(image)
            metadata = probe(output)
            self.assertEqual(metadata["fps_fraction"], "24000/1001")
            self.assertEqual(metadata["frame_count"], 5)
            self.assertEqual(metadata["color_space"], "bt709")
            self.assertEqual(metadata["color_transfer"], "bt709")
            self.assertEqual(metadata["color_primaries"], "bt709")
            self.assertEqual(metadata["color_range"], "tv")
            decoded = read_frame(output, 0)
            # Avoid chroma-subsampling boundaries when checking matrix/range
            # correctness. Limited-range quantization allows a small difference.
            centers = np.stack([
                decoded[16:48, 32 * i + 8:32 * i + 24].mean(axis=(0, 1))
                for i in range(len(colors))
            ])
            self.assertLessEqual(float(abs(centers - colors).max()), 2)


if __name__ == "__main__":
    unittest.main()
