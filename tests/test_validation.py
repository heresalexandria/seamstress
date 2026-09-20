from __future__ import annotations

import copy
import unittest

from seamstress.validation import validate_plan


class PlanValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.metadata = {
            "width": 320, "height": 180, "fps": 24.0, "fps_fraction": "24/1",
            "frame_count": 200, "duration": 200 / 24,
        }
        self.plan = {
            "schema_version": 1, "source": self.metadata.copy(), "source_sha256": "a" * 64,
            "config": {"flow_width": 160},
            "seams": [{"frame": 50, "before": 14, "after": 22},
                      {"frame": 100, "before": 14, "after": 22}],
        }

    def test_valid_plan_is_not_mutated(self) -> None:
        original = copy.deepcopy(self.plan)
        self.assertIsNone(validate_plan(self.plan, self.metadata))
        self.assertEqual(self.plan, original)

    def test_odd_analysis_width_is_valid(self) -> None:
        self.plan["config"]["flow_width"] = 161
        validate_plan(self.plan, self.metadata)

    def test_schema_and_fingerprint_are_required(self) -> None:
        for field, value in (("schema_version", 2), ("schema_version", True),
                             ("source_sha256", "not-a-fingerprint")):
            with self.subTest(field=field, value=value):
                plan = copy.deepcopy(self.plan)
                plan[field] = value
                with self.assertRaisesRegex(ValueError, field):
                    validate_plan(plan, self.metadata)

    def test_metadata_must_be_positive_finite_and_match(self) -> None:
        for field, value in (("width", 0), ("height", -1), ("frame_count", True),
                             ("fps", float("nan")), ("fps", 0), ("fps_fraction", "0/0"),
                             ("width", 640), ("frame_count", 201)):
            with self.subTest(field=field, value=value):
                plan = copy.deepcopy(self.plan)
                plan["source"][field] = value
                with self.assertRaisesRegex(ValueError, field):
                    validate_plan(plan, self.metadata)

    def test_config_rejects_nonfinite_and_out_of_bounds_values(self) -> None:
        cases = {
            "flow_width": [63, 321, 160.5, True],
            "max_displacement": [-0.1, 0.21, float("nan")],
            "color_spatial_sigma": [0, -1, 161, float("inf")],
            "local_color_strength": [-0.1, 1.1],
            "geometry_strength": [-0.1, 1.1, "1"],
            "color_strength": [float("nan"), True],
            "velocity_strength": [-1, 1.1],
            "max_crop_fraction": [-0.1, 0.21],
            "crop_fraction": [-0.1, 0.05],
            "texture_strength": [-0.1, 0.51, float("nan")],
            "texture_window": [1, 201, 2.5, True],
        }
        for field, values in cases.items():
            for value in values:
                with self.subTest(field=field, value=value):
                    plan = copy.deepcopy(self.plan)
                    plan["config"][field] = value
                    with self.assertRaisesRegex(ValueError, field):
                        validate_plan(plan, self.metadata)

    def test_unknown_config_fields_are_not_silently_ignored(self) -> None:
        self.plan["config"]["geometry_strenght"] = 0.5
        with self.assertRaisesRegex(ValueError, "unknown config"):
            validate_plan(self.plan, self.metadata)

    def test_seams_must_be_sorted_unique_integer_indices(self) -> None:
        for frames in ([100, 50], [50, 50], [50.5, 100], [True, 100], [0, 100], [50, 200]):
            with self.subTest(frames=frames):
                plan = copy.deepcopy(self.plan)
                plan["seams"] = [{"frame": frame} for frame in frames]
                with self.assertRaises(ValueError):
                    validate_plan(plan, self.metadata)

    def test_short_or_clipped_windows_are_rejected(self) -> None:
        for record in ({"frame": 50, "before": 4}, {"frame": 50, "after": 4},
                       {"frame": 4}, {"frame": 196}, {"frame": 50, "before": 5.5}):
            with self.subTest(record=record):
                plan = copy.deepcopy(self.plan)
                plan["seams"] = [record]
                with self.assertRaises(ValueError):
                    validate_plan(plan, self.metadata)

    def test_overlapping_enabled_windows_are_rejected(self) -> None:
        self.plan["seams"][1]["frame"] = 65
        with self.assertRaisesRegex(ValueError, "overlaps"):
            validate_plan(self.plan, self.metadata)

    def test_touching_windows_and_disabled_overlaps_are_valid(self) -> None:
        self.plan["seams"][1]["frame"] = 86  # First ends at 72; second starts at 72.
        validate_plan(self.plan, self.metadata)
        self.plan["seams"][1].update(frame=65, enabled=False)
        validate_plan(self.plan, self.metadata)

    def test_enabled_flags_cannot_be_strings(self) -> None:
        for field in ("enabled", "fill_startup_hold"):
            with self.subTest(field=field):
                plan = copy.deepcopy(self.plan)
                plan["seams"][0][field] = "false"
                with self.assertRaisesRegex(ValueError, field):
                    validate_plan(plan, self.metadata)


if __name__ == "__main__":
    unittest.main()
