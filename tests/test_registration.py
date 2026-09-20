"""Synthetic evidence checks for coordinate direction and estimator honesty."""

import unittest

import cv2
import numpy as np

from seamstress.registration import dense_correspondence, register_pair


def cartoon(width=640, height=400):
    rng = np.random.default_rng(42)
    image = np.full((height, width, 3), [100, 135, 160], dtype=np.uint8)
    for _ in range(110):
        center = (int(rng.integers(15, width - 15)), int(rng.integers(15, height - 15)))
        color = tuple(int(c) for c in rng.integers(35, 205, 3))
        radius = int(rng.integers(5, 30))
        cv2.circle(image, center, radius, color, -1, lineType=cv2.LINE_AA)
        cv2.circle(image, center, radius, (30, 35, 40), 2, lineType=cv2.LINE_AA)
    for j in range(9):
        cv2.putText(image, str(j), (35 + 65 * j, 200 + int(70 * np.sin(j))),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 175, 160), 2)
    return image


class RegistrationTests(unittest.TestCase):
    def test_source_to_reference_direction_and_resize(self):
        left = cartoon(1000, 650)
        expected = np.eye(3)
        expected[:2] = cv2.getRotationMatrix2D((500, 325), 0.8, 1.028)
        expected[:2, 2] += [7.0, -4.0]
        right = cv2.warpAffine(left, np.linalg.inv(expected)[:2], (1000, 650),
                               borderMode=cv2.BORDER_REFLECT101)
        result = register_pair(left, right, max_width=640)
        actual = np.asarray(result["matrix"])
        corners = np.float32([[100, 100], [900, 100], [100, 550], [900, 550]])
        actual_points = corners @ actual[:2, :2].T + actual[:2, 2]
        expected_points = corners @ expected[:2, :2].T + expected[:2, 2]
        self.assertLess(float(np.max(np.linalg.norm(actual_points - expected_points, axis=1))), 1.2)
        self.assertTrue(result["reliable"], result["diagnostics"])
        self.assertLess(result["metrics"]["after"]["corrected_mae"],
                        result["metrics"]["before"]["corrected_mae"] * 0.4)

    def test_colour_transform_maps_right_back_to_left(self):
        left = cartoon()
        applied_gain = np.array([1.08, 0.91, 1.04])
        applied_bias = np.array([5.0, 8.0, -4.0])
        right = np.rint(left * applied_gain + applied_bias).clip(0, 255).astype(np.uint8)
        result = register_pair(left, right)
        np.testing.assert_allclose(result["color"]["gain"], 1 / applied_gain, atol=0.015)
        np.testing.assert_allclose(result["color"]["bias"], -applied_bias / applied_gain, atol=1.3)
        self.assertLess(result["metrics"]["after"]["corrected_mae"], 0.5)

    def test_solid_background_does_not_claim_camera_evidence(self):
        left = np.full((240, 400, 3), [80, 100, 120], np.uint8)
        right = np.full_like(left, [87, 106, 131])
        result = register_pair(left, right)
        self.assertFalse(result["reliable"])
        self.assertLess(result["confidence"], 0.1)
        np.testing.assert_allclose(result["matrix"], np.eye(3))
        np.testing.assert_allclose(result["color"]["bias"], [-7, -6, -11])

    def test_bidirectional_flow_direction_and_occlusion_mask(self):
        left = cartoon(480, 320)
        right = cv2.warpAffine(left, np.float32([[1, 0, 5], [0, 1, -3]]),
                               (480, 320), borderMode=cv2.BORDER_REFLECT101)
        result = dense_correspondence(left, right, max_width=320)
        middle = result["forward"][30:-30, 30:-30]
        np.testing.assert_allclose(np.median(middle, axis=(0, 1)), [5, -3], atol=0.45)
        self.assertGreater(result["valid_fraction"], 0.80)
        self.assertEqual(result["valid"].shape, left.shape[:2])


if __name__ == "__main__":
    unittest.main()
