import unittest
import numpy as np

from seamstress.retime import analyze_startup_hold, fill_startup_hold


def square_frame(x):
    frame = np.zeros((96, 160, 3), np.uint8)
    frame[28:64, x:x + 30] = (240, 210, 180)
    frame[35:42, x + 5:x + 13] = (50, 65, 80)
    return frame


class StartupHoldTests(unittest.TestCase):
    def test_on_twos_is_untouched(self):
        frames = [square_frame(10 + 4 * (i // 2)) for i in range(20)]
        result, report = fill_startup_hold(frames, 4)
        self.assertFalse(report['applied'])
        self.assertEqual(report['startup_held_intervals'], 1)
        for original, actual in zip(frames, result):
            np.testing.assert_array_equal(original, actual)

    def test_extra_hold_filled_without_timeline_change(self):
        positions = [12, 14, 16, 18, 20, 20, 20, 20,
                     24, 24, 28, 28, 32, 32, 36, 36, 40, 40]
        frames = [square_frame(x) for x in positions]
        result, report = fill_startup_hold(frames, 4)
        self.assertTrue(report['applied'])
        self.assertEqual(len(result), len(frames))
        self.assertEqual(report['replaced_offsets'], [2, 3])
        for i in set(range(len(frames))) - {6, 7}:
            np.testing.assert_array_equal(frames[i], result[i])
        centers = [np.mean(np.where(frame[..., 0] > 120)[1])
                   for frame in [frames[5], result[6], result[7], frames[8]]]
        self.assertTrue(all(a < b for a, b in zip(centers, centers[1:])), centers)
        self.assertEqual(report['source_time_shift'], 0.)

    def test_no_future_motion_is_left_alone(self):
        frames = [square_frame(20) for _ in range(18)]
        result, report = fill_startup_hold(frames, 4)
        self.assertFalse(report['applied'])
        for original, actual in zip(frames, result):
            np.testing.assert_array_equal(original, actual)

    def test_later_long_hold_cadence_is_preserved(self):
        positions = [12, 14, 20, 20, 20, 24, 24, 24, 28, 28, 28, 32]
        frames = [square_frame(x) for x in positions]
        report = analyze_startup_hold(frames, 2)
        self.assertEqual(report['normal_held_intervals'], 2)
        self.assertFalse(report['eligible'])


if __name__ == '__main__':
    unittest.main()
