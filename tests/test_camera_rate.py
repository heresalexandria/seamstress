import unittest

import cv2
import numpy as np

from seamstress.camera_rate import recover_cadence_rate
from seamstress.design import _matrix


CENTER = np.array([639.5, 359.5])


def synthetic_motion(rates, side='pre', upscale=None):
    cut = 100
    start = cut-13 if side == 'pre' else cut
    indices = list(range(start, start+13, 2))
    trajectory = {indices[0]: np.eye(3)}
    for a, b, rate in zip(indices, indices[1:], rates):
        trajectory[b] = _matrix(2*np.asarray(rate), CENTER)@trajectory[a]
    frames = {index: np.array(index) for index in indices}
    upscale = np.eye(3) if upscale is None else upscale

    def pair_fit(a, b):
        backward = trajectory[int(a)]@np.linalg.inv(trajectory[int(b)])
        return {'accepted': True, 'confidence': .98,
                'matrix': (np.linalg.inv(upscale)@backward@upscale).tolist()}

    return frames, cut, pair_fit, upscale


def recover(rates, side='pre', **kwargs):
    frames, cut, pair_fit, upscale = synthetic_motion(rates, side)
    return recover_cadence_rate(frames, cut, side, pair_fit=pair_fit,
                                upscale=upscale, center=CENTER, **kwargs)


class CameraRateTests(unittest.TestCase):
    def test_repeated_cadence_recovers_the_trajectory_in_both_phases_and_sides(self):
        for side in ('pre', 'post'):
            for phase in (-1, 1):
                rates = [[0., 0., 3+phase*(-1)**i*2, -1.] for i in range(6)]
                result = recover(rates, side)
                self.assertTrue(result['accepted'], result)
                np.testing.assert_allclose(result['rate'], [0., 0., 3., -1.], atol=1e-10)
                # Three final interval samples alias toward one drawing phase.
                self.assertAlmostEqual(abs(np.median(np.array(rates)[-3:, 2])-result['rate'][2]), 2.)
                self.assertEqual(len(result['diagnostics']['source_indices']), 7)
                self.assertEqual(len(result['diagnostics']['four_frame_composition_errors_native_pixels']), 5)
                self.assertEqual(result['diagnostics']['native_rate'], result['rate'])

    def test_uniform_zoom_cadence_and_analysis_scaling_use_native_units(self):
        rates = [[-.006+(-1)**i*.003, .0002, 0., 0.] for i in range(6)]
        frames, cut, pair_fit, upscale = synthetic_motion(rates, upscale=np.diag([2., 2., 1.]))
        result = recover_cadence_rate(frames, cut, 'pre', pair_fit=pair_fit,
                                      upscale=upscale, center=CENTER)
        self.assertTrue(result['accepted'], result)
        np.testing.assert_allclose(result['rate'], [-.006, .0002, 0., 0.], atol=1e-10)

    def test_steady_motion_and_smooth_acceleration_do_not_change_existing_rate(self):
        cases = [np.ones(6)*3, np.linspace(1, 6, 6), np.arange(6.)**2,
                 [0., .1, .2, .3, .4, .5]]
        for speeds in cases:
            with self.subTest(speeds=speeds):
                result = recover([[0., 0., speed, 0.] for speed in speeds])
                self.assertFalse(result['accepted'], result)
                self.assertNotIn('rate', result)

    def test_smooth_acceleration_can_coexist_with_proven_cadence(self):
        rates = [[0., 0., 3+.15*(i-2.5)+2*(-1)**i, 0.] for i in range(6)]
        result = recover(rates)
        self.assertTrue(result['accepted'], result)
        np.testing.assert_allclose(result['rate'], [0., 0., 3., 0.], atol=1e-10)

    def test_strong_acceleration_with_exact_cadence_keeps_the_shorter_rate(self):
        for side in ('pre', 'post'):
            rates = [[0., 0., 3+.5*(i-2.5)+1.6*(-1)**i, 0.] for i in range(6)]
            result = recover(rates, side)
            self.assertFalse(result['accepted'])
            self.assertIn('acceleration', result['reason'])

    def test_nonperiodic_motion_phase_break_and_single_outlier_are_rejected(self):
        for speeds in ([1., 1., 1., 9., 1., 1.], [5., 1., 5., 1., 1., 5.],
                       [2., 8., -1., 3., 4., 2.], [3., 4., 5., 4., 3., 2.]):
            with self.subTest(speeds=speeds):
                self.assertFalse(recover([[0., 0., speed, 0.] for speed in speeds])['accepted'])

    def test_inconsequential_cadence_does_not_unlock_recovery(self):
        result = recover([[0., 0., 3+.2*(-1)**i, 0.] for i in range(6)])
        self.assertFalse(result['accepted'])

    def test_every_short_and_long_interval_needs_reliable_geometry(self):
        rates = [[0., 0., 3+2*(-1)**i, 0.] for i in range(6)]
        frames, cut, original, upscale = synthetic_motion(rates)
        for failure in ('short', 'long', 'confidence'):
            def pair_fit(a, b):
                fit = original(a, b)
                if int(a) == cut-13:
                    if failure == 'confidence':
                        fit['confidence'] = .7
                    elif int(b)-int(a) == (2 if failure == 'short' else 4):
                        fit['accepted'] = False
                return fit
            with self.subTest(failure=failure):
                result = recover_cadence_rate(frames, cut, 'pre', pair_fit=pair_fit,
                                              upscale=upscale, center=CENTER)
                self.assertFalse(result['accepted'])

    def test_independent_four_frame_motion_rejects_oscillating_fitting_errors(self):
        rates = [[0., 0., 3+2*(-1)**i, 0.] for i in range(6)]
        frames, cut, original, upscale = synthetic_motion(rates)
        def pair_fit(a, b):
            fit = original(a, b)
            if int(b)-int(a) == 4:
                fit['matrix'][0][2] += 4
            return fit
        result = recover_cadence_rate(frames, cut, 'pre', pair_fit=pair_fit,
                                      upscale=upscale, center=CENTER)
        self.assertFalse(result['accepted'])
        self.assertIn('disagrees', result['reason'])

    def test_missing_source_handles_and_cancellation_fail_safely(self):
        rates = [[0., 0., 3+2*(-1)**i, 0.] for i in range(6)]
        frames, cut, pair_fit, upscale = synthetic_motion(rates)
        del frames[cut-13]
        result = recover_cadence_rate(frames, cut, 'pre', pair_fit=pair_fit,
                                      upscale=upscale, center=CENTER)
        self.assertFalse(result['accepted'])
        with self.assertRaises(InterruptedError):
            recover(rates, cancelled=lambda: True)

    def test_estimation_errors_and_singular_matrices_fail_closed(self):
        rates = [[0., 0., 3+2*(-1)**i, 0.] for i in range(6)]
        frames, cut, _, upscale = synthetic_motion(rates)
        def failure(a, b):
            raise cv2.error('No usable camera model')
        for pair_fit in (failure, lambda a, b: {'accepted': True, 'confidence': 1., 'matrix': np.zeros((3, 3))}):
            result = recover_cadence_rate(frames, cut, 'pre', pair_fit=pair_fit,
                                          upscale=upscale, center=CENTER)
            self.assertFalse(result['accepted'])

    def test_real_feature_registration_of_a_cadenced_camera(self):
        from seamstress.calibration import calibrate_pair
        rng = np.random.default_rng(92)
        image = cv2.GaussianBlur(rng.integers(0, 256, (360, 640, 3), dtype=np.uint8), (5, 5), .8)
        for _ in range(80):
            x, y = rng.integers([25, 25], [615, 335])
            cv2.circle(image, (int(x), int(y)), 5, (240, 180, 100), 2)
        center = np.array([319.5, 179.5])
        indices = list(range(87, 100, 2))
        frames = {87: image}
        transform = np.eye(3)
        for i, index in enumerate(indices[1:]):
            transform = _matrix(np.array([0., 0., 2*(1.5+.9*(-1)**i), 0.]), center)@transform
            frames[index] = cv2.warpAffine(image, transform[:2], (640, 360), borderMode=cv2.BORDER_REFLECT)
        result = recover_cadence_rate(frames, 100, 'pre', pair_fit=calibrate_pair,
                                      upscale=np.eye(3), center=center)
        self.assertTrue(result['accepted'], result)
        np.testing.assert_allclose(result['rate'][2:], [1.5, 0.], atol=.03)


if __name__ == '__main__':
    unittest.main()
