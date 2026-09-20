from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from seamstress.framing_evidence import _infer_endpoint, recover_endpoint_edit


def observations(phase=0., affine_motion=False):
    rng = np.random.default_rng(281)
    left = rng.uniform([20, 20], [620, 340], (500, 2))
    # Local depth motion cannot be represented by an affine camera rate.
    velocity = np.column_stack((.65*np.sin(left[:, 1]/37), .45*np.cos(left[:, 0]/71)))
    stationary = (np.floor(left[:, 0]/80)+np.floor(left[:, 1]/60)).astype(int) % 2 == 0
    velocity[stationary] = 0
    if affine_motion:
        velocity = .003*(left-[320, 180])+[.2, -.1]
    matrix = np.array([[1.008, .001, -2.5], [0., 1.025, -4.5], [0., 0., 1.]])
    right = (left+phase*velocity-matrix[:2, 2])@np.linalg.inv(matrix[:2, :2]).T
    pre = np.array([velocity]*3)
    post = pre@np.linalg.inv(matrix[:2, :2]).T
    return left, right, pre, post, matrix


class FramingEvidenceTests(unittest.TestCase):
    def test_reprojected_repeated_endpoint_recovers_full_affine(self):
        left, right, pre, post, matrix = observations()
        result = _infer_endpoint(left, right, pre, post, matrix, (360, 640))
        self.assertTrue(result['accepted'], result)
        np.testing.assert_allclose(result['edit_matrix'], matrix, atol=1e-12)
        phases = [fold['phase'] for item in result['diagnostics']['intervals'] for fold in item['folds']]
        np.testing.assert_allclose(phases, 0, atol=1e-9)

    def test_ordinary_continuous_parallax_does_not_become_framing_repair(self):
        result = _infer_endpoint(*observations(phase=1.), (360, 640))
        self.assertFalse(result['accepted'], result)
        self.assertIn('does not identify a repeated endpoint', result['reason'])

    def test_continuous_affine_zoom_cannot_identify_endpoint_phase(self):
        result = _infer_endpoint(*observations(phase=1., affine_motion=True), (360, 640))
        self.assertFalse(result['accepted'], result)
        self.assertIn('cannot distinguish', result['reason'])

    def test_repeated_pure_affine_motion_is_also_ambiguous(self):
        result = _infer_endpoint(*observations(affine_motion=True), (360, 640))
        self.assertFalse(result['accepted'], result)

    def test_animation_hold_under_continuous_camera_pan_is_not_a_recrop(self):
        left, _, _, _, _ = observations()
        velocity = np.column_stack((np.full(len(left), 3.), np.where(left[:, 1] < 180, -3., 3.)))
        right = left+[3., 0.]
        matrix = np.array([[1., 0., -3.], [0., 1., 0.], [0., 0., 1.]])
        result = _infer_endpoint(left, right, np.array([velocity]*3), np.array([velocity]*3), matrix, (360, 640))
        self.assertFalse(result['accepted'], result)
        self.assertIn('stationary camera reference', result['reason'])

    def test_noncoherent_boundary_does_not_authorize_common_recrop(self):
        left, right, pre, post, matrix = observations()
        right[left[:, 0] < 250, 1] += 5
        result = _infer_endpoint(left, right, pre, post, matrix, (360, 640))
        self.assertFalse(result['accepted'], result)
        self.assertIn('disagree with', result['reason'])

    def test_one_side_phase_disagreement_rejects(self):
        left, right, pre, post, matrix = observations(phase=.2)
        post *= .2
        result = _infer_endpoint(left, right, pre, post, matrix, (360, 640))
        self.assertFalse(result['accepted'], result)

    def test_small_shared_patch_is_not_spatial_evidence(self):
        left, right, pre, post, matrix = observations()
        keep = left[:, 0] < 270
        result = _infer_endpoint(left[keep], right[keep], pre[:, keep], post[:, keep], matrix, (360, 640))
        self.assertFalse(result['accepted'], result)
        self.assertIn('spatial support', result['reason'])

    def test_identical_endpoint_with_no_recrop_does_not_trigger(self):
        left, _, pre, post, _ = observations()
        result = _infer_endpoint(left, left.copy(), pre, post, np.eye(3), (360, 640))
        self.assertFalse(result['accepted'], result)
        self.assertIn('no substantial framing reset', result['reason'])

    def test_cancellation_is_not_hidden_as_evidence_rejection(self):
        frame = np.zeros((360, 640, 3), np.uint8)
        frames = {i: frame for i in range(14)}
        fit = {'accepted': True, 'scene_consistent': True, 'matrix': np.eye(3).tolist(),
               'diagnostics': {'inlier_fraction': 1., 'coverage': 1.,
                               'reprojection_p90_analysis_pixels': 0., 'quality': {'gradient_ncc': 1.}}}
        with self.assertRaises(InterruptedError):
            recover_endpoint_edit(frames, 7, fit, cancelled=lambda: True)

    def test_numerical_failure_rejects_without_aborting_video_analysis(self):
        frame = np.zeros((360, 640, 3), np.uint8)
        frames = {i: frame for i in range(14)}
        fit = {'accepted': True, 'scene_consistent': True, 'matrix': np.eye(3).tolist(),
               'diagnostics': {'inlier_fraction': 1., 'coverage': 1.,
                               'reprojection_p90_analysis_pixels': 0., 'quality': {'gradient_ncc': 1.}}}
        with patch('seamstress.framing_evidence._matches', side_effect=cv2.error('fixture failure')):
            result = recover_endpoint_edit(frames, 7, fit)
        self.assertFalse(result['accepted'])
        self.assertEqual(result['diagnostic_error'], 'error')

    @unittest.skipUnless(shutil.which('ffmpeg'), 'FFmpeg is needed for an encoded image-track regression')
    def test_encoded_camera_pan_survives_an_independent_animation_hold(self):
        from seamstress.calibration import calibrate_pair, _camera_rate, _options
        from seamstress.media import VideoWriter, read_frames
        rng = np.random.default_rng(614)
        base = rng.integers(0, 256, (520, 800, 3), dtype=np.uint8)
        base = cv2.GaussianBlur(base, (3, 3), .65)
        for _ in range(300):
            center = tuple(rng.integers([2, 2], [798, 518]).tolist())
            color = tuple(rng.integers(0, 256, 3).tolist())
            cv2.circle(base, center, int(rng.integers(3, 13)), color, -1)
        y, x = np.mgrid[:360, :640].astype(np.float32)
        images = []
        for index in range(14):
            t = index-7
            phase = t if t < 0 else t-1
            world_x = x+80-3*t  # Camera advances even while the two planes hold.
            group = np.where(world_x < 400, -1., 1.)
            world_y = y+80-3*group*phase
            images.append(cv2.remap(base, world_x, world_y.astype(np.float32), cv2.INTER_LINEAR))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'camera-pan-animation-hold.mp4'
            with VideoWriter(path, 640, 360, '24', crf=14, preset='fast') as writer:
                for frame in images:
                    writer.write(frame)
            frames = dict(enumerate(read_frames(path, 0, len(images))))
        options = _options()
        fit = calibrate_pair(frames[6], frames[7], options)
        self.assertTrue(fit['accepted'], fit)
        _, ordinary_reliable, _ = _camera_rate(frames, list(range(7)), np.eye(3),
                                               np.array([319.5, 179.5]), options, None)
        self.assertFalse(ordinary_reliable)
        result = recover_endpoint_edit(frames, 7, fit, options=options)
        self.assertFalse(result['accepted'], result)
        self.assertIn('stationary camera reference', result['reason'])

    @unittest.skipUnless(shutil.which('ffmpeg'), 'FFmpeg is needed for an encoded image-track regression')
    def test_camera_oscillation_is_not_aliased_to_stationary_anchors(self):
        from seamstress.calibration import calibrate_pair, _camera_rate, _options
        from seamstress.media import VideoWriter, read_frames
        rng = np.random.default_rng(614)
        base = rng.integers(0, 256, (520, 800, 3), dtype=np.uint8)
        base = cv2.GaussianBlur(base, (3, 3), .65)
        for _ in range(300):
            center = tuple(rng.integers([2, 2], [798, 518]).tolist())
            color = tuple(rng.integers(0, 256, 3).tolist())
            cv2.circle(base, center, int(rng.integers(3, 13)), color, -1)
        y, x = np.mgrid[:360, :640].astype(np.float32)
        images = []
        for index in range(14):
            t = index-7
            phase = t if t < 0 else t-1
            world_x = x+80-3*(-1)**index
            bands = (world_x//40).astype(int) % 4
            group = np.where(bands == 1, -1., np.where(bands == 3, 1., 0.))
            world_y = y+80-3*group*phase
            images.append(cv2.remap(base, world_x, world_y.astype(np.float32), cv2.INTER_LINEAR))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'camera-oscillation-animation-hold.mp4'
            with VideoWriter(path, 640, 360, '24', crf=14, preset='fast') as writer:
                for frame in images:
                    writer.write(frame)
            frames = dict(enumerate(read_frames(path, 0, len(images))))
        options = _options()
        fit = calibrate_pair(frames[6], frames[7], options)
        self.assertTrue(fit['accepted'], fit)
        _, ordinary_reliable, _ = _camera_rate(frames, list(range(7)), np.eye(3),
                                               np.array([319.5, 179.5]), options, None)
        self.assertFalse(ordinary_reliable)
        result = recover_endpoint_edit(frames, 7, fit, options=options)
        self.assertFalse(result['accepted'], result)
        self.assertIn('stationary camera reference', result['reason'])
        self.assertEqual(result['diagnostics']['stationary_anchors']['checked_intervals'], 12)

    @unittest.skipUnless(shutil.which('ffmpeg'), 'FFmpeg is needed for an encoded image-track regression')
    def test_encoded_layered_motion_repeated_endpoint_is_recovered(self):
        from seamstress.calibration import calibrate_pair
        from seamstress.media import VideoWriter, read_frames
        rng = np.random.default_rng(193)
        canvas = np.full((360, 640, 3), 42, np.uint8)
        for _ in range(1700):
            center = tuple(rng.integers([4, 4], [636, 356]).tolist())
            color = tuple(rng.integers(60, 240, 3).tolist())
            cv2.circle(canvas, center, int(rng.integers(2, 6)), color, -1, cv2.LINE_AA)
        matrix = np.array([[1.008, 0., -2.5], [0., 1.025, -4.5], [0., 0., 1.]])
        inverse = np.linalg.inv(matrix)
        images = []
        for index in range(14):
            t = index-6 if index < 7 else index-7
            image = np.empty_like(canvas)
            for row, speed in enumerate([0., .9, 0., -.9, 0., .9]):
                moving = cv2.warpAffine(canvas, np.float32([[1, 0, t*speed], [0, 1, 0]]),
                                        (640, 360), borderMode=cv2.BORDER_REFLECT)
                image[row*60:(row+1)*60] = moving[row*60:(row+1)*60]
            if index >= 7:
                image = cv2.warpAffine(image, inverse[:2], (640, 360),
                                       flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
            images.append(image)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'moving-layers.mp4'
            with VideoWriter(path, 640, 360, '24', crf=14, preset='fast') as writer:
                for frame in images:
                    writer.write(frame)
            frames = dict(enumerate(read_frames(path, 0, len(images))))
        fit = calibrate_pair(frames[6], frames[7])
        result = recover_endpoint_edit(frames, 7, fit, upscale=np.diag([2., 2., 1.]))
        self.assertTrue(result['accepted'], result)
        actual = np.asarray(result['edit_matrix'])
        expected = np.diag([2., 2., 1.])@matrix@np.diag([.5, .5, 1.])
        points = np.array([[0., 0., 1.], [1279., 719., 1.], [639.5, 359.5, 1.]])
        self.assertLess(np.max(np.linalg.norm((points@(actual-expected).T)[:, :2], axis=1)), .6)
        limited = recover_endpoint_edit(frames, 7, fit, options={'max_scale_change': .001})
        self.assertFalse(limited['accepted'])
        self.assertIn('conservative global affine bounds', limited['reason'])


if __name__ == '__main__':
    unittest.main()
