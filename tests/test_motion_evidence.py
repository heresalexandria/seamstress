import math
from pathlib import Path
import shutil
import tempfile

import cv2
import numpy as np
import unittest

from seamstress.motion_evidence import (
    _infer_edit, _matrix, _motion_models, _power, recover_partial_edit,
)
from seamstress.calibration import calibrate_pair
from seamstress.media import VideoWriter, read_frames


CENTER = np.array([319.5, 179.5])


def translation(x, y):
    return _matrix(np.array([0., 0., x, y]), CENTER)


def model(matrix, fraction=.45, coverage=.5):
    return {'matrix': matrix.tolist(), 'fraction': fraction,
            'coverage': coverage, 'span': [.8, .4]}


def evidence(angle=0., normal_motion=1.3, edit_strength=.025):
    direction = np.array([math.cos(angle), math.sin(angle)])
    normal = np.array([-direction[1], direction[0]])
    common = translation(*(normal*normal_motion))
    post = translation(*(normal*.7))
    groups = [[model(_power(translation(*(normal*normal_motion+direction*speed)), 2))
               for speed in (-2., 2.)] for _ in range(3)]
    edit = np.eye(3)
    edit[:2, :2] += edit_strength*np.outer(normal, normal)
    edit[:2, 2] = CENTER-edit[:2, :2]@CENTER
    registration = np.linalg.inv(common)@edit
    cross = [(np.linalg.matrix_power(np.linalg.inv(common), a)@edit@
              np.linalg.matrix_power(np.linalg.inv(post), b), a, b)
             for a, b in ((3, 0), (2, 1), (1, 2))]
    return direction, edit, registration, groups, cross, post


class MotionEvidenceTests(unittest.TestCase):
    def test_partial_edit_preserves_common_camera_motion_for_arbitrary_axes(self):
        for angle in (0., .37, math.pi/2, 2.4):
            direction, expected, registration, groups, cross, post = evidence(angle)
            result = _infer_edit(registration, groups, cross, post, (360, 640))
            assert result['accepted'], result
            actual = np.asarray(result['edit_matrix'])
            np.testing.assert_allclose(actual, expected, atol=1e-10)
            # No displacement, including through neutral-return matrix powers, is
            # introduced along the direction that the evidence cannot resolve.
            for exponent in (-.5, .2, .5, 1.):
                np.testing.assert_allclose(direction@(_power(actual, exponent)-np.eye(3))[:2], 0, atol=1e-10)

    def test_ordinary_shared_motion_is_not_misidentified_as_an_edit(self):
        _, _, registration, groups, cross, post = evidence(normal_motion=4., edit_strength=0.)
        result = _infer_edit(registration, groups, cross, post, (360, 640))
        assert not result['accepted']
        assert 'stable above motion uncertainty' in result['reason']

    def test_unsupported_direction_of_edit_is_left_uncorrected(self):
        _, expected, registration, groups, cross, post = evidence()
        registration = translation(8, 0)@registration
        cross = [(translation(8, 0)@matrix, a, b) for matrix, a, b in cross]
        result = _infer_edit(registration, groups, cross, post, (360, 640))
        assert result['accepted'], result
        np.testing.assert_allclose(result['edit_matrix'], expected, atol=1e-10)

    def test_two_dimensional_motion_disagreement_rejects_recovery(self):
        _, _, registration, _, cross, post = evidence()
        groups = [[model(translation(4, 0), .3), model(translation(-4, 0), .3),
                   model(translation(0, 8), .3)] for _ in range(3)]
        result = _infer_edit(registration, groups, cross, post, (360, 640))
        assert not result['accepted']
        assert 'single direction' in result['reason']

    def test_temporally_inconsistent_candidate_rejects_recovery(self):
        _, _, registration, groups, cross, post = evidence()
        cross[1] = (translation(0, 3)@cross[1][0], *cross[1][1:])
        result = _infer_edit(registration, groups, cross, post, (360, 640))
        assert not result['accepted']

    def test_excessive_projected_shape_change_rejects_recovery(self):
        _, _, registration, groups, cross, post = evidence(edit_strength=.08)
        result = _infer_edit(registration, groups, cross, post, (360, 640))
        assert not result['accepted']
        assert 'affine bounds' in result['reason']

    def test_stricter_caller_bounds_are_respected(self):
        _, _, registration, groups, cross, post = evidence()
        result = _infer_edit(registration, groups, cross, post, (360, 640),
                             {'max_scale_change': .005})
        assert not result['accepted']
        assert 'affine bounds' in result['reason']

    def test_original_scene_rejection_cannot_be_overridden_by_color_evidence(self):
        result = recover_partial_edit({}, 8, {'accepted': False, 'scene_consistent': False,
                                             'scene_continuity': {'accepted': True}},
                                      pair_fit=lambda *_: self.fail('Must reject before fitting'))
        assert not result['accepted']

    def test_missing_independent_handles_rejects_recovery(self):
        result = recover_partial_edit({}, 8, {'accepted': True, 'scene_consistent': True},
                                      pair_fit=lambda *_: self.fail('Must reject before fitting'))
        assert not result['accepted']

    def test_motion_peeling_finds_both_broad_motion_layers(self):
        rng = np.random.default_rng(9)
        image = cv2.GaussianBlur(rng.integers(0, 256, (360, 640, 3), dtype=np.uint8), (0, 0), 1.1)
        output = image.copy()
        for x0, x1, dx in ((0, 320, -4), (320, 640, 4)):
            output[:, x0:x1] = cv2.warpAffine(image[:, x0:x1], np.array([[1., 0, dx], [0, 1., 2]]),
                                              (x1-x0, 360), borderMode=cv2.BORDER_REFLECT)
        groups = _motion_models(image, output)
        assert len(groups) == 2
        assert all(group['coverage'] >= .3125 for group in groups)
        rates = sorted(np.asarray(group['matrix'])[0, 2] for group in groups)
        np.testing.assert_allclose(rates, [-4, 4], atol=.15)

    def test_small_foreground_cannot_supply_a_recovery_motion_layer(self):
        rng = np.random.default_rng(9)
        image = cv2.GaussianBlur(rng.integers(0, 256, (360, 640, 3), dtype=np.uint8), (0, 0), 1.1)
        output = cv2.warpAffine(image, np.array([[1., 0, 4], [0, 1., 0]]), (640, 360),
                               borderMode=cv2.BORDER_REFLECT)
        output[140:220, 280:360] = image[140:220, 280:360]
        assert len(_motion_models(image, output)) <= 1

    @unittest.skipUnless(shutil.which('ffmpeg'), 'FFmpeg is required for encoded evidence')
    def test_encoded_two_layer_video_recovers_edit_without_freezing_shared_pan(self):
        rng = np.random.default_rng(29)
        image = cv2.GaussianBlur(rng.integers(0, 256, (360, 640, 3), np.uint8), (0, 0), 1.2)
        frames = []
        for index in range(8):
            frame = image.copy()
            for y0, y1, dx in ((0, 180, -1.8), (180, 360, 1.8)):
                frame[y0:y1] = cv2.warpAffine(
                    image[y0:y1], np.array([[1., 0, index*dx], [0, 1., index*.4]]),
                    (640, 180), borderMode=cv2.BORDER_REFLECT)
            frames.append(frame)
        expected = np.eye(3)
        expected[1, 1] = 1.025
        expected[1, 2] = -.025*CENTER[1]
        registration = translation(0, -.4)@expected
        incoming = cv2.warpAffine(frames[-1], np.linalg.inv(registration)[:2],
                                  (640, 360), borderMode=cv2.BORDER_REFLECT)
        for index in range(8):
            frames.append(cv2.warpAffine(incoming, translation(0, index*.3)[:2],
                                         (640, 360), borderMode=cv2.BORDER_REFLECT))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'mixed-motion.mp4'
            writer = VideoWriter(path, 640, 360, '24', crf=14)
            try:
                for frame in frames:
                    writer.write(frame)
            finally:
                writer.close()
            decoded = dict(enumerate(read_frames(path, 0, len(frames))))
        fit = calibrate_pair(decoded[7], decoded[8])
        result = recover_partial_edit(decoded, 8, fit, pair_fit=calibrate_pair)
        assert result['accepted'], result
        # The true edit is measured after cancelling the shared .4px/frame pan.
        # A missing cancellation creates a systematic .4px error here.
        points = np.array([[0., 0., 1.], [639., 0., 1.], [0., 359., 1.], [639., 359., 1.]])
        error = points@(np.array(result['edit_matrix'])-expected).T
        assert np.linalg.norm(error[:, :2], axis=1).max() < .12
        assert abs(result['supported_rate'][3]-.4) < .08
