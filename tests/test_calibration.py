"""Source-specific automatic calibration: real encoded synthetic cartoon clips."""
from pathlib import Path
import json
import shutil
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from seamstress.calibration import (CalibrationCancelled, calibrate_pair, calibrate_video,
                                   _protected_luts, _color_scene_evidence)
from seamstress.conform import validate_conform_plan
from seamstress.media import VideoWriter, probe


def cartoon(width=192, height=128, seed=22):
    rng = np.random.default_rng(seed)
    frame = np.full((height, width, 3), (175, 196, 215), np.uint8)
    for _ in range(60):
        x, y = int(rng.integers(3, width-15)), int(rng.integers(3, height-15))
        color = tuple(map(int, rng.integers(35, 235, 3)))
        cv2.rectangle(frame, (x, y), (x+10, y+9), color, -1)
        cv2.rectangle(frame, (x, y), (x+10, y+9), (15, 15, 15), 1)
    return frame


def multilayer_cartoon():
    """Two broad scene regions continue with incompatible motion and a grade."""
    left = np.concatenate([
        np.concatenate([cartoon(seed=22), cartoon(seed=33)], axis=1),
        np.concatenate([cartoon(seed=44), cartoon(seed=55)], axis=1)], axis=0)
    right = np.empty_like(left)
    for rows, dx, dy in [(slice(0, 128), 8, 4), (slice(128, 256), -8, -4)]:
        right[rows] = cv2.warpAffine(left, np.float32([[1, 0, dx], [0, 1, dy]]),
                                    (384, 256), borderMode=cv2.BORDER_REFLECT101)[rows]
    right = np.clip(right.astype(float)*[1.015, .99, 1.01]+[3, -2, 3], 0, 255).astype(np.uint8)
    return left, right


OPTIONS = {'analysis_max_size': 192, 'max_samples': 1800, 'min_samples': 80,
           'local_centers': 8, 'geometry_support_seconds': .5,
           'rate_support_frames': 3, 'local_iterations': 5}


class CalibrationPairTests(unittest.TestCase):
    def test_independent_motion_establishes_color_evidence_without_global_camera(self):
        left, right = multilayer_cartoon()
        self.assertFalse(calibrate_pair(left, right, {'analysis_max_size': 384})['scene_consistent'])
        evidence = _color_scene_evidence(left, right)
        self.assertTrue(evidence['accepted'], evidence)
        self.assertGreaterEqual(evidence['heldout_matches'], 24)
        self.assertGreaterEqual(evidence['heldout_coverage'], .5)
        self.assertGreaterEqual(evidence['heldout_convex_hull_fraction'], .4)

    def test_local_continuity_rejects_shared_palette_small_copy_and_large_motion(self):
        left, _ = multilayer_cartoon()
        unrelated = np.concatenate([
            np.concatenate([cartoon(seed=71), cartoon(seed=82)], axis=1),
            np.concatenate([cartoon(seed=93), cartoon(seed=104)], axis=1)], axis=0)
        small_copy = unrelated.copy()
        small_copy[70:170, 130:250] = left[70:170, 130:250]
        # Same exact palette and image tiles, rearranged into a different scene.
        shuffled = np.concatenate([left[128:, 192:], left[128:, :192]], axis=1)
        shuffled = np.concatenate([shuffled, np.concatenate([left[:128, 192:], left[:128, :192]], axis=1)])
        shifted = cv2.warpAffine(left, np.float32([[1, 0, 70], [0, 1, 0]]),
                                (384, 256), borderMode=cv2.BORDER_REFLECT101)
        for label, right in [('same palette', unrelated), ('small shared area', small_copy),
                             ('shuffled tiles', shuffled), ('large motion', shifted)]:
            with self.subTest(case=label):
                self.assertFalse(_color_scene_evidence(left, right)['accepted'])

    def test_native_transform_direction_at_varied_aspects(self):
        for width, height in [(192, 128), (128, 192), (384, 128)]:
            with self.subTest(size=(width, height)):
                left = cartoon(width, height)
                forward = np.float32([[1, 0, 3], [0, 1, -2]])
                right = cv2.warpAffine(left, forward, (width, height), borderMode=cv2.BORDER_REFLECT101)
                right = np.clip(right.astype(float)*[1.02, .98, 1.01]+[3, -2, 4], 0, 255).astype(np.uint8)
                result = calibrate_pair(left, right, OPTIONS)
                self.assertTrue(result['accepted'], result)
                np.testing.assert_allclose(np.asarray(result['matrix'])[:2, 2], [-3, 2], atol=.45)
                self.assertGreater(result['diagnostics']['coverage'], .3)

    def test_scene_change_and_large_geometry_are_not_accepted(self):
        left = cartoon()
        unrelated = cartoon(seed=818)
        self.assertFalse(calibrate_pair(left, unrelated, OPTIONS)['scene_consistent'])
        shifted = cv2.warpAffine(left, np.float32([[1, 0, 28], [0, 1, 0]]), (192, 128), borderMode=cv2.BORDER_REFLECT101)
        result = calibrate_pair(left, shifted, OPTIONS)
        self.assertFalse(result['accepted'])
        self.assertIn('exceeds', result['reason'])

    def test_protected_tone_maps_preserve_endpoints_and_monotonicity(self):
        a, b = _protected_luts(np.array([1.02, .99, 1.01]), np.array([2., -2., 1.]))
        for table in (a, b):
            np.testing.assert_array_equal(table[0], [0, 0, 0])
            np.testing.assert_allclose(table[-1], [255, 255, 255])
            self.assertTrue(np.all(np.diff(table, axis=0) > 0))

    def test_pchip_inverse_endpoint_roundoff_is_normalized_exactly(self):
        gain = np.array([.9105630864231296, .9194660924098947, 1.0422492679937663])
        bias = np.array([-.537396886443406, -3.572082515242508, -1.8633197383608482])
        # This real numerical case previously returned255.00000000000006.
        for table in _protected_luts(gain, bias):
            self.assertGreaterEqual(float(table.min()), 0.)
            self.assertLessEqual(float(table.max()), 255.)
            np.testing.assert_array_equal(table[0], [0., 0., 0.])
            np.testing.assert_array_equal(table[-1], [255., 255., 255.])
            self.assertTrue(np.all(np.diff(table, axis=0) >= 0))


@unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'), 'FFmpeg required for real video calibration')
class CalibrationVideoTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.folder = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def video(self, name, left, right=None, cut=16, count=36):
        destination = self.folder/name
        right = left if right is None else right
        with VideoWriter(destination, left.shape[1], left.shape[0], '24', crf=0, preset='ultrafast') as writer:
            for n in range(count):
                writer.write(left if n < cut else right)
        return destination

    def run_calibration(self, source, seams, name='calibration.json', **kwargs):
        return calibrate_video(source, seams, self.folder/name, options=OPTIONS, **kwargs)

    def test_translated_and_regraded_cartoon_improves_without_frame_changes(self):
        left = cartoon()
        right = cv2.warpAffine(left, np.float32([[1, 0, 3], [0, 1, -2]]), (192, 128), borderMode=cv2.BORDER_REFLECT101)
        right = np.clip(right.astype(float)*[1.025, .985, 1.01]+[3, -2, 4], 0, 255).astype(np.uint8)
        source = self.video('shift.mp4', left, right)
        original = source.read_bytes(); events = []
        with patch('seamstress.calibration._color_scene_evidence',
                   side_effect=AssertionError('Established global path must not use fallback')), \
                patch('seamstress.calibration.recover_partial_edit',
                      side_effect=AssertionError('Reliable camera motion must not use partial recovery')):
            result = self.run_calibration(source, [16], progress=events.append)
        plan = json.loads(Path(result['plan_path']).read_text()); report = result['report']; color = report['seams'][0]['color']
        validate_conform_plan(plan, probe(source))
        self.assertEqual(report['summary']['geometry_accepted'], 1, report)
        self.assertTrue(color['global_accepted'] or color['local_accepted'], color)
        if color['global_accepted']:
            self.assertLess(color['global_validation'][0]['after']['mae'], color['before']['mae']*.7)
        self.assertEqual(len(plan['frame_matrices']), 36)
        self.assertEqual(plan['source']['fps_fraction'], '24/1')
        self.assertLess(report['summary']['constant_crop_fraction'], .08)
        self.assertEqual(source.read_bytes(), original)
        self.assertEqual(events[-1]['stage'], 'complete')

    def test_multilayer_color_recovery_keeps_geometry_timing_and_source_unchanged(self):
        left, right = multilayer_cartoon()
        source = self.video('multilayer.mp4', left, right)
        original = source.read_bytes()
        result = calibrate_video(source, [16], self.folder/'multilayer.json',
                                 options={**OPTIONS, 'analysis_max_size': 384})
        seam = result['report']['seams'][0]
        self.assertFalse(seam['geometry']['scene_consistent'])
        self.assertTrue(seam['geometry_excluded_reason'])
        self.assertTrue(seam['scene_continuity']['accepted'], seam)
        self.assertEqual(seam['scene_continuity']['method'], 'independent-local-correspondence')
        self.assertEqual(len(seam['scene_continuity']['pairs']), 3)
        self.assertTrue(seam['color']['global_accepted'] or seam['color']['local_accepted'], seam['color'])
        self.assertEqual(result['unresolved_seams'], [16])
        plan = json.loads(Path(result['plan_path']).read_text())
        np.testing.assert_allclose(plan['frame_matrices'], np.repeat(np.eye(3)[None], 36, axis=0))
        np.testing.assert_array_equal(plan['view_matrix'], np.eye(3))
        self.assertEqual(plan['source']['frame_count'], 36)
        self.assertEqual(plan['source']['fps_fraction'], '24/1')
        self.assertEqual(source.read_bytes(), original)

    def test_one_related_pair_cannot_unlock_color_for_unrelated_following_frames(self):
        left, right = multilayer_cartoon()
        unrelated = np.tile(cartoon(seed=818), (2, 2, 1))
        source = self.folder/'single-related-frame.mp4'
        with VideoWriter(source, 384, 256, '24', crf=0, preset='ultrafast') as writer:
            for n in range(36):
                writer.write(left if n < 16 else right if n == 16 else unrelated)
        result = calibrate_video(source, [16], self.folder/'single-related-frame.json',
                                 options={**OPTIONS, 'analysis_max_size': 384})
        seam = result['report']['seams'][0]
        self.assertTrue(seam['scene_continuity']['pairs'][0]['accepted'], seam)
        self.assertFalse(seam['scene_continuity']['accepted'])
        self.assertEqual(seam['color']['status'], 'excluded')
        self.assertEqual(result['calibration']['grade_curves'], [])
        self.assertEqual(result['calibration']['local_color_curves'], [])

    def test_spatial_color_shift_uses_heldout_validated_local_model(self):
        left = cartoon()
        y, x = np.mgrid[:128, :192]
        offset = np.stack([8*np.sin(x/192*np.pi*2), 5*np.cos(y/128*np.pi*2), 6*np.cos(x/192*np.pi)], -1)
        right = np.rint(np.clip(left.astype(float)+offset, 0, 255)).astype(np.uint8)
        source = self.video('local.mp4', left, right)
        result = calibrate_video(source, [16], self.folder/'local.json', options={**OPTIONS, 'local_centers': 16})
        color = result['report']['seams'][0]['color']
        self.assertTrue(color['local_accepted'], color)
        self.assertEqual(len(result['calibration']['local_color_curves']), 1)
        self.assertLess(color['local_validation'][0]['after']['mae'], color['local_validation'][0]['before']['mae']*.5)
        self.assertTrue(all(check['accepted'] for check in color['local_safety']))

    def test_crop_budget_excludes_geometry_instead_of_filling_borders(self):
        left = cartoon()
        right = cv2.warpAffine(left, np.float32([[1, 0, 3], [0, 1, -2]]), (192, 128), borderMode=cv2.BORDER_REFLECT101)
        source = self.video('crop.mp4', left, right)
        result = calibrate_video(source, [16], self.folder/'crop.json', options={**OPTIONS, 'max_view_crop_fraction': .0001})
        self.assertEqual(result['unresolved_seams'], [16])
        self.assertIn('source coverage', result['report']['seams'][0]['geometry_excluded_reason'])
        self.assertEqual(result['report']['summary']['constant_crop_fraction'], 0)

    def test_portrait_identity_and_zero_seams_remain_identity(self):
        source = self.video('identity.mp4', cartoon(128, 192))
        for seams, name in [([16], 'identity.json'), ([], 'none.json')]:
            result = self.run_calibration(source, seams, name)
            plan = json.loads(Path(result['plan_path']).read_text())
            np.testing.assert_allclose(plan['view_matrix'], np.eye(3), atol=1e-8)
            if 'frame_matrices' in plan:
                np.testing.assert_allclose(plan['frame_matrices'], np.repeat(np.eye(3)[None], 36, axis=0), atol=1e-8)
            self.assertEqual(plan['grade_curves'], [])
            self.assertEqual(plan['local_color_curves'], [])
            self.assertEqual(result['report']['summary']['seam_count'], len(seams))

    def test_unrelated_scene_and_short_handles_are_explicitly_excluded(self):
        source = self.video('change.mp4', cartoon(), cartoon(seed=818))
        result = self.run_calibration(source, [1, 16, 17, 35])
        self.assertEqual(result['unresolved_seams'], [1, 16, 17, 35])
        for seam in result['report']['seams']:
            self.assertTrue(seam['geometry_excluded_reason'])
            self.assertEqual(seam['color']['status'], 'excluded')
        plan = json.loads(Path(result['plan_path']).read_text())
        np.testing.assert_allclose(plan['frame_matrices'], np.repeat(np.eye(3)[None], 36, axis=0))

    def test_extended_cadence_evidence_never_crosses_another_marked_join(self):
        source = self.video('nearby-markers.mp4', cartoon(), count=48)
        markers = [12, 24, 36]
        observed = []

        def inspect_window(frames, cut, side, **kwargs):
            position = markers.index(cut)
            previous = markers[position-1] if position else 0
            following = markers[position+1] if position+1 < len(markers) else 48
            self.assertGreaterEqual(min(frames), previous)
            self.assertLess(max(frames), following)
            observed.append((cut, side))
            return {'accepted': False, 'reason': 'Independent cadence handles unavailable'}

        with patch('seamstress.calibration.recover_cadence_rate', side_effect=inspect_window):
            result = self.run_calibration(source, markers)
        self.assertEqual(len(observed), 6)
        self.assertEqual(result['unresolved_seams'], [])
        plan = json.loads(Path(result['plan_path']).read_text())
        np.testing.assert_allclose(plan['frame_matrices'], np.repeat(np.eye(3)[None], 48, axis=0))

    def test_scene_change_without_nearby_cuts_is_not_color_corrected(self):
        source = self.video('unrelated.mp4', cartoon(), cartoon(seed=818))
        result = self.run_calibration(source, [16])
        self.assertEqual(result['unresolved_seams'], [16])
        self.assertEqual(result['report']['seams'][0]['color']['status'], 'excluded')

    def test_cancel_during_analysis_publishes_nothing(self):
        source = self.video('cancel.mp4', cartoon())
        state = {'cancel': False}
        def progress(event):
            if event['stage'] == 'geometry':
                state['cancel'] = True
        with self.assertRaises(CalibrationCancelled):
            self.run_calibration(source, [16], progress=progress, cancelled=lambda: state['cancel'])
        self.assertFalse((self.folder/'calibration.json').exists())
        self.assertFalse((self.folder/'calibration.plan.json').exists())
        self.assertFalse(list(self.folder.glob('.seamstress-*')))

    def test_overwrite_source_alias_and_bad_indices_are_rejected(self):
        source = self.video('protected.mp4', cartoon())
        original = source.read_bytes()
        with self.assertRaises(ValueError):
            calibrate_video(source, [], source, options=OPTIONS)
        destination = self.folder/'exists.json'; destination.write_text('keep')
        with self.assertRaises(ValueError):
            calibrate_video(source, [], destination, options=OPTIONS)
        for seams in ([16, 16], [17, 16], [0], [36], [True]):
            with self.subTest(seams=seams), self.assertRaises(ValueError):
                self.run_calibration(source, seams)
        self.assertEqual(destination.read_text(), 'keep')
        self.assertEqual(source.read_bytes(), original)


if __name__ == '__main__':
    unittest.main()
