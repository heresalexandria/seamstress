"""Per-seam choices preserve drawings, strict provenance and crop safety."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from seamstress.calibration import calibrate_video, _assemble
from seamstress.corrections import DEFAULT_CORRECTION, normalize_correction
from seamstress.media import VideoWriter


ROOT = Path(__file__).resolve().parents[1]
OPTIONS = {'analysis_max_size': 192, 'max_samples': 1200, 'min_samples': 80,
           'local_centers': 8, 'geometry_support_seconds': .25,
           'rate_support_frames': 3, 'local_iterations': 4}


def manual(dx=0., dy=0., **kwargs):
    return {'right_to_left_matrix': [[1., 0., dx], [0., 1., dy], [0., 0., 1.]],
            'pre_rate': [0., 0., 0., 0.], 'post_rate': [0., 0., 0., 0.],
            'ease_rate': False, **kwargs}


def cartoon():
    rng = np.random.default_rng(22)
    frame = np.full((128, 192, 3), (175, 196, 215), np.uint8)
    for _ in range(60):
        x, y = int(rng.integers(3, 177)), int(rng.integers(3, 113))
        color = tuple(map(int, rng.integers(35, 235, 3)))
        cv2.rectangle(frame, (x, y), (x+10, y+9), color, -1)
        cv2.rectangle(frame, (x, y), (x+10, y+9), (15, 15, 15), 1)
    return frame


class CorrectionSchemaTests(unittest.TestCase):
    def test_explicit_defaults_and_toggle_back_preserve_detached_manual(self):
        self.assertEqual(normalize_correction(), DEFAULT_CORRECTION)
        measurements = manual(2., -1.)
        for mode in ('auto', 'off', 'manual'):
            result = normalize_correction({'geometry': mode, 'manual': measurements})
            self.assertEqual(result['manual'], measurements)
            result['manual']['pre_rate'][2] = 999
            self.assertEqual(measurements['pre_rate'][2], 0.)
        self.assertEqual(DEFAULT_CORRECTION['geometry'], 'auto')

    def test_unknown_keys_invalid_types_nonfinite_and_unsafe_measurements_reject(self):
        bad = [[], True, {'geometry': 'morph'}, {'color': []}, {'unknown': 1},
               {'cadence': 1}, {'rate_easing': None}, {'geometry': 'manual'},
               {'manual': {}}, {'manual': {**manual(), 'other': 1}},
               {'manual': manual(ease_rate=1)},
               {'manual': manual(pre_rate=[0, 0, '1', 0])},
               {'manual': manual(post_rate=[0, 0, True, 0])},
               {'manual': manual(pre_rate=[0, 0, float('nan'), 0])},
               {'manual': manual(pre_rate=[0, 0, 10**1000, 0])},
               {'manual': manual(post_rate=[.2, 0, 0, 0])},
               {'manual': manual(pre_rate=[0, 0, 49, 0])},
               {'manual': manual(right_to_left_matrix=[[1, 0, 0], [0, 1, 0], [0, .1, 1]])},
               {'manual': manual(right_to_left_matrix=[[1, 0, 0], [0, -1, 0], [0, 0, 1]])},
               {'manual': manual(right_to_left_matrix=[[1.3, 0, 0], [0, 1, 0], [0, 0, 1]])},
               {'manual': manual(right_to_left_matrix=[[True, 0, 0], [0, 1, 0], [0, 0, 1]])},
               {'manual': manual(500, 0)}]
        for value in bad:
            with self.subTest(value=value), self.assertRaises(ValueError):
                normalize_correction(value, metadata={'width': 192, 'height': 128})

    def test_import_provenance_is_checked_even_when_temporarily_disabled(self):
        provenance = {'kind': 'calibration', 'label': 'Reviewed framing',
                      'source_sha256': 'a'*64, 'frame': 16, 'calibration_sha256': 'b'*64}
        value = {'geometry': 'off', 'manual': manual(provenance=provenance)}
        self.assertEqual(normalize_correction(value, source_sha256='a'*64, frame=16)['manual']['provenance'], provenance)
        for arguments in ({'source_sha256': 'c'*64}, {'frame': 17}):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                normalize_correction(value, **arguments)
        for changed in ({'frame': True}, {'label': ''}, {'source_sha256': 'abc'}, {'unknown': 'x'}):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                normalize_correction({'manual': manual(provenance={**provenance, **changed})})

    def test_accepted_75_second_measurements_rebuild_the_exact_reviewed_movie(self):
        calibration = json.loads((ROOT/'plans/IYTYT-framing-reviewed-calibration.json').read_text())
        reference = json.loads((ROOT/'plans/IYTYT-framing-reviewed.json').read_text())
        record = next(cut for cut in calibration['cuts'] if cut['frame'] == 1805)
        measurements = {key: record[key] for key in ('right_to_left_matrix', 'pre_rate', 'post_rate', 'ease_rate')}
        measurements['provenance'] = {'kind': 'calibration', 'label': 'Previously accepted 1:15',
                                      'source_sha256': calibration['source_sha256'], 'frame': 1805}
        policy = normalize_correction({'geometry': 'manual', 'manual': measurements},
                                      metadata=calibration['source'], source_sha256=calibration['source_sha256'], frame=1805)
        calibration['correction_settings'] = [{'frame': 1805, 'correction': policy}]
        record.update({key: policy['manual'][key] for key in ('right_to_left_matrix', 'pre_rate', 'post_rate', 'ease_rate')})
        plan = _assemble(calibration, calibration['source'])
        np.testing.assert_allclose(plan['frame_matrices'], reference['frame_matrices'], rtol=0, atol=1e-12)
        np.testing.assert_array_equal(plan['view_matrix'], reference['view_matrix'])
        self.assertEqual(plan['correction_settings'], calibration['correction_settings'])
        self.assertEqual(plan['grade_curves'], reference['grade_curves'])
        self.assertEqual(plan['local_color_curves'], reference['local_color_curves'])
        self.assertFalse(record['ease_rate'])
        self.assertNotEqual(record['pre_rate'], [0., 0., 0., 0.])


@unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'), 'FFmpeg required')
class CorrectionVideoTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.folder = Path(self.temporary.name)
        self.counter = 0
        left = cartoon()
        shifted = cv2.warpAffine(left, np.float32([[1, 0, 3], [0, 1, -2]]),
                                 (192, 128), borderMode=cv2.BORDER_REFLECT101)
        self.right = np.clip(shifted.astype(float)*[1.025, .985, 1.01]+[3, -2, 4], 0, 255).astype(np.uint8)
        self.source = self.folder/'source.mp4'
        with VideoWriter(self.source, 192, 128, '24', crf=0, preset='ultrafast') as writer:
            for index in range(36):
                writer.write(left if index < 16 else self.right)

    def tearDown(self):
        self.temporary.cleanup()

    def run_policy(self, correction=None, **kwargs):
        self.counter += 1
        return calibrate_video(self.source, [16], self.folder/f'{self.counter}.json',
                               options=OPTIONS, seam_settings={16: correction} if correction is not None else None, **kwargs)

    def test_omitted_and_explicit_defaults_produce_identical_render_fields(self):
        one = self.run_policy()
        two = self.run_policy(copy.deepcopy(DEFAULT_CORRECTION))
        first, second = [json.loads(Path(result['plan_path']).read_text()) for result in (one, two)]
        for key in ('frame_matrices', 'view_matrix', 'grade_curves', 'local_color_curves', 'segments'):
            self.assertEqual(first[key], second[key], key)
        self.assertEqual(one['report']['summary'], two['report']['summary'])
        self.assertTrue(one['report']['seams'][0]['color']['global_accepted'])

    def test_geometry_off_keeps_color_and_is_distinct_from_unresolved(self):
        with patch('seamstress.calibration.recover_cadence_rate', side_effect=AssertionError('off must skip cadence')):
            result = self.run_policy({'geometry': 'off'})
        plan = json.loads(Path(result['plan_path']).read_text())
        np.testing.assert_array_equal(plan['frame_matrices'], np.repeat(np.eye(3)[None], 36, axis=0))
        self.assertTrue(result['calibration']['grade_curves'])
        self.assertEqual(result['unresolved_seams'], [])
        self.assertEqual(result['report']['seams'][0]['geometry_status'], 'off')
        self.assertEqual(result['report']['summary']['geometry_disabled_frames'], [16])

    def test_color_off_skips_observations_and_tone_disables_local_fit(self):
        with patch('seamstress.calibration._flow_observations', side_effect=AssertionError('off must skip all color')):
            result = self.run_policy({'color': 'off'})
        self.assertEqual(result['calibration']['grade_curves'], [])
        self.assertEqual(result['calibration']['local_color_curves'], [])
        self.assertEqual(result['report']['seams'][0]['color']['status'], 'off')
        with patch('seamstress.calibration._fit_local', side_effect=AssertionError('tone must skip local fit')):
            tone = self.run_policy({'color': 'tone'})
        self.assertTrue(tone['calibration']['grade_curves'])
        self.assertEqual(tone['calibration']['local_color_curves'], [])

    def test_manual_overrides_rejected_auto_geometry_without_applying_diagnostic_rates(self):
        from seamstress.calibration import calibrate_pair
        actual = calibrate_pair
        calls = []
        def reject_seam(left, right, options):
            fit = actual(left, right, options)
            if not calls:
                fit.update(accepted=False, reason='Rejected automatic geometry')
            calls.append(1)
            return fit
        measurements = manual(-3, 2, pre_rate=[.0001, 0., .2, -.1],
                              post_rate=[0., 0., -.3, .1], ease_rate=True)
        provenance = {'kind': 'manual', 'label': 'Reviewed shift', 'frame': 16,
                      'source_sha256': hashlib.sha256(self.source.read_bytes()).hexdigest()}
        measurements['provenance'] = provenance
        with patch('seamstress.calibration.calibrate_pair', side_effect=reject_seam), \
             patch('seamstress.calibration.recover_cadence_rate', side_effect=AssertionError('manual must skip cadence')), \
             patch('seamstress.calibration.recover_partial_edit', side_effect=AssertionError('manual must skip recovery')):
            result = self.run_policy({'geometry': 'manual', 'manual': measurements, 'rate_easing': False, 'color': 'tone'})
        record = result['calibration']['cuts'][0]
        for key in ('right_to_left_matrix', 'pre_rate', 'post_rate'):
            self.assertEqual(record[key], measurements[key])
        self.assertFalse(record['ease_rate'])
        self.assertFalse(result['report']['seams'][0]['geometry']['accepted'])
        self.assertEqual(result['report']['seams'][0]['geometry_status'], 'manual')
        self.assertTrue(result['calibration']['grade_curves'])
        self.assertEqual(result['report']['summary']['cadence_adjusted_frames'], [])
        self.assertEqual(result['report']['summary']['manual_geometry_frames'], [16])
        self.assertEqual(result['calibration']['review_decisions'][0]['provenance'], provenance)
        plan = json.loads(Path(result['plan_path']).read_text())
        # Disabling easing still retains the user's expected one-frame motion.
        matrix = np.asarray(record['right_to_left_matrix'])
        self.assertGreater(np.linalg.norm(np.asarray(plan['frame_matrices'][15])@matrix-
                                          np.asarray(plan['frame_matrices'][16])), .05)

    def test_automatic_easing_can_be_disabled_without_erasing_expected_motion(self):
        measured = [(np.array([0., 0., .4, 0.]), True, {}), (np.zeros(4), True, {})]
        with patch('seamstress.calibration._camera_rate', side_effect=measured), \
             patch('seamstress.calibration.recover_cadence_rate', side_effect=AssertionError('cadence disabled')):
            result = self.run_policy({'rate_easing': False, 'cadence': False, 'color': 'off'})
        record = result['calibration']['cuts'][0]
        self.assertFalse(record['ease_rate'])
        self.assertEqual(record['pre_rate'], [0., 0., .4, 0.])
        self.assertEqual(result['report']['summary']['cadence_adjusted_frames'], [])

    def test_recovery_toggles_gate_each_estimator_independently(self):
        recovered = {'accepted': True, 'edit_matrix': manual(-3, 2)['right_to_left_matrix']}
        for partial_enabled, endpoint_enabled in ((True, False), (False, True), (False, False)):
            with self.subTest(partial=partial_enabled, endpoint=endpoint_enabled), \
                 patch('seamstress.calibration._camera_rate', return_value=(np.zeros(4), False, {})), \
                 patch('seamstress.calibration.recover_partial_edit', return_value=copy.deepcopy(recovered)) as partial, \
                 patch('seamstress.calibration.recover_endpoint_edit', return_value=copy.deepcopy(recovered)) as endpoint:
                result = self.run_policy({'partial_recovery': partial_enabled, 'endpoint_recovery': endpoint_enabled, 'color': 'off'})
            self.assertEqual(partial.call_count, int(partial_enabled))
            self.assertEqual(endpoint.call_count, int(endpoint_enabled and not partial_enabled))
            self.assertEqual(result['unresolved_seams'], [] if partial_enabled or endpoint_enabled else [16])

    def test_cadence_summary_only_counts_rates_actually_used_by_assembly(self):
        for accepted_side, easing in (('post', False), ('post', True), ('pre', False)):
            def cadence(frames, cut, side, **kwargs):
                return {'accepted': side == accepted_side, 'rate': [0., 0., .4, 0.]}
            with self.subTest(side=accepted_side, easing=easing), \
                 patch('seamstress.calibration._camera_rate', side_effect=lambda *args: (np.zeros(4), True, {})), \
                 patch('seamstress.calibration.recover_cadence_rate', side_effect=cadence):
                result = self.run_policy({'rate_easing': easing, 'color': 'off'})
            expected = [16] if accepted_side == 'pre' or easing else []
            self.assertEqual(result['report']['summary']['cadence_adjusted_frames'], expected)

    def test_invalid_policy_and_source_provenance_publish_no_artifacts(self):
        digest = hashlib.sha256(self.source.read_bytes()).hexdigest()
        invalid = [{17: {}}, {'16': {}}, {True: {}}, {16: {'geometry': 'bad'}},
                   {16: {'manual': manual(provenance={'kind': 'calibration', 'label': 'Wrong source',
                                                    'source_sha256': ('0' if digest[0] != '0' else '1')*64, 'frame': 16})}}]
        for index, settings in enumerate(invalid):
            destination = self.folder/f'new-{index}'/'analysis.json'
            with self.subTest(settings=settings), self.assertRaises(ValueError), \
                 patch('seamstress.calibration.iter_frames', side_effect=AssertionError('invalid input must not decode')):
                calibrate_video(self.source, [16], destination, options=OPTIONS, seam_settings=settings)
            self.assertFalse(destination.parent.exists())

    def test_manual_excessive_crop_fails_instead_of_silently_ignoring_user_geometry(self):
        output = self.folder/'unsafe.json'
        with self.assertRaisesRegex(ValueError, 'Manual geometry.*crop budget'):
            calibrate_video(self.source, [16], output, options=OPTIONS,
                            seam_settings={16: {'geometry': 'manual', 'manual': manual(-25, 0), 'color': 'off'}})
        self.assertFalse(output.exists())
        self.assertFalse(output.with_suffix('.plan.json').exists())
        self.assertFalse(output.with_suffix('.report.json').exists())
        self.assertFalse(list(self.folder.glob('.seamstress-*')))


if __name__ == '__main__':
    unittest.main()
