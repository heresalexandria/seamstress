"""Persisted seam decisions, reviewed imports, and applied app readouts."""
import copy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from seamstress import pipeline, projects
from seamstress.corrections import DEFAULT_CORRECTION
from seamstress.design import ALGORITHM, build_conform_plan
from seamstress.media import VideoWriter


class AppliedReadoutTests(unittest.TestCase):
    def test_unused_post_rate_diagnostic_is_not_reported_as_applied_cadence(self):
        project={'seams':[{'frame':24,'correction':dict(DEFAULT_CORRECTION)}]}
        result={'calibration':{},'report':{'seams':[{'frame':24,'rate_easing':False,
            'pre_rate':{'cadence_recovery':{'accepted':False}},
            'post_rate':{'cadence_recovery':{'accepted':True}},'color':{'status':'unchanged'}}]}}
        self.assertFalse(pipeline._seam_results(project,result)[0]['cadence'])
        result['report']['seams'][0]['rate_easing']=True
        self.assertTrue(pipeline._seam_results(project,result)[0]['cadence'])
        result['report']['seams'][0]['rate_easing']=False
        result['report']['seams'][0]['pre_rate']['cadence_recovery']['accepted']=True
        self.assertTrue(pipeline._seam_results(project,result)[0]['cadence'])


@unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'), 'FFmpeg required')
class ProjectCorrectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.media_dir = tempfile.TemporaryDirectory()
        cls.source = Path(cls.media_dir.name)/'source.mp4'
        picture = np.random.default_rng(29).integers(30, 220, (64, 96, 3), dtype=np.uint8)
        with VideoWriter(cls.source, 96, 64, '12', crf=0, preset='ultrafast') as writer:
            for _ in range(48):writer.write(picture)

    @classmethod
    def tearDownClass(cls):
        cls.media_dir.cleanup()

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.folder = Path(self.directory.name)
        self.project = projects.create_project(self.source, self.folder/'project')
        self.project = projects.set_seams(Path(self.project['projectPath']), [24])
        self.path = Path(self.project['projectPath'])

    def tearDown(self):
        self.directory.cleanup()

    def calibration(self):
        return {'schema_version': 1, 'method': 'source_conform_calibration', 'algorithm': ALGORITHM,
                'source': self.project['metadata'], 'source_sha256': self.project['sourceSha256'],
                'cuts': [{'frame': 24, 'right_to_left_matrix': [[1, 0, -.7], [0, 1, .2], [0, 0, 1]],
                          'pre_rate': [.0001, 0, .1, 0], 'post_rate': [0, 0, 0, 0], 'ease_rate': False}],
                'excluded_geometry': [], 'grade_curves': [], 'local_color_curves': []}

    def imported(self):
        file = self.folder/'reviewed.json'
        projects.atomic_json(file, self.calibration())
        return projects.import_seam_correction(self.path, 24, file), file

    def test_old_projects_default_to_current_automatic_behavior(self):
        data = json.loads(self.path.read_text())
        data['seams'][0].pop('correction')
        projects.atomic_json(self.path, data)
        loaded = projects.load_project(self.path)
        self.assertEqual(loaded['seams'][0]['correction'], DEFAULT_CORRECTION)
        loaded['artifacts']['plan'] = 'existing-plan'
        projects.save_project(loaded)
        identical = projects.set_seams(self.path, data['seams'])
        self.assertEqual(identical['revision'], loaded['revision'])
        self.assertEqual(identical['artifacts']['plan'], 'existing-plan')

    def test_every_setting_invalidates_plan_previews_and_readout(self):
        values = {'geometry': 'off', 'color': 'tone', 'partial_recovery': False,
                  'endpoint_recovery': False, 'cadence': False, 'rate_easing': False}
        for key, value in values.items():
            with self.subTest(field=key):
                current = projects.load_project(self.path)
                current['seams'][0]['correction'] = dict(DEFAULT_CORRECTION)
                current['artifacts'] = {key: 'historical' for key in ('proxy', 'detection', 'plan', 'report', 'calibration',
                    'fullPreview', 'seamPreviews', 'export', 'verification')}
                current['seamResults'] = [{'frame': 24, 'geometry': 'auto'}]
                projects.save_project(current)
                rows = copy.deepcopy(current['seams']);rows[0]['correction'][key] = value
                saved = projects.set_seams(self.path, rows)
                self.assertEqual(saved['revision'], current['revision']+1)
                self.assertEqual(set(saved['artifacts']), {'proxy', 'detection'})
                self.assertNotIn('seamResults', saved)
                self.assertEqual(projects.load_project(self.path)['seams'][0]['correction'][key], value)
                with self.assertRaisesRegex(RuntimeError, 'Seams changed'):
                    pipeline._commit(current, {'plan': 'stale'}, 'analyzed')

    def test_import_copies_exact_measurements_and_survives_file_removal(self):
        project, file = self.imported()
        manual = project['seams'][0]['correction']['manual']
        for key in ('right_to_left_matrix', 'pre_rate', 'post_rate', 'ease_rate'):
            self.assertEqual(manual[key], self.calibration()['cuts'][0][key])
        self.assertEqual(manual['provenance']['calibration_sha256'], hashlib.sha256(file.read_bytes()).hexdigest())
        self.assertEqual(manual['provenance']['source_sha256'], project['sourceSha256'])
        self.assertEqual(manual['provenance']['frame'], 24)
        self.assertFalse(project['seams'][0]['correction']['rate_easing'])
        file.unlink()
        loaded = projects.load_project(self.path)
        self.assertEqual(loaded['seams'][0]['correction']['manual'], manual)
        analyzed = pipeline.analyze_project(loaded, options={'analysis_max_size': 96, 'enable_local_color': False})
        calibration = json.loads(Path(analyzed['artifacts']['calibration']).read_text())
        for key in ('right_to_left_matrix', 'pre_rate', 'post_rate', 'ease_rate'):
            self.assertEqual(calibration['cuts'][0][key], manual[key])
        readout = analyzed['seamResults'][0]
        self.assertEqual(readout['geometry'], 'manual')
        self.assertFalse(readout['cadence'])
        self.assertFalse(readout['rateEasing'])
        self.assertEqual(readout['manual']['provenance'], manual['provenance'])
        plan = json.loads(Path(analyzed['artifacts']['plan']).read_text())
        rebuilt = build_conform_plan(calibration, analyzed['metadata'])
        for key in ('frame_matrices', 'view_matrix', 'correction_settings', 'review_decisions'):
            self.assertEqual(rebuilt[key], plan[key])

    def test_bad_imports_leave_project_bytes_unchanged(self):
        changes = [lambda c: c.update(source_sha256='0'*64),
                   lambda c: c['source'].update(width=98),
                   lambda c: c['cuts'][0].update(frame=25),
                   lambda c: c['cuts'].append(copy.deepcopy(c['cuts'][0])),
                   lambda c: c['excluded_geometry'].append({'frame': 24, 'reason': 'unsafe'}),
                   lambda c: c['cuts'][0].update(pre_rate=[float('nan'), 0, 0, 0]),
                   lambda c: c['cuts'][0].update(ease_rate=1),
                   lambda c: c.update(method='source_conform')]
        original = self.path.read_bytes()
        file = self.folder/'bad.json'
        for index, change in enumerate(changes):
            with self.subTest(case=index):
                data = copy.deepcopy(self.calibration());change(data)
                file.write_text(json.dumps(data))
                with self.assertRaises(ValueError):projects.import_seam_correction(self.path, 24, file)
                self.assertEqual(self.path.read_bytes(), original)
        file.write_text('not json')
        with self.assertRaisesRegex(ValueError, 'valid reviewed'):projects.import_seam_correction(self.path, 24, file)
        self.assertEqual(self.path.read_bytes(), original)

    def test_moving_marker_discards_bound_manual_but_keeps_other_choices(self):
        project, _ = self.imported()
        rows = copy.deepcopy(project['seams']);rows[0]['frame'] = 25
        rows[0]['correction']['color'] = 'tone'
        moved = projects.set_seams(self.path, rows)
        settings = moved['seams'][0]['correction']
        self.assertEqual(settings['geometry'], 'auto')
        self.assertEqual(settings['color'], 'tone')
        self.assertNotIn('manual', settings)

    def test_redetection_keeps_reviewed_markers_at_their_exact_frames(self):
        project, _ = self.imported()
        detected = {'seams': [{'frame': 25, 'origin': 'detected', 'confidence': .9}]}
        with patch('seamstress.detection.detect_seams', return_value=detected):
            redetected = pipeline.detect_project(project)
        self.assertEqual([row['frame'] for row in redetected['seams']], [24, 25])
        self.assertEqual(redetected['seams'][0]['correction'], project['seams'][0]['correction'])
        self.assertNotIn('manual', redetected['seams'][1]['correction'])
        detected = {'seams': [{'frame': 24, 'origin': 'detected', 'confidence': .8}]}
        with patch('seamstress.detection.detect_seams', return_value=detected):
            matching = pipeline.detect_project(redetected)
        self.assertEqual(matching['seams'][0]['id'], project['seams'][0]['id'])
        self.assertEqual(matching['seams'][0]['correction'], project['seams'][0]['correction'])

    def test_disabled_framing_is_intentional_and_app_readout_matches_plan(self):
        rows = self.project['seams'];rows[0]['correction'].update(geometry='off', color='off')
        project = projects.set_seams(self.path, rows)
        result = pipeline.analyze_project(project, options={'analysis_max_size': 96})
        self.assertEqual(result['warnings'], [])
        self.assertEqual(result['seamResults'][0]['geometry'], 'off')
        self.assertEqual(result['seamResults'][0]['color'], 'off')
        self.assertNotIn('manual', result['seamResults'][0])
        calibration = json.loads(Path(result['artifacts']['calibration']).read_text())
        plan = build_conform_plan(calibration, result['metadata'])
        self.assertEqual(plan['unresolved_seams'], [])
        self.assertEqual(plan['grade_curves'], [])
        np.testing.assert_allclose(plan['view_matrix'], np.eye(3))

    def test_desktop_worker_import_protocol_uses_same_validation(self):
        reviewed = self.folder/'reviewed.json';projects.atomic_json(reviewed, self.calibration())
        request = {'operation': 'importSeamCorrection', 'args': {'projectPath': str(self.path),
                   'frame': 24, 'reviewedPath': str(reviewed)}}
        result = subprocess.run([sys.executable, '-m', 'seamstress.desktop_worker'], input=json.dumps(request)+'\n',
                                text=True, capture_output=True, check=True, timeout=30)
        events = [json.loads(line) for line in result.stdout.splitlines()]
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]['type'], 'complete')
        self.assertEqual(events[0]['project']['seams'][0]['correction']['geometry'], 'manual')


if __name__ == '__main__':unittest.main()
