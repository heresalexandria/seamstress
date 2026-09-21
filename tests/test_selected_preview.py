"""A selected-seam preview renders its actual source range without a full-shot job."""
import copy
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from seamstress import conform, pipeline, projects
from seamstress.design import ALGORITHM, build_conform_plan
from seamstress.media import VideoWriter, probe, read_frames


@unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'), 'FFmpeg required')
class SelectedPreviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.media = tempfile.TemporaryDirectory(prefix='seamstress-selected-preview-')
        root = Path(cls.media.name)
        silent, cls.source = root/'silent.mp4', root/'source.mp4'
        with VideoWriter(silent, 96, 64, '24000/1001', crf=0, preset='ultrafast') as writer:
            for index in range(72):
                picture = np.full((64, 96, 3), (35+index*2, 185-index, 75+index), np.uint8)
                picture[8:28, 10+index%40:30+index%40] = (205, 40, 120)
                writer.write(picture)
        subprocess.run(['ffmpeg', '-v', 'error', '-nostdin', '-n', '-i', str(silent),
                        '-f', 'lavfi', '-i', 'aevalsrc=0.15*sin(2*PI*(300*t+70*t*t)):s=48000:d=3.003',
                        '-map', '0:v', '-map', '1:a', '-c:v', 'copy', '-c:a', 'aac',
                        '-b:a', '192k', str(cls.source)], check=True, capture_output=True)

    @classmethod
    def tearDownClass(cls):
        cls.media.cleanup()

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.folder = Path(self.directory.name)
        self.project = projects.create_project(self.source, self.folder/'project')
        self.path = Path(self.project['projectPath'])
        self.project = projects.set_seams(self.path, [18, {'frame': 30, 'enabled': False}, 54])
        calibration = {'schema_version': 1, 'method': 'source_conform_calibration', 'algorithm': ALGORITHM,
                       'source': self.project['metadata'], 'source_sha256': self.project['sourceSha256'],
                       'parameters': {'geometry_support': 4, 'rate_support': 2,
                                      'source_margin_pixels': 0, 'max_view_crop_fraction_total_dimension': .08},
                       'cuts': [{'frame': frame, 'right_to_left_matrix': np.eye(3).tolist(),
                                 'pre_rate': [0, 0, 0, 0], 'post_rate': [0, 0, 0, 0], 'ease_rate': False}
                                for frame in (18, 54)],
                       'excluded_geometry': [], 'grade_curves': [], 'local_color_curves': []}
        plan = build_conform_plan(calibration, self.project['metadata'])
        self.plan_path = self.folder/'accepted.plan.json'
        projects.atomic_json(self.plan_path, plan)
        self.project['artifacts']['plan'] = str(self.plan_path)
        self.project = projects.save_project(self.project)

    def tearDown(self):
        self.directory.cleanup()

    def test_direct_encoded_range_preserves_other_previews_and_source_audio_alignment(self):
        other = {'frame': 18, 'path': str(self.source), 'startFrame': 12, 'endFrame': 24}
        self.project['artifacts'].update(fullPreview=str(self.source), seamPreviews=[other,
            {'frame': 54, 'path': str(self.source), 'startFrame': 42, 'endFrame': 66}])
        self.project = projects.save_project(self.project)
        with patch.object(pipeline, 'prepare_media', side_effect=AssertionError('No full proxy job')), \
                patch.object(pipeline, 'analyze_project', side_effect=AssertionError('No analysis job')), \
                patch.object(conform, 'render_conform', wraps=conform.render_conform) as render:
            result = pipeline.run_stage(self.path, 'preview',
                options={'frame': 54, 'previewWidth': 96, 'previewSeconds': .5})
        self.assertEqual(render.call_count, 1)
        self.assertEqual((render.call_args.kwargs['start_frame'], render.call_args.kwargs['end_frame']), (48, 60))
        self.assertEqual(result['artifacts']['fullPreview'], str(self.source))
        self.assertEqual(result['artifacts']['seamPreviews'][0], other)
        self.assertEqual(len(result['artifacts']['seamPreviews']), 2)
        clip = result['artifacts']['seamPreviews'][1]
        self.assertEqual((clip['frame'], clip['startFrame'], clip['endFrame']), (54, 48, 60))
        metadata = probe(clip['path'])
        self.assertEqual((metadata['frame_count'], metadata['fps_fraction']), (12, '24000/1001'))
        sidecar = json.loads(Path(clip['path']).with_suffix('.repair.json').read_text())
        self.assertEqual((sidecar['source_start_frame'], sidecar['source_end_frame_exclusive']), (48, 60))
        actual, expected = read_frames(clip['path'], 0, 12), read_frames(self.source, 48, 12)
        self.assertLess(float(np.abs(actual.astype(float)-expected).mean()), 2.)
        # Varying-frequency audio makes a wrong source range observably different.
        def pcm(path):
            result = subprocess.run(['ffmpeg', '-v', 'error', '-i', str(path), '-map', '0:a:0',
                '-ac', '1', '-ar', '48000', '-f', 'f32le', 'pipe:1'], check=True, capture_output=True)
            return np.frombuffer(result.stdout, dtype='<f4')
        original, audio = pcm(self.source), pcm(clip['path'])
        expected_audio = original[48*2002:60*2002]
        count = min(len(audio), len(expected_audio))
        self.assertGreater(np.corrcoef(audio[2048:count-1024], expected_audio[2048:count-1024])[0, 1], .99)
        self.assertLess(abs(len(audio)-12*2002), 1025)

    def test_missing_or_disabled_target_and_missing_plan_fail_without_whole_shot_work(self):
        unchanged = self.path.read_bytes()
        cases = [({'frame': value}, self.project) for value in [12, 30, True, None, '54']]
        for plan in [None, str(self.folder/'missing.plan.json')]:
            project = copy.deepcopy(self.project)
            if plan is None:
                project['artifacts'].pop('plan')
            else:
                project['artifacts']['plan'] = plan
            cases.append(({'frame': 54}, project))
        for options, project in cases:
            with self.subTest(options=options, plan=project['artifacts'].get('plan')), \
                    patch.object(pipeline, 'prepare_media', side_effect=AssertionError('No proxy job')), \
                    patch.object(pipeline, 'analyze_project', side_effect=AssertionError('No global analysis')), \
                    patch.object(conform, 'render_conform', side_effect=AssertionError('No render')), \
                    self.assertRaises(ValueError):
                pipeline.preview_project(project, options=options)
        self.assertEqual(self.path.read_bytes(), unchanged)

    def test_selected_preview_does_not_create_a_full_preview(self):
        result = pipeline.run_stage(self.path, 'preview',
            options={'frame': 18, 'previewWidth': 48, 'previewSeconds': .25})
        self.assertNotIn('fullPreview', result['artifacts'])
        self.assertEqual([row['frame'] for row in result['artifacts']['seamPreviews']], [18])
        self.assertEqual(probe(result['artifacts']['seamPreviews'][0]['path'])['frame_count'], 6)
