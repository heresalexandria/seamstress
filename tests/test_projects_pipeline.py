"""Saved-project contracts and the real, encoded source-conform workflow."""
from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import cv2
import numpy as np

from seamstress import cli, pipeline, projects
from seamstress.media import VideoWriter, probe, read_frame


class ProjectValueTests(unittest.TestCase):
    def test_seconds_and_fractional_frame_rate_non_drop_timecodes(self):
        for text in ['8', '00:08.000', '00:00:08.000']:
            self.assertEqual(projects.timecode_to_frame(text, '12/1'), 96)
        # Decimal seconds describe elapsed time, but NDF labels count nominal frames.
        self.assertEqual(projects.timecode_to_frame('60', '30000/1001'), 1798)
        self.assertEqual(projects.timecode_to_frame('00:01:00:00', '30000/1001'), 1800)
        self.assertEqual(projects.timecode_to_frame('01:00:00:17', '30000/1001'), 108017)
        self.assertEqual(projects.timecode_to_frame('01:00:00:00', '24000/1001'), 86400)
        for text in ['', '-1', '1:60', '00:00:00:30', '1.2:03', 'nan', '1:2:3:4:5']:
            with self.subTest(text=text), self.assertRaises(ValueError):
                projects.timecode_to_frame(text, '30000/1001')

    def test_markers_are_exact_sorted_unique_and_keep_disabled_state(self):
        metadata = {'frame_count': 120, 'fps': 12}
        seams = projects.normalized_seams([
            {'frame': 96, 'enabled': False, 'origin': 'manual', 'id': 'keep-me'}, 12, 96], metadata)
        self.assertEqual([s['frame'] for s in seams], [12, 96])
        self.assertEqual(seams[1]['time'], 8)
        self.assertFalse(seams[1]['enabled'])
        self.assertEqual(seams[1]['id'], 'keep-me')
        for invalid in [0, 120, -1, True, 12.5, {'frame': 12, 'enabled': 1},
                        {'frame': 12, 'confidence': float('nan')}]:
            with self.subTest(invalid=invalid), self.assertRaises((ValueError, TypeError)):
                projects.normalized_seams([invalid], metadata)

    def test_failed_atomic_save_keeps_last_valid_project(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'project.json'
            projects.atomic_json(path, {'complete': True})
            original = path.read_bytes()
            with self.assertRaises(ValueError):
                projects.atomic_json(path, {'bad': float('nan')})
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(list(Path(folder).glob('.seamstress-*')), [])

    def test_hdr_high_bit_depth_odd_and_single_frame_sources_rejected(self):
        base = {'frame_count': 24, 'width': 160, 'height': 96, 'fps': 12,
                'pix_fmt': 'yuv420p', 'color_transfer': 'bt709'}
        formats = {'pixel_formats': [{'name': name, 'components': [{'bit_depth': depth}]}
                                    for name, depth in [('yuv420p', 8), ('yuv420p10le', 10)]]}
        for override in [{'color_transfer': 'smpte2084'}, {'color_transfer': 'arib-std-b67'},
                         {'pix_fmt': 'yuv420p10le'}, {'width': 159}, {'height': 30}, {'frame_count': 1}]:
            with self.subTest(override=override), patch.object(projects, 'probe', return_value={**base, **override}), \
                    patch.object(projects, '_run', return_value=SimpleNamespace(stdout=json.dumps(formats).encode())), \
                    patch.object(projects.subprocess, 'Popen') as process, self.assertRaises(ValueError):
                projects.inspect_source(Path('/unused.mp4'))
            process.assert_not_called()


@unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'), 'FFmpeg is required')
class RealProjectPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix='seamstress-pipeline-test-')
        cls.root = Path(cls.temporary.name)
        cls.silent = cls.root/'silent.mp4'
        cls.audio = cls.root/'audio-first-two-streams.mp4'
        image = np.full((96, 160, 3), (43, 78, 114), np.uint8)
        rng = np.random.default_rng(714)
        for _ in range(55):
            x, y = rng.integers(2, 148), rng.integers(2, 84)
            color = tuple(int(v) for v in rng.integers(35, 224, 3))
            cv2.rectangle(image, (x, y), (x+8, y+8), color, -1)
        cv2.circle(image, (82, 45), 17, (204, 146, 98), -1)
        cv2.circle(image, (82, 45), 17, (16, 17, 20), 2)
        with VideoWriter(cls.silent, 160, 96, '12/1', crf=10, preset='ultrafast') as writer:
            for _ in range(120):
                writer.write(image)
        # Nonzero video stream index and multiple audio streams exercise actual mapping.
        subprocess.run(['ffmpeg', '-v', 'error', '-nostdin', '-n', '-i', str(cls.silent),
                        '-f', 'lavfi', '-i', 'sine=frequency=440:duration=10',
                        '-f', 'lavfi', '-i', 'sine=frequency=660:duration=10',
                        '-map', '1:a', '-map', '0:v', '-map', '2:a', '-c:v', 'copy',
                        '-c:a', 'aac', '-b:a', '64k', '-shortest', str(cls.audio)],
                       check=True, capture_output=True)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def setUp(self):
        self.folder = self.root/self._testMethodName
        self.folder.mkdir()

    def make_project(self, source=None):
        return projects.create_project(source or self.silent, self.folder/'project')

    def assert_export(self, source, project):
        output = Path(project['artifacts']['export'])
        before, after = probe(source), probe(output)
        for key in ['frame_count', 'fps_fraction', 'width', 'height', 'has_audio']:
            self.assertEqual(before[key], after[key], key)
        verification = json.loads(Path(project['artifacts']['verification']).read_text())
        self.assertTrue(verification['passed'], verification)
        self.assertFalse(verification['visual_perfection_verified'])
        if before['has_audio']:
            self.assertEqual(pipeline._audio_hash(source), pipeline._audio_hash(output))
        return output

    def test_import_checks_actual_audio_first_video_timestamps(self):
        metadata = projects.inspect_source(self.audio)
        self.assertEqual(metadata['video_stream_index'], 1)
        self.assertEqual(metadata['frame_count'], 120)
        self.assertEqual(read_frame(self.audio, 119).shape, (96, 160, 3))

    def test_persistence_and_edit_invalidation(self):
        project = self.make_project()
        project['artifacts'] = {key: str(self.folder/key) for key in
                                ['proxy', 'thumbnails', 'detection', 'plan', 'calibration',
                                 'report', 'fullPreview', 'seamPreviews', 'export', 'verification']}
        projects.save_project(project)
        changed = projects.set_seams(Path(project['projectPath']), [96, 12])
        self.assertEqual(changed['revision'], 1)
        self.assertEqual(set(changed['artifacts']), {'proxy', 'thumbnails', 'detection'})
        self.assertEqual(changed['sourceSha256'], project['sourceSha256'])
        changed['artifacts']['plan'] = 'existing-plan'
        projects.save_project(changed)
        same = projects.set_seams(Path(project['projectPath']), [12, 96])
        self.assertEqual(same['revision'], 1)
        self.assertEqual(same['artifacts']['plan'], 'existing-plan')
        disabled = projects.set_seams(Path(project['projectPath']), [12, {'frame': 96, 'enabled': False}])
        self.assertEqual(disabled['revision'], 2)
        self.assertNotIn('plan', disabled['artifacts'])
        self.assertFalse(projects.load_project(Path(project['projectPath']))['seams'][1]['enabled'])

    def test_changed_source_and_duplicate_project_are_rejected(self):
        source = self.folder/'source.mp4'
        shutil.copyfile(self.silent, source)
        project = self.make_project(source)
        with self.assertRaises(FileExistsError):
            projects.create_project(source, Path(project['projectPath']).parent)
        with source.open('ab') as handle:
            handle.write(b'changed')
        with self.assertRaisesRegex(ValueError, 'source video has changed'):
            projects.load_project(Path(project['projectPath']))
        source.unlink()
        with self.assertRaises(FileNotFoundError):
            projects.load_project(Path(project['projectPath']))

    def test_actual_vfr_source_is_rejected_without_retiming(self):
        irregular = self.folder/'irregular.mp4'
        subprocess.run(['ffmpeg', '-v', 'error', '-nostdin', '-n', '-i', str(self.silent),
                        '-vf', r'setpts=if(lt(N\,60)\,N/(12*TB)\,(60+2*(N-60))/(12*TB))',
                        '-fps_mode', 'vfr', '-c:v', 'libx264', '-preset', 'ultrafast', str(irregular)],
                       check=True, capture_output=True)
        with self.assertRaisesRegex(ValueError, 'Variable or discontinuous frame timing'):
            projects.inspect_source(irregular)

    def test_real_gray16_and_rgb48_sources_are_rejected(self):
        for pixel_format in ['gray16le', 'rgb48be']:
            with self.subTest(pixel_format=pixel_format):
                source = self.folder/(pixel_format+'.mkv')
                # PNG stores packed RGB48; FFV1 stores gray16 without changing its depth.
                codec = 'png' if pixel_format.startswith('rgb48') else 'ffv1'
                subprocess.run(['ffmpeg', '-v', 'error', '-nostdin', '-n', '-i', str(self.silent),
                                '-frames:v', '12', '-pix_fmt', pixel_format, '-c:v', codec, str(source)],
                               check=True, capture_output=True)
                self.assertEqual(probe(source)['pix_fmt'], pixel_format)
                with self.assertRaisesRegex(ValueError, '8-bit SDR'):
                    projects.inspect_source(source)

    def test_anamorphic_source_is_rejected_before_display_shape_changes(self):
        source = self.folder/'anamorphic.mp4'
        subprocess.run(['ffmpeg', '-v', 'error', '-nostdin', '-n', '-i', str(self.silent),
                        '-frames:v', '12', '-vf', 'setsar=2/1', '-c:v', 'libx264', str(source)],
                       check=True, capture_output=True)
        self.assertEqual(probe(source)['sample_aspect_ratio'], '2:1')
        with self.assertRaisesRegex(ValueError, 'Anamorphic'):
            projects.inspect_source(source)

    def test_delayed_video_track_is_rejected_instead_of_desynchronizing_audio(self):
        source = self.folder/'delayed-video.mp4'
        subprocess.run(['ffmpeg', '-v', 'error', '-nostdin', '-n', '-itsoffset', '1',
                        '-i', str(self.silent), '-f', 'lavfi', '-i', 'sine=duration=11',
                        '-map', '0:v', '-map', '1:a', '-c:v', 'copy', '-c:a', 'aac', str(source)],
                       check=True, capture_output=True)
        metadata = probe(source)
        self.assertEqual(metadata['frame_count'], 120)
        self.assertEqual(metadata['fps_fraction'], '12/1')
        self.assertAlmostEqual(metadata['video_start_time'], 1., places=3)
        self.assertAlmostEqual(metadata['audio_start_times'][0], 0., places=3)
        with self.assertRaisesRegex(ValueError, 'start|timestamp|offset'):
            projects.inspect_source(source)

    def test_delayed_audio_track_retains_its_relative_start_and_bytes(self):
        source = self.folder/'delayed-audio.mp4'
        subprocess.run(['ffmpeg', '-v', 'error', '-nostdin', '-n', '-i', str(self.silent),
                        '-itsoffset', '0.5', '-f', 'lavfi', '-i', 'sine=duration=9.5',
                        '-map', '0:v', '-map', '1:a', '-c:v', 'copy', '-c:a', 'aac', str(source)],
                       check=True, capture_output=True)
        metadata = probe(source)
        self.assertAlmostEqual(metadata['video_start_time'], 0., places=3)
        # AAC priming can place the first packet just before the requested half-second.
        self.assertGreater(metadata['audio_start_times'][0], .45)
        self.assertLess(metadata['audio_start_times'][0], .55)
        project = self.make_project(source)
        result = pipeline.run_stage(project['projectPath'], 'export',
            options={'exportPath': str(self.folder/'delayed-audio-corrected.mp4')})
        output = self.assert_export(source, result)
        after = probe(output)
        self.assertAlmostEqual(after['video_start_time'], metadata['video_start_time'], places=4)
        self.assertEqual(len(after['audio_start_times']), len(metadata['audio_start_times']))
        for a, b in zip(after['audio_start_times'], metadata['audio_start_times']):
            self.assertAlmostEqual(a, b, places=4)
        verification = json.loads(Path(result['artifacts']['verification']).read_text())
        self.assertTrue(verification['checks']['audio_start_times_preserved'])
        self.assertTrue(verification['checks']['video_start_time_preserved'])
        project = projects.set_seams(Path(project['projectPath']), [6])
        pipeline.run_stage(project['projectPath'], 'analyze', options={'enable_local_color': False})
        previewed = pipeline.run_stage(project['projectPath'], 'preview',
            options={'previewWidth': 80, 'previewSeconds': .5})
        whole = probe(previewed['artifacts']['fullPreview'])
        self.assertGreater(whole['audio_start_times'][0], .4)
        self.assertAlmostEqual(whole['audio_start_times'][0], metadata['audio_start_times'][0], delta=.03)
        snippet = previewed['artifacts']['seamPreviews'][0]
        self.assertEqual(snippet['startFrame'], 3)
        clip = probe(snippet['path'])
        self.assertEqual(clip['frame_count'], snippet['endFrame']-snippet['startFrame'])
        self.assertEqual(clip['frame_count'], 6)
        self.assertEqual(clip['fps_fraction'], metadata['fps_fraction'])
        # Clip audio comes directly from source, so priming must not accumulate.
        expected = metadata['audio_start_times'][0]-snippet['startFrame']/metadata['fps']
        self.assertGreater(clip['audio_start_times'][0], .15)
        self.assertAlmostEqual(clip['audio_start_times'][0], expected, delta=.03)

    def test_export_verification_rejects_shifted_audio_even_when_bytes_match(self):
        project = self.make_project(self.audio)
        destination = self.folder/'shifted-audio.mp4'
        def changed_timing(path):
            metadata = probe(path)
            if Path(path).resolve() == destination.resolve():
                metadata = {**metadata, 'audio_start_times': [v+.25 for v in metadata['audio_start_times']]}
            return metadata
        # Isolate the verification contract: stream bytes still match, timestamps do not.
        with patch.object(pipeline, 'probe', side_effect=changed_timing), \
                self.assertRaisesRegex(RuntimeError, 'Export verification failed'):
            pipeline.run_stage(project['projectPath'], 'export', options={'exportPath': str(destination)})
        verification = json.loads(destination.with_suffix('.verification.json').read_text())
        self.assertTrue(verification['checks']['audio_streams_unchanged'])
        self.assertFalse(verification['checks']['audio_start_times_preserved'])
        self.assertFalse(verification['passed'])
        self.assertNotIn('export', projects.load_project(Path(project['projectPath']))['artifacts'])

    def test_no_seam_process_preserves_framing_timing_and_both_audio_streams(self):
        project = self.make_project(self.audio)
        events = []
        result = pipeline.run_stage(project['projectPath'], 'process',
            options={'interval_hints': [], 'previewWidth': 80, 'crf': 10,
                     'exportPath': str(self.folder/'corrected.mp4')}, progress=events.append)
        self.assertEqual(result['seams'], [])
        plan = json.loads(Path(result['artifacts']['plan']).read_text())
        self.assertEqual(plan['grade_curves'], [])
        self.assertEqual(plan['local_color_curves'], [])
        np.testing.assert_allclose(np.asarray(plan['view_matrix']), np.eye(3))
        self.assertEqual(result['artifacts']['seamPreviews'], [])
        preview = probe(result['artifacts']['fullPreview'])
        self.assertEqual((preview['width'], preview['height'], preview['frame_count']), (80, 48, 120))
        output = self.assert_export(self.audio, result)
        difference = np.abs(read_frame(output, 60).astype(float)-read_frame(self.audio, 60))
        self.assertLess(float(np.mean(difference)), 2.5)
        self.assertTrue(events)
        self.assertTrue(all(0 <= event.get('fraction', 0) <= 1 for event in events))

    def test_manual_arbitrary_marker_and_edge_marker_survive_analysis_preview_export(self):
        project = self.make_project()
        project = projects.set_seams(Path(project['projectPath']), [1, 96])
        analyzed = pipeline.run_stage(project['projectPath'], 'analyze',
                                      options={'enable_local_color': False, 'previewWidth': 80})
        self.assertEqual([s['frame'] for s in analyzed['seams']], [1, 96])
        self.assertTrue(analyzed['warnings'], 'The edge marker needs an explicit insufficient-handles advisory')
        previewed = pipeline.run_stage(project['projectPath'], 'preview',
                                      options={'previewWidth': 80, 'previewSeconds': 2})
        snippets = previewed['artifacts']['seamPreviews']
        self.assertEqual([s['frame'] for s in snippets], [1, 96])
        for snippet in snippets:
            info = probe(snippet['path'])
            self.assertEqual(info['frame_count'], snippet['endFrame']-snippet['startFrame'])
            self.assertEqual((info['width'], info['height']), (80, 48))
        exported = pipeline.run_stage(project['projectPath'], 'export',
                                     options={'exportPath': str(self.folder/'manual.mp4'), 'crf': 10})
        self.assert_export(self.silent, exported)

    def test_cleared_markers_stay_cleared_and_completed_stages_are_reused(self):
        project = self.make_project()
        project = projects.set_seams(Path(project['projectPath']), [])
        with patch.object(pipeline, 'detect_project', side_effect=AssertionError('Must respect explicit empty list')):
            result = pipeline.run_stage(project['projectPath'], 'process', options={'previewWidth': 80})
        with patch.object(pipeline, 'analyze_project', side_effect=AssertionError('Already analyzed')), \
                patch.object(pipeline, 'preview_project', side_effect=AssertionError('Already previewed')):
            repeated = pipeline.run_stage(project['projectPath'], 'process')
        self.assertEqual(result['artifacts'], repeated['artifacts'])

    def test_cancellation_leaves_saved_project_and_published_artifacts_unchanged(self):
        project = self.make_project()
        before = Path(project['projectPath']).read_bytes()
        with self.assertRaises(InterruptedError):
            pipeline.run_stage(project['projectPath'], 'process', cancelled=lambda: True)
        self.assertEqual(Path(project['projectPath']).read_bytes(), before)
        analyzed = pipeline.run_stage(project['projectPath'], 'analyze')
        before = Path(project['projectPath']).read_bytes()
        state = {'cancel': False}
        def progress(event):
            if event.get('stage') == 'export' and event.get('fraction', 0) < 1:
                state['cancel'] = True
        destination = self.folder/'cancelled.mp4'
        with self.assertRaises(InterruptedError):
            pipeline.run_stage(analyzed['projectPath'], 'export',
                options={'exportPath': str(destination)}, progress=progress, cancelled=lambda: state['cancel'])
        self.assertEqual(Path(project['projectPath']).read_bytes(), before)
        self.assertFalse(destination.exists())

    def test_stale_job_cannot_publish_after_marker_edit(self):
        project = self.make_project()
        projects.set_seams(Path(project['projectPath']), [96])
        with self.assertRaisesRegex(RuntimeError, 'Seams changed'):
            pipeline._commit(project, {'export': 'stale.mp4'}, 'exported')
        self.assertNotIn('export', projects.load_project(Path(project['projectPath']))['artifacts'])

    def test_reanalysis_invalidates_all_old_plan_dependent_artifacts(self):
        project = self.make_project()
        first = pipeline.run_stage(project['projectPath'], 'process', options={'previewWidth': 80})
        old_preview = first['artifacts']['fullPreview']
        first['artifacts'].update({'export': 'previous-export.mp4', 'verification': 'previous-check.json'})
        projects.save_project(first)
        second = pipeline.run_stage(project['projectPath'], 'analyze')
        self.assertEqual(second['revision'], first['revision'])
        self.assertNotEqual(second['artifacts']['plan'], first['artifacts']['plan'])
        for name in ['fullPreview', 'seamPreviews', 'export', 'verification']:
            self.assertNotIn(name, second['artifacts'])
        self.assertTrue(Path(old_preview).is_file(), 'Old files remain available as historical artifacts')
        resumed = pipeline.run_stage(project['projectPath'], 'process', options={'previewWidth': 80})
        self.assertNotEqual(resumed['artifacts']['fullPreview'], old_preview)

    def test_concurrent_reanalysis_cannot_publish_a_preview_of_the_old_plan(self):
        project = self.make_project()
        project = pipeline.run_stage(project['projectPath'], 'analyze')
        updated = {'value': False}
        def change_plan(event):
            if event.get('stage') == 'preview' and not updated['value']:
                updated['value'] = True
                pipeline.run_stage(project['projectPath'], 'analyze')
        with self.assertRaisesRegex(RuntimeError, 'correction plan changed'):
            pipeline.run_stage(project['projectPath'], 'preview', options={'previewWidth': 80},
                               progress=change_plan)
        current = projects.load_project(Path(project['projectPath']))
        self.assertEqual(current['revision'], project['revision'])
        self.assertNotEqual(current['artifacts']['plan'], project['artifacts']['plan'])
        self.assertNotIn('fullPreview', current['artifacts'])

    def test_cli_stages_and_interrupt_exit_code(self):
        project_dir = self.folder/'cli-project'
        def invoke(args):
            stdout, stderr = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                code = cli.main(args)
            self.assertEqual(code, 0, stderr.getvalue())
            return json.loads(stdout.getvalue())
        detected = invoke(['detect', str(self.silent), '--work-dir', str(project_dir), '--intervals', ''])
        self.assertEqual(detected['seams'], [])
        marked = invoke(['mark', '--project', str(project_dir), '--timecodes', '00:00:08:00'])
        self.assertEqual(marked['seams'][0]['frame'], 96)
        invoke(['mark', '--project', str(project_dir), '--clear'])
        invoke(['calibrate', '--project', str(project_dir)])
        previewed = invoke(['preview', '--project', str(project_dir), '--preview-width', '80'])
        self.assertEqual(probe(previewed['artifacts']['fullPreview'])['width'], 80)
        exported = invoke(['export', '--project', str(project_dir), '--output', str(self.folder/'cli.mp4')])
        self.assert_export(self.silent, exported)
        inspected = invoke(['inspect', '--project', str(project_dir)])
        self.assertEqual(inspected['artifacts'], exported['artifacts'])
        with patch.object(pipeline, 'run_stage', side_effect=KeyboardInterrupt()), \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(cli.main(['calibrate', '--project', str(project_dir)]), 130)

    def test_desktop_worker_json_protocol_and_signal_cancellation(self):
        project = self.make_project()
        request = {'operation': 'get', 'args': {'projectPath': project['projectPath']}}
        result = subprocess.run([sys.executable, '-m', 'seamstress.desktop_worker'],
                                input=json.dumps(request)+'\n', capture_output=True, text=True,
                                check=True, timeout=30)
        events = [json.loads(line) for line in result.stdout.splitlines()]
        self.assertEqual([e['type'] for e in events], ['complete'])
        self.assertEqual(events[0]['project']['sourceSha256'], project['sourceSha256'])
        with tempfile.TemporaryFile(mode='w+') as errors:
            child = subprocess.Popen([sys.executable, '-m', 'seamstress.desktop_worker'],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=errors, text=True)
            try:
                request = {'operation': 'create', 'args': {'source': str(self.audio),
                           'folder': str(self.folder/'cancelled-project')}}
                child.stdin.write(json.dumps(request)+'\n')
                child.stdin.flush()
                first = json.loads(child.stdout.readline())
                self.assertEqual(first['type'], 'progress')
                child.send_signal(signal.SIGTERM)
                remaining, _ = child.communicate(timeout=30)
                events = [json.loads(line) for line in remaining.splitlines()]
                self.assertEqual(child.returncode, 130)
                self.assertEqual(events[-1]['type'], 'cancelled')
                self.assertFalse(any(event['type'] == 'complete' for event in events))
            finally:
                if child.poll() is None:
                    child.kill()
                    child.wait()
                child.stdin.close()
                child.stdout.close()


if __name__ == '__main__':
    unittest.main()
