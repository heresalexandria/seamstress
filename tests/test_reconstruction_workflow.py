"""Project-level reconstruction transactions preserve accepted continuity work."""
import copy
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from seamstress import pipeline, projects, reconstruction as engine
from seamstress.design import ALGORITHM, build_conform_plan
from seamstress.media import VideoWriter, probe, read_frames
from seamstress.reconstruction.bundle import sha256
from seamstress.reconstruction_render import FrameReconstruction
from seamstress.reconstruction_workflow import reconstruct_project


@unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'), 'FFmpeg required')
class ReconstructionWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.media = tempfile.TemporaryDirectory(prefix='seamstress-reconstruction-workflow-')
        cls.source = Path(cls.media.name)/'source.mp4'
        rng = np.random.default_rng(315)
        plate = rng.integers(30, 190, (64, 96, 3), dtype=np.uint8)
        with VideoWriter(cls.source, 96, 64, '12', crf=0, preset='ultrafast') as writer:
            for n in range(48):
                rgb = plate.copy(); rgb[20:36, 24+n%8:36+n%8] = [235, 45, 120]
                writer.write(rgb)
        cls.frames = read_frames(cls.source, 0, 48)

    @classmethod
    def tearDownClass(cls):
        cls.media.cleanup()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.project = projects.create_project(self.source, self.root/'project')
        self.path = Path(self.project['projectPath'])
        self.project = projects.set_seams(self.path, [12, 36])
        calibration = {
            'schema_version': 1, 'method': 'source_conform_calibration', 'algorithm': ALGORITHM,
            'source': self.project['metadata'], 'source_sha256': self.project['sourceSha256'],
            'parameters': {'geometry_support': 3, 'rate_support': 2, 'source_margin_pixels': 0,
                           'max_view_crop_fraction_total_dimension': .08},
            'cuts': [{'frame': n, 'right_to_left_matrix': [[1, 0, .3], [0, 1, 0], [0, 0, 1]],
                      'pre_rate': [0]*4, 'post_rate': [0]*4, 'ease_rate': False} for n in (12, 36)],
            'excluded_geometry': [], 'grade_curves': [], 'local_color_curves': []}
        self.baseline = build_conform_plan(calibration, self.project['metadata'])
        self.plan_path = self.root/'accepted.plan.json'
        projects.atomic_json(self.plan_path, self.baseline)
        self.project['artifacts']['plan'] = str(self.plan_path)
        self.project['artifacts']['seamPreviews'] = [
            {'frame': n, 'path': str(self.source), 'startFrame': n-3, 'endFrame': n+4} for n in (12, 36)]
        self.project = projects.save_project(self.project)
        self.baseline_bytes = self.plan_path.read_bytes()

    def tearDown(self):
        self.temp.cleanup()

    def run_action(self, action, frame=36, **settings):
        self.project = pipeline.run_stage(self.path, 'reconstruct', options={
            'frame': frame, 'action': action, 'reconstruction': settings, 'previewWidth': 96})
        return self.project

    def authored(self, frame, name='authored'):
        assets = self.root/f'{name}-assets'; assets.mkdir()
        layers = [{'id': 'background', 'name': 'Background', 'kind': 'background', 'order': 0,
                   'mask_by_frame': {}, 'matrices': {}, 'keyframes': [frame]},
                  {'id': 'foreground', 'name': 'Subject', 'kind': 'foreground', 'order': 1,
                   'mask_by_frame': {}, 'matrices': {}, 'keyframes': [frame]}]
        rows = {}
        for n in range(frame-3, frame+4):
            rgb = self.frames[n]
            alpha = np.zeros((64, 96), np.float32); alpha[20:36, 24+n%8:36+n%8] = 1
            plate = rgb.copy(); plate[alpha > 0] = [80, 110, 140]
            source, background, matte = assets/f'source-{n}.png', assets/f'plate-{n}.png', assets/f'matte-{n}.npz'
            Image.fromarray(rgb).save(source); Image.fromarray(plate).save(background)
            np.savez_compressed(matte, alpha=alpha, premultiplied=rgb.astype(np.float32)*alpha[:, :, None],
                                emission=np.zeros_like(rgb, dtype=np.float32), background_sha256=np.asarray(sha256(background)))
            rows[n] = {'source': source, 'plate': background, 'matte': matte}
            for layer, mask in zip(layers, [np.ones_like(alpha), alpha]):
                image = assets/f'{layer["id"]}-{n}.png'; Image.fromarray(np.rint(mask*255).astype(np.uint8)).save(image)
                layer['mask_by_frame'][str(n)] = image
                matrix = np.eye(3)
                if n == frame:
                    matrix = np.array([[1.06, 0, -2.88], [0, 1.06, -1.92], [0, 0, 1]]) if layer['kind'] == 'background' else np.array([[1, 0, 1], [0, 1, 0], [0, 0, 1]])
                layer['matrices'][str(n)] = matrix.tolist()
        return engine.import_authored(self.source, frame, self.root/name, support=(frame-3, frame+4),
                                      frames=rows, layers=layers, baseline_plan=self.plan_path)

    def import_candidate(self, frame=36, name='authored'):
        manifest = self.authored(frame, name)
        self.run_action('import', frame, manifestPath=str(manifest))
        return manifest

    def accept(self, frame):
        self.run_action('render', frame)
        return self.run_action('accept', frame, review_approved=True)

    def test_actual_proposal_and_mask_edit_keep_accepted_plan_and_default_network_off(self):
        with patch('seamstress.reconstruction_cloud.generate_background_anchor', side_effect=AssertionError('No upload consent')):
            self.run_action('propose', reachFrames=3, motionStrength=.4, segmentation='classic')
            before = copy.deepcopy(self.project['reconstructions']['36']['candidate'])
            self.assertEqual((before['reachFrames'], before['motionStrength']), (3, .4))
            self.assertNotIn('accepted', self.project['reconstructions']['36'])
            self.run_action('edit', strokes=[{'frame': 36, 'layer_id': 'foreground',
                            'mode': 'protect', 'radius': 2, 'points': [[30, 24], [31, 24]]}])
        record = self.project['reconstructions']['36']
        self.assertNotEqual(record['candidate']['manifestPath'], before['manifestPath'])
        self.assertEqual(record['history'][0]['manifestPath'], before['manifestPath'])
        self.assertEqual(self.project['artifacts']['plan'], str(self.plan_path))
        self.assertEqual(self.plan_path.read_bytes(), self.baseline_bytes)
        self.assertTrue(Path(before['manifestPath']).is_file())

    def test_accept_edit_reject_and_revert_preserve_other_repairs_and_baseline(self):
        self.import_candidate(12, 'first'); self.accept(12)
        first = copy.deepcopy(self.project['reconstructions']['12']['accepted'])
        plan = json.loads(Path(self.project['artifacts']['plan']).read_text())
        first_entry = copy.deepcopy(plan['reconstructions'][0])
        self.import_candidate(36, 'second'); self.accept(36)
        self.assertEqual(self.project['reconstructions']['12']['accepted'], first)
        plan = json.loads(Path(self.project['artifacts']['plan']).read_text())
        self.assertEqual(plan['reconstructions'][0], first_entry)
        self.assertEqual({k: v for k, v in plan.items() if k != 'reconstructions'}, self.baseline)
        accepted_plan = self.project['artifacts']['plan']; accepted_bytes = Path(accepted_plan).read_bytes()
        self.run_action('edit', strokes=[{'frame': 36, 'layer_id': 'foreground', 'mode': 'protect',
                                          'radius': 1, 'points': [[30, 24]]}])
        self.assertIn('accepted', self.project['reconstructions']['36'])
        self.assertIn('candidate', self.project['reconstructions']['36'])
        self.assertEqual(self.project['artifacts']['plan'], accepted_plan)
        self.assertEqual(Path(accepted_plan).read_bytes(), accepted_bytes)
        self.run_action('reject')
        self.assertNotIn('candidate', self.project['reconstructions']['36'])
        self.run_action('revert')
        reverted = json.loads(Path(self.project['artifacts']['plan']).read_text())
        self.assertEqual(reverted['reconstructions'], [first_entry])
        self.assertEqual({k: v for k, v in reverted.items() if k != 'reconstructions'}, self.baseline)
        self.assertNotIn('accepted', self.project['reconstructions']['36'])
        self.assertTrue(self.project['reconstructions']['36']['history'])
        self.assertEqual(self.plan_path.read_bytes(), self.baseline_bytes)

    def test_actual_candidate_preview_has_correct_range_and_acceptance_requires_review(self):
        self.import_candidate()
        before = self.path.read_bytes()
        for settings in ({}, {'review_approved': True}):
            with self.assertRaisesRegex(ValueError, 'Review|preview'):
                self.run_action('accept', **settings)
            self.assertEqual(self.path.read_bytes(), before)
        self.run_action('render')
        candidate = self.project['reconstructions']['36']['candidate']
        self.assertEqual((candidate['startFrame'], candidate['endFrame']), (33, 40))
        self.assertEqual((candidate['previewStartFrame'], candidate['previewEndFrame']), (21, 48))
        self.assertEqual(probe(candidate['candidatePreviewPath'])['frame_count'], 27)
        self.assertEqual(self.project['artifacts']['plan'], str(self.plan_path))
        self.run_action('accept', review_approved=True)
        self.assertNotIn('candidate', self.project['reconstructions']['36'])
        self.assertEqual(self.project['artifacts']['seamPreviews'][0]['frame'], 12)
        self.assertEqual(len(self.project['artifacts']['seamPreviews']), 1)

    def test_cancellation_and_stale_reconstruction_never_publish(self):
        before = self.path.read_bytes()
        with self.assertRaises(InterruptedError):
            pipeline.run_stage(self.path, 'reconstruct', options={'frame': 36, 'action': 'propose'}, cancelled=lambda: True)
        self.assertEqual(self.path.read_bytes(), before)
        real = engine.propose
        def newer_job(*args, **kwargs):
            result = real(*args, **kwargs)
            current = projects.load_project(self.path); current['reconstructionRevision'] = 99
            projects.save_project(current)
            return result
        with patch.object(engine, 'propose', side_effect=newer_job), self.assertRaisesRegex(RuntimeError, 'newer reconstruction'):
            self.run_action('propose', reachFrames=3, segmentation='classic')
        current = projects.load_project(self.path)
        self.assertEqual(current['reconstructionRevision'], 99)
        self.assertNotIn('reconstructions', current)
        self.assertEqual(current['artifacts']['plan'], str(self.plan_path))

    def test_changed_marker_during_work_cannot_be_applied(self):
        real = engine.propose
        def move_marker(*args, **kwargs):
            result = real(*args, **kwargs)
            projects.set_seams(self.path, [12, 35])
            return result
        with patch.object(engine, 'propose', side_effect=move_marker), self.assertRaisesRegex(RuntimeError, 'settings changed'):
            self.run_action('propose', reachFrames=3, segmentation='classic')
        current = projects.load_project(self.path)
        self.assertEqual([row['frame'] for row in current['seams']], [12, 35])
        self.assertNotIn('reconstructions', current)

    def test_target_context_and_manifest_tampering_fail_closed(self):
        self.import_candidate()
        old = self.project['artifacts']['plan']
        changed = copy.deepcopy(self.baseline); changed['frame_matrices'][36][0][2] += .1
        new_plan = self.root/'new-context.plan.json'; projects.atomic_json(new_plan, changed)
        self.project['artifacts']['plan'] = str(new_plan); self.project = projects.save_project(self.project)
        before = self.path.read_bytes()
        with self.assertRaisesRegex(ValueError, 'framing or grade changed'):
            self.run_action('render')
        self.assertEqual(self.path.read_bytes(), before)
        self.project['artifacts']['plan'] = old; self.project = projects.save_project(self.project)
        candidate = self.project['reconstructions']['36']['candidate']
        manifest = Path(candidate['manifestPath']); manifest.write_bytes(manifest.read_bytes()+b' ')
        with self.assertRaisesRegex(ValueError, 'changed outside'):
            self.run_action('render')
        self.assertEqual(self.project['artifacts']['plan'], old)

    def test_other_source_and_wrong_target_import_cannot_change_project(self):
        manifest = self.authored(12)
        before = self.path.read_bytes()
        with self.assertRaisesRegex(ValueError, 'different seam'):
            self.run_action('import', manifestPath=str(manifest))
        self.assertEqual(self.path.read_bytes(), before)
        wrong = copy.deepcopy(self.project); wrong['sourceSha256'] = '0'*64
        with self.assertRaisesRegex(ValueError, 'different source'):
            reconstruct_project(wrong, options={'frame': 36, 'action': 'propose'})
        self.assertEqual(self.path.read_bytes(), before)

    def test_native_provider_preserves_all_non_target_source_pixels(self):
        self.import_candidate(); self.accept(36)
        recipe = json.loads(Path(self.project['artifacts']['plan']).read_text())
        provider = FrameReconstruction(self.source, recipe, self.project['metadata'])
        try:
            for n in range(48):
                result = provider.apply(n, self.frames[n])
                if n != 36:
                    np.testing.assert_array_equal(result, self.frames[n])
            self.assertFalse(np.array_equal(provider.apply(36, self.frames[36]), self.frames[36]))
        finally:
            provider.close()
        accepted = Path(self.project['reconstructions']['36']['accepted']['manifestPath'])
        accepted.write_bytes(accepted.read_bytes()+b' ')
        with self.assertRaisesRegex(ValueError, 'manifest changed'):
            FrameReconstruction(self.source, recipe, self.project['metadata'])

    def test_revert_after_unrelated_marker_edit_uses_frozen_plan_without_reanalysis(self):
        self.import_candidate(12, 'first'); self.accept(12)
        self.import_candidate(36, 'second'); self.accept(36)
        accepted = copy.deepcopy(self.project['reconstructions']['12']['accepted'])
        self.run_action('edit', strokes=[{'frame': 36, 'layer_id': 'foreground', 'mode': 'include',
                                          'radius': 1, 'points': [[30, 24]]}])
        self.project = projects.set_seams(self.path, [12, 24, 36])
        self.assertNotIn('plan', self.project['artifacts'])
        with patch.object(pipeline, 'analyze_project', side_effect=AssertionError('Revert must not analyze')):
            self.run_action('reject')
            self.run_action('revert')
        self.assertNotIn('plan', self.project['artifacts'])
        frozen = self.project['refinementBaseline']
        recipe = json.loads(Path(frozen['plan']).read_text())
        self.assertEqual([row['frame'] for row in recipe['reconstructions']], [12])
        self.assertEqual(self.project['reconstructions']['12']['accepted'], accepted)
        self.assertEqual([row['frame'] for row in self.project['seams']], [12, 24, 36])
        self.assertNotIn('accepted', self.project['reconstructions']['36'])
        self.assertEqual({key: value for key, value in recipe.items() if key != 'reconstructions'}, self.baseline)

    def test_new_marker_inside_accepted_support_is_refused_without_data_loss(self):
        self.import_candidate(); self.accept(36)
        before = self.path.read_bytes()
        with self.assertRaisesRegex(ValueError, 'accepted|reconstruction|support'):
            projects.set_seams(self.path, [12, 35, 36])
        self.assertEqual(self.path.read_bytes(), before)

    def test_source_only_background_step_never_uses_provider_even_when_key_is_present(self):
        self.import_candidate()
        old = self.project['reconstructions']['36']['candidate']['manifestPath']
        with patch('seamstress.reconstruction_cloud.generate_background_anchor', side_effect=AssertionError('Source-only must not upload')):
            result = pipeline.run_stage(self.path, 'reconstruct', provider_key='test-key-do-not-use',
                options={'frame': 36, 'action': 'background', 'reconstruction': {'allowAI': False}})
        new = result['reconstructions']['36']['candidate']['manifestPath']
        self.assertNotEqual(new, old)
        self.assertEqual(result['artifacts']['plan'], str(self.plan_path))
        self.assertNotIn('background_generation', engine.load_bundle(new)['provenance'])

    def test_candidate_publish_merges_an_unrelated_preview_completed_during_work(self):
        real = engine.propose
        preview = {'frame': 12, 'path': str(self.root/'new-other-preview.mp4'), 'startFrame': 6, 'endFrame': 18}
        def preview_completes(*args, **kwargs):
            result = real(*args, **kwargs)
            current = projects.load_project(self.path)
            current['artifacts']['seamPreviews'] = [preview]
            current['artifacts']['fullPreview'] = str(self.root/'new-whole-preview.mp4')
            projects.save_project(current)
            return result
        with patch.object(engine, 'propose', side_effect=preview_completes):
            self.run_action('propose', reachFrames=3, segmentation='classic')
        self.assertEqual(self.project['artifacts']['seamPreviews'], [preview])
        self.assertEqual(self.project['artifacts']['fullPreview'], str(self.root/'new-whole-preview.mp4'))

    def test_old_analysis_commit_cannot_discard_a_new_reconstruction(self):
        old = copy.deepcopy(self.project)
        self.import_candidate(); self.accept(36)
        before = self.path.read_bytes()
        with self.assertRaisesRegex(RuntimeError, '[Rr]econstruction'):
            pipeline._commit(old, {'plan': str(self.plan_path)}, 'analyzed')
        self.assertEqual(self.path.read_bytes(), before)

    def test_automatic_attempt_abstains_at_one_unsupported_seam_and_continues(self):
        real = engine.propose; attempted = []
        def one_unsupported(source, frame, *args, **kwargs):
            attempted.append(frame)
            if frame == 12:
                raise ValueError('Independent layers could not be verified')
            return real(source, frame, *args, **kwargs)
        with patch.object(engine, 'propose', side_effect=one_unsupported):
            result = pipeline.run_stage(self.path, 'reconstruct', options={
                'action': 'auto', 'reconstruction': {'segmentation': 'classic', 'reachFrames': 3}})
        self.assertEqual(attempted, [12, 36])
        self.assertTrue(any('Frame 12' in warning for warning in result['warnings']))
        self.assertTrue(result['reconstructions']['36'].get('candidate') or result['reconstructions']['36'].get('accepted'))
        self.assertEqual(self.plan_path.read_bytes(), self.baseline_bytes)

    def test_default_proposal_reach_is_bounded_by_the_next_marker(self):
        # An identity treatment at an extra marker is a valid saved baseline;
        # its marker still constrains a new repair even without camera change.
        self.project['seams'] = projects.normalized_seams([12, 18, 36], self.project['metadata'], self.project['sourceSha256'])
        self.project = projects.save_project(self.project)
        self.run_action('propose', 12, segmentation='classic')
        candidate = self.project['reconstructions']['12']['candidate']
        self.assertEqual(candidate['reachFrames'], 5)
        self.assertEqual((candidate['startFrame'], candidate['endFrame']), (7, 18))
