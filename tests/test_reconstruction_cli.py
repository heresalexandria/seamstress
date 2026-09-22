"""CLI opt-in, stage routing and a real reconstruction review/export round trip."""
import contextlib
import copy
import io
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from seamstress import cli, projects
from seamstress.media import probe
from tests import test_reconstruction_workflow as workflow_fixture


def invoke(argv):
    output, errors = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(output), contextlib.redirect_stderr(errors):
        status = cli.main(argv)
    return status, json.loads(output.getvalue()) if output.getvalue().strip() else None, errors.getvalue()


class ReconstructionCliRoutingTests(unittest.TestCase):
    def setUp(self):
        self.project = {'projectPath': '/test/project.json', 'source': '/test/source.mp4',
                        'metadata': {'fps_fraction': '24/1'}, 'seams': [{'frame': 48, 'enabled': True}]}

    def test_mutually_exclusive_targets_and_positive_frame_arguments(self):
        for argv in [
            ['--frame', '48', '--timecode', '2'], ['--all-seams', '--frame', '48'],
            ['--frame', '0'], ['--reach-frames', '-1'], ['--max-ai-requests', '0']]:
            with self.subTest(argv=argv), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                cli.parser().parse_args(['reconstruct', '--project', '/project.json', *argv])

    def test_default_resume_does_not_enable_reconstruction_or_expose_environment_key(self):
        with patch('seamstress.projects.load_project', return_value=self.project), \
             patch('seamstress.pipeline.run_stage', return_value=self.project) as run, \
             patch.dict(os.environ, {'OPENAI_API_KEY': 'test-placeholder-never-upload'}):
            code, _, error = invoke(['resume', '--project', '/test/project.json'])
        self.assertEqual(code, 0, error)
        self.assertEqual(run.call_args.args, ('/test/project.json', 'process'))
        self.assertNotIn('reconstructionEnabled', run.call_args.kwargs['options'])
        self.assertIsNone(run.call_args.kwargs['provider_key'])

    def test_opt_in_workflow_and_cloud_budget_are_forwarded_only_when_requested(self):
        for cloud in (False, True):
            with self.subTest(cloud=cloud), patch('seamstress.projects.load_project', return_value=self.project), \
                 patch('seamstress.pipeline.run_stage', return_value=self.project) as run, \
                 patch.dict(os.environ, {'OPENAI_API_KEY': 'test-placeholder-never-upload'}):
                code, _, error = invoke(['resume', '--project', '/test/project.json', '--reconstruct',
                                        '--max-ai-requests', '3', *(['--allow-ai'] if cloud else [])])
            self.assertEqual(code, 0, error)
            self.assertTrue(run.call_args.kwargs['options']['reconstructionEnabled'])
            self.assertEqual(run.call_args.kwargs['options']['reconstruction'], {'allowAI': cloud, 'maxAIRequests': 3})
            self.assertEqual(run.call_args.kwargs['provider_key'], 'test-placeholder-never-upload' if cloud else None)

    def test_illegal_cloud_and_source_combinations_refuse_before_work(self):
        variants = [
            ['reconstruct', 'input.mp4', '--project', 'p'],
            ['reconstruct', '--project', 'p', '--base-plan', 'b'],
            ['reconstruct', '--project', 'p', '--work-dir', 'w'],
            ['reconstruct'],
            ['reconstruct', '--project', 'p', '--stage', 'render', '--frame', '48', '--allow-ai'],
            ['reconstruct', '--project', 'p', '--stage', 'edit', '--all-seams'],
            ['reconstruct', '--project', 'p', '--stage', 'import', '--frame', '48'],
            ['resume', '--project', 'p', '--allow-ai'],
        ]
        for argv in variants:
            with self.subTest(argv=argv), patch('seamstress.projects.load_project', return_value=self.project), \
                 patch('seamstress.pipeline.run_stage') as run:
                code, _, error = invoke(argv)
            self.assertEqual(code, 1, error); run.assert_not_called()

    def test_edit_does_not_receive_proposal_only_segmentation_option(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'edits.json'
            edits = {'strokes': [{'frame': 48, 'layer_id': 'foreground', 'mode': 'include', 'radius': 2, 'points': [[10, 12]]}]}
            path.write_text(json.dumps(edits))
            with patch('seamstress.projects.load_project', return_value=self.project), \
                 patch('seamstress.pipeline.run_stage', return_value=self.project) as run:
                code, _, error = invoke(['reconstruct', '--project', 'p', '--timecode', '00:02', '--stage', 'edit', '--edits', str(path)])
            self.assertEqual(code, 0, error)
            self.assertEqual(run.call_args.kwargs['options'], {'frame': 48, 'action': 'edit', 'reconstruction': edits})
            self.assertIsNone(run.call_args.kwargs['provider_key'])

    def test_local_model_install_is_explicit_and_prompted_segment_is_forwarded(self):
        with patch('seamstress.segmentation_model.setup_model', return_value={'available': True}) as install:
            code, _, error = invoke(['setup-segmentation-model', '--download'])
        self.assertEqual(code, 0, error)
        self.assertTrue(install.call_args.kwargs['allow_download'])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'points.json'; prompt = {'neuralPrompt': {'frame': 48, 'layerId': 'foreground', 'points': [{'x': 30, 'y': 20, 'label': 1}]}}
            path.write_text(json.dumps(prompt))
            with patch('seamstress.projects.load_project', return_value=self.project), \
                 patch('seamstress.pipeline.run_stage', return_value=self.project) as run:
                code, _, error = invoke(['reconstruct', '--project', 'p', '--frame', '48', '--stage', 'segment', '--edits', str(path)])
            self.assertEqual(code, 0, error)
            self.assertEqual(run.call_args.kwargs['options']['reconstruction'], prompt)


@unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'), 'FFmpeg required')
class ReconstructionCliRoundTripTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        workflow_fixture.ReconstructionWorkflowTests.setUpClass()

    @classmethod
    def tearDownClass(cls):
        workflow_fixture.ReconstructionWorkflowTests.tearDownClass()

    def setUp(self):
        self.fixture = workflow_fixture.ReconstructionWorkflowTests()
        self.fixture.setUp()

    def tearDown(self):
        self.fixture.tearDown()

    def test_import_edit_render_explicit_accept_and_full_export(self):
        fixture = self.fixture; manifest = fixture.authored(36)
        command = ['reconstruct', '--project', str(fixture.path), '--timecode', '00:03']
        code, project, error = invoke([*command, '--stage', 'import', '--bundle', str(manifest)])
        self.assertEqual(code, 0, error)
        self.assertIn('candidate', project['reconstructions']['36'])
        edits = fixture.root/'edits.json'
        edits.write_text(json.dumps({'layer_matrices': [{'frame': 36, 'layer_id': 'foreground',
                      'matrix': [[1, 0, .75], [0, 1, 0], [0, 0, 1]]}]}))
        code, project, error = invoke([*command, '--stage', 'edit', '--edits', str(edits)])
        self.assertEqual(code, 0, error)
        self.assertEqual(project['artifacts']['plan'], str(fixture.plan_path))
        code, project, error = invoke([*command, '--stage', 'render'])
        self.assertEqual(code, 0, error)
        self.assertTrue(Path(project['reconstructions']['36']['candidate']['candidatePreviewPath']).is_file())
        before = fixture.path.read_bytes()
        code, _, error = invoke([*command, '--stage', 'accept'])
        self.assertEqual(code, 1); self.assertIn('confirm acceptance', error)
        self.assertEqual(fixture.path.read_bytes(), before)
        destination = fixture.root/'final.mp4'
        code, project, error = invoke([*command, '--stage', 'accept', '--reviewed', '--output', str(destination), '--crf', '18'])
        self.assertEqual(code, 0, error)
        self.assertEqual(probe(destination)['frame_count'], 48)
        self.assertIn('accepted', project['reconstructions']['36'])
        self.assertTrue(json.loads(destination.with_suffix('.verification.json').read_text())['passed'])
        self.assertEqual(fixture.plan_path.read_bytes(), fixture.baseline_bytes)

    def test_existing_movie_sidecars_or_source_refuse_before_reconstruction_mutates_project(self):
        fixture = self.fixture
        for suffix in ('.mp4', '.repair.json', '.verification.json'):
            with self.subTest(suffix=suffix):
                protected = fixture.root/f'protected{suffix}'; protected.write_text('keep this')
                before = fixture.path.read_bytes()
                with patch('seamstress.pipeline.run_stage') as run:
                    code, _, error = invoke(['reconstruct', '--project', str(fixture.path), '--frame', '36',
                        '--stage', 'propose', '--output', str(fixture.root/'protected.mp4')])
                self.assertEqual(code, 1, error); run.assert_not_called()
                self.assertEqual(fixture.path.read_bytes(), before)
                self.assertEqual(protected.read_text(), 'keep this'); protected.unlink()
        with patch('seamstress.pipeline.run_stage') as run:
            code, _, error = invoke(['reconstruct', '--project', str(fixture.path), '--frame', '36', '--output', str(fixture.source)])
        self.assertEqual(code, 1, error); run.assert_not_called()

    def test_pending_candidate_is_not_exported_without_acceptance(self):
        fixture = self.fixture; manifest = fixture.authored(36)
        destination = fixture.root/'must-not-export.mp4'
        code, _, error = invoke(['reconstruct', '--project', str(fixture.path), '--frame', '36', '--stage', 'import',
                                 '--bundle', str(manifest), '--output', str(destination)])
        self.assertEqual(code, 1); self.assertIn('Candidate needs review', error)
        self.assertFalse(destination.exists())
        saved = projects.load_project(fixture.path)
        self.assertIn('candidate', saved['reconstructions']['36'])
        self.assertEqual(saved['artifacts']['plan'], str(fixture.plan_path))
