"""CLI one-off refinement uses the shared frozen-plan guarantees."""
import contextlib
import copy
import io
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from seamstress import cli, projects
from seamstress.corrections import DEFAULT_CORRECTION
from seamstress.design import ALGORITHM, build_conform_plan
from seamstress.media import VideoWriter, probe


def manual(shift):
    return {'right_to_left_matrix': [[1, 0, shift], [0, 1, 0], [0, 0, 1]],
            'pre_rate': [0, 0, 0, 0], 'post_rate': [0, 0, 0, 0], 'ease_rate': False}


class RefineParserTests(unittest.TestCase):
    def test_target_is_required_exclusive_and_positive(self):
        for argv in [['refine', '--project', 'p'],
                     ['refine', '--project', 'p', '--frame', '1', '--timecode', '1'],
                     ['refine', '--project', 'p', '--frame', '0'],
                     ['refine', '--project', 'p', '--frame', '1', '--support-frames', '-1']]:
            with self.subTest(argv=argv), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                cli.parser().parse_args(argv)

    def test_selected_preview_frame_is_forwarded(self):
        project={'projectPath':'/project.json'}
        with patch('seamstress.projects.load_project',return_value=project), patch('seamstress.pipeline.run_stage',return_value=project) as run, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(cli.main(['preview','--project','/project.json','--frame','42']),0)
        self.assertEqual(run.call_args.args,('/project.json','preview'))
        self.assertEqual(run.call_args.kwargs['options']['frame'],42)

    def test_conflicting_source_modes_refused_before_load(self):
        variants=[['input.mp4','--project','p'],['--project','p','--base-plan','b'],
                  ['--project','p','--work-dir','w'],[],['input.mp4'],['--base-plan','b']]
        for variant in variants:
            with self.subTest(variant=variant), patch('seamstress.projects.load_project') as load, contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(cli.main(['refine',*variant,'--frame','1']),1)
                load.assert_not_called()


@unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'),'FFmpeg required')
class RefineCliTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.media=tempfile.TemporaryDirectory();cls.source=Path(cls.media.name)/'source.mp4'
        silent=Path(cls.media.name)/'silent.mp4'
        picture=np.random.default_rng(550).integers(25,220,(64,96,3),dtype=np.uint8)
        with VideoWriter(silent,96,64,'12',crf=0,preset='ultrafast') as writer:
            for _ in range(72):writer.write(picture)
        subprocess.run(['ffmpeg','-v','error','-nostdin','-n','-i',str(silent),
            '-f','lavfi','-i','sine=frequency=440:duration=6',
            '-f','lavfi','-i','sine=frequency=660:duration=6',
            '-map','0:v:0','-map','1:a:0','-map','2:a:0','-c:v','copy','-c:a','aac',str(cls.source)],check=True)

    @classmethod
    def tearDownClass(cls):cls.media.cleanup()

    def setUp(self):
        self.directory=tempfile.TemporaryDirectory();self.folder=Path(self.directory.name)
        self.project=projects.create_project(self.source,self.folder/'project');self.path=Path(self.project['projectPath'])
        self.project=projects.set_seams(self.path,[18,54])
        settings=[]
        for row,shift in zip(self.project['seams'],(.8,.5)):
            row['correction']={**DEFAULT_CORRECTION,'geometry':'manual','manual':manual(shift),'color':'off','rate_easing':False}
            settings.append({'frame':row['frame'],'correction':copy.deepcopy(row['correction'])})
        recipe={'schema_version':1,'method':'source_conform_calibration','algorithm':ALGORITHM,
            'source':self.project['metadata'],'source_sha256':self.project['sourceSha256'],
            'parameters':{'geometry_support':4,'rate_support':2,'source_margin_pixels':0,'max_view_crop_fraction_total_dimension':.08},
            'cuts':[{'frame':18,**manual(.8)},{'frame':54,**manual(.5)}],
            'grade_curves':[],'local_color_curves':[],'excluded_geometry':[],'correction_settings':settings}
        self.baseline=build_conform_plan(recipe,self.project['metadata'])
        self.plan=self.folder/'accepted.plan.json';projects.atomic_json(self.plan,self.baseline)
        self.project['artifacts']['plan']=str(self.plan);projects.save_project(self.project)

    def tearDown(self):self.directory.cleanup()

    def invoke(self,args):
        out,err=io.StringIO(),io.StringIO()
        with contextlib.redirect_stdout(out),contextlib.redirect_stderr(err):code=cli.main(args)
        return code,json.loads(out.getvalue()) if out.getvalue().strip() else None,err.getvalue()

    def test_standalone_inherits_reviewed_settings_and_exports_full_video(self):
        folder=self.folder/'standalone';output=self.folder/'refined.mp4';before=self.plan.read_bytes()
        code,result,error=self.invoke(['refine',str(self.source),'--base-plan',str(self.plan),
            '--timecode','00:04.500','--work-dir',str(folder),'--support-frames','4','--output',str(output),'--crf','18'])
        self.assertEqual(code,0,error)
        plan=json.loads(Path(result['plan_path']).read_text())
        measured=json.loads(Path(result['calibration_path']).read_text())
        self.assertEqual(measured['cuts'][0]['right_to_left_matrix'],manual(.5)['right_to_left_matrix'])
        self.assertEqual(measured['correction_settings'][0]['correction']['geometry'],'manual')
        self.assertEqual(plan['view_matrix'],self.baseline['view_matrix'])
        self.assertEqual(plan['frame_matrices'][:49],self.baseline['frame_matrices'][:49])
        self.assertEqual(plan['frame_matrices'][59:],self.baseline['frame_matrices'][59:])
        self.assertEqual(self.plan.read_bytes(),before)
        self.assertEqual(probe(output)['frame_count'],72)
        verification=json.loads(Path(result['verification_path']).read_text())
        self.assertTrue(verification['passed'])
        self.assertEqual(len(verification['output']['audio_start_times']),2)
        self.assertTrue(verification['checks']['audio_streams_unchanged'])

    def test_project_override_changes_only_selected_settings(self):
        correction=self.folder/'correction.json';projects.atomic_json(correction,{'geometry':'off','color':'off'})
        old=copy.deepcopy(self.project['seams'][0])
        code,result,error=self.invoke(['refine','--project',str(self.path),'--frame','54','--support-frames','4','--correction',str(correction)])
        self.assertEqual(code,0,error)
        self.assertEqual(result['seams'][0],old)
        self.assertEqual(result['seams'][1]['correction']['geometry'],'off')
        self.assertTrue(result['refinementBaseline'])
        plan=json.loads(Path(result['artifacts']['plan']).read_text())
        self.assertEqual(plan['frame_matrices'][:49],self.baseline['frame_matrices'][:49])
        self.assertEqual(plan['frame_matrices'][59:],self.baseline['frame_matrices'][59:])
        self.assertEqual(plan['view_matrix'],self.baseline['view_matrix'])

    def test_existing_output_or_sidecar_refused_before_settings_or_analysis(self):
        correction=self.folder/'correction.json';projects.atomic_json(correction,{'geometry':'off'})
        for ending in ('.mp4','.repair.json','.verification.json'):
            output=self.folder/f'protected{ending}';output.write_text('keep')
            destination=self.folder/'protected.mp4';before=self.path.read_bytes()
            with patch('seamstress.pipeline.run_stage') as run:
                code,_,error=self.invoke(['refine','--project',str(self.path),'--frame','54',
                    '--correction',str(correction),'--output',str(destination)])
            self.assertEqual(code,1,error);run.assert_not_called()
            self.assertEqual(self.path.read_bytes(),before);self.assertEqual(output.read_text(),'keep')
            output.unlink()

    def test_existing_standalone_folder_and_invalid_correction_never_mutate_project(self):
        original=self.path.read_bytes()
        code,_,error=self.invoke(['refine',str(self.source),'--base-plan',str(self.plan),'--frame','54','--work-dir',str(self.folder)])
        self.assertEqual(code,1);self.assertIn('new --work-dir',error)
        correction=self.folder/'bad.json';correction.write_text('{"geometry":"force"}')
        code,_,error=self.invoke(['refine','--project',str(self.path),'--frame','54','--correction',str(correction)])
        self.assertEqual(code,1);self.assertIn('geometry must be',error)
        self.assertEqual(self.path.read_bytes(),original)

    def test_standalone_validates_cfr_before_any_artifact_creation(self):
        folder=self.folder/'invalid-timing'
        with patch('seamstress.projects.inspect_source',side_effect=ValueError('Variable timing')) as inspect:
            code,_,error=self.invoke(['refine',str(self.source),'--base-plan',str(self.plan),
                '--frame','54','--work-dir',str(folder)])
        self.assertEqual(code,1);self.assertIn('Variable timing',error);inspect.assert_called_once()
        self.assertFalse(folder.exists())


if __name__=='__main__':unittest.main()
