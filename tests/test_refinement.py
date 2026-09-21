"""A single-seam job must keep its accepted full-shot context exactly."""
import copy
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from seamstress import pipeline, projects
from seamstress.corrections import DEFAULT_CORRECTION
from seamstress.design import ALGORITHM, build_conform_plan
from seamstress.media import VideoWriter, probe
from seamstress.refinement import (RefinementError, merge_refinement_plan,
                                  refinement_window, refine_video)


def calibration(metadata, digest, cuts):
    return {'schema_version': 1, 'method': 'source_conform_calibration', 'algorithm': ALGORITHM,
            'source': metadata, 'source_sha256': digest,
            'parameters': {'geometry_support': 4, 'rate_support': 2,
                           'source_margin_pixels': 0, 'max_view_crop_fraction_total_dimension': .08},
            'cuts': [{'frame': frame, 'right_to_left_matrix': [[1, 0, shift], [0, 1, 0], [0, 0, 1]],
                      'pre_rate': [0, 0, 0, 0], 'post_rate': [0, 0, 0, 0], 'ease_rate': False}
                     for frame, shift in cuts], 'excluded_geometry': [], 'grade_curves': [], 'local_color_curves': []}


class RefinementPlanTests(unittest.TestCase):
    def setUp(self):
        self.meta={'width':96,'height':64,'frame_count':72,'fps_fraction':'12/1','fps':12}
        self.base=build_conform_plan(calibration(self.meta,'1'*64,[(18,.8),(54,.5)]),self.meta)
        self.next=build_conform_plan(calibration(self.meta,'1'*64,[(54,.3)]),self.meta)

    def test_exact_other_geometry_view_and_curves_are_retained(self):
        window=refinement_window(self.base,54,self.meta,[18,54],4)
        self.assertEqual(window[:3],(49,59,4))
        merged=merge_refinement_plan(self.base,self.next,self.meta,frame=54,start=49,end=59,baseline_sha256='2'*64)
        for key in ('view_matrix','segments','edge_extension_pixels'):
            self.assertEqual(merged[key],self.base[key])
        self.assertEqual(merged['frame_matrices'][:49],self.base['frame_matrices'][:49])
        self.assertEqual(merged['frame_matrices'][59:],self.base['frame_matrices'][59:])
        self.assertNotEqual(merged['frame_matrices'][49:59],self.base['frame_matrices'][49:59])
        self.assertTrue(merged['refinement']['outside_window_identical'])
        self.assertEqual(merged['design_report']['constant_view_crop_fraction_total_dimension'],self.base['design_report']['constant_view_crop_fraction_total_dimension'])

    def test_omitted_identity_view_is_preserved_for_valid_baselines(self):
        baseline=build_conform_plan(calibration(self.meta,'1'*64,[(18,0),(54,0)]),self.meta)
        baseline.pop('view_matrix')
        candidate=build_conform_plan(calibration(self.meta,'1'*64,[(54,0)]),self.meta)
        merged=merge_refinement_plan(baseline,candidate,self.meta,frame=54,start=49,end=59,baseline_sha256='2'*64)
        self.assertNotIn('view_matrix',merged)
        self.assertTrue(merged['refinement']['outside_window_identical'])

    def test_never_recrops_to_accommodate_exposed_source_edges(self):
        candidate=build_conform_plan(calibration(self.meta,'1'*64,[(54,5)]),self.meta)
        with self.assertRaisesRegex(RefinementError,'preserved view'):
            merge_refinement_plan(self.base,candidate,self.meta,frame=54,start=49,end=59,baseline_sha256='2'*64)

    def test_window_cannot_shrink_old_correction_or_overlap_neighbor(self):
        with self.assertRaisesRegex(RefinementError,'previous seam correction'):
            refinement_window(self.base,54,self.meta,[18,54],3)
        # Long support hits a preserved correction even without crossing its marker.
        with self.assertRaisesRegex(RefinementError,'overlaps a preserved neighbor'):
            refinement_window(self.base,54,self.meta,[18,54],31)

    def test_unattributed_baseline_geometry_is_not_discarded(self):
        baseline=copy.deepcopy(self.base)
        baseline['frame_matrices'][48]=[[1,0,.1],[0,1,0],[0,0,1]]
        with self.assertRaisesRegex(RefinementError,'not attributed'):
            refinement_window(baseline,54,self.meta,[18,54],5)

    def test_non_target_color_is_copied_exactly_and_target_extent_is_checked(self):
        identity=np.repeat(np.arange(256,dtype=float)[:,None],3,axis=1).tolist()
        curve={'frame':18,'support_before':4,'support_after':4,'left_lut':identity,'right_lut':identity}
        self.base['grade_curves']=[curve]
        merged=merge_refinement_plan(self.base,self.next,self.meta,frame=54,start=49,end=59,baseline_sha256='2'*64)
        self.assertEqual(merged['grade_curves'],[curve])
        invalid=copy.deepcopy(self.next);invalid['grade_curves']=[{**curve,'frame':54,'support_before':5}]
        with self.assertRaisesRegex(RefinementError,'Target color exceeds'):
            merge_refinement_plan(self.base,invalid,self.meta,frame=54,start=49,end=59,baseline_sha256='2'*64)


@unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'),'FFmpeg required')
class ProjectRefinementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.media=tempfile.TemporaryDirectory()
        cls.source=Path(cls.media.name)/'source.mp4'
        picture=np.random.default_rng(927).integers(25,220,(64,96,3),dtype=np.uint8)
        with VideoWriter(cls.source,96,64,'12',crf=0,preset='ultrafast') as writer:
            for _ in range(72):writer.write(picture)
        cls.fractional_source=Path(cls.media.name)/'fractional-source.mp4'
        with VideoWriter(cls.fractional_source,96,64,'24000/1001',crf=0,preset='ultrafast') as writer:
            for _ in range(72):writer.write(picture)

    @classmethod
    def tearDownClass(cls):cls.media.cleanup()

    def setUp(self):
        self.directory=tempfile.TemporaryDirectory();self.folder=Path(self.directory.name)
        self.project=projects.create_project(self.source,self.folder/'project')
        self.path=Path(self.project['projectPath'])
        self.project=projects.set_seams(self.path,[18,54])
        recipe=calibration(self.project['metadata'],self.project['sourceSha256'],[(18,.8),(54,.5)])
        self.baseline=build_conform_plan(recipe,self.project['metadata'])
        self.plan=self.folder/'accepted.plan.json';projects.atomic_json(self.plan,self.baseline)
        self.project['artifacts']['plan']=str(self.plan)
        self.project['seamResults']=[{'frame':18,'geometry':'manual','color':'unchanged','notes':['Keep this exact readout']},
                                     {'frame':54,'geometry':'manual','color':'unchanged','notes':[]}]
        projects.save_project(self.project)

    def tearDown(self):self.directory.cleanup()

    def modified(self):
        rows=copy.deepcopy(self.project['seams'])
        record=calibration(self.project['metadata'],self.project['sourceSha256'],[(54,.3)])['cuts'][0]
        manual={key:record[key] for key in ('right_to_left_matrix','pre_rate','post_rate','ease_rate')}
        rows[1]['correction']={**DEFAULT_CORRECTION,'geometry':'manual','manual':manual,'color':'off'}
        return projects.set_seams(self.path,rows)

    def test_settings_retain_baseline_then_refine_real_selected_seam_only(self):
        modified=self.modified()
        self.assertNotIn('plan',modified['artifacts'])
        frozen=modified['refinementBaseline']
        self.assertEqual(frozen['planSha256'],hashlib.sha256(self.plan.read_bytes()).hexdigest())
        self.assertEqual(frozen['seams'],self.project['seams'])
        result=pipeline.run_stage(self.path,'refine',options={'frame':54,'supportFrames':4,'analysis_max_size':96})
        plan=json.loads(Path(result['artifacts']['plan']).read_text())
        self.assertEqual(plan['view_matrix'],self.baseline['view_matrix'])
        self.assertEqual(plan['frame_matrices'][:49],self.baseline['frame_matrices'][:49])
        self.assertEqual(plan['frame_matrices'][59:],self.baseline['frame_matrices'][59:])
        self.assertEqual(result['seamResults'][0],self.project['seamResults'][0])
        self.assertEqual(result['seamResults'][1]['geometry'],'manual')
        self.assertEqual(result['seamResults'][1]['color'],'off')
        self.assertEqual(self.plan.read_bytes(),json.dumps(self.baseline,indent=2,allow_nan=False).encode()+b'\n')
        saved=json.loads(Path(result['artifacts']['calibration']).read_text())
        self.assertEqual([row['frame'] for row in saved['cuts']],[54])
        self.assertEqual(saved['refinement_context']['baseline_plan_sha256'],frozen['planSha256'])

    def test_other_pending_edits_refuse_targeted_analysis(self):
        modified=self.modified();rows=copy.deepcopy(modified['seams']);rows[0]['correction']['color']='off'
        projects.set_seams(self.path,rows)
        with self.assertRaisesRegex(RefinementError,'Other seam markers or settings changed'):
            pipeline.run_stage(self.path,'refine',options={'frame':54,'supportFrames':4})

    def test_mutated_baseline_cannot_be_used_or_published(self):
        modified=self.modified()
        self.plan.write_text(self.plan.read_text()+' ')
        with self.assertRaisesRegex(RefinementError,'missing or changed'):
            pipeline.refine_project(modified,options={'frame':54,'supportFrames':4})
        with self.assertRaisesRegex(RuntimeError,'baseline changed'):
            pipeline._commit(modified,{'plan':'unsafe'},'analyzed',require_refinement_baseline=modified['refinementBaseline'])

    def test_marker_edit_during_refinement_cannot_publish_stale_plan(self):
        modified=self.modified()
        from seamstress.refinement import refine_video as real_refine
        def concurrent(*args,**kwargs):
            result=real_refine(*args,**kwargs)
            current=projects.load_project(self.path)
            current['seams'][0]['correction']['color']='off'
            projects.set_seams(self.path,current['seams'])
            return result
        with patch('seamstress.refinement.refine_video',side_effect=concurrent), self.assertRaisesRegex(RuntimeError,'Seams changed'):
            pipeline.refine_project(modified,options={'frame':54,'supportFrames':4,'analysis_max_size':96})
        current=projects.load_project(self.path)
        self.assertNotIn('plan',current['artifacts'])
        self.assertEqual(current['refinementBaseline'],modified['refinementBaseline'])

    def test_source_mismatch_and_unsafe_crop_leave_project_unchanged(self):
        modified=self.modified();before=self.path.read_bytes()
        with self.assertRaisesRegex(ValueError,'different source'):
            from seamstress.refinement import load_baseline
            load_baseline(self.plan,self.project['metadata'],'0'*64)
        rows=copy.deepcopy(modified['seams']);rows[1]['correction']['manual']['right_to_left_matrix'][0][2]=5
        modified=projects.set_seams(self.path,rows);before=self.path.read_bytes()
        with self.assertRaisesRegex(RefinementError,'preserved view'):
            pipeline.refine_project(modified,options={'frame':54,'supportFrames':4,'analysis_max_size':96})
        self.assertEqual(self.path.read_bytes(),before)

    def test_color_observations_use_final_pinned_geometry(self):
        import cv2
        import seamstress.calibration as engine
        from seamstress.media import read_frame
        seen=[]
        def capture(left,right,options):
            seen.append((left.copy(),right.copy()))
            return {'observed':True}
        record=calibration(self.project['metadata'],self.project['sourceSha256'],[(54,.3)])['cuts'][0]
        manual={key:record[key] for key in ('right_to_left_matrix','pre_rate','post_rate','ease_rate')}
        with patch.object(engine,'_flow_observations',side_effect=capture), patch.object(engine,'_color_fit',
                return_value=(None,None,{'global_accepted':False,'local_accepted':False,'reasons':[]})):
            result=refine_video(self.source,54,self.plan,self.folder/'color.json',support_frames=4,
                correction={**DEFAULT_CORRECTION,'geometry':'manual','manual':manual,'color':'tone'},
                options={'analysis_max_size':96})
        self.assertEqual(len(seen),3)
        final=json.loads(Path(result['plan_path']).read_text())
        self.assertEqual(final['view_matrix'],self.baseline['view_matrix'])
        for side,number in enumerate((53,54)):
            source=read_frame(self.source,number,(96,64))
            matrix=np.asarray(final['view_matrix'])@np.asarray(final['frame_matrices'][number])
            expected=cv2.warpAffine(source,matrix[:2],(96,64),flags=cv2.INTER_CUBIC)
            np.testing.assert_array_equal(seen[0][side],expected)

    def test_unrelated_close_boundaries_do_not_shorten_selected_support(self):
        import seamstress.calibration as engine
        recipe=calibration(self.project['metadata'],self.project['sourceSha256'],[(18,.8)])
        baseline=build_conform_plan(recipe,self.project['metadata'])
        path=self.folder/'one-accepted-seam.json';projects.atomic_json(path,baseline)
        record=recipe['cuts'][0]
        manual={key:record[key] for key in ('right_to_left_matrix','pre_rate','post_rate','ease_rate')}
        levels=np.repeat(np.arange(256,dtype=float)[:,None],3,axis=1)
        table=255*(levels/255)**1.02
        with patch.object(engine,'_flow_observations',return_value={}), patch.object(engine,'_color_fit',return_value=(
                (table,table),None,{'global_accepted':True,'local_accepted':False,'reasons':[]})):
            result=refine_video(self.source,18,path,self.folder/'distant-boundaries.json',
                analysis_boundaries=[18,54,58],support_frames=4,
                correction={**DEFAULT_CORRECTION,'geometry':'manual','manual':manual,'color':'tone'},
                options={'analysis_max_size':96})
        self.assertEqual(result['calibration']['parameters']['geometry_support'],4)
        plan=json.loads(Path(result['plan_path']).read_text())
        self.assertEqual(plan['design_report']['actual_geometry_supports'],
                         [{'frame':18,'before_frames':4,'after_frames':4}])
        self.assertEqual(plan['grade_curves'][0]['support_before'],4)
        self.assertEqual(plan['grade_curves'][0]['support_after'],4)
        self.assertEqual(plan['frame_matrices'],baseline['frame_matrices'])

    def test_explicit_all_markers_keep_existing_full_analysis_behavior(self):
        from seamstress.calibration import calibrate_video
        options={'analysis_max_size':96,'enable_local_color':False,'geometry_support_seconds':4/12}
        original=calibrate_video(self.source,[18,54],self.folder/'original.json',options=options)
        explicit=calibrate_video(self.source,[18,54],self.folder/'explicit.json',options=options,
                                 analysis_boundaries=[18,54])
        self.assertEqual(original['calibration'],explicit['calibration'])
        self.assertEqual(json.loads(Path(original['plan_path']).read_text()),
                         json.loads(Path(explicit['plan_path']).read_text()))

    def test_exact_frame_support_at_fractional_rate_handles_one_and_maximum_frames(self):
        from seamstress.repair import fingerprint
        meta=probe(self.fractional_source);digest=fingerprint(self.fractional_source)
        recipe=calibration(meta,digest,[(54,.5)])
        recipe['parameters'].update(geometry_support=1,rate_support=1)
        path=self.folder/'fractional-baseline.json'
        projects.atomic_json(path,build_conform_plan(recipe,meta))
        record=recipe['cuts'][0]
        manual={key:record[key] for key in ('right_to_left_matrix','pre_rate','post_rate','ease_rate')}
        for support in (1,round(60*meta['fps'])):
            with self.subTest(support=support):
                result=refine_video(self.fractional_source,54,path,self.folder/f'exact-{support}.json',
                    support_frames=support,
                    correction={**DEFAULT_CORRECTION,'geometry':'manual','manual':manual,'color':'off'},
                    options={'analysis_max_size':96})
                self.assertEqual(result['calibration']['parameters']['geometry_support'],support)
                self.assertEqual(result['calibration']['generator']['geometry_support_frames'],support)
                plan=json.loads(Path(result['plan_path']).read_text())
                self.assertEqual(plan['design_report']['actual_geometry_supports'],
                    [{'frame':54,'before_frames':min(support,53),'after_frames':min(support,17)}])



if __name__=='__main__':unittest.main()
