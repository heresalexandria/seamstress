"""Calibrated plan regeneration, source coverage, and publication guards."""
from __future__ import annotations
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from seamstress.design import ALGORITHM, build_conform_plan, design_conform

ROOT=Path(__file__).resolve().parents[1]


class DesignTests(unittest.TestCase):
    def setUp(self):
        self.metadata={'width':80,'height':60,'frame_count':96,'fps_fraction':'24000/1001'}
        self.calibration={'schema_version':1,'method':'source_conform_calibration','algorithm':ALGORITHM,
                          'source':self.metadata.copy(),'source_sha256':'a'*64,
                          'parameters':{'geometry_support':16,'rate_support':4,'source_margin_pixels':2,
                                        'max_view_crop_fraction_total_dimension':.15},
                          'cuts':[{'frame':40,'right_to_left_matrix':[[1,0,1],[0,1,.5],[0,0,1]],
                                   'pre_rate':[0,0,0,0],'post_rate':[0,0,0,0],'ease_rate':False}],
                          'excluded_geometry':[],'grade_curves':[]}

    def test_source_dimensions_control_coverage_and_cut_alignment(self):
        plan=build_conform_plan(self.calibration,self.metadata)
        matrices=np.array(plan['frame_matrices']);view=np.array(plan['view_matrix'])
        self.assertEqual(matrices.shape,(96,3,3))
        # Paired cut transforms must align the reviewed correspondence.
        np.testing.assert_allclose(matrices[39]@self.calibration['cuts'][0]['right_to_left_matrix'],matrices[40],atol=1e-12)
        np.testing.assert_array_equal(matrices[0],np.eye(3))
        np.testing.assert_array_equal(matrices[-1],np.eye(3))
        corners=np.array([[0,0,1],[79,0,1],[0,59,1],[79,59,1]],float).T
        points=np.linalg.inv(view@matrices)@corners
        self.assertGreaterEqual(points[:,0].min(),2)
        self.assertGreaterEqual(points[:,1].min(),2)
        self.assertLessEqual(points[:,0].max(),77)
        self.assertLessEqual(points[:,1].max(),57)

    def test_explicit_exclusion_retains_original_geometry_and_reason(self):
        self.calibration['excluded_geometry']=[{'frame':40,'reason':'Incompatible foreground/background evidence'}]
        plan=build_conform_plan(self.calibration,self.metadata)
        np.testing.assert_array_equal(plan['frame_matrices'],np.repeat(np.eye(3)[None],96,axis=0))
        self.assertEqual(plan['unresolved_seams'],[40])
        self.assertEqual(plan['design_report']['excluded_geometry'],self.calibration['excluded_geometry'])
        self.assertIn('EXPERIMENTAL',plan['status'])

    def test_short_final_return_is_explicit_and_finishes_at_identity(self):
        self.calibration['cuts'][0]['frame']=82
        plan=build_conform_plan(self.calibration,self.metadata)
        self.assertEqual(plan['design_report']['actual_geometry_supports'],[{'frame':82,'before_frames':16,'after_frames':13}])
        np.testing.assert_array_equal(plan['frame_matrices'][-1],np.eye(3))

    def test_invalid_measurements_overlap_and_excessive_crop_are_rejected(self):
        variants=[]
        bad=copy.deepcopy(self.calibration);bad['cuts'][0]['pre_rate'][0]=float('nan');variants.append(bad)
        bad=copy.deepcopy(self.calibration);bad['cuts'][0]['ease_rate']='yes';variants.append(bad)
        bad=copy.deepcopy(self.calibration);bad['cuts'].append(dict(bad['cuts'][0],frame=50));variants.append(bad)
        bad=copy.deepcopy(self.calibration);bad['excluded_geometry']=[{'frame':70,'reason':'unknown cut'}];variants.append(bad)
        bad=copy.deepcopy(self.calibration);bad['parameters']['max_view_crop_fraction_total_dimension']=.001;variants.append(bad)
        for calibration in variants:
            with self.subTest(calibration=calibration),self.assertRaises(ValueError):
                build_conform_plan(calibration,self.metadata)
        for geo,rate in ((0,4),(4,8),(True,4),(16,-2)):
            with self.subTest(geometry_support=geo,rate_support=rate),self.assertRaises(ValueError):
                build_conform_plan(self.calibration,self.metadata,geo,rate)

    def test_source_hash_and_existing_paths_are_guarded_before_publication(self):
        with tempfile.TemporaryDirectory() as folder:
            folder=Path(folder);source=folder/'source.mp4';source.write_bytes(b'original source')
            original=source.read_bytes();calibration=folder/'calibration.json';output=folder/'plan.json'
            self.calibration['source_sha256']=hashlib.sha256(original).hexdigest()
            calibration.write_text(json.dumps(self.calibration))
            with patch('seamstress.design.probe',return_value=self.metadata):
                report=design_conform(source,calibration,output)
                plan=json.loads(output.read_text())
                self.assertEqual(plan['generator']['calibration_sha256'],hashlib.sha256(calibration.read_bytes()).hexdigest())
                self.assertEqual(report['plan_sha256'],hashlib.sha256(output.read_bytes()).hexdigest())
                saved=output.read_bytes()
                for collision in (source,calibration,output):
                    with self.assertRaisesRegex(ValueError,'new plan output'):
                        design_conform(source,calibration,collision)
                self.calibration['source_sha256']='0'*64;calibration.write_text(json.dumps(self.calibration))
                with self.assertRaisesRegex(ValueError,'fingerprint'):
                    design_conform(source,calibration,folder/'wrong.json')
                self.assertFalse((folder/'wrong.json').exists())
                self.assertEqual(output.read_bytes(),saved)
                self.assertEqual(source.read_bytes(),original)

    @unittest.skipUnless((ROOT/'plans/IYTYT-calibration.json').exists(), 'Source-specific calibration not available')
    def test_all_3347_frames_match_the_current_eight_join_reference_plan(self):
        calibration=json.loads((ROOT/'plans/IYTYT-calibration.json').read_text())
        reference=json.loads((ROOT/'plans/IYTYT-eight-joins.json').read_text())
        regenerated=build_conform_plan(calibration,reference['source'])
        np.testing.assert_allclose(regenerated['frame_matrices'],reference['frame_matrices'],rtol=0,atol=1e-11)
        np.testing.assert_allclose(regenerated['view_matrix'],reference['view_matrix'],rtol=0,atol=1e-12)
        self.assertEqual(regenerated['grade_curves'],reference['grade_curves'])
        self.assertEqual(regenerated['unresolved_seams'],[2888])
        self.assertEqual(regenerated['design_report']['actual_geometry_supports'][-1]['after_frames'],106)


if __name__=='__main__':unittest.main()
