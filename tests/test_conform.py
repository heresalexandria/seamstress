import copy
import unittest
import numpy as np

from seamstress.conform import conform_frame, tone_lut_at, validate_conform_plan


class ConformTests(unittest.TestCase):
    def setUp(self):
        self.meta = dict(width=24, height=16, frame_count=8, fps_fraction='24000/1001')
        self.plan = dict(schema_version=3, method='source_conform', source=self.meta,
            source_sha256='a'*64, segments=[dict(start=0, end=8,
            matrix=np.eye(3).tolist(), gain=[1,1,1], bias=[0,0,0])])

    def test_original_drawing_survives_identity_exactly(self):
        source = np.random.default_rng(1).integers(0, 256, (16,24,3), dtype=np.uint8)
        result, outside, clipped = conform_frame(source, np.eye(3), np.ones(3), np.zeros(3))
        np.testing.assert_array_equal(result, source)
        self.assertEqual((outside, clipped), (0, 0))

    def test_rejects_missing_frames_and_nonrigid_distortion(self):
        validate_conform_plan(self.plan, self.meta)
        for transform in ([[1,.1,0],[0,1,0],[0,0,1]], [[1,0,0],[0,.9,0],[0,0,1]], [[-1,0,23],[0,1,0],[0,0,1]]):
            changed=copy.deepcopy(self.plan);changed['segments'][0]['matrix']=transform
            with self.assertRaises(ValueError):validate_conform_plan(changed, self.meta)
        changed=copy.deepcopy(self.plan);changed['segments'][0]['end']=7
        with self.assertRaises(ValueError):validate_conform_plan(changed, self.meta)

    def test_refuses_exposed_border(self):
        matrix=np.eye(3);matrix[0,2]=2
        frame=np.zeros((16,24,3),np.uint8)
        with self.assertRaisesRegex(ValueError, 'beyond source'):
            conform_frame(frame,matrix,np.ones(3),np.zeros(3))

    def test_constant_grade_is_not_temporal_blending(self):
        frame=np.full((16,24,3),100,np.uint8)
        result,_,_=conform_frame(frame,np.eye(3),np.array([1,1.1,.8]),np.array([2,0,-5]))
        np.testing.assert_array_equal(result[0,0],[102,110,75])

    def test_per_frame_matrix_count(self):
        self.plan['frame_matrices']=[np.eye(3).tolist()]*7
        with self.assertRaisesRegex(ValueError,'exactly one matrix'):
            validate_conform_plan(self.plan,self.meta)

    def test_affine_aspect_correction_requires_explicit_mode(self):
        self.plan['segments'][0]['matrix']=[[1,0,0],[0,1.02,0],[0,0,1]]
        with self.assertRaises(ValueError):validate_conform_plan(self.plan,self.meta)
        self.plan['geometry_mode']='affine'
        validate_conform_plan(self.plan,self.meta)
        self.plan['segments'][0]['matrix'][1][1]=1.3
        with self.assertRaises(ValueError):validate_conform_plan(self.plan,self.meta)

    def test_grade_lut_keeps_black_white_and_cut_strength(self):
        identity=np.repeat(np.arange(256)[:,None],3,axis=1)
        lut=255*(identity/255)**1.1
        curve=dict(frame=4,support_before=2,support_after=2,left_lut=lut.tolist(),right_lut=lut.tolist())
        self.plan['grade_curves']=[curve]
        validate_conform_plan(self.plan,self.meta)
        self.assertIsNone(tone_lut_at(1,[curve]))
        self.assertIsNone(tone_lut_at(6,[curve]))
        for i in (3,4):
            np.testing.assert_allclose(tone_lut_at(i,[curve]),lut,atol=2e-5)
        table=tone_lut_at(2,[curve])
        np.testing.assert_array_equal(table[[0,255]],[[0,0,0],[255,255,255]])
        self.assertTrue(np.all(np.diff(table,axis=0)>=0))


if __name__=='__main__':unittest.main()
