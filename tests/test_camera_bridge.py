import unittest
from unittest.mock import patch
import numpy as np
import cv2
from seamstress.camera_bridge import synthesize_camera_bridge

class LinearModel:
    def synthesize(self,a,b,progress):
        return [np.round((1-t)*a.astype(float)+t*b.astype(float)).astype(np.uint8) for t in progress]

class CameraBridgeTests(unittest.TestCase):
    def test_camera_translation_has_no_interpolator_endpoint_stall(self):
        rng=np.random.default_rng(7);a=rng.integers(40,180,(96,160,3),dtype=np.uint8)
        def translated(dx):return cv2.warpAffine(a,np.float32([[1,0,dx],[0,1,0]]),(160,96),borderMode=cv2.BORDER_REFLECT_101)
        def registration(dx):return {'confidence':1.,'matrix':[[1,0,dx],[0,1,0],[0,0,1]]}
        before=[translated(i) for i in range(-4,0)]
        after=[translated(i) for i in range(13,17)]
        with patch('seamstress.camera_bridge.register_pair',side_effect=[registration(-12),registration(4),registration(-4)]):
            frames,r=synthesize_camera_bridge(LinearModel(),a,translated(12),before,after,np.linspace(0,1,13))
        np.testing.assert_array_equal(frames[0],a);np.testing.assert_array_equal(frames[-1],translated(12))
        np.testing.assert_allclose(np.asarray(r['camera_parameters'])[:,2],np.arange(1,12),atol=1e-6)
        self.assertEqual(len(frames),13);self.assertEqual(r['source_coverage_min'],1.)
    def test_missing_handles_rejected(self):
        a=np.zeros((96,160,3),np.uint8)
        with self.assertRaisesRegex(ValueError,'handles'):
            synthesize_camera_bridge(LinearModel(),a,a,[],[a],np.linspace(0,1,5))

if __name__=='__main__':unittest.main()
