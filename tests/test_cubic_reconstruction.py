"""Reconstruction semantics without a checkpoint or neural inference."""
import unittest
import numpy as np
from seamstress.bridge import RifeModel,_cubic_reconstruct


class CubicReconstructionTests(unittest.TestCase):
    def test_soft_mask_sharpening_preserves_expected_source_ownership(self):
        a=np.full((12,16,3),.2,np.float32);b=np.full_like(a,.8)
        flow=np.zeros((12,16,4),np.float32)
        logits=np.full((12,16),np.log(3),np.float32)
        # sigmoid(log(3))=.75; sigmoid(2*log(3))=.9, still a soft mix.
        result=_cubic_reconstruct(a,b,flow,logits)
        np.testing.assert_allclose(result,.2*.9+.8*.1,atol=1e-6)

    def test_gather_direction_and_nonreflecting_border(self):
        a=np.zeros((8,12,3),np.float32)
        a[:]=np.arange(12,dtype=np.float32)[None,:,None]/11
        flow=np.zeros((8,12,4),np.float32);flow[...,0]=2
        result=_cubic_reconstruct(a,a,flow,np.full((8,12),30,np.float32))
        np.testing.assert_allclose(result[:,0],a[:,2],atol=1e-6)
        np.testing.assert_allclose(result[:,-2:],1,atol=1e-6)

    def test_cubic_overshoot_is_bounded(self):
        a=np.zeros((16,16,3),np.float32);a[:,8:]=1
        flow=np.zeros((16,16,4),np.float32);flow[...,0]=.4;flow[...,2]=.4
        result=_cubic_reconstruct(a,a,flow,np.zeros((16,16),np.float32))
        self.assertGreaterEqual(float(result.min()),0)
        self.assertLessEqual(float(result.max()),1)

    def test_invalid_mode_is_rejected_before_weights_access(self):
        with self.assertRaisesRegex(ValueError,'reconstruction'):
            RifeModel('not-a-file',reconstruction='hard-mask')


if __name__=='__main__':unittest.main()
