import tempfile
import unittest
from pathlib import Path
import numpy as np
from seamstress.bridge import progress_curve,validate_bridge_plan,RifeModel

class BridgeTests(unittest.TestCase):
    def test_progress_is_monotone_and_endpoint_exact_for_extreme_slopes(self):
        for a,b in [(1,1),(0,0),(3,0),(0,3),(8,4)]:
            p=progress_curve(100,a,b)
            self.assertEqual(p[0],0);self.assertEqual(p[-1],1)
            self.assertTrue(np.all(np.diff(p)>=-1e-12))
    def test_progress_rejects_invalid_timing(self):
        for a,b in [(-1,0),(float('nan'),1),(1,float('inf'))]:
            with self.assertRaises(ValueError):progress_curve(12,a,b)
    def test_bridge_windows_cannot_overlap(self):
        m={'width':1280,'height':720,'frame_count':100,'fps_fraction':'24/1','fps':24.}
        p={'schema_version':2,'method':'rife_bridge','source':m,'source_sha256':'0'*64,'seams':[{'frame':20,'time':20/24,'bridge_start':10,'bridge_end':25},{'frame':30,'time':30/24,'bridge_start':24,'bridge_end':35}]}
        with self.assertRaisesRegex(ValueError,'overlap'):validate_bridge_plan(p,m)
    def test_wrong_weights_rejected_before_torch_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            f=Path(tmp)/'fake.pkl';f.write_bytes(b'not a checkpoint')
            with self.assertRaisesRegex(ValueError,'checksum'):RifeModel(f)

if __name__=='__main__':unittest.main()
