"""Verification's common native-resolution view and minimal conform schema."""
from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from seamstress import analysis
from seamstress.conform import validate_conform_plan


class VerificationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source, self.output = self.root/"source.mp4", self.root/"candidate.mp4"
        self.meta = dict(width=960, height=540, frame_count=4, fps_fraction="24000/1001",
                         fps=24000/1001, duration=4*1001/24000, has_audio=False)
        self.plan = dict(schema_version=3, method="source_conform",
                         source={k:self.meta[k] for k in ("width","height","frame_count","fps_fraction")},
                         source_sha256="a"*64,
                         segments=[dict(start=0,end=4,matrix=np.eye(3).tolist(),gain=[1,1,1],bias=[0,0,0])])
        y,x = np.indices((540,960))
        self.frames = [np.stack(((x*31+i*7)%256, (y*43+i*13)%256, (x*17+y*19)%256),axis=-1).astype(np.uint8) for i in range(4)]

    def run_verification(self, output_frames=None):
        recipe = self.root/"plan.json"; recipe.write_text(json.dumps(self.plan))
        pairs=[]
        class Writer:
            def __init__(self,*args,**kwargs):pass
            def __enter__(self):return self
            def __exit__(self,*args):pass
            def write(self,frame):pass
        def metrics(a,b):
            pairs.append((a.copy(),b.copy()))
            return dict(registered_mae=1.,sharpness_left=1.,sharpness_right=1.,motion_p50=1.)
        def read(path,start,count):
            frames = self.frames if path == self.source else (output_frames or self.frames)
            return np.stack(frames[start:start+count])
        validate_conform_plan(self.plan,self.meta)
        with patch.object(analysis,"probe",return_value=self.meta),patch.object(analysis,"fingerprint",return_value="a"*64), \
             patch.object(analysis,"read_frames",side_effect=read),patch.object(analysis,"VideoWriter",Writer), \
             patch.object(analysis,"pair_metrics",side_effect=metrics),contextlib.redirect_stdout(io.StringIO()):
            result=analysis.verify(self.source,self.output,recipe,self.root/"review")
        return result,pairs

    def test_valid_minimal_conform_plan_needs_neither_fps_float_nor_seams(self):
        result,_=self.run_verification()
        self.assertEqual(result['seam_error_reductions'],[])
        page=(self.root/'review/review.html').read_text()
        self.assertIn('fps=23.976023976023978',page)
        self.assertIn('const seams=[]',page)
        self.assertIn('>Candidate<',page)

    def test_metric_view_is_applied_at_native_resolution_and_status_remains_visible(self):
        view=np.array([[1.075,0,-34.25],[0,1.075,-19.85],[0,0,1]])
        self.plan.update(view_matrix=view.tolist(),seams=[{'frame':2}],
                         unresolved_seams=[2],status='Pending <review>')
        output_frames=[cv2.warpAffine(f,view[:2],(960,540),flags=cv2.INTER_CUBIC,borderMode=cv2.BORDER_REPLICATE) for f in self.frames]
        result,pairs=self.run_verification(output_frames)
        expected=[cv2.resize(f,(480,270),interpolation=cv2.INTER_AREA) for f in output_frames]
        np.testing.assert_array_equal(pairs[0][0],expected[1])
        np.testing.assert_array_equal(pairs[0][1],expected[2])
        np.testing.assert_array_equal(pairs[0],pairs[1])
        self.assertEqual(result['unresolved_seams'],[2])
        self.assertEqual(result['metric_reference_view_matrix'],view.tolist())
        page=(self.root/'review/review.html').read_text()
        self.assertIn('Pending &lt;review&gt;',page)
        self.assertIn('Unresolved geometry joins: 2',page)
        self.assertIn('"unresolved": true',page)

    def test_preview_sidecar_is_rejected_before_decoding(self):
        self.output.with_suffix('.repair.json').write_text(json.dumps({'preview':True,'source_start_frame':100}))
        recipe=self.root/'plan.json';recipe.write_text(json.dumps(self.plan))
        with patch.object(analysis,'probe',return_value=self.meta),patch.object(analysis,'fingerprint',return_value='a'*64), \
             patch.object(analysis,'read_frames') as read,patch.object(analysis,'VideoWriter') as writer:
            with self.assertRaisesRegex(ValueError,'full-timeline'):
                analysis.verify(self.source,self.output,recipe,self.root/'review')
            read.assert_not_called();writer.assert_not_called()


if __name__=='__main__':unittest.main()
