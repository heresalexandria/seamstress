import hashlib
import io
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np

from seamstress import segmentation_model as model


class ModelSetupTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.source=self.root/'source';self.source.mkdir()
        self.records={}
        for name in model.FILES:
            data=('mock pinned bytes '+name).encode();(self.source/name).write_bytes(data)
            self.records[name]={'size':len(data),'sha256':hashlib.sha256(data).hexdigest()}
        self.patch=patch.object(model,'FILES',self.records);self.patch.start();self.addCleanup(self.patch.stop)
        self.target=self.root/'models'

    def test_download_requires_explicit_choice_and_missing_model_never_triggers_network(self):
        with patch('urllib.request.build_opener') as opener:
            self.assertFalse(model.model_status(self.target)['downloaded'])
            self.assertFalse(model.available(self.target))
            with self.assertRaisesRegex(model.SegmentationError,'explicit'):model.setup_model(self.target)
            with self.assertRaises(model.SegmentationError):
                model.predict_mask(np.zeros((10,20,3),np.uint8),points=[{'x':4,'y':5,'label':1}],model_dir=self.target)
            opener.assert_not_called()

    def test_local_import_verifies_hashes_and_preserves_existing_files(self):
        status=model.setup_model(self.target,source_dir=self.source)
        self.assertTrue(status['downloaded'])
        self.assertEqual(status['modelId'],model.MODEL_ID)
        provenance=json.loads((self.target/'provenance.json').read_text())
        self.assertEqual(provenance['files'],self.records)
        model.setup_model(self.target,source_dir=self.source)
        target=self.target/next(iter(self.records));target.write_bytes(b'accepted local bytes')
        with self.assertRaisesRegex(model.SegmentationError,'Existing'):model.setup_model(self.target,source_dir=self.source)
        self.assertEqual(target.read_bytes(),b'accepted local bytes')

    def test_corrupt_import_does_not_publish_model_or_leave_partial_file(self):
        source=self.source/next(iter(self.records));source.write_bytes(b'bad')
        with self.assertRaisesRegex(model.SegmentationError,'SHA-256'):model.setup_model(self.target,source_dir=self.source)
        self.assertEqual(list(self.target.iterdir()),[])

    def test_download_uses_pinned_https_urls_and_bounded_bytes(self):
        events=[]
        def read(request,timeout):
            self.assertIn('/resolve/'+model.REVISION+'/',request.full_url)
            self.assertTrue(request.full_url.startswith(model.PUBLISHER+'/'))
            return io.BytesIO((self.source/request.full_url.rsplit('/',1)[-1]).read_bytes())
        with patch('urllib.request.build_opener') as opener:
            opener.return_value.open.side_effect=read
            self.assertTrue(model.setup_model(self.target,allow_download=True,progress=events.append)['downloaded'])
            self.assertEqual(opener.return_value.open.call_count,2)
        self.assertEqual(events[-1]['fraction'],1)

    def test_cancelled_import_never_publishes_partial_file(self):
        cancelled=[False]
        with self.assertRaises(InterruptedError):
            model.setup_model(self.target,source_dir=self.source,cancelled=lambda:cancelled[0],
                              progress=lambda _event:cancelled.__setitem__(0,True))
        self.assertEqual(list(self.target.iterdir()),[])

    def test_redirects_to_nonpublisher_hosts_are_rejected(self):
        for url in ['http://huggingface.co/model','https://evil.example/model','https://user:password@huggingface.co/model']:
            with self.assertRaises(model.SegmentationError):
                model._ModelRedirect().redirect_request(None,None,302,'redirect',{},url)

    def test_model_status_reports_runtime_missing_without_installing(self):
        model.setup_model(self.target,source_dir=self.source)
        with patch('importlib.util.find_spec',return_value=None):
            status=model.model_status(self.target)
            self.assertTrue(status['downloaded']);self.assertFalse(status['available'])
            self.assertIn('runtime',status['reason'])


class InferenceContractTests(unittest.TestCase):
    def test_invalid_prompts_are_rejected_before_loading_model(self):
        rgb=np.zeros((40,80,3),np.uint8)
        cases=[{}, {'points':[{'x':80,'y':0,'label':1}]}, {'points':[{'x':1,'y':2,'label':2}]},
               {'points':[{'x':True,'y':2,'label':1}]}, {'points':[{'x':1,'y':float('nan'),'label':1}]},
               {'box':[5,6,4,12]}, {'box':[-1,0,12,12]}, {'box':[0,0,81,40]}]
        with patch.object(model,'model_status') as status:
            for kwargs in cases:
                with self.subTest(kwargs=kwargs),self.assertRaises(ValueError):model.predict_mask(rgb,**kwargs)
            status.assert_not_called()

    def test_cpu_inference_scales_prompts_returns_native_mask_and_provenance(self):
        calls=[]
        class Session:
            def __init__(self,path,sess_options,providers):
                self.encoder='encoder' in str(path)
                self.assert_provider=providers
                if providers!=['CPUExecutionProvider']:raise AssertionError('Not CPU')
            def run(self,names,inputs):
                calls.append(inputs)
                if self.encoder:
                    self_shape=inputs['input_image'].shape
                    if self_shape!=(512,1024,3):raise AssertionError(self_shape)
                    return [np.zeros((1,256,64,64),np.float32)]
                h,w=inputs['orig_im_size'].astype(int)
                logits=np.full((1,1,h,w),-1,np.float32);logits[:,:,3:7,4:9]=2
                return [logits,np.array([[.9]],np.float32),np.zeros((1,1,256,256),np.float32)]
        fake=types.SimpleNamespace(SessionOptions=lambda:types.SimpleNamespace(),ExecutionMode=types.SimpleNamespace(ORT_SEQUENTIAL=0),InferenceSession=Session)
        with patch.object(model,'model_status',return_value={'available':True}),patch.dict('sys.modules',{'onnxruntime':fake}):
            result=model.predict_mask(np.zeros((40,80,3),np.uint8),points=[{'x':40,'y':20,'label':1}],box=[20,10,60,30],model_dir='/unused')
        self.assertEqual(result['mask'].shape,(40,80));self.assertEqual(result['mask'].sum(),20)
        np.testing.assert_allclose(calls[1]['point_coords'],[[[512,256],[256,128],[768,384]]])
        np.testing.assert_array_equal(calls[1]['point_labels'],[[1,2,3]])
        self.assertEqual(result['model']['provider'],'CPUExecutionProvider')
        self.assertEqual(result['model']['revision'],model.REVISION)

    def test_points_without_box_include_not_a_point_token(self):
        coords,labels=model._prompts(100,50,[{'x':40,'y':20,'label':1},{'x':70,'y':20,'label':0}],None)
        np.testing.assert_array_equal(labels,[1,0,-1])
        np.testing.assert_array_equal(coords[-1],[0,0])


if __name__=='__main__':unittest.main()
