"""Source-frame, geometry, provenance and recovery invariants for layered repair."""
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image

from seamstress.media import probe, read_frames
from seamstress.reconstruction import (apply_frame, edit_bundle, import_authored,
    import_bundle, load_bundle, prepare_backgrounds, propose, render_bundle, summary)
from seamstress.reconstruction.bundle import sha256
from seamstress.reconstruction.vision import recover_background


def encode(path, frames, fps='24000/1001',audio=False):
    h,w=frames.shape[1:3]
    args=['ffmpeg','-v','error','-nostdin','-n','-f','rawvideo','-pix_fmt','rgb24',
          '-s',f'{w}x{h}','-framerate',fps,'-i','pipe:0']
    if audio:args+=['-f','lavfi','-i','sine=frequency=700:sample_rate=48000','-map','0:v','-map','1:a','-c:a','flac','-shortest']
    args+=['-c:v','ffv1','-pix_fmt','bgr0','-frames:v',str(len(frames)),str(path)]
    subprocess.run(args,input=frames.tobytes(),check=True,capture_output=True)


@unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'),'FFmpeg required')
class ReconstructionTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        h,w=48,64; y,x=np.mgrid[:h,:w]
        self.plate=np.stack(((x*13+y*7)%160+20,(x*3+y*11)%140+50,(x*7+y*3)%100+70),axis=2).astype(np.uint8)
        self.frames=np.stack([self.plate.copy() for _ in range(10)])
        for n,rgb in enumerate(self.frames):rgb[16:28,22+n:30+n]=[225,35+n,65]
        self.source=self.root/'source.mkv';encode(self.source,self.frames,audio=True)
        np.testing.assert_array_equal(read_frames(self.source,0,10),self.frames)

    def tearDown(self):self.temp.cleanup()

    def authored(self,*,output='bundle',motion=True,unknown=False):
        assets=self.root/f'assets-{output}';assets.mkdir()
        rows={};layers=[{'id':'background','kind':'background','order':0,'mask_by_frame':{},'matrices':{},'keyframes':[5]},
                         {'id':'foreground','kind':'foreground','order':1,'mask_by_frame':{},'matrices':{},'keyframes':[5]}]
        for n in range(2,8):
            alpha=np.zeros((48,64),np.float32);alpha[16:28,22+n:30+n]=1
            for kind,rgb in [('source',self.frames[n]),('plate',self.plate)]:Image.fromarray(rgb).save(assets/f'{kind}-{n}.png')
            platefile=assets/f'plate-{n}.png';matte=assets/f'matte-{n}.npz'
            np.savez_compressed(matte,alpha=alpha,premultiplied=self.frames[n].astype(np.float32)*alpha[:,:,None],
                emission=np.zeros((48,64,3),np.float32),background_sha256=np.asarray(sha256(platefile)))
            rows[n]={'source':assets/f'source-{n}.png','plate':platefile,'matte':matte}
            if unknown:
                path=assets/f'unknown-{n}.png';Image.fromarray((alpha*255).astype(np.uint8)).save(path);rows[n]['unknown']=path
            for layer,mask in zip(layers,[np.ones((48,64),np.float32),alpha]):
                path=assets/f'{layer["id"]}-{n}.png';Image.fromarray((mask*255).astype(np.uint8)).save(path)
                layer['mask_by_frame'][str(n)]=path
                m=np.eye(3)
                if motion and n in (4,5):
                    if layer['kind']=='foreground':m[0,2]=2
                    else:m=np.array([[1.04,0,-1.28],[0,1.04,-.96],[0,0,1]])
                layer['matrices'][str(n)]=m.tolist()
        return import_authored(self.source,5,self.root/output,support=(2,8),frames=rows,layers=layers)

    def test_same_frame_actor_and_exact_outside_support(self):
        path=self.authored();b=load_bundle(path)
        for n in (0,1,2,3,6,7,8,9):np.testing.assert_array_equal(apply_frame(b,n,self.frames[n]),self.frames[n])
        result=apply_frame(b,5,self.frames[5])
        np.testing.assert_array_equal(result[16:28,29:37],self.frames[5,16:28,27:35])
        self.assertFalse(np.array_equal(result,self.frames[5]))
        wrong=self.frames[4]
        with self.assertRaisesRegex(ValueError,'original source RGB'):apply_frame(b,5,wrong)
        self.assertEqual(summary(path)['sourceFrame'],5)
        self.assertEqual(len(summary(path)['frames']),6)

    def test_immutable_manifest_assets_and_transform_validation(self):
        path=self.authored()
        with self.assertRaises(FileExistsError):self.authored(output='bundle')
        with self.assertRaisesRegex(ValueError,'similarity'):
            edit_bundle(path,self.root/'shear',{'layer_matrices':[{'layer_id':'foreground','frame':5,'matrix':[[1,.3,0],[0,1,0],[0,0,1]]}]})
        original=path.read_bytes()
        edited=edit_bundle(path,self.root/'approved',{'review_approved':True})
        self.assertEqual(path.read_bytes(),original);self.assertEqual(summary(edited)['status'],'approved')
        target=Path(load_bundle(path)['_root'])/'plates/plate-5.png';target.write_bytes(target.read_bytes()+b'tamper')
        with self.assertRaisesRegex(ValueError,'hash mismatch'):load_bundle(path)

    def test_brush_propagation_respects_generation_boundary(self):
        path=self.authored();before=load_bundle(path)
        edited=edit_bundle(path,self.root/'edit',{'strokes':[{'frame':5,'layer_id':'foreground','mode':'protect','radius':2,'points':[[40,20],[42,20]]}]})
        after=load_bundle(edited)
        for n in (2,3,4):
            self.assertEqual(before['assets'][before['layers'][1]['mask_by_frame'][str(n)]],after['assets'][after['layers'][1]['mask_by_frame'][str(n)]])
        alpha=np.load(Path(after['_root'])/after['frames']['5']['matte'])['alpha']
        self.assertEqual(alpha[20,41],1)
        self.assertEqual(summary(edited)['status'],'needs-review')
        with self.assertRaisesRegex(ValueError,'inside native'):
            edit_bundle(path,self.root/'badbrush',{'strokes':[{'frame':5,'layer_id':'foreground','mode':'include','radius':2,'points':[[-1,20]]}]})
        # Adding a second authored frame must not overwrite the first one's
        # source-specific contour while still listing it as a keyframe.
        second=edit_bundle(edited,self.root/'edit-again',{'strokes':[{'frame':6,'layer_id':'foreground','mode':'exclude','radius':3,'points':[[41,20]]}]})
        later=load_bundle(second)
        old_mask=Path(after['_root'])/after['layers'][1]['mask_by_frame']['5']
        new_mask=Path(later['_root'])/later['layers'][1]['mask_by_frame']['5']
        self.assertEqual(sha256(old_mask),sha256(new_mask))
        self.assertEqual(later['layers'][1]['authored_keyframes'],[5,6])

    def test_lossless_window_render_preserves_frames_rate_audio(self):
        path=self.authored();b=load_bundle(path)
        output=render_bundle(path,self.root/'render')
        info=probe(output);self.assertEqual(info['frame_count'],6);self.assertTrue(info['has_audio'])
        self.assertAlmostEqual(info['fps'],24000/1001,places=5)
        actual=read_frames(output,0,6)
        expected=np.stack([apply_frame(b,n,self.frames[n]) for n in range(2,8)])
        np.testing.assert_array_equal(actual,expected)
        report=json.loads((output.parent/'render-report.json').read_text())
        self.assertEqual(report['source_start_frame'],2)
        self.assertEqual(report['source_end_frame_exclusive'],8)

    def test_import_rejects_different_source_and_cancel_never_publishes(self):
        path=self.authored();copy=import_bundle(path,self.root/'import',source=self.source)
        np.testing.assert_array_equal(apply_frame(copy,5,self.frames[5]),apply_frame(path,5,self.frames[5]))
        other=self.root/'other.mkv';encode(other,self.frames[::-1])
        with self.assertRaisesRegex(ValueError,'another source'):import_bundle(path,self.root/'wrong',source=other)
        with self.assertRaises(InterruptedError):render_bundle(path,self.root/'cancel',cancelled=lambda:True)
        self.assertFalse((self.root/'cancel/native.mkv').exists())

    def test_portable_import_works_after_original_source_path_is_gone(self):
        path=self.authored();relocated=self.root/'relocated.mkv';shutil.copyfile(self.source,relocated);self.source.unlink()
        imported=import_bundle(path,self.root/'relocated-bundle',source=relocated)
        self.assertEqual(load_bundle(imported)['source']['path'],str(relocated.resolve()))
        np.testing.assert_array_equal(apply_frame(imported,5,self.frames[5])[16:28,29:37],self.frames[5,16:28,27:35])

    def test_review_approval_cannot_override_bad_background_coverage(self):
        path=self.authored()
        bad=edit_bundle(path,self.root/'badcoverage',{'layer_matrices':[{'layer_id':'background','frame':5,
            'matrix':[[.25,0,24],[0,.25,18],[0,0,1]]}],'review_approved':True})
        info=summary(bad)
        self.assertFalse(info['canAccept']);self.assertEqual(info['status'],'needs-review')
        self.assertTrue(info['qa']['blocking_errors'])
        self.assertIn(info['qa']['blocking_errors'][0],info['issues'])
        clipped=edit_bundle(path,self.root/'clippedactor',{'layer_matrices':[{'layer_id':'foreground','frame':5,
            'matrix':[[1,0,100],[0,1,0],[0,0,1]]}],'review_approved':True})
        self.assertFalse(summary(clipped)['canAccept'])
        boundary=edit_bundle(path,self.root/'boundaryjump',{'layer_matrices':[{'layer_id':'foreground','frame':2,
            'matrix':[[1,0,1],[0,1,0],[0,0,1]]}],'review_approved':True})
        self.assertFalse(summary(boundary)['canAccept'])

    def test_one_cloud_anchor_keeps_masks_and_requires_review(self):
        path=self.authored(unknown=True);before=load_bundle(path);calls=[]
        def provider(source,mask,frame):
            calls.append((frame,mask.copy()))
            output=self.root/'generated.png';rgb=source.copy();rgb[mask]=[50,60,70];Image.fromarray(rgb).save(output)
            return {'image_path':str(output),'request_id':'fake-test-only'}
        final=prepare_backgrounds(path,self.root/'filled',provider=provider)
        after=load_bundle(final);self.assertEqual(len(calls),1)
        self.assertFalse(summary(final)['autoEligible']);self.assertEqual(summary(final)['status'],'needs-review')
        for n in range(2,8):
            with np.load(Path(before['_root'])/before['frames'][str(n)]['matte']) as left, np.load(Path(after['_root'])/after['frames'][str(n)]['matte']) as right:
                np.testing.assert_array_equal(left['alpha'],right['alpha']);np.testing.assert_array_equal(left['emission'],right['emission'])
        with self.assertRaises(FileExistsError):prepare_backgrounds(path,self.root/'filled',provider=provider)
        self.assertEqual(len(calls),1,'Existing revision must be rejected before making a paid request')

    def test_source_only_background_recovery_is_an_explicit_immutable_revision(self):
        path=self.authored();before=path.read_bytes()
        revised=prepare_backgrounds(path,self.root/'source-only')
        self.assertEqual(path.read_bytes(),before)
        b=load_bundle(revised)
        self.assertEqual(b['provenance']['background_recovery'][-1]['method'],'original source donors only')
        self.assertNotIn('background_generation',b['provenance'])
        for n in range(2,8):
            with np.load(Path(b['_root'])/b['frames'][str(n)]['matte']) as data:
                core=data['alpha']==1
                np.testing.assert_array_equal(data['premultiplied'][core],self.frames[n][core])

    def test_guarded_automatic_route_accepts_only_verified_single_motion(self):
        rng=np.random.default_rng(22);image=rng.integers(0,256,(192,256,3),dtype=np.uint8)
        image=cv2.GaussianBlur(image,(3,3),.7)
        frames=np.repeat(image[None],8,axis=0);source=self.root/'static.mkv';encode(source,frames)
        manifest=propose(source,4,self.root/'static-proposal',options={'reachFrames':3,'segmentation':'classic'})
        info=summary(manifest)
        self.assertTrue(info['autoEligible'],info['metrics'])
        self.assertEqual(info['qa']['metrics']['unrecovered_background_pixels'],0)
        # A spatially localized color/content change must abstain despite all
        # matching background features still agreeing on a rigid camera.
        changed=frames.copy();changed[4:,30:150,40:160]=[240,10,30]
        second=self.root/'changed.mkv';encode(second,changed)
        candidate=propose(second,4,self.root/'changed-proposal',options={'reachFrames':3,'segmentation':'classic'})
        self.assertFalse(summary(candidate)['autoEligible'])

    def test_installed_neural_seed_records_provenance_but_requires_matte_review(self):
        from seamstress import segmentation_model
        points=np.array([[10,10],[20,10],[30,10],[40,10],[10,20],[20,20],[30,20],[40,20]],np.float32)
        clusters=[{'matrix':np.eye(3),'points':points,'targets':points,'inliers':8,'p90':0.},
                  {'matrix':np.array([[1,0,2],[0,1,0],[0,0,1]]),'points':points+3,'targets':points+5,'inliers':8,'p90':0.}]
        mask=np.zeros((48,64),np.float32);mask[16:28,27:35]=1
        with patch('seamstress.reconstruction.vision.motion_clusters',return_value=(clusters,points,points)), \
                patch.object(segmentation_model,'available',return_value=True), \
                patch.object(segmentation_model,'predict_mask',return_value={'mask':mask,'score':.9,'model':{'name':'test-model'}}) as predict:
            path=propose(self.source,5,self.root/'neural',options={'reachFrames':3,'segmentation':'auto'})
        self.assertEqual(predict.call_count,1)
        self.assertEqual(load_bundle(path)['provenance']['segmentation_model']['name'],'test-model')
        self.assertFalse(summary(path)['autoEligible'])

    def test_existing_baseline_alignment_is_not_applied_a_second_time(self):
        # B_right already supplies the measured incoming→outgoing translation.
        # The native repair runs before B_right, so its residual must be I.
        correction=np.array([[1,0,2],[0,1,0],[0,0,1]],float)
        plan={'schema_version':3,'method':'source_conform','source':probe(self.source),
              'source_sha256':sha256(self.source),'segments':[{'start':0,'end':10,
              'matrix':np.eye(3).tolist(),'gain':[1,1,1],'bias':[0,0,0]}],
              'frame_matrices':[np.eye(3).tolist()]*5+[correction.tolist()]*5}
        baseline=self.root/'baseline.json';baseline.write_text(json.dumps(plan))
        points=np.array([[5,5],[55,5],[5,40],[55,40],[30,20],[40,30],[15,30],[30,40]],np.float32)
        clusters=[{'matrix':correction,'points':points,'targets':points+[2,0],'inliers':8,'p90':0.}]
        with patch('seamstress.reconstruction.vision.motion_clusters',return_value=(clusters,points,points+[2,0])), \
                patch('seamstress.reconstruction.vision.match_points',return_value=(points,points)), \
                patch('seamstress.reconstruction.vision.fit_similarity',return_value=(np.eye(3),np.ones(8,bool),0.)):
            path=propose(self.source,5,self.root/'residual',baseline_plan=baseline,options={'reachFrames':3,'segmentation':'classic'})
        b=load_bundle(path)
        for layer in b['layers']:
            for m in layer['matrices'].values():np.testing.assert_array_equal(m,np.eye(3))
        for n in range(2,9):np.testing.assert_array_equal(apply_frame(b,n,self.frames[n]),self.frames[n])
        self.assertFalse(summary(path)['autoEligible'],'Varying baseline geometry requires final-context review')

    def test_memory_guard_runs_before_decode_or_directory_creation(self):
        metadata={'width':3840,'height':2160,'fps':60.,'frame_count':2400}
        with patch('seamstress.reconstruction.engine.probe',return_value=metadata), \
                patch('seamstress.reconstruction.engine.read_frames',side_effect=AssertionError('Must reject before decode')), \
                self.assertRaisesRegex(ValueError,'Shorten reachFrames'):
            propose(self.source,1200,self.root/'too-large',options={'reachFrames':600})
        self.assertFalse((self.root/'too-large').exists())

    def test_actual_proposal_end_to_end_and_confidence_abstention(self):
        path=propose(self.source,5,self.root/'proposal',options={'reachFrames':3,'motionStrength':.4})
        info=summary(path)
        self.assertTrue(info['canRender']);self.assertFalse(info['autoEligible'])
        self.assertEqual(info['status'],'needs-review')
        b=load_bundle(path)
        self.assertIn('classical-cv',b['provenance']['method'])
        output=render_bundle(path,self.root/'proposed-render')
        self.assertEqual(probe(output)['frame_count'],7)
        with self.assertRaisesRegex(ValueError,'propose again'):
            edit_bundle(path,self.root/'extend',{'reachFrames':9})

    def test_odd_dimensions_and_other_frame_rate_are_supported(self):
        frames=np.full((8,45,63,3),125,np.uint8);source=self.root/'odd.mkv';encode(source,frames,'30/1')
        path=propose(source,4,self.root/'odd-proposal',options={'reachFrames':3})
        output=render_bundle(path,self.root/'odd-render');info=probe(output)
        self.assertEqual((info['width'],info['height'],info['frame_count']),(63,45,7))
        np.testing.assert_array_equal(read_frames(output,0,7),frames[1:])


class BackgroundRecoveryTests(unittest.TestCase):
    def test_source_donor_recovery_excludes_donor_actor(self):
        rng=np.random.default_rng(13);background=rng.integers(0,256,(160,192,3),dtype=np.uint8)
        background=cv2.GaussianBlur(background,(3,3),.5)
        target=background.copy();donor=background.copy();mask=np.zeros((160,192),np.float32);donormask=mask.copy()
        mask[50:100,60:100]=1;donormask[50:100,115:155]=1
        target[mask>0]=[255,0,0];donor[donormask>0]=[0,255,0]
        plate,unknown,records=recover_background(target,mask,[(donor,donormask,9)])
        self.assertEqual(int(unknown.sum()),0);self.assertGreater(records[0]['inliers'],8)
        np.testing.assert_array_equal(plate[mask==0],target[mask==0])
        self.assertLess(np.abs(plate[mask>0].astype(float)-background[mask>0]).mean(),.5)
