"""Bounded original-pixel grade curves with exact shared-midpoint semantics."""
from __future__ import annotations
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from scipy.interpolate import PchipInterpolator
from PIL import Image,ImageDraw
from segment_color import OUT, SOURCE, CUTS, read_frames, matches, VideoWriter
from slow_segment_grade import HALF_SUPPORT, smoothstep

KNOTS=np.array([0.,16.,48.,96.,144.,192.,240.,255.])


def build(estimate):
    gain,bias=np.array(estimate['gain']),np.array(estimate['bias'])
    target=KNOTS[:,None]*gain+bias;target[0]=0;target[-1]=255
    if np.any(np.diff(target,axis=0)<=0):raise ValueError('Tone knots are not monotone')
    left,right=[],[];checks=[];levels=np.arange(256,dtype=float)
    for c in range(3):
        transfer=PchipInterpolator(KNOTS,target[:,c])
        dense=np.linspace(0,255,65537);mapped=transfer(dense)
        if np.any(np.diff(mapped)<=0):raise ValueError('Transfer is not strictly monotone')
        inverse=PchipInterpolator(mapped,dense)
        l=.5*(levels+inverse(levels));r=.5*(levels+transfer(levels))
        left.append(l);right.append(r)
        probe=np.linspace(0,255,4097)
        checks.append(float(np.max(abs(.5*(transfer(probe)+inverse(transfer(probe)))-.5*(probe+transfer(probe))))))
    l,r=np.stack(left,axis=1),np.stack(right,axis=1)
    for lut in (l,r):
        assert np.all(np.diff(lut,axis=0)>0)
        assert np.allclose(lut[0],0) and np.allclose(lut[-1],255)
        assert np.min(lut)>=0 and np.max(lut)<=255
    return {'frame':estimate['frame'],'support_before':HALF_SUPPORT,'support_after':HALF_SUPPORT,
            'left_lut':l.tolist(),'right_lut':r.tolist()}, {'continuous_consistency_max_byte_error':max(checks)}


def evaluate(values,lut):
    return np.stack([np.interp(values[...,c],np.arange(256),np.array(lut)[:,c]) for c in range(3)],axis=-1)


def grade(frame_index,image,curves):
    active=[]
    for curve in curves:
        cut=curve['frame'];side='left' if frame_index<cut else 'right'
        distance=cut-1-frame_index if side=='left' else frame_index-cut
        support=curve['support_before'] if side=='left' else curve['support_after']
        if 0<=distance<=support:
            active.append(curve)
            if len(active)>1:raise ValueError('Supports overlap')
            w=1-smoothstep(distance/support)
            transformed=evaluate(image,curve[side+'_lut'])
            # Interpolate a pointwise transfer function, not images or poses.
            image=image.astype(float)+w*(transformed-image)
    return np.rint(image).astype(np.uint8)


def preview(curves,cut):
    start=cut-HALF_SUPPORT-6;count=2*HALF_SUPPORT+13
    destination=OUT/f'{cut}-slow-grade-comparison.mp4'
    if destination.exists():raise FileExistsError(destination)
    # Stream one native source decode, retaining timing and spatial samples.
    from seamstress.media import iter_frames
    with VideoWriter(destination,2560,752,'24000/1001',crf=16) as writer:
        for index,original in enumerate(iter_frames(SOURCE,start,count)):
            corrected=grade(start+index,original,curves)
            canvas=Image.new('RGB',(2560,752),'#141922')
            canvas.paste(Image.fromarray(original),(0,32));canvas.paste(Image.fromarray(corrected),(1280,32))
            draw=ImageDraw.Draw(canvas)
            draw.text((16,10),f'ORIGINAL / frame {start+index}',fill='white')
            draw.text((1296,10),'SLOW BOUNDED COLOR ONLY / original geometry and poses',fill='white')
            writer.write(np.array(canvas))
    return str(destination)


def main():
    estimates=json.loads((OUT/'estimates.json').read_text());curves=[];results=[]
    for estimate in estimates:
        curve,checks=build(estimate);curves.append(curve);cut=curve['frame']
        frames=read_frames(SOURCE,cut-1,2,size=(640,360))
        x,y,train,_,coverage=matches(frames[0],frames[1])
        left,right=evaluate(y,curve['left_lut']),evaluate(x,curve['right_lut'])
        residual=abs(left[~train]-right[~train]);raw=abs(y[~train]-x[~train])
        # Probe exact relation L=T(R) after LUT discretization, not just the
        # dense continuous inverse used to construct the two maps.
        levels=np.linspace(0,255,4097)[:,None]*np.ones((1,3))
        target=KNOTS[:,None]*estimate['gain']+estimate['bias'];target[0]=0;target[-1]=255
        transformed=np.stack([PchipInterpolator(KNOTS,target[:,c])(levels[:,c]) for c in range(3)],axis=1)
        discrete_error=abs(evaluate(transformed,curve['left_lut'])-evaluate(levels,curve['right_lut']))
        peak=0
        for side in ['left_lut','right_lut']:
            delta=np.array(curve[side])-np.arange(256)[:,None]
            weights=1-smoothstep(np.arange(HALF_SUPPORT+1)/HALF_SUPPORT)
            peak=max(peak,float(np.max(abs(np.diff(weights)))*abs(delta).max()))
        result={'frame':cut,'raw_heldout_mae':float(raw.mean()),'corrected_heldout_mae':float(residual.mean()),
                'raw_heldout_p90':float(np.percentile(raw,90)),'corrected_heldout_p90':float(np.percentile(residual,90)),
                'new_clipping_fraction':0.,'min_lut_increment':min(float(np.diff(curve[k],axis=0).min()) for k in ['left_lut','right_lut']),
                'max_lut_roundtrip_consistency_error':float(discrete_error.max()),'peak_added_byte_change_per_frame':peak,**checks}
        results.append(result);print(result,flush=True)
    for a,b in zip(curves,curves[1:]):
        assert a['frame']+a['support_after']<b['frame']-1-b['support_before']
    (OUT/'protected-midpoint-curves.json').write_text(json.dumps(curves,indent=2))
    (OUT/'protected-midpoint-evaluation.json').write_text(json.dumps(results,indent=2))
    print('LUTS_READY',OUT/'protected-midpoint-curves.json',flush=True)
    print('PREVIEW',preview(curves,722),flush=True)


if __name__=='__main__':main()
