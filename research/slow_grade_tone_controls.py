"""Check black/white-preserving monotone tone curves for long grade tracks."""
from __future__ import annotations
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from scipy.interpolate import PchipInterpolator
from segment_color import OUT, SOURCE, CUTS, read_frames, matches
from slow_segment_grade import transform, HALF_SUPPORT

KNOTS=np.array([0.,16.,48.,96.,144.,192.,240.,255.])


def curves(gain,bias):
    target=KNOTS[:,None]*gain+bias
    target[0]=0.;target[-1]=255.
    if np.any(np.diff(target,axis=0)<=0):
        raise ValueError("A stronger grade requires constrained monotone knot fitting")
    return [PchipInterpolator(KNOTS,target[:,c]) for c in range(3)]


def mapped(rgb,curve):
    return np.column_stack([curve[c](rgb[:,c]) for c in range(3)])


def main():
    estimates=json.loads((OUT/'estimates.json').read_text());rows=[]
    for cut in CUTS:
        images=read_frames(SOURCE,cut-1,2,size=(640,360))
        x,y,train,_,coverage=matches(images[0],images[1])
        gl,bl=transform(cut-1,estimates);gr,br=transform(cut,estimates)
        left,right=curves(gl,bl),curves(gr,br)
        raw=abs(y[~train]-x[~train])
        affine=abs(y[~train]*gl+bl-(x[~train]*gr+br))
        tone=abs(mapped(y[~train],left)-mapped(x[~train],right))
        row={'cut':cut,'raw_heldout_mae':float(raw.mean()),'symmetric_affine_heldout_mae':float(affine.mean()),
             'protected_tone_heldout_mae':float(tone.mean()),'tone_min_derivative':min(float(c.derivative()(np.linspace(0,255,1025)).min()) for c in left+right)}
        rows.append(row);print(row,flush=True)
    # Verify every track sample remains strictly ordered, maps 0→0 and 255→255,
    # and compute worst introduced change on a dense tonal grid.
    palette=np.arange(256,dtype=float)[:,None]*np.ones((1,3));peaks=[]
    for cut in CUTS:
        peak=0.; previous=None
        for frame in range(max(0,cut-HALF_SUPPORT-2),min(3347,cut+HALF_SUPPORT+3)):
            g,b=transform(frame,estimates);lut=mapped(palette,curves(g,b))
            if not (np.min(np.diff(lut,axis=0))>0 and np.allclose(lut[0],0) and np.allclose(lut[-1],255)):
                raise ValueError('Nonmonotone or nonanchored tone map')
            if previous is not None and frame!=cut:peak=max(peak,float(abs(lut-previous).max()))
            previous=lut
        peaks.append({'cut':cut,'peak_per_frame_tone_change':peak})
    (OUT/'slow-grade-tone-controls.json').write_text(json.dumps({'knots':KNOTS.tolist(),'seam_errors':rows,'track_peaks':peaks},indent=2))


if __name__=='__main__':main()
