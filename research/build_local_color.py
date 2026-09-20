"""Fit fixed color-only corrections against the committed conform baseline.

Held-out spatial tiles and neighboring source pairs are never used in fitting.
Source framing, poses, timing, global grades, and original media are unchanged.
"""
from pathlib import Path
import json,sys,hashlib
import cv2,numpy as np
import local_color_probe as p

OUT=p.ROOT/'research/local-color-fit';OUT.mkdir(exist_ok=True)
SCALE=[48,48,48,.2,.2]


def check(pair,left,right,use_heldout):
    a,b,ap,bp,training=pair
    x=p.apply_samples(a,ap,left);y=p.apply_samples(b,bp,right)
    use=~training if use_heldout else np.ones(len(a),bool)
    raw=a-b;err=x-y;tile=(ap[:,1]*9).astype(int)*16+(ap[:,0]*16).astype(int)
    prior=[];after=[]
    for t in np.unique(tile):
        mask=(tile==t)&use
        if mask.sum()<30:continue
        prior.append(np.median(raw[mask],axis=0));after.append(np.median(err[mask],axis=0))
    return {'samples':int(use.sum()),'mae_before':float(abs(raw[use]).mean()),'mae_after':float(abs(err[use]).mean()),
            'regional_signed_bias_before':float(abs(np.array(prior)).mean()),'regional_signed_bias_after':float(abs(np.array(after)).mean()),
            'peak_matched_channel_change':float(max(abs(x-a).max(),abs(y-b).max()))}


def fit_cut(cut):
    raw=p.read_frames(p.ROOT/'IYTYT.mp4',cut-3,6)
    frames=[p.baseline(f,cut-3+i) for i,f in enumerate(raw)]
    pairs=[p.observations(frames[a],frames[b]) for a,b in [(2,3),(1,4),(0,5)]]
    a,b,ap,bp,train=pairs[0];target=(a+b)/2
    x=p.features(target[train],(ap[train]+bp[train])/2,'hybrid',SCALE)
    cv2.setRNGSeed(19)
    _,_,centers=cv2.kmeans(x,64,None,(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,60,.02),3,cv2.KMEANS_PP_CENTERS)
    left=p.fit(a[train],ap[train],target[train],'hybrid',centers,SCALE,.003)
    right=p.fit(b[train],bp[train],target[train],'hybrid',centers,SCALE,.003)
    checks=[check(pair,left,right,i==0) for i,pair in enumerate(pairs)]
    record={'frame':cut,'support_before':168,'support_after':168,'left':left,'right':right}
    report={'frame':cut,'checks':checks,'geometry_changed':False,'fit_frames':[cut-1,cut],
            'validation_frames':[[cut-2,cut+1],[cut-3,cut+2]],
            'all_validation_biases_improve':all(c['regional_signed_bias_after']<c['regional_signed_bias_before'] for c in checks),
            'all_validation_maes_improve':all(c['mae_after']<c['mae_before'] for c in checks),
            'status':'Numerical color experiment; native/temporal review required'}
    (OUT/f'{cut}-model.json').write_text(json.dumps(record,indent=2))
    (OUT/f'{cut}-report.json').write_text(json.dumps(report,indent=2))
    print(report,flush=True)
    return record,report


def main():
    cuts=[int(x) for x in sys.argv[1:]] if len(sys.argv)>1 else [s['frame'] for s in p.PLAN['seams']]
    models=[];reports=[]
    for cut in cuts:
        model,report=fit_cut(cut);models.append(model);reports.append(report)
    baseline=p.ROOT/'plans/IYTYT-eight-joins.json'
    data={'baseline_commit':'f4ea566','baseline_plan_sha256':hashlib.sha256(baseline.read_bytes()).hexdigest(),
          'source_sha256':p.PLAN['source_sha256'],'method':'bounded color/position RBF residual after protected baseline LUT',
          'local_color_curves':models,'reports':reports,'status':'Experimental; not a perceptual guarantee'}
    (OUT/'all-models.json').write_text(json.dumps(data,indent=2))


if __name__=='__main__':main()
