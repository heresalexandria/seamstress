"""Research: color offsets with soft directional gamut bounds.

True RGB black and white are preserved. A zero channel of a saturated material
may move toward the shared color; matching never requires crushing both sides
toward that channel endpoint. No output geometry, blending, or generation.
"""
import sys,json
from pathlib import Path
import numpy as np,cv2
import local_color_probe as p
OUT=p.ROOT/'research/local-color-directional';OUT.mkdir(exist_ok=True)
LIMIT=18.;SCALE=[48,48,48,.2,.2]


def response(rgb,raw,limit=LIMIT,derivative=False):
    c=np.asarray(rgb,dtype=float)
    # Smoothly protect actual black and white, not saturated-color channels.
    lo=np.mean(c*c,axis=1,keepdims=True)
    hi=np.mean((255-c)**2,axis=1,keepdims=True)
    gate=(1-np.exp(-lo/64))*(1-np.exp(-hi/64))
    t=np.tanh(raw/limit);offset=limit*t*gate
    available=np.where(offset>=0,255-c,c)
    safe=np.maximum(available,1e-12);z=offset/safe
    correction=available*np.tanh(z)
    if derivative:
        return correction,(1-np.tanh(z)**2)*gate*(1-t*t)*(available>0)
    return correction


def apply_samples(rgb,xy,model):
    c=np.asarray(rgb,dtype=np.float32)
    b=p.basis(p.features(c,xy,'hybrid',model['feature_scale']),np.array(model['centers'],np.float32))
    return c+response(c,b@np.array(model['coefficients']),model['limit'])


def fit(rgb,xy,target,centers,weights,ridge_factor=.003):
    b=p.basis(p.features(rgb,xy,'hybrid',SCALE),centers).astype(float)
    delta=target-rgb
    ridge=np.eye(len(centers))*(len(rgb)/len(centers)*ridge_factor)
    co=np.linalg.solve(b.T@(b*weights[:,None])+ridge,b.T@(weights[:,None]*delta))
    for iteration in range(20):
        prediction,deriv=response(rgb,b@co,derivative=True)
        error=delta-prediction
        steps=[]
        for c in range(3):
            robust=np.minimum(1,2/np.maximum(abs(error[:,c]),1e-6));w=weights*robust
            jac=b*deriv[:,c,None]
            steps.append(np.linalg.solve(jac.T@(jac*w[:,None])+ridge,jac.T@(w*error[:,c])-ridge@co[:,c]))
        step=np.stack(steps,axis=1)
        def loss(v):
            e=delta-response(rgb,b@v);ae=abs(e)
            return np.sum(weights[:,None]*np.where(ae<2,.5*e*e,2*(ae-1)))+.5*np.sum(v*(ridge@v))
        prior=loss(co);rate=1
        while rate>1/128 and loss(co+rate*step)>prior:rate*=.5
        co+=rate*step
        if np.max(abs(rate*step))<.002:break
    return {'mode':'hybrid','response':'directional-gamut-v2','feature_scale':SCALE,'centers':centers.tolist(),'coefficients':co.tolist(),'limit':LIMIT}


def check(pair,left,right,use_heldout):
    a,b,ap,bp,training=pair
    x=apply_samples(a,ap,left);y=apply_samples(b,bp,right)
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


def run(cut,center_count=64,balance_exponent=.5,balance_cap=8,ridge_factor=.003):
    raw=p.read_frames(p.ROOT/'IYTYT.mp4',cut-3,6)
    frames=[p.baseline(f,cut-3+i) for i,f in enumerate(raw)]
    pairs=[p.observations(frames[a],frames[b]) for a,b in [(2,3),(1,4),(0,5)]]
    a,b,ap,bp,tr=pairs[0];mid=(a+b)/2;position=(ap+bp)/2
    bins=np.floor(mid[tr]/24).astype(int);keys=bins[:,0]*121+bins[:,1]*11+bins[:,2]
    _,inverse,counts=np.unique(keys,return_inverse=True,return_counts=True)
    weights=np.clip((np.median(counts)/counts[inverse])**balance_exponent,.25,balance_cap);weights/=weights.mean()
    feature=p.features(mid[tr],position[tr],'hybrid',SCALE)
    rng=np.random.default_rng(221);chosen=rng.choice(len(feature),len(feature),replace=True,p=weights/weights.sum())
    cv2.setRNGSeed(19)
    _,_,centers=cv2.kmeans(feature[chosen],center_count,None,(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,60,.02),3,cv2.KMEANS_PP_CENTERS)
    left=fit(a[tr],ap[tr],mid[tr],centers,weights,ridge_factor);right=fit(b[tr],bp[tr],mid[tr],centers,weights,ridge_factor)
    record={'frame':cut,'support_before':168,'support_after':168,'left':left,'right':right}
    report={'frame':cut,'limit':LIMIT,'fit_parameters':{'centers':center_count,'feature_scale':SCALE,'balance_exponent':balance_exponent,'balance_cap':balance_cap,'ridge_factor':ridge_factor},'checks':[check(pair,left,right,i==0) for i,pair in enumerate(pairs)],
            'maximum_coefficient':max(float(abs(np.array(x['coefficients'])).max()) for x in [left,right]),
            'training':'Adjacent pair alternating spatial tiles only; palette-balanced; nonlinear Huber+ridge, shared RGB midpoint',
            'status':'Research only; requires native ROI and color Jacobian review'}
    (OUT/f'{cut}-model.json').write_text(json.dumps(record,indent=2));(OUT/f'{cut}-report.json').write_text(json.dumps(report,indent=2));print(report,flush=True)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('cuts',nargs='+',type=int)
    parser.add_argument('--output-dir',type=Path,default=OUT)
    parser.add_argument('--centers',type=int,default=64)
    parser.add_argument('--color-scale',type=float,default=48)
    parser.add_argument('--position-scale',type=float,default=.2)
    parser.add_argument('--balance-exponent',type=float,default=.5)
    parser.add_argument('--balance-cap',type=float,default=8)
    parser.add_argument('--ridge-factor',type=float,default=.003)
    args=parser.parse_args();OUT=args.output_dir;OUT.mkdir(parents=True,exist_ok=True)
    SCALE=[args.color_scale]*3+[args.position_scale]*2
    for cut in args.cuts:run(cut,args.centers,args.balance_exponent,args.balance_cap,args.ridge_factor)
