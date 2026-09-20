"""Test stronger bounded near-black response without changing geometry.

The v1 limit18 cannot match clipped/near-zero colors. This study keeps the
same range-preserving model but fits the actual nonlinear response and chooses
shared targets inside the two attainable color intervals where possible.
"""
import sys,json
from pathlib import Path
import numpy as np,cv2
import local_color_probe as p
from build_local_color import check
OUT=p.ROOT/'research/local-color-strong';OUT.mkdir(exist_ok=True)
LIMIT=63.75;SCALE=[48,48,48,.2,.2]


def targets(l,r):
    lh=4*(l/255)*(1-l/255);rh=4*(r/255)*(1-r/255)
    cap=.99*LIMIT
    lower=np.maximum(l-cap*lh,r-cap*rh);upper=np.minimum(l+cap*lh,r+cap*rh)
    common=np.minimum(np.maximum((l+r)/2,lower),upper)
    # If intervals do not overlap, each side approaches its closest feasible
    # value. A true zero channel stays zero under this model.
    left=np.clip(common,l-cap*lh,l+cap*lh)
    right=np.clip(common,r-cap*rh,r+cap*rh)
    return left,right


def fit(rgb,xy,target,centers,weights):
    b=p.basis(p.features(rgb,xy,'hybrid',SCALE),centers).astype(float)
    h=4*(rgb/255)*(1-rgb/255);delta=target-rgb;coefs=[]
    ridge=np.eye(len(centers))*(len(rgb)/len(centers)*.0001)
    for c in range(3):
        a=b*h[:,c,None];co=np.linalg.solve(a.T@(a*weights[:,None])+ridge,a.T@(weights*delta[:,c]))
        for iteration in range(25):
            z=b@co/LIMIT;t=np.tanh(z);prediction=LIMIT*t*h[:,c]
            error=delta[:,c]-prediction
            robust=np.minimum(1,2/np.maximum(abs(error),1e-6));w=weights*robust
            jac=b*(h[:,c]*(1-t*t))[:,None]
            step=np.linalg.solve(jac.T@(jac*w[:,None])+ridge,jac.T@(w*error)-ridge@co)
            def loss(v):
                e=delta[:,c]-LIMIT*np.tanh(b@v/LIMIT)*h[:,c]
                ae=abs(e);return np.sum(weights*np.where(ae<2,.5*e*e,2*(ae-1)))+.5*v@ridge@v
            prior=loss(co);rate=1
            while rate>1/128 and loss(co+rate*step)>prior:rate*=.5
            co+=rate*step
            if np.max(abs(rate*step))<.002:break
        coefs.append(co)
    return {'mode':'hybrid','feature_scale':SCALE,'centers':centers.tolist(),'coefficients':np.stack(coefs,axis=1).tolist(),'limit':LIMIT}


def run(cut):
    raw=p.read_frames(p.ROOT/'IYTYT.mp4',cut-3,6)
    frames=[p.baseline(f,cut-3+i) for i,f in enumerate(raw)]
    pairs=[p.observations(frames[a],frames[b]) for a,b in [(2,3),(1,4),(0,5)]]
    a,b,ap,bp,tr=pairs[0];mid=(a+b)/2;position=(ap+bp)/2
    # Balance coarse palette bins so small outfits/hair receive evidence
    # weight as well as the large flat backgrounds. Cap weights for noise.
    bins=np.floor(mid[tr]/24).astype(int);keys=bins[:,0]*121+bins[:,1]*11+bins[:,2]
    _,inverse,counts=np.unique(keys,return_inverse=True,return_counts=True)
    weights=np.clip(np.sqrt(np.median(counts)/counts[inverse]),.25,8);weights/=weights.mean()
    feature=p.features(mid[tr],position[tr],'hybrid',SCALE)
    rng=np.random.default_rng(221);chosen=rng.choice(len(feature),len(feature),replace=True,p=weights/weights.sum())
    cv2.setRNGSeed(19)
    _,_,centers=cv2.kmeans(feature[chosen],64,None,(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,60,.02),3,cv2.KMEANS_PP_CENTERS)
    lt,rt=targets(a,b)
    left=fit(a[tr],ap[tr],lt[tr],centers,weights);right=fit(b[tr],bp[tr],rt[tr],centers,weights)
    record={'frame':cut,'support_before':168,'support_after':168,'left':left,'right':right}
    report={'frame':cut,'limit':LIMIT,'checks':[check(pair,left,right,i==0) for i,pair in enumerate(pairs)],
            'maximum_coefficient':max(float(abs(np.array(x['coefficients'])).max()) for x in [left,right]),
            'training':'Adjacentpair/alternating spatialtiles only; palette-balanced; nonlinear Huber+ridge with feasible common targets',
            'status':'Research only; requires native ROI and color Jacobian review'}
    (OUT/f'{cut}-model.json').write_text(json.dumps(record,indent=2));(OUT/f'{cut}-report.json').write_text(json.dumps(report,indent=2));print(report,flush=True)


if __name__=='__main__':
    for cut in map(int,sys.argv[1:]):run(cut)
