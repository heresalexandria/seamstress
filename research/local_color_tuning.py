"""Small spatial-support study with independent cross-pair validation."""
import json,sys
import numpy as np,cv2
import local_color_probe as p

cut=int(sys.argv[1]) if len(sys.argv)>1 else 361
frames=p.read_frames(p.ROOT/'IYTYT.mp4',cut-3,6)
base=[p.baseline(f,cut-3+i) for i,f in enumerate(frames)]
pairs=[p.observations(base[a],base[b]) for a,b in [(2,3),(1,4),(0,5)]]
# Fit adjacent frames only; validate both held-out locations and independent
# pictures up to three source frames away. No neighbor data is in the fit.
l,r,lp,rp,train=pairs[0];target=(l+r)/2
report=[];models={}
for label,scale,k,ridge in [('broad',[48,48,48,.4,.4],48,.015),('local',[48,48,48,.2,.2],64,.015),('local-light-prior',[48,48,48,.2,.2],64,.003),('color-detail',[32,32,32,.3,.3],64,.008),('local-fine',[48,48,48,.12,.12],96,.003),('local-fine-regularized',[48,48,48,.12,.12],96,.01)]:
    feature=p.features(target[train],(lp[train]+rp[train])/2,'hybrid',scale)
    cv2.setRNGSeed(19)
    _,_,centers=cv2.kmeans(feature,k,None,(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,60,.02),3,cv2.KMEANS_PP_CENTERS)
    left=p.fit(l[train],lp[train],target[train],'hybrid',centers,scale,ridge)
    right=p.fit(r[train],rp[train],target[train],'hybrid',centers,scale,ridge)
    checks=[]
    for i,(a,b,ap,bp,tr) in enumerate(pairs):
        x=p.apply_samples(a,ap,left);y=p.apply_samples(b,bp,right)
        use=~tr if i==0 else np.ones(len(a),bool)
        err=x-y;raw=a-b
        tile=(ap[:,1]*9).astype(int)*16+(ap[:,0]*16).astype(int)
        means=[];prior=[]
        for t in np.unique(tile):
            mask=(tile==t)&use
            if mask.sum()<30:continue
            means.append(np.median(err[mask],axis=0));prior.append(np.median(raw[mask],axis=0))
        checks.append({'pair':i,'holdout_mae_before':float(abs(raw[use]).mean()),'holdout_mae_after':float(abs(err[use]).mean()),'regional_signed_bias_before':float(abs(np.array(prior)).mean()),'regional_signed_bias_after':float(abs(np.array(means)).mean()),'maximum_channel_change':float(max(abs(x-a).max(),abs(y-b).max()))})
    report.append({'label':label,'checks':checks});models[label]={'frame':cut,'left':left,'right':right,'support_before':168,'support_after':168}
    print(report[-1],flush=True)
(p.OUT/f'{cut}-tuning-report.json').write_text(json.dumps(report,indent=2))
(p.OUT/f'{cut}-tuning-models.json').write_text(json.dumps(models,indent=2))
