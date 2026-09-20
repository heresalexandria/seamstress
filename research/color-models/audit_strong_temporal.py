"""Native ±84-frame extrapolation of saved strong color models; no fitting."""
from audit_strong_rbf import *

output={'offset_frames':84,'effective_residual_weight':.5,'note':'Halfway through168frame quintic support. Raw full-strength diagnostics expose extrapolation; effective diagnostics apply the actual0.5 residual weight.','reports':[]}
for path in sorted(MODELDIR.glob('*-model.json'),key=lambda p:int(p.name.split('-')[0])):
    model=json.loads(path.read_text());cut=model['frame'];rec={'frame':cut,'sides':{}}
    for side,n in [('left',cut-85),('right',cut+84)]:
        image=read_frames(ROOT/'IYTYT.mp4',n,1)[0];base=p.baseline(image,n);h,w=base.shape[:2];yy,xx=np.mgrid[0:h:4,0:w:4]
        c=base[::4,::4].reshape(-1,3).astype(float);xy=np.stack([xx/(w-1),yy/(h-1)],-1).reshape(-1,2);m=model[side]
        e=extrema(c,xy,m);out=predict(c,xy,m);cf=confidence(c,xy,m);use=(c@[.2126,.7152,.0722]<24)&(c.max(1)<60);luma=c[use]@[.2126,.7152,.0722];effective=(c+.5*(out-c));eluma=effective[use]@[.2126,.7152,.0722]
        rec['sides'][side]={'source_frame':n,'full_strength':e,'effective_half_strength':e['attenuation_diagnostics']['0.5'],'effective_max_change':float(abs(effective-c).max()),'effective_p99_change':float(np.percentile(abs(effective-c),99)),'effective_shadow_median_luma_gain':float(np.median((eluma/np.maximum(luma,1))[luma>=2])) if (luma>=2).any() else None,'fraction_confidence_lt_0.1':float((cf<.1).mean())}
        print(cut,side,'raw min singular',e['min_singular'],'half',e['attenuation_diagnostics']['0.5']['min_singular'],'shadowgain',rec['sides'][side]['effective_shadow_median_luma_gain'],flush=True)
    output['reports'].append(rec);(OUT/'strong-temporal-audit.json').write_text(json.dumps(output,indent=2))
