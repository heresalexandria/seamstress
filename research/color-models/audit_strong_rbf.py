"""Saved strong-RBF safety study. No fitting, renderer changes, or video output."""
from audit_rbf import *
import hashlib
MODELDIR=ROOT/'research/local-color-strong'


def extrema(c,xy,m):
    out=predict(c,xy,m);jac=[]
    for ch in range(3):
        plus=c.copy();minus=c.copy();plus[:,ch]=np.minimum(c[:,ch]+.25,255);minus[:,ch]=np.maximum(c[:,ch]-.25,0)
        jac.append((predict(plus,xy,m)-predict(minus,xy,m))/(plus[:,ch]-minus[:,ch])[:,None])
    jac=np.stack(jac,-1);diag=np.diagonal(jac,axis1=1,axis2=2);sing=np.linalg.svd(jac,compute_uv=False);det=np.linalg.det(jac)
    i,ch=np.unravel_index(np.argmin(diag),diag.shape);j=int(np.argmin(sing[:,-1]))
    # Identity/residual mixing: all quantities refer to the actual fitted
    # transform; attenuation is evaluated only as a diagnostic if needed.
    scales={}
    for alpha in [1.,.85,.75,.5]:
        ja=np.eye(3)+alpha*(jac-np.eye(3));ss=np.linalg.svd(ja,compute_uv=False)
        scales[str(alpha)]={'min_own_derivative':float(np.diagonal(ja,axis1=1,axis2=2).min()),'min_singular':float(ss[:,-1].min()),'min_determinant':float(np.linalg.det(ja).min())}
    return {'min_own_derivative':float(diag.min()),'min_singular':float(sing[:,-1].min()),'max_singular':float(sing[:,0].max()),'min_determinant':float(det.min()),'negative_own_derivative_fraction':float((diag<0).mean()),'nonpositive_determinant_fraction':float((det<=0).mean()),'max_abs_change':float(abs(out-c).max()),'p99_abs_change':float(np.percentile(abs(out-c),99)),'clip_fraction':float(((out<-.0001)|(out>255.0001)).mean()),'worst_own_derivative':{'input_rgb':c[i].tolist(),'xy':xy[i].tolist(),'output_rgb':out[i].tolist(),'channel':int(ch),'jacobian':jac[i].tolist()},'worst_singular':{'input_rgb':c[j].tolist(),'xy':xy[j].tolist(),'output_rgb':out[j].tolist(),'jacobian':jac[j].tolist()},'attenuation_diagnostics':scales}


def shadow_audit(c,xy,m):
    use=(c@np.array([.2126,.7152,.0722])<24)&(c.max(1)<60)
    if not use.any():return {'samples':0}
    c=c[use];xy=xy[use];out=predict(c,xy,m)
    lum=c@[.2126,.7152,.0722];newlum=out@[.2126,.7152,.0722];ratio=newlum/np.maximum(lum,1)
    result=extrema(c,xy,m)
    result.update(samples=len(c),median_luma_gain=float(np.median(ratio[lum>=2])) if (lum>=2).any() else None,p01_luma_gain=float(np.percentile(ratio[lum>=2],1)) if (lum>=2).any() else None,p99_luma_gain=float(np.percentile(ratio[lum>=2],99)) if (lum>=2).any() else None,p99_abs_luma_change=float(np.percentile(abs(newlum-lum),99)))
    return result


def main():
    paths=sorted(MODELDIR.glob('*-model.json'),key=lambda x:int(x.name.split('-')[0]));start=time.perf_counter()
    old=json.loads((ROOT/'research/local-color-fit/all-models.json').read_text());old={m['frame']:m for m in old['local_color_curves']}
    levels=[0,1,2,4,8,12,16,24,32,48,64,96,128,160,192,224,255]
    palette=np.stack(np.meshgrid(*[levels]*3,indexing='ij'),-1).reshape(-1,3).astype(float)
    positions=np.stack(np.meshgrid(np.linspace(0,1,7),np.linspace(0,1,5),indexing='xy'),-1).reshape(-1,2)
    c=np.tile(palette,(len(positions),1));xy=np.repeat(positions,len(palette),axis=0)
    output={'saved_model_fitting':False,'grid_levels':levels,'spatial_grid':[7,5],'native_sampling':'Every fourth original pixel after unchanged baseline geometry and LUT; no RGB resizing','color_jacobian_step':.25,'reports':[]}
    for path in paths:
        model=json.loads(path.read_text());cut=model['frame'];frames=read_frames(ROOT/'IYTYT.mp4',cut-1,2);record={'frame':cut,'model_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'sides':{}}
        for side,n,image in [('left',cut-1,frames[0]),('right',cut,frames[1])]:
            m=model[side];base=p.baseline(image,n);h,w=base.shape[:2];yy,xx=np.mgrid[0:h:4,0:w:4]
            native=base[::4,::4].reshape(-1,3).astype(float);nxy=np.stack([xx/(w-1),yy/(h-1)],-1).reshape(-1,2)
            rec={'limit':m['limit'],'broad_grid':extrema(c,xy,m),'native_anchor':extrema(native,nxy,m),'native_motion':eval_points(native,nxy,m,True),'native_shadows':shadow_audit(native,nxy,m),'v1_native_shadows':shadow_audit(native,nxy,old[cut][side])}
            record['sides'][side]=rec
            print(cut,side,'grid',rec['broad_grid']['min_own_derivative'],rec['broad_grid']['min_singular'],'native',rec['native_anchor']['min_own_derivative'],rec['native_anchor']['min_singular'],'shadow gain',rec['native_shadows'].get('median_luma_gain'),flush=True)
        output['reports'].append(record);output['runtime_seconds']=time.perf_counter()-start
        (OUT/'strong-rbf-audit.json').write_text(json.dumps(output,indent=2))
    print('DONE',time.perf_counter()-start,flush=True)

if __name__=='__main__':main()
