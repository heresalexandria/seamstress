"""Disambiguate gamut-boundary flatness from color folds in directional models."""
from audit_directional import *

def evaluate(c,xy,m):
    out=predict(c,xy,m);jac=[]
    for ch in range(3):
        plus=c.copy();minus=c.copy();plus[:,ch]=np.minimum(c[:,ch]+.25,255);minus[:,ch]=np.maximum(c[:,ch]-.25,0)
        jac.append((predict(plus,xy,m)-predict(minus,xy,m))/(plus[:,ch]-minus[:,ch])[:,None])
    jac=np.stack(jac,-1);diag=np.diagonal(jac,axis1=1,axis2=2);det=np.linalg.det(jac);sing=np.linalg.svd(jac,compute_uv=False);interior=((c>=8)&(c<=247));strict=interior.all(1);active_min=[]
    # Restrict the differential to input channels having >=8codevalues of
    # room at both boundaries. Output columns remain all3channels.
    bits=interior@np.array([1,2,4])
    for code in range(1,8):
        use=bits==code
        if not use.any():continue
        channels=[i for i in range(3) if code&(1<<i)];restricted=jac[use][:,:,channels]
        active_min.append(float(np.linalg.svd(restricted,compute_uv=False)[:,-1].min()))
    result={'sample_count':len(c),'min_active_own_derivative':float(diag[interior].min()) if interior.any() else None,'min_active_column_singular':min(active_min) if active_min else None,'strict_rgb_interior_sample_count':int(strict.sum()),'strict_rgb_interior_min_singular':float(sing[strict,-1].min()) if strict.any() else None,'strict_rgb_interior_min_determinant':float(det[strict].min()) if strict.any() else None,'fraction_negative_own_derivative':float((diag< -1e-5).mean()),'fraction_negative_determinant':float((det< -1e-5).mean()),'min_gray_direction_luma_derivative':float((jac.sum(-1)@np.array([.2126,.7152,.0722])).min()),'max_change':float(abs(out-c).max()),'p99_change':float(np.percentile(abs(out-c),99))}
    k=int(np.argmin(sing[strict,-1])) if strict.any() else 0
    if strict.any():
        i=np.where(strict)[0][k];result['worst_strict_rgb_interior']={'input':c[i].tolist(),'xy':xy[i].tolist(),'output':out[i].tolist(),'jacobian':jac[i].tolist()}
    return result

levels=[8,16,32,64,96,128,160,192,224,247]
palette=np.stack(np.meshgrid(*[levels]*3,indexing='ij'),-1).reshape(-1,3).astype(float)
positions=np.stack(np.meshgrid(np.linspace(0,1,7),np.linspace(0,1,5),indexing='xy'),-1).reshape(-1,2)
c=np.tile(palette,(len(positions),1));xy=np.repeat(positions,len(palette),axis=0)
paths=sorted(MODELDIR.glob('*-model.json'),key=lambda a:int(a.name.split('-')[0]))+[ROOT/'research/local-color-directional-fine/3240-model.json']
result={'rgb_interior_definition':'Every input channel8..247; excludes asymptotic boundary flatness','active_columns':'Input channels8..247, all output channels retained','reports':[]}
for path in paths:
    model=json.loads(path.read_text());cut=model['frame'];label=('fine-' if 'fine' in str(path.parent) else '')+str(cut);rec={'label':label,'frame':cut,'model_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'sides':{}}
    for side,n,away in [('left',cut-1,cut-85),('right',cut,cut+84)]:
        rgb,pos=native_sample(n);awayrgb,awaypos=native_sample(away);m=model[side]
        d={'interior_grid':evaluate(c,xy,m),'native_anchor':evaluate(rgb,pos,m),'native_away_full_strength':evaluate(awayrgb,awaypos,m)}
        if label.startswith('fine'):
            d['native_motion']=motion(rgb,pos,m);d['native_shadows']=q.shadow_audit(rgb,pos,m)
        rec['sides'][side]=d
        print(label,side,'grid',d['interior_grid']['strict_rgb_interior_min_singular'],'nativeactive',d['native_anchor']['min_active_column_singular'],'awayactive',d['native_away_full_strength']['min_active_column_singular'],'negative',d['native_anchor']['fraction_negative_determinant'],flush=True)
    result['reports'].append(rec);(OUT/'directional-interior-audit.json').write_text(json.dumps(result,indent=2))
