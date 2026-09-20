"""Read-only native/gamut/temporal audit of saved directional color models."""
import audit_strong_rbf as q
from audit_strong_rbf import ROOT,OUT,p,read_frames,np,json,Path,time,hashlib
import directional_local_color as d

MODELDIR=ROOT/'research/local-color-directional'
def predict(c,xy,m):
    return np.concatenate([d.apply_samples(c[i:i+8192],xy[i:i+8192],m) for i in range(0,len(c),8192)])
q.predict=predict

def motion(c,xy,m):
    out=predict(c,xy,m);result={}
    for label,shift in [('16x8',[16/1279,8/719]),('32x16',[32/1279,16/719])]:
        change=predict(c,np.clip(xy+shift,0,1),m)-out
        result[label]={'max':float(abs(change).max()),'p99':float(np.percentile(abs(change),99))}
    result['max_quintic_weight_drift_byte_per_frame']=float(abs(out-c).max()*1.875/168)
    return result

def native_sample(n):
    frame=read_frames(ROOT/'IYTYT.mp4',n,1)[0];base=p.baseline(frame,n);h,w=base.shape[:2];yy,xx=np.mgrid[0:h:4,0:w:4]
    return base[::4,::4].reshape(-1,3).astype(float),np.stack([xx/(w-1),yy/(h-1)],-1).reshape(-1,2)

def main():
    levels=[0,1,2,4,8,12,16,24,32,48,64,96,128,160,192,224,255]
    palette=np.stack(np.meshgrid(*[levels]*3,indexing='ij'),-1).reshape(-1,3).astype(float)
    positions=np.stack(np.meshgrid(np.linspace(0,1,7),np.linspace(0,1,5),indexing='xy'),-1).reshape(-1,2)
    c=np.tile(palette,(len(positions),1));xy=np.repeat(positions,len(palette),axis=0)
    result={'method':'Saved directional-gamut-v2 models, no fitting','jacobian_difference':'±0.25byte interior; one-sided at RGB0/255, always within valid gamut','grid_levels':levels,'spatial_grid':[7,5],'native_pixels':'Every fourth row/column, baseline geometry and LUT unchanged; no resizing before color sampling','reports':[]}
    start=time.perf_counter()
    for path in sorted(MODELDIR.glob('*-model.json'),key=lambda a:int(a.name.split('-')[0])):
        m=json.loads(path.read_text());cut=m['frame'];rec={'frame':cut,'model_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'sides':{}}
        for side,n,away in [('left',cut-1,cut-85),('right',cut,cut+84)]:
            model=m[side];color,pos=native_sample(n);away_c,away_pos=native_sample(away)
            g=q.extrema(c,xy,model);a=q.extrema(color,pos,model);t=q.extrema(away_c,away_pos,model);away_out=away_c+.5*(predict(away_c,away_pos,model)-away_c)
            shadow=q.shadow_audit(color,pos,model)
            checks={'grid':g,'native_anchor':a,'native_shadows':shadow,'native_motion':motion(color,pos,model),'away_frame':away,'away_full_strength':t,'away_effective_half_strength':t['attenuation_diagnostics']['0.5'],'away_effective_max_change':float(abs(away_out-away_c).max()),'away_effective_p99_change':float(np.percentile(abs(away_out-away_c),99)),'black_exact':bool(np.all(predict(np.zeros((len(positions),3)),positions,model)==0)),'white_exact':bool(np.all(predict(np.full((len(positions),3),255.),positions,model)==255))}
            rec['sides'][side]=checks
            print(cut,side,'grid min own/singular/det',g['min_own_derivative'],g['min_singular'],g['min_determinant'],'native',a['min_own_derivative'],a['min_singular'],a['min_determinant'],'away raw',t['min_own_derivative'],t['min_singular'],t['min_determinant'],'shadow gain',shadow.get('median_luma_gain'),flush=True)
        result['reports'].append(rec);result['runtime_seconds']=time.perf_counter()-start
        (OUT/'directional-rbf-audit.json').write_text(json.dumps(result,indent=2))
    print('DONE',time.perf_counter()-start,flush=True)

if __name__=='__main__':main()
