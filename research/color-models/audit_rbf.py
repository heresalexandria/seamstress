"""Independent evaluation of a saved RBF model. Never fits or resamples output."""
from pathlib import Path
import json,sys,time
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'research'),str(ROOT)]
import cv2,numpy as np
from seamstress.media import read_frames
import local_color_probe as p
OUT=Path(__file__).parent
MODEL=ROOT/'research/local-color-probe/361-tuning-models.json'
LABEL='local-light-prior'
cv2.setNumThreads(2)


def predict(c,xy,m):
    return np.concatenate([p.apply_samples(c[i:i+8192],xy[i:i+8192],m) for i in range(0,len(c),8192)])


def confidence(c,xy,m):
    values=p.features(c,xy,m['mode'],m.get('feature_scale'));centers=np.array(m['centers'])
    parts=[]
    for i in range(0,len(c),8192):
        v=values[i:i+8192];d=np.maximum(np.sum(v*v,1)[:,None]+np.sum(centers*centers,1)[None,:]-2*v@centers.T,0)
        parts.append(np.exp(-np.maximum(d.min(1)-3,0)/3))
    return np.concatenate(parts)


def eval_points(c,xy,m,temporal=True):
    out=predict(c,xy,m);jac=[]
    for ch in range(3):
        delta=np.zeros_like(c);delta[:,ch]=.5
        jac.append((predict(c+delta,xy,m)-predict(c-delta,xy,m)))
    jac=np.stack(jac,-1);sing=np.linalg.svd(jac,compute_uv=False);det=np.linalg.det(jac);diag=np.diagonal(jac,axis1=1,axis2=2);conf=confidence(c,xy,m)
    report={}
    for label,use in [('all',np.ones(len(c),bool)),('supported_confidence_gt_0.5',conf>.5),('weak_confidence_lt_0.1',conf<.1)]:
        if not use.sum():continue
        report[label]={'count':int(use.sum()),'min_color_jacobian_diagonal':float(diag[use].min()),'min_color_jacobian_singular':float(sing[use,-1].min()),'max_color_jacobian_singular':float(sing[use,0].max()),'min_color_jacobian_determinant':float(det[use].min()),'fraction_nonpositive_diagonal':float((diag[use]<=0).mean()),'fraction_nonpositive_determinant':float((det[use]<=0).mean()),'max_abs_channel_change':float(abs(out-c)[use].max()),'p99_abs_channel_change':float(np.percentile(abs(out-c)[use],99)), 'clipped_channel_fraction':float(((out[use]<-.0001)|(out[use]>255.0001)).mean())}
    if not temporal:return report
    # Same color translated by 16px horizontally and8px vertically: isolates
    # correction field's drift from any actual scene/color/pose change.
    for name,shift in [('pan16x8',[16/1279,8/719]),('pan32x16',[32/1279,16/719])]:
        shifted=np.clip(xy+shift,0,1);difference=predict(c,shifted,m)-out
        report[name]={'p95_abs_channel_change':float(np.percentile(abs(difference),95)),'p99_abs_channel_change':float(np.percentile(abs(difference),99)),'max_abs_channel_change':float(abs(difference).max())}
    # No temporal fit: if the residual weight uses quintic easing over168
    # source frames, its derivative is at most1.875/168 perframe.
    report['maximum_168frame_quintic_weight_change_byte_per_frame']=float(abs(out-c).max()*1.875/168)
    return report


def main():
    start=time.perf_counter();model=json.loads(MODEL.read_text())[LABEL]
    palette=np.stack(np.meshgrid(*[np.linspace(0,255,13)]*3,indexing='ij'),-1).reshape(-1,3)
    positions=np.stack(np.meshgrid(np.linspace(0,1,7),np.linspace(0,1,5),indexing='xy'),-1).reshape(-1,2)
    c=np.tile(palette,(len(positions),1));xy=np.repeat(positions,len(palette),axis=0)
    result={'model_path':str(MODEL),'model_key':LABEL,'model_was_refit':False,'mapping':'pointwiseRGB+normalizedXY; no output sampling or convolution','finite_difference_byte_step':.5,'grid_size':[13,13,13,7,5],'sides':{}}
    source=ROOT/'output/IYTYT-source-conform-eight-joins.mp4'
    for side,n in [('left',360),('right',361)]:
        m=model[side];rec={'cube_and_spatial_grid':eval_points(c,xy,m),'black_at_positions':predict(np.zeros((len(positions),3)),positions,m).tolist(),'white_at_positions':predict(np.full((len(positions),3),255.),positions,m).tolist(),'actual_frames':{}}
        for frame_n in [n-84,n,n+84]:
            image=read_frames(source,frame_n,1,size=(640,360))[0]
            yy,xx=np.mgrid[0:360:2,0:640:2];color=image[::2,::2].reshape(-1,3).astype(float);pos=np.stack([xx/639,yy/359],-1).reshape(-1,2)
            out=predict(color,pos,m);conf=confidence(color,pos,m)
            rec['actual_frames'][str(frame_n)]={'point_count':len(color),'p05_confidence':float(np.percentile(conf,5)),'fraction_confidence_lt_0.1':float((conf<.1).mean()),'max_abs_channel_change':float(abs(out-color).max()),'p99_abs_channel_change':float(np.percentile(abs(out-color),99))}
            if frame_n==n:rec['anchor_palette']=eval_points(color,pos,m)
        result['sides'][side]=rec
        print(side,json.dumps({'grid':rec['cube_and_spatial_grid'],'actual_frames':rec['actual_frames']}),flush=True)
    result['runtime_seconds']=time.perf_counter()-start
    (OUT/'rbf-audit.json').write_text(json.dumps(result,indent=2))

if __name__=='__main__':main()
