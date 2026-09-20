"""Adjacent-cut background-only optical registration. Diagnostic, not output."""
from pathlib import Path
import sys,json,cv2,numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from PIL import Image
from scipy.ndimage import distance_transform_edt
from seamstress.media import read_frames
from seamstress.repair import flow,grid
from seamstress.registration import _fit_color
cv2.setNumThreads(4)
OUT=ROOT/'research/layer2888/dense';OUT.mkdir(exist_ok=True)
r=json.load(open(ROOT/'research/layer2888/v2/report.json'));FG=np.array(r['foreground_matrix_incoming_to_outgoing']);BG=np.array(r['background_matrix_incoming_to_outgoing'])
a,b=read_frames(ROOT/'IYTYT.mp4',2887,2);h,w=a.shape[:2];xy=grid(a.shape)
ma,mb=[(cv2.imread(str(ROOT/f'research/layer2888/mask-{i}.png'),0)>127).astype(np.uint8) for i in [2887,2888]]
def sample(im,d,mode=cv2.INTER_LINEAR):
 q=xy+d;return cv2.remap(im,q[:,:,0],q[:,:,1],mode,borderMode=cv2.BORDER_CONSTANT)
def warp(im,m,mode=cv2.INTER_LINEAR):return cv2.warpAffine(im,m[:2].astype(np.float32),(w,h),flags=mode,borderMode=cv2.BORDER_CONSTANT)
forward,backward=flow(a,b),flow(b,a)
fb=np.linalg.norm(forward+sample(backward,forward),axis=2)
valid=(cv2.dilate(ma,np.ones((31,31),np.uint8))==0)&(sample(cv2.dilate(mb,np.ones((31,31),np.uint8)).astype(np.float32),forward)<.01)
confidence=np.exp(-(fb/1.25)**2).astype(np.float32)*valid
iv=np.linalg.inv(BG);base=xy@iv[:2,:2].T+iv[:2,2]-xy
residual=forward-base
num=cv2.GaussianBlur((residual*confidence[:,:,None]).astype(np.float32),(0,0),8)
den=cv2.GaussianBlur(confidence,(0,0),8)
field=num/np.maximum(den[:,:,None],.0001)
known=den>.08
_,idx=distance_transform_edt(~known,return_indices=True)
field[~known]=field[idx[0][~known],idx[1][~known]]
raw=base+field
checks=[]
for sigma in [0,4,8,12,18,24]:
 mapping=raw.astype(np.float32) if sigma==0 else cv2.GaussianBlur(raw.astype(np.float32),(0,0),sigma)
 uy,ux=np.gradient(mapping[:,:,0]);vy,vx=np.gradient(mapping[:,:,1]);det=(1+ux)*(1+vy)-uy*vx
 q={'sigma':sigma,'jacobian_min':float(det.min()),'jacobian_max':float(det.max()),'background_p01':float(np.percentile(det[valid],1)),'background_p99':float(np.percentile(det[valid],99))};checks.append(q)
 if det.min()>=.7 and det.max()<=1.4:break
accepted=bool(det.min()>=.7 and det.max()<=1.4)
# Invert final gather map at straight source line samples; measure departure
# of the transformed line from its best straight line, independently of MAE.
edges=cv2.Canny(cv2.cvtColor(b,cv2.COLOR_RGB2GRAY),35,100)
edges[cv2.dilate(mb,np.ones((21,21),np.uint8))>0]=0
lines=cv2.HoughLinesP(edges,1,np.pi/720,70,minLineLength=100,maxLineGap=4)
line_rows=[]
if lines is not None:
 ls=sorted(lines.reshape(-1,4).tolist(),key=lambda x:-(x[2]-x[0])**2-(x[3]-x[1])**2)[:60]
 for x0,y0,x1,y1 in ls:
  p=np.column_stack([np.linspace(x0,x1,60),np.linspace(y0,y1,60)]).astype(np.float32);out=p.copy()
  for _ in range(16):
   d=cv2.remap(mapping,out[:,0].reshape(1,-1),out[:,1].reshape(1,-1),cv2.INTER_LINEAR).reshape(-1,2);out=p-d
  center=out.mean(0);_,_,vh=np.linalg.svd(out-center);dist=np.abs((out-center)@vh[-1])
  line_rows.append({'source_line':[x0,y0,x1,y1],'length':float(np.linalg.norm(p[-1]-p[0])),'transformed_straightness_p95':float(np.percentile(dist,95)),'max':float(dist.max())})
alpha=warp(mb.astype(np.float32),FG).clip(0,1)
fgnum=warp(b.astype(np.float32)*mb[:,:,None],FG);fg=np.divide(fgnum,alpha[:,:,None],out=np.zeros_like(fgnum),where=alpha[:,:,None]>1e-5)
background=sample(b,mapping,cv2.INTER_LANCZOS4).astype(np.float32)
coverage=sample((1-cv2.dilate(mb,np.ones((5,5),np.uint8))).astype(np.float32),mapping)>.999
safe=coverage&(ma==0);safe[:20]=False;safe[-20:]=False;safe[:,:20]=False;safe[:,-20:]=False
color=_fit_color(a,background.astype(np.uint8),safe);gain=np.array(color['gain']);bias=np.array(color['bias']);background=background*gain+bias;fg=fg*gain+bias
# Exact previous original frame supplies only already visible background.
donor_available=cv2.dilate(ma,np.ones((3,3),np.uint8))==0
take=(~coverage)&donor_available;background[take]=a[take];coverage[take]=True
holes=(~coverage)&(alpha<.999)
result=np.clip(background*(1-alpha[:,:,None])+fg*alpha[:,:,None],0,255).astype(np.uint8);result[holes]=[255,0,255]
Image.fromarray(result).save(OUT/'result-2888.png');Image.fromarray(holes.astype(np.uint8)*255).save(OUT/'holes.png')
Image.fromarray(np.concatenate([a,result],axis=1)).save(OUT/'cut-adjacent.png')
report={'adjacent_source_frames':[2887,2888],'neural':False,'pose_blend':False,'foreground_map':FG.tolist(),'background_method':'DIS forward/backward confidence, normalized Gaussian residual smoothing; independent source foreground','accepted_jacobian':accepted,'smoothing_trials':checks,'foreground_or_disocclusion_holes':int(holes.sum()),'previous_source_background_pixels_used':int((take&(alpha<.999)).sum()),'straight_lines':line_rows,'worst_line_p95_curvature':max(x['transformed_straightness_p95'] for x in line_rows),'status':'Diagnostic only. Magenta explicitly marks uncovered background.'}
(OUT/'report.json').write_text(json.dumps(report,indent=2));print({k:v for k,v in report.items() if k not in ['straight_lines']},flush=True)
np.savez_compressed(OUT/'mapping.npz',mapping=mapping,confidence=confidence,valid=valid)
