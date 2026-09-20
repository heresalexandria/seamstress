from pathlib import Path
import sys,json,cv2,numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(Path(__file__).parent))
from prototype import BASE,grab,OUT,saveim,W,H
from seamstress.media import read_frames,VideoWriter
from seamstress.registration import _fit_color
from research.rigid_feasibility import features,metrics
cv2.setNumThreads(4)
CUT=2888
indices=[2883,2885,2887,2888,2889,2896,2904,2920]
source={i:read_frames(ROOT/'IYTYT.mp4',i,1)[0] for i in indices}
ref=source[CUT]
MASKS={CUT:np.load(OUT/'mask-2888.npy')}

def at(m,p):
 q=np.rint(p).astype(int);q[:,0]=q[:,0].clip(0,W-1);q[:,1]=q[:,1].clip(0,H-1)
 return m[q[:,1],q[:,0]]>0

def getmask(i):
 if i in MASKS:return MASKS[i]
 _,p,q,_=features(ref,source[i]);keep=at(cv2.erode(MASKS[CUT],np.ones((3,3),np.uint8)),p)
 a,good=cv2.estimateAffinePartial2D(q[keep],p[keep],method=cv2.RANSAC,ransacReprojThreshold=2,maxIters=5000,confidence=.999)
 if a is None:raise RuntimeError('Foreground tracking failed')
 inv=np.linalg.inv(np.vstack([a,[0,0,1]]))
 seed=cv2.warpAffine(BASE,inv[:2],(W,H),flags=cv2.INTER_NEAREST)
 # For this short bounded test the manual fixed core seeds remain within
 # same foreground poses; masks are inspected and never claimed production.
 MASKS[i]=grab(source[i],seed)
 saveim(f'mask-{i}.png',MASKS[i]*255)
 return MASKS[i]

def register(a,b,ma,mb,kind):
 _,pa,pb,_=features(a,b)
 if kind=='foreground':
  keep=at(cv2.erode(ma,np.ones((3,3),np.uint8)),pa)&at(cv2.erode(mb,np.ones((3,3),np.uint8)),pb)
  fn=cv2.estimateAffinePartial2D
 else:
  keep=~at(cv2.dilate(ma,np.ones((13,13),np.uint8)),pa)&~at(cv2.dilate(mb,np.ones((13,13),np.uint8)),pb)
  fn=cv2.estimateAffine2D
 p,q=pa[keep],pb[keep]
 m,good=fn(q,p,method=cv2.RANSAC,ransacReprojThreshold=2.5,maxIters=10000,confidence=.999,refineIters=50)
 if m is None:raise RuntimeError('Insufficient '+kind+' registration')
 error=np.linalg.norm(q@m[:2,:2].T+m[:2,2]-p,axis=1)
 return np.vstack([m,[0,0,1]]),{'matches':len(p),'inliers':int(good.sum()),'residual_p50':float(np.median(error)),'residual_p90':float(np.percentile(error,90))}

def warp(im,m,mode=cv2.INTER_LANCZOS4):return cv2.warpAffine(im,m[:2].astype(np.float32),(W,H),flags=mode,borderMode=cv2.BORDER_CONSTANT)

a=source[2887];ma=getmask(2887);mb=getmask(2888)
BG,bgq=register(a,ref,ma,mb,'background');FG,fgq=register(a,ref,ma,mb,'foreground')
print('BG',BG.tolist(),bgq,flush=True);print('FG',FG.tolist(),fgq,flush=True)
# Same grade for both layers, estimated on registered source background.
wb=warp(ref,BG);v=(warp((1-mb).astype(np.float32),BG,cv2.INTER_LINEAR)>.999)&(ma==0)
v[:20]=False;v[-20:]=False;v[:,:20]=False;v[:,-20:]=False
color=_fit_color(a,wb,v);gain=np.array(color['gain']);bias=np.array(color['bias'])
reports=[]
for target in [2888,2889,2896]:
 b=source[target];m=getmask(target)
 alpha=warp(m.astype(np.float32),FG,cv2.INTER_LINEAR).clip(0,1)
 premul=warp(b.astype(np.float32)*m[:,:,None],FG,cv2.INTER_LINEAR)
 fg=np.divide(premul,alpha[:,:,None],out=np.zeros_like(premul),where=alpha[:,:,None]>1e-5)
 clean=(1-cv2.dilate(m,np.ones((5,5),np.uint8))).astype(np.float32)
 cov=warp(clean,BG,cv2.INTER_LINEAR)>.999
 bg=warp(b,BG).astype(float);bg=bg*gain+bias
 holes_initial=(~cov)&(alpha<.999)
 copied=np.zeros((H,W),np.uint8); donor_report=[]
 for donor in [2887,2885,2883,2904,2920]:
  md=getmask(donor)
  D,dq=register(b,source[donor],m,md,'background')
  M=BG@D
  can=warp((1-cv2.dilate(md,np.ones((7,7),np.uint8))).astype(np.float32),M,cv2.INTER_LINEAR)>.999
  patch=warp(source[donor],M).astype(float)
  # Color normalize each donor against target background on original pixels.
  sample=cov&can
  cc=_fit_color(np.clip(bg,0,255).astype(np.uint8),patch.astype(np.uint8),sample)
  patch=patch*np.array(cc['gain'])+np.array(cc['bias'])
  take=(~cov)&can
  bg[take]=patch[take];cov[take]=True;copied[take]=1
  donor_report.append({'frame':donor,'pixels_used':int(np.count_nonzero(take&(alpha<.999))), 'registration':dq})
 fg=np.clip(fg*gain+bias,0,255)
 holes=(~cov)&(alpha<.999)
 result=np.clip(bg*(1-alpha[:,:,None])+fg*alpha[:,:,None],0,255).astype(np.uint8)
 result[holes]=[255,0,255] # Explicit missing pixels; never hide by inpainting.
 saveim(f'result-{target}.png',result)
 saveim(f'holes-{target}.png',holes.astype(np.uint8)*255)
 overlay=result.copy();overlay[copied.astype(bool)&(alpha<.999)]=(.6*overlay[copied.astype(bool)&(alpha<.999)]+np.array([0,102,0])).clip(0,255).astype(np.uint8)
 saveim(f'donor-pixels-{target}.png',overlay)
 # Outer 30px are classified separately from subject disocclusion holes.
 interior=np.zeros((H,W),bool);interior[30:-30,30:-30]=True
 character_zone=np.zeros((H,W),bool);character_zone[200:650,320:900]=True
 rep={'frame':target,'initial_holes':int(holes_initial.sum()),'final_holes':int(holes.sum()),'interior_final_holes':int((holes&interior).sum()),'character_zone_final_holes':int((holes&character_zone).sum()),'donors':donor_report}
 reports.append(rep);print(rep,flush=True)
 if target==2888:
  saveim('cut-adjacent.png',np.concatenate([a,result],axis=1))
  with VideoWriter(OUT/'cut-blink.mp4',W,H,'24',crf=12) as writer:
   for _ in range(8):
    for fr in [a,result]:
     for _ in range(8):writer.write(fr)
(OUT/'report.json').write_text(json.dumps({'cut':CUT,'source':'IYTYT.mp4','foreground_matrix_incoming_to_outgoing':FG.tolist(),'background_matrix_incoming_to_outgoing':BG.tolist(),'foreground_registration':fgq,'background_registration':bgq,'color':color,'frames':reports,'status':'Experimental. Magenta marks uncovered pixels; masks and donor joins require visual review. No poses are synthesized or blended.'},indent=2))
