"""Independent native flat-region validation of already-fitted models; fits nothing."""
from pathlib import Path
import sys,json,cv2,numpy as np,hashlib
from PIL import Image
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'research')]
from seamstress.repair import flow,sample,resize_flow
from local_color_probe import apply_samples
D=Path(__file__).resolve().parent;cv2.setNumThreads(2)
rs={361:[{'id':r['id'],'material':r['material'],'left_rect_xyxy':r['rect_xyxy']} for r in json.load(open(D/'visual-361.json'))['regions']]}
rs.update({r['frame']:r['regions'] for r in json.load(open(D/'visual-other-joins.json'))['cuts']})
def ed(im):
 im=cv2.GaussianBlur(im,(0,0),.7);edge=np.zeros(im.shape[:2],np.uint8)
 for c in range(3):edge|=cv2.Canny(im[:,:,c],18,45)
 return cv2.distanceTransform((edge==0).astype(np.uint8),cv2.DIST_L2,5)
def get(im,xy):
 out=[]
 for a in range(0,len(xy),15000):
  q=xy[a:a+15000];out.append(cv2.remap(im,q[:,0,None].astype(np.float32),q[:,1,None].astype(np.float32),cv2.INTER_LINEAR).reshape((len(q),)+im.shape[2:]))
 return np.concatenate(out)
def stat(l,r):
 if len(l)==0:return None
 delta=r-l;med=np.median(delta,0)
 return {'n':len(l),'signed_rgb_mean':delta.mean(0).tolist(),'signed_rgb_median':med.tolist(),'mean_absolute_signed_bias':float(abs(delta.mean(0)).mean()),'rgb_mae':float(abs(delta).mean()),'residual_mad_about_median':np.median(abs(delta-med),0).tolist()}
rows=[]
for cut,regions in rs.items():
 path=ROOT/'research/local-color-fit'/f'{cut}-model.json';model=json.load(open(path))
 a,b=[np.array(Image.open(D/str(cut)/f'{s}.png').convert('RGB')) for s in ['left','right']]
 ap,bp=[cv2.resize(im,(640,360),interpolation=cv2.INTER_AREA) for im in [a,b]]
 f,bk=flow(ap,bp),flow(bp,ap);fw=resize_flow(f,(1280,720));fb=cv2.resize(np.linalg.norm(f+sample(bk,f),axis=2),(1280,720))*2
 smooth=[cv2.GaussianBlur(im.astype(np.float32),(0,0),.8) for im in [a,b]];eds=[ed(im) for im in [a,b]];regionsout=[]
 for region in regions:
  x0,y0,x1,y1=region['left_rect_xyxy'];yy,xx=np.mgrid[y0:y1,x0:x1];xy=np.column_stack([xx.ravel(),yy.ravel()]).astype(np.float32);rq=xy+get(fw,xy)
  l,r=get(smooth[0],xy),get(smooth[1],rq)
  keep=(get(fb,xy)<1.2)&(get(eds[0],xy)>4)&(get(eds[1],rq)>4)&(abs(l-r).max(1)<35)
  xy,rq,l,r=[v[keep] for v in [xy,rq,l,r]]
  rr={k:region[k] for k in ['id','material','left_rect_xyxy']};rr['baseline']=stat(l,r)
  if len(l):
   cl,cr=apply_samples(l,xy/[1279,719],model['left']),apply_samples(r,rq/[1279,719],model['right']);rr['candidate']=stat(cl,cr)
   rr['absolute_signed_bias_change']=rr['candidate']['mean_absolute_signed_bias']-rr['baseline']['mean_absolute_signed_bias']
   print(cut,region['id'],'n',len(l),'baseline',np.round(rr['baseline']['signed_rgb_mean'],2),'candidate',np.round(rr['candidate']['signed_rgb_mean'],2),'biaschange',round(rr['absolute_signed_bias_change'],3),flush=True)
  else:rr['candidate']=None;print(cut,region['id'],'NO_VALID_INTERIOR',flush=True)
  regionsout.append(rr)
 rows.append({'frame':cut,'model_path':str(path),'model_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'regions':regionsout})
result={'method':'Native1280x720 Gaussian0.8px samples; DIS measurement flow at640x360; forward-backward<1.2nativepx; distance>4nativepx from per-channel contours in both frames; absRGB difference<35. RGB models only applied at measured samples; no fit or geometry changes. Signed bias incoming minus outgoing,8bitRGB.','cuts':rows}
(D/'all-models-native-material-evaluation.json').write_text(json.dumps(result,indent=2))
