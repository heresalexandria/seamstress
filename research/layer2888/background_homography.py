from pathlib import Path
import sys,json,cv2,numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from PIL import Image
from research.rigid_feasibility import features,metrics
from seamstress.registration import _fit_color,_measure
cv2.setNumThreads(4)
OUT=ROOT/'research/layer2888/background-homography';OUT.mkdir(exist_ok=True)
a,b=[np.array(Image.open(ROOT/f'research/affine-feasibility/2888/{name}.png').convert('RGB')) for name in ['left-original','right-original']]
ma,mb=[np.array(Image.open(ROOT/f'research/layer2888/mask-{n}.png'))>127 for n in [2887,2888]]
_,p,q,_=features(a,b)
def at(mask,p):
 z=np.rint(p).astype(int);return mask[z[:,1].clip(0,719),z[:,0].clip(0,1279)]>0
keep=~at(cv2.dilate(ma.astype(np.uint8),np.ones((15,15),np.uint8)),p)&~at(cv2.dilate(mb.astype(np.uint8),np.ones((15,15),np.uint8)),q)
h,good=cv2.findHomography(q[keep],p[keep],cv2.RANSAC,2.5,maxIters=20000,confidence=.999)
base=np.array(json.load(open(ROOT/'research/layer2888/v2/report.json'))['background_matrix_incoming_to_outgoing'])
points=np.array([[0,0],[1279,0],[1279,719],[0,719]],float)
def project(m,p):
 z=np.column_stack([p,np.ones(len(p))])@m.T;return z[:,:2]/z[:,2:]
rows=[]
for name,m in [('affine',base),('homography',h)]:
 im=cv2.warpPerspective(b,m,(1280,720),flags=cv2.INTER_LANCZOS4)
 valid=cv2.warpPerspective((cv2.dilate(mb.astype(np.uint8),np.ones((17,17),np.uint8))==0).astype(np.uint8),m,(1280,720),flags=cv2.INTER_NEAREST)>0
 valid&=~cv2.dilate(ma.astype(np.uint8),np.ones((17,17),np.uint8)).astype(bool);valid[:20]=False;valid[-20:]=False;valid[:,:20]=False;valid[:,-20:]=False
 col=_fit_color(a,im,valid);out=np.clip(im*col['gain']+col['bias'],0,255).astype(np.uint8)
 quality=_measure(a,im,valid,col);error=np.linalg.norm(project(m,q[keep])-p[keep],axis=1)
 corners=project(m,points);delta=np.linalg.norm(corners-project(base,points),axis=1)
 row={'model':name,'matrix_incoming_to_outgoing':m.tolist(),'corner_delta_from_affine':delta.tolist(),'max_corner_delta_from_affine':float(delta.max()),'feature_residual_p50':float(np.median(error)),'feature_residual_p90':float(np.percentile(error,90)),'color':col,'metrics':metrics(a,out,valid),'score':quality['trimmed_corrected_mae']+.4*quality['gradient_mae']}
 rows.append(row);Image.fromarray(out).save(OUT/f'{name}.png')
(OUT/'report.json').write_text(json.dumps({'matches':int(keep.sum()),'homography_inliers':int(good.sum()),'models':rows},indent=2));print(rows)
