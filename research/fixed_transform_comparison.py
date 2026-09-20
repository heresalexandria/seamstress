"""Bounded source-only affine/projective fit comparison at join361."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import cv2,numpy as np
from PIL import Image,ImageDraw
from research.rigid_feasibility import features,metrics,ROIS
from seamstress.registration import _fit_color,_measure
from seamstress.media import VideoWriter
P=ROOT/'research/rigid-feasibility/361';OUT=P/'fixed-transform-comparison';OUT.mkdir(exist_ok=True)
a=np.array(Image.open(P/'left-original.png'));b=np.array(Image.open(P/'right-original.png'))
base=json.load(open(P/'report.json'));similarity=np.array(base['matrix_right_to_left']);h,w=a.shape[:2]
_,pa,pb,_=features(a,b)
A,mask=cv2.estimateAffine2D(pb,pa,method=cv2.RANSAC,ransacReprojThreshold=2,maxIters=12000,confidence=.999,refineIters=50)
H,inliers=cv2.findHomography(pb,pa,cv2.RANSAC,2,maxIters=12000,confidence=.999)
candidates={'similarity':[('previous_best',similarity)],'affine':[('native_sift',np.vstack([A,[0,0,1]]))],'homography':[('native_sift',H)]}
# ECC candidates can account for broad low-frequency structure; the strict
# geometry checks below reject candidates supported by image distortion.
aa,bb=[cv2.resize(x,(640,360),interpolation=cv2.INTER_AREA) for x in [a,b]]
gray=[cv2.GaussianBlur(cv2.cvtColor(x,cv2.COLOR_RGB2GRAY).astype(np.float32)/255,(0,0),1.1) for x in [aa,bb]]
down=np.diag([.5,.5,1]);up=np.diag([2,2,1])
for kind,mode in [('affine',cv2.MOTION_AFFINE),('homography',cv2.MOTION_HOMOGRAPHY)]:
 for seedname,seed in [('similarity',similarity),('sift',candidates[kind][0][1])]:
  inv=np.linalg.inv(down@seed@up).astype(np.float32)
  try:
   score,v=cv2.findTransformECC(gray[0],gray[1],inv if kind=='homography' else inv[:2],mode,(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,160,1e-6),None,5)
   if kind=='affine':v=np.vstack([v,[0,0,1]])
   m=up@np.linalg.inv(v)@down;m/=m[2,2]
   candidates[kind].append(('ecc_from_'+seedname,m))
  except cv2.error:pass
corners=np.array([[0,0],[w-1,0],[0,h-1],[w-1,h-1]],float)
def points(m,p):
 q=np.column_stack((p,np.ones(len(p))))@m.T
 return q[:,:2]/q[:,2:]
def constraints(m):
 m=m/m[2,2];landmarks=np.r_[corners,[[w/2,h/2],[850,330],[390,380]]]
 anis=[];areas=[]
 for p in landmarks:
  origin=points(m,p[None])[0];j=np.column_stack(((points(m,(p+[1,0])[None])[0]-origin),(points(m,(p+[0,1])[None])[0]-origin)))
  s=np.linalg.svd(j,compute_uv=False);anis.append(float(s.max()/s.min()));areas.append(float(np.linalg.det(j)))
 difference=np.linalg.norm(points(m,corners)-points(similarity,corners),axis=1)
 result={'maximum_corner_difference_from_similarity_px':float(max(difference)),'corner_differences_px':difference.tolist(),
 'maximum_local_anisotropy':max(anis),'local_area_scale_range':[min(areas),max(areas)],'projective_row':m[2].tolist()}
 result['accepted']=bool(max(difference)<20 and max(anis)<1.025 and min(areas)>.95 and max(areas)<1.08 and max(areas)/min(areas)<1.05)
 return result
chosen={};reports={}
for kind,options in candidates.items():
 rows=[]
 for name,m in options:
  bounds=constraints(m)
  warped=cv2.warpPerspective(b,m.astype(np.float32),(w,h),flags=cv2.INTER_LANCZOS4,borderMode=cv2.BORDER_CONSTANT)
  valid=cv2.warpPerspective(np.ones((h,w),np.uint8),m.astype(np.float32),(w,h),flags=cv2.INTER_NEAREST)>0
  valid[:10]=False;valid[-10:]=False;valid[:,:10]=False;valid[:,-10:]=False
  color=_fit_color(a,warped,valid);c=np.clip(np.rint(warped.astype(float)*color['gain']+color['bias']),0,255).astype(np.uint8)
  q=_measure(a,warped,valid,color);score=q['trimmed_corrected_mae']+.4*q['gradient_mae']
  record={'method':name,'matrix_right_to_left':m.tolist(),'bounds':bounds,'color':color,'metrics':metrics(a,c,valid),'score':float(score)}
  rows.append(record)
  if bounds['accepted'] and (kind not in chosen or score<chosen[kind][0]):chosen[kind]=(score,c,record)
 reports[kind]=rows
for kind,(score,frame,record) in chosen.items():
 Image.fromarray(frame).save(OUT/f'{kind}.png')
 with VideoWriter(OUT/f'{kind}-blink.mp4',1280,720,'24',crf=12,preset='fast') as writer:
  for _ in range(6):
   for im in [a,frame]:
    for _ in range(8):writer.write(im)
for label,roi in ROIS[361]:
 x0,y0,x1,y1=roi;cw,ch=x1-x0,y1-y0;sheet=Image.new('RGB',(cw*(len(chosen)+1),ch+32));draw=ImageDraw.Draw(sheet)
 for i,(name,frame) in enumerate([('Original frame360',a)]+[(k,v[1]) for k,v in chosen.items()]):
  sheet.paste(Image.fromarray(frame[y0:y1,x0:x1]),(i*cw,32));draw.text((i*cw+8,10),name,fill='white')
 sheet.save(OUT/(label.lower().replace(' ','-')+'.png'))
result={'chosen':{k:v[2] for k,v in chosen.items()},'all_candidates':reports,'restrictions':'One fixed transform and single source resampling; global affine RGB grade; no frame interpolation, dense warp, crossfade, or pose blending.'}
(OUT/'report.json').write_text(json.dumps(result,indent=2))
for kind,(_,_,r) in chosen.items():print(kind,r['method'],r['metrics'],r['bounds'],flush=True)
