"""Prepare a bounded 48-frame source-only layer proof, without rendering yet."""
from pathlib import Path
import sys,json,cv2,numpy as np
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from PIL import Image
from seamstress.media import read_frames
from research.rigid_feasibility import features
cv2.setNumThreads(4)
OUT=Path(__file__).resolve().parent/'proof';OUT.mkdir(exist_ok=True)
frames=read_frames(ROOT/'IYTYT.mp4',2864,48);np.save(OUT/'source-2864-2911.npy',frames)
ref=frames[24];maskref=(np.array(Image.open(ROOT/'research/layer2888/mask-2888.png'))>127).astype(np.uint8)
BG=np.array(json.load(open(ROOT/'research/layer2888/v2/report.json'))['background_matrix_incoming_to_outgoing'])
flame=np.zeros((720,1280),np.uint8)
polys=[[(429,427),(460,427),(461,450),(454,480),(450,507),(441,524),(430,495),(422,452)],[(557,428),(566,428),(578,452),(576,480),(562,525),(554,514),(559,485),(555,460)],[(661,393),(694,394),(704,418),(709,451),(705,481),(692,505),(675,500),(652,455),(653,418)],[(772,396),(788,397),(801,427),(798,451),(789,469),(782,450)]]
for p in polys:cv2.fillPoly(flame,[np.array(p,np.int32)],1)
for x0,y0,x1,y1 in [(420,425,470,441),(542,426,581,443),(649,390,716,408),(770,393,807,411)]:flame[y0:y1,x0:x1]=0

def at(m,p):
 z=np.rint(p).astype(int);return m[z[:,1].clip(0,719),z[:,0].clip(0,1279)]>0
rows=[]
for n in range(2881,2912):
 frame=frames[n-2864]
 if n==2888:
  mask=maskref;F=np.eye(3);D=np.eye(3);fgq={'matches':0,'p90':0};bgq={'matches':0,'p90':0}
 else:
  _,p,q,_=features(ref,frame)
  fg=at(cv2.erode(maskref,np.ones((3,3),np.uint8)),p)
  fm,fi=cv2.estimateAffinePartial2D(q[fg],p[fg],method=cv2.RANSAC,ransacReprojThreshold=2.5,maxIters=10000,confidence=.999,refineIters=50)
  F=np.linalg.inv(np.vstack([fm,[0,0,1]]))
  cached=ROOT/f'research/layer2888/mask-{n}.png'
  if cached.exists():mask=(np.array(Image.open(cached))>127).astype(np.uint8)
  else:
   seed=cv2.warpAffine(maskref,F[:2].astype(np.float32),(1280,720),flags=cv2.INTER_NEAREST)
   labels=np.zeros((720,1280),np.uint8);labels[cv2.dilate(seed,np.ones((15,15),np.uint8))>0]=cv2.GC_PR_BGD;labels[seed>0]=cv2.GC_PR_FGD;labels[cv2.erode(seed,np.ones((9,9),np.uint8))>0]=cv2.GC_FGD
   cv2.grabCut(cv2.cvtColor(frame,cv2.COLOR_RGB2BGR),labels,None,np.zeros((1,65)),np.zeros((1,65)),2,cv2.GC_INIT_WITH_MASK)
   mask=((labels==cv2.GC_FGD)|(labels==cv2.GC_PR_FGD)).astype(np.uint8)
  bg=~at(cv2.dilate(maskref,np.ones((17,17),np.uint8)),p)&~at(cv2.dilate(mask,np.ones((17,17),np.uint8)),q)
  # Matrix from reference-frame BG into current-source coordinates.
  dm,di=cv2.estimateAffine2D(p[bg],q[bg],method=cv2.RANSAC,ransacReprojThreshold=2.5,maxIters=10000,confidence=.999,refineIters=50)
  D=np.vstack([dm,[0,0,1]])
  if n==2887:D=np.linalg.inv(BG)
  errfg=np.linalg.norm(q[fg]@fm[:2,:2].T+fm[:2,2]-p[fg],axis=1);errbg=np.linalg.norm(p[bg]@D[:2,:2].T+D[:2,2]-q[bg],axis=1)
  fgq={'matches':int(fg.sum()),'inliers':int(fi.sum()),'p50':float(np.median(errfg)),'p90':float(np.percentile(errfg,90))};bgq={'matches':int(bg.sum()),'inliers':int(di.sum()),'p50':float(np.median(errbg)),'p90':float(np.percentile(errbg,90))}
 fm=cv2.warpAffine(flame,F[:2].astype(np.float32),(1280,720),flags=cv2.INTER_NEAREST)
 np.savez_compressed(OUT/f'mask-{n}.npz',foreground=mask,flame=fm)
 rows.append({'frame':n,'foreground_reference_to_current_matrix':F.tolist(),'background_reference_to_current_matrix':D.tolist(),'foreground_tracking':fgq,'background_tracking':bgq})
 (OUT/'tracking.json').write_text(json.dumps({'first':2864,'end_exclusive':2912,'frames':rows},indent=2))
 print(n,fgq,bgq,flush=True)
