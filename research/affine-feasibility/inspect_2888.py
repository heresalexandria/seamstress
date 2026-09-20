from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import cv2,numpy as np
from seamstress.media import read_frames
from research.rigid_feasibility import features
cv2.setNumThreads(4)
frames=list(read_frames(ROOT/'IYTYT.mp4',2884,9))
rows=[]
for i in range(1,len(frames)):
 a,b=frames[i-1],frames[i];_,pa,pb,_=features(a,b)
 rois={'characters':(445,220,870,610),'empire_state':(320,5,440,710),'distant_right':(925,300,1270,540),'near_bottom':(5,580,1250,710)}
 fields={}
 for label,(x0,y0,x1,y1) in rois.items():
  k=(pa[:,0]>x0)&(pa[:,0]<x1)&(pa[:,1]>y0)&(pa[:,1]<y1)
  p,q=pa[k],pb[k]
  if len(p)<6:continue
  m,good=cv2.estimateAffine2D(q,p,method=cv2.RANSAC,ransacReprojThreshold=2,maxIters=5000,confidence=.999)
  if m is None:continue
  disp=q-p
  fields[label]={'matches':len(p),'median_raw_displacement_xy':np.median(disp,axis=0).tolist(),'matrix':m.tolist(),'inliers':int(good.sum())}
 rows.append({'frames':[2883+i,2884+i],'regions':fields})
(ROOT/'research/affine-feasibility/2888/region_motion.json').write_text(json.dumps(rows,indent=2))
for x in rows:
 print(x['frames'],[(k,v['matches'],np.around(v['median_raw_displacement_xy'],2).tolist()) for k,v in x['regions'].items()])
