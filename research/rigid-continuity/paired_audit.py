"""Track one set of source landmarks and transport exactly through export matrices.

This avoids changing the estimated scene depth merely because a 1% image
resample made a different RANSAC consensus win. No new image deformation.
"""
from pathlib import Path
import json,sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import cv2,numpy as np
from seamstress.media import read_frames
OUT=Path(__file__).parent/'all-joins';CUTS=[361,722,1083,1444,1805,2166,2527,3240]


def affine_similarity(p,q):
 pm,qm=p.mean(0),q.mean(0);pc=(p-pm)[:,0]+1j*(p-pm)[:,1];qc=(q-qm)[:,0]+1j*(q-qm)[:,1]
 ab=np.vdot(pc,qc)/np.vdot(pc,pc);a=np.array([[ab.real,-ab.imag],[ab.imag,ab.real]])
 return np.column_stack([a,qm-a@pm])


def describe(a,gap=4):
 c=np.array([319.75,179.75]);d=a[:,:2]@c+a[:,2]-c
 return {'zoom_pct_per_frame':float(np.log(np.sqrt(np.linalg.det(a[:,:2])))*100/gap),'dx_native_per_frame':float(d[0]*2/gap),'dy_native_per_frame':float(d[1]*2/gap)}


def points(a,b):
 ga,gb=[cv2.cvtColor(f,cv2.COLOR_RGB2GRAY) for f in [a,b]]
 p=cv2.goodFeaturesToTrack(ga,1800,.009,6,blockSize=5)
 cells={};take=[]
 for i,(x,y) in enumerate(p[:,0]):
  cell=(int(x/640*6),int(y/360*4))
  if cells.get(cell,0)<24:take.append(i);cells[cell]=cells.get(cell,0)+1
 p=p[take];q,s,_=cv2.calcOpticalFlowPyrLK(ga,gb,p,None,winSize=(25,25),maxLevel=3)
 back,r,_=cv2.calcOpticalFlowPyrLK(gb,ga,q,None,winSize=(25,25),maxLevel=3)
 good=(s[:,0]>0)&(r[:,0]>0)&(np.linalg.norm(back[:,0]-p[:,0],axis=1)<1)
 p,q=p[good,0],q[good,0]
 m,inside=cv2.estimateAffinePartial2D(p,q,method=cv2.RANSAC,ransacReprojThreshold=1.25,maxIters=5000,confidence=.999,refineIters=30)
 keep=inside[:,0]>0
 return p[keep],q[keep],float(inside.mean())


recipe=json.load(open(OUT/'eight-global-matrices.json'));m=np.array(recipe['frame_matrices']);view=np.array(recipe['view_matrix']);down=np.diag([.5,.5,1]);up=np.linalg.inv(down);mv=down@view@m@up;v=down@view@up
reports=[]
for cut in CUTS:
 start=cut-20;frames=list(read_frames(ROOT/'IYTYT.mp4',start,41,size=(640,360)));rows=[]
 for i in range(4,41):
  n=start+i;p,q,conf=points(frames[i-4],frames[i]);bp=p@v[:2,:2].T+v[:2,2];bq=q@v[:2,:2].T+v[:2,2]
  cp=p@mv[n-4,:2,:2].T+mv[n-4,:2,2];cq=q@mv[n,:2,:2].T+mv[n,:2,2]
  rows.append({'frame':n,'inlier_fraction':conf,'source':describe(affine_similarity(bp,bq)),'candidate':describe(affine_similarity(cp,cq))})
 means={}
 for name,lo,hi in [('before',cut-8,cut-1),('across',cut,cut+3),('after',cut+4,cut+11),('outside',cut+15,cut+20)]:
  selected=[r for r in rows if lo<=r['frame']<=hi]
  means[name]={kind:{key:float(np.mean([r[kind][key] for r in selected])) for key in rows[0][kind]} for kind in ['source','candidate']}
 flags=[]
 for key,threshold in [('zoom_pct_per_frame',.1),('dx_native_per_frame',1),('dy_native_per_frame',1)]:
  pre,post=means['before']['source'][key],means['outside']['source'][key];dominant=pre if abs(pre)>abs(post) else post
  if abs(dominant)<threshold or pre*post < -threshold**2:continue
  opposite=[r['frame'] for r in rows if cut-10<=r['frame']<=cut+12 and r['candidate'][key]*np.sign(dominant)<-threshold and r['source'][key]*np.sign(dominant)>=-threshold]
  if opposite:flags.append({'component':key,'frames':opposite,'native_before':pre,'native_after':post})
 reports.append({'cut':cut,'interval_means':means,'new_strong_reversal_flags':flags,'curves':rows});print(cut,'means',means,'flags',flags,flush=True)
 (OUT/'eight-global-paired-camera-audit.json').write_text(json.dumps({'method':'Original4-frame sparse tracks, spatial balancing and RANSAC; same inlier landmarks transported through exact source-to-output matrices. Reports actual camera effect without changing dominant scene-plane consensus after resampling.','strong_reversal_thresholds':{'zoom_pct_per_frame':.1,'center_native_px_per_frame':1},'joins':reports},indent=2))
