import cv2,numpy as np,json,time
from pathlib import Path
cv2.setNumThreads(2)
p=Path(__file__).parent
cap=cv2.VideoCapture('IYTYT.mp4')
rows=[]; prev=None;i=0
while True:
 ok,img=cap.read()
 if not ok: break
 small=cv2.resize(img,(480,270))
 gray=cv2.cvtColor(small,cv2.COLOR_BGR2GRAY)
 if prev is not None:
  p0=cv2.goodFeaturesToTrack(prev,maxCorners=450,qualityLevel=.012,minDistance=5,blockSize=5)
  r={'frame':i,'time':i*1001/24000,'raw':float(np.mean(cv2.absdiff(gray,prev))),'color':np.mean(small.astype(float)-psmall,axis=(0,1)).tolist()}
  if p0 is not None:
   p1,st,err=cv2.calcOpticalFlowPyrLK(prev,gray,p0,None,winSize=(25,25),maxLevel=3)
   back,sb,eb=cv2.calcOpticalFlowPyrLK(gray,prev,p1,None,winSize=(25,25),maxLevel=3)
   good=(st.ravel()!=0)&(sb.ravel()!=0)&(np.linalg.norm(back-p0,axis=2).ravel()<1.5)
   a,b=p0[good].reshape(-1,2),p1[good].reshape(-1,2)
   M,inl=cv2.estimateAffinePartial2D(a,b,method=cv2.RANSAC,ransacReprojThreshold=1.5,maxIters=1000) if len(a)>=5 else (None,None)
   if M is not None:
    warp=cv2.warpAffine(psmall,M,(480,270))
    mask=cv2.warpAffine(np.ones_like(prev),M,(480,270))>0
    residual=small.astype(float)-warp.astype(float)
    r.update({'valid':float(np.mean(good)),'inlier':float(np.mean(inl)),'scale':float(np.sqrt(M[0,0]**2+M[0,1]**2)),'tx':float(M[0,2]),'ty':float(M[1,2]),'angle':float(np.arctan2(M[1,0],M[0,0])),'warped_color':np.median(residual[mask],axis=0).tolist(),'warped_mae':float(np.mean(abs(residual[mask]))),'matrix':M.tolist()})
  rows.append(r)
 prev=gray;psmall=small;i+=1
 if i%400==0:print(i,flush=True)
(p/'motion_metrics.json').write_text(json.dumps(rows,indent=2))
for key,func in [('scale',lambda r:abs(r.get('scale',1)-1)),('color',lambda r:sum(abs(v) for v in r.get('warped_color',[0]))),('warped_mae',lambda r:r.get('warped_mae',0))]:
 print('\nTOP',key)
 for r in sorted(rows,key=func,reverse=True)[:35]:print(r['frame'],round(r['time'],4),round(func(r),4),r.get('warped_color'),round(r.get('valid',0),3),round(r.get('inlier',0),3))
