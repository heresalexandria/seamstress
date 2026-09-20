"""Native source-only join evidence: one similarity and one affine RGB grade."""
from pathlib import Path
import sys,json,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import cv2,numpy as np
from scipy.optimize import minimize
from PIL import Image,ImageDraw
from seamstress.media import read_frames,VideoWriter
from seamstress.registration import register_pair,_fit_color,_measure
OUT=ROOT/'research/rigid-feasibility';OUT.mkdir(exist_ok=True)
cv2.setNumThreads(4)

def matrix(p,w,h):
 s=np.exp(p[0]/1000);r=p[1]/1000;c=np.array([(w-1)/2,(h-1)/2])
 a=s*np.array([[np.cos(r),-np.sin(r)],[np.sin(r),np.cos(r)]])
 return np.vstack([np.column_stack((a,c+np.array(p[2:])-a@c)),[0,0,1]])
def parameters(m,w,h):
 c=np.array([(w-1)/2,(h-1)/2]);s=np.sqrt(np.linalg.det(m[:2,:2]));r=np.arctan2(m[1,0],m[0,0])
 return np.r_[1000*np.log(s),1000*r,m[:2,:2]@c+m[:2,2]-c]
def project(m,w,h):
 c=np.array([(w-1)/2,(h-1)/2]);pos=m[:2,:2]@c+m[:2,2]
 u,s,vh=np.linalg.svd(m[:2,:2]);a=(u@vh)*np.mean(s)
 out=np.eye(3);out[:2,:2]=a;out[:2,2]=pos-a@c
 return out

def warp(im,m):
 h,w=im.shape[:2];out=cv2.warpAffine(im,m[:2].astype(np.float32),(w,h),flags=cv2.INTER_LANCZOS4,borderMode=cv2.BORDER_CONSTANT)
 valid=cv2.warpAffine(np.ones((h,w),np.uint8),m[:2].astype(np.float32),(w,h),flags=cv2.INTER_NEAREST)>0
 valid[:8]=False;valid[-8:]=False;valid[:,:8]=False;valid[:,-8:]=False
 valid=cv2.erode(valid.astype(np.uint8),np.ones((5,5),np.uint8))>0
 return out,valid

def features(a,b):
 sift=cv2.SIFT_create(nfeatures=7000,contrastThreshold=.012,edgeThreshold=14)
 ag,bg=[cv2.cvtColor(x,cv2.COLOR_RGB2GRAY) for x in [a,b]]
 ka,da=sift.detectAndCompute(ag,None);kb,db=sift.detectAndCompute(bg,None)
 matcher=cv2.BFMatcher();ab=matcher.knnMatch(da,db,k=2);ba=matcher.knnMatch(db,da,k=2)
 rev={m.queryIdx:m.trainIdx for m,n in ba if m.distance<.76*n.distance}
 matched=[m for m,n in ab if m.distance<.76*n.distance and rev.get(m.trainIdx)==m.queryIdx]
 pa=np.array([ka[m.queryIdx].pt for m in matched],np.float32);pb=np.array([kb[m.trainIdx].pt for m in matched],np.float32)
 m,inliers=cv2.estimateAffinePartial2D(pb,pa,method=cv2.RANSAC,ransacReprojThreshold=2,maxIters=10000,confidence=.999,refineIters=50)
 return np.vstack([m,[0,0,1]]),pa,pb,inliers.ravel()>0

def edges(im):
 g=cv2.cvtColor(im,cv2.COLOR_RGB2GRAY);g=cv2.GaussianBlur(g,(5,5),.8)
 return cv2.Canny(g,25,70)>0

def metrics(a,b,valid):
 errors=np.mean(np.abs(a.astype(float)-b.astype(float)),axis=2)
 ea,eb=edges(a),edges(b);ea&=valid;eb&=valid
 da=cv2.distanceTransform((~ea).astype(np.uint8),cv2.DIST_L2,5);db=cv2.distanceTransform((~eb).astype(np.uint8),cv2.DIST_L2,5)
 distances=np.r_[db[ea],da[eb]]
 return {'rgb_mae':float(np.mean(errors[valid])),'pixel_fraction_rgb_mae_gt15':float(np.mean(errors[valid]>15)),
   'edge_symmetric_median_px':float(np.median(distances)),'edge_symmetric_p90_px':float(np.percentile(distances,90)),
   'edge_fraction_farther_than_2px':float(np.mean(distances>2)),'edge_fraction_farther_than_4px':float(np.mean(distances>4)),
   'overlap_fraction':float(np.mean(valid))}

def fit(a,b,refine=True):
 h,w=a.shape[:2];f,pa,pb,inliers=features(a,b)
 production=register_pair(a,b,max_width=640)
 candidates={'sift_similarity':f,'projected_ecc_similarity':project(np.array(production['matrix']),w,h)}
 if refine:
  # Constrained 4-parameter similarity, not affine or local deformation.
  aa=cv2.resize(a,(640,360),interpolation=cv2.INTER_AREA);bb=cv2.resize(b,(640,360),interpolation=cv2.INTER_AREA)
  ga,gb=[cv2.GaussianBlur(cv2.cvtColor(x,cv2.COLOR_RGB2GRAY).astype(np.float32),(0,0),1) for x in [aa,bb]]
  yy,xx=np.mgrid[10:350:3,10:630:3];xy=np.stack((xx.ravel(),yy.ravel(),np.ones(xx.size)),1)
  target=ga[yy,xx].ravel().astype(float)
  seed=np.diag([.5,.5,1])@f@np.diag([2,2,1]);p0=parameters(seed,640,360)
  def cost(p):
   m=matrix(p,640,360);inverse=np.linalg.inv(m);q=xy@inverse.T
   val=cv2.remap(gb,q[:,0].reshape(-1,1).astype(np.float32),q[:,1].reshape(-1,1).astype(np.float32),cv2.INTER_LINEAR).ravel().astype(float)
   keep=(q[:,0]>2)&(q[:,0]<637)&(q[:,1]>2)&(q[:,1]<357)
   x=val[keep];y=target[keep];A=np.column_stack((x,np.ones_like(x)))
   coef=np.linalg.lstsq(A,y,rcond=None)[0]
   err=np.abs(y-A@coef)
   return float(np.mean(np.minimum(err,25)))
  res=minimize(cost,p0,method='Powell',bounds=[(p0[0]-20,p0[0]+20),(p0[1]-10,p0[1]+10),(p0[2]-12,p0[2]+12),(p0[3]-12,p0[3]+12)],options={'maxiter':30,'xtol':.015,'ftol':1e-5})
  candidates['robust_photometric_similarity']=np.diag([2,2,1])@matrix(res.x,640,360)@np.diag([.5,.5,1])
 measured=[]
 for name,m in candidates.items():
  wb,valid=warp(b,m);color=_fit_color(a,wb,valid)
  corrected=np.clip(np.rint(wb.astype(float)*color['gain']+color['bias']),0,255).astype(np.uint8)
  quality=_measure(a,wb,valid,color);score=quality['trimmed_corrected_mae']+.4*quality['gradient_mae']
  measured.append((score,name,m,wb,corrected,valid,color,quality))
 _,name,m,wb,corrected,valid,color,quality=min(measured,key=lambda x:x[0])
 pred=pb@m[:2,:2].T+m[:2,2];error=np.linalg.norm(pred-pa,axis=1)
 report={'chosen':name,'matrix_right_to_left':m.tolist(),'scale':float(np.sqrt(np.linalg.det(m[:2,:2]))),'rotation_degrees':float(np.degrees(np.arctan2(m[1,0],m[0,0]))),
  'color':color,'raw':metrics(a,b,valid),'geometry_only':metrics(a,wb,valid),'geometry_and_grade':metrics(a,corrected,valid),
  'feature_matches':len(pa),'feature_inlier_fraction':float(inliers.mean()),'feature_residual_p50':float(np.median(error)),
  'feature_residual_p90':float(np.percentile(error,90)),
  'candidates':[{'method':r[1],'score':float(r[0])} for r in measured]}
 return corrected,valid,report,pa,pb

ROIS={361:[('Woman face and hair',(650,175,1030,520)),('Sloth and windshield',(170,205,615,590)),('Windshield and hood',(200,410,1110,690))],
722:[('Woman and roof',(330,40,660,560)),('Car and sloth',(570,320,1100,680)),('Bridge/city',(620,120,1260,435))],
1444:[('Woman and hand',(110,180,450,670)),('Convertible rear',(370,315,920,690)),('Background vehicles',(850,200,1240,610))]}

def main():
 summary=[]
 for n in [361,722,1444]:
  out=OUT/str(n);out.mkdir(exist_ok=True);sequence=list(read_frames(ROOT/'IYTYT.mp4',n-3,6));a,b=sequence[2],sequence[3]
  c,valid,report,pa,pb=fit(a,b);report.update(frame=n,time_seconds=n*1001/24000)
  neighbors=[]
  for i,j in [(0,1),(1,2),(3,4),(4,5)]:
   _,_,r,_,_=fit(sequence[i],sequence[j],refine=False);neighbors.append({'frames':[n-3+i,n-3+j],**r['geometry_and_grade']})
  report['neighbors']=neighbors
  for name,im in [('left-original',a),('right-original',b),('right-similarity-grade',c)]:Image.fromarray(im).save(out/f'{name}.png')
  adjacent=Image.new('RGB',(2560,752));draw=ImageDraw.Draw(adjacent)
  adjacent.paste(Image.fromarray(a),(0,32));adjacent.paste(Image.fromarray(c),(1280,32))
  draw.text((12,10),f'Original frame {n-1}',fill='white');draw.text((1292,10),f'Original frame {n}: one similarity + affine RGB grade',fill='white');adjacent.save(out/'adjacent-native.png')
  with VideoWriter(out/'registered-blink.mp4',1280,720,'24',crf=12,preset='fast') as writer:
   for _ in range(8):
    for frame in [a,c]:
     for _ in range(8):writer.write(frame)
  crop_reports=[]
  for label,roi in ROIS[n]:
   x0,y0,x1,y1=roi;wa=x1-x0;ha=y1-y0
   crops=Image.new('RGB',(wa*2,ha+32));d=ImageDraw.Draw(crops)
   crops.paste(Image.fromarray(a[y0:y1,x0:x1]),(0,32));crops.paste(Image.fromarray(c[y0:y1,x0:x1]),(wa,32));d.text((8,10),f'{label}: before',fill='white');d.text((wa+8,10),'After rigid/grade alignment',fill='white')
   crops.save(out/(label.lower().replace(' ','-').replace('/','-')+'.png'))
   v=valid[y0:y1,x0:x1];crop_reports.append({'label':label,'roi':roi,'metrics':metrics(a[y0:y1,x0:x1],c[y0:y1,x0:x1],v)})
  report['regions']=crop_reports
  (out/'report.json').write_text(json.dumps(report,indent=2));summary.append(report)
  print(n,report['chosen'],report['geometry_and_grade'],flush=True)
 (OUT/'report.json').write_text(json.dumps(summary,indent=2))
if __name__=='__main__':main()
