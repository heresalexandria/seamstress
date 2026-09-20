"""Independent camera measurements on original and freshly encoded proof."""
from pathlib import Path
import json,sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import cv2,numpy as np
from PIL import Image,ImageDraw
from seamstress.media import read_frames,probe
OUT=Path(__file__).resolve().parent


def track(a,b,gap=1):
 a,b=[cv2.cvtColor(f,cv2.COLOR_RGB2GRAY) for f in [a,b]]
 mask=np.ones(a.shape,np.uint8)*255;mask[85:265,100:270]=0;mask[40:240,320:565]=0
 p=cv2.goodFeaturesToTrack(a,1600,.007,6,mask=mask,blockSize=5)
 q,ok,_=cv2.calcOpticalFlowPyrLK(a,b,p,None,winSize=(25,25),maxLevel=3)
 back,reverse,_=cv2.calcOpticalFlowPyrLK(b,a,q,None,winSize=(25,25),maxLevel=3)
 use=(ok[:,0]>0)&(reverse[:,0]>0)&(np.linalg.norm(back[:,0]-p[:,0],axis=1)<.8)
 p,q=p[use,0],q[use,0]
 affine,inside=cv2.estimateAffine2D(p,q,method=cv2.RANSAC,ransacReprojThreshold=1.2,maxIters=5000,confidence=.999,refineIters=30)
 c=np.array([319.75,179.75]);drift=affine[:,:2]@c+affine[:,2]-c
 residual=np.linalg.norm(p@affine[:,:2].T+affine[:,2]-q,axis=1)*2
 return {'x_scale_pct_per_frame':float((np.linalg.norm(affine[:,0])**(1/gap)-1)*100),'y_scale_pct_per_frame':float((np.linalg.norm(affine[:,1])**(1/gap)-1)*100),'center_x_native_px_per_frame':float(drift[0]*2/gap),'center_y_native_px_per_frame':float(drift[1]*2/gap),'inlier_fraction':float(inside.mean()),'median_inlier_residual_native_px':float(np.median(residual[inside[:,0]>0]))}


def main():
 original=list(read_frames(ROOT/'IYTYT.mp4',325,76,size=(640,360)))
 repaired=list(read_frames(OUT/'original-fixed-affine-24.mp4',12,76,size=(640,360)))
 results={}
 for label,frames in [('original',original),('fixed_affine',repaired)]:
  results[label]=[{'frame':325+i,'step':track(frames[i-1],frames[i]),'four_frame':track(frames[i-4],frames[i],4) if i>=4 else None} for i in range(1,len(frames))]
  print(label,'CUT',next(r for r in results[label] if r['frame']==361),flush=True)
  for lo,hi in [(343,360),(362,367),(379,385),(389,400)]:
   selected=[r['step'] for r in results[label] if lo<=r['frame']<=hi]
   print(label,lo,hi,{key:round(float(np.mean([s[key] for s in selected])),5) for key in selected[0]},flush=True)
 report=json.loads((OUT/'original-fixed-affine-24-transforms.json').read_text());matrices=np.array([f['source_to_output_matrix'] for f in report['frames']])
 # Fixed transform persists; the ratio of singular values is exactly fixed
 # throughout incoming frames, although isotropic scale/translation ease.
 condition=np.linalg.cond(matrices[361:,:2,:2]);corners=np.array([[0,0,1],[1279,0,1],[0,719,1],[1279,719,1]]).T
 minimum=1e9
 for m in matrices:
  xy=np.linalg.inv(m)@corners;minimum=min(minimum,xy[0].min(),xy[1].min(),1279-xy[0].max(),719-xy[1].max())
 assert np.max(abs(matrices[385:]-matrices[385]))<1e-12
 assert np.ptp(condition)<1e-12
 assert minimum>1.99
 metadata=probe(OUT/'original-fixed-affine-24.mp4');assert metadata['frame_count']==96 and metadata['width']==1280 and metadata['height']==720
 results['checks']={'encoded_frame_count':96,'native_dimensions':[1280,720],'persistent_from_frame':385,'persistent_through_frame':721,'anisotropy_singular_ratio':float(condition[0]),'anisotropy_ratio_variation':float(np.ptp(condition)),'minimum_native_border_clearance_px':float(minimum)}
 (OUT/'camera-audit.json').write_text(json.dumps(results,indent=2))
 # Static plot of independently measured rates. Single-frame measurements
 # intentionally retain the original animation cadence.
 image=Image.new('RGB',(1200,850),'#161b23');draw=ImageDraw.Draw(image)
 fields=[('x_scale_pct_per_frame','Horizontal scale % per frame'),('y_scale_pct_per_frame','Vertical scale % per frame'),('center_x_native_px_per_frame','Horizontal center px per frame'),('center_y_native_px_per_frame','Vertical center px per frame')]
 for row,(key,title) in enumerate(fields):
  lo_y=35+row*200;hi_y=lo_y+155
  series=[[ (r['frame'],r['step'][key]) for r in results[name]] for name in ['original','fixed_affine']]
  vals=[v for s in series for _,v in s];low,high=min(vals),max(vals);pad=max(.05,(high-low)*.08);low-=pad;high+=pad
  xy=lambda f,v:(70+(f-326)/74*1090,hi_y-(v-low)/(high-low)*(hi_y-lo_y))
  draw.text((70,lo_y-19),title,fill='white');draw.text((3,lo_y),f'{high:.3f}',fill='gray');draw.text((3,hi_y-10),f'{low:.3f}',fill='gray')
  if low<0<high:draw.line((70,xy(326,0)[1],1160,xy(326,0)[1]),fill='#333b44')
  draw.line((xy(361,0)[0],lo_y,xy(361,0)[0],hi_y),fill='#677386')
  for points,color in zip(series,['#ffba65','#72d8f5']):draw.line([xy(f,v) for f,v in points],fill=color,width=2)
  for f in [330,340,350,361,370,380,390,400]:draw.text((xy(f,low)[0]-10,hi_y+5),str(f),fill='gray')
 draw.text((780,5),'Original orange / corrected cyan',fill='white');image.save(OUT/'camera-rate-audit.png')
 print(results['checks'],flush=True)

if __name__=='__main__':main()
