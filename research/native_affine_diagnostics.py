"""Fixed affine and RGB-grade source-only feasibility at remaining video joins."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import cv2,numpy as np
from PIL import Image,ImageDraw
from research.rigid_feasibility import features,metrics
from seamstress.registration import _fit_color,_measure,_gradient
from seamstress.media import read_frames,VideoWriter
cv2.setNumThreads(4)
OUT=ROOT/'research/affine-feasibility';OUT.mkdir(exist_ok=True)
ROIS={1083:[('Skateboarder',(755,230,1130,710)),('Sloth and convertible',(0,285,820,700)),('Bridge lines',(100,60,1200,325))],
1805:[('Distant protagonists',(495,295,830,610)),('Left buildings and car',(0,80,530,635)),('Right shop and fruitstall',(815,50,1280,620))],
2166:[('City grid center',(355,270,925,670)),('Skyline',(0,20,1280,295)),('Left city grid',(0,300,435,720))],
2527:[('Woman face and hands',(920,120,1250,580)),('Sloth and sofa',(180,150,850,665)),('Shelves and lamp',(525,0,940,490))],
2888:[('Flying characters',(355,175,910,685)),('Empire State and skyline',(220,10,575,710)),('Water and right skyline',(870,260,1280,695))],
3240:[('Characters',(470,220,795,610)),('Door and inscription',(285,0,975,345)),('Right lion and column',(895,0,1280,645))]}

def fit(a,b,refine=True):
 h,w=a.shape[:2];_,pa,pb,_=features(a,b)
 A,inliers=cv2.estimateAffine2D(pb,pa,method=cv2.RANSAC,ransacReprojThreshold=2.25,maxIters=10000,confidence=.999,refineIters=50)
 if A is None:raise RuntimeError('No affine registration')
 initial=np.vstack([A,[0,0,1]]);candidates=[('native_sift_affine',initial)]
 if refine:
  aa,bb=[cv2.resize(x,(640,360),interpolation=cv2.INTER_AREA) for x in [a,b]]
  gray=[cv2.GaussianBlur(cv2.cvtColor(x,cv2.COLOR_RGB2GRAY).astype(np.float32)/255,(0,0),1.1) for x in [aa,bb]]
  down=np.diag([.5,.5,1]);up=np.diag([2,2,1])
  try:
   cc,v=cv2.findTransformECC(gray[0],gray[1],np.linalg.inv(down@initial@up)[:2].astype(np.float32),cv2.MOTION_AFFINE,(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,130,1e-6),None,5)
   candidates.append(('ecc_affine',up@np.linalg.inv(np.vstack([v,[0,0,1]]))@down))
  except cv2.error:pass
 rows=[];best=None
 for name,m in candidates:
  singular=np.linalg.svd(m[:2,:2],compute_uv=False);anis=float(max(singular)/min(singular))
  bounds={'singular_values':singular.tolist(),'anisotropy':anis,'determinant':float(np.linalg.det(m[:2,:2]))}
  accepted=bool(.85<min(singular) and max(singular)<1.18 and anis<1.055)
  if not accepted:
   rows.append({'method':name,'accepted':False,'bounds':bounds});continue
  c=cv2.warpAffine(b,m[:2].astype(np.float32),(w,h),flags=cv2.INTER_LANCZOS4,borderMode=cv2.BORDER_CONSTANT)
  valid=cv2.warpAffine(np.ones((h,w),np.uint8),m[:2].astype(np.float32),(w,h),flags=cv2.INTER_NEAREST)>0
  valid=cv2.erode(valid.astype(np.uint8),np.ones((5,5),np.uint8))>0
  valid[:10]=False;valid[-10:]=False;valid[:,:10]=False;valid[:,-10:]=False
  color=_fit_color(a,c,valid);graded=np.clip(np.rint(c.astype(float)*color['gain']+color['bias']),0,255).astype(np.uint8)
  quality=_measure(a,c,valid,color);score=quality['trimmed_corrected_mae']+.4*quality['gradient_mae']
  record={'method':name,'accepted':True,'bounds':bounds,'matrix_right_to_left':m.tolist(),'color':color,'score':float(score),
    'raw':metrics(a,b,valid),'geometry_only':metrics(a,c,valid),'geometry_and_grade':metrics(a,graded,valid)}
  rows.append(record)
  if best is None or score<best[0]:best=(score,graded,valid,record,c)
 if best is None:raise RuntimeError('All affine candidates exceeded geometric bounds')
 _,graded,valid,record,warped=best;report=dict(record);report['candidates']=rows
 report['feature_matches']=len(pa);report['feature_inlier_fraction']=float(inliers.mean())
 ga,gb=[_gradient(cv2.cvtColor(x,cv2.COLOR_RGB2GRAY)) for x in [a,warped]]
 flat=cv2.erode(((ga<3)&(gb<3)).astype(np.uint8),np.ones((9,9),np.uint8))>0
 report['flat_region_color_fit']=_fit_color(a,warped,valid&flat)
 return graded,valid,report

results=[]
known=json.load(open(ROOT/'research/rigid-feasibility/361/fixed-transform-comparison/report.json'))['chosen']['affine']
results.append({'frame':361,'left_source_frame':360,'right_source_frame':361,'matrix_right_to_left':known['matrix_right_to_left'],'color':known['color'],'geometry_and_grade':known['metrics'],'bounds':known['bounds'],'artifacts':'research/rigid-feasibility/361/fixed-transform-comparison'})
for n in [1083,1805,2166,2527,2888,3240]:
 out=OUT/str(n);out.mkdir(exist_ok=True)
 source=list(read_frames(ROOT/'IYTYT.mp4',n-2,5));a,b=source[1],source[2]
 c,valid,r=fit(a,b);r.update(frame=n,left_source_frame=n-1,right_source_frame=n,time_seconds=n*1001/24000)
 r['neighbors']=[]
 for i,j in [(0,1),(2,3),(3,4)]:
  _,_,rr=fit(source[i],source[j],refine=False)
  r['neighbors'].append({'frames':[n-2+i,n-2+j],**rr['geometry_and_grade']})
 for name,im in [('left-original',a),('right-original',b),('right-affine-grade',c)]:Image.fromarray(im).save(out/f'{name}.png')
 adjacent=Image.new('RGB',(2560,752));d=ImageDraw.Draw(adjacent)
 adjacent.paste(Image.fromarray(a),(0,32));adjacent.paste(Image.fromarray(c),(1280,32));d.text((12,10),f'Original frame {n-1}',fill='white');d.text((1292,10),f'Original frame {n}, fixed affine + RGB grade',fill='white');adjacent.save(out/'adjacent-native.png')
 with VideoWriter(out/'registered-blink.mp4',1280,720,'24',crf=12,preset='fast') as writer:
  for _ in range(6):
   for frame in [a,c]:
    for _ in range(8):writer.write(frame)
 r['regions']=[]
 for name,roi in ROIS[n]:
  x0,y0,x1,y1=roi;cw,ch=x1-x0,y1-y0;sheet=Image.new('RGB',(cw*2,ch+32));draw=ImageDraw.Draw(sheet)
  sheet.paste(Image.fromarray(a[y0:y1,x0:x1]),(0,32));sheet.paste(Image.fromarray(c[y0:y1,x0:x1]),(cw,32));draw.text((8,10),name+' / original',fill='white');draw.text((cw+8,10),'After fixed affine / grade',fill='white');sheet.save(out/(name.lower().replace(' ','-')+'.png'))
  r['regions'].append({'name':name,'roi':roi,'metrics':metrics(a[y0:y1,x0:x1],c[y0:y1,x0:x1],valid[y0:y1,x0:x1])})
 (out/'report.json').write_text(json.dumps(r,indent=2));results.append(r)
 (ROOT/'research/affine-all-joins.json').write_text(json.dumps({'source':'IYTYT.mp4','frame_indexing':'zero-based first incoming frame; matrices map incoming source pixels to outgoing frame','method':'fixed affine, one resampling, per-channel affine RGB grade; no temporal synthesis','seams':results},indent=2))
 print(n,r['method'],r['geometry_and_grade'],flush=True)
