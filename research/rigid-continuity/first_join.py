"""Original-frame-only persistent geometry correction of the first generation cut.

No model inference, optical-flow warp, crossfade, frame replacement, or colour
processing. Feature correspondences estimate a global camera transform only.
Each output picture samples one source picture. A constant optional affine
restores the cut aspect ratio; only the similarity camera motion then varies.
"""
from pathlib import Path
import argparse, json, sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import cv2,numpy as np
from scipy.optimize import least_squares,brentq
from PIL import Image,ImageDraw
from seamstress.media import read_frames,VideoWriter

OUT=Path(__file__).resolve().parent
CENTER=np.array([639.5,359.5]);W,H=1280,720


def params(m):
 a=m[:2,:2]
 return np.array([np.log(np.sqrt(np.linalg.det(a))),np.arctan2(a[1,0],a[0,0]),*(a@CENTER+m[:2,2]-CENTER)])


def matrix(p):
 s,a=np.exp(p[0]),p[1]
 m=np.eye(3);m[:2,:2]=s*np.array([[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]])
 m[:2,2]=CENTER+p[2:]-m[:2,:2]@CENTER
 return m


def features(frame):
 gray=cv2.cvtColor(cv2.resize(frame,(640,360),interpolation=cv2.INTER_AREA),cv2.COLOR_RGB2GRAY)
 # Exclude moving drawn faces, arms, and torso; retain windshield, hood,
 # mirror, and distributed background. This mask is specific to first join.
 mask=np.ones(gray.shape,np.uint8)*255
 mask[85:265,100:270]=0;mask[40:240,320:565]=0
 sift=cv2.SIFT_create(nfeatures=5000,contrastThreshold=.009,edgeThreshold=14)
 k,d=sift.detectAndCompute(gray,mask)
 return np.float32([x.pt for x in k]),d


def fit(reference,moving):
 lp,ld=reference;rp,rd=moving
 bf=cv2.BFMatcher()
 lr=bf.knnMatch(ld,rd,k=2);rl=bf.knnMatch(rd,ld,k=2)
 rev={m.queryIdx:m.trainIdx for m,n in rl if m.distance<.8*n.distance}
 matches=[m for m,n in lr if m.distance<.8*n.distance and rev.get(m.trainIdx)==m.queryIdx]
 cells={};balanced=[]
 for m in sorted(matches,key=lambda x:x.distance):
  cell=tuple(np.minimum((lp[m.queryIdx]/[640,360]*[6,4]).astype(int),[5,3]))
  if cells.get(cell,0)<20:balanced.append(m);cells[cell]=cells.get(cell,0)+1
 p=np.array([rp[m.trainIdx] for m in balanced]);q=np.array([lp[m.queryIdx] for m in balanced])
 affine,inside=cv2.estimateAffinePartial2D(p,q,method=cv2.RANSAC,ransacReprojThreshold=1.25,maxIters=10000,confidence=.999,refineIters=50)
 if affine is None:raise RuntimeError('Similarity fit failed')
 inside=inside[:,0].astype(bool);pred=p@affine[:,:2].T+affine[:,2]
 error=np.linalg.norm(pred-q,axis=1)*2
 grid=np.minimum((q[inside]/[640,360]*[4,4]).astype(int),[3,3])
 coverage=len(np.unique(grid[:,1]*4+grid[:,0]))/16
 m=np.vstack([affine,[0,0,1]]);m[:2,2]*=2
 regions={}
 for name,select in [('hood',q[:,1]>265),('windshield_edge',(q[:,1]>75)&(q[:,1]<115)),('background',(q[:,1]<75)|((q[:,0]<90)&(q[:,1]<270)))]:
  if select.sum():regions[name]={'count':int(select.sum()),'median_native_px':float(np.median(error[select])),'p90_native_px':float(np.percentile(error[select],90))}
 return m,{'matches':len(p),'inliers':int(inside.sum()),'inlier_fraction':float(inside.mean()),'coverage_4x4':coverage,'median_inlier_error_native_px':float(np.median(error[inside])),'p90_all_error_native_px':float(np.percentile(error,90)),'regions':regions}


def slope(frames,values):
 x=np.asarray(frames,dtype=float);x-=x.mean();y=np.asarray(values)
 initial=np.linalg.lstsq(np.column_stack([np.ones(len(x)),x]),y,rcond=None)[0]
 # Scale each component before robust regression; do not let pixel translation
 # dominate estimates of zoom/rotation.
 scales=np.array([.001,.0005,.7,.7])
 def residual(coeff):return ((coeff.reshape(2,4)[0]+x[:,None]*coeff.reshape(2,4)[1]-y)/scales).ravel()
 fit=least_squares(residual,initial.ravel(),loss='soft_l1',f_scale=1).x.reshape(2,4)
 return fit[1],np.sqrt(np.mean((fit[0]+x[:,None]*fit[1]-y)**2,axis=0))


def integral_decay(t,duration):
 u=np.clip(np.asarray(t,dtype=float)/duration,0,1)
 # Integral of 1-smoothstep(u); slope begins at 1, ends at 0 with zero
 # acceleration at either boundary, then stays constant indefinitely.
 return duration*(u-u**3+.5*u**4)


def crop_scale(transforms):
 corners=np.array([[0,0,1],[W-1,0,1],[0,H-1,1],[W-1,H-1,1]],float).T
 def room(z):
  zoom=matrix(np.array([np.log(z),0,0,0]));smallest=1e9
  for transform in transforms:
   source=np.linalg.inv(zoom@transform)@corners
   smallest=min(smallest,source[0].min()-2,source[1].min()-2,W-3-source[0].max(),H-3-source[1].max())
  return smallest
 if room(1)>=0:return 1.
 if room(1.2)<0:raise RuntimeError('Required crop exceeds 20% enlargement')
 return float(brentq(room,1,1.2)+1e-6)


def render(source,transforms):
 return [cv2.warpAffine(f,m[:2],(W,H),flags=cv2.INTER_CUBIC,borderMode=cv2.BORDER_CONSTANT) for f,m in zip(source,transforms)]


def main():
 parser=argparse.ArgumentParser();parser.add_argument('--duration',type=int,default=36);parser.add_argument('--skip-preview',action='store_true');parser.add_argument('--geometry',choices=['similarity','fixed-affine'],default='similarity');args=parser.parse_args()
 cache=OUT/'camera-measurements.json'
 if cache.exists():measured=json.loads(cache.read_text())
 else:
  source=read_frames(ROOT/'IYTYT.mp4',325,76);all_features=[features(f) for f in source];reference=all_features[360-325]
  measured=[]
  for n,feature in zip(range(325,401),all_features):
   m,diagnostics=fit(reference,feature);forward=np.linalg.inv(m)
   measured.append({'frame':n,'source_to_360':m.tolist(),'pose_from_360':forward.tolist(),'parameters':params(forward).tolist(),'diagnostics':diagnostics})
   if n%10==0:print('Measured',n,flush=True)
  cache.write_text(json.dumps(measured,indent=2))
 byframe={r['frame']:r for r in measured}
 pre_indices=list(range(343,361));post_indices=list(range(365,397))
 vpre,pre_rmse=slope(pre_indices,[byframe[n]['parameters'] for n in pre_indices])
 cut=np.array(byframe[361]['pose_from_360'])
 affine_evidence=None
 if args.geometry=='fixed-affine':
  from seamstress.registration import register_pair
  cut_images=read_frames(ROOT/'IYTYT.mp4',360,2)
  affine_evidence=register_pair(cut_images[0],cut_images[1])
  cut=np.linalg.inv(np.array(affine_evidence['matrix']))
 initial=matrix(vpre)@np.linalg.inv(cut)
 # Estimate outgoing rate in the corrected coordinate system.
 vpost,post_rmse=slope(post_indices,[params(initial@np.array(byframe[n]['pose_from_360'])) for n in post_indices])
 delta=vpre-vpost
 transforms=[]
 for n in range(722):
  correction=np.eye(3) if n<361 else matrix(delta*integral_decay(n-361,args.duration))@initial
  transforms.append(correction)
 enlargement=crop_scale(transforms);zoom=matrix(np.array([np.log(enlargement),0,0,0]));output=[zoom@t for t in transforms]
 name=f'original-{"fixed-affine" if args.geometry=="fixed-affine" else "rigid"}-{args.duration}'
 report={'method':'One original source frame per output frame, persistent fixed geometry correction plus similarity camera-rate easing',
  'geometry':args.geometry,'fixed_affine_evidence':affine_evidence,
  'fixed_anisotropy_singular_ratio':float(np.linalg.cond(initial[:2,:2])),
  'source':'IYTYT.mp4','join_frame':361,'fit_window':[325,400],'rate_fit_frames_before':pre_indices,'rate_fit_frames_after':post_indices,
  'parameter_units':['log scale','radians','native center x px','native center y px'],
  'incoming_rate_per_frame':vpre.tolist(),'outgoing_rate_per_frame':vpost.tolist(),
  'incoming_fit_rmse':pre_rmse.tolist(),'outgoing_fit_rmse':post_rmse.tolist(),
  'cut_raw_forward_parameters':params(cut).tolist(),'cut_similarity_evidence':byframe[361]['diagnostics'],
  'rate_reconciliation_frames':args.duration,'initial_correction':initial.tolist(),
  'persistent_correction':transforms[-1].tolist(),'persistent_correction_parameters':params(transforms[-1]).tolist(),
  'constant_enlargement':enlargement,'crop_fraction_each_dimension':1-1/enlargement,'crop_per_side_native_px':[W*(1-1/enlargement)/2,H*(1-1/enlargement)/2],
  'same_source_frame_and_draw_cadence':True,'no_colour_correction':True,'borders':'No fill/reflection; constant crop keeps every sample inside original image by two pixels',
  'requires_visual_review':True,'limitations':['Similarity cannot remove changed illustration geometry or depth parallax.','Feature fit to rigid car/background may not describe every scene depth.','Camera moves over the original held drawings; no pose or frame synthesis.'],
  'frames':[{'frame':n,'source_frame':n,'source_to_output_matrix':m.tolist()} for n,m in enumerate(output)]}
 (OUT/f'{name}-transforms.json').write_text(json.dumps(report,indent=2))
 print(json.dumps({k:v for k,v in report.items() if k!='frames'},indent=2),flush=True)
 if args.skip_preview:return
 start,count=313,96
 source=list(read_frames(ROOT/'IYTYT.mp4',start,count));candidate=render(source,output[start:start+count]);baseline=render(source,[zoom]*count)
 with VideoWriter(OUT/f'{name}.mp4',W,H,'24000/1001',crf=14,preset='fast') as writer:
  for f in candidate:writer.write(f)
 with VideoWriter(OUT/f'{name}-comparison.mp4',W*2,H,'24000/1001',crf=14,preset='fast') as writer:
  for a,b in zip(baseline,candidate):
   frame=np.concatenate([a,b],axis=1)
   cv2.putText(frame,'ORIGINAL - equal constant crop',(12,28),cv2.FONT_HERSHEY_SIMPLEX,.7,(255,255,255),1,cv2.LINE_AA)
   cv2.putText(frame,'ORIGINAL FRAMES + fixed geometry and camera ease',(1292,28),cv2.FONT_HERSHEY_SIMPLEX,.7,(255,255,255),1,cv2.LINE_AA)
   writer.write(frame)
 sheet=Image.new('RGB',(1280,400*5),'#151820');draw=ImageDraw.Draw(sheet)
 for row,n in enumerate([359,360,361,370,397]):
  for col,frames in enumerate([baseline,candidate]):
   sheet.paste(Image.fromarray(frames[n-start]).resize((640,360)),(col*640,row*400));draw.text((col*640+8,row*400+370),f'{"ORIGINAL" if col==0 else "FIXED GEOMETRY"} {n}',fill='white')
 sheet.save(OUT/f'{name}-contact.jpg',quality=95)
 print('Saved',str(OUT/f'{name}.mp4'),flush=True)

if __name__=='__main__':main()
