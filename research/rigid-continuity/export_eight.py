"""Export bounded eight-join geometry; frame2888 is explicitly layer-pending."""
from pathlib import Path
import json,sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(Path(__file__).parent))
import cv2,numpy as np
from seamstress.media import read_frames
from all_joins import crop_needed,CUTS,describe
from first_join import matrix
OUT=Path(__file__).parent/'all-joins'


def camera(a,b,gap=4):
 ga,gb=[cv2.cvtColor(f,cv2.COLOR_RGB2GRAY) for f in [a,b]]
 points=cv2.goodFeaturesToTrack(ga,2400,.007,5,blockSize=5)
 # Enforce spatially distributed evidence before fitting camera motion.
 cells={};take=[]
 for i,(x,y) in enumerate(points[:,0]):
  cell=(int(x/640*6),int(y/360*4))
  if cells.get(cell,0)<35:take.append(i);cells[cell]=cells.get(cell,0)+1
 points=points[take]
 q,ok,_=cv2.calcOpticalFlowPyrLK(ga,gb,points,None,winSize=(25,25),maxLevel=3)
 back,valid,_=cv2.calcOpticalFlowPyrLK(gb,ga,q,None,winSize=(25,25),maxLevel=3)
 use=(ok[:,0]>0)&(valid[:,0]>0)&(np.linalg.norm(back[:,0]-points[:,0],axis=1)<1)
 p,q=points[use,0],q[use,0]
 affine,inside=cv2.estimateAffinePartial2D(p,q,method=cv2.RANSAC,ransacReprojThreshold=1.25,maxIters=5000,confidence=.999,refineIters=30)
 center=np.array([319.75,179.75]);drift=affine[:,:2]@center+affine[:,2]-center
 return {'log_scale_pct_per_frame':float(np.log(np.sqrt(np.linalg.det(affine[:,:2])))*100/gap),'center_x_native_px_per_frame':float(drift[0]*2/gap),'center_y_native_px_per_frame':float(drift[1]*2/gap),'inlier_fraction':float(inside.mean()),'point_count':len(p)}


def main():
 old=json.load(open(OUT/'neutral168_balanced12-matrices.json'));values=np.array(old['frame_matrices'])
 values[2888-1-168:2888+168+1]=np.eye(3)
 zoom=crop_needed(values);view=matrix(np.array([np.log(zoom),0,0,0]))
 combined=view@values;corners=np.array([[0,0,1],[1279,0,1],[0,719,1],[1279,719,1]],float).T
 source=np.linalg.inv(combined)@corners
 clear=np.minimum.reduce([source[:,0].min(axis=1),source[:,1].min(axis=1),1279-source[:,0].max(axis=1),719-source[:,1].max(axis=1)])
 assert clear.min()>1.99
 assert np.allclose(values[2888],np.eye(3))
 assert len(values)==3347
 summary={'frame_count':3347,'source_frame_mapping':'identity','global_join_corrections':[n for n in CUTS if n!=2888],
  'unresolved_joins':[{'cut':2888,'global_matrix':'identity','reason':'Depth-dependent cut: global affine shifts foreground10–18px to favor city. Separate source-only layer correction pending.'}],
  'geometry_support_frames_each_side':168,'rate_support_frames_each_side':12,'last_geometry_return_duration_frames':106,
  'constant_view_enlargement':zoom,'constant_crop_fraction_total_dimension':1-1/zoom,'constant_crop_per_side_native_px':[1280*(1-1/zoom)/2,720*(1-1/zoom)/2],
  'minimum_native_source_clearance_px':float(clear.min()),'minimum_clearance_frame':int(clear.argmin()),'exposed_source_pixel_count':0,
  'coverage_proof':'Every inverse-transformed output corner is at least2 pixels inside the original frame. Affine convexity proves the complete output rectangle is inside; two pixels cover cubic interpolation support.',
  'edge_extension_pixels':0,'maximum_affine_singular_ratio':float(np.linalg.cond(values[:,:2,:2]).max()),
  'production_status':'Research candidate; no full-film render. Join2888 explicitly unresolved. Geometry and rate estimates require visual review.'}
 recipe={'source':{'path':'IYTYT.mp4','width':1280,'height':720,'frame_count':3347,'fps_fraction':'24000/1001'},'frame_indexing':'zero-based; frame_matrices map original source pixels to output before view_matrix','geometry_mode':'affine','frame_matrices':values.tolist(),'view_matrix':view.tolist(),'edge_extension_pixels':0,'summary':summary}
 (OUT/'eight-global-matrices.json').write_text(json.dumps(recipe,indent=2))
 print(json.dumps(summary,indent=2),flush=True)
 # Re-measure actual original and transformed pictures, without rendering
 # a movie. These are independent four-frame balanced sparse tracks.
 down=np.diag([.5,.5,1]);audits=[]
 for cut in CUTS:
  start=cut-20;frames=list(read_frames(ROOT/'IYTYT.mp4',start,41,size=(640,360)))
  baseline=[cv2.warpAffine(f,(down@view@np.linalg.inv(down))[:2],(640,360),flags=cv2.INTER_CUBIC) for f in frames]
  transformed=[cv2.warpAffine(f,(down@combined[start+i]@np.linalg.inv(down))[:2],(640,360),flags=cv2.INTER_CUBIC) for i,f in enumerate(frames)]
  curves={}
  for name,sequence in [('source',baseline),('candidate',transformed)]:
   curves[name]=[{'frame':start+i,**camera(sequence[i-4],sequence[i],4)} for i in range(4,len(sequence))]
  fields=['log_scale_pct_per_frame','center_x_native_px_per_frame','center_y_native_px_per_frame']
  intervals={}
  for label,lo,hi in [('before',cut-8,cut-1),('cut',cut,cut+3),('after',cut+4,cut+11),('outside_rate_support',cut+15,cut+20)]:
   intervals[label]={name:{key:float(np.mean([r[key] for r in rows if lo<=r['frame']<=hi])) for key in fields} for name,rows in curves.items()}
  # Strong reversal flags compare the sign of the native motion on both
  # sides, not isolated noisy zero-velocity tracks.
  flags=[]
  for key,threshold in [('log_scale_pct_per_frame',.1),('center_x_native_px_per_frame',1),('center_y_native_px_per_frame',1)]:
   before=intervals['before']['source'][key];after=intervals['outside_rate_support']['source'][key]
   dominant=before if abs(before)>abs(after) else after
   if abs(dominant)<threshold or before*after < -threshold**2:continue
   opposite=[r['frame'] for r in curves['candidate'] if cut-10<=r['frame']<=cut+12 and r[key]*np.sign(dominant)<-threshold]
   if opposite:flags.append({'component':key,'frames':opposite,'native_before':before,'native_after':after,'interpretation':'Candidate opposite to the shared/dominant native direction; inspect parallax and camera estimate.'})
  audits.append({'cut':cut,'status':'layer_pending_unchanged_global' if cut==2888 else 'global_candidate','interval_means':intervals,'strong_direction_reversal_flags':flags,'curves':curves})
  print(cut,'means',intervals,'reversals',flags,flush=True)
  (OUT/'eight-global-camera-audit.json').write_text(json.dumps({'summary':summary,'join_audits':audits,'measurement':'Independent4-frame sparse correspondence, spatially balanced RANSAC similarity,640px analysis images; no encoded full film.'},indent=2))
 print('READY',OUT/'eight-global-matrices.json',flush=True)

if __name__=='__main__':main()
