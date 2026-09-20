"""Single bounded later-frame check of the same plate; exact hole masks only."""
from pathlib import Path
import sys,json,cv2,numpy as np
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from PIL import Image
from research.rigid_feasibility import features,metrics
from seamstress.media import read_frames
cv2.setNumThreads(4)
OUT=Path(__file__).resolve().parent
ref=np.array(Image.open(ROOT/'research/affine-feasibility/2888/right-original.png').convert('RGB'))
target=read_frames(ROOT/'IYTYT.mp4',2896,1)[0]
mr,mt=[np.array(Image.open(ROOT/f'research/layer2888/mask-{n}.png'))>127 for n in [2888,2896]]
_,p,q,_=features(target,ref)
def at(m,p):
 z=np.rint(p).astype(int);return m[z[:,1].clip(0,719),z[:,0].clip(0,1279)]>0
keep=~at(cv2.dilate(mt.astype(np.uint8),np.ones((15,15),np.uint8)),p)&~at(cv2.dilate(mr.astype(np.uint8),np.ones((15,15),np.uint8)),q)
m,good=cv2.estimateAffine2D(q[keep],p[keep],method=cv2.RANSAC,ransacReprojThreshold=2.5,maxIters=10000,confidence=.999,refineIters=50)
D=np.vstack([m,[0,0,1]]);err=np.linalg.norm(q[keep]@m[:2,:2].T+m[:2,2]-p[keep],axis=1)
reg=json.load(open(OUT/'registration-report.json'));layer=json.load(open(ROOT/'research/layer2888/v2/report.json'))
P=np.array(layer['background_matrix_incoming_to_outgoing'])@D@np.array(reg['matrix_generated_to_source_full'])
gen=np.array(Image.open(OUT/'generated-roi.png').convert('RGB'))
plate=cv2.warpAffine(gen,P[:2].astype(np.float32),(1280,720),flags=cv2.INTER_LANCZOS4).astype(np.float32)
col=reg['chosen_registration']['color_generated_to_source'];plate=plate*np.array(col['gain'])+np.array(col['bias']);plate=plate*np.array(layer['color']['gain'])+np.array(layer['color']['bias']);plate=np.clip(np.rint(plate),0,255).astype(np.uint8)
cov=cv2.warpAffine(np.ones(gen.shape[:2],np.uint8),P[:2].astype(np.float32),(1280,720),flags=cv2.INTER_NEAREST)>0
base=np.array(Image.open(ROOT/'research/layer2888/v2/result-2896.png').convert('RGB'));holes=np.array(Image.open(ROOT/'research/layer2888/v2/holes-2896.png'))>127
result=base.copy();take=holes&cov;result[take]=plate[take];assert np.array_equal(result[~holes],base[~holes])
Image.fromarray(result).save(OUT/'patched-2896.png');Image.fromarray(np.concatenate([base[200:625,325:883],result[200:625,325:883]],axis=1)).save(OUT/'comparison-2896-characters.png')
previous=np.array(Image.open(OUT/'patched-2888.png'));Image.fromarray(np.concatenate([previous[210:622,325:883],result[210:622,325:883]],axis=1)).save(OUT/'temporal-2888-2896.png')
report={'source_frame':2896,'plate_reference_frame':2888,'plate_reference_to_target_background_matrix':D.tolist(),'background_matches':int(keep.sum()),'background_inliers':int(good.sum()),'background_feature_residual_p50':float(np.median(err)),'background_feature_residual_p90':float(np.percentile(err,90)),'matrix_generated_to_output':P.tolist(),'hole_pixels':int(holes.sum()),'filled_hole_pixels':int(take.sum()),'remaining_holes':int((holes&~cov).sum()),'remaining_hole_bounds_xyxy':None,'changed_pixels_outside_holes':int(np.count_nonzero(np.any(result!=base,axis=2)&~holes)),'status':'Single later-frame feasibility check; not certified temporal repair.'}
ys,xs=np.where(holes&~cov)
if len(xs):report['remaining_hole_bounds_xyxy']=[int(xs.min()),int(ys.min()),int(xs.max()+1),int(ys.max()+1)]
(OUT/'frame-2896-report.json').write_text(json.dumps(report,indent=2));print(report)
