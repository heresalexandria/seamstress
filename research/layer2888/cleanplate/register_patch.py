"""Register one generated static background plate; use only explicit holes."""
from pathlib import Path
import sys,json,cv2,numpy as np
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from PIL import Image,ImageDraw
from research.rigid_feasibility import features,metrics
from seamstress.registration import _fit_color,_measure
cv2.setNumThreads(4)
OUT=Path(__file__).resolve().parent
source=np.array(Image.open(OUT/'source-2888-roi.png').convert('RGB'))
generated=np.array(Image.open(OUT/'generated-roi.png').convert('RGB'))
g=cv2.resize(generated,(640,512),interpolation=cv2.INTER_LANCZOS4)
fg=np.array(Image.open(ROOT/'research/layer2888/mask-2888.png'))[160:672,288:928]>127
visible=cv2.dilate(fg.astype(np.uint8),np.ones((19,19),np.uint8))==0
_,pa,pb,_=features(source,g)
p=np.rint(pa).astype(int);keep=visible[p[:,1].clip(0,511),p[:,0].clip(0,639)]
a,good=cv2.estimateAffine2D(pb[keep],pa[keep],method=cv2.RANSAC,ransacReprojThreshold=1.6,maxIters=10000,confidence=.999,refineIters=50)
M=np.vstack([a,[0,0,1]])
# Test only fixed global affine refinement, never a generated architecture warp.
candidates=[('sift_affine',M)]
try:
 ag,bg=[cv2.GaussianBlur(cv2.cvtColor(x,cv2.COLOR_RGB2GRAY).astype(np.float32)/255,(0,0),1) for x in [source,g]]
 _,mm=cv2.findTransformECC(ag,bg,np.linalg.inv(M)[:2].astype(np.float32),cv2.MOTION_AFFINE,(cv2.TERM_CRITERIA_COUNT|cv2.TERM_CRITERIA_EPS,150,1e-7),visible.astype(np.uint8)*255,5)
 candidates.append(('ecc_affine',np.linalg.inv(np.vstack([mm,[0,0,1]]))))
except cv2.error:pass
rows=[]
for name,m in candidates:
 aligned=cv2.warpAffine(g,m[:2].astype(np.float32),(640,512),flags=cv2.INTER_LANCZOS4)
 valid=visible&(cv2.warpAffine(np.ones((512,640),np.uint8),m[:2].astype(np.float32),(640,512))>0)
 valid[:8]=False;valid[-8:]=False;valid[:,:8]=False;valid[:,-8:]=False
 col=_fit_color(source,aligned,valid);meas=_measure(source,aligned,valid,col)
 score=meas['trimmed_corrected_mae']+.4*meas['gradient_mae']
 error=np.linalg.norm(pb[keep]@m[:2,:2].T+m[:2,2]-pa[keep],axis=1)
 row={'method':name,'matrix_generated_resized_to_source_roi':m.tolist(),'color_generated_to_source':col,'score':score,'visible_background_metrics':metrics(source,np.clip(aligned*col['gain']+col['bias'],0,255).astype(np.uint8),valid),'feature_residual_p50':float(np.median(error)),'feature_residual_p90':float(np.percentile(error,90))}
 rows.append(row)
chosen=min(rows,key=lambda x:x['score']);m=np.array(chosen['matrix_generated_resized_to_source_roi']);col=chosen['color_generated_to_source']
# Compose generated-native -> resized ROI -> sourcefull -> outgoing-coordinate.
S=np.diag([640/generated.shape[1],512/generated.shape[0],1]);T=np.array([[1,0,288],[0,1,160],[0,0,1.]])
plate_to_source=T@m@S
r=json.load(open(ROOT/'research/layer2888/v2/report.json'));BG=np.array(r['background_matrix_incoming_to_outgoing'])
P=BG@plate_to_source
plate=cv2.warpAffine(generated,P[:2].astype(np.float32),(1280,720),flags=cv2.INTER_LANCZOS4).astype(np.float32)
plate=plate*np.array(col['gain'])+np.array(col['bias']);plate=plate*np.array(r['color']['gain'])+np.array(r['color']['bias']);plate=np.clip(np.rint(plate),0,255).astype(np.uint8)
coverage=cv2.warpAffine(np.ones(generated.shape[:2],np.uint8),P[:2].astype(np.float32),(1280,720),flags=cv2.INTER_NEAREST)>0
base=np.array(Image.open(ROOT/'research/layer2888/v2/result-2888.png').convert('RGB'));holes=np.array(Image.open(ROOT/'research/layer2888/v2/holes-2888.png'))>127
take=holes&coverage;result=base.copy();result[take]=plate[take]
assert np.array_equal(result[~holes],base[~holes])
Image.fromarray(result).save(OUT/'patched-2888.png');Image.fromarray(plate).save(OUT/'registered-plate-in-output.png')
# Native paired crop showing untouched result and only-hole replacement.
for name,box in [('characters',(330,210,880,620)),('sloth-feet',(425,420,581,560)),('woman-flame-and-feet',(649,381,848,614)),('arms',(340,305,868,386))]:
 x0,y0,x1,y1=box;left=base[y0:y1,x0:x1];right=result[y0:y1,x0:x1];sheet=np.concatenate([left,right],axis=1)
 Image.fromarray(sheet).save(OUT/f'comparison-{name}.png')
report={'source_frame':2888,'method':'one built-in static clean plate, fixed global affine registration, exact hole-mask copy only','feature_matches_visible':int(keep.sum()),'sift_inliers':int(good.sum()),'chosen_registration':chosen,'candidates':rows,'matrix_generated_to_source_full':plate_to_source.tolist(),'matrix_generated_to_output':P.tolist(),'hole_pixels':int(holes.sum()),'filled_hole_pixels':int(take.sum()),'remaining_uncovered_pixels':int((holes&~coverage).sum()),'changed_pixels_outside_holes':int(np.count_nonzero(np.any(result!=base,axis=2)&~holes)),'foreground_source_pixels_unchanged':True,'status':'Awaiting native edge/texture review; not integrated'}
(OUT/'registration-report.json').write_text(json.dumps(report,indent=2));print(report,flush=True)
