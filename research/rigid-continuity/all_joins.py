"""Feasibility of persistent versus slowly distributed original-frame geometry.

Creates matrices and selected stills only. No full video or neural inference.
"""
from pathlib import Path
import json,sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(Path(__file__).parent))
import cv2,numpy as np
from scipy.linalg import logm,expm
from scipy.optimize import brentq
from PIL import Image,ImageDraw
from seamstress.media import read_frames
from seamstress.registration import register_pair
from first_join import matrix,params,integral_decay,slope
OUT=Path(__file__).parent/'all-joins';OUT.mkdir(exist_ok=True)
CUTS=[361,722,1083,1444,1805,2166,2527,2888,3240];N=3347;W,H=1280,720;CENTER=np.array([639.5,359.5])


def sim(m):
 a=m[:2,:2];u,s,v=np.linalg.svd(a);rotation=u@v
 r=np.eye(3);r[:2,:2]=rotation*s.mean();r[:2,2]=a@CENTER+m[:2,2]-r[:2,:2]@CENTER
 return r


def fit_rate(ref,images,offsets):
 ps=[];conf=[]
 for image,t in zip(images,offsets):
  reg=register_pair(ref,image);ps.append(params(sim(np.linalg.inv(np.array(reg['matrix'])))));conf.append(reg['confidence'])
 velocity,rmse=slope(offsets,ps)
 return velocity,{'confidence_min':float(min(conf)),'confidence_all':conf,'fit_rmse':rmse.tolist(),'offsets':offsets,'parameters':np.asarray(ps).tolist()}


def describe(m):
 a=m[:2,:2];singular=np.linalg.svd(a,compute_uv=False)
 return {'scale_geometric_mean':float(np.sqrt(np.linalg.det(a))),'axis_x_scale':float(np.linalg.norm(a[:,0])),'axis_y_scale':float(np.linalg.norm(a[:,1])),'axis_y_over_x':float(np.linalg.norm(a[:,1])/np.linalg.norm(a[:,0])),'singular_ratio':float(singular[0]/singular[1]),'center_shift_native_px':(a@CENTER+m[:2,2]-CENTER).tolist()}


def crop_needed(matrices):
 corners=np.array([[0,0,1],[1279,0,1],[0,719,1],[1279,719,1]],float).T
 inverse=np.linalg.inv(matrices)
 def room(z):
  q=np.linalg.inv(matrix(np.array([np.log(z),0,0,0])))@corners
  points=inverse@q
  return min(points[:,0].min()-2,points[:,1].min()-2,1277-points[:,0].max(),717-points[:,1].max())
 if room(1)>=0:return 1.
 if room(4)<0:return None
 return float(brentq(room,1,4)+1e-6)


def main():
 cache=OUT/'cut-measurements.json'
 if cache.exists():records=json.loads(cache.read_text())
 else:
  records=[]
  for cut in CUTS:
   frames=list(read_frames(ROOT/'IYTYT.mp4',cut-13,26))
   reg=register_pair(frames[12],frames[13]);r=np.array(reg['matrix'])
   pre,pre_evidence=fit_rate(frames[12],[frames[i] for i in [0,4,8,12]],[-12,-8,-4,0])
   post,post_evidence=fit_rate(frames[13],[frames[i] for i in [13,17,21,25]],[0,4,8,12])
   delta=pre-post
   radius=np.sqrt((W/2)**2+(H/2)**2)
   speed=float(np.sqrt(np.sum(delta[2:]**2)+(radius**2)*(delta[0]**2+delta[1]**2)))
   reliable=min(pre_evidence['confidence_min'],post_evidence['confidence_min'])>.3
   use_rate=speed>.5 and reliable
   edit=matrix(pre)@r
   records.append({'cut':cut,'registration':reg,'pre_rate':pre.tolist(),'post_rate':post.tolist(),'pre_evidence':pre_evidence,'post_evidence':post_evidence,'rate_difference_image_rms_native_px_per_frame':speed,'ease_rate':use_rate,'rate_decision':'rate difference exceeds 0.5 native px/frame and fit confidence exceeds .3' if use_rate else 'small rate difference or inadequate camera evidence','edit_correction':edit.tolist(),'edit_description':describe(edit)})
   cache.write_text(json.dumps(records,indent=2));print('Measured',cut,describe(edit),'rate_delta',speed,'ease',use_rate,flush=True)
 variants={name:np.repeat(np.eye(3)[None],N,axis=0) for name in ['cumulative_static','cumulative_rate24','neutral168','neutral168_tangent','neutral168_balanced12']}
 accum_static=np.eye(3);accum_rate=np.eye(3)
 for j,rec in enumerate(records):
  cut=rec['cut'];end=CUTS[j+1] if j+1<len(CUTS) else N
  edit=np.array(rec['edit_correction']);delta=np.array(rec['pre_rate'])-np.array(rec['post_rate']);delta*=int(rec['ease_rate'])
  accum_static=accum_static@edit;variants['cumulative_static'][cut:end]=accum_static
  for n in range(cut,end):variants['cumulative_rate24'][n]=accum_rate@matrix(delta*integral_decay(n-cut,24))@edit
  accum_rate=variants['cumulative_rate24'][end-1].copy()
  log_edit=np.real_if_close(logm(edit)).astype(float)
  balanced_edit=matrix((np.array(rec['pre_rate'])+np.array(rec['post_rate']))/2 if rec['ease_rate'] else np.array(rec['pre_rate']))@np.array(rec['registration']['matrix'])
  balanced_log=np.real_if_close(logm(balanced_edit)).astype(float)
  for side in [-1,1]:
   anchor=cut-1 if side<0 else cut
   available=anchor if side<0 else N-1-anchor
   # Last segment is short (107 frames). It cannot support a 168-frame
   # return, so the return is truncated at the movie endpoint, reported.
   duration=min(168,available)
   for d in range(duration+1):
    n=anchor+side*d;u=d/duration if duration else 1;weight=1-(3*u*u-2*u*u*u)
    base=expm(side*.5*log_edit*weight)
    variants['neutral168'][n]=base
    # Symmetric endpoint rate reconciliation using a slow compact tangent.
    # Corrected derivatives meet at mean(pre,post), at the cost of a
    # gradual camera excursion / reversal required to return to identity.
    tangent=delta*.5*d*(1-u)**3
    variants['neutral168_tangent'][n]=matrix(tangent)@base
    half_distance=d+.5
    u_rate=np.clip(1-half_distance/12,0,1)
    # delta=pre-post; rate correction integrates to a negative shared
    # offset for positive incoming zoom, then returns to zero as native
    # outgoing motion takes over. The output camera need not reverse.
    balanced_offset=-delta*6*(u_rate**3-.5*u_rate**4)
    variants['neutral168_balanced12'][n]=matrix(balanced_offset)@expm(side*.5*balanced_log*weight)
 summaries={}
 for name,values in variants.items():
  crop=crop_needed(values);all_desc=[describe(m) for m in values]
  summaries[name]={'minimum_global_enlargement_no_border_fill':crop,'crop_fraction_total_dimension':None if crop is None else 1-1/crop,'final_geometry':all_desc[-1],'minimum_scale':min(d['scale_geometric_mean'] for d in all_desc),'maximum_scale':max(d['scale_geometric_mean'] for d in all_desc),'maximum_anisotropy_ratio':max(d['singular_ratio'] for d in all_desc),'maximum_center_displacement_native_px':max(np.linalg.norm(d['center_shift_native_px']) for d in all_desc),'cut_geometries':[{'cut':cut,'before':all_desc[cut-1],'after':all_desc[cut],'late':all_desc[min(cut+24,N-1)]} for cut in CUTS]}
  (OUT/f'{name}-matrices.json').write_text(json.dumps({'method':name,'source':'IYTYT.mp4','frame_matrices':values.tolist(),'view_matrix':matrix(np.array([np.log(crop),0,0,0])).tolist() if crop else None,'summary':summaries[name]},indent=2))
 print(json.dumps({k:{x:v for x,v in s.items() if x!='cut_geometries'} for k,s in summaries.items()},indent=2),flush=True)
 (OUT/'summary.json').write_text(json.dumps({'cuts':records,'variants':summaries,'notes':['All estimates are provisional; affine correspondence is not proof of global edit geometry.','neutral168 returns toward original framing over seven seconds on either side and exactly matches the fitted boundary; its zero correction slopes leave original camera-rate discontinuities.','neutral168_tangent also matches local camera rates, but returning toward identity necessarily introduces camera excursions or reversals.','Final segment is only107 frames: recovery has to finish in4.46 seconds or persist beyond the movie.','No color correction; no nonrigid image deformation; one original frame per output.']},indent=2))
 for n in [3264,3346]:
  frame=read_frames(ROOT/'IYTYT.mp4',n,1)[0]
  sheet=Image.new('RGB',(1280*2,765*3),'#111');draw=ImageDraw.Draw(sheet)
  panels=[('ORIGINAL',frame)]
  for name,values in variants.items():
   # Display unavoidable blank boundaries in a magenta matte before any
   # global crop, so the feasibility cost is visible, not concealed.
   warped=cv2.warpAffine(frame,values[n,:2],(W,H),flags=cv2.INTER_CUBIC,borderMode=cv2.BORDER_CONSTANT,borderValue=(255,0,255))
   panels.append((name,warped))
  for i,(name,img) in enumerate(panels):
   x=(i%2)*1280;y=(i//2)*765;sheet.paste(Image.fromarray(img),(x,y));draw.text((x+10,y+730),f'{n} {name} — no global crop; magenta = unavailable source',fill='white')
  sheet.save(OUT/f'late-scene-{n}.jpg',quality=95)
 print('Done: matrices, summaries, selected stills only',flush=True)

if __name__=='__main__':main()
