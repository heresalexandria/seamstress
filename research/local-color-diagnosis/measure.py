"""Color evidence only: correspondence samples flat native source-conform regions.
No new grading or geometry is applied to any source frame or output video.
"""
from pathlib import Path
import sys,json,cv2,numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'research'))
from PIL import Image,ImageDraw
from skimage.color import rgb2lab,deltaE_ciede2000
from local_color_probe import observations,apply_samples
cv2.setNumThreads(2)
OUT=Path(__file__).resolve().parent
CUTS=[361,722,1083,1444,1805,2166,2527,2888,3240]

def at(im,xy):
 return np.concatenate([cv2.remap(im,z[:,0].reshape(-1,1).astype(np.float32),z[:,1].reshape(-1,1).astype(np.float32),cv2.INTER_LINEAR).ravel() for z in np.array_split(xy,max(1,(len(xy)+15000)//15001))])
def edge_distance(im):
 g=cv2.GaussianBlur(im,(0,0),.7);edges=np.zeros(im.shape[:2],np.uint8)
 for ch in range(3):edges|=cv2.Canny(g[:,:,ch],18,45)
 return cv2.distanceTransform((edges==0).astype(np.uint8),cv2.DIST_L2,5)
def stats(l,r):
 if len(l)==0:return {'samples':0}
 d=r-l;lb,rb=rgb2lab(l[None]/255)[0],rgb2lab(r[None]/255)[0];labd=rb-lb;med=np.median(d,axis=0);axis=int(np.argmax(abs(med)))
 return {'samples':len(l),'left_rgb_median':np.median(l,axis=0).tolist(),'right_rgb_median':np.median(r,axis=0).tolist(),'signed_rgb_mean_right_minus_left':d.mean(axis=0).tolist(),'signed_rgb_median_right_minus_left':med.tolist(),'rgb_mae':float(abs(d).mean()),'median_bias_amplitude_rgb':float(abs(med).mean()),'residual_rgb_mad_about_median':np.median(abs(d-med),axis=0).tolist(),'dominant_channel_sign_agreement':float(np.mean(np.sign(d[:,axis])==np.sign(med[axis]))) if med[axis]!=0 else None,'signed_lab_median_right_minus_left':np.median(labd,axis=0).tolist(),'median_delta_e_2000':float(np.median(deltaE_ciede2000(lb,rb)))}

summary=[]
for cut in CUTS:
 folder=OUT/str(cut)
 if (folder/'measurements.json').exists():
  summary.append(json.load(open(folder/'measurements.json')));print(cut,'cached',flush=True);continue
 a,b=[np.array(Image.open(folder/(s+'.png')).convert('RGB')) for s in ['left','right']]
 l,r,lp,rp,train=observations(a,b);lxy=lp*[1279,719];rxy=rp*[1279,719]
 # Further exclude a native5px neighborhood of contours in either original
 # drawing. Flow forward/backward visibility was already checked upstream.
 valid=(at(edge_distance(a),lxy)>5)&(at(edge_distance(b),rxy)>5)
 l,r,lp,rp,train,lxy,rxy=[x[valid] for x in [l,r,lp,rp,train,lxy,rxy]]
 np.savez_compressed(folder/'matched-flat-samples.npz',left_rgb=l,right_rgb=r,left_xy_native=lxy,right_xy_native=rxy,train=train)
 mid=(l+r)/2;d=r-l
 cv2.setRNGSeed(361);k=min(20,max(4,len(mid)//350))
 _,labs,centers=cv2.kmeans(mid.astype(np.float32),k,None,(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,80,.01),4,cv2.KMEANS_PP_CENTERS);labs=labs.ravel()
 clusters=[]
 for i in range(k):
  q=labs==i
  if q.sum()<40:continue
  st=stats(l[q],r[q]);st.update(cluster=i,mid_rgb_centroid=centers[i].tolist(),native_xy_median=np.median(lxy[q],axis=0).tolist(),native_xy_p10=np.percentile(lxy[q],10,axis=0).tolist(),native_xy_p90=np.percentile(lxy[q],90,axis=0).tolist())
  clusters.append(st)
 clusters.sort(key=lambda z:-z['samples'])
 # Same/nearly-identical color in separated scene cells: residual differences
 # can expose limits of a globally color-conditioned mapping.
 bins=np.floor(mid/8).astype(int);cells=np.column_stack([(lp[:,0]*4).astype(int).clip(0,3),(lp[:,1]*3).astype(int).clip(0,2)])
 keys=bins[:,0]*1024+bins[:,1]*32+bins[:,2];comparisons=[]
 for key in np.unique(keys):
  which=keys==key
  if which.sum()<35:continue
  groups=[]
  for cell in np.unique(cells[which],axis=0):
   q=which&np.all(cells==cell,axis=1)
   if q.sum()<12:continue
   groups.append({'cell':cell.tolist(),'n':int(q.sum()),'rgb':np.median(mid[q],axis=0),'delta':np.median(d[q],axis=0),'xy':np.median(lxy[q],axis=0)})
  for i,g in enumerate(groups):
   for h in groups[i+1:]:
    rgbdist=float(np.linalg.norm(g['rgb']-h['rgb']));posdist=float(np.linalg.norm(g['xy']-h['xy']));diff=float(np.linalg.norm(g['delta']-h['delta']))
    if rgbdist<=4 and posdist>=160 and diff>=3:
     comparisons.append({'mid_color_distance_rgb':rgbdist,'spatial_distance_px':posdist,'residual_difference_rgb_l2':diff,'regions':[{k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in x.items()} for x in [g,h]]})
 comparisons.sort(key=lambda z:-z['residual_difference_rgb_l2'])
 rec={'frame':cut,'measurement_only':True,'input_geometry_and_luts':'unchanged baseline f4ea566','correspondence_resolution':[640,360],'additional_native_contour_exclusion_px':5,'flat_match_count':len(l),'global_matched':stats(l,r),'palette_clusters':clusters,'same_color_different_location_examples':comparisons[:12]}
 (folder/'measurements.json').write_text(json.dumps(rec,indent=2));summary.append(rec)
 # Native diagnostic dots show where signed residuals were actually measured.
 heat=np.zeros((720,1280),np.float32);weight=np.zeros_like(heat);p=np.rint(lxy).astype(int)
 for (x,y),val in zip(p,np.mean(abs(d),axis=1)):heat[y,x]+=val;weight[y,x]+=1
 hblur=cv2.GaussianBlur(heat,(0,0),10);wblur=cv2.GaussianBlur(weight,(0,0),10);field=hblur/np.maximum(wblur,1e-6)
 colored=cv2.cvtColor(cv2.applyColorMap(np.clip(field/12*255,0,255).astype(np.uint8),cv2.COLORMAP_TURBO),cv2.COLOR_BGR2RGB)
 vis=(a*.32).astype(np.uint8);keep=wblur>.001;vis[keep]=colored[keep]
 canvas=Image.new('RGB',(1280,753));canvas.paste(Image.fromarray(vis),(0,33));draw=ImageDraw.Draw(canvas);draw.text((10,10),f'{cut}: matched flat-region RGB error, 0-12 byte color scale; contours/occlusions excluded. NOT a correction.',fill='white');canvas.save(folder/'flat-color-residual-map.png')
 # Compact palette evidence carries actual before/after median swatches and
 # signed RGB differences rather than magnified contrast in the drawings.
 sheet=Image.new('RGB',(940,42+len(clusters)*52),'#181818');draw=ImageDraw.Draw(sheet);draw.text((10,10),f'{cut}: protected-LUT samples. Incoming minus outgoing; RGB levels0-255.',fill='white')
 for j,z in enumerate(clusters):
  yy=42+j*52;lc=np.rint(z['left_rgb_median']).astype(int);rc=np.rint(z['right_rgb_median']).astype(int);med=z['signed_rgb_median_right_minus_left']
  draw.rectangle((10,yy,83,yy+40),fill=tuple(lc));draw.rectangle((86,yy,159,yy+40),fill=tuple(rc));draw.text((175,yy+4),f"n={z['samples']:5d}  RGBdelta={np.round(med,2).tolist()}  dE00={z['median_delta_e_2000']:.2f}",fill='white');draw.text((175,yy+24),f"median location={np.rint(z['native_xy_median']).astype(int).tolist()}  RGB={lc.tolist()} -> {rc.tolist()}",fill='#cccccc')
 sheet.save(folder/'palette-evidence.png')
 print(cut,len(l),rec['global_matched']['rgb_mae'],len(comparisons),flush=True)
(OUT/'all-joins-measurements.json').write_text(json.dumps({'cuts':summary},indent=2))
