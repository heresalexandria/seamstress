"""One bounded native two-second layer proof, including entry/exit source frames."""
from pathlib import Path
import sys,json,cv2,numpy as np,subprocess
from scipy.linalg import logm,expm
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from PIL import Image
from seamstress.media import VideoWriter
cv2.setNumThreads(4)
CP=Path(__file__).resolve().parent;OUT=CP/'proof'
src=np.load(OUT/'source-2864-2911.npy',mmap_mode='r')
tr={x['frame']:x for x in json.load(open(OUT/'tracking.json'))['frames']}
cam=json.load(open(ROOT/'research/layer2888/camera-ease.json'));cams={x['frame']:np.array(x['background_matrix_source_to_output']) for x in cam['frames']}
layer=json.load(open(ROOT/'research/layer2888/v2/report.json'));reg=json.load(open(CP/'registration-report.json'))
FG=np.array(layer['foreground_matrix_incoming_to_outgoing']);fglog=logm(FG).real
M=np.array(reg['matrix_generated_to_source_full']);gain=np.array(layer['color']['gain']);bias=np.array(layer['color']['bias']);pc=reg['chosen_registration']['color_generated_to_source']
gen=np.array(Image.open(CP/'generated-roi.png').convert('RGB'))
Z=np.array([[1.06,0,-.06*639.5],[0,1.06,-.06*359.5],[0,0,1.]])
ones=np.ones((720,1280),np.float32);genones=np.ones(gen.shape[:2],np.float32)
def warp(im,m,mode=cv2.INTER_LINEAR):return cv2.warpAffine(im,m[:2].astype(np.float32),(1280,720),flags=mode,borderMode=cv2.BORDER_CONSTANT)
def plate_at(m):
 p=warp(gen,m,cv2.INTER_LANCZOS4).astype(float)*np.array(pc['gain'])+np.array(pc['bias']);return np.clip(p,0,255)
def quint(x):x=np.clip(x,0,1);return x*x*x*(10+x*(-15+6*x))
reports=[];previous=None;prev_original=None
with VideoWriter(OUT/'candidate-silent.mp4',1280,720,'24000/1001',crf=12,preset='fast') as writer,VideoWriter(OUT/'comparison-silent.mp4',2560,748,'24000/1001',crf=14,preset='fast') as compare:
 for n in range(2864,2912):
  source=src[n-2864].astype(np.float32);original=warp(source.astype(np.uint8),Z,cv2.INTER_LANCZOS4)
  weight=1.-quint((n-2900)/10) if n>=2900 else 1.
  # Two exact source frames terminate the experiment. All camera, layer, and
  # local grade corrections reach identity with zero derivative beforehand.
  active=n>=2882 and n<2910
  r={'frame':n,'active':active,'return_weight':float(weight),'uncovered_pixels':0,'changed_original_foreground_core_pixels':0}
  if not active:
   result=original.copy()
  else:
   masks=np.load(OUT/f'mask-{n}.npz');mask=masks['foreground'];flame=masks['flame'];opaque=(mask>0)&(flame==0)
   D=np.array(tr[n]['background_reference_to_current_matrix'])
   bgm=cams[n];fgm=np.eye(3) if n<2888 else FG
   if n>=2900:bgm=expm(weight*logm(bgm).real).real;fgm=expm(weight*fglog).real
   BG=Z@bgm;F=Z@fgm
   outgain=np.ones(3) if n<2888 else 1+weight*(gain-1)
   outbias=np.zeros(3) if n<2888 else weight*bias
   psource=plate_at(D@M)
   if n<2888:psource=psource*gain+bias
   din=cv2.distanceTransform(opaque.astype(np.uint8),cv2.DIST_L2,5);dout=cv2.distanceTransform((~opaque).astype(np.uint8),cv2.DIST_L2,5)
   alpha=np.clip((din-dout+.65)/1.8,0,1)
   delta=source-psource
   physical=np.where(delta>=0,delta/np.maximum(255-psource,1),-delta/np.maximum(psource,1)).max(axis=2).clip(0,1)
   warm=np.maximum(source[:,:,0]-source[:,:,2],0);excess=np.maximum(delta[:,:,0],0)
   fc=np.clip((warm-10)/25,0,1)*np.clip((excess-2)/12,0,1)
   fa=cv2.GaussianBlur((physical*fc).astype(np.float32),(0,0),.55);alpha[flame>0]=fa[flame>0]
   removal=cv2.dilate(np.maximum(mask,flame),np.ones((7,7),np.uint8))
   remove=warp(removal.astype(np.float32),BG)>.001
   aout=warp(alpha.astype(np.float32),F).clip(0,1)
   sourcebg=warp(source.astype(np.uint8),BG,cv2.INTER_LANCZOS4).astype(float)*outgain+outbias
   pout=plate_at(BG@D@M)
   if n<2888:pout=pout*gain+bias
   pout=pout*outgain+outbias
   rim=cv2.distanceTransform(remove.astype(np.uint8),cv2.DIST_L2,5);pa=np.minimum(rim/2,1)
   # Only the speculative outer guard fades on the return. Actual transformed
   # source-subject coverage still gets a clean background, not a ghost pose.
   if n>=2900:
    subject=warp((alpha>.001).astype(np.float32),BG).clip(0,1)
    pa=np.maximum(subject,pa*weight)
   bg=sourcebg*(1-pa[:,:,None])+pout*pa[:,:,None]
   premul=source-(1-alpha[:,:,None])*psource;premul=np.minimum(np.maximum(premul,0),255*alpha[:,:,None]).astype(np.float32)
   foreground=warp(premul,F)*outgain+aout[:,:,None]*outbias
   result=np.clip(bg*(1-aout[:,:,None])+foreground,0,255).astype(np.uint8)
   core=cv2.erode(opaque.astype(np.uint8),np.ones((5,5),np.uint8));coreout=warp(core.astype(np.float32),F)>.999
   source_foreground=np.clip(warp(source.astype(np.uint8),F,cv2.INTER_LANCZOS4).astype(float)*outgain+outbias,0,255).astype(np.uint8)
   result[coreout]=source_foreground[coreout]
   cv=warp(ones,BG)>.999;pv=warp(genones,BG@D@M)>.999
   uncovered=((~cv)&(pa<.999)|(~pv)&(pa>.001))&(aout<.999)
   result[uncovered]=[255,0,255]
   preserved=(~remove)&(aout<.001)&cv
   # Outside every layer/removal support, retain source BG warp exactly.
   result[preserved]=np.clip(sourcebg[preserved],0,255).astype(np.uint8)
   r.update(uncovered_pixels=int(uncovered.sum()),original_foreground_core_pixels=int(coreout.sum()),changed_original_foreground_core_pixels=int(np.count_nonzero(np.any(result!=source_foreground,axis=2)&coreout)),modified_background_support_pixels=int(np.count_nonzero((pa>.001)&(aout<.999))))
   ring=(cv2.dilate(remove.astype(np.uint8),np.ones((9,9),np.uint8))>0)&~remove&cv&pv&(aout<.001)
   err=np.mean(np.abs(sourcebg-pout),axis=2)
   if ring.any():r['plate_to_visible_bg_boundary_mae']=float(err[ring].mean());r['plate_to_visible_bg_boundary_p90']=float(np.percentile(err[ring],90))
  if previous is not None:
   r['mean_rgb_change']=float(np.abs(result.astype(float)-previous).mean());r['original_mean_rgb_change']=float(np.abs(original.astype(float)-prev_original).mean())
  if n in [2864,2881,2882,2883,2884,2885,2886,2887,2888,2889,2890,2891,2892,2893,2894,2896,2900,2904,2907,2908,2909,2910,2911]:Image.fromarray(result).save(OUT/f'frame-{n}.png')
  writer.write(result)
  strip=np.zeros((748,2560,3),np.uint8);strip[28:,:1280]=original;strip[28:,1280:]=result
  cv2.putText(strip,'Original (same 1.06 view)',(12,20),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),1);cv2.putText(strip,'Experimental layers / magenta = uncovered',(1292,20),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),1)
  compare.write(strip);reports.append(r);previous=result;prev_original=original
  print(n,r,flush=True)
for name in ['candidate','comparison']:
 subprocess.run(['ffmpeg','-v','error','-nostdin','-i',str(OUT/f'{name}-silent.mp4'),'-ss',str(2864*1001/24000),'-t',str(48*1001/24000),'-i',str(ROOT/'IYTYT.mp4'),'-map','0:v:0','-map','1:a:0','-c:v','copy','-c:a','aac','-b:a','192k','-movflags','+faststart','-shortest',str(OUT/f'{name}.mp4')],check=True)
report={'source_window':[2864,2912],'frame_count':48,'view_matrix':Z.tolist(),'background_ease_file':'research/layer2888/camera-ease.json','background_ease_half_window':cam['half_window_frames'],'foreground':'one current original source drawing per frame, independent transform; no pose synthesis or crossfade','cleanplate_generation_calls':1,'return_to_source':{'neutralization_frames':[2900,2910],'exact_source_frames':[2910,2911],'curve':'quintic zero endpoint velocity; geometry and local grade return toidentity'},'frames':reports,'status':'Experimental proof. Requires native frame and playback review; not integrated.'}
(OUT/'render-report.json').write_text(json.dumps(report,indent=2))
