"""Rebuild actor-removal regions from one static generated plate.
Source foreground interiors and original visible background are asserted intact.
"""
from pathlib import Path
import sys,json,cv2,numpy as np
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from PIL import Image
from seamstress.media import read_frames
cv2.setNumThreads(4)
OUT=Path(__file__).resolve().parent/'recomposed';OUT.mkdir(exist_ok=True)
CP=OUT.parent
layer=json.load(open(ROOT/'research/layer2888/v2/report.json'));reg=json.load(open(CP/'registration-report.json'));later=json.load(open(CP/'frame-2896-report.json'))
BG=np.array(layer['background_matrix_incoming_to_outgoing']);FG=np.array(layer['foreground_matrix_incoming_to_outgoing'])
M=np.array(reg['matrix_generated_to_source_full']);D=np.array(later['plate_reference_to_target_background_matrix'])
gain=np.array(layer['color']['gain']);bias=np.array(layer['color']['bias']);pc=reg['chosen_registration']['color_generated_to_source']
gen=np.array(Image.open(CP/'generated-roi.png').convert('RGB'))
srcs=read_frames(ROOT/'IYTYT.mp4',2888,9)
flame_polys=[[(429,427),(460,427),(461,450),(454,480),(450,507),(441,524),(430,495),(422,452)],[(557,428),(566,428),(578,452),(576,480),(562,525),(554,514),(559,485),(555,460)],[(661,393),(694,394),(704,418),(709,451),(705,481),(692,505),(675,500),(652,455),(653,418)],[(772,396),(788,397),(801,427),(798,451),(789,469),(782,450)]]
flame=np.zeros((720,1280),np.uint8)
for p in flame_polys:cv2.fillPoly(flame,[np.array(p,np.int32)],1)
# Preserve the actual opaque red/black nozzles and first bright flame core.
for x0,y0,x1,y1 in [(420,425,470,441),(542,426,581,443),(649,390,716,408),(770,393,807,411)]:flame[y0:y1,x0:x1]=0

def warp(im,m,mode=cv2.INTER_LINEAR):return cv2.warpAffine(im,m[:2].astype(np.float32),(1280,720),flags=mode,borderMode=cv2.BORDER_CONSTANT)
def plate_at(m):
 p=warp(gen,m,cv2.INTER_LANCZOS4).astype(np.float32)*np.array(pc['gain'])+np.array(pc['bias'])
 return np.clip(p,0,255).astype(np.float32)
reports=[]
for n in [2888,2896]:
 source=srcs[n-2888].astype(np.float32);mask=(np.array(Image.open(ROOT/f'research/layer2888/mask-{n}.png'))>127).astype(np.uint8)
 old=np.array(Image.open(ROOT/f'research/layer2888/v2/result-{n}.png').convert('RGB'))
 motion=np.eye(3) if n==2888 else D
 source_plate=plate_at(motion@M)
 # Matte opaque linework with a narrow boundary only. Colored flame alpha
 # is estimated from the source against the registered static clean plate.
 opaque=(mask>0)&(flame==0)
 dist_in=cv2.distanceTransform(opaque.astype(np.uint8),cv2.DIST_L2,5)
 dist_out=cv2.distanceTransform((~opaque).astype(np.uint8),cv2.DIST_L2,5)
 alpha=np.clip((dist_in-dist_out+.65)/1.8,0,1)
 delta=source-source_plate
 physical=np.where(delta>=0,delta/np.maximum(255-source_plate,1),-delta/np.maximum(source_plate,1)).max(axis=2).clip(0,1)
 # Orange/yellow hue and excess red over the reconstructed background limit
 # the flame estimate; no generated pixels are used as foreground.
 warm=np.maximum(source[:,:,0]-source[:,:,2],0)
 excess=np.maximum(delta[:,:,0],0)
 fc=np.clip((warm-10)/25,0,1)*np.clip((excess-2)/12,0,1)
 fa=physical*fc
 fa=cv2.GaussianBlur(fa.astype(np.float32),(0,0),.55)
 alpha[flame>0]=fa[flame>0]
 # Include uncertain flame glow in background removal, even where its
 # reconstructed foreground alpha is small.
 removal=cv2.dilate(np.maximum(mask,flame),np.ones((17,17),np.uint8))
 remove=warp(removal.astype(np.float32),BG)>.001
 plate_output=plate_at(BG@motion@M)*gain+bias
 bg=warp(source.astype(np.uint8),BG,cv2.INTER_LANCZOS4).astype(np.float64)*gain+bias
 # Blend only inside the removal guard, keeping its exterior exactly intact.
 rim=cv2.distanceTransform(remove.astype(np.uint8),cv2.DIST_L2,5)
 pa=np.minimum(rim/3,1)
 bg=bg*(1-pa[:,:,None])+plate_output*pa[:,:,None]
 # Recover premultiplied source foreground from source colors and known BG.
 # This removes source-background contamination at anti-aliased/glowing edges.
 premul=source-(1-alpha[:,:,None])*source_plate
 premul=np.minimum(np.maximum(premul,0),255*alpha[:,:,None]).astype(np.float32)
 aout=warp(alpha.astype(np.float32),FG).clip(0,1)
 pout=warp(premul,FG)*gain+aout[:,:,None]*bias
 result=np.clip(bg*(1-aout[:,:,None])+pout,0,255).astype(np.uint8)
 # Match existing original foreground interiors exactly, including the earlier
 # source resampling and grade rounding. Generated pixels never enter them.
 core=cv2.erode(opaque.astype(np.uint8),np.ones((5,5),np.uint8))
 core_out=warp(core.astype(np.float32),FG)>.999
 result[core_out]=old[core_out]
 work=remove|(aout>.001)
 result[~work]=old[~work]
 assert np.array_equal(result[~work],old[~work]);assert np.array_equal(result[core_out],old[core_out])
 Image.fromarray(result).save(OUT/f'result-{n}.png')
 Image.fromarray((alpha*255).astype(np.uint8)).save(OUT/f'alpha-source-{n}.png')
 iso=np.clip(premul+160*(1-alpha[:,:,None]),0,255).astype(np.uint8);Image.fromarray(iso).save(OUT/f'matte-isolation-{n}.png')
 change=np.any(result!=old,axis=2)
 Image.fromarray(np.concatenate([old[200:625,325:883],result[200:625,325:883]],axis=1)).save(OUT/f'comparison-{n}.png')
 row={'frame':n,'source_foreground_core_pixels':int(core_out.sum()),'changed_foreground_core_pixels':int(np.count_nonzero(change&core_out)),'work_region_pixels':int(work.sum()),'changed_pixels_outside_work_region':int(np.count_nonzero(change&~work)),'remaining_magenta_pixels':int(np.count_nonzero(np.all(result==[255,0,255],axis=2))),'removal_region_pixels':int(remove.sum()),'original_visible_background_preserved_outside_removal_and_fg':True,'status':'Native matte inspection required; not integrated'}
 reports.append(row);print(row,flush=True)
(OUT/'report.json').write_text(json.dumps({'source':'IYTYT.mp4','cleanplate_generations':1,'source_foreground_pose_blending':False,'background_map':'fixed affine (v2)','frames':reports},indent=2))
