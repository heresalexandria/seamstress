"""Bounded experiment: propagate only masked generated contour repairs in time."""
from pathlib import Path
import sys,time,json
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import cv2,numpy as np
from PIL import Image,ImageDraw
from seamstress.bridge import RifeModel,progress_curve
from seamstress.camera_bridge import synthesize_camera_bridge,_transform
from seamstress.media import read_frames,VideoWriter
from seamstress.repair import sample,smoothstep

OUT=ROOT/'research/anchor-refinement';OUT.mkdir(exist_ok=True)

class Capture:
 def synthesize(self,left,right,progress):
  self.left=left.copy();self.right=right.copy()
  return [left]+[left for _ in progress[1:-1]]+[right]

class Replay:
 def __init__(self,frames):self.frames=frames
 def synthesize(self,left,right,progress):return [left]+self.frames[1:-1]+[right]

def tensors(model,left,right):
 torch=model.torch;h,w=left.shape[:2]
 def tensor(x):return torch.from_numpy(x.copy()).permute(2,0,1)[None].float().to(model.device)/255
 pad=(0,(-w)%64,0,(-h)%64)
 a=torch.nn.functional.pad(tensor(left),pad);b=torch.nn.functional.pad(tensor(right),pad)
 return torch.cat((a,b),1)

def contour_mask():
 m=np.zeros((384,384),np.uint8)
 # Corridors around the existing silhouette, not entire character heads/bodies.
 hair=np.array([(258,62),(268,59),(278,61),(285,68),(289,84),(291,105),(291,121),(286,135),(276,143),(266,137),(257,131),(252,117),(249,102),(251,83),(254,69)],np.int32)
 sloth=np.array([(91,170),(89,152),(91,136),(95,123),(104,116),(114,114),(126,117),(134,125),(138,139),(137,155),(132,170)],np.int32)
 cv2.polylines(m,[hair],True,255,8,cv2.LINE_AA)
 cv2.polylines(m,[sloth],False,255,8,cv2.LINE_AA)
 for center,axes in [((234,216),(10,17)),((305,210),(9,14)),((66,278),(12,16)),((154,277),(11,16))]:
  cv2.ellipse(m,center,axes,0,0,360,255,-1,cv2.LINE_AA)
 m=cv2.GaussianBlur(m.astype(np.float32)/255,(9,9),1.3)
 m[m<.015]=0
 return m

def main():
 started=time.monotonic();lo,hi=3228,3244;span=hi-lo;mid=span//2
 source=list(read_frames(ROOT/'IYTYT.mp4',lo-4,span+9));left,right=source[4],source[-5]
 progress=progress_curve(span+1);capture=Capture()
 _,camera=synthesize_camera_bridge(capture,left,right,source[:4],source[-4:],progress)
 h,w=left.shape[:2];ch,cw=capture.left.shape[:2];center=np.array([(w-1)/2,(h-1)/2])
 # The source left endpoint is translated by an integer offset on common canvas.
 x0,y0,tw,th=400,250,200,160
 scores=cv2.matchTemplate(capture.left,left[y0:y0+th,x0:x0+tw],cv2.TM_CCOEFF_NORMED)
 _,maximum,_,loc=cv2.minMaxLoc(scores)
 offset=np.eye(3);offset[:2,2]=[loc[0]-x0,loc[1]-y0]
 assert maximum>.999,'Could not recover exact common-view offset'
 camera_mid=_transform(np.array(camera['camera_parameters'][mid-1]),center)@np.linalg.inv(offset)
 native_to_common=np.linalg.inv(camera_mid)
 model=RifeModel(ROOT/'models/rife425/flownet.pkl',device='cpu')
 baseline_common=model.synthesize(capture.left,capture.right,progress)
 baseline_native,_=synthesize_camera_bridge(Replay(baseline_common),left,right,source[:4],source[-4:],progress)
 print('baseline complete',time.monotonic()-started,flush=True)
 mask_native=np.zeros((h,w),np.float32);mask_native[192:576,448:832]=contour_mask()
 generated=np.array(Image.open(ROOT/'research/contour-imagegen/generated-native-roi.png').convert('RGB'))
 generated_full=baseline_native[mid].copy();generated_full[192:576,448:832]=generated
 native_delta=(generated_full.astype(np.float32)-baseline_native[mid])*mask_native[...,None]
 gain=np.array(camera['color_gain'],np.float32);bias=np.array(camera['color_bias'],np.float32)
 grade_factor=.5+.5/gain
 common_delta=cv2.warpAffine(native_delta,native_to_common[:2].astype(np.float32),(cw,ch),flags=cv2.INTER_CUBIC)/grade_factor
 common_mask=cv2.warpAffine(mask_native,native_to_common[:2].astype(np.float32),(cw,ch),flags=cv2.INTER_LINEAR)
 anchor=np.clip(np.rint(baseline_common[mid].astype(np.float32)+common_delta),0,255).astype(np.uint8)
 # Only this masked contour anchor is a generative asset. No new generation.
 Image.fromarray((mask_native*255).round().astype(np.uint8)).save(OUT/'native-edit-mask.png')
 Image.fromarray(anchor).save(OUT/'common-midpoint-anchor.png')
 output_common=list(baseline_common);masks_common=[np.zeros((ch,cw),np.float32) for _ in progress]
 output_common[mid]=anchor;masks_common[mid]=common_mask
 torch=model.torch
 for side,a,b,first in [('left',capture.left,anchor,0),('right',anchor,capture.right,mid)]:
  pair=tensors(model,a,b)
  with torch.inference_mode():
   for j in range(1,mid):
    t=j/mid;flows,_,merged=model.model(pair,t,[16,8,4,2,1])
    frame=merged[-1][0,:,:ch,:cw].permute(1,2,0).clamp(0,1).cpu().numpy()*255
    selected=flows[-1][:,2:4] if side=='left' else flows[-1][:,:2]
    motion=selected[0,:,:ch,:cw].permute(1,2,0).cpu().numpy()
    transported=np.clip(sample(common_mask,motion),0,1)
    envelope=float(smoothstep(t if side=='left' else 1-t))
    weight=transported*envelope
    index=first+j;base=baseline_common[index].astype(np.float32)
    output_common[index]=np.clip(np.rint(base+(frame-base)*weight[...,None]),0,255).astype(np.uint8)
    masks_common[index]=weight
    print(side,j,'changed area',float(np.mean(weight>.015)),flush=True)
 refined_native,_=synthesize_camera_bridge(Replay(output_common),left,right,source[:4],source[-4:],progress)
 # Reapply the original native baseline outside transported mask support, making
 # the scope invariant explicit even through two coordinate resamplings.
 masks_native=[]
 for i in range(len(progress)):
  if i in (0,span):mask=np.zeros((h,w),np.float32)
  else:
   transform=_transform(np.array(camera['camera_parameters'][i-1]),center)@np.linalg.inv(offset)
   mask=cv2.warpAffine(masks_common[i],transform[:2].astype(np.float32),(w,h),flags=cv2.INTER_CUBIC)
  masks_native.append(np.clip(mask,0,1))
  support=cv2.dilate((mask>.005).astype(np.uint8),np.ones((5,5),np.uint8))>0
  refined_native[i][~support]=baseline_native[i][~support]
  assert np.array_equal(refined_native[i][~support],baseline_native[i][~support])
 assert np.array_equal(refined_native[0],left) and np.array_equal(refined_native[-1],right)
 np.savez_compressed(OUT/'result.npz',baseline=np.stack(baseline_native),refined=np.stack(refined_native),native_masks=np.stack(masks_native))
 for name,frames in [('baseline',baseline_native),('refined',refined_native)]:
  Image.fromarray(frames[mid]).save(OUT/f'{name}-midpoint.png')
  with VideoWriter(OUT/f'{name}.mp4',w,h,'24000/1001',crf=14,preset='fast') as writer:
   for _ in range(4):
    for frame in source[:4]+frames+source[-4:]:writer.write(frame)
 with VideoWriter(OUT/'comparison.mp4',w*2,h,'24000/1001',crf=14,preset='fast') as writer:
  for _ in range(5):
   for a,b in zip(source[:4]+baseline_native+source[-4:],source[:4]+refined_native+source[-4:]):writer.write(np.concatenate((a,b),1))
 sheet=Image.new('RGB',(768,414))
 for i,(name,frames) in enumerate([('Baseline',baseline_native),('Masked midpoint anchor',refined_native)]):
  sheet.paste(Image.fromarray(frames[mid]).crop((448,192,832,576)),(384*i,30));ImageDraw.Draw(sheet).text((384*i+8,10),name,fill='white')
 sheet.save(OUT/'midpoint-comparison.png')
 samples=[2,5,7,8,9,11,14];temporal=Image.new('RGB',(384*len(samples),414*2))
 for row,(name,frames) in enumerate([('Baseline',baseline_native),('Refined',refined_native)]):
  for col,j in enumerate(samples):
   temporal.paste(Image.fromarray(frames[j]).crop((448,192,832,576)),(384*col,414*row+30));ImageDraw.Draw(temporal).text((384*col+6,414*row+9),f'{name} {lo+j}',fill='white')
 temporal.save(OUT/'temporal-contours.jpg')
 def temporal_curve(frames):
  return [float(np.mean(np.abs(a[192:576,448:832].astype(float)-b[192:576,448:832].astype(float)))) for a,b in zip(frames[:-1],frames[1:])]
 def sharpness(frame):
  g=cv2.cvtColor(frame[192:576,448:832],cv2.COLOR_RGB2GRAY).astype(np.float32)
  return float(np.mean(np.abs(cv2.Laplacian(g,cv2.CV_32F))))
 delta=[b.astype(float)-a.astype(float) for a,b in zip(baseline_native,refined_native)]
 report={'anchors':[lo,lo+mid,hi],'native_resolution':[w,h],'camera_parameters':camera['camera_parameters'],
  'source_endpoints_exact':True,'outside_mask_baseline_exact':True,'mask_changed_fraction':[float(np.mean(m>.015)) for m in masks_native],
  'baseline_contour_temporal':temporal_curve(baseline_native),'refined_contour_temporal':temporal_curve(refined_native),
  'baseline_contour_sharpness':[sharpness(f) for f in baseline_native],'refined_contour_sharpness':[sharpness(f) for f in refined_native],
  'edit_difference_mean':[float(np.mean(abs(d))) for d in delta],
  'edit_difference_acceleration':[float(np.mean(abs(delta[i+1]-2*delta[i]+delta[i-1]))) for i in range(1,span)],
  'offset':offset.tolist(),'offset_match_score':maximum,'runtime_seconds':time.monotonic()-started,'production_integrated':False,'generation_calls':0}
 (OUT/'report.json').write_text(json.dumps(report,indent=2))
 print('DONE',report['runtime_seconds'],flush=True)
if __name__=='__main__':main()
