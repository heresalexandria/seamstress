"""Experiment only: align RIFE endpoint linework before mixing its common views."""
from pathlib import Path
import sys,json,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import cv2,numpy as np
from PIL import Image,ImageDraw
from scipy.ndimage import gaussian_filter1d
from seamstress.bridge import RifeModel,progress_curve
from seamstress.camera_bridge import synthesize_camera_bridge
from seamstress.repair import flow,sample,safe_mapping
from seamstress.media import read_frames,VideoWriter
from seamstress._vendor.rife.warplayer import warp

OUT=ROOT/'research/contour-refinement-v2';OUT.mkdir(exist_ok=True)

class RefinedRife(RifeModel):
 def synthesize(self,left,right,progress):
  torch=self.torch;h,w=left.shape[:2]
  def tensor(x):return torch.from_numpy(x.copy()).permute(2,0,1)[None].float().to(self.device)/255
  pad=(0,(-w)%64,0,(-h)%64)
  a=torch.nn.functional.pad(tensor(left),pad);b=torch.nn.functional.pad(tensor(right),pad)
  pair=torch.cat([a,b],1);items=[];baselines=[left.copy()]
  def array(t):return t[0,:,:h,:w].permute(1,2,0).cpu().numpy()
  with torch.inference_mode():
   for index,t in enumerate(progress[1:-1],1):
    flows,logits,merged=self.model(pair,float(t),[16,8,4,2,1])
    f=flows[-1];wa=array(warp(a,f[:,:2]));wb=array(warp(b,f[:,2:]))
    baseline=np.clip(np.rint(array(merged[-1])*255),0,255).astype(np.uint8)
    baselines.append(baseline)
    aa=np.clip(np.rint(wa*255),0,255).astype(np.uint8);bb=np.clip(np.rint(wb*255),0,255).astype(np.uint8)
    forward,backward=flow(aa,bb),flow(bb,aa)
    efa=np.linalg.norm(forward+sample(backward,forward),axis=2)
    efb=np.linalg.norm(backward+sample(forward,backward),axis=2)
    mask=array(torch.sigmoid(logits))
    def gather(field,error,other_weight):
     confidence=np.exp(-np.square(error/1.5)).astype(np.float32)
     displacement=-field*confidence[...,None]*other_weight
     # Keep the correction local and let it vanish at untouched source anchors.
     length=np.linalg.norm(displacement,axis=2)
     displacement*=np.minimum(1.,3./np.maximum(length,.01))[...,None]
     return (displacement*float(np.sin(np.pi*float(t))**2)).astype(np.float32)
    items.append({'t':float(t),'ra':array(f[:,:2]).copy(),'rb':array(f[:,2:]).copy(),
      'mask':mask, 'da':gather(forward,efa,1-mask),'db':gather(backward,efb,mask),
      'consistent':float(np.mean((efa<1.5)&(efb<1.5)))})
    print('inference',index,'time',round(float(t),3),'consistent',items[-1]['consistent'],flush=True)
  baselines.append(right.copy())
  da=gaussian_filter1d(np.stack([x['da'] for x in items]),.8,axis=0,mode='nearest')
  db=gaussian_filter1d(np.stack([x['db'] for x in items]),.8,axis=0,mode='nearest')
  self.variants={'baseline':baselines,'refined':[left.copy()],'cubic':[left.copy()],'masksharp':[left.copy()],'refined_masksharp':[left.copy()]};self.detail=[]
  for i,item in enumerate(items):
   ma,ja,sa=safe_mapping(da[i],floor=.4);mb,jb,sb=safe_mapping(db[i],floor=.4)
   fa=ma+sample(item['ra'],ma);fb=mb+sample(item['rb'],mb)
   aligned_a=sample(left,fa,cv2.INTER_CUBIC).astype(np.float32)
   aligned_b=sample(right,fb,cv2.INTER_CUBIC).astype(np.float32)
   m=item['mask'];refined=aligned_a*m+aligned_b*(1-m)
   ca=sample(left,item['ra'],cv2.INTER_CUBIC).astype(np.float32)
   cb=sample(right,item['rb'],cv2.INTER_CUBIC).astype(np.float32)
   cubic=ca*m+cb*(1-m)
   sharp_m=m*m/np.maximum(m*m+(1-m)*(1-m),.001)
   self.variants['masksharp'].append(np.clip(np.rint(ca*sharp_m+cb*(1-sharp_m)),0,255).astype(np.uint8))
   self.variants['refined_masksharp'].append(np.clip(np.rint(aligned_a*sharp_m+aligned_b*(1-sharp_m)),0,255).astype(np.uint8))
   self.variants['refined'].append(np.clip(np.rint(refined),0,255).astype(np.uint8))
   self.variants['cubic'].append(np.clip(np.rint(cubic),0,255).astype(np.uint8))
   self.detail.append({'t':item['t'],'fb_consistent_fraction':item['consistent'],
     'minimum_jacobian':min(ja,jb),'regularized':bool(sa or sb),
     'correction_p95':float(np.percentile(np.linalg.norm(ma,axis=2),95))})
  for key in ['refined','cubic','masksharp','refined_masksharp']:self.variants[key].append(right.copy())
  return self.variants['refined']

class Replay:
 def __init__(self,frames):self.frames=frames
 def synthesize(self,left,right,progress):return [left.copy()]+self.frames[1:-1]+[right.copy()]

def sharpness(frame):
 g=cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY).astype(np.float32)
 return float(np.mean(np.abs(cv2.Laplacian(g,cv2.CV_32F))))

def main():
 start=time.monotonic();lo,hi=3228,3244
 source=list(read_frames(ROOT/'IYTYT.mp4',lo-4,hi-lo+9))
 model=RefinedRife(ROOT/'models/rife425/flownet.pkl',device='cpu')
 progress=progress_curve(hi-lo+1)
 refined,report=synthesize_camera_bridge(model,source[4],source[-5],source[:4],source[-4:],progress)
 variants={'refined':refined}
 for key in ['baseline','cubic','masksharp','refined_masksharp']:
  variants[key],_=synthesize_camera_bridge(Replay(model.variants[key]),source[4],source[-5],source[:4],source[-4:],progress)
 for key,frames in variants.items():
  np.savez_compressed(OUT/f'{key}.npz',frames=np.stack(frames))
  Image.fromarray(frames[8]).save(OUT/f'{key}-midpoint.png')
  with VideoWriter(OUT/f'{key}.mp4',1280,720,'24000/1001',crf=14,preset='fast') as writer:
   for _ in range(4):
    for frame in source[:4]+frames+source[-4:]:writer.write(frame)
 roi=(470,200,795,550)
 keys=['baseline','refined','refined_masksharp'];sheet=Image.new('RGB',(650*3,740))
 for i,key in enumerate(keys):
  crop=Image.fromarray(variants[key][8]).crop(roi).resize((650,700))
  sheet.paste(crop,(650*i,40));ImageDraw.Draw(sheet).text((650*i+12,12),key,fill='white')
 sheet.save(OUT/'midpoint-contours.jpg')
 with VideoWriter(OUT/'comparison.mp4',1920,720,'24000/1001',crf=14,preset='fast') as writer:
  for _ in range(5):
   for j in range(len(refined)):
    parts=[]
    for key in keys:
     frame=variants[key][j];crop=frame[roi[1]:roi[3],roi[0]:roi[2]]
     image=cv2.resize(crop,(640,690),interpolation=cv2.INTER_NEAREST)
     image=cv2.copyMakeBorder(image,30,0,0,0,cv2.BORDER_CONSTANT)
     cv2.putText(image,key,(12,22),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),1)
     parts.append(image)
    writer.write(np.concatenate(parts,axis=1))
 metrics={}
 for key,frames in variants.items():
  crops=[f[roi[1]:roi[3],roi[0]:roi[2]] for f in frames]
  metrics[key]={'sharpness_full':[sharpness(f) for f in frames],
    'sharpness_contours':[sharpness(f) for f in crops],
    'temporal_difference_contours':[float(np.mean(np.abs(a.astype(float)-b.astype(float)))) for a,b in zip(crops[:-1],crops[1:])]}
 report.update(runtime=time.monotonic()-start,metrics=metrics,refinement=model.detail)
 (OUT/'report.json').write_text(json.dumps(report,indent=2))
 print('DONE',OUT,'runtime',report['runtime'],flush=True)
if __name__=='__main__':main()
