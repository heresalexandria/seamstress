"""Native diagnostic bridge candidates for the difficult street parallax seam."""
from pathlib import Path
import json,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import cv2,numpy as np
from PIL import Image,ImageDraw
from seamstress.bridge import RifeModel,progress_curve
from seamstress.camera_bridge import synthesize_camera_bridge
from seamstress.media import read_frames,VideoWriter
from seamstress.analysis import pair_metrics

def sharpness(frame):
 g=cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY).astype(np.float32)
 return float(np.mean(abs(cv2.Laplacian(g,cv2.CV_32F))))

out=ROOT/'research/1805-parallax';out.mkdir(exist_ok=True)
context_start,context_end=1793,1815
source=list(read_frames(ROOT/'IYTYT.mp4',context_start,context_end-context_start+1))
model=RifeModel(ROOT/'models/rife425/flownet.pkl',reconstruction='baseline')
variants=[('long-camera-baseline',1797,1811,True,'baseline'),
 ('long-camera-cubic',1797,1811,True,'cubic'),('long-raw-cubic',1797,1811,False,'cubic'),
 ('medium-camera-cubic',1801,1809,True,'cubic'),('medium-raw-cubic',1801,1809,False,'cubic'),
 ('short-camera-cubic',1803,1807,True,'cubic'),('short-raw-cubic',1803,1807,False,'cubic')]
reports=[];images=[('source',source)]
for name,lo,hi,camera,reconstruction in variants:
 model.reconstruction=reconstruction;left,right=source[lo-context_start],source[hi-context_start]
 progress=progress_curve(hi-lo+1);start=time.perf_counter()
 try:
  if camera:
   frames,report=synthesize_camera_bridge(model,left,right,
       source[lo-context_start-4:lo-context_start],source[hi-context_start+1:hi-context_start+5],progress)
  else:frames=model.synthesize(left,right,progress);report={}
 except Exception as exc:
  reports.append({'name':name,'error':str(exc)});print(name,'FAILED',str(exc),flush=True);continue
 candidate=source.copy();candidate[lo-context_start:hi-context_start+1]=frames
 np.savez_compressed(out/f'{name}.npz',frames=np.stack(candidate),source_indices=np.arange(context_start,context_end+1))
 for index in [1804,1805]:Image.fromarray(candidate[index-context_start]).save(out/f'{name}-{index}.png')
 small=[cv2.resize(f,(640,360),interpolation=cv2.INTER_AREA) for f in candidate]
 metrics=[pair_metrics(a,b) for a,b in zip(small[:-1],small[1:])]
 result={'name':name,'anchors':[lo,hi],'camera':camera,'reconstruction':reconstruction,
   'runtime':time.perf_counter()-start,'camera_report':report,
   'motion_curve':[m['motion_p50'] for m in metrics],
   'registered_curve':[m['registered_mae'] for m in metrics],
   'sharpness_buildings':[sharpness(f[130:560,30:450]) for f in candidate]}
 reports.append(result);images.append((name,candidate))
 with VideoWriter(out/f'{name}.mp4',2560,720,'24000/1001',crf=14,preset='fast') as writer:
  for _ in range(4):
   for a,b in zip(source,candidate):
    image=np.concatenate((a,b),axis=1).copy()
    cv2.putText(image,'ORIGINAL',(15,26),cv2.FONT_HERSHEY_SIMPLEX,.7,(255,255,255),1,cv2.LINE_AA)
    cv2.putText(image,name,(1295,26),cv2.FONT_HERSHEY_SIMPLEX,.7,(255,255,255),1,cv2.LINE_AA)
    writer.write(image)
 print(name,'runtime',round(result['runtime'],2),'camera rate',report.get('camera_rate_retention'),
       'motion',np.round(result['motion_curve'],2).tolist(),flush=True)
 (out/'report.json').write_text(json.dumps(reports,indent=2))
sheet=Image.new('RGB',(640*4,384*len(images)),'#141821');draw=ImageDraw.Draw(sheet)
for row,(name,frames) in enumerate(images):
 for col,index in enumerate([1804,1805]):
  frame=frames[index-context_start]
  small=Image.fromarray(frame).resize((640,360),Image.Resampling.LANCZOS)
  sheet.paste(small,(col*640,row*384));draw.text((col*640+6,row*384+365),f'{name} {index}',fill='white')
  crop=Image.fromarray(frame).crop((40,120,360,300)).resize((640,360),Image.Resampling.NEAREST)
  sheet.paste(crop,((col+2)*640,row*384));draw.text(((col+2)*640+6,row*384+365),'near buildings 2x',fill='white')
sheet.save(out/'contact.jpg',quality=95)
