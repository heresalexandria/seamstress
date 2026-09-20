"""Reproduce the first join proof without synthesizing or blending drawings."""
import json
import subprocess
from pathlib import Path
import cv2
import numpy as np
from seamstress.media import read_frames, VideoWriter
from seamstress.conform import conform_frame

out=Path('output/source-proof');out.mkdir(parents=True,exist_ok=True)
start,end,cut=289,433,361
camera=json.loads(Path('research/rigid-continuity/original-fixed-affine-24-transforms.json').read_text())
grade=json.loads(Path('research/segment-color/estimates.json').read_text())[0]
matrices={r['frame']:np.array(r['source_to_output_matrix']) for r in camera['frames']}
source=read_frames('IYTYT.mp4',start,end-start)
with VideoWriter(out/'candidate-silent.mp4',1280,720,'24000/1001',crf=14) as cand, VideoWriter(out/'comparison-silent.mp4',2560,748,'24000/1001',crf=14) as compare:
 for i,frame in enumerate(source,start):
  gain=np.array(grade['gain']) if i>=cut else np.ones(3)
  bias=np.array(grade['bias']) if i>=cut else np.zeros(3)
  fixed,_,_=conform_frame(frame,matrices[i],gain,bias)
  cand.write(fixed)
  strip=np.zeros((748,2560,3),np.uint8)
  strip[28:,:1280]=frame;strip[28:,1280:]=fixed
  cv2.putText(strip,'Original',(12,20),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),1)
  cv2.putText(strip,'Original drawings: fixed framing + grade',(1292,20),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),1)
  compare.write(strip)
  if i in (cut-1,cut):cv2.imwrite(str(out/f'{i}.jpg'),cv2.cvtColor(strip,cv2.COLOR_RGB2BGR))
for name in ('candidate','comparison'):
 subprocess.run(['ffmpeg','-v','error','-nostdin','-i',str(out/f'{name}-silent.mp4'),'-ss',str(start*1001/24000),'-t',str((end-start)*1001/24000),'-i','IYTYT.mp4','-map','0:v:0','-map','1:a:0','-c:v','copy','-c:a','aac','-b:a','192k','-movflags','+faststart','-shortest',str(out/f'first-join-{name}.mp4')],check=True)
report={'source_window':[start,end],'join_frame':cut,'synthesized_frames':0,'blended_frames':0,'one_original_frame_per_output_frame':True,'geometry':'Fixed affine correction persists through incoming clip, 24-frame camera-rate easing','color_gain':grade['gain'],'color_bias':grade['bias'],'source_frame_matrices':str(Path('research/rigid-continuity/original-fixed-affine-24-transforms.json')),'constant_safety_crop_fraction':camera['crop_fraction_each_dimension'],'status':'Short perceptual test, not certified seamless'}
(out/'report.json').write_text(json.dumps(report,indent=2))
print('Source drawing proof ready',flush=True)
