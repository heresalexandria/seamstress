"""Discrete original-drawing hold repair experiment; no production edits."""
from pathlib import Path
import json,sys,subprocess
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import cv2,numpy as np
from PIL import Image,ImageDraw
from seamstress.media import read_frames,VideoWriter
OUT=Path(__file__).parent;CUT=2527;START=CUT-24;COUNT=120


def picture_change(a,b):
 ga,gb=[cv2.GaussianBlur(cv2.cvtColor(f,cv2.COLOR_RGB2GRAY).astype(np.float32),(5,5),1.1) for f in [a,b]]
 d=abs(ga-gb)
 return {'mean':float(d.mean()),'p99':float(np.percentile(d,99)),'fraction_over5':float(np.mean(d>5)),'held_drawing':bool(d.mean()<.55 and np.percentile(d,99)<5)}


def held_runs(frames,indices):
 runs=[];start=0
 for i in range(1,len(frames)):
  if not picture_change(frames[i-1],frames[i])['held_drawing']:
   runs.append({'start':int(indices[start]),'end':int(indices[i-1]),'length':i-start});start=i
 runs.append({'start':int(indices[start]),'end':int(indices[-1]),'length':len(frames)-start})
 return runs


def source_index(output):
 if output<CUT:return output
 u=np.clip((output-(CUT+8))/72,0,1);smooth=u*u*(3-2*u)
 return int(np.floor(output+6*(1-smooth)+.5))


def main():
 proxy=list(read_frames(ROOT/'IYTYT.mp4',START,COUNT,size=(640,360)))
 indices=np.arange(START,START+COUNT);mapping=np.array([source_index(n) for n in indices]);retimed=[proxy[n-START] for n in mapping]
 assert len(mapping)==COUNT and np.all(np.diff(mapping)>=0) and mapping[-1]==indices[-1]
 repeat_outputs=[int(indices[i]) for i in range(1,COUNT) if mapping[i]==mapping[i-1]]
 skipped=[int(n) for n in sorted(set(indices)-set(mapping))];assert len(skipped)==6 and len(repeat_outputs)==6
 original_runs=held_runs(proxy,indices);candidate_runs=held_runs(retimed,indices)
 report={'source':'IYTYT.mp4','seam':CUT,'output_start':START,'output_count':COUNT,'duration_seconds':COUNT*1001/24000,
  'strategy':'Omit6 redundant startup pictures, then repay6 exact drawing repetitions using a smooth3-second reduction of source-time offset; no new pictures or blending.',
  'maximum_picture_advance_frames':int((mapping-indices).max()),'maximum_picture_advance_seconds':float((mapping-indices).max()*1001/24000),
  'audio':'Unmodified original timeline, source timing for output window; mouth/action drawings temporarily advance by up to250.25ms.',
  'recovery_interval':[CUT+8,CUT+80],'skipped_source_frames':skipped,'repeated_output_frames':repeat_outputs,
  'source_mapping':[{'output_frame':int(n),'source_frame':int(m)} for n,m in zip(indices,mapping)],
  'hold_detection':'640px luma,Gaussian sigma1.1; mean change<0.55 and99th-percentile change<5, allowing compression noise; no geometry compensation.',
  'original_drawing_runs':original_runs,'retimed_drawing_runs':candidate_runs,
  'first20_step_metrics_original':[{'frame':int(n),**picture_change(proxy[n-1-START],proxy[n-START])} for n in range(CUT+1,CUT+21)],
  'first20_step_metrics_retimed':[{'frame':int(n),**picture_change(retimed[n-1-START],retimed[n-START])} for n in range(CUT+1,CUT+21)],
  'outside_window':'Identity mapping; duration and audio never change.',
  'not_addressed':['Camera motion discontinuity','Redrawn furniture/body/background','Grade/texture differences','Speech or musical gesture synchronization']}
 (OUT/'source-map.json').write_text(json.dumps(report,indent=2))
 print('Original first runs',[r for r in original_runs if CUT<=r['start']<CUT+30],flush=True)
 print('Retimed first runs',[r for r in candidate_runs if CUT<=r['start']<CUT+90],flush=True)
 print('Skipped',skipped,'Repeated outputs',repeat_outputs,flush=True)
 # Compare full-image versus subject regions before the cut to distinguish
 # source camera movement from articulated pose change.
 region_metrics=[]
 for n in range(CUT-12,CUT+21):
  a,b=proxy[n-1-START],proxy[n-START]
  region_metrics.append({'frame':n,'whole':picture_change(a,b),'woman':picture_change(a[20:340,460:635],b[20:340,460:635]),'sloth':picture_change(a[85:285,120:290],b[85:285,120:290]),'furniture':picture_change(a[200:350,30:430],b[200:350,30:430])})
 (OUT/'region-cadence.json').write_text(json.dumps(region_metrics,indent=2))
 # Native proof contains only direct index selections, byte-identical to the
 # decoded originals before encoding. No resize or transform of the proof.
 source=list(read_frames(ROOT/'IYTYT.mp4',START,COUNT))
 with VideoWriter(OUT/'retimed.mp4',1280,720,'24000/1001',crf=14,preset='fast') as w:
  for n in mapping:w.write(source[n-START])
 with VideoWriter(OUT/'comparison.mp4',2560,720,'24000/1001',crf=14,preset='fast') as w:
  for i,n in enumerate(mapping):
   f=np.concatenate([source[i],source[n-START]],axis=1)
   cv2.putText(f,'ORIGINAL',(12,26),cv2.FONT_HERSHEY_SIMPLEX,.7,(255,255,255),1,cv2.LINE_AA)
   cv2.putText(f,f'EXACT DRAWING RETIME: source {n}',(1292,26),cv2.FONT_HERSHEY_SIMPLEX,.7,(255,255,255),1,cv2.LINE_AA);w.write(f)
 for name in ['retimed','comparison']:
  subprocess.run(['ffmpeg','-v','error','-nostdin','-y','-i',str(OUT/f'{name}.mp4'),'-ss',str(START*1001/24000),'-i',str(ROOT/'IYTYT.mp4'),'-map','0:v:0','-map','1:a:0','-t',str(COUNT*1001/24000),'-c:v','copy','-c:a','aac','-b:a','192k','-movflags','+faststart',str(OUT/f'{name}-audio.mp4')],check=True)
 sheet=Image.new('RGB',(1280,384*4),'#111');draw=ImageDraw.Draw(sheet)
 for row,n in enumerate([CUT,CUT+2,CUT+6,CUT+10]):
  for col,(label,index) in enumerate([('original',n),('retimed',source_index(n))]):
   sheet.paste(Image.fromarray(source[index-START]).resize((640,360)),(col*640,row*384));draw.text((col*640+8,row*384+366),f'output{n} {label} source{index}',fill='white')
 sheet.save(OUT/'startup-contact.jpg',quality=95)
 print('DONE native5-second proof, same duration/audio',flush=True)

if __name__=='__main__':main()
