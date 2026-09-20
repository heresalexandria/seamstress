"""Two native proof windows for balanced camera-rate correction, no full render."""
from pathlib import Path
import sys,json,subprocess
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(Path(__file__).parent))
import cv2,numpy as np
from PIL import Image,ImageDraw
from seamstress.media import read_frames,VideoWriter
from first_join import matrix
from all_joins import crop_needed
from audit import track
OUT=Path(__file__).parent/'all-joins'
recipe=json.load(open(OUT/'neutral168_balanced12-matrices.json'));matrices=np.array(recipe['frame_matrices'])
reports=[]
for cut in [361,2166]:
 start,count=cut-48,96;frames=list(read_frames(ROOT/'IYTYT.mp4',start,count))
 # Constant per-proof viewing crop; compare against identically cropped source.
 # This is reported separately from the all-movie crop bound.
 zoom_factor=crop_needed(matrices[start:start+count]);view=matrix(np.array([np.log(zoom_factor),0,0,0]))
 output=[];baseline=[]
 for n,f in enumerate(frames,start):
  output.append(cv2.warpAffine(f,(view@matrices[n])[:2],(1280,720),flags=cv2.INTER_CUBIC,borderMode=cv2.BORDER_CONSTANT))
  baseline.append(cv2.warpAffine(f,view[:2],(1280,720),flags=cv2.INTER_CUBIC,borderMode=cv2.BORDER_CONSTANT))
 video=OUT/f'balanced-{cut}.mp4'
 with VideoWriter(video,1280,720,'24000/1001',crf=14,preset='fast') as writer:
  for f in output:writer.write(f)
 with VideoWriter(OUT/f'balanced-{cut}-comparison.mp4',2560,720,'24000/1001',crf=14,preset='fast') as writer:
  for a,b in zip(baseline,output):
   f=np.concatenate([a,b],axis=1);cv2.putText(f,'ORIGINAL (same viewing crop)',(12,26),cv2.FONT_HERSHEY_SIMPLEX,.7,(255,255,255),1,cv2.LINE_AA);cv2.putText(f,'ORIGINAL PIXELS / slow affine + balanced camera',(1292,26),cv2.FONT_HERSHEY_SIMPLEX,.7,(255,255,255),1,cv2.LINE_AA);writer.write(f)
 audio=OUT/f'balanced-{cut}-audio.mp4'
 subprocess.run(['ffmpeg','-v','error','-nostdin','-y','-i',str(video),'-ss',str(start*1001/24000),'-i',str(ROOT/'IYTYT.mp4'),'-map','0:v:0','-map','1:a:0','-t',str(count*1001/24000),'-c:v','copy','-c:a','aac','-b:a','192k','-movflags','+faststart',str(audio)],check=True)
 # Independent fresh-encoded affine tracking across the cut and camera support.
 encoded=list(read_frames(video,0,count,size=(640,360)));original_small=[cv2.resize(f,(640,360),interpolation=cv2.INTER_AREA) for f in baseline]
 curves={}
 for name,version in [('original',original_small),('balanced',encoded)]:
  curves[name]=[{'frame':start+i,**track(version[i-2],version[i],2)} for i in range(2,count)]
 report={'cut':cut,'source_start':start,'frame_count':count,'preview_constant_enlargement':zoom_factor,'preview_crop_fraction_total_dimension':1-1/zoom_factor,'original_frames_only':True,'color_correction':False,'curves':curves}
 reports.append(report)
 sheet=Image.new('RGB',(1280,388*4),'#111');draw=ImageDraw.Draw(sheet)
 for row,n in enumerate([cut-1,cut,cut+6,cut+24]):
  for col,version in enumerate([baseline,output]):
   sheet.paste(Image.fromarray(version[n-start]).resize((640,360)),(col*640,row*388));draw.text((col*640+8,row*388+366),f'{"original" if col==0 else "balanced source geometry"} {n}',fill='white')
 sheet.save(OUT/f'balanced-{cut}-contact.jpg',quality=94)
 print(cut,'view zoom',zoom_factor,'cut camera',next(r for r in curves['balanced'] if r['frame']==cut),'saved',audio,flush=True)
 (OUT/'balanced-proofs.json').write_text(json.dumps(reports,indent=2))
