from pathlib import Path
import sys,json,hashlib
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from PIL import Image,ImageDraw
import cv2,numpy as np
from seamstress.conform import conform_frame,tone_lut_at
OUT=Path(__file__).resolve().parent
PLAN=ROOT/'plans/IYTYT-eight-joins.json';p=json.load(open(PLAN));view=np.array(p['view_matrix'])
for cut in [361,722,1083,1444,1805,2166,2527,2888,3240]:
 cache=ROOT/'research'/('rigid-feasibility' if cut in [361,722,1444] else 'affine-feasibility')/str(cut)
 out=OUT/str(cut);out.mkdir(exist_ok=True);images=[]
 for idx,side,name in [(cut-1,'left','left-original.png'),(cut,'right','right-original.png')]:
  original=np.array(Image.open(cache/name).convert('RGB'));seg=next(s for s in p['segments'] if s['start']<=idx<s['end'])
  matrix=view@np.array(p['frame_matrices'][idx]);im,_,_=conform_frame(original,matrix,np.array(seg['gain']),np.array(seg['bias']),p.get('edge_extension_pixels',0))
  lut=tone_lut_at(idx,p['grade_curves'])
  if lut is not None:im=cv2.LUT(im,lut[:,None,:]).round().astype(np.uint8)
  Image.fromarray(im).save(out/f'{side}.png');images.append(im)
  (out/f'{side}-metadata.json').write_text(json.dumps({'source_frame':idx,'source_cache':str(cache/name),'source_to_output_matrix':matrix.tolist(),'protected_grade_active':lut is not None,'modified_geometry':False},indent=2))
 sheet=Image.new('RGB',(2560,750));d=ImageDraw.Draw(sheet)
 for i,(side,im) in enumerate(zip(['Outgoing protected LUT','Incoming protected LUT'],images)):
  sheet.paste(Image.fromarray(im),(i*1280,30));d.text((i*1280+12,10),f'{cut}: {side}',fill='white')
 sheet.save(out/'baseline-adjacent.png')
print('Prepared nine exact pre-encode baseline frame pairs')
(OUT/'baseline.json').write_text(json.dumps({'commit':'f4ea566','plan':str(PLAN),'plan_sha256':hashlib.sha256(PLAN.read_bytes()).hexdigest(),'source':'IYTYT.mp4','method':'Existing conform_frame geometry and protected tone_lut_at reproduced without edits; no new pixel correction or temporal rendering.'},indent=2))
