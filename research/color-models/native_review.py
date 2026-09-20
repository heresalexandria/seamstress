"""Native still review only; no change to output video or production plan."""
import json
from global_models import *

reports=json.loads((OUT/'results.json').read_text())['reports']
for cut in [361,2166,2527,2888]:
    report=next(r for r in reports if r['cut']==cut)
    model=report['models']['protected_quadratic']['parameters']
    base=list(read_frames(SOURCE,cut-1,2))
    adjusted=[np.rint(np.clip(apply(f,model[side]),0,255)).astype(np.uint8) for f,side in zip(base,['left','right'])]
    sheet=Image.new('RGB',(2560,1488),'#111');draw=ImageDraw.Draw(sheet)
    for row,frames in enumerate([base,adjusted]):
        for col,frame in enumerate(frames):sheet.paste(Image.fromarray(frame),(col*1280,row*744+24))
        draw.text((10,row*744+5),'ACCEPTED ENCODED BASELINE' if row==0 else 'PROTECTED QUADRATIC COLOR ONLY — RESEARCH, NOT ACCEPTED',fill='white')
    sheet.save(OUT/f'native-{cut}.jpg',quality=96)
    print(cut,flush=True)
