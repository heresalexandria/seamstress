"""Native frame comparison sheets for color-only research models."""
import json,sys
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from seamstress.local_color import apply_image
OUT=ROOT/'research/local-color-strong'
for cut in [361,722,2527,3240]:
    base=[np.array(Image.open(ROOT/'research/local-color-diagnosis'/str(cut)/f'{side}.png')) for side in ['left','right']]
    rows=[('Checkpoint',base)]
    for label,folder in [('Conservative local color','local-color-fit'),('Stronger local color','local-color-strong')]:
        m=json.loads((ROOT/'research'/folder/f'{cut}-model.json').read_text())
        corrected=[apply_image(im,m[side]) for im,side in zip(base,['left','right'])]
        rows.append((label,corrected))
        if folder=='local-color-strong':
            for im,side in zip(corrected,['left','right']):Image.fromarray(im).save(OUT/f'{cut}-{side}.png')
    sheet=Image.new('RGB',(2560,3*752),'#141414');d=ImageDraw.Draw(sheet)
    for row,(label,frames) in enumerate(rows):
        d.text((10,row*752+8),f'{cut} / {label} / outgoing LEFT, incoming RIGHT',fill='white')
        for col,im in enumerate(frames):sheet.paste(Image.fromarray(im),(col*1280,row*752+32))
    sheet.save(OUT/f'{cut}-comparison.jpg',quality=95)
