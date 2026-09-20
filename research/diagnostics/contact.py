import cv2
from PIL import Image,ImageDraw
from pathlib import Path
import json
out=Path(__file__).parent
cap=cv2.VideoCapture('IYTYT.mp4')
def sheet(frames,name,cols=4):
    thumbs=[]
    for f in frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES,f); ok,img=cap.read()
        if not ok: continue
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        thumb=Image.fromarray(img).resize((480,270))
        cell=Image.new('RGB',(480,300),(24,24,24));cell.paste(thumb,(0,30))
        ImageDraw.Draw(cell).text((8,8),f'frame {f} | {f*1001/24000:.5f}s',fill='white')
        thumbs.append(cell)
    result=Image.new('RGB',(480*cols,300*((len(thumbs)+cols-1)//cols)),(0,0,0))
    for i,t in enumerate(thumbs):result.paste(t,(480*(i%cols),300*(i//cols)))
    result.save(out/name)
for k in range(1,10):
    center=round(15*k*24000/1001)
    sheet([center-12,center-1,center,center+12],f'near_{k*15:03}.jpg')
