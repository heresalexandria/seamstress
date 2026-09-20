"""Draw Seamstress's original interlaced-thread app icon, with no remote assets."""
from pathlib import Path
import subprocess,tempfile
from PIL import Image,ImageDraw,ImageFilter
root=Path(__file__).resolve().parents[1];size=1024
image=Image.new('RGBA',(size,size));draw=ImageDraw.Draw(image)
draw.rounded_rectangle((42,42,982,982),radius=215,fill='#152729')
for x in range(64,960,8):draw.line((x,90,x,934),fill=(117,167,157,9),width=2)
for y in range(64,960,8):draw.line((90,y,934,y),fill=(203,165,164,9),width=2)
def curve(points,color,width):
    sampled=[]
    for i in range(0,len(points)-1,3):
        a,b,c,d=points[i:i+4]
        for n in range(101):
            t=n/100;sampled.append(tuple((1-t)**3*a[k]+3*(1-t)**2*t*b[k]+3*(1-t)*t*t*c[k]+t**3*d[k] for k in (0,1)))
    draw.line(sampled,fill=color,width=width,joint='curve')
    for x,y in (sampled[0],sampled[-1]):draw.ellipse((x-width/2,y-width/2,x+width/2,y+width/2),fill=color)
curve([(735,305),(625,183),(281,220),(303,408),(320,547),(702,472),(718,626),(740,822),(403,855),(283,714)],'#77d3c4',77)
curve([(681,260),(574,360),(462,581),(349,784)],'#17292a',109)
curve([(681,260),(574,360),(462,581),(349,784)],'#d365aa',57)
for x,y in [(699,736),(728,701),(749,658)]:draw.ellipse((x-5,y-5,x+5,y+5),fill='#e7cfac')
image.save(root/'app/assets/icon.png')
image.save(root/'app/assets/icon.icns',format='ICNS')
