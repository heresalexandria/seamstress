"""Independent local landmark evidence for cut geometry; never renders a video."""
from pathlib import Path
import cv2,numpy as np,json
from PIL import Image,ImageDraw
OUT=Path(__file__).resolve().parent
images=[np.array(Image.open(OUT/f'raw{n}.png')) for n in [360,361]]
gray=[cv2.GaussianBlur(cv2.cvtColor(a,cv2.COLOR_RGB2GRAY).astype(np.float32),(0,0),.65) for a in images]
landmarks=[('windshield upper left',239,200),('windshield upper right',1054,201),('windshield lower left',207,532),('windshield lower right',1203,508),('headlight top',671,679),('hood blue curve',407,667),('mirror upper left',28,477),('mirror lower right',128,541),('left wiper pivot',642,485),('right wiper pivot',1036,486),('left wiper tip',329,477),('left wiper joint',430,454),('right streetlamp top',1098,53),('left building top',117,304),('left lamp head',181,279),('right door mirror corner',1225,458)]
records=[]
for label,x,y in landmarks:
 radius=18;search=18
 template=gray[0][y-radius:y+radius+1,x-radius:x+radius+1]
 x0=max(0,x-radius-search);y0=max(0,y-radius-search)
 area=gray[1][y0:min(720,y+radius+search+1),x0:min(1280,x+radius+search+1)]
 result=cv2.matchTemplate(area,template,cv2.TM_CCOEFF_NORMED)
 _,peak,_,pos=cv2.minMaxLoc(result);px,py=pos
 def offset(left,mid,right):return float(np.clip(.5*(left-right)/(left-2*mid+right),-.75,.75))
 dx=offset(*result[py,px-1:px+2]) if 0<px<result.shape[1]-1 else 0
 dy=offset(*result[py-1:py+2,px]) if 0<py<result.shape[0]-1 else 0
 target=[float(x0+px+radius+dx),float(y0+py+radius+dy)]
 records.append({'label':label,'source360':[x,y],'target361':target,'NCC':peak})
p=np.array([r['source360'] for r in records]);q=np.array([r['target361'] for r in records])
weights=np.array([r['NCC'] for r in records])>.78
models={}
for kind,fn in [('similarity',cv2.estimateAffinePartial2D),('affine',cv2.estimateAffine2D)]:
 m,inside=fn(p[weights],q[weights],method=cv2.RANSAC,ransacReprojThreshold=2.5,maxIters=10000,confidence=.999,refineIters=50)
 err=np.linalg.norm(p@m[:,:2].T+m[:,2]-q,axis=1)
 models[kind]={'forward_matrix':m.tolist(),'median_error_native_px':float(np.median(err[weights])),'p90_error_native_px':float(np.percentile(err[weights],90)),'all_errors_native_px':err.tolist(),'inlier_count':int(inside.sum())}
 for r,e in zip(records,err):r[kind+'_error_native_px']=float(e)
sheet=Image.new('RGB',(2560,760),'#111');draw=ImageDraw.Draw(sheet)
for c,image in enumerate(images):
 sheet.paste(Image.fromarray(image),(1280*c,40))
 for i,r in enumerate(records):
  x,y=r['source360' if c==0 else 'target361'];x+=1280*c;y+=40
  draw.ellipse((x-6,y-6,x+6,y+6),outline='#ffff00',width=2);draw.text((x+7,y-14),str(i+1),fill='#ffff00',stroke_width=1,stroke_fill='black')
 draw.text((1280*c+10,10),f'Original {360+c} - independent static landmark matches',fill='white')
sheet.save(OUT/'landmark-evidence.jpg',quality=97)
(OUT/'landmark-evidence.json').write_text(json.dumps({'landmarks':records,'models':models},indent=2))
print(json.dumps({'landmarks':records,'models':models},indent=2))
