from pathlib import Path
import sys,json,cv2,numpy as np,hashlib
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'research'))
from PIL import Image,ImageDraw
from seamstress.repair import flow,sample,resize_flow
from local_color_probe import apply_samples
cv2.setNumThreads(2)
OUT=Path(__file__).resolve().parent
regions=json.load(open(OUT/'visual-361.json'))['regions']
a,b=[np.array(Image.open(OUT/'361'/f'{s}.png').convert('RGB')) for s in ['left','right']]
as_,bs_=[cv2.resize(x,(640,360),interpolation=cv2.INTER_AREA) for x in [a,b]]
f,bk=flow(as_,bs_),flow(bs_,as_);fw=resize_flow(f,(1280,720));fb=cv2.resize(np.linalg.norm(f+sample(bk,f),axis=2),(1280,720))*2

def edge_distance(im):
 x=cv2.GaussianBlur(im,(0,0),.7);edge=np.zeros(x.shape[:2],np.uint8)
 for c in range(3):edge|=cv2.Canny(x[:,:,c],18,45)
 return cv2.distanceTransform((edge==0).astype(np.uint8),cv2.DIST_L2,5)

def get(im,xy):return cv2.remap(im,xy[:,0].reshape(-1,1).astype(np.float32),xy[:,1].reshape(-1,1).astype(np.float32),cv2.INTER_LINEAR).reshape((len(xy),)+im.shape[2:])

def stat(l,r):
 d=r-l;med=np.median(d,axis=0);return {'samples':len(l),'mean_rgb_delta':d.mean(axis=0).tolist(),'median_rgb_delta':med.tolist(),'mean_absolute_signed_bias':float(abs(d.mean(axis=0)).mean()),'rgb_mae':float(abs(d).mean()),'mad_about_median':np.median(abs(d-med),axis=0).tolist()}

smooth=[cv2.GaussianBlur(x.astype(np.float32),(0,0),.8) for x in [a,b]];eds=[edge_distance(x) for x in [a,b]]
oldpath=ROOT/'research/local-color-probe/361-models.json';newpath=ROOT/'research/local-color-probe/361-tuning-models.json'
old=json.load(open(oldpath));new=json.load(open(newpath));models={'color':old['color'],'hybrid':old['hybrid'],**{k:new[k] for k in ['local-light-prior','local-fine','local-fine-regularized'] if k in new}}
rows=[];annot=Image.fromarray(a);draw=ImageDraw.Draw(annot)
for i,roi in enumerate(regions):
 x0,y0,x1,y1=roi['rect_xyxy'];yy,xx=np.mgrid[y0:y1,x0:x1];xy=np.column_stack([xx.ravel(),yy.ravel()]).astype(np.float32);rq=xy+get(fw,xy)
 l,r=get(smooth[0],xy),get(smooth[1],rq)
 keep=(get(fb,xy)<1.2)&(get(eds[0],xy)>4)&(get(eds[1],rq)>4)&(abs(l-r).max(axis=1)<35)
 xy,rq,l,r=[v[keep] for v in [xy,rq,l,r]]
 row={'id':roi['id'],'material':roi['material'],'rect_xyxy':roi['rect_xyxy'],'selected_native_pixels':(x1-x0)*(y1-y0),'accepted_matched_interior_pixels':len(l),'baseline':stat(l,r),'models':{}}
 for name,model in models.items():
  cl,cr=apply_samples(l,xy/[1279,719],model['left']),apply_samples(r,rq/[1279,719],model['right']);row['models'][name]=stat(cl,cr)
 rows.append(row);draw.rectangle((x0,y0,x1,y1),outline='yellow',width=1);draw.text((x1+3,y0),str(i+1),fill='yellow')
 print(roi['id'],len(l),'baseline',np.round(row['baseline']['mean_rgb_delta'],2),'hybrid',np.round(row['models']['hybrid']['mean_rgb_delta'],2),'local',np.round(row['models']['local-light-prior']['mean_rgb_delta'],2),flush=True)
 # Native crop pixels receive only the candidate pointwise color function;
 # no geometry or correspondences are used to generate these display crops.
 orig_a,orig_b=a[y0:y1,x0:x1],b[y0:y1,x0:x1];y,x=np.mgrid[y0:y1,x0:x1];loc=np.column_stack([x.ravel()/1279,y.ravel()/719])
 model=models['local-light-prior'];ca=apply_samples(orig_a.reshape(-1,3),loc,model['left']).reshape(orig_a.shape);cb=apply_samples(orig_b.reshape(-1,3),loc,model['right']).reshape(orig_b.shape)
 scale=4;cw=(x1-x0)*scale;ch=(y1-y0)*scale;step=max(cw+8,170);sheet=Image.new('RGB',(step*4+16,ch+56),'#181818');dd=ImageDraw.Draw(sheet)
 dd.text((8,5),roi['material']+' / native flat interiors enlarged4x',fill='white')
 for j,(label,im) in enumerate([('Baseline outgoing',orig_a),('Baseline incoming',orig_b),('Test outgoing',ca),('Test incoming',cb)]):
  im=Image.fromarray(np.clip(np.rint(im),0,255).astype(np.uint8)).resize((cw,ch),Image.Resampling.NEAREST);sheet.paste(im,(8+j*step,48));dd.text((8+j*step,29),label,fill='white')
 sheet.save(OUT/'361'/f'material-{roi["id"]}.png')
annot.save(OUT/'361'/'material-regions-annotated.png')
report={'frame':361,'method':'Native Gaussian0.8px correspondence samples; flow640x360 then native interpolation; FB<1.2nativepx; contours excluded4nativepx. Signed delta is incoming minus outgoing. Models alter color values only.','model_files':{'prior':str(oldpath),'tuned':str(newpath)},'model_sha256':{'prior':hashlib.sha256(oldpath.read_bytes()).hexdigest(),'tuned':hashlib.sha256(newpath.read_bytes()).hexdigest()},'regions':rows}
(OUT/'361'/'material-model-evaluation.json').write_text(json.dumps(report,indent=2))
