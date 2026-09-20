"""Validate stronger response only; preserve all baseline geometry and production files."""
from pathlib import Path
import sys,json,cv2,numpy as np,hashlib
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'research')]
from seamstress.repair import flow,sample,resize_flow
from local_color_probe import apply_samples as apply_v1
from directional_local_color import apply_samples as apply_directional
def apply_samples(rgb,xy,model):
 return (apply_directional if model.get("response")=="directional-gamut-v2" else apply_v1)(rgb,xy,model)
D=Path(__file__).resolve().parent;OUT=D/'directional-response';OUT.mkdir(exist_ok=True);cv2.setNumThreads(2)
rs={361:[{'id':r['id'],'material':r['material'],'left_rect_xyxy':r['rect_xyxy']} for r in json.load(open(D/'visual-361.json'))['regions']]}
rs.update({r['frame']:r['regions'] for r in json.load(open(D/'visual-other-joins.json'))['cuts']})
def ed(im):
 im=cv2.GaussianBlur(im,(0,0),.7);edge=np.zeros(im.shape[:2],np.uint8)
 for c in range(3):edge|=cv2.Canny(im[:,:,c],18,45)
 return cv2.distanceTransform((edge==0).astype(np.uint8),cv2.DIST_L2,5)
def get(im,xy):
 out=[]
 for a in range(0,len(xy),15000):
  q=xy[a:a+15000];out.append(cv2.remap(im,q[:,0,None].astype(np.float32),q[:,1,None].astype(np.float32),cv2.INTER_LINEAR).reshape((len(q),)+im.shape[2:]))
 return np.concatenate(out)
def stat(l,r):
 if not len(l):return None
 delta=r-l;med=np.median(delta,0)
 return {'n':len(l),'signed_rgb_mean':delta.mean(0).tolist(),'signed_rgb_median':med.tolist(),'mean_absolute_signed_bias':float(abs(delta.mean(0)).mean()),'rgb_mae':float(abs(delta).mean()),'residual_mad_about_median':np.median(abs(delta-med),0).tolist()}
def crop_values(image,rect,model):
 x0,y0,x1,y1=rect;im=image[y0:y1,x0:x1];y,x=np.mgrid[y0:y1,x0:x1];xy=np.column_stack([x.ravel()/1279,y.ravel()/719]);rgb=im.reshape(-1,3)
 if model is not None:rgb=apply_samples(rgb,xy,model)
 return np.rint(rgb.reshape(im.shape)).clip(0,255).astype(np.uint8)
def sheet(cut,name,rect,a,b,models):
 x0,y0,x1,y1=rect;w=x1-x0;h=y1-y0;scale=2 if w<250 else 1;w*=scale;h*=scale;step=max(w,230);canvas=Image.new('RGB',(step*2+32,(h+30)*4+24),'#161616');dr=ImageDraw.Draw(canvas)
 for j,(label,model) in enumerate([('Baseline',None),('v1 limit18',models['v1']),('Strong limit63.75',models['strong']),('Directional gamut',models['directional'])]):
  for i,(im,side) in enumerate([(a,'left'),(b,'right')]):
   crop=crop_values(im,rect,model[side] if model else None);crop=Image.fromarray(crop).resize((w,h),Image.Resampling.NEAREST)
   x=8+i*(step+16);y=8+j*(h+30);dr.text((x,y),f'{label} / '+('outgoing' if i==0 else 'incoming'),fill='white');canvas.paste(crop,(x,y+20))
 canvas.save(OUT/f'{cut}-{name}.png')
context={722:[('hair-context',[365,28,495,169]),('car-context',[165,455,781,630])],2527:[('table-context',[261,576,821,667]),('outfit-context',[1065,340,1280,645]),('wall-couch-context',[180,20,346,510])],3240:[('characters-context',[486,265,776,544]),('stable-stone-context',[157,481,373,642])],1444:[('hair-context',[178,134,380,239])],2166:[('sky-context',[900,20,1180,111])],361:[('character-context',[743,245,869,420])],1083:[('tracksuit-context',[843,380,1024,581])],1805:[('foliage-context',[1107,10,1278,202])],2888:[('fur-context',[446,367,540,441])]}
rows=[]
for cut,regions in rs.items():
 paths={'v1':ROOT/'research/local-color-fit'/f'{cut}-model.json','strong':ROOT/'research/local-color-strong'/f'{cut}-model.json','directional':ROOT/'research/local-color-directional'/f'{cut}-model.json'}
 if not paths['directional'].exists():continue
 models={n:json.load(open(p)) for n,p in paths.items()}
 a,b=[np.array(Image.open(D/str(cut)/f'{s}.png').convert('RGB')) for s in ['left','right']]
 ap,bp=[cv2.resize(im,(640,360),interpolation=cv2.INTER_AREA) for im in [a,b]]
 f,bk=flow(ap,bp),flow(bp,ap);fw=resize_flow(f,(1280,720));fb=cv2.resize(np.linalg.norm(f+sample(bk,f),axis=2),(1280,720))*2
 smooth=[cv2.GaussianBlur(im.astype(np.float32),(0,0),.8) for im in [a,b]];eds=[ed(im) for im in [a,b]];rrs=[]
 for region in regions:
  x0,y0,x1,y1=region['left_rect_xyxy'];yy,xx=np.mgrid[y0:y1,x0:x1];xy=np.column_stack([xx.ravel(),yy.ravel()]).astype(np.float32);rq=xy+get(fw,xy);l,r=get(smooth[0],xy),get(smooth[1],rq)
  keep=(get(fb,xy)<1.2)&(get(eds[0],xy)>4)&(get(eds[1],rq)>4)&(abs(l-r).max(1)<35);xy,rq,l,r=[v[keep] for v in [xy,rq,l,r]]
  rr={k:region[k] for k in ['id','material','left_rect_xyxy']};rr['baseline']=stat(l,r)
  for n,m in models.items():
   if len(l):cl,cr=apply_samples(l,xy/[1279,719],m['left']),apply_samples(r,rq/[1279,719],m['right']);rr[n]=stat(cl,cr)
   else:rr[n]=None
  if len(l):
   rr['bias_change_strong_vs_v1']=rr['strong']['mean_absolute_signed_bias']-rr['v1']['mean_absolute_signed_bias'];rr['bias_change_strong_vs_baseline']=rr['strong']['mean_absolute_signed_bias']-rr['baseline']['mean_absolute_signed_bias'];rr['bias_change_directional_vs_v1']=rr['directional']['mean_absolute_signed_bias']-rr['v1']['mean_absolute_signed_bias'];rr['bias_change_directional_vs_strong']=rr['directional']['mean_absolute_signed_bias']-rr['strong']['mean_absolute_signed_bias']
   print(cut,region['id'],'n',len(l),'base',np.round(rr['baseline']['signed_rgb_mean'],2),'v1',np.round(rr['v1']['signed_rgb_mean'],2),'strong',np.round(rr['strong']['signed_rgb_mean'],2),'directional',np.round(rr['directional']['signed_rgb_mean'],2),'dbias_vs_v1',round(rr['bias_change_directional_vs_v1'],3),flush=True)
  rrs.append(rr)
 for name,rect in context.get(cut,[]):sheet(cut,name,rect,a,b,models)
 rows.append({'frame':cut,'models':{n:{'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'limit':models[n]['left']['limit'],'response':models[n]['left'].get('response','hybrid-rbf-headroom-v1')} for n,p in paths.items()},'regions':rrs})
result={'method':'Same independent native flat-interior measurements as all-models-native-material-evaluation.json. No models fitted. Signed RGB=incoming minus outgoing. Model-specific limit recorded; image geometry/drawings never remapped. Context crops are unchanged native pixels passed only through RGB correction, then optionally enlarged2x with nearest neighbor.','cuts':rows}
(OUT/'native-material-evaluation.json').write_text(json.dumps(result,indent=2))
