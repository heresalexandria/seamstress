"""Pointwise RGB residual models on the accepted source-conform baseline.

Optical flow supplies correspondence samples only. Neither geometry nor frame
selection is modified; every proposed output is a function of its own RGB.
"""
from pathlib import Path
import json,sys,time
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import cv2,numpy as np
from PIL import Image,ImageDraw
from seamstress.media import read_frames
from research.segment_color import matches
OUT=Path(__file__).parent;SOURCE=ROOT/'output/IYTYT-source-conform-eight-joins.mp4'
CUTS=[361,722,1083,1444,1805,2166,2527,2888,3240]
MODELS=['diagonal_affine','cross_affine','protected_cross','protected_quadratic']
MATERIALS=['dark','light_neutral','blue','warm','green','neutral','other']
cv2.setNumThreads(4)


def material(x):
 rgb=x/255;lo=rgb.min(1);hi=rgb.max(1);sat=hi-lo;luma=rgb@[.2126,.7152,.0722]
 labels=np.full(len(x),6,int)
 labels[sat<.14]=5
 labels[(rgb[:,1]>rgb[:,0]*1.15)&(rgb[:,1]>rgb[:,2]*1.1)&(sat>.08)]=4
 labels[(rgb[:,0]>rgb[:,1]*1.08)&(rgb[:,1]>rgb[:,2]*1.04)&(sat>.08)]=3
 labels[(rgb[:,2]>rgb[:,0]*1.15)&(rgb[:,2]>rgb[:,1]*1.05)&(sat>.08)]=2
 labels[(sat<.13)&(luma>.65)]=1
 labels[luma<.09]=0
 return labels


def basis(x,kind,c):
 z=x/255;v=2*z-1
 if kind=='diagonal_affine':return np.column_stack([np.ones(len(z)),v[:,c]])
 linear=np.column_stack([np.ones(len(z)),v])
 if kind!='protected_quadratic':return linear
 return np.column_stack([linear,v*v,v[:,0]*v[:,1],v[:,0]*v[:,2],v[:,1]*v[:,2]])


def fit_half(x,target,kind):
 coefs=[];protected=kind.startswith('protected')
 for c in range(3):
  design=basis(x,kind,c);d=(target[:,c]-x[:,c])/255
  if protected:design=design*(x[:,c]/255*(1-x[:,c]/255))[:,None]
  strength=np.array([.00002]+[.0001]*(design.shape[1]-1))
  if kind=='protected_quadratic':strength[4:]=.0008
  if not protected:strength*=.1
  regularizer=np.diag(strength)*len(x)
  beta=np.zeros(design.shape[1])
  for _ in range(10):
   err=d-design@beta;sigma=max(.8/255,1.4826*np.median(abs(err-np.median(err))))
   w=np.minimum(1,1.5*sigma/np.maximum(abs(err),1e-9))
   beta=np.linalg.solve(design.T@(design*w[:,None])+regularizer,design.T@(d*w))
  coefs.append(beta)
 model={'kind':kind,'coefs':np.array(coefs).tolist(),'protection_scale':1.}
 # x + x(1-x)*delta stays inside [0,1] whenever |delta|<=1. Limit to.5
 # over a dense RGB cube, providing margin against numerical overshoot.
 if protected:
  grid=np.stack(np.meshgrid(*[np.linspace(0,255,17)]*3,indexing='ij'),-1).reshape(-1,3)
  peak=max(float(np.max(abs(basis(grid,kind,c)@coefs[c]))) for c in range(3))
  model['protection_scale']=min(1.,.5/max(peak,1e-12))
 return model


def apply(x,model):
 shape=x.shape;flat=np.asarray(x,dtype=float).reshape(-1,3);out=flat.copy();kind=model['kind'];coef=np.array(model['coefs'])
 for c in range(3):
  shift=basis(flat,kind,c)@coef[c]*model['protection_scale']
  if kind.startswith('protected'):shift*=flat[:,c]/255*(1-flat[:,c]/255)
  out[:,c]+=255*shift
 return out.reshape(shape)


def fit(x,y,kind):
 target=(x+y)*.5
 return {'right':fit_half(x,target,kind),'left':fit_half(y,target,kind)}


def mapped(x,y,model):return apply(x,model['right']),apply(y,model['left'])


def metrics(x,y,model=None):
 a,b=(x,y) if model is None else mapped(x,y,model)
 err=abs(a-b);return {'mae':float(err.mean()),'p90_rgb_error':float(np.percentile(err,90)),'signed_median_rgb':np.median(b-a,axis=0).tolist()}


def load_observations(cut):
 cache=OUT/f'observations-{cut}.npz'
 if cache.exists():return dict(np.load(cache))
 frames=list(read_frames(SOURCE,cut-10,20,size=(640,360)))
 data={};observations=[]
 for pair,(a,b) in enumerate([(9,10),(8,11),(7,12)]):
  x,y,train,tile,coverage=matches(frames[a],frames[b]);take=np.arange(0,len(x),3)
  observations.append((x[take],y[take],train[take],tile[take],np.full(len(take),pair)))
 for i,key in enumerate(['x','y','train','tile','pair']):data[key]=np.concatenate([o[i] for o in observations])
 for j,(a,b) in enumerate([(1,3),(16,18)]):
  x,y,t,tile,cov=matches(frames[a],frames[b]);take=np.arange(0,len(x),3)
  for key,value in [('x',x[take]),('y',y[take]),('train',t[take]),('tile',tile[take])]:data[f'control{j}_{key}']=value
 data['left_image']=frames[9];data['right_image']=frames[10]
 np.savez_compressed(cache,**data);return data


def color_safety(model,image):
 values=apply(image,model);flat=image.reshape(-1,3).astype(float);out=values.reshape(-1,3)
 gray=flat.mean(1);ink=(flat.max(1)<20);skin=(flat[:,0]>flat[:,1]*1.08)&(flat[:,1]>flat[:,2]*1.05)&(gray>70)&(gray<220);sky=(flat[:,2]>flat[:,0]*1.15)&(flat[:,2]>flat[:,1]*1.05)
 result={'clip_channel_fraction':float(np.mean((out<0)|(out>255))),'mean_abs_change':float(abs(out-flat).mean()),'p99_abs_change':float(np.percentile(abs(out-flat),99)),'black_mapped':apply(np.zeros((1,3)),model)[0].tolist(),'white_mapped':apply(np.full((1,3),255.),model)[0].tolist()}
 for name,mask in [('ink',ink),('warm_skin_proxy',skin),('blue_sky_proxy',sky)]:
  if mask.sum():result[name]={'pixels':int(mask.sum()),'p95_abs_change_rgb':np.percentile(abs(out-flat)[mask],95,axis=0).tolist(),'mean_change_rgb':np.mean((out-flat)[mask],axis=0).tolist()}
 grid=np.stack(np.meshgrid(*[np.linspace(0,255,13)]*3,indexing='ij'),-1).reshape(-1,3)
 outg=apply(grid,model);jac=[]
 for c in range(3):
  plus=grid.copy();minus=grid.copy();plus[:,c]+=.01;minus[:,c]-=.01
  jac.append((apply(plus,model)-apply(minus,model))/.02)
 jac=np.stack(jac,axis=-1);sing=np.linalg.svd(jac,compute_uv=False)
 result.update(rgb_cube_clip_fraction=float(np.mean((outg<-.001)|(outg>255.001))),jacobian_min_singular=float(sing[:,-1].min()),jacobian_max_singular=float(sing[:,0].max()),jacobian_min_diagonal=float(np.diagonal(jac,axis1=1,axis2=2).min()))
 return result


def analyze(cut):
 d=load_observations(cut);x,y,train=d['x'],d['y'],d['train'];test=~train;labels=material((x+y)*.5)
 report={'cut':cut,'sample_count':len(x),'baseline_heldout':metrics(x[test],y[test]),'models':{}}
 candidates={}
 for kind in MODELS:
  model=fit(x[train],y[train],kind);candidates[kind]=model
  rec={'parameters':model,'spatial_holdout':metrics(x[test],y[test],model),'materials':{}}
  # A completely unseen coarse color family tests extrapolation. This is
  # stricter than spatial tiles, whose same colors occur in train and test.
  for label,name in enumerate(MATERIALS):
   held=test&(labels==label);seen=train&(labels!=label)
   if held.sum()<120 or seen.sum()<1000:continue
   loo=fit(x[seen],y[seen],kind)
   rec['materials'][name]={'n':int(held.sum()),'baseline':metrics(x[held],y[held]),'spatial_fit':metrics(x[held],y[held],model),'leave_color_family_out':metrics(x[held],y[held],loo)}
  # Separate cross-cut pair fits should predict comparable correction on
  # exactly the same observed colors, avoiding changed sample distributions.
  anchor=x[test][::10];anchor_left=y[test][::10];predictions=[]
  for pair in range(3):
   use=train&(d['pair']==pair);individual=fit(x[use],y[use],kind)
   predictions.append(np.concatenate([apply(anchor,individual['right'])-anchor,apply(anchor_left,individual['left'])-anchor_left]))
  spread=np.ptp(np.stack(predictions),axis=0)
  rec['across_pair_prediction_spread_p95_rgb']=np.percentile(spread,95,axis=0).tolist()
  rec['safety']={'left':color_safety(model['left'],d['left_image']),'right':color_safety(model['right'],d['right_image'])}
  controls=[]
  for j in range(2):
   cx,cy,ct=d[f'control{j}_x'],d[f'control{j}_y'],d[f'control{j}_train'];cm=fit(cx[ct],cy[ct],kind);a,b=mapped(cx[~ct],cy[~ct],cm)
   controls.append({'baseline':metrics(cx[~ct],cy[~ct]),'fitted':metrics(cx[~ct],cy[~ct],cm),'p95_added_color_change':float(np.percentile(abs(np.concatenate([a-cx[~ct],b-cy[~ct]])),95))})
  rec['within_clip_controls']=controls;report['models'][kind]=rec
  print('FIT',cut,kind,'baseline',round(report['baseline_heldout']['mae'],3),'holdout',round(rec['spatial_holdout']['mae'],3),'controls',np.round([c['p95_added_color_change'] for c in controls],2).tolist(),flush=True)
 # Show only pointwise point corrections of existing baseline geometry.
 panels=[('CURRENT protected diagonal',d['left_image'],d['right_image'])]
 for kind in MODELS:
  m=candidates[kind];panels.append((kind,np.clip(apply(d['left_image'],m['left']),0,255).round().astype(np.uint8),np.clip(apply(d['right_image'],m['right']),0,255).round().astype(np.uint8)))
 sheet=Image.new('RGB',(1280,390*len(panels)),'#111');draw=ImageDraw.Draw(sheet)
 for row,(name,a,b) in enumerate(panels):
  sheet.paste(Image.fromarray(a),(0,row*390));sheet.paste(Image.fromarray(b),(640,row*390));draw.text((10,row*390+366),f'{name}: before {cut-1}',fill='white');draw.text((650,row*390+366),f'after {cut}',fill='white')
 sheet.save(OUT/f'comparison-{cut}.jpg',quality=95)
 return report


def main():
 reports=[];start=time.perf_counter()
 for cut in CUTS:
  result=analyze(cut);reports.append(result);(OUT/'results.json').write_text(json.dumps({'baseline':str(SOURCE),'geometry_and_frame_timing':'Unchanged; output mapping is pointwise RGB only. Flow was used only for sample correspondence.','reports':reports},indent=2))
 print('DONE seconds',time.perf_counter()-start,flush=True)

if __name__=='__main__':main()
