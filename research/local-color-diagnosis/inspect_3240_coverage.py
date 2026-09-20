from pathlib import Path
import sys,json,numpy as np,cv2
from PIL import Image
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'research')]
import local_color_probe as p
from strong_local_color import targets
D=Path(__file__).resolve().parent;OUT=D/'strong-response';cv2.setNumThreads(2)
a,b=[np.array(Image.open(D/'3240'/f'{s}.png').convert('RGB')) for s in ['left','right']]
l,r,lp,rp,tr=p.observations(a,b);mid=(l+r)/2;xy=lp*[1279,719]
strong=json.load(open(ROOT/'research/local-color-strong/3240-model.json'));v1=json.load(open(ROOT/'research/local-color-fit/3240-model.json'));lt,rt=targets(l,r)
# Match the exact calibration palette-weight computation; no fitting.
bins=np.floor(mid[tr]/24).astype(int);keys=bins[:,0]*121+bins[:,1]*11+bins[:,2];_,inv,counts=np.unique(keys,return_inverse=True,return_counts=True)
w=np.clip(np.sqrt(np.median(counts)/counts[inv]),.25,8);w/=w.mean();weights=np.zeros(len(l));weights[tr]=w
# Readable cyan class for the character outfit, excluding dark outlines.
cyan=(mid[:,0]<75)&(mid[:,1]>100)&(mid[:,1]<170)&(mid[:,2]>170)&(mid[:,2]<230)
rs=[('diagnostic_jacket_roi',[704,359,730,377],False),('pants_roi',[700,425,745,505],False),('whole_outfit_cyan_class',[663,318,770,534],True)]
def summary(q):
 return {'mean':q.mean(0).tolist(),'median':np.median(q,0).tolist(),'p05':np.quantile(q,.05,axis=0).tolist(),'p95':np.quantile(q,.95,axis=0).tolist(),'channel_zero_fraction':(q<.01).mean(0).tolist()}
rows=[]
for name,rect,classmask in rs:
 x0,y0,x1,y1=rect;m=(xy[:,0]>=x0)&(xy[:,0]<x1)&(xy[:,1]>=y0)&(xy[:,1]<y1)
 if classmask:m&=cyan
 row={'id':name,'rect':rect,'accepted_calibration_samples':int(m.sum()),'training_count':int((m&tr).sum()),'heldout_count':int((m&~tr).sum()),'training_weight_mean':float(weights[m&tr].mean()) if (m&tr).any() else None,'left_rgb':summary(l[m]),'right_rgb':summary(r[m]),'left_target_rgb':summary(lt[m]),'right_target_rgb':summary(rt[m])}
 for label,models in [('v1',v1),('strong',strong)]:
  sl=p.apply_samples(l[m],lp[m],models['left']);sr=p.apply_samples(r[m],rp[m],models['right']);row[label]={'left_rgb':summary(sl),'right_rgb':summary(sr),'delta_mean':(sr-sl).mean(0).tolist(),'left_target_error_mean':(sl-lt[m]).mean(0).tolist(),'right_target_error_mean':(sr-rt[m]).mean(0).tolist()}
 center=np.array(strong['left']['centers']);scale=np.array(strong['left']['feature_scale']);query=p.features(mid[m],(lp[m]+rp[m])/2,'hybrid',scale.tolist()).mean(0);order=np.argsort(((center-query)**2).sum(1))[:5]
 row['nearest_centers']=[{'index':int(i),'center_rgb':(center[i,:3]*scale[:3]).tolist(),'center_xy_native':(center[i,3:]*scale[3:]*[1279,719]).tolist(),'squared_distance':float(((center[i]-query)**2).sum())} for i in order]
 rows.append(row)
 print(name,'n',int(m.sum()),'train',int((m&tr).sum()),'hold',int((m&~tr).sum()),'weight',row['training_weight_mean'])
 for k in ['left_rgb','right_rgb','left_target_rgb','right_target_rgb']:print(' ',k,np.round(row[k]['mean'],2),'zeros',np.round(row[k]['channel_zero_fraction'],2))
 print(' strongout',np.round(row['strong']['left_rgb']['mean'],2),np.round(row['strong']['right_rgb']['mean'],2),'targeterr',np.round(row['strong']['left_target_error_mean'],2),np.round(row['strong']['right_target_error_mean'],2))
 print(' centers',row['nearest_centers'][:2])
blue_total=int(cyan.sum());blue_train=int((cyan&tr).sum());print('ALL',len(l),'training',int(tr.sum()),'cyan',blue_total,'cyantrain',blue_train)
json.dump({'frame':3240,'measurement':'Exact parent calibration observations on locked baseline native pairs; unchanged samples, train tiles and palette weights. No fit. Cyan class has R<75,G100..170,B170..230; outfit-restricted where stated. Targets are strong model feasible common targets.','all_calibration_samples':len(l),'all_training_samples':int(tr.sum()),'cyan_palette_total':blue_total,'cyan_palette_train':blue_train,'regions':rows},open(OUT/'3240-coverage.json','w'),indent=2)
