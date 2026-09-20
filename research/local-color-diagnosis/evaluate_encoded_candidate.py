"""Read-only native ROI audit of completed encoded baseline/candidate movies.

Freezes the pre-encode baseline correspondence and material rectangles. Both
encoded versions are measured at exactly those locations. A second shared gate
requires both encoded pairs to pass the same strict contour/flow checks, so
changed sample selection cannot manufacture an apparent improvement.
"""
from pathlib import Path
import argparse,hashlib,json,subprocess,sys
import cv2,numpy as np
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from seamstress.media import probe
from seamstress.repair import flow,sample,resize_flow
from seamstress.local_color import apply_image
D=Path(__file__).resolve().parent;cv2.setNumThreads(2)

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def canonical(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def regions():
 out={361:[{'id':r['id'],'material':r['material'],'left_rect_xyxy':r['rect_xyxy']} for r in json.load(open(D/'visual-361.json'))['regions']]}
 out.update({r['frame']:r['regions'] for r in json.load(open(D/'visual-other-joins.json'))['cuts']})
 out[3240]=[dict(r) for r in out[3240]]+[
  {'id':'pants_extra','material':'Woman blue trousers','left_rect_xyxy':[700,425,745,505]},
  {'id':'outfit_cyan_extra','material':'Entire outfit cyan class','left_rect_xyxy':[663,318,770,534]}]
 return out

def get(im,xy):
 vals=[]
 for start in range(0,len(xy),15000):
  q=xy[start:start+15000];vals.append(cv2.remap(im,q[:,0,None].astype(np.float32),q[:,1,None].astype(np.float32),cv2.INTER_LINEAR).reshape((len(q),)+im.shape[2:]))
 return np.concatenate(vals)

def edges(im):
 blur=cv2.GaussianBlur(im,(0,0),.7);edge=np.zeros(im.shape[:2],np.uint8)
 for c in range(3):edge|=cv2.Canny(blur[:,:,c],18,45)
 return cv2.distanceTransform((edge==0).astype(np.uint8),cv2.DIST_L2,5)

def pair_fields(pair):
 ap,bp=[cv2.resize(im,(640,360),interpolation=cv2.INTER_AREA) for im in pair]
 f,bk=flow(ap,bp),flow(bp,ap)
 return {'flow':resize_flow(f,(1280,720)),'fb':cv2.resize(np.linalg.norm(f+sample(bk,f),axis=2),(1280,720))*2,
         'rgb':[cv2.GaussianBlur(im.astype(np.float32),(0,0),.8) for im in pair],
         'edges':[edges(im) for im in pair]}

def stats(left,right):
 if not len(left):return None
 delta=right-left;med=np.median(delta,0)
 return {'samples':len(left),'left_rgb_mean':left.mean(0).tolist(),'right_rgb_mean':right.mean(0).tolist(),
         'signed_rgb_mean':delta.mean(0).tolist(),'signed_rgb_median':med.tolist(),
         'mean_absolute_signed_bias':float(abs(delta.mean(0)).mean()),'rgb_mae':float(abs(delta).mean()),
         'residual_mad_about_median':np.median(abs(delta-med),0).tolist()}

def decode_selected(path,indices,metadata):
 # A single sequential decode per movie; no inaccurate time seeking and no
 # repeated decoding from frame zero for every seam.
 expression='+'.join(f'eq(n\\,{index})' for index in indices)
 command=['ffmpeg','-v','error','-nostdin','-i',str(path),'-map','0:v:0','-an','-sn','-dn',
          '-vf','select='+expression,'-sws_flags','accurate_rnd+full_chroma_int',
          '-fps_mode','passthrough','-frames:v',str(len(indices)),'-f','rawvideo','-pix_fmt','rgb24','pipe:1']
 result=subprocess.run(command,stdout=subprocess.PIPE,stderr=subprocess.PIPE,check=False)
 if result.returncode:raise RuntimeError(result.stderr.decode(errors='replace'))
 expected=len(indices)*metadata['width']*metadata['height']*3
 if len(result.stdout)!=expected:raise RuntimeError(f'Expected {expected} decoded bytes, got {len(result.stdout)}')
 arr=np.frombuffer(result.stdout,np.uint8).reshape(len(indices),metadata['height'],metadata['width'],3)
 return dict(zip(indices,arr))

def write_crop_sheet(out,cut,pairs):
 rect={361:[735,235,875,425],722:[365,28,495,169],1083:[840,380,1030,585],1444:[178,134,380,239],1805:[1080,8,1278,210],2166:[900,20,1180,111],2527:[250,570,825,670],2888:[446,367,545,447],3240:[485,265,780,544]}[cut]
 x0,y0,x1,y1=rect;w=x1-x0;h=y1-y0;step=max(w,245)
 sheet=Image.new('RGB',(step*2+24,(h+24)*len(pairs)+12),'#151515');dr=ImageDraw.Draw(sheet)
 for j,(name,pair) in enumerate(pairs.items()):
  for i,im in enumerate(pair):
   x=8+i*(step+8);y=6+j*(h+24);dr.text((x,y),name+' / '+('out' if i==0 else 'in'),fill='white');sheet.paste(Image.fromarray(im[y0:y1,x0:x1]),(x,y+18))
 sheet.save(out/f'{cut}-encoded-context.png')

def prepare(plan,out,cuts):
 baseline=json.load(open(ROOT/'plans/IYTYT-eight-joins.json'))
 for key in ['source_sha256','view_matrix','frame_matrices','grade_curves','segments']:
  if canonical(plan[key])!=canonical(baseline[key]):raise ValueError(f'Candidate changes frozen baseline {key}; cached input stills invalid')
 curves={c['frame']:c for c in plan['local_color_curves']};prepared={}
 for cut in cuts:
  clean=[np.array(Image.open(D/str(cut)/f'{side}.png').convert('RGB')) for side in ['left','right']]
  dest=out/'preencode'/str(cut);dest.mkdir(parents=True,exist_ok=True);stamp=dest/'identity.json'
  identity={'model_sha256':canonical(curves[cut]),'baseline_left_sha256':digest(D/str(cut)/'left.png'),'baseline_right_sha256':digest(D/str(cut)/'right.png')}
  if stamp.exists() and json.load(open(stamp))==identity and all((dest/f'{s}.png').exists() for s in ['left','right']):
   corrected=[np.array(Image.open(dest/f'{s}.png').convert('RGB')) for s in ['left','right']]
  else:
   corrected=[apply_image(im,curves[cut][side],weight=1.) for im,side in zip(clean,['left','right'])]
   for im,side in zip(corrected,['left','right']):Image.fromarray(im).save(dest/f'{side}.png')
   stamp.write_text(json.dumps(identity,indent=2))
  prepared[cut]=(clean,corrected);print('prepared native color-only anchors',cut,flush=True)
 return prepared

def main():
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--plan',type=Path,default=ROOT/'plans/IYTYT-color-refined.json')
 parser.add_argument('--baseline',type=Path,default=ROOT/'output/IYTYT-source-conform-eight-joins.mp4')
 parser.add_argument('--candidate',type=Path,default=ROOT/'output/IYTYT-source-conform-color-refined.mp4')
 parser.add_argument('--out',type=Path,default=D/'encoded-candidate')
 parser.add_argument('--prepare-only',action='store_true',help='Cache exact pre-encode candidate anchors without reading either movie')
 args=parser.parse_args();plan=json.load(open(args.plan));cuts=[c['frame'] for c in plan['local_color_curves']];args.out.mkdir(parents=True,exist_ok=True)
 prepared=prepare(plan,args.out,cuts)
 if args.prepare_only:return
 meta={name:probe(path) for name,path in [('baseline',args.baseline),('candidate',args.candidate)]}
 for name,m in meta.items():
  for key in ['width','height','frame_count','fps_fraction']:
   if m[key]!=plan['source'][key]:raise ValueError(f'{name} {key} differs from source; require completed frame-exact output')
  if m['frame_count_estimated']:raise ValueError(f'{name} frame count is estimated; wait for completed finalized movie')
 indices=sorted({i for cut in cuts for i in [cut-1,cut]})
 encoded={name:decode_selected(path,indices,meta[name]) for name,path in [('baseline',args.baseline),('candidate',args.candidate)]}
 allregions=regions();results=[]
 for cut in cuts:
  clean,corrected=prepared[cut];pairs={'baseline_preencode':clean,'candidate_preencode':corrected,
   'baseline_encoded':[encoded['baseline'][cut-1],encoded['baseline'][cut]],'candidate_encoded':[encoded['candidate'][cut-1],encoded['candidate'][cut]]}
  fields={name:pair_fields(pair) for name,pair in pairs.items()};ref=fields['baseline_preencode'];rows=[]
  for region in allregions[cut]:
   x0,y0,x1,y1=region['left_rect_xyxy'];yy,xx=np.mgrid[y0:y1,x0:x1];xy=np.column_stack([xx.ravel(),yy.ravel()]).astype(np.float32)
   rq=xy+get(ref['flow'],xy);left,right=[get(ref['rgb'][i],q) for i,q in enumerate([xy,rq])]
   frozen=(get(ref['fb'],xy)<1.2)&(get(ref['edges'][0],xy)>4)&(get(ref['edges'][1],rq)>4)&(abs(left-right).max(1)<35)
   if region['id']=='outfit_cyan_extra':
    mid=(left+right)/2;frozen&=(mid[:,0]<75)&(mid[:,1]>100)&(mid[:,1]<170)&(mid[:,2]>170)&(mid[:,2]<230)
   joint=frozen.copy();values={}
   for name,field in fields.items():
    l,r=get(field['rgb'][0],xy),get(field['rgb'][1],rq);values[name]=(l,r)
    if name.endswith('_encoded'):
     flow_agreement=np.linalg.norm(get(field['flow'],xy)-get(ref['flow'],xy),axis=1)<1.2
     joint&=(get(field['fb'],xy)<1.2)&(get(field['edges'][0],xy)>4)&(get(field['edges'][1],rq)>4)&(abs(l-r).max(1)<35)&flow_agreement
   record={**region,'selected_rectangle_pixels':len(xy),'frozen_preencode_count':int(frozen.sum()),'joint_encoded_gate_count':int(joint.sum()),'retained_fraction':float(joint.sum()/max(frozen.sum(),1)),'frozen':{},'joint':{}}
   for gate_name,gate in [('frozen',frozen),('joint',joint)]:
    for name,(l,r) in values.items():record[gate_name][name]=stats(l[gate],r[gate])
   if joint.any():
    before=record['joint']['baseline_encoded'];after=record['joint']['candidate_encoded'];expected=record['joint']['candidate_preencode']
    record['encoded_signed_bias_change']=after['mean_absolute_signed_bias']-before['mean_absolute_signed_bias']
    record['candidate_encoding_added_signed_rgb_bias']=(np.array(after['signed_rgb_mean'])-np.array(expected['signed_rgb_mean'])).tolist()
    print(cut,region['id'],'n',int(joint.sum()),'baselineEncoded',np.round(before['signed_rgb_mean'],2),'candidateEncoded',np.round(after['signed_rgb_mean'],2),'candidatePreencode',np.round(expected['signed_rgb_mean'],2),flush=True)
   rows.append(record)
  results.append({'frame':cut,'regions':rows})
  for name,pair in pairs.items():
   if name.endswith('_encoded'):
    for side,im in zip(['left','right'],pair):Image.fromarray(im).save(args.out/f'{cut}-{name}-{side}.png')
  write_crop_sheet(args.out,cut,pairs)
  # Write incremental evidence after each completed cut without claiming all done.
  payload={'completed':len(results)==len(cuts),'method':__doc__,'plan_path':str(args.plan),'plan_sha256':digest(args.plan),'videos':meta,'frame_indices':indices,'preserved_regions_sha256':{'first':digest(D/'visual-361.json'),'others':digest(D/'visual-other-joins.json')},'gate':'Frozen baseline-preencode DIS flow640; Gaussian0.8native; FB<1.2native,contour distance>4native,absRGBdiff<35. Joint additionally requires both encoded pairs to pass the same gates and flow agreement<1.2native. Same points for all four comparisons; no fitting.','cuts':results}
  (args.out/'native-material-evaluation.json').write_text(json.dumps(payload,indent=2))
 print('ENCODED NATIVE MATERIAL AUDIT COMPLETE',flush=True)
if __name__=='__main__':main()
