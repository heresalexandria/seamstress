"""Color-only residual study. Correspondence measures colors; output never warps.

Experiments operate after the checkpoint's unchanged global framing and LUTs.
They are not automatically accepted production corrections.
"""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import cv2,numpy as np
from PIL import Image,ImageDraw
from seamstress.media import read_frames
from seamstress.conform import conform_frame,tone_lut_at
from seamstress.repair import flow,sample,grid
cv2.setNumThreads(2)
OUT=ROOT/'research/local-color-probe';OUT.mkdir(exist_ok=True)
PLAN=json.loads((ROOT/'plans/IYTYT-eight-joins.json').read_text())


def baseline(image,n):
    matrix=np.array(PLAN['view_matrix'])@np.array(PLAN['frame_matrices'][n])
    image=conform_frame(image,matrix,np.ones(3),np.zeros(3))[0]
    lut=tone_lut_at(n,PLAN['grade_curves'])
    return image if lut is None else cv2.LUT(image,lut[:,None,:]).round().astype(np.uint8)


def observations(left,right):
    left,right=[cv2.resize(x,(640,360),interpolation=cv2.INTER_AREA) for x in [left,right]]
    fw,bw=flow(left,right),flow(right,left);xy=grid(fw.shape);rq=xy+fw
    good=np.linalg.norm(fw+sample(bw,fw),axis=2)<.6
    left=cv2.GaussianBlur(left.astype(np.float32),(0,0),.65)
    right=sample(cv2.GaussianBlur(right.astype(np.float32),(0,0),.65),fw)
    edges=[]
    for x in [left,right]:
        dx=cv2.Sobel(x,cv2.CV_32F,1,0);dy=cv2.Sobel(x,cv2.CV_32F,0,1)
        edges.append(np.sqrt(dx*dx+dy*dy).max(axis=2))
    good&=(np.maximum(*edges)<22)&(abs(left-right).max(axis=2)<35)
    good&=(rq[:,:,0]>5)&(rq[:,:,0]<634)&(rq[:,:,1]>5)&(rq[:,:,1]<354)
    good[:5]=False;good[-5:]=False;good[:,:5]=False;good[:,-5:]=False
    train=((xy[:,:,0]//40+xy[:,:,1]//40).astype(int)%2)==0
    yy,xx=np.where(good);take=np.arange(len(xx))[::max(1,len(xx)//20000)]
    yy,xx=yy[take],xx[take]
    return left[yy,xx],right[yy,xx],xy[yy,xx]/[639,359],rq[yy,xx]/[639,359],train[yy,xx]


def features(rgb,xy,mode,scale=None):
    if mode=='color':return rgb.astype(np.float32)/np.array(scale or [48,48,48],np.float32)
    if mode=='spatial':return xy.astype(np.float32)/np.array(scale or [.35,.35],np.float32)
    return np.concatenate([rgb,xy],axis=1).astype(np.float32)/np.array(scale or [48,48,48,.4,.4],np.float32)


def basis(values,centers):
    d=np.maximum(np.sum(values*values,axis=1)[:,None]+np.sum(centers*centers,axis=1)[None,:]-2*values@centers.T,0)
    # A partition of smooth color/position neighborhoods, tapered to identity
    # for samples far outside the calibrated colors and positions.
    nearest=d.min(axis=1);b=np.exp(-.5*(d-nearest[:,None]))
    b/=np.maximum(b.sum(axis=1)[:,None],1e-9)
    confidence=np.exp(-np.maximum(nearest-3,0)/3)
    return b*confidence[:,None]


def fit(rgb,xy,target,mode,centers,scale=None,ridge_factor=.015):
    x=features(rgb,xy,mode,scale);b=basis(x,centers)
    headroom=4*(rgb/255)*(1-rgb/255)
    coefs=[]
    for c in range(3):
        a=b*headroom[:,c,None]
        y=target[:,c]-rgb[:,c];weights=np.ones(len(y));ridge=np.eye(a.shape[1])*(len(a)/len(centers)*ridge_factor)
        for _ in range(5):
            coef=np.linalg.solve(a.T@(a*weights[:,None])+ridge,a.T@(y*weights))
            e=a@coef-y;weights=np.minimum(1,2/np.maximum(abs(e),1e-6))
        coefs.append(coef)
    return {'mode':mode,'feature_scale':scale,'centers':centers.tolist(),'coefficients':np.stack(coefs,axis=1).tolist(),'limit':18.}


def apply_samples(rgb,xy,model):
    c=rgb.astype(np.float32);b=basis(features(c,xy,model['mode'],model.get('feature_scale')),np.array(model['centers'],np.float32))
    raw=b@np.array(model['coefficients'],np.float32)
    correction=model['limit']*np.tanh(raw/model['limit'])*4*(c/255)*(1-c/255)
    return c+correction


def apply_image(image,model):
    h,w=image.shape[:2];rgb=image.reshape(-1,3);xy=(grid((h,w))/[w-1,h-1]).reshape(-1,2)
    out=np.empty_like(rgb)
    for start in range(0,len(rgb),32768):
        result=apply_samples(rgb[start:start+32768],xy[start:start+32768],model)
        assert result.min()>=-1e-4 and result.max()<=255.0001
        out[start:start+32768]=np.rint(result).astype(np.uint8)
    return out.reshape(image.shape)


def main(cut=361):
    frames=read_frames(ROOT/'IYTYT.mp4',cut-3,6)
    base=[baseline(f,cut-3+i) for i,f in enumerate(frames)]
    pairs=[observations(base[a],base[b]) for a,b in [(2,3),(1,4),(0,5)]]
    l,r,lp,rp,train=[np.concatenate([p[i] for p in pairs]) for i in range(5)]
    target=(l+r)/2;report=[];models={}
    for mode,k in [('color',24),('spatial',24),('hybrid',48)]:
        mid=features(target[train],(lp[train]+rp[train])/2,mode)
        cv2.setRNGSeed(19)
        _,_,centers=cv2.kmeans(mid,k,None,(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,60,.02),3,cv2.KMEANS_PP_CENTERS)
        a=fit(l[train],lp[train],target[train],mode,centers);b=fit(r[train],rp[train],target[train],mode,centers)
        fixed_l=apply_samples(l,lp,a);fixed_r=apply_samples(r,rp,b)
        row={'mode':mode,'sample_count':len(l),'heldout_count':int((~train).sum()),'baseline_heldout_mae':float(abs(l-r)[~train].mean()),'candidate_heldout_mae':float(abs(fixed_l-fixed_r)[~train].mean()),'baseline_heldout_p90':float(np.percentile(abs(l-r)[~train],90)),'candidate_heldout_p90':float(np.percentile(abs(fixed_l-fixed_r)[~train],90)),'peak_channel_change':float(max(abs(fixed_l-l).max(),abs(fixed_r-r).max()))}
        report.append(row);models[mode]={'frame':cut,'left':a,'right':b}
        print(row,flush=True)
        corrected=[apply_image(base[2],a),apply_image(base[3],b)]
        sheet=Image.new('RGB',(2560,1488),'#141414');d=ImageDraw.Draw(sheet)
        for row_idx,images in enumerate([base[2:4],corrected]):
            for col_idx,im in enumerate(images):sheet.paste(Image.fromarray(im),(1280*col_idx,744*row_idx+24))
            d.text((10,744*row_idx+5),'CHECKPOINT' if row_idx==0 else f'{mode.upper()} COLOR ONLY',fill='white')
        sheet.save(OUT/f'{cut}-{mode}-comparison.jpg',quality=95)
    (OUT/f'{cut}-models.json').write_text(json.dumps(models,indent=2))
    (OUT/f'{cut}-report.json').write_text(json.dumps(report,indent=2))


if __name__=='__main__':main(int(sys.argv[1]) if len(sys.argv)>1 else 361)
