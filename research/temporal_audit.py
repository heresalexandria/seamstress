"""Read-only comparison of original and encoded v2 picture evolution."""
from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
from PIL import Image, ImageDraw

from seamstress.media import read_frames
from seamstress.repair import crop_frame

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "research" / "temporal_audit"
SEAMS = [361, 2166, 3240, 722, 1083, 1444, 1805, 2527, 2888]


def track(a, b, gap=1):
    ga = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)
    gb = cv2.cvtColor(b, cv2.COLOR_RGB2GRAY)
    points = cv2.goodFeaturesToTrack(ga, 1600, 0.008, 7, blockSize=5)
    if points is None or len(points) < 12:
        return None
    target, status, err = cv2.calcOpticalFlowPyrLK(ga, gb, points, None, winSize=(21,21), maxLevel=3)
    back, reverse, _ = cv2.calcOpticalFlowPyrLK(gb, ga, target, None, winSize=(21,21), maxLevel=3)
    p, q = points[:,0], target[:,0]
    valid = (status[:,0]>0)&(reverse[:,0]>0)&(np.linalg.norm(back[:,0]-p,axis=1)<1.5)
    p,q=p[valid],q[valid]
    if len(p)<12:return None
    affine,inliers=cv2.estimateAffinePartial2D(p,q,method=cv2.RANSAC,ransacReprojThreshold=1.5,maxIters=3000)
    if affine is None:return None
    pred=p@affine[:,:2].T+affine[:,2]
    residual=np.linalg.norm(pred-q,axis=1)
    center=np.array([a.shape[1]/2,a.shape[0]/2])
    drift=affine[:,:2]@center+affine[:,2]-center
    scale=np.sqrt(np.linalg.det(affine[:,:2]))
    return {'scale_pct_per_frame':float((scale**(1/gap)-1)*100),
            'dx_per_frame':float(drift[0]/gap),'dy_per_frame':float(drift[1]/gap),
            'nonrigid_p90':float(np.percentile(residual,90)),
            'inlier_fraction':float(inliers.mean()),'point_count':len(p),
            'motion_p50_per_frame':float(np.median(np.linalg.norm(q-p,axis=1))/gap)}


def curves(frames,start):
    records=[]
    for i,frame in enumerate(frames):
        gray=cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)
        blur=cv2.GaussianBlur(gray,(0,0),1)
        rec={'frame':start+i,'mean_rgb':frame.mean((0,1)).tolist(),
             'edge_energy':float(np.mean(abs(cv2.Laplacian(blur,cv2.CV_32F))))}
        if i:
            rec['picture_change_mae']=float(np.mean(abs(frame.astype(float)-frames[i-1].astype(float))))
            rec['step']=track(frames[i-1],frame)
        if i>=2:rec['two_frame']=track(frames[i-2],frame,2)
        if i>=4:rec['four_frame']=track(frames[i-4],frame,4)
        records.append(rec)
    return records


def strip(original,repaired,seam,start):
    offsets=[-8,-4,-2,-1,0,1,2,4,6,8,10,12]
    thumb=(320,180); band=205
    sheet=Image.new('RGB',(6*thumb[0],4*band),'#151922');draw=ImageDraw.Draw(sheet)
    for group in range(2):
        for version,frames in enumerate((original,repaired)):
            row=group*2+version
            for col,offset in enumerate(offsets[group*6:group*6+6]):
                frame=frames[seam+offset-start]
                sheet.paste(Image.fromarray(frame).resize(thumb),(col*thumb[0],row*band))
                label=f'{"SOURCE" if version==0 else "V2"} {seam+offset} ({offset:+d})'
                draw.text((col*thumb[0]+7,row*band+184),label,fill='#ffce57' if offset==0 else 'white')
    sheet.save(OUT/f'strip-{seam}.jpg',quality=94)


def draw_curves(records,seam):
    width,height=1200,720
    image=Image.new('RGB',(width,height),'#141820');draw=ImageDraw.Draw(image)
    fields=[('picture_change_mae','Picture change MAE'),('two_frame.scale_pct_per_frame','Scale % / frame (two-frame estimate)'),
            ('two_frame.dy_per_frame','Vertical center motion px / frame (two-frame estimate)'),
            ('edge_energy','Edge energy (blurred Laplacian)')]
    for row,(key,title) in enumerate(fields):
        top=25+row*175;left=65;right=width-30;bottom=top+135
        values=[]
        curves=[]
        for source in records:
            samples=[]
            for item in source:
                value=item
                for part in key.split('.'):
                    value=value.get(part) if isinstance(value,dict) else None
                if value is not None:samples.append((item['frame'],value));values.append(value)
            curves.append(samples)
        lo,hi=min(values),max(values);padding=max((hi-lo)*.1,.01);lo-=padding;hi+=padding
        xmin=records[0][0]['frame'];xmax=records[0][-1]['frame']
        xy=lambda x,y:(left+(right-left)*(x-xmin)/(xmax-xmin),bottom-(bottom-top)*(y-lo)/(hi-lo))
        draw.text((left,top-17),title,fill='white')
        draw.text((5,top),f'{hi:.2f}',fill='gray');draw.text((5,bottom-12),f'{lo:.2f}',fill='gray')
        xc=xy(seam,0)[0];draw.line((xc,top,xc,bottom),fill='#667080')
        if lo<0<hi:draw.line((left,xy(seam,0)[1],right,xy(seam,0)[1]),fill='#343943')
        for samples,color in zip(curves,('#ffbb66','#67d5f5')):
            draw.line([xy(x,y) for x,y in samples],fill=color,width=2)
        for offset in (-20,-10,0,10,20):
            draw.text((xy(seam+offset,lo)[0]-10,bottom+4),str(offset),fill='gray')
    draw.text((880,5),'ORIGINAL orange   V2 cyan',fill='white')
    image.save(OUT/f'curves-{seam}.png')


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    report=json.loads((ROOT/'output/v2/repaired.repair.json').read_text())
    results=[]
    for seam in SEAMS:
        start=seam-22;count=47
        cache=OUT/f'frames-{seam}.npz'
        if cache.exists():
            data=np.load(cache);original,repaired=data['original'],data['repaired']
        else:
            raw=read_frames(ROOT/'IYTYT.mp4',start,count,size=(640,360))
            original=np.stack([crop_frame(frame,report['crop_fraction']) for frame in raw])
            repaired=read_frames(ROOT/'output/v2/repaired.mp4',start,count,size=(640,360))
            np.savez_compressed(cache,original=original,repaired=repaired)
        a,b=curves(original,start),curves(repaired,start)
        strip(original,repaired,seam,start);draw_curves((a,b),seam)
        entry={'seam':seam,'original':a,'v2':b}
        results.append(entry)
        (OUT/'curves.json').write_text(json.dumps(results,indent=2))
        print('AUDIT',seam,flush=True)
        for name,records in (('original',a),('v2',b)):
            cut=next(x for x in records if x['frame']==seam)
            print(name,'picture',round(cut['picture_change_mae'],2),'camera',cut['two_frame'],flush=True)
    (OUT/'curves.json').write_text(json.dumps(results,indent=2))


if __name__=='__main__':main()
