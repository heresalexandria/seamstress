"""Audit encoded bridge starts, ends, and surrounding camera/picture evolution."""
from __future__ import annotations
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
sys.path.insert(0,str(Path(__file__).resolve().parent))
import cv2
import numpy as np
from PIL import Image,ImageDraw
from temporal_audit import track
from seamstress.media import read_frames
from seamstress.repair import flow,sample

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'research/final-bridge-audit'

def metrics(frames,start):
    result=[]
    for i,f in enumerate(frames):
        gray=cv2.cvtColor(f,cv2.COLOR_RGB2GRAY)
        gray=cv2.GaussianBlur(gray,(0,0),1).astype(np.float32)
        row={'frame':start+i,'mean_rgb':f.mean((0,1)).tolist(),
             'luma':float(gray.mean()),'edge_energy':float(np.mean(abs(cv2.Laplacian(gray,cv2.CV_32F))))}
        if i:
            a,b=frames[i-1],f
            warp=sample(b,flow(a,b));error=abs(a.astype(float)-warp.astype(float))
            row['registered_mae']=float(error.mean())
            row['registered_p95']=float(np.percentile(error.mean(-1),95))
            row['raw_mae']=float(abs(a.astype(float)-b.astype(float)).mean())
            ag=cv2.GaussianBlur(cv2.cvtColor(a,cv2.COLOR_RGB2GRAY),(0,0),1).astype(float)
            changed=abs(gray-ag)
            row['held']=bool(changed.mean()<=1.25 and np.percentile(changed,99)<=6 and np.mean(changed>5)<=.016)
            row['camera1']=track(a,b)
        if i>=2:row['camera2']=track(frames[i-2],f,2)
        if i>=4:row['camera4']=track(frames[i-4],f,4)
        result.append(row)
    return result

def strip(original,final,seam,start):
    n,lo,hi=seam['frame'],seam['bridge_start'],seam['bridge_end']
    indices=[lo-2,lo,lo+1,lo+2,n-1,n,n+1,hi-2,hi-1,hi,hi+1,hi+2]
    canvas=Image.new('RGB',(1920,820),'#121620');draw=ImageDraw.Draw(canvas)
    for group in range(2):
        for version,frames in enumerate((original,final)):
            row=group*2+version
            for col,index in enumerate(indices[group*6:group*6+6]):
                canvas.paste(Image.fromarray(frames[index-start]).resize((320,180)),(col*320,row*205))
                label=f'{"SOURCE" if version==0 else "BRIDGE"} {index}'+(' / anchor' if index in (lo,hi) else '')
                draw.text((col*320+6,row*205+185),label,fill='#ffd579' if index in(lo,hi) else 'white')
    canvas.save(OUT/f'edges-{n}.jpg',quality=95)

def plot(rows,seam):
    original,final=rows
    n,lo,hi=seam['frame'],seam['bridge_start'],seam['bridge_end']
    fields=[('registered_mae','Registered difference'),('camera2.scale_pct_per_frame','Camera scale % / frame (2-frame estimate)'),
            ('camera2.dy_per_frame','Vertical center motion px / frame'),('edge_energy','Edge energy'),('luma','Mean luminance')]
    image=Image.new('RGB',(1280,900),'#121722');draw=ImageDraw.Draw(image)
    xmin,xmax=original[0]['frame'],original[-1]['frame']
    for row,(key,title) in enumerate(fields):
        top=28+row*175;bottom=top+128;left=65;right=1250;series=[];values=[]
        for records in rows:
            samples=[]
            for item in records:
                value=item
                for p in key.split('.'):value=value.get(p) if isinstance(value,dict) else None
                if value is not None:samples.append((item['frame'],value));values.append(value)
            series.append(samples)
        ymin,ymax=min(values),max(values);pad=max((ymax-ymin)*.08,.005);ymin-=pad;ymax+=pad
        xy=lambda x,y:(left+(right-left)*(x-xmin)/(xmax-xmin),bottom-(bottom-top)*(y-ymin)/(ymax-ymin))
        draw.rectangle((xy(lo,ymax)[0],top,xy(hi,ymin)[0],bottom),fill='#202d3a')
        for x,color in ((lo,'#679676'),(n,'#606c7a'),(hi,'#679676')):
            draw.line((xy(x,ymin)[0],top,xy(x,ymin)[0],bottom),fill=color)
        if ymin<0<ymax:draw.line((left,xy(n,0)[1],right,xy(n,0)[1]),fill='#354457')
        draw.text((left,top-20),title,fill='white');draw.text((2,top),f'{ymax:.2f}',fill='gray');draw.text((2,bottom-10),f'{ymin:.2f}',fill='gray')
        for samples,color in zip(series,('#ffbb68','#66daf7')):draw.line([xy(x,y) for x,y in samples],fill=color,width=2)
        for x in (xmin,lo,n,hi,xmax):draw.text((xy(x,ymin)[0]-12,bottom+5),str(x),fill='gray')
    draw.text((850,4),'SOURCE orange / ENCODED BRIDGE cyan',fill='white')
    image.save(OUT/f'curves-{n}.png')

def summarize(seam,original,final):
    n,lo,hi=seam['frame'],seam['bridge_start'],seam['bridge_end'];a={r['frame']:r for r in original};b={r['frame']:r for r in final}
    baseline=[r['registered_mae'] for r in original if 'registered_mae' in r and r['frame']!=n]
    threshold=max(2.5*float(np.median(baseline)),float(np.percentile(baseline,90))+1.5)
    edge_frames=[lo,lo+1,lo+2,hi-1,hi,hi+1]
    edges=[]
    for index in edge_frames:
        ar,br=a[index],b[index]
        edges.append({'frame':index,'source_registered':ar['registered_mae'],'bridge_registered':br['registered_mae'],
                      'bridge_luma_step':br['luma']-b[index-1]['luma'],
                      'source_luma_step':ar['luma']-a[index-1]['luma'],
                      'bridge_edge_change_pct':100*(br['edge_energy']/b[index-1]['edge_energy']-1),
                      'strong_new_registered_spike':br['registered_mae']>threshold and br['registered_mae']>ar['registered_mae']+.75})
    interior=[b[i] for i in range(lo+1,hi)]
    return {'frame':n,'start':lo,'end':hi,'edges':edges,'registered_outlier_threshold':threshold,
            'source_join_registered':a[n]['registered_mae'],'bridge_join_registered':b[n]['registered_mae'],
            'source_held_intervals_in_bridge':[i for i in range(lo+1,hi+1) if a[i]['held']],
            'bridge_held_intervals_in_bridge':[i for i in range(lo+1,hi+1) if b[i]['held']],
            'largest_bridge_luma_steps':sorted([{'frame':i,'delta':b[i]['luma']-b[i-1]['luma']} for i in range(lo+1,hi+1)],key=lambda x:abs(x['delta']),reverse=True)[:3],
            'largest_bridge_registered':sorted([{'frame':r['frame'],'value':r['registered_mae']} for r in interior],key=lambda r:r['value'],reverse=True)[:3],
            'edge_energy_ratio_min':min(r['edge_energy']/a[r['frame']]['edge_energy'] for r in interior),
            'edge_energy_ratio_max':max(r['edge_energy']/a[r['frame']]['edge_energy'] for r in interior)}

def main():
    OUT.mkdir(exist_ok=True)
    plan=json.loads((ROOT/'plans/IYTYT-bridges.json').read_text())
    seams=sorted(plan['seams'],key=lambda s:([361,2166,3240,2888].index(s['frame']) if s['frame'] in [361,2166,3240,2888] else 5+s['frame']))
    data=[];summary=[]
    for seam in seams:
        lo,hi=seam['bridge_start'],seam['bridge_end'];start=lo-8;count=hi-lo+17
        cache=OUT/f'proxy-{seam["frame"]}.npz'
        if cache.exists():
            cached=np.load(cache);original,final=cached['original'],cached['final']
        else:
            # Decode actual native frames before making equal analysis proxies.
            source=read_frames(ROOT/'IYTYT.mp4',start,count)
            encoded=read_frames(ROOT/'output/IYTYT-bridged.mp4',start,count)
            original=np.stack([cv2.resize(f,(640,360),interpolation=cv2.INTER_AREA) for f in source])
            final=np.stack([cv2.resize(f,(640,360),interpolation=cv2.INTER_AREA) for f in encoded])
            for index in (lo,lo+1,seam['frame'],hi-1,hi):
                pair=np.concatenate((source[index-start],encoded[index-start]),axis=1)
                Image.fromarray(pair).save(OUT/f'native-{seam["frame"]}-{index}.jpg',quality=96)
            del source,encoded
            np.savez_compressed(cache,original=original,final=final)
        a,b=metrics(original,start),metrics(final,start)
        data.append({'seam':seam,'source':a,'bridge':b});summary.append(summarize(seam,a,b))
        strip(original,final,seam,start);plot((a,b),seam)
        (OUT/'measurements.json').write_text(json.dumps(data,indent=2));(OUT/'summary.json').write_text(json.dumps(summary,indent=2))
        s=summary[-1]
        print('AUDITED',seam['frame'],'cut registered',round(s['source_join_registered'],2),'->',round(s['bridge_join_registered'],2),'new edgespikes',[r['frame'] for r in s['edges'] if r['strong_new_registered_spike']],'edgeenergy',round(s['edge_energy_ratio_min'],3),round(s['edge_energy_ratio_max'],3),flush=True)

if __name__=='__main__':main()
