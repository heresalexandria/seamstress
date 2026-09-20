"""Source inspection, explicit repair plans, and output-based quality reports."""
from __future__ import annotations
import json
from pathlib import Path
import cv2
import numpy as np
from scipy.ndimage import median_filter
from PIL import Image,ImageDraw
from .media import probe,iter_frames,read_frames,VideoWriter
from .repair import fingerprint,flow,sample,crop_frame


def pair_metrics(a,b):
    f=flow(a,b);warped=sample(b,f)
    delta=np.abs(warped.astype(np.float32)-a.astype(np.float32))
    ga=cv2.cvtColor(a,cv2.COLOR_RGB2GRAY).astype(np.float32)
    gb=cv2.cvtColor(b,cv2.COLOR_RGB2GRAY).astype(np.float32)
    return {'raw_mae':float(np.mean(abs(a.astype(np.float32)-b.astype(np.float32)))),
            'registered_mae':float(np.mean(delta)),
            'registered_p90':float(np.percentile(delta,90)),
            'mean_rgb_step':(b.astype(np.float32).mean((0,1))-a.astype(np.float32).mean((0,1))).tolist(),
            'motion_p50':float(np.median(np.linalg.norm(f,axis=2))),
            'motion_p90':float(np.percentile(np.linalg.norm(f,axis=2),90)),
            'sharpness_left':float(np.mean(abs(cv2.Laplacian(ga,cv2.CV_32F)))),
            'sharpness_right':float(np.mean(abs(cv2.Laplacian(gb,cv2.CV_32F)))),
            'sharpness_step':float(np.mean(abs(cv2.Laplacian(gb,cv2.CV_32F)))-np.mean(abs(cv2.Laplacian(ga,cv2.CV_32F))))}


def detect(input,meta,interval):
    import tempfile
    with tempfile.TemporaryDirectory(prefix='seamstress-scan-') as tmp:
        proxies=np.memmap(Path(tmp)/'frames.rgb',mode='w+',dtype=np.uint8,
                          shape=(meta['frame_count'],180,320,3))
        signals=[];prev=None;means=[]
        for i,frame in enumerate(iter_frames(input,size=(320,180))):
            if i>=len(proxies):raise ValueError('Decoded frame count exceeds metadata; constant frame rate input required.')
            proxies[i]=frame
            low=cv2.GaussianBlur(frame,(0,0),1).astype(np.float32)
            signals.append(0 if prev is None else float(np.mean(abs(low-prev))))
            means.append(low.mean((0,1)));prev=low
        raw=np.array(signals)
        baseline=np.array([np.percentile(raw[max(0,i-24):min(len(raw),i+25)],75) for i in range(len(raw))])
        color=np.r_[0,np.linalg.norm(np.diff(means,axis=0),axis=1)]
        score=raw/np.maximum(baseline,.5)+.6*color
        seams=[];expected=interval*meta['fps'];position=expected
        while position<len(raw)-8:
            lo=max(8,round(position-meta['fps']*.9));hi=min(len(raw)-8,round(position+meta['fps']*.9))
            ranking=[]
            # A continuation can restart nearly stationary while its preceding
            # camera move is fast. Raw-difference screening would discard that
            # real seam, and unnormalized residual would favor disocclusions in
            # the legitimate camera move. Inspect every cached neighboring pair
            # and normalize by motion measured at this proxy resolution.
            motion_scale=.15*(480/proxies.shape[2])
            for n in range(lo,hi):
                m=pair_metrics(proxies[n-1],proxies[n])
                residual=m['registered_mae']/(1+motion_scale*m['motion_p90'])
                ranking.append((residual+.15*score[n],n))
            if not ranking:break
            _,n=max(ranking);seams.append(n);position=n+expected
        del proxies
    return seams,{'raw_difference':raw.tolist(),'candidate_score':score.tolist()}


def contact(input,seams,path):
    w,h=400,225
    sheet=Image.new('RGB',(w*4,(h+28)*max(1,len(seams))), '#151922');d=ImageDraw.Draw(sheet)
    if not seams:d.text((30,100),'No candidate joins found. Supply --seams to select frame boundaries manually.',fill='white')
    for row,n in enumerate(seams):
        frames=read_frames(input,n-2,4,size=(w,h))
        for col,f in enumerate(frames):
            sheet.paste(Image.fromarray(f),(w*col,(h+28)*row))
            d.text((w*col+8,(h+28)*row+h+6),f'frame {n-2+col}'+(' / JOIN' if col==2 else ''),fill='white')
    sheet.save(path)


def analyze(input:Path,work_dir:Path,interval:float=15.,seams:list[int]|None=None):
    input,work_dir=Path(input),Path(work_dir);work_dir.mkdir(parents=True,exist_ok=True)
    plan_path=work_dir/'plan.json'
    if plan_path.exists():raise ValueError('Plan already exists; use a new work directory.')
    meta=probe(input)
    if not meta['frame_count'] or meta['frame_count']<10:raise ValueError('Video too short.')
    automatic=seams is None
    signals=None
    if seams is None:seams,signals=detect(input,meta,interval)
    seams=sorted(set(seams))
    if any(n<5 or n>=meta['frame_count']-5 for n in seams):raise ValueError('Seams must have at least five neighboring frames on each side.')
    records=[]
    for n in seams:
        frames=read_frames(input,n-5,10,size=(640,360))
        metrics=pair_metrics(frames[4],frames[5])
        records.append({'frame':n,'time':n/meta['fps'],'enabled':True,'before':14,'after':22,'metrics':metrics})
        print(f'Analyzed boundary {n} ({n/meta["fps"]:.3f}s)',flush=True)
    plan={'schema_version':1,'source':{'path':str(input.resolve()),**meta},
          'source_sha256':fingerprint(input),'detection':'automatic candidates; inspect contact sheet' if automatic else 'explicit frame indices',
          'config':{'flow_width':min(640,meta['width']),'max_displacement':.045,'color_spatial_sigma':24,
                    'local_color_strength':.8,'geometry_strength':1.,'color_strength':1.,
                    'crop_fraction':0.,'max_crop_fraction':.04},'seams':records,
          'notes':['Frame indices are zero based, and identify the first frame after the edit.',
                   'Candidate detection is heuristic, not proof of an edit.',
                   'No image-generation service is called. Reproduction uses only local deterministic processing.',
                   'Output quality requires visual review; numerical improvement is not perceptual certification.']}
    plan_path.write_text(json.dumps(plan,indent=2))
    contact(input,seams,work_dir/'boundaries.jpg')
    if signals:(work_dir/'scan.json').write_text(json.dumps(signals))
    return {'plan_path':str(plan_path.resolve()),'seam_count':len(seams),'seams':seams,'contact_sheet':str((work_dir/'boundaries.jpg').resolve())}


def verify(input:Path,output:Path,plan:Path,work_dir:Path):
    input,output,plan,work_dir=map(Path,(input,output,plan,work_dir));work_dir.mkdir(parents=True,exist_ok=True)
    p=json.loads(plan.read_text());before=probe(input);after=probe(output)
    if fingerprint(input)!=p['source_sha256']:raise ValueError('Source does not match repair plan.')
    repair_report=json.loads(output.with_suffix('.repair.json').read_text()) if output.with_suffix('.repair.json').exists() else {}
    if repair_report.get('preview'):
        raise ValueError('verify requires a full-timeline render; preview frame indices have a source offset.')
    crop=repair_report.get('crop_fraction',0.)
    metric_view=np.asarray(p.get('view_matrix',np.eye(3)),dtype=float) if p.get('method')=='source_conform' else None
    if metric_view is not None:
        crop=max(0.,(1-1/np.sqrt(np.linalg.det(metric_view[:2,:2])))/2)
    results=[];w,h=480,270
    seams=[dict(s,time=s.get('time',s['frame']/before['fps'])) for s in p.get('seams',[])]
    review_seams=seams or [{'frame':min(30,before['frame_count']//2),'time':0.0}]
    montage=Image.new('RGB',(w*4,(h+26)*len(review_seams)), '#141820');draw=ImageDraw.Draw(montage)
    comparison=work_dir/'comparison.mp4'
    if comparison.exists():raise ValueError('Verification comparison exists; use a new directory.')
    with VideoWriter(comparison,w*2,h,before['fps_fraction'],crf=16,preset='fast') as writer:
        for row,seam in enumerate(review_seams):
            n=seam['frame'];lo=max(0,n-30);hi=min(before['frame_count'],n+42)
            original_full=read_frames(input,lo,hi-lo)
            a=np.stack([cv2.resize(f,(w,h),interpolation=cv2.INTER_AREA) for f in original_full])
            # Apply framing at source resolution before identical downsampling.
            def reference_view(f):
                if metric_view is not None:
                    return cv2.warpAffine(f,metric_view[:2],(f.shape[1],f.shape[0]),flags=cv2.INTER_CUBIC,borderMode=cv2.BORDER_REPLICATE)
                return crop_frame(f,crop)
            ac=np.stack([cv2.resize(reference_view(f),(w,h),interpolation=cv2.INTER_AREA) for f in original_full])
            del original_full
            b=np.stack([cv2.resize(f,(w,h),interpolation=cv2.INTER_AREA) for f in read_frames(output,lo,hi-lo)])
            k=n-lo
            ma=pair_metrics(ac[k-1],ac[k]);mb=pair_metrics(b[k-1],b[k])
            # Check the entire window, including new discontinuities at patch edges.
            pairs_a=[pair_metrics(ac[j-1],ac[j]) for j in range(1,len(ac))]
            pairs_b=[pair_metrics(b[j-1],b[j]) for j in range(1,len(b))]
            ca=[m['registered_mae'] for m in pairs_a];cb=[m['registered_mae'] for m in pairs_b]
            result={'frame':n,'time':seam['time'],'before':ma,'after':mb,
                    'registered_error_reduction':1-mb['registered_mae']/max(ma['registered_mae'],.001),
                    'window_max_registered_before':max(ca),'window_max_registered_after':max(cb),
                    'window_curve_before':ca,'window_curve_after':cb,
                    'motion_curve_before':[m['motion_p50'] for m in pairs_a],
                    'motion_curve_after':[m['motion_p50'] for m in pairs_b],
                    'sharpness_retained':(mb['sharpness_left']+mb['sharpness_right'])/max(ma['sharpness_left']+ma['sharpness_right'],.001),
                    'needs_review':True}
            if seams:results.append(result)
            for col,f in enumerate((a[k-1],a[k],b[k-1],b[k])):
                montage.paste(Image.fromarray(f),(col*w,row*(h+26)))
                label=['Original last','Original first','Candidate last','Candidate first'][col]
                draw.text((col*w+8,row*(h+26)+h+5),f'{seam["time"]:.3f}s / {label}',fill='white')
            for i in range(len(a)):
                pair=np.concatenate((a[i],b[i]),axis=1).copy()
                cv2.putText(pair,f'ORIGINAL   {seam["time"]:.3f}s',(12,24),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),1,cv2.LINE_AA)
                cv2.putText(pair,'CANDIDATE',(w+12,24),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),1,cv2.LINE_AA)
                writer.write(pair)
            print(f'Verified boundary {n}: residual {ma["registered_mae"]:.2f} -> {mb["registered_mae"]:.2f}',flush=True)
    montage.save(work_dir/'before-after.jpg')
    from .review import write_review
    review_path=write_review(input,output,p,work_dir/'review.html')
    import subprocess
    def audio_hash(path):
        if not probe(path)['has_audio']:return None
        r=subprocess.run(['ffmpeg','-v','error','-i',str(path),'-map','0:a:0','-c:a','copy','-f','hash','-hash','sha256','-'],capture_output=True,check=True)
        return r.stdout.decode().strip()
    report={'audio_bitstream_preserved':audio_hash(input)==audio_hash(output),'review_player':review_path,'source':str(input.resolve()),'output':str(output.resolve()),
            'frame_count_preserved':before['frame_count']==after['frame_count'],
            'fps_preserved':abs(before['fps']-after['fps'])<1e-8,
            'dimensions_preserved':(before['width'],before['height'])==(after['width'],after['height']),
            'duration_difference':after['duration']-before['duration'],
            'has_audio':after['has_audio'],'crop_fraction':crop,'seams':results,
            'metric_reference_view_matrix':metric_view.tolist() if metric_view is not None else None,
            'plan_status':p.get('status'),'unresolved_seams':p.get('unresolved_seams',[]),
            'perceptual_status':'Needs normal-speed visual review; metrics do not certify invisible seams.',
            'comparison':str(comparison.resolve())}
    (work_dir/'report.json').write_text(json.dumps(report,indent=2))
    return {k:v for k,v in report.items() if k!='seams'}|{'report_path':str((work_dir/'report.json').resolve()),'seam_error_reductions':[round(r['registered_error_reduction'],3) for r in results]}
