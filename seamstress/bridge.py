"""Reconstruct short, endpoint-constrained motion bridges using RIFE 4.25."""
from __future__ import annotations
import json
import os
from pathlib import Path
import tempfile
import numpy as np
from .media import probe,read_frames,iter_frames,VideoWriter,mux_audio
from .repair import fingerprint

WEIGHTS_SHA256='6615790efd627772917205db291f51cd392528a157ecbb2ecaeec3bff8eb6de2'


def _cubic_reconstruct(left,right,flow,mask_logits):
    """Reconstruct the same learned gathers with cubic sampling and soft masks.

    Inputs are padded float RGB endpoints in [0,1], H×W×4 gathers, and H×W
    mask logits. Replicated padded-image borders match RIFE's grid_sample
    padding_mode='border'; no reflected endpoint pixels are introduced.
    """
    import cv2
    if left.shape!=right.shape or left.ndim!=3 or left.shape[2]!=3:
        raise ValueError('Cubic reconstruction needs matching RGB endpoints.')
    h,w=left.shape[:2]
    if flow.shape!=(h,w,4) or mask_logits.shape!=(h,w):
        raise ValueError('Flow and mask dimensions must match padded endpoints.')
    if not np.all(np.isfinite(flow)) or not np.all(np.isfinite(mask_logits)):
        raise ValueError('Interpolator returned nonfinite flow or mask values.')
    yy,xx=np.mgrid[:h,:w].astype(np.float32)
    a=cv2.remap(left,xx+flow[...,0],yy+flow[...,1],cv2.INTER_CUBIC,borderMode=cv2.BORDER_REPLICATE)
    b=cv2.remap(right,xx+flow[...,2],yy+flow[...,3],cv2.INTER_CUBIC,borderMode=cv2.BORDER_REPLICATE)
    # A factor of two strengthens confident source ownership while retaining
    # a continuous blend near ambiguous occlusions. It cannot fix wrong flow.
    mask=1/(1+np.exp(-np.clip(mask_logits*2,-50,50)))
    return np.clip(a*mask[...,None]+b*(1-mask[...,None]),0,1)


def progress_curve(count, start_slope=1., end_slope=1.):
    """Monotone cubic Hermite timing; limit derivatives to prevent reversals."""
    if count<2:raise ValueError('A bridge needs two endpoints.')
    slopes=np.asarray([start_slope,end_slope],dtype=float)
    if not np.all(np.isfinite(slopes)) or np.any(slopes<0):raise ValueError('Progress slopes must be finite and nonnegative.')
    norm=float(np.linalg.norm(slopes))
    if norm>3:slopes*=3/norm
    a,b=slopes;t=np.linspace(0,1,count)
    result=(-2*t**3+3*t*t)+(t**3-2*t*t+t)*a+(t**3-t*t)*b
    result[0]=0;result[-1]=1
    return np.clip(result,0,1)


class RifeModel:
    def __init__(self,weights,device='auto',reconstruction='baseline'):
        if reconstruction not in ('baseline','cubic'):raise ValueError('reconstruction must be baseline or cubic.')
        self.reconstruction=reconstruction
        weights=Path(weights)
        if not weights.is_file():raise FileNotFoundError(f'RIFE weights missing: {weights}. See README setup instructions.')
        if fingerprint(weights)!=WEIGHTS_SHA256:raise ValueError('RIFE checkpoint checksum mismatch; expected official 4.25 weights.')
        try:
            import torch
        except ImportError as exc:raise RuntimeError('Install the bridge extra: pip install -e ".[bridge]"') from exc
        from ._vendor.rife.ifnet import IFNet
        self.torch=torch
        if device=='auto':device='cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        if device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable in this process; choose another device.')
        if device=='mps' and not torch.backends.mps.is_available():raise RuntimeError('MPS unavailable in this process; choose CPU or run with GPU access.')
        self.device=torch.device(device);torch.set_num_threads(4)
        self.model=IFNet().eval().to(self.device)
        checkpoint=torch.load(weights,map_location='cpu',weights_only=True)
        state={k.removeprefix('module.'):v for k,v in checkpoint.items()}
        state={k:v for k,v in state.items() if not k.startswith(('teacher.','caltime.'))}
        self.model.load_state_dict(state,strict=True)

    def synthesize(self,left,right,progress):
        torch=self.torch
        if left.shape!=right.shape or left.ndim!=3 or left.shape[2]!=3 or left.dtype!=np.uint8 or right.dtype!=np.uint8:raise ValueError('Endpoints must be matching RGB uint8 frames.')
        progress=np.asarray(progress,dtype=float)
        if progress.ndim!=1 or len(progress)<3 or not np.all(np.isfinite(progress)) or progress[0]!=0 or progress[-1]!=1 or np.any(np.diff(progress)<=0):raise ValueError('Progress must strictly increase from 0 to 1.')
        h,w=left.shape[:2]
        def tensor(x):return torch.from_numpy(x.copy()).permute(2,0,1)[None].float().to(self.device)/255
        pad=(0,(-w)%64,0,(-h)%64)
        a=torch.nn.functional.pad(tensor(left),pad);b=torch.nn.functional.pad(tensor(right),pad)
        pair=torch.cat([a,b],dim=1);out=[left.copy()]
        if self.reconstruction=='cubic':
            padded_left=a[0].permute(1,2,0).cpu().numpy()
            padded_right=b[0].permute(1,2,0).cpu().numpy()
        with torch.inference_mode():
            for t in progress[1:-1]:
                flows,logits,merged=self.model(pair,float(t),[16,8,4,2,1])
                if self.reconstruction=='cubic':
                    field=flows[-1][0].permute(1,2,0).cpu().numpy()
                    mask_logits=logits[0,0].cpu().numpy()
                    frame=_cubic_reconstruct(padded_left,padded_right,field,mask_logits)[:h,:w]
                    out.append((frame*255).round().astype(np.uint8))
                else:
                    frame=merged[-1][0,:,:h,:w].permute(1,2,0).clamp(0,1)
                    out.append((frame.cpu().numpy()*255).round().astype(np.uint8))
        out.append(right.copy())
        return out


def validate_bridge_plan(plan,meta):
    if not isinstance(plan,dict):raise ValueError('Bridge plan must be a JSON object.')
    if plan.get('schema_version')!=2 or plan.get('method')!='rife_bridge':raise ValueError('Expected schema_version 2, method rife_bridge.')
    for key in ('width','height','frame_count','fps_fraction'):
        if plan.get('source',{}).get(key)!=meta[key]:raise ValueError(f'Source {key} differs from plan.')
    import re
    if not isinstance(plan.get('source_sha256'),str) or not re.fullmatch('[0-9a-f]{64}',plan['source_sha256']):raise ValueError('Bridge plan needs source SHA-256 fingerprint.')
    if not isinstance(plan.get('seams'),list):raise ValueError('seams must be a list.')
    last=-1
    for s in plan['seams']:
        if not isinstance(s,dict):raise ValueError('Each seam must be an object.')
        if type(s.get('camera_control',True)) is not bool:raise ValueError('camera_control must be boolean.')
        lo,hi,n=s.get('bridge_start'),s.get('bridge_end'),s.get('frame')
        if any(type(v) is not int for v in (lo,hi,n)):raise ValueError('Bridge indices must be integers.')
        if not 0<=lo<n<=hi<meta['frame_count'] or hi-lo<2:raise ValueError('Invalid bridge endpoints.')
        if s.get('camera_control',True) and (lo==0 or hi==meta['frame_count']-1):raise ValueError('Camera bridges require source handles on both sides.')
        if not isinstance(s.get('time'),(int,float)) or not np.isfinite(s['time']) or abs(s['time']-n/meta['fps'])>1/meta['fps']:raise ValueError('Bridge time does not match frame index.')
        if lo<=last:raise ValueError('Bridge windows must be ordered and cannot overlap.')
        if hi-lo>meta['fps']*3:raise ValueError('Bridge exceeds three seconds; shorter local windows required.')
        progress_curve(hi-lo+1,s.get('start_slope',1.),s.get('end_slope',1.))
        last=hi


def render_bridges(input,plan,output,weights,device='auto',crf=16):
    input,plan,output=map(Path,(input,plan,output));meta=probe(input)
    if output.resolve()==input.resolve() or output.exists() or output.with_suffix('.repair.json').exists():raise ValueError('Choose a new output path distinct from source and existing sidecars.')
    p=json.loads(plan.read_text());validate_bridge_plan(p,meta)
    if fingerprint(input)!=p.get('source_sha256'):raise ValueError('Source fingerprint differs from bridge plan.')
    output.parent.mkdir(parents=True,exist_ok=True)
    model=RifeModel(weights,device);reports=[]
    with tempfile.TemporaryDirectory(prefix='seamstress-bridge-',dir=output.parent) as tmp:
        tmp=Path(tmp);patches={}
        for s in p['seams']:
            lo,hi=s['bridge_start'],s['bridge_end']
            handle_lo=max(0,lo-4);handle_hi=min(meta['frame_count']-1,hi+4)
            neighborhood=read_frames(input,handle_lo,handle_hi-handle_lo+1)
            source=neighborhood[lo-handle_lo:hi-handle_lo+1]
            progress=progress_curve(len(source),s.get('start_slope',1.),s.get('end_slope',1.))
            camera_report=None
            if s.get('camera_control',True):
                from .camera_bridge import synthesize_camera_bridge
                frames,camera_report=synthesize_camera_bridge(model,source[0],source[-1],neighborhood[:lo-handle_lo],neighborhood[hi-handle_lo+1:],progress)
            else:
                frames=model.synthesize(source[0],source[-1],progress)
            if len(frames)!=len(source) or any(f.shape!=source[0].shape or f.dtype!=np.uint8 for f in frames):raise RuntimeError('Bridge frame count, dimensions, or RGB type changed.')
            if not np.array_equal(frames[0],source[0]) or not np.array_equal(frames[-1],source[-1]):raise RuntimeError('Bridge endpoint integrity failed.')
            path=tmp/f'{s["frame"]}.npy';np.save(path,np.stack(frames))
            for i in range(lo+1,hi):patches[i]=(path,i-lo)
            reports.append({'frame':s['frame'],'time':s['time'],'start':lo,'end':hi,'synthesized_frames':hi-lo-1,'endpoints_exact':True,'progress':progress.tolist(),'camera_control':camera_report})
            print(f'Synthesized {hi-lo-1} frames across boundary {s["frame"]} ({s["time"]:.3f}s)',flush=True)
        silent=tmp/'video.mp4';cache={}
        with VideoWriter(silent,meta['width'],meta['height'],meta['fps_fraction'],crf=crf) as writer:
            for i,frame in enumerate(iter_frames(input)):
                if i in patches:
                    path,j=patches[i]
                    if path not in cache:cache={path:np.load(path,mmap_mode='r')}
                    frame=cache[path][j]
                writer.write(frame)
        complete=tmp/'complete.mp4';mux_audio(silent,input,complete);os.link(complete,output)
    report={'output':str(output.resolve()),'source_sha256':p['source_sha256'],'method':'RIFE 4.25 endpoint-constrained motion bridges',
            'model_sha256':WEIGHTS_SHA256,'device':str(model.device),'frame_count':meta['frame_count'],
            'crop_fraction':0.,'seams':reports,'requires_visual_review':True}
    report_path=output.with_suffix('.repair.json');report_path.write_text(json.dumps(report,indent=2))
    return {k:v for k,v in report.items() if k!='seams'}|{'seam_count':len(reports),'report_path':str(report_path.resolve())}
