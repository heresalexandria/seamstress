"""Motion-aware, transported elastic and photometric seam correction.

Each frame retains its own time sample. Dense *excess* displacement at an edit is
split between the two sides and transported with the scene before fading out.
"""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path
import tempfile
import cv2
import numpy as np

cv2.setNumThreads(4)


def fingerprint(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def grid(shape):
    h, w = shape[:2]
    return np.stack(np.meshgrid(np.arange(w), np.arange(h)), -1).astype(np.float32)


def sample(image, displacement, interpolation=cv2.INTER_LINEAR):
    coords = grid(displacement.shape) + displacement
    return cv2.remap(image, coords[..., 0], coords[..., 1], interpolation,
                     borderMode=cv2.BORDER_REFLECT_101)


def gray(image):
    g = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # A fixed local contrast transform tolerates modest grading drift.
    return cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8)).apply(g)


def flow(a, b):
    estimator = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    estimator.setFinestScale(0)
    estimator.setGradientDescentIterations(32)
    estimator.setVariationalRefinementIterations(8)
    return estimator.calc(gray(a), gray(b), None)


def resize_flow(field, size):
    h, w = field.shape[:2]
    out = cv2.resize(field, size, interpolation=cv2.INTER_CUBIC)
    out[..., 0] *= size[0] / w
    out[..., 1] *= size[1] / h
    return out


def smoothstep(x):
    x = np.clip(x, 0., 1.)
    return x*x*x*(10 + x*(-15 + 6*x))


def robust_color(source, target, valid):
    """Bounded diagonal color fit with iterative outlier rejection."""
    s = cv2.GaussianBlur(source.astype(np.float32), (0, 0), 1.2)
    t = cv2.GaussianBlur(target.astype(np.float32), (0, 0), 1.2)
    mask = valid & (np.max(abs(s-t), axis=2) < 45)
    gains, biases = [], []
    for c in range(3):
        x, y = s[...,c][mask][::3], t[...,c][mask][::3]
        if len(x) < 100 or np.std(x) < 8:
            gains.append(1.); biases.append(float(np.median(y-x)) if len(x) else 0.)
            continue
        A = np.column_stack((x, np.ones_like(x)))
        coef = np.array([1., 0.])
        for _ in range(5):
            err = y-A@coef
            weights = np.minimum(1., 5/np.maximum(abs(err), .01))
            coef = np.linalg.lstsq(A*weights[:,None],y*weights,rcond=None)[0]
            coef[0] = np.clip(coef[0], .80, 1.25)
            coef[1] = np.clip(coef[1], -25, 25)
        gains.append(float(coef[0])); biases.append(float(coef[1]))
    return np.array(gains,np.float32), np.array(biases,np.float32)


def build_fields(frames, cut, config):
    """Return correction fields at the two edit anchors, in analysis pixels."""
    a, b = frames[cut-1], frames[cut]
    forward, backward = flow(a,b), flow(b,a)
    fb = np.linalg.norm(forward + sample(backward,forward), axis=2)
    bf = np.linalg.norm(backward + sample(forward,backward), axis=2)
    # Estimate actual animation velocity over several frames (handles on-twos).
    span = min(4,cut-1,len(frames)-cut-1)
    va = -flow(a,frames[cut-1-span])/max(1,span)
    vb = flow(b,frames[cut+span])/max(1,span)
    expected_a = .5*(va+sample(vb,forward))
    expected_b = .5*(vb+sample(va,backward))
    excess_a = forward-expected_a
    excess_b = backward+expected_b
    # A mild vector-field regularizer discourages rubbery linework.
    excess_a = cv2.GaussianBlur(excess_a,(0,0),1.0)
    excess_b = cv2.GaussianBlur(excess_b,(0,0),1.0)
    limit = config.get('max_displacement', .045)*a.shape[1]
    for v in (excess_a,excess_b):
        mag=np.linalg.norm(v,axis=2)
        v *= np.minimum(1.,limit/np.maximum(mag,.01))[...,None]
    # Uncertain disocclusions retain smoothly extrapolated neighboring motion.
    for v,e in ((excess_a,fb),(excess_b,bf)):
        confidence = np.exp(-np.square(e/2.5)).astype(np.float32)
        numerator = cv2.GaussianBlur(v*confidence[...,None],(0,0),4)
        denominator = cv2.GaussianBlur(confidence,(0,0),4)
        filled = numerator/np.maximum(denominator[...,None], .03)
        mix=np.clip(confidence,.1,1)[...,None]
        v[:] = mix*v+(1-mix)*filled
    # Gather maps point opposite the desired movement of content.
    left_map = -.5*excess_a
    right_map = -.5*excess_b
    left_half=sample(a,left_map).astype(np.float32)
    right_half=sample(b,right_map).astype(np.float32)
    valid=(sample(fb,left_map)<2.5)&(sample(bf,right_map)<2.5)
    gain,bias=robust_color(right_half,left_half,valid)
    # Invert the half transform for the outgoing side; average both grades.
    # Fit broad spatial color drift after correspondence, avoiding outline ghosts.
    aligned_b=sample(b,forward).astype(np.float32)
    residual=a.astype(np.float32)-(aligned_b*gain+bias)
    sigma=config.get('color_spatial_sigma',24)
    weight=np.exp(-np.square(fb/2.5)).astype(np.float32)
    low=cv2.GaussianBlur(residual*weight[...,None],(0,0),sigma)
    low/=np.maximum(cv2.GaussianBlur(weight,(0,0),sigma)[...,None],.05)
    low=np.clip(low,-12,12)*config.get('local_color_strength',.8)
    left_color = -.5*low/gain
    right_color = .5*sample(low,backward)
    return {
      'left_map':left_map,'right_map':right_map,
      'left_tangent':va-expected_a,'right_tangent':vb-expected_b,
      'left_color':left_color,'right_color':right_color,
      'gain':gain,'bias':bias,
      'forward':forward,'backward':backward,
      'diagnostics':{'flow_consistent_fraction':float(np.mean((fb<2)&(bf<2))),
      'excess_displacement_p95':float(np.percentile(np.linalg.norm(excess_a,axis=2),95)),
      'expected_motion_p50':float(np.median(np.linalg.norm(expected_a,axis=2))),
      'color_gain':gain.tolist(),'color_bias':bias.tolist(),
      'color_residual_mae':float(np.mean(abs(residual))),
      'field_limit_hit_fraction':float(np.mean(np.linalg.norm(excess_a,axis=2)>.98*limit))}}


def transport_vector(field, motion):
    """Pull back a displacement vector through the current-to-anchor Jacobian."""
    pulled=sample(field,motion)
    smoothed=cv2.GaussianBlur(motion,(0,0),1.5)
    uy,ux=np.gradient(smoothed[...,0]);vy,vx=np.gradient(smoothed[...,1])
    a,b,c,d=1+ux,uy,vx,1+vy
    det=a*d-b*c
    valid=(det>.3)&(det<3.)
    denominator=np.where(valid,det,1.)
    result=np.stack(((d*pulled[...,0]-b*pulled[...,1])/denominator,
                     (-c*pulled[...,0]+a*pulled[...,1])/denominator),axis=-1)
    bounded=np.linalg.norm(result,axis=2)<np.maximum(1.,np.linalg.norm(pulled,axis=2)*2.)
    return np.where((valid&bounded)[...,None],result,pulled).astype(np.float32)


def safe_mapping(mapping, floor=.25):
    """Regularize a gather map until its local Jacobian cannot fold linework."""
    def determinant(v):
        uy,ux=np.gradient(v[...,0]);vy,vx=np.gradient(v[...,1])
        return (1+ux)*(1+vy)-uy*vx
    result=mapping
    softened=False
    for sigma in (0,1,2,3,5,8):
        if sigma:result=cv2.GaussianBlur(mapping,(0,0),sigma);softened=True
        det=determinant(result)
        if float(det.min())>=floor:return result,float(det.min()),softened
    for _ in range(12):
        result=result*.8;det=determinant(result)
        if float(det.min())>=floor:return result,float(det.min()),True
    return np.zeros_like(mapping),1.,True


def corrected_window(frames, cut, config, full_frames=None):
    fields=build_fields(frames,cut,config)
    out=[]
    border=0.
    minimum_jacobian=1.;softened_frames=0
    for i,current in enumerate(frames):
        side='left' if i<cut else 'right'
        anchor=frames[cut-1] if i<cut else frames[cut]
        weight=smoothstep(i/max(cut-1,1)) if i<cut else smoothstep((len(frames)-1-i)/max(len(frames)-1-cut,1))
        source=current if full_frames is None else full_frames[i]
        if weight==0:
            out.append(source.copy());continue
        mapping=fields[side+'_map'].copy()
        tangent=fields[side+'_tangent']
        anchor_index=cut-1 if i<cut else cut
        # Hermite-like tangent correction matches velocity at the shared boundary.
        mapping += tangent*(i-anchor_index)*config.get('velocity_strength',1.)
        color=fields[side+'_color']
        if not np.array_equal(current,anchor):
            motion=flow(current,anchor)
            mapping=transport_vector(mapping,motion)
            color=sample(color,motion)
        mapping=mapping*float(weight)*config.get('geometry_strength',1.)
        color=color*float(weight)*config.get('color_strength',1.)
        mapping,jacobian,softened=safe_mapping(mapping)
        minimum_jacobian=min(minimum_jacobian,jacobian);softened_frames+=int(softened)
        if source.shape[:2]!=current.shape[:2]:
            mapping=resize_flow(mapping,(source.shape[1],source.shape[0]))
            color=cv2.resize(color,(source.shape[1],source.shape[0]),interpolation=cv2.INTER_CUBIC)
        coords=grid(source.shape)+mapping
        h,w=source.shape[:2]
        # Find every output pixel whose sampling kernel reaches past the source.
        invalid=(coords[...,0]<0)|(coords[...,0]>w-1)|(coords[...,1]<0)|(coords[...,1]>h-1)
        yy,xx=np.nonzero(invalid)
        if len(xx):
            depth=np.minimum.reduce([xx/w,(w-1-xx)/w,yy/h,(h-1-yy)/h])
            border=max(border,float(depth.max())+1/min(w,h))
        warped=sample(source,mapping,cv2.INTER_CUBIC).astype(np.float32)
        color=sample(color,mapping)
        gain,bias=fields['gain'],fields['bias']
        affine=((warped-bias)/gain-warped) if side=='left' else (warped*gain+bias-warped)
        color+=affine*(.5*float(weight)*config.get('color_strength',1.))
        out.append(np.clip(warped+color,0,255).round().astype(np.uint8))
    return out,fields['diagnostics']|{'required_crop_fraction':border,'minimum_jacobian':minimum_jacobian,'regularized_frames':softened_frames}


def crop_frame(frame,fraction):
    if not math.isfinite(fraction) or not 0<=fraction<.5:raise ValueError('Crop fraction must be finite and between 0 and 0.5.')
    if fraction==0:return frame
    h,w=frame.shape[:2];x=math.ceil(w*fraction);y=math.ceil(h*fraction)
    return cv2.resize(frame[y:h-y,x:w-x],(w,h),interpolation=cv2.INTER_LANCZOS4)


def repair(input:Path,plan:Path,output:Path,crf:int=16):
    from .media import probe,read_frames,iter_frames,VideoWriter,mux_audio
    input,plan,output=Path(input),Path(plan),Path(output)
    if input.resolve()==output.resolve() or output.exists() or output.with_suffix('.repair.json').exists():
        raise ValueError('Output must be a new path distinct from the source.')
    p=json.loads(plan.read_text());meta=probe(input)
    from .validation import validate_plan
    validate_plan(p,meta)
    if p.get('source_sha256')!=fingerprint(input):
        raise ValueError('Source fingerprint differs from the analyzed source; analyze again.')
    if p['source']['frame_count']!=meta['frame_count']:
        raise ValueError('Frame count differs from the plan.')
    output.parent.mkdir(parents=True,exist_ok=True)
    config=p['config'];patches={};reports=[];required_crop=0.
    # Temporary patch arrays keep full movie memory bounded and all writes atomic.
    with tempfile.TemporaryDirectory(prefix='seamstress-',dir=output.parent) as temp:
        temp=Path(temp)
        for seam in p['seams']:
            if not seam.get('enabled',True):continue
            n=seam['frame'];start=max(0,n-seam.get('before',12));end=min(meta['frame_count'],n+seam.get('after',18))
            full=read_frames(input,start,end-start)
            hold_report=None
            if seam.get('fill_startup_hold',False):
                from .retime import fill_startup_hold
                full,hold_report=fill_startup_hold(full,n-start)
            width=min(config.get('flow_width',640),meta['width'])
            size=(width,round(meta['height']*width/meta['width']))
            small=[cv2.resize(f,size,interpolation=cv2.INTER_AREA) for f in full]
            fixed,report=corrected_window(small,n-start,config,full)
            if hold_report is not None:report['startup_hold']=hold_report
            if config.get('texture_strength',0)>0:
                from .texture import texture_transition
                radius=config.get('texture_window',10)
                low=max(0,n-start-radius);high=min(len(fixed),n-start+radius)
                appearance,appearance_report=texture_transition(fixed[low:high],n-start-low,config)
                fixed[low:high]=appearance
                report['appearance']=appearance_report
            path=temp/f'{n}.npy';np.save(path,np.stack(fixed))
            for i in range(start,end):
                if i in patches:raise ValueError('Repair windows overlap; shorten plan windows.')
                patches[i]=(path,i-start)
            reports.append({'frame':n,'time':n/meta['fps'],**report})
            required_crop=max(required_crop,report['required_crop_fraction'])
            print(f'Repaired boundary {n} ({n/meta["fps"]:.3f}s)',flush=True)
        crop=max(config.get('crop_fraction',0.),required_crop+(.001 if required_crop>0 else 0))
        if crop>config.get('max_crop_fraction',.04):
            raise ValueError(f'Repair requires {crop:.1%} edge crop; exceeds plan limit. Inspect seam fields.')
        silent=temp/'video.mp4';cache={}
        with VideoWriter(silent,meta['width'],meta['height'],meta['fps_fraction'],crf=crf) as writer:
            for i,frame in enumerate(iter_frames(input)):
                if i in patches:
                    path,k=patches[i]
                    if path not in cache:cache={path:np.load(path,mmap_mode='r')}
                    frame=cache[path][k]
                writer.write(crop_frame(frame,crop))
        import os
        complete=temp/'complete.mp4'
        mux_audio(silent,input,complete)
        os.link(complete,output)  # Atomic exclusive publication on the same filesystem.
    result={'output':str(output.resolve()),'source_sha256':p['source_sha256'],
            'crop_fraction':crop,'seams':reports,'frame_count':meta['frame_count'],
            'method':'transported residual flow and spatial photometric correction',
            'requires_visual_review':True}
    output.with_suffix('.repair.json').write_text(json.dumps(result,indent=2))
    return {k:v for k,v in result.items() if k!='seams'}|{'seam_count':len(reports),'report_path':str(output.with_suffix('.repair.json').resolve())}
