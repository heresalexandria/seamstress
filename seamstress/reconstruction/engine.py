"""Source-first proposal/edit/import workflow with explicit abstention gates."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from ..media import probe, read_frames, iter_frames
from .bundle import (asset_path, check_cancel, copy_asset, freeze, load_bundle,
                     matrix, new_bundle, register, report, rgb_hash, save_rgb, sha256, validate, layer_id)
from .compositor import compose_frame
from . import vision

MAX_DECODED_RGB_BYTES=512*1024*1024


def _memory_bound(metadata,start,end):
    required=(end-start)*metadata['width']*metadata['height']*3
    if required>MAX_DECODED_RGB_BYTES:
        maximum=MAX_DECODED_RGB_BYTES//(metadata['width']*metadata['height']*3)
        reach=max(0,(maximum-1)//2)
        raise ValueError(f'Reconstruction window needs {required/(1024**2):.0f} MiB of decoded RGB; '
            f'the limit is 512 MiB. Shorten reachFrames to {reach} or less at this resolution, or use a smaller source.')


def _read(b,name):
    return np.array(Image.open(asset_path(b,name)).convert('RGB'))


def _mask(b,name):
    return np.asarray(Image.open(asset_path(b,name)).convert('L')).astype(np.float32)/255


def _put_mask(b,layer,n,mask):
    name=f'masks/{layer["id"]}-{n}.png'
    layer['mask_by_frame'][str(n)]=save_rgb(b,name,np.rint(mask*255).clip(0,255).astype(np.uint8))


def _put_frame(b,n,rgb,plate,alpha,emission=None,*,unknown=None,donors=None,residual_strength=1.):
    key=str(n); row=b['frames'].setdefault(key,{})
    row['source']=save_rgb(b,f'source/frame-{n}.png',rgb)
    row['plate']=save_rgb(b,f'plates/plate-{n}.png',plate)
    row['source_rgb_sha256']=rgb_hash(rgb)
    row['foreground_source_frame']=n
    row['residual_strength']=float(residual_strength)
    emission=np.zeros_like(rgb,np.float32) if emission is None else emission.astype(np.float32)
    # Exact source RGB in all opaque cores. Additive effects have their own
    # channel and are subtracted from P rather than counted twice.
    premultiplied=np.maximum(rgb.astype(np.float32)*alpha[:,:,None]-emission,0)
    path=Path(b['_root'])/f'mattes/matte-{n}.npz';path.parent.mkdir(exist_ok=True)
    np.savez_compressed(path,alpha=alpha.astype(np.float32),premultiplied=premultiplied.astype(np.float32),
        emission=emission,background_sha256=np.asarray(b['assets'][row['plate']]))
    row['matte']=register(b,path)
    if unknown is not None:
        row['unknown']=save_rgb(b,f'unknown/mask-{n}.png',unknown.astype(np.uint8)*255)
    row['donors']=donors or []


def _baseline_context(b):
    """Native geometry before the common view, exactly as conform applies it."""
    if not b.get('baseline'):return np.eye(3),np.eye(3),True
    plan=json.loads(asset_path(b,b['baseline']['asset']).read_text())
    def at(n):
        if plan.get('frame_matrices') is not None:return np.asarray(plan['frame_matrices'][n],np.float64)
        return np.asarray(next(s for s in plan['segments'] if s['start']<=n<s['end'])['matrix'],np.float64)
    n=b['frame'];left,right=at(n-1),at(n)
    geometries=[at(i) for i in range(b['support']['start'],b['support']['end'])]
    uniform=all(np.allclose(m,geometries[0],atol=1e-10,rtol=0) for m in geometries)
    grades=[(s['gain'],s['bias']) for s in plan['segments'] if s['start']<b['support']['end'] and s['end']>b['support']['start']]
    uniform=uniform and all(g==grades[0] for g in grades) and not plan.get('grade_curves') and not plan.get('local_color_curves')
    return left,right,bool(uniform)


def _residual_motion(left,right,velocity,measured):
    # Reconstruction executes before accepted conform geometry. Reapplying the
    # raw jump would double-correct seams already aligned by that baseline.
    result=np.linalg.inv(right)@left@velocity@measured
    if np.allclose(result,np.eye(3),atol=1e-7,rtol=0):return np.eye(3)
    return matrix(result)


def _assess(b,*,manual=False,cancelled=None):
    """Numeric gates are necessary, never a semantic segmentation certificate."""
    issues=[];blocking=[];records=[];unknown_pixels=0
    endpoints=[str(b['support']['start']),str(b['support']['end']-1)]
    if any(not np.allclose(layer['matrices'][n],np.eye(3),atol=1e-8,rtol=0) for layer in b['layers'] for n in endpoints):
        blocking.append('Correction must return to identity at both support endpoints; add return frames or shorten the motion envelope.')
    for key,row in b['frames'].items():
        check_cancel(cancelled)
        _,record=compose_frame(b,int(key),_read(b,row['source']));records.append(record)
        if row.get('unknown'):unknown_pixels+=int((_mask(b,row['unknown'])>.001).sum())
    max_gap=max((r.get('maximum_gap_radius',0) for r in records),default=0)
    max_visible=max((r.get('visible_gap_pixels',0) for r in records),default=0)
    if any(r.get('foreground_lost_pixels',0)>max(4,r.get('foreground_expected_pixels',0)*.002) for r in records):
        blocking.append('Foreground correction clips original actor pixels at the frame edge; reduce motion or expand source coverage.')
    if max_gap>max(4,min(b['source']['width'],b['source']['height'])*.015):
        blocking.append('Layer motion exposes a background gap too wide for bounded reconstruction; reduce motion or add coverage.')
    if max_visible and b['compositor'].get('gap_inpaint_radius',0)<=0:
        blocking.append('Layer motion leaves uncovered background pixels without an approved fill.')
    if unknown_pixels:
        issues.append('Hidden background still contains classical inpaint; inspect it or supply a source donor/AI background anchor.')
    if any(l.get('segmentation') in ('classical-cv-proposal','neural-seed-with-cv-tracking') for l in b['layers']):
        issues.append('Foreground and depth masks are proposals; inspect thin outlines, overlaps and intermediate frames.')
    tracking=b.get('provenance',{}).get('tracking_confidence',[])
    if tracking and min(tracking)<.65:issues.append('Local mask tracking has uncertain frames; add corrected keyframes there.')
    metrics=b['qa'].get('metrics',{})
    metrics.update(maximum_gap_radius=float(max_gap),maximum_visible_gap_pixels=int(max_visible),
                   unrecovered_background_pixels=int(unknown_pixels),support_frames=len(records))
    # Guarded automatic route is intentionally narrow: all features must agree
    # with one small similarity, no segmented actors/hidden content, complete
    # output coverage, and an independently measured adjacent-frame velocity.
    semantic_edits=any(set(e)-{'review_approved'} for e in b.get('provenance',{}).get('edits',[]))
    automatic=(not semantic_edits and metrics.get('single_motion_model',False) and metrics.get('inlier_count',0)>=40
        and metrics.get('inlier_fraction',0)>=.97 and metrics.get('fit_p90',999)<.55
        and metrics.get('feature_hull_fraction',0)>=.5 and metrics.get('photometric_p90',999)<=3
        and metrics.get('adjacent_velocity_verified',False)
        and metrics.get('baseline_context_uniform',False)
        and metrics.get('maximum_correction_pixels',999)<=3
        and not unknown_pixels and not max_visible and not any(r.get('edge_missing_pixels',0) for r in records)
        and not issues and not blocking)
    b['qa']={'status':'approved' if manual and not blocking else ('auto-eligible' if automatic else 'needs-review'),
        'auto_eligible':bool(automatic),'blocking_errors':blocking,'issues':issues,'metrics':metrics,
        'frame_checks':records,'reviewed_by_user':bool(manual and not blocking)}


def propose(source,frame,output_dir,*,baseline_plan=None,options=None,progress=None,cancelled=None):
    """Make a runnable, reviewable native-layer proposal from an arbitrary shot.

    Automatic labels are classical CV estimates. The high-confidence automatic
    route only handles verified small single-layer corrections; articulated
    subjects, layered occlusions and invented pixels require review.
    """
    opts=dict(options or {});meta=probe(source);check_cancel(cancelled)
    allowed={'reachFrames','motionStrength','boundaries','allowAI','maxAIRequests','quality','segmentation'}
    if set(opts)-allowed:raise ValueError('Unknown reconstruction options: '+', '.join(sorted(set(opts)-allowed)))
    reach=opts.get('reachFrames',max(3,round(meta['fps']*.6)))
    strength=opts.get('motionStrength',1.)
    if type(reach) is not int or not 2<=reach<=max(2,round(meta['fps']*10)):
        raise ValueError('reachFrames must be 2 frames to 10 seconds')
    if isinstance(strength,bool) or not isinstance(strength,(float,int)) or not 0<=strength<=1:
        raise ValueError('motionStrength must be between zero and one')
    start,end=max(0,frame-reach),min(meta['frame_count'],frame+reach+1)
    if frame-start<2 or end-frame<2:
        raise ValueError('Reconstruction needs two source frames on both sides of the seam')
    boundaries=opts.get('boundaries',[])
    if any(type(n) is not int or not 0<n<meta['frame_count'] for n in boundaries):
        raise ValueError('boundaries must be incoming source frame indices')
    if any(n!=frame and start<n<end for n in boundaries):
        raise ValueError('Reconstruction support crosses another marked seam; shorten reachFrames')
    _memory_bound(meta,start,end)
    b=new_bundle(source,frame,output_dir,(start,end),baseline_plan)
    b['options']={'reachFrames':reach,'motionStrength':float(strength),'segmentation':opts.get('segmentation','auto')}
    b['provenance']['method']='classical-cv motion groups + seeded GrabCut + local mask tracking + source-first donors'
    b['compositor']={'edge_band':0,'gap_threshold':.7,'gap_inpaint_radius':3}
    report(progress,.03,'Reading original frames and matching coherent motion')
    rgb=read_frames(source,start,end-start);images={n:rgb[n-start] for n in range(start,end)}
    incoming,outgoing=images[frame],images[frame-1]
    clusters,p,q=vision.motion_clusters(incoming,outgoing)
    measured=bool(clusters)
    if not clusters:
        clusters=[{'matrix':np.eye(3),'points':np.empty((0,2)), 'targets':np.empty((0,2)),'inliers':0,'p90':999.}]
    h,w=incoming.shape[:2]
    # A smaller interior independent-motion cluster is an actor candidate.
    # It is explicitly not classified semantically and cannot auto-accept.
    actor_index=None
    for i,c in enumerate(clusters[1:],1):
        points=c['points'];lo=points.min(axis=0);hi=points.max(axis=0)
        if np.prod(hi-lo)<w*h*.6 and lo[0]>w*.03 and hi[0]<w*.97:
            actor_index=i;break
    actor=vision.segment_cluster(incoming,clusters[actor_index]['points']) if actor_index is not None else np.zeros((h,w),np.float32)
    segmentation=opts.get('segmentation','auto')
    if segmentation not in ('auto','classic','neural'):raise ValueError('segmentation must be auto, classic or neural')
    model_provenance=None
    if actor_index is not None and segmentation!='classic':
        try:
            from .. import segmentation_model
            points=clusters[actor_index]['points'];lo=points.min(axis=0);hi=points.max(axis=0)
            # Auto uses only an already installed model; downloads are an
            # explicit model-setup action owned by the app.
            if segmentation=='neural' or segmentation_model.available():
                result=segmentation_model.predict_mask(incoming,points=[{'x':float(p[0]),'y':float(p[1]),'label':1} for p in points[::max(1,len(points)//6)]],
                    box=[float(lo[0]),float(lo[1]),float(hi[0]),float(hi[1])],cancelled=cancelled)
                actor=np.asarray(result['mask'],np.float32)
                if actor.shape!=(h,w) or not np.isfinite(actor).all():raise ValueError('Segmentation model returned an invalid native mask')
                actor=actor.clip(0,1);model_provenance=result.get('model',{})
        except (ImportError,FileNotFoundError) as exc:
            if segmentation=='neural':raise ValueError('Install the optional local segmentation model before selecting neural mode') from exc
    elif segmentation=='neural':
        raise ValueError('No independent foreground group was found; use a prompted neural mask on the source frame')
    background_clusters=[c for i,c in enumerate(clusters) if i!=actor_index]
    reference_masks=vision.label_planes((h,w),background_clusters)
    layers=[]
    for i,c in enumerate(background_clusters):
        layers.append({'id':f'background-{i}','name':f'Background plane {i+1}','kind':'background','order':i,
            'matrices':{},'mask_by_frame':{},'keyframes':[frame],'confidence':float(min(1,c['inliers']/60)),
            'segmentation':'classical-cv-proposal' if len(background_clusters)>1 else 'single-motion-plane'})
    fg={'id':'foreground','name':'Original foreground','kind':'foreground','order':len(layers),
        'matrices':{},'mask_by_frame':{},'keyframes':[frame],'confidence':0. if actor_index is None else .5,
        'segmentation':'classical-cv-proposal' if actor_index is not None else 'empty'}
    b['layers']=[*layers,fg]
    if model_provenance is not None:
        fg['segmentation']='neural-seed-with-cv-tracking';b['provenance']['segmentation_model']=model_provenance
    # Remove expected one-frame camera movement from the cut discrepancy.
    pp,qq=vision.match_points(images[frame-2],outgoing)
    velocity,velocity_keep,velocity_error=vision.fit_similarity(pp,qq)
    velocity_verified=velocity_keep.sum()>=20 and velocity_error<.7
    if not velocity_verified:velocity=np.eye(3)
    baseline_left,baseline_right,baseline_uniform=_baseline_context(b)
    corrections=[_residual_motion(baseline_left,baseline_right,velocity,c['matrix']) for c in background_clusters] if measured else [np.eye(3) for _ in background_clusters]
    fg_correction=_residual_motion(baseline_left,baseline_right,velocity,clusters[actor_index]['matrix']) if measured and actor_index is not None else np.eye(3)
    clock=vision.cadence_clock(rgb,start);masks={};tracking=[]
    corners=np.array([[0,0,1],[w-1,0,1],[0,h-1,1],[w-1,h-1,1]],float).T
    maximum=0.
    for n in range(start,end):
        check_cancel(cancelled)
        if n==frame:
            fgmask=actor.copy();bgmasks=reference_masks;confidence=1.
        else:
            fgmask,confidence=vision.transport_mask(incoming,images[n],actor) if actor.any() else (actor.copy(),1.)
            if len(layers)==1:bgmasks=reference_masks
            else:
                x,y=vision.local_flow(incoming,images[n]); scores=np.stack([cv2.remap(m,x,y,cv2.INTER_LINEAR) for m in reference_masks]); labels=np.argmax(scores,axis=0)
                bgmasks=[(labels==i).astype(np.float32) for i in range(len(layers))]
        masks[n]=fgmask;tracking.append(confidence)
        # Incoming-only correction avoids moving accepted outgoing drawings.
        amount=vision.gate(n,start,end,frame,clock)*strength if n>=frame else 0.
        for layer,refmask,correction in zip(layers,bgmasks,corrections):
            m=np.eye(3)+amount*(correction-np.eye(3));matrix(m)
            layer['matrices'][str(n)]=m.tolist();_put_mask(b,layer,n,refmask)
            maximum=max(maximum,float(np.linalg.norm((m@corners-corners)[:2],axis=0).max()))
        fm=np.eye(3)+amount*(fg_correction-np.eye(3));fg['matrices'][str(n)]=fm.tolist();_put_mask(b,fg,n,fgmask)
        report(progress,.10+.35*(n-start+1)/(end-start),f'Tracking source-native layer masks at {n}')
    b['provenance']['tracking_confidence']=tracking
    b['provenance']['cadence_clock']={str(n):v for n,v in clock.items()}
    b['provenance']['correction_matrices']={l['id']:m.tolist() for l,m in zip([*layers,fg],[*corrections,fg_correction])}
    for n in range(start,end):
        check_cancel(cancelled)
        # Farthest same-shot temporal donors reveal the most hidden content.
        candidates=sorted((j for j in images if j!=n),key=lambda j:-abs(j-n))[:4]
        plate,unknown,records=vision.recover_background(images[n],masks[n],[(images[j],masks[j],j) for j in candidates])
        _put_frame(b,n,images[n],plate,masks[n],unknown=unknown,donors=records)
        report(progress,.45+.4*(n-start+1)/(end-start),f'Recovering observed background at {n}')
    matched=cv2.warpPerspective(incoming,clusters[0]['matrix'],(w,h))
    overlap=cv2.warpPerspective(np.ones((h,w),np.float32),clusters[0]['matrix'],(w,h))>.999
    photo=np.mean(np.abs(matched.astype(np.float32)-outgoing.astype(np.float32)),axis=2)
    hull=float(cv2.contourArea(cv2.convexHull(clusters[0]['points'].astype(np.float32))))/(w*h) if len(clusters[0]['points'])>=3 else 0.
    b['qa']['metrics']={'single_motion_model':len(clusters)==1 and actor_index is None and measured,
        'inlier_count':clusters[0]['inliers'],'inlier_fraction':clusters[0]['inliers']/max(1,len(p)),
        'fit_p90':clusters[0]['p90'],'adjacent_velocity_verified':bool(velocity_verified),
        'baseline_context_uniform':baseline_uniform,
        'feature_hull_fraction':hull,'photometric_p90':float(np.quantile(photo[overlap],.90)) if overlap.any() else 999.,
        'maximum_correction_pixels':maximum}
    _assess(b,cancelled=cancelled)
    if not measured:b['qa']['issues'].append('Insufficient coherent features; identity proposal requires authored masks or motion.')
    check_cancel(cancelled);report(progress,.98,'Validating immutable reconstruction assets')
    return freeze(b)


def _clone(original,output_dir):
    old=load_bundle(original)
    b=new_bundle(old['source']['path'],old['frame'],output_dir,
        (old['support']['start'],old['support']['end']),asset_path(old,old['baseline']['asset']) if old.get('baseline') else None)
    reserved={k:b[k] for k in ('id','_root','source','baseline')}
    b=copy.deepcopy(old);b.update(reserved);b.pop('_manifest',None);b['assets']={}
    for name in old['assets']:copy_asset(b,asset_path(old,name),name)
    b['provenance']['parent_manifest_sha256']=sha256(Path(old['_root'])/'manifest.json')
    return b


def _strokes(mask,strokes):
    value=mask.copy();h,w=value.shape
    for stroke in strokes:
        mode=stroke.get('mode');radius=stroke.get('radius');points=stroke.get('points')
        if mode not in ('include','exclude','protect','emission') or isinstance(radius,bool) or not isinstance(radius,(float,int)) or not 0<radius<=max(h,w):
            raise ValueError('Invalid mask brush mode/radius')
        p=np.asarray(points,float)
        if p.ndim!=2 or p.shape[1]!=2 or len(p)==0 or len(p)>10000 or not np.isfinite(p).all() or (p<0).any() or (p[:,0]>=w).any() or (p[:,1]>=h).any():
            raise ValueError('Brush points must lie inside native source coordinates')
        color=0. if mode=='exclude' else 1.
        coords=np.rint(p).astype(np.int32);thickness=max(1,round(radius*2))
        for point in coords:cv2.circle(value,tuple(point),max(1,round(radius)),color,-1)
        if len(coords)>1:cv2.polylines(value,[coords],False,color,thickness)
    return value.clip(0,1)


def edit_bundle(manifest,output_dir,edits,*,progress=None,cancelled=None):
    """Create a new immutable revision; never mutate accepted assets."""
    old=load_bundle(manifest);check_cancel(cancelled)
    edits=dict(edits or {})
    for key in ('allowAI','maxAIRequests','quality'):edits.pop(key,None)
    allowed={'mask_keyframes','strokes','layer_matrices','review_approved','motionStrength','reachFrames'}
    if set(edits)-allowed:raise ValueError('Unknown reconstruction edit fields: '+', '.join(sorted(set(edits)-allowed)))
    edits_masks=bool(edits.get('mask_keyframes') or edits.get('strokes'))
    if edits_masks:_memory_bound(old['source'],old['support']['start'],old['support']['end'])
    b=_clone(old,output_dir);start,end=b['support']['start'],b['support']['end'];frame=b['frame']
    layers={l['id']:l for l in b['layers']};images={int(k):_read(b,r['source']) for k,r in b['frames'].items()} if edits_masks else {}
    keyframes={};emission_keys={}
    for row in [*edits.get('mask_keyframes',[]),*edits.get('strokes',[])]:
        n=row.get('frame');ident=row.get('layer_id')
        if type(n) is not int or not start<=n<end or ident not in layers:
            raise ValueError('Mask edit must target an existing layer and supported source frame')
        key=(ident,n)
        if 'path' in row:
            im=np.asarray(Image.open(row['path']).convert('L')).astype(np.float32)/255
            if im.shape!=(b['source']['height'],b['source']['width']):raise ValueError('Edited mask must use native source dimensions')
            keyframes[key]=im
        else:
            target=emission_keys if row.get('mode')=='emission' else keyframes
            if key not in target:
                prior_emission=layers[ident].get('emission_keyframes',{}).get(str(n))
                target[key]=(_mask(b,prior_emission) if prior_emission else np.zeros(images[n].shape[:2],np.float32)) if target is emission_keys else _mask(b,layers[ident]['mask_by_frame'][str(n)])
            target[key]=_strokes(target[key],[row])
    affected=set();explicit={}
    for (ident,n),mask in keyframes.items():
        explicit.setdefault(ident,[]).append(n)
        layers[ident]['keyframes']=sorted(set(layers[ident].get('keyframes',[])+[n]))
    for ident,refs in explicit.items():
        layer=layers[ident]
        # Generated seeds are not user anchors. Retain every previously
        # authored keyframe exactly when a later edit introduces a new anchor.
        prior_refs=layer.get('authored_keyframes',[])
        for n in prior_refs:
            if (ident,n) not in keyframes:keyframes[ident,n]=_mask(b,layer['mask_by_frame'][str(n)])
        refs=sorted(set(refs+prior_refs));layer['authored_keyframes']=refs
        for n in range(start,end):
            check_cancel(cancelled)
            # Local edits never propagate through the generation seam without
            # another authored keyframe on that side.
            choices=[j for j in refs if (j<frame)==(n<frame)]
            if not choices:continue
            j=min(choices,key=lambda j:abs(j-n));mask=keyframes[ident,j]
            if n!=j:mask,confidence=vision.transport_mask(images[j],images[n],mask)
            _put_mask(b,layer,n,mask);affected.add(n)
            if layer['kind']=='background':
                others=[l for l in b['layers'] if l['kind']=='background' and l['id']!=ident]
                if not others:
                    if np.min(mask)<.999:raise ValueError('The only background plane must cover the source; add another plane before excluding pixels')
                else:
                    for other in others:_put_mask(b,other,n,_mask(b,other['mask_by_frame'][str(n)])*(1-mask))
                    coverage=sum(_mask(b,l['mask_by_frame'][str(n)]) for l in b['layers'] if l['kind']=='background')
                    _put_mask(b,others[0],n,np.maximum(_mask(b,others[0]['mask_by_frame'][str(n)]),1-coverage))
    fg=next(l for l in b['layers'] if l['kind']=='foreground')
    emissions={};emission_layers=set()
    for (ident,j),mask in emission_keys.items():
        if ident!=fg['id']:raise ValueError('Emission strokes target the foreground layer')
        path=save_rgb(b,f'masks/emission-keyframe-{ident}-{j}.png',np.rint(mask*255).astype(np.uint8))
        layers[ident].setdefault('emission_keyframes',{})[str(j)]=path;emission_layers.add(ident)
    for ident in emission_layers:
        refs={int(j):_mask(b,path) for j,path in layers[ident]['emission_keyframes'].items()}
        for n in range(start,end):
            choices=[j for j in refs if (j<frame)==(n<frame)]
            if not choices:continue
            j=min(choices,key=lambda j:abs(j-n));mask=refs[j]
            m=mask if n==j else vision.transport_mask(images[j],images[n],mask)[0]
            raw=images[n].astype(np.float32);warm=np.maximum(raw-raw[:,:,2:3],0);warm[:,:,2]=0
            emissions[n]=np.maximum(emissions.get(n,np.zeros_like(warm)),warm*m[:,:,None]);affected.add(n)
    masks={n:_mask(b,fg['mask_by_frame'][str(n)]) for n in images}
    for index,n in enumerate(sorted(affected)):
        check_cancel(cancelled)
        donors=sorted((j for j in images if j!=n),key=lambda j:-abs(j-n))[:4]
        plate,unknown,records=vision.recover_background(images[n],masks[n],[(images[j],masks[j],j) for j in donors])
        if n not in emissions:
            with np.load(asset_path(old,old['frames'][str(n)]['matte'])) as data:emissions[n]=data['emission']
        _put_frame(b,n,images[n],plate,masks[n],emissions[n],unknown=unknown,donors=records)
        report(progress,(index+1)/max(1,len(affected)),f'Propagating authored masks and recovering background at {n}')
    if 'motionStrength' in edits or 'reachFrames' in edits:
        corrections=b['provenance'].get('correction_matrices')
        if not corrections:raise ValueError('This authored bundle has explicit motion; edit layer matrices instead')
        opts=b.setdefault('options',{});strength=edits.get('motionStrength',opts.get('motionStrength',1.));reach=edits.get('reachFrames',opts.get('reachFrames',frame-start))
        if isinstance(strength,bool) or not isinstance(strength,(int,float)) or not 0<=strength<=1:raise ValueError('motionStrength must be zero to one')
        if type(reach) is not int or reach<2 or reach>max(frame-start,end-frame-1):raise ValueError('Edited reachFrames must fit the existing decoded bundle; propose again to extend it')
        opts.update(motionStrength=float(strength),reachFrames=reach)
        clock={int(k):v for k,v in b['provenance']['cadence_clock'].items()}
        lo,hi=max(start,frame-reach),min(end,frame+reach+1)
        for layer in b['layers']:
            for n in range(start,end):
                amount=vision.gate(n,lo,hi,frame,clock)*strength if frame<=n<hi else 0.
                layer['matrices'][str(n)]=(np.eye(3)+amount*(np.asarray(corrections[layer['id']])-np.eye(3))).tolist()
    for row in edits.get('layer_matrices',[]):
        n=row.get('frame');ident=row.get('layer_id')
        if type(n) is not int or not start<=n<end or ident not in layers:raise ValueError('Motion edit targets an unavailable frame/layer')
        layers[ident]['matrices'][str(n)]=matrix(row['matrix']).tolist()
    b['provenance'].setdefault('edits',[]).append(copy.deepcopy(edits))
    if edits.get('review_approved') not in (None,True,False):raise ValueError('review_approved must be boolean')
    _assess(b,manual=edits.get('review_approved') is True,cancelled=cancelled)
    return freeze(b)


def import_bundle(manifest,output_dir,*,source,baseline_plan=None,progress=None,cancelled=None):
    """Copy a complete immutable bundle without rebinding its accepted context."""
    old=load_bundle(manifest,verify=False);validate(old,verify=True,verify_source=False);check_cancel(cancelled)
    if sha256(source)!=old['source']['sha256']:raise ValueError('Imported bundle belongs to another source')
    baseline=old.get('baseline')
    if (baseline is None)!=(baseline_plan is None):
        raise ValueError('Imported bundle baseline differs from the current accepted plan')
    if baseline:
        from ..reconstruction_render import context_signature
        before=json.loads(asset_path(old,baseline['asset']).read_text())
        current=json.loads(Path(baseline_plan).read_text())
        start,end=old['support']['start'],old['support']['end']
        if context_signature(before,start,end)!=context_signature(current,start,end):
            raise ValueError('Imported bundle baseline framing or grade differs within its support')
    old=copy.deepcopy(old)
    old['provenance']['original_source_path']=old['source']['path']
    old['source']['path']=str(Path(source).expanduser().resolve())
    b=_clone(old,output_dir);b['source']['path']=str(Path(source).expanduser().resolve())
    return freeze(b)


def import_authored(source,frame,output_dir,*,support,frames,layers,baseline_plan=None,
                    compositor=None,provenance=None,review_approved=False,progress=None,cancelled=None):
    """Import native authored masks/plates/mattes and explicit rigid transforms.

    frames maps source index to {source,plate,matte,residual_strength?}, where
    source/plate are PNG paths and matte is the documented alpha/P/emission NPZ.
    layers contain mask_by_frame paths and matrices, all in native coordinates.
    All files are copied, hash-bound, and verified before READY publication.
    """
    b=new_bundle(source,frame,output_dir,tuple(support),baseline_plan)
    b['compositor'].update(compositor or {})
    b['provenance'].update(provenance or {})
    b['provenance']['method']='authored native layer bundle'
    start,end=support
    if set(map(int,frames))!=set(range(start,end)):raise ValueError('Authored assets do not cover the complete support')
    b['layers']=copy.deepcopy(layers)
    for layer in b['layers']:
        layer_id(layer['id'])
        layer.setdefault('authored_keyframes',copy.deepcopy(layer.get('keyframes',[])))
        layer['matrices']={str(n):matrix(m).tolist() for n,m in layer['matrices'].items()}
        layer['mask_by_frame']={str(n):copy_asset(b,path,f'masks/{layer["id"]}-{n}.png') for n,path in layer['mask_by_frame'].items()}
    for n in range(start,end):
        check_cancel(cancelled);row=copy.deepcopy(frames.get(n,frames.get(str(n))))
        for kind,folder,suffix in [('source','source','.png'),('plate','plates','.png'),('matte','mattes','.npz')]:
            row[kind]=copy_asset(b,row[kind],f'{folder}/{kind}-{n}{suffix}')
        if row.get('unknown'):
            row['unknown']=copy_asset(b,row['unknown'],f'unknown/mask-{n}.png')
        row['source_rgb_sha256']=rgb_hash(_read(b,row['source']));row['foreground_source_frame']=n
        b['frames'][str(n)]=row
        report(progress,(n-start+1)/(end-start),f'Importing source frame {n}')
    # Source PNGs must match decoded original frames, not a graded proxy or
    # silently substituted actor drawing.
    count=0
    for n,rgb in enumerate(iter_frames(source,start,end-start),start):
        check_cancel(cancelled);count+=1
        if rgb_hash(rgb)!=b['frames'][str(n)]['source_rgb_sha256']:raise ValueError(f'Authored RGB is not exact original source at {n}')
    if count!=end-start:raise ValueError('Authored support exceeds decoded source frames')
    _assess(b,manual=review_approved,cancelled=cancelled)
    return freeze(b)


def _rebase_plate(b,old,n,plate,remaining):
    row=b['frames'][str(n)];prior=old['frames'][str(n)]
    plate_name=save_rgb(b,f'plates/plate-{n}.png',plate);row['plate']=plate_name
    with np.load(asset_path(old,prior['matte']),allow_pickle=False) as data:
        arrays={k:data[k].copy() for k in data.files}
    before=_read(old,prior['plate']).astype(np.float32)
    arrays['premultiplied']=(arrays['premultiplied']+(1-arrays['alpha'][:,:,None])*(before-plate.astype(np.float32))).clip(0,255)
    arrays['background_sha256']=np.asarray(b['assets'][plate_name])
    path=asset_path(b,row['matte']);np.savez_compressed(path,**arrays);register(b,path)
    row['unknown']=save_rgb(b,f'unknown/mask-{n}.png',remaining.astype(np.uint8)*255)


def prepare_backgrounds(manifest,output_dir,*,provider=None,progress=None,cancelled=None):
    """Generate one missing-background anchor, then source-first propagation.

    provider(source_rgb, editable_mask, source_frame) returns image_path plus
    optional provenance. Root owns upload approval, API key and request budget.
    """
    old=load_bundle(manifest);check_cancel(cancelled)
    if provider is None:
        _memory_bound(old['source'],old['support']['start'],old['support']['end'])
        b=_clone(old,output_dir)
        images={int(k):_read(old,row['source']) for k,row in old['frames'].items()}
        masks={}
        for key,row in old['frames'].items():
            with np.load(asset_path(old,row['matte']),allow_pickle=False) as data:
                masks[int(key)]=np.maximum(data['alpha'],np.any(data['emission']>.1,axis=2).astype(np.float32))
        for index,n in enumerate(sorted(images)):
            check_cancel(cancelled)
            donors=sorted((j for j in images if j!=n),key=lambda j:-abs(j-n))[:4]
            plate,remaining,records=vision.recover_background(images[n],masks[n],[(images[j],masks[j],j) for j in donors])
            _rebase_plate(b,old,n,plate,remaining);b['frames'][str(n)]['donors']=records
            report(progress,(index+1)/len(images),f'Recovering original donor background at {n}')
        b['provenance'].setdefault('background_recovery',[]).append({'method':'original source donors only','frame_count':len(images)})
        _assess(b,cancelled=cancelled)
        return freeze(b)
    candidates=[]
    for key,row in old['frames'].items():
        if row.get('unknown'):
            mask=_mask(old,row['unknown'])>.001
            if mask.any():candidates.append((int(mask.sum()),int(key),mask))
    if not candidates:raise ValueError('This bundle has no unrecovered background holes to generate')
    b=_clone(old,output_dir)
    _,anchor,hole=max(candidates,key=lambda row:row[0])
    source=_read(old,old['frames'][str(anchor)]['source'])
    result=provider(source,hole,anchor);check_cancel(cancelled)
    generated=np.asarray(Image.open(result['image_path']).convert('RGB'))
    if generated.shape!=source.shape:raise ValueError('Generated background does not match native dimensions')
    if not np.array_equal(generated[~hole],source[~hole]):raise ValueError('Background provider changed observed source pixels outside requested holes')
    anchor_plate=_read(old,old['frames'][str(anchor)]['plate']);anchor_plate[hole]=generated[hole]
    foreground=next(l for l in b['layers'] if l['kind']=='foreground')
    for key,row in b['frames'].items():
        check_cancel(cancelled);n=int(key);original=_read(b,row['source']);plate=_read(b,row['plate'])
        unknown=_mask(b,row['unknown'])>.001 if row.get('unknown') else np.zeros(source.shape[:2],bool)
        remaining=unknown.copy()
        if n==anchor:plate[hole]=anchor_plate[hole];remaining[hole]=False
        elif unknown.any():
            source_visible=(_mask(b,foreground['mask_by_frame'][str(anchor)])<.001).astype(np.uint8)*255
            target_visible=(_mask(b,foreground['mask_by_frame'][key])<.001).astype(np.uint8)*255
            p,q=vision.match_points(source,original,source_visible,target_visible)
            m,keep,error=vision.fit_similarity(p,q)
            if keep.sum()>=8 and error<=1.5:
                h,w=unknown.shape;warped=cv2.warpPerspective(anchor_plate,m,(w,h))
                coverage=cv2.warpPerspective(np.ones((h,w),np.float32),m,(w,h))>.999
                use=unknown&coverage;plate[use]=warped[use];remaining[use]=False
        _rebase_plate(b,old,n,plate,remaining)
        row['generated_anchor_frame']=anchor
        report(progress,(n-b['support']['start']+1)/(b['support']['end']-b['support']['start']),f'Propagating one background anchor to {n}')
    b['provenance'].setdefault('background_generation',[]).append({'anchor_frame':anchor,
        'generated_sha256':sha256(result['image_path']), 'request_id':result.get('request_id'),
        'cache_hit':bool(result.get('cache_hit',False))})
    _assess(b,cancelled=cancelled)
    b['qa']['auto_eligible']=False;b['qa']['status']='needs-review'
    b['qa']['issues'].append('Generated hidden background is plausible synthesis, not recovered original content; review its geometry and temporal stability.')
    return freeze(b)
