"""Experiment: separate camera support from appearance replacement windows."""
from pathlib import Path
import json
import cv2
import numpy as np
from seamstress.media import read_frames,VideoWriter
from seamstress.bridge import RifeModel,progress_curve

src=Path('seamstress/camera_bridge.py').read_text().replace('from .registration','from seamstress.registration').replace('from .repair','from seamstress.repair')
src=src.replace('pre_frames, post_frames, progress):','pre_frames, post_frames, progress, appearance_frames=None, seam_index=None, radius=2):')
old='''    synthesized = model.synthesize(left_common.round().astype(np.uint8),
                                   right_common.round().astype(np.uint8), progress)'''
new='''    if appearance_frames is None:
        synthesized = model.synthesize(left_common.round().astype(np.uint8), right_common.round().astype(np.uint8), progress)
    else:
        anchors=[]
        for j,frame in enumerate(appearance_frames):
            if j==0: anchors.append(left_common.round().astype(np.uint8));continue
            if j==span: anchors.append(right_common.round().astype(np.uint8));continue
            reg=_camera_registration(left,frame)
            relative=_similarity(np.array(reg['matrix']),center)
            common=common_warp(frame,offset@relative).astype(np.float32)
            mask=common_warp(ones,offset@relative,cv2.INTER_NEAREST)>0
            if j>=seam_index:common=common*gain+bias
            nearest_j=distance_transform_edt(~mask,return_distances=False,return_indices=True)
            common[~mask]=common[tuple(nearest_j[:,~mask])]
            anchors.append(np.clip(common,0,255).round().astype(np.uint8))
        lo=max(1,seam_index-radius);hi=min(span-1,seam_index+radius)
        tween=model.synthesize(anchors[lo],anchors[hi],np.linspace(0,1,hi-lo+1))
        synthesized=anchors[:lo]+tween+anchors[hi+1:]
'''
assert old in src
src=src.replace(old,new)
namespace={};exec(compile(src,'research/multi_anchor_module.py','exec'),namespace)
bridge=namespace['synthesize_camera_bridge']
out=Path('research/multi-anchor');out.mkdir(exist_ok=True)
model=RifeModel('models/rife425/flownet.pkl','mps')
plan=json.loads(Path('plans/IYTYT-bridges.json').read_text())
for seam in plan['seams']:
    n=seam['frame']
    if n not in (722,1805):continue
    lo,hi=seam['bridge_start'],seam['bridge_end'];start=lo-12
    originals=read_frames('IYTYT.mp4',start,hi-lo+25)
    source=originals[12:12+hi-lo+1]
    baseline=read_frames('output/IYTYT-bridged.mp4',start,len(originals))
    frames,report=bridge(model,source[0],source[-1],originals[8:12],originals[12+len(source):16+len(source)],progress_curve(len(source)),appearance_frames=source,seam_index=n-lo,radius=2)
    np.savez_compressed(out/f'{n}.npz',frames=np.stack(frames))
    (out/f'{n}.json').write_text(json.dumps(report,indent=2))
    cv2.imwrite(str(out/f'{n}-middle.png'),cv2.cvtColor(frames[n-lo],cv2.COLOR_RGB2BGR))
    with VideoWriter(out/f'{n}-comparison.mp4',1920,384,'24000/1001',crf=14) as writer:
        for repeat in range(2):
            for j,a in enumerate(originals):
                k=j-12;f=frames[k] if 0<=k<len(frames) else a
                tiles=[cv2.resize(x,(640,360),interpolation=cv2.INTER_AREA) for x in (a,baseline[j],f)]
                image=np.zeros((384,1920,3),np.uint8)
                for t,(label,x) in enumerate(zip(('Original','Long bridge','Short appearance bridge'),tiles)):
                    image[24:,t*640:(t+1)*640]=x
                    cv2.putText(image,label,(t*640+10,17),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255),1)
                writer.write(image)
    print(n,'complete',flush=True)
