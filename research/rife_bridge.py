"""Bounded true-frame-interpolation bridge experiment using official RIFE 4.25.

The eight interior frames replace existing samples; both endpoint images,
total frame count, and playback rate remain exact. This is an experiment,
not a claim that invented in-between drawings are perceptually correct.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn.functional as F

from seamstress.analysis import pair_metrics
from seamstress.media import read_frames, VideoWriter
from seamstress.repair import corrected_window, flow, sample


class RifeInterpolator:
    def __init__(self, root=ROOT/"research/rife", device="auto"):
        root = Path(root)
        sys.path.insert(0, str(root/"Practical-RIFE"))
        spec = importlib.util.spec_from_file_location("official_rife_ifnet", root/"model425/train_log/IFNet_HDv3.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if device == "auto":
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = torch.device(device)
        torch.set_num_threads(4)
        self.model = module.IFNet().eval().to(self.device)
        state = torch.load(root/"model425/train_log/flownet.pkl", map_location="cpu", weights_only=True)
        state = {key.removeprefix("module."): value for key, value in state.items()}
        # The official checkpoint retains training-only teacher/timing heads,
        # explicitly commented out in the supplied inference architecture.
        state = {key: value for key, value in state.items()
                 if not key.startswith(("teacher.", "caltime."))}
        self.model.load_state_dict(state, strict=True)

    def sync(self):
        if self.device.type == "mps":
            torch.mps.synchronize()

    def interpolate(self, left, right, interior_count=8, timesteps=None):
        """Return unchanged endpoints around independently inferred tween frames."""
        h, w = left.shape[:2]
        a = torch.from_numpy(left.copy()).permute(2,0,1)[None].float().to(self.device)/255
        b = torch.from_numpy(right.copy()).permute(2,0,1)[None].float().to(self.device)/255
        padding = (0, (-w)%64, 0, (-h)%64)
        # Same constant padding and RGB input as upstream inference_video.py.
        a, b = F.pad(a,padding), F.pad(b,padding)
        pair = torch.cat([a,b],dim=1)
        output=[left.copy()]
        diagnostics=[]
        self.sync()
        start=time.perf_counter()
        if timesteps is None:
            timesteps=[i/(interior_count+1) for i in range(1,interior_count+1)]
        if len(timesteps)!=interior_count or any(not 0<t<1 for t in timesteps):
            raise ValueError("Interior timesteps must be strictly between zero and one")
        for timestep in timesteps:
            with torch.inference_mode():
                flows,mask,merged=self.model(pair,timestep,[16,8,4,2,1])
            image=merged[-1][0,:,:h,:w].permute(1,2,0).clamp(0,1)
            output.append((image.cpu().numpy()*255).round().astype(np.uint8))
            blend_mask=mask[0,0,:h,:w].sigmoid().cpu().numpy()
            diagnostics.append({"timestep":timestep,
                                "mixed_mask_fraction":float(np.mean((blend_mask>.15)&(blend_mask<.85)))})
        self.sync()
        elapsed=time.perf_counter()-start
        output.append(right.copy())
        return output,{"runtime_seconds":elapsed,"interior_count":interior_count,
                       "endpoint_preserved":bool(np.array_equal(output[0],left) and np.array_equal(output[-1],right)),
                       "inference":diagnostics}


def sharpness(frame):
    gray=cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY).astype(np.float32)
    return float(np.mean(abs(cv2.Laplacian(gray,cv2.CV_32F))))


def velocity_progress(frames,lo,hi):
    """Monotone scalar progress fitted to robust endpoint motion projections.

    A single scalar cannot resolve mutually conflicting subject trajectories.
    Return the measured slopes so callers can reject implausible timing fits.
    """
    a,b=frames[lo],frames[hi]
    forward=flow(a,b);backward=flow(b,a)
    fb=np.linalg.norm(forward+sample(backward,forward),axis=-1)
    va=-flow(a,frames[lo-4])/4
    vb=sample(flow(b,frames[hi+4])/4,forward)
    magnitude=np.sum(forward**2,axis=-1)
    valid=(magnitude>1)&(fb<2)
    span=hi-lo
    estimates=[]
    for velocity in (va,vb):
        projected=np.sum(velocity*forward,axis=-1)/np.maximum(magnitude,.01)
        estimates.append(float(np.median(projected[valid]))*span if valid.any() else 1.)
    slopes=np.clip(estimates,0,3)
    length=np.linalg.norm(slopes)
    if length>3:slopes*=3/length
    # Keep the sequence strictly monotone with nonzero tiny endpoint movement.
    u=np.arange(1,span,dtype=float)/span
    progress=(-2*u**3+3*u**2)+(u**3-2*u**2+u)*slopes[0]+(u**3-u**2)*slopes[1]
    progress=np.clip(progress,1e-5,1-1e-5)
    return progress.tolist(),{"raw_normalized_slopes":estimates,"used_normalized_slopes":slopes.tolist(),
                              "valid_motion_fraction":float(valid.mean()),"progress":progress.tolist()}


def raw_variants(args,plan,model):
    reports=[]
    for seam in args.seams:
        frames=list(read_frames(args.source,seam-24,48,size=(640,360)))
        methods=[("Original",frames)]
        variants=[]
        for before,after in ((4,7),(6,9)):
            lo,hi=24-before,24+after
            interior=hi-lo-1
            hermite,timing_report=velocity_progress(frames,lo,hi)
            for timing,timesteps in (("linear",None),("hermite",hermite)):
                name=f"raw-{before+after+1}-{timing}"
                tweens,inference=model.interpolate(frames[lo],frames[hi],interior,timesteps)
                candidate=frames.copy();candidate[lo:hi+1]=tweens
                methods.append((name,candidate))
                curve=[pair_metrics(candidate[j-1],candidate[j]) for j in range(lo,hi+2)]
                basecurve=[pair_metrics(frames[j-1],frames[j]) for j in range(lo,hi+2)]
                sharp=[sharpness(f) for f in tweens]
                variant={"name":name,"endpoint_frames":[seam-before,seam+after],
                         "inference":inference,"timing":timing_report if timing=="hermite" else None,
                         "registered_curve":[m["registered_mae"] for m in curve],
                         "raw_curve":[m["raw_mae"] for m in curve],
                         "motion_curve":[m["motion_p50"] for m in curve],
                         "baseline_motion_curve":[m["motion_p50"] for m in basecurve],
                         "sharpness_curve":sharp,
                         "endpoint_sharpness_min":min(sharp[0],sharp[-1]),
                         "tween_sharpness_min":min(sharp[1:-1]),
                         "shape_preserved":True}
                variants.append(variant)
                np.savez_compressed(args.out/f"seam-{seam}-{name}.npz",frames=np.stack(candidate),
                                    source_indices=np.arange(seam-24,seam+24))
                with VideoWriter(args.out/f"seam-{seam}-{name}-comparison.mp4",1280,360,
                                 plan["source"]["fps_fraction"],crf=15,preset="fast") as writer:
                    for _ in range(3):
                        for a,b in zip(frames,candidate):
                            image=np.concatenate((a,b),axis=1).copy()
                            cv2.putText(image,"ORIGINAL",(10,22),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1,cv2.LINE_AA)
                            cv2.putText(image,name,(650,22),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1,cv2.LINE_AA)
                            writer.write(image)
                print(f"{seam} {name}: mean residual={np.mean(variant['registered_curve']):.3f}, "
                      f"min sharpness ratio={variant['tween_sharpness_min']/variant['endpoint_sharpness_min']:.3f}, "
                      f"motion={np.round(variant['motion_curve'],2).tolist()}",flush=True)
        sheet=Image.new("RGB",(480*5,344*len(methods)),"#131820")
        draw=ImageDraw.Draw(sheet)
        for row,(name,sequence) in enumerate(methods):
            for col,index in enumerate((20,22,24,26,30)):
                crop=Image.fromarray(sequence[index]).crop((200,100,440,260)).resize((480,320),Image.Resampling.NEAREST)
                sheet.paste(crop,(col*480,row*344))
                draw.text((col*480+5,row*344+324),f"{name} | {seam-24+index}",fill="white")
        sheet.save(args.out/f"seam-{seam}-raw-variants-crops.png")
        reports.append({"frame":seam,"variants":variants})
        (args.out/"raw-variants-report.json").write_text(json.dumps({"device":str(model.device),"seams":reports},indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",type=Path,default=ROOT/"IYTYT.mp4")
    parser.add_argument("--plan",type=Path,default=ROOT/"output/v1/plan.json")
    parser.add_argument("--seams",type=int,nargs="+",default=[361,3240])
    parser.add_argument("--out",type=Path,default=ROOT/"research/rife/results")
    parser.add_argument("--device",default="auto",choices=["auto","cpu","mps"])
    parser.add_argument("--raw-variants",action="store_true")
    args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    plan=json.loads(args.plan.read_text())
    model=RifeInterpolator(device=args.device)
    print(f"Official RIFE4.25 loaded on {model.device}",flush=True)
    if args.raw_variants:
        raw_variants(args,plan,model)
        return
    reports=[]
    fps=plan["source"]["fps_fraction"]
    for seam in args.seams:
        raw=list(read_frames(args.source,seam-24,48,size=(640,360)))
        geometry=raw.copy()
        geometry[10:46],geometry_report=corrected_window(raw[10:46],14,plan["config"])
        # Anchors n-4,n+5 enclose exactly eight newly inferred sample positions.
        lo,hi=20,29
        source_tweens,source_report=model.interpolate(raw[lo],raw[hi])
        geometry_tweens,tween_report=model.interpolate(geometry[lo],geometry[hi])
        source_bridge=raw.copy();source_bridge[lo:hi+1]=source_tweens
        bridge=geometry.copy();bridge[lo:hi+1]=geometry_tweens
        methods=[("original",raw),("geometry",geometry),
                 ("source RIFE bridge",source_bridge),("geometry RIFE bridge",bridge)]
        report={"frame":seam,"source_bridge":source_report,"geometry_bridge":tween_report,
                "geometry_report":geometry_report,"methods":{}}
        for label,frames in methods:
            metrics=[pair_metrics(frames[j-1],frames[j]) for j in range(lo,hi+2)]
            sharp=[sharpness(frames[j]) for j in range(lo,hi+1)]
            report["methods"][label]={"registered_curve":[m["registered_mae"] for m in metrics],
                                       "raw_curve":[m["raw_mae"] for m in metrics],
                                       "motion_curve":[m["motion_p50"] for m in metrics],
                                       "sharpness":sharp}
        # Individual lossless frames enable scrutiny of actual line contours.
        frame_dir=args.out/f"seam-{seam}-frames"
        frame_dir.mkdir(exist_ok=True)
        for i,frame in enumerate(geometry_tweens):
            Image.fromarray(frame).save(frame_dir/f"{seam-4+i:05d}.png")
        montage=Image.new("RGB",(320*10,204*4),"#141821")
        draw=ImageDraw.Draw(montage)
        for row,(label,frames) in enumerate(methods):
            for col,index in enumerate(range(lo,hi+1)):
                small=Image.fromarray(frames[index]).resize((320,180),Image.Resampling.LANCZOS)
                montage.paste(small,(col*320,row*204))
                draw.text((col*320+5,row*204+184),f"{label} | {seam-24+index}",fill="white")
        montage.save(args.out/f"seam-{seam}-montage.jpg",quality=95)
        # 2x crops of selected generated frames expose doubled contour ghosts.
        crop=Image.new("RGB",(480*5,344*2),"#141821")
        draw=ImageDraw.Draw(crop)
        boxes=[(200,100,440,260),(40,35,280,195)]
        for row,box in enumerate(boxes):
            for col,index in enumerate([0,2,4,6,9]):
                image=Image.fromarray(geometry_tweens[index]).crop(box).resize((480,320),Image.Resampling.NEAREST)
                crop.paste(image,(col*480,row*344))
                draw.text((col*480+6,row*344+324),f"frame {seam-4+index}"+(" ENDPOINT" if index in [0,9] else " RIFE TWEEN"),fill="white")
        crop.save(args.out/f"seam-{seam}-tween-crops.png")
        for name,left,right in (("comparison",geometry,bridge),("source-comparison",raw,source_bridge)):
            with VideoWriter(args.out/f"seam-{seam}-{name}.mp4",1280,360,fps,crf=15,preset="fast") as writer:
                for _ in range(4):
                    for index,(a,b) in enumerate(zip(left,right)):
                        image=np.concatenate([a,b],axis=1).copy()
                        cv2.putText(image,"BASELINE",(10,22),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1,cv2.LINE_AA)
                        cv2.putText(image,"RIFE BRIDGE",(650,22),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1,cv2.LINE_AA)
                        writer.write(image)
        with VideoWriter(args.out/f"seam-{seam}-bridge-loop.mp4",640,360,fps,crf=15,preset="fast") as writer:
            for _ in range(4):
                for frame in bridge:writer.write(frame)
        reports.append(report)
        print(f"{seam}:8 tweens {tween_report['runtime_seconds']:.2f}s, "
              f"endpoint exact={tween_report['endpoint_preserved']}, "
              f"mean registered baseline={np.mean(report['methods']['geometry']['registered_curve']):.3f}, "
              f"bridge={np.mean(report['methods']['geometry RIFE bridge']['registered_curve']):.3f}",flush=True)
        (args.out/"report.json").write_text(json.dumps({"device":str(model.device),"model":"official RIFE4.25","seams":reports},indent=2))


if __name__=="__main__":
    main()
