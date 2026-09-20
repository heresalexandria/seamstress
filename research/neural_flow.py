"""Compare official pretrained Torchvision RAFT with the current DIS estimator.

All model downloads are confined to research/models. Input is the actual source
video; no generated textures or blended frames influence correspondence scores.
Official preprocessing reference:
https://docs.pytorch.org/vision/0.24/auto_examples/others/plot_optical_flow.html
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
from PIL import Image, ImageDraw
import torch
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights

from seamstress.media import read_frames
from seamstress.repair import flow, grid, sample, robust_color


class RaftEstimator:
    def __init__(self, device="auto", model_dir=Path("research/models"), updates=12):
        if device == "auto":
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = torch.device(device)
        self.updates = updates
        torch.set_num_threads(4)
        weights = Raft_Large_Weights.DEFAULT
        model_dir.mkdir(parents=True, exist_ok=True)
        state = torch.hub.load_state_dict_from_url(weights.url, model_dir=str(model_dir),
                                                  check_hash=True, weights_only=True)
        self.model = raft_large(weights=None).eval().to(self.device)
        self.model.load_state_dict(state)
        self.transforms = weights.transforms()
        self.weights_url = weights.url

    def synchronize(self):
        if self.device.type == "mps":
            torch.mps.synchronize()

    def pair(self, left, right):
        """Both directions as one batch; output flow is in input-pixel units."""
        h, w = left.shape[:2]
        if h % 8 or w % 8:
            raise ValueError("RAFT inputs must have dimensions divisible by eight")
        a = torch.from_numpy(np.stack([left, right])).permute(0, 3, 1, 2)
        b = torch.from_numpy(np.stack([right, left])).permute(0, 3, 1, 2)
        a, b = self.transforms(a, b)
        a, b = a.to(self.device), b.to(self.device)
        self.synchronize()
        start = time.perf_counter()
        with torch.inference_mode():
            result = self.model(a, b, num_flow_updates=self.updates)[-1]
        self.synchronize()
        seconds = time.perf_counter() - start
        array = result.detach().cpu().permute(0, 2, 3, 1).numpy()
        return array[0], array[1], seconds


def evaluate(left, right, forward, backward, runtime):
    warped = sample(right, forward, cv2.INTER_CUBIC)
    fb = np.linalg.norm(forward + sample(backward, forward), axis=2)
    coordinates = grid(left.shape) + forward
    h, w = left.shape[:2]
    inside = ((coordinates[..., 0] >= 2) & (coordinates[..., 0] < w-3) &
              (coordinates[..., 1] >= 2) & (coordinates[..., 1] < h-3))
    valid = inside & (fb < 2)
    gain, bias = robust_color(warped, left, valid)
    corrected = np.clip(warped.astype(float)*gain+bias, 0, 255)
    raw_error = np.mean(np.abs(left.astype(float) - warped.astype(float)), axis=2)
    corrected_error = np.mean(np.abs(left.astype(float) - corrected), axis=2)
    gray_a = cv2.cvtColor(left, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gray_b = cv2.cvtColor(warped, cv2.COLOR_RGB2GRAY).astype(np.float32)
    edges_a = cv2.Canny(left, 45, 100)
    edges_b = cv2.Canny(warped, 45, 100)
    dist_a = cv2.distanceTransform((edges_a == 0).astype(np.uint8), cv2.DIST_L2, 5)
    dist_b = cv2.distanceTransform((edges_b == 0).astype(np.uint8), cv2.DIST_L2, 5)
    edge_missing = ((edges_a > 0) & (dist_b > 1)) | ((edges_b > 0) & (dist_a > 1))
    dy, dx = np.gradient(forward, axis=(0, 1))
    determinant = (1+dx[..., 0])*(1+dy[..., 1]) - dy[..., 0]*dx[..., 1]
    report = {
        "bidirectional_runtime_seconds": runtime,
        "registered_mae_all": float(raw_error.mean()),
        "registered_mae_inside": float(raw_error[inside].mean()),
        "color_corrected_mae_inside": float(corrected_error[inside].mean()),
        "color_corrected_mae_consistent": float(corrected_error[valid].mean()),
        "fb_error_p50": float(np.median(fb[inside])),
        "fb_error_p95": float(np.percentile(fb[inside],95)),
        "consistent_fraction_2px": float(valid.mean()),
        "consistent_fraction_1px": float(np.mean(inside & (fb < 1))),
        "inbounds_fraction": float(inside.mean()),
        "unmatched_edges_fraction": float(edge_missing[inside].mean()),
        "raw_flow_fold_fraction": float(np.mean(determinant[inside] <= 0)),
        "warped_sharpness_ratio": float(np.mean(np.abs(cv2.Laplacian(gray_b,cv2.CV_32F))) /
                                         np.mean(np.abs(cv2.Laplacian(gray_a,cv2.CV_32F)))),
        "gain": gain.tolist(), "bias": bias.tolist(),
    }
    confidence = np.exp(-(fb/1.5)**2)*inside
    return report, warped, np.clip(corrected_error[...,None]*6,0,255).repeat(3,axis=2).astype(np.uint8), confidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("IYTYT.mp4"))
    parser.add_argument("--out", type=Path, default=Path("research/neural"))
    parser.add_argument("--seams", type=int, nargs="+", default=[2888,3240])
    parser.add_argument("--device", default="auto", choices=["auto","cpu","mps"])
    parser.add_argument("--updates", type=int, default=12)
    args = parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    estimator = RaftEstimator(device=args.device, updates=args.updates)
    print(f"RAFT loaded on {estimator.device}; updates={args.updates}",flush=True)
    records=[]
    for seam in args.seams:
        left,right=read_frames(args.source,seam-1,2,size=(640,360))
        start=time.perf_counter()
        dis_forward,dis_backward=flow(left,right),flow(right,left)
        dis_runtime=time.perf_counter()-start
        raft_forward,raft_backward,raft_runtime=estimator.pair(left,right)
        variants=[("DIS",dis_forward,dis_backward,dis_runtime),
                  ("RAFT",raft_forward,raft_backward,raft_runtime)]
        sheet=Image.new("RGB",(640*4,390*2),"#161920")
        draw=ImageDraw.Draw(sheet)
        record={"frame":seam,"methods":{}}
        crop_sheet=Image.new("RGB",(480*4,340*2),"#161920")
        crop_draw=ImageDraw.Draw(crop_sheet)
        crop_regions=[(200,100,440,260),(40,35,280,195)]
        warps={}
        for row,(label,forward,backward,runtime) in enumerate(variants):
            report,warped,error,confidence=evaluate(left,right,forward,backward,runtime)
            record["methods"][label]=report
            warps[label]=warped
            np.savez_compressed(args.out/f"seam-{seam}-{label.lower()}-flow.npz",
                                forward=forward,backward=backward)
            confidence_rgb=(confidence[...,None]*255).repeat(3,axis=2).astype(np.uint8)
            for col,(tag,image) in enumerate((("reference left",left),("aligned right",warped),
                                              ("corrected residual x6",error),("flow consistency",confidence_rgb))):
                sheet.paste(Image.fromarray(image),(col*640,row*390))
                draw.text((col*640+8,row*390+365),f"{seam} | {label} | {tag}",fill="white")
        for row,box in enumerate(crop_regions):
            for col,(label,image) in enumerate((("reference left",left),("source right",right),
                                                ("DIS aligned",warps["DIS"]),("RAFT aligned",warps["RAFT"]))):
                crop=Image.fromarray(image).crop(box).resize((480,320),Image.Resampling.NEAREST)
                crop_sheet.paste(crop,(col*480,row*340))
                crop_draw.text((col*480+6,row*340+324),label,fill="white")
        sheet.save(args.out/f"seam-{seam}.jpg",quality=95)
        crop_sheet.save(args.out/f"seam-{seam}-crops.png")
        records.append(record)
        print(json.dumps(record),flush=True)
    report={"torch":torch.__version__,"device":str(estimator.device),
            "weights":estimator.weights_url,"updates":args.updates,"seams":records}
    (args.out/"report.json").write_text(json.dumps(report,indent=2))


if __name__=="__main__":
    main()
