"""Measure long, symmetric photometric easing without rendering a full video."""
from __future__ import annotations
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from PIL import Image, ImageDraw
from segment_color import ROOT, OUT, SOURCE, CUTS, matches, robust_fit, iter_frames

FPS = 24000/1001
HALF_SUPPORT = round(7*FPS)


def smoothstep(t):
    t = np.clip(t, 0., 1.)
    return t*t*t*(10+t*(-15+6*t))


def power(grade, exponent):
    gain, bias = np.array(grade["gain"]), np.array(grade["bias"])
    powered_gain = gain**exponent
    ratio = np.where(abs(gain-1) < 1e-7, exponent, (powered_gain-1)/(gain-1))
    return powered_gain, bias*ratio


def transform(frame, estimates):
    gain, bias = np.ones(3), np.zeros(3)
    active = []
    for fit in estimates:
        cut = fit["frame"]
        distance = cut-1-frame if frame < cut else frame-cut
        if 0 <= distance <= HALF_SUPPORT:
            exponent = (.5 if frame >= cut else -.5)*(1-smoothstep(distance/HALF_SUPPORT))
            g, b = power(fit, exponent)
            # Supports are disjoint for this source; do not hide overlap.
            if active: raise ValueError("Photometric supports overlap")
            gain, bias = g, b
            active.append(cut)
    return gain, bias


def main():
    estimates = json.loads((OUT/"estimates.json").read_text())
    wanted = {}; control_starts = []
    for index in range(24, 1080, 24):
        if not any(index < cut <= index+4 for cut in CUTS):
            control_starts.append(index); wanted[index] = True; wanted[index+4] = True
    for cut in CUTS:
        for delta in [-HALF_SUPPORT, -126, -84, -42, -1, 0, 42, 84, 126, HALF_SUPPORT]:
            frame = cut+delta
            if 0 <= frame < 3347: wanted[frame] = True
    frames = {}
    for index, frame in enumerate(iter_frames(SOURCE, size=(640, 360))):
        if index in wanted: frames[index] = frame
    controls=[]
    palette = np.array([16., 48., 96., 144., 192., 240.])[:, None]
    for index in control_starts:
        x, y, train, _, coverage = matches(frames[index], frames[index+4])
        gain, bias = robust_fit(x[train], y[train])
        predicted_delta = ((gain-1)*palette+bias)/4
        record = {"frame": index, "gap": 4, "gain_later_to_earlier": gain.tolist(), "bias_later_to_earlier": bias.tolist(),
                  "median_abs_grade_change_per_frame": float(np.median(abs(predicted_delta))),
                  "max_abs_grade_change_per_frame": float(np.max(abs(predicted_delta))),
                  "signed_median_matched_pixel_change_per_frame": (np.median(y-x, axis=0)/4).tolist()}
        controls.append(record)
        if len(controls)%10==0:print("CONTROL", len(controls), "/", len(control_starts), flush=True)
    tracks = []
    for cut in CUTS:
        records=[]
        for frame in range(max(0, cut-HALF_SUPPORT-2), min(3347, cut+HALF_SUPPORT+3)):
            gain, bias = transform(frame, estimates)
            records.append({"frame": frame, "gain": gain.tolist(), "bias": bias.tolist(),
                            "palette": (gain*palette+bias).tolist()})
        rates=[]
        for left, right in zip(records, records[1:]):
            if right["frame"] == cut: continue  # This parameter reset cancels the source grade reset.
            rate=abs(np.array(right["palette"])-np.array(left["palette"]))
            rates.append({"frame": right["frame"], "median": float(np.median(rate)), "max": float(rate.max())})
        peak=max(rates,key=lambda r:r["max"])
        a,b=transform(cut-1,estimates),transform(cut,estimates)
        tracks.append({"cut":cut,"peak_change":peak,"left_endpoint_gain":a[0].tolist(),"left_endpoint_bias":a[1].tolist(),
                       "right_endpoint_gain":b[0].tolist(),"right_endpoint_bias":b[1].tolist(),"rates":rates,"track":records})
    clips=[]
    for frame, image in frames.items():
        gain,bias=transform(frame,estimates)
        mapped=image.astype(float)*gain+bias
        clips.append({"frame":frame,"new_underflow_fraction_rgb":np.mean((mapped<0)&(image>0),axis=(0,1)).tolist(),
                      "new_overflow_fraction_rgb":np.mean((mapped>255)&(image<255),axis=(0,1)).tolist()})
    result={"half_support_frames":HALF_SUPPORT,"half_support_seconds":HALF_SUPPORT/FPS,"controls":controls,"tracks":tracks,"range_checks":clips}
    (OUT/"slow-grade-analysis.json").write_text(json.dumps(result,indent=2))
    summary={"normal_control_median_rates_quantiles":np.percentile([c["median_abs_grade_change_per_frame"] for c in controls],[10,50,75,90,95]).tolist(),
             "normal_control_max_rates_quantiles":np.percentile([c["max_abs_grade_change_per_frame"] for c in controls],[10,50,75,90,95]).tolist(),
             "peak_correction_rate_by_cut":[{"cut":t["cut"],**t["peak_change"]} for t in tracks],
             "largest_new_underflows":sorted(clips,key=lambda c:max(c["new_underflow_fraction_rgb"]),reverse=True)[:10],
             "largest_new_overflows":sorted(clips,key=lambda c:max(c["new_overflow_fraction_rgb"]),reverse=True)[:10]}
    (OUT/"slow-grade-summary.json").write_text(json.dumps(summary,indent=2))
    canvas=Image.new("RGB",(1400,850),"#131922");draw=ImageDraw.Draw(canvas)
    for row,cut in enumerate([361,722]):
        t=next(t for t in tracks if t["cut"]==cut); records=t["track"]
        left,right,top,bottom=75,1370,40+row*410,390+row*410
        xmin,xmax=records[0]["frame"],records[-1]["frame"]
        values=[r["palette"][2][c]-96 for r in records for c in range(3)]
        ymin,ymax=min(values)-1,max(values)+1
        xy=lambda x,y:(left+(right-left)*(x-xmin)/(xmax-xmin),bottom-(bottom-top)*(y-ymin)/(ymax-ymin))
        draw.text((left,top-22),f"{cut}: added correction at RGB 96, byte values; 7 s each side",fill="white")
        draw.line((left,xy(cut,0)[1],right,xy(cut,0)[1]),fill="#46515e")
        draw.line((xy(cut,0)[0],top,xy(cut,0)[0],bottom),fill="#46515e")
        for c,color in enumerate(["#ff8d8d","#8ef0af","#90b5ff"]):
            for part in [[r for r in records if r["frame"]<cut],[r for r in records if r["frame"]>=cut]]:
                draw.line([xy(r["frame"],r["palette"][2][c]-96) for r in part],fill=color,width=2)
        for x in [cut-HALF_SUPPORT,cut,cut+HALF_SUPPORT]:draw.text((xy(x,0)[0]-15,bottom+8),str(x),fill="white")
        draw.text((8,top),f"{ymax:.1f}",fill="white");draw.text((8,bottom-12),f"{ymin:.1f}",fill="white")
    canvas.save(OUT/"slow-grade-track.png")
    print(json.dumps(summary,indent=2),flush=True)


if __name__=="__main__":main()
