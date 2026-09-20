"""Color-only research: robust matched-pixel grades held constant per source clip.

No frame interpolation, geometric warp, temporal blend, or source overwrite.
Flow is used only to identify samples for estimating one grade per segment.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
from PIL import Image, ImageDraw

from seamstress.media import VideoWriter, probe, read_frames, iter_frames
from seamstress.repair import flow, grid, sample

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "research/segment-color"
SOURCE = ROOT / "IYTYT.mp4"
CUTS = [361, 722, 1083, 1444, 1805, 2166, 2527, 2888, 3240]


def matches(left, right):
    """Return right→left color observations from consistent interior matches."""
    fw, bw = flow(left, right), flow(right, left)
    xy = grid(fw.shape) + fw
    consistent = np.linalg.norm(fw + sample(bw, fw), axis=2) < 1.0
    left = cv2.GaussianBlur(left.astype(np.float32), (0, 0), .8)
    right = sample(cv2.GaussianBlur(right.astype(np.float32), (0, 0), .8), fw)
    ga, gb = [cv2.cvtColor(x, cv2.COLOR_RGB2GRAY) for x in (left, right)]
    gradient = np.maximum(cv2.magnitude(cv2.Sobel(ga, cv2.CV_32F, 1, 0), cv2.Sobel(ga, cv2.CV_32F, 0, 1)),
                          cv2.magnitude(cv2.Sobel(gb, cv2.CV_32F, 1, 0), cv2.Sobel(gb, cv2.CV_32F, 0, 1)))
    h, w = consistent.shape
    valid = consistent & (gradient < 28) & (np.max(abs(left-right), axis=2) < 36)
    valid &= (xy[..., 0] > 6) & (xy[..., 0] < w-7) & (xy[..., 1] > 6) & (xy[..., 1] < h-7)
    valid[:6] = False; valid[-6:] = False; valid[:, :6] = False; valid[:, -6:] = False
    # Balanced spatial sampling: a large flat sky does not dominate the fit.
    x, y, train, tiles = [], [], [], []
    rng = np.random.default_rng(9183)
    for row in range(8):
        for col in range(12):
            y0, y1, x0, x1 = row*h//8, (row+1)*h//8, col*w//12, (col+1)*w//12
            yy, xx = np.nonzero(valid[y0:y1, x0:x1]); yy += y0; xx += x0
            if len(yy) < 30:
                continue
            chosen = rng.choice(len(yy), min(250, len(yy)), replace=False)
            yy, xx = yy[chosen], xx[chosen]
            x.append(right[yy, xx]); y.append(left[yy, xx])
            train.extend([((row+col) % 2) == 0] * len(yy))
            tiles.extend([row*12+col]*len(yy))
    if not x:
        raise ValueError("No consistent matched interior pixels")
    return np.concatenate(x), np.concatenate(y), np.array(train), np.array(tiles), float(valid.mean())


def robust_fit(x, y):
    """Diagonal RGB affine fit with Huber residual weights and gain regularization."""
    gains, biases = [], []
    for c in range(3):
        valid = (x[:, c] > 8) & (x[:, c] < 247) & (y[:, c] > 8) & (y[:, c] < 247)
        xc, yc = x[valid, c]/255., y[valid, c]/255.
        if len(xc) < 200 or xc.std() < .06:
            raise ValueError("Insufficient channel intensity range for grade estimation")
        a = np.column_stack((xc, np.ones(len(xc))))
        coef = np.array([1., np.median(yc-xc)])
        # Mild identity-gain prior limits unstable black/white extrapolation.
        # This is deliberately reported rather than silently forcing exact matching.
        regularizer = np.diag([len(xc)*.002, len(xc)*.00002])
        prior = np.array([1., 0.])
        for _ in range(15):
            err = yc-a@coef
            scale = max(1.5/255, 1.4826*float(np.median(abs(err-np.median(err)))))
            weights = np.minimum(1., (1.5*scale)/np.maximum(abs(err), 1e-8))
            coef = np.linalg.solve(a.T@(a*weights[:, None])+regularizer, a.T@(yc*weights)+regularizer@prior)
        gains.append(float(coef[0])); biases.append(float(coef[1]*255))
    return np.array(gains), np.array(biases)


def pair_error(x, y, gain, bias):
    raw = abs(y-x); fixed = abs(y-(x*gain+bias))
    return {"raw_mean_abs": float(raw.mean()), "fixed_mean_abs": float(fixed.mean()),
            "raw_p90_abs": float(np.percentile(raw, 90)), "fixed_p90_abs": float(np.percentile(fixed, 90)),
            "raw_signed_median_rgb": np.median(y-x, axis=0).tolist(),
            "fixed_signed_median_rgb": np.median(y-(x*gain+bias), axis=0).tolist()}


def estimate(cut):
    frames = read_frames(SOURCE, cut-3, 6, size=(640, 360))
    observations = []; fits = []
    for a, b in [(2, 3), (1, 4), (0, 5)]:
        x, y, train, tiles, coverage = matches(frames[a], frames[b])
        gain, bias = robust_fit(x[train], y[train])
        fits.append({"left_frame": cut-3+a, "right_frame": cut-3+b,
                     "gain": gain.tolist(), "bias": bias.tolist(), "valid_fraction": coverage,
                     "samples": len(x), "heldout_spatial_tiles": pair_error(x[~train], y[~train], gain, bias)})
        observations.append((x, y, train))
    x, y, train = [np.concatenate([o[i] for o in observations]) for i in range(3)]
    gain, bias = robust_fit(x[train], y[train])
    # Cross-pair coefficient agreement is a check against actual lighting/motion
    # being mistaken for the generation's grade reset.
    return {"frame": cut, "gain": gain.tolist(), "bias": bias.tolist(),
            "gain_pair_spread": np.ptp([f["gain"] for f in fits], axis=0).tolist(),
            "bias_pair_spread": np.ptp([f["bias"] for f in fits], axis=0).tolist(),
            "heldout_spatial_tiles": pair_error(x[~train], y[~train], gain, bias), "pairs": fits}


def apply(frame, grade):
    # Every sample retains its original frame index and pixel location.
    return np.rint(np.clip(frame.astype(np.float32)*grade["gain"]+grade["bias"], 0, 255)).astype(np.uint8)


def propagate(estimates):
    grades = [{"segment": 0, "start": 0, "end_exclusive": CUTS[0], "gain": [1., 1., 1.], "bias": [0., 0., 0.]}]
    for i, fit in enumerate(estimates):
        previous = grades[-1]
        pg, pb = np.array(previous["gain"]), np.array(previous["bias"])
        gain, bias = pg*np.array(fit["gain"]), pb+pg*np.array(fit["bias"])
        grades.append({"segment": i+1, "start": fit["frame"], "end_exclusive": CUTS[i+1] if i+1 < len(CUTS) else 3347,
                       "gain": gain.tolist(), "bias": bias.tolist()})
    return grades


def dynamic_range(grades):
    """Read sparse samples across all source segments; this is not a full render."""
    wanted = {}
    for grade in grades:
        indices = np.linspace(grade["start"]+3, grade["end_exclusive"]-4, 7).round().astype(int)
        wanted.update({int(i): grade["segment"] for i in indices})
    samples = {i: [] for i in range(len(grades))}
    # One ffmpeg decode prevents repeated long seeks through the same source.
    for index, frame in enumerate(iter_frames(SOURCE, size=(320, 180))):
        if index in wanted:
            samples[wanted[index]].append(frame)
    result = []
    for grade in grades:
        raw = np.stack(samples[grade["segment"]]).astype(np.float32)
        mapped = raw*np.array(grade["gain"])+np.array(grade["bias"])
        result.append({"segment": grade["segment"], "sample_count": len(raw),
                       "original_channel_percentiles": np.percentile(raw, [0.1, 1, 50, 99, 99.9], axis=(0, 1, 2)).tolist(),
                       "mapped_channel_percentiles": np.percentile(mapped, [0.1, 1, 50, 99, 99.9], axis=(0, 1, 2)).tolist(),
                       "underflow_fraction_rgb": np.mean(mapped<0, axis=(0, 1, 2)).tolist(),
                       "overflow_fraction_rgb": np.mean(mapped>255, axis=(0, 1, 2)).tolist(),
                       "mean_abs_change_rgb": np.mean(abs(mapped-raw), axis=(0, 1, 2)).tolist()})
    return result


def previews(grades, metadata):
    for cut in CUTS[:2]:
        start, count = cut-48, 96
        original = read_frames(SOURCE, start, count)
        fixed = [apply(frame, grades[int(np.searchsorted(CUTS, start+i, side="right"))]) for i, frame in enumerate(original)]
        for name, frames in (("source", original), ("color", fixed)):
            destination = OUT/f"{cut}-{name}.mp4"
            if destination.exists():
                raise FileExistsError(destination)
            with VideoWriter(destination, 1280, 720, metadata["fps_fraction"], crf=16) as writer:
                for frame in frames:
                    writer.write(frame)
        comparison = OUT/f"{cut}-comparison.mp4"
        with VideoWriter(comparison, 2560, 752, metadata["fps_fraction"], crf=16) as writer:
            for i, (a, b) in enumerate(zip(original, fixed)):
                image = Image.new("RGB", (2560, 752), "#121820")
                image.paste(Image.fromarray(a), (0, 32)); image.paste(Image.fromarray(b), (1280, 32))
                draw = ImageDraw.Draw(image)
                draw.text((20, 10), f"ORIGINAL - frame {start+i}", fill="white")
                draw.text((1300, 10), "CONSTANT SEGMENT COLOR - original positions/poses/cadence", fill="white")
                writer.write(np.array(image))
        # Native before/after columns for the last pre-cut and first post-cut frame.
        sheet = Image.new("RGB", (2560, 1472), "#121820"); draw = ImageDraw.Draw(sheet)
        for row, i in enumerate((47, 48)):
            sheet.paste(Image.fromarray(original[i]), (0, row*736+16))
            sheet.paste(Image.fromarray(fixed[i]), (1280, row*736+16))
            draw.text((8, row*736), f"SOURCE {start+i}", fill="white")
            draw.text((1288, row*736), f"COLOR ONLY {start+i}", fill="white")
        sheet.save(OUT/f"{cut}-cut.jpg", quality=96)
        print("PREVIEW", cut, flush=True)


def main():
    OUT.mkdir(exist_ok=True)
    estimates = []
    for cut in CUTS:
        result = estimate(cut); estimates.append(result)
        (OUT/"estimates.json").write_text(json.dumps(estimates, indent=2))
        print("FIT", cut, "gain", np.round(result["gain"], 4), "bias", np.round(result["bias"], 3),
              "holdout", result["heldout_spatial_tiles"], flush=True)
    grades = propagate(estimates)
    (OUT/"grades.json").write_text(json.dumps(grades, indent=2))
    ranges = dynamic_range(grades)
    (OUT/"dynamic-range.json").write_text(json.dumps(ranges, indent=2))
    previews(grades, probe(SOURCE))


if __name__ == "__main__":
    main()
