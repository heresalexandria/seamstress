"""Distinguish a color discontinuity from regression bias and within-clip change."""
from __future__ import annotations
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from segment_color import SOURCE, OUT, matches, robust_fit, read_frames


def symmetric_fit(x, y):
    values = []
    for c in range(3):
        keep = (x[:, c]>8)&(x[:, c]<247)&(y[:, c]>8)&(y[:, c]<247)
        a, b = x[keep, c].astype(float), y[keep, c].astype(float)
        weight = np.ones(len(a)); gain=1.; bias=0.
        for _ in range(15):
            am, bm = np.average(a, weights=weight), np.average(b, weights=weight)
            av = np.average((a-am)**2, weights=weight); bv = np.average((b-bm)**2, weights=weight)
            cov = np.average((a-am)*(b-bm), weights=weight)
            gain = (bv-av+np.sqrt((bv-av)**2+4*cov**2))/(2*cov)
            bias = bm-gain*am
            residual = b-(gain*a+bias)
            scale = max(1.5, 1.4826*np.median(abs(residual-np.median(residual))))
            weight = np.minimum(1., 1.5*scale/np.maximum(abs(residual), 1e-8))
        values.append([float(gain), float(bias)])
    return {"gain": [v[0] for v in values], "bias": [v[1] for v in values]}


def main():
    results=[]
    for cut in [361, 722, 2166, 3240]:
        frames = read_frames(SOURCE, cut-6, 12, size=(640, 360))
        pairs = []
        for left, right, label in [(0, 2, "within_before"), (5, 6, "across_cut"), (8, 10, "within_after")]:
            x, y, train, _, coverage = matches(frames[left], frames[right])
            gain, bias = robust_fit(x[train], y[train])
            record = {"kind": label, "left": cut-6+left, "right": cut-6+right,
                      "regularized_gain": gain.tolist(), "regularized_bias": bias.tolist(),
                      "symmetric_total_least_squares": symmetric_fit(x[train], y[train]), "coverage": coverage}
            pairs.append(record)
            print(cut, label, "regularized", np.round(gain, 4), np.round(bias, 2),
                  "symmetric", record["symmetric_total_least_squares"], flush=True)
        results.append({"frame": cut, "pairs": pairs})
        (OUT/"estimator-controls.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__": main()
