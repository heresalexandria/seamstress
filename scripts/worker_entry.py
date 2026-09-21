"""Frozen desktop-worker entry point; keep this module free of developer paths."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys


def configure_environment():
    if getattr(sys, 'frozen', False):
        backend = Path(sys.executable).resolve().parent.parent
        tools = backend/'bin'
        os.environ['PATH'] = str(tools)+os.pathsep+os.environ.get('PATH', '')
    else:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
    os.environ.setdefault('OMP_NUM_THREADS', '2')


def self_test():
    import cv2
    import numpy as np
    import scipy
    import onnxruntime
    from scipy.linalg import expm, logm
    from scipy.ndimage import percentile_filter
    from scipy.signal import find_peaks
    from scipy.interpolate import PchipInterpolator
    from seamstress.calibration import calibrate_pair
    from seamstress.media import _tool
    from seamstress.pipeline import run_stage
    cv2.SIFT_create()
    assert 'CPUExecutionProvider' in onnxruntime.get_available_providers()
    np.testing.assert_allclose(expm(logm(np.eye(3))), np.eye(3))
    assert percentile_filter(np.arange(9).reshape(3, 3), 50, size=3).shape == (3, 3)
    assert len(find_peaks([0., 1., 0.])[0]) == 1
    assert float(PchipInterpolator([0, 1], [0, 1])(.5)) == .5
    ffmpeg, ffprobe = _tool('ffmpeg'), _tool('ffprobe')
    for program in (ffmpeg, ffprobe):
        subprocess.run([program, '-version'], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    frame = np.full((40, 64, 3), 90, np.uint8)
    assert calibrate_pair(frame, frame)['accepted']
    # Exercise bundled libx264 encode and decode, not merely version parsing.
    args = [ffmpeg, '-v', 'error', '-f', 'lavfi', '-i', 'color=c=blue:s=64x48:r=24',
            '-frames:v', '2', '-c:v', 'libx264', '-preset', 'ultrafast', '-f', 'matroska', 'pipe:1']
    movie = subprocess.run(args, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout
    info = subprocess.run([ffprobe, '-v', 'error', '-show_streams', '-of', 'json', 'pipe:0'],
                          input=movie, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert json.loads(info.stdout)['streams'][0]['width'] == 64
    decoded = subprocess.run([ffmpeg, '-v', 'error', '-i', 'pipe:0', '-f', 'rawvideo',
                              '-pix_fmt', 'rgb24', 'pipe:1'], input=movie, check=True,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout
    assert len(decoded) == 2*64*48*3
    decoded_rgb = np.frombuffer(decoded, np.uint8).reshape(2, 48, 64, 3)
    assert float(decoded_rgb[..., 2].mean()) > 240
    assert float(decoded_rgb[..., :2].mean()) < 10
    print(json.dumps({'self_test': 'ok', 'numpy': np.__version__, 'scipy': scipy.__version__,
                      'opencv': cv2.__version__, 'onnxruntime': onnxruntime.__version__,
                      'ffmpeg': ffmpeg, 'ffprobe': ffprobe}))
    return 0


def main():
    configure_environment()
    if sys.argv[1:] == ['--self-test']:
        return self_test()
    from seamstress.desktop_worker import main as desktop_main
    return desktop_main()


if __name__ == '__main__':
    raise SystemExit(main())
