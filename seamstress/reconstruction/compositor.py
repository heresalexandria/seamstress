"""Rigid native layers; no whole-image flow or interpolation of actor poses."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from ..media import _tool, iter_frames
from .bundle import asset_path, check_cancel, fresh_directory, load_bundle, matrix, report, rgb_hash


def warp(image, transform, size, *, border=cv2.BORDER_CONSTANT):
    return cv2.warpPerspective(image, np.asarray(transform, np.float64), size,
                               flags=cv2.INTER_LINEAR, borderMode=border)


def compose_frame(bundle, n, rgb):
    b = load_bundle(bundle)
    w, h = b['source']['width'], b['source']['height']
    if type(n) is not int or not 0 <= n < b['source']['frame_count']:
        raise ValueError('Invalid source frame index')
    if rgb.shape != (h,w,3) or rgb.dtype != np.uint8:
        raise ValueError('Reconstruction requires native RGB uint8 source frames')
    lo, hi = b['support']['start'], b['support']['end']
    if not lo <= n < hi:
        return rgb.copy(), {'frame':n, 'identity':True}
    row = b['frames'][str(n)]
    if row.get('source_rgb_sha256') and rgb_hash(rgb) != row['source_rgb_sha256']:
        raise ValueError(f'Frame {n} is not the original source RGB used by this bundle')
    transforms = [matrix(l['matrices'][str(n)]) for l in b['layers']]
    if all(np.array_equal(m,np.eye(3)) for m in transforms):
        return rgb.copy(), {'frame':n, 'identity':True}
    plate = np.asarray(Image.open(asset_path(b,row['plate'])).convert('RGB'))
    with np.load(asset_path(b,row['matte']), allow_pickle=False) as data:
        alpha, premultiplied, emission = (data[k].astype(np.float32) for k in ('alpha','premultiplied','emission'))
    color = np.zeros((h,w,3),np.float32)
    coverage = np.zeros((h,w),np.float32)
    weights = []
    backgrounds = sorted((l for l in b['layers'] if l['kind']=='background'),key=lambda l:l.get('order',0))
    for layer in backgrounds:
        mask = np.asarray(Image.open(asset_path(b,layer['mask_by_frame'][str(n)])).convert('L')).astype(np.float32)/255
        m = layer['matrices'][str(n)]
        p = warp(plate.astype(np.float32)*mask[:,:,None],m,(w,h))
        weight = warp(mask,m,(w,h)).clip(0,1)
        color = p + color*(1-weight[:,:,None])
        coverage = weight + coverage*(1-weight)
        weights.append(weight)
    color = np.divide(color,coverage[:,:,None],out=np.zeros_like(color),where=coverage[:,:,None]>1e-5)
    config = b.get('compositor',{})
    gaps = np.sum(weights,axis=0) < config.get('gap_threshold',.70)
    edge = np.zeros((h,w),bool)
    band = int(config.get('edge_band',0))
    if band:
        edge[:band]=True; edge[-band:]=True; edge[:,:band]=True; edge[:,-band:]=True
        fallback = warp(plate,backgrounds[0]['matrices'][str(n)],(w,h),border=cv2.BORDER_REPLICATE)
        color[gaps&edge] = fallback[gaps&edge]
    interior = gaps&~edge
    radius = float(config.get('gap_inpaint_radius',0))
    filled = np.rint(color).clip(0,255).astype(np.uint8)
    if interior.any() and radius > 0:
        filled = cv2.inpaint(filled,interior.astype(np.uint8)*255,radius,cv2.INPAINT_TELEA)
    elif interior.any():
        # A visible gap is a review failure; source fallback keeps a preview
        # inspectable without pretending unobserved content was recovered.
        filled[interior] = rgb[interior]
    foreground = next(l for l in b['layers'] if l['kind']=='foreground')
    f = foreground['matrices'][str(n)]
    aw = warp(alpha,f,(w,h)).clip(0,1)
    expected_area=float(alpha.sum())*float(np.linalg.det(np.asarray(f)[:2,:2]))
    lost=max(0.,expected_area-float(aw.sum()))
    result = warp(premultiplied,f,(w,h)) + (1-aw[:,:,None])*filled.astype(np.float32) + warp(emission,f,(w,h))
    # This is a same-frame reconstruction residual, never a pose crossfade.
    strength = float(row.get('residual_strength',1.))
    if strength < 1:
        residual = rgb.astype(np.float32) - (premultiplied+(1-alpha[:,:,None])*plate.astype(np.float32)+emission)
        result += warp(residual,f,(w,h))*(1-strength)
    distance = cv2.distanceTransform(interior.astype(np.uint8),cv2.DIST_L2,3)
    return np.rint(result).clip(0,255).astype(np.uint8), {
        'frame':n,'identity':False,'visible_gap_pixels':int((interior&(aw<.95)).sum()),
        'edge_missing_pixels':int((gaps&edge).sum()),'maximum_gap_radius':float(distance.max()),
        'foreground_lost_pixels':lost,'foreground_expected_pixels':expected_area,
        'alpha_pixels':int((alpha>.001).sum())}


def apply_frame(manifest, n, rgb):
    """Apply before accepted conform geometry/color; exact identity off support."""
    return compose_frame(manifest,n,rgb)[0]


def render_bundle(manifest, output_dir, *, progress=None, cancelled=None):
    """Lossless native window, including range-aligned lossless source audio.

    FFV1/Matroska supports odd as well as even dimensions. The app creates its
    viewing encode after applying the accepted conform plan to these frames.
    """
    b = load_bundle(manifest)
    check_cancel(cancelled)
    root = fresh_directory(output_dir)
    output = root/'native.mkv'
    source = b['source']; start,end = b['support']['start'],b['support']['end']
    args = [_tool('ffmpeg'),'-v','error','-nostdin','-n','-f','rawvideo','-pix_fmt','rgb24',
            '-s',f"{source['width']}x{source['height']}",'-framerate',source['fps_fraction'],
            '-i','pipe:0','-i',source['path'],'-map','0:v:0','-map','1:a?',
            '-c:v','ffv1','-level','3','-pix_fmt','bgr0','-c:a','flac']
    if source.get('has_audio'):
        # This matches exact source-frame decode semantics and does not retime
        # audio. FFmpeg retains offsets before atrim; reset this window to zero.
        fps = source['fps']; offset = source.get('video_start_time',0)
        args += ['-af',f'atrim=start={offset+start/fps:.12f}:end={offset+end/fps:.12f},asetpts=PTS-STARTPTS']
    args += ['-frames:v',str(end-start),str(output)]
    error_path = root/'encoder.log'
    records=[]
    with error_path.open('wb') as errors:
        process = subprocess.Popen(args,stdin=subprocess.PIPE,stderr=errors)
        try:
            count=0
            for n,rgb in enumerate(iter_frames(source['path'],start,end-start),start):
                check_cancel(cancelled)
                result,record=compose_frame(b,n,rgb);records.append(record)
                process.stdin.write(result.tobytes()); count+=1
                report(progress,count/(end-start),f'Reconstructing source frame {n}')
            if count != end-start:
                raise ValueError('Source ended before reconstruction window')
            process.stdin.close()
            if process.wait():
                raise RuntimeError('Reconstruction encode failed: '+error_path.read_text()[-2000:])
        except BaseException:
            if process.poll() is None: process.terminate()
            if process.stdin and not process.stdin.closed: process.stdin.close()
            process.wait()
            output.unlink(missing_ok=True)
            raise
    (root/'render-report.json').write_text(json.dumps({'manifest':str(Path(b['_root'])/'manifest.json'),
        'source_start_frame':start,'source_end_frame_exclusive':end,'frames':records,
        'native_before_baseline':True,'codec':'ffv1','audio':'source range decoded losslessly'},indent=2)+'\n')
    return output
