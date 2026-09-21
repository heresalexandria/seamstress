"""Optional, verified MobileSAM ONNX inference on the local CPU.

Weights are downloaded only by setup_model(allow_download=True). No video
pixels leave this machine. The ONNX exporter publishes file SHA-256s in its
Hugging Face LFS metadata; both revision and exact bytes are pinned below.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import urllib.parse
import urllib.request

import numpy as np
from PIL import Image

MODEL_ID = 'mobilesam-onnx-v1'
REVISION = '0d3b403339b4674a82493d5e97964dd78089ddc8'
PUBLISHER = 'https://huggingface.co/Acly/MobileSAM'
UPSTREAM = 'https://github.com/ChaoningZhang/MobileSAM'
FILES = {
    'mobile_sam_image_encoder.onnx': {
        'size': 28157093, 'sha256': '580f5fb648ea1062c0aabc26217aed56921985f03f0cbbd852bba81d760cc749'},
    'sam_mask_decoder_single.onnx': {
        'size': 16501323, 'sha256': '93915fc7c993ab9d59ab8c9ccd3bce37f7509c81ab4150a74abd4d2abbd8570d'},
}
DOWNLOAD_BYTES = sum(v['size'] for v in FILES.values())


class SegmentationError(RuntimeError):
    pass


def default_model_dir():
    if os.environ.get('SEAMSTRESS_MODEL_DIR'):
        return Path(os.environ['SEAMSTRESS_MODEL_DIR']).expanduser().resolve()
    if sys.platform == 'darwin': base = Path.home()/'Library/Application Support/Seamstress'
    elif sys.platform == 'win32': base = Path(os.environ.get('LOCALAPPDATA', Path.home()))/'Seamstress'
    else: base = Path(os.environ.get('XDG_CACHE_HOME', Path.home()/'.cache'))/'seamstress'
    return base/'models'/MODEL_ID


def _directory(value):
    return Path(value).expanduser().resolve() if value is not None else default_model_dir()


def _check(cancelled):
    if cancelled and cancelled(): raise InterruptedError('Local segmentation cancelled')


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''): digest.update(block)
    return digest.hexdigest()


def _verified(path, record):
    return path.is_file() and path.stat().st_size == record['size'] and _sha(path) == record['sha256']


def model_status(model_dir=None):
    directory = _directory(model_dir)
    runtime = importlib.util.find_spec('onnxruntime') is not None
    try: downloaded = all(_verified(directory/name, record) for name, record in FILES.items())
    except OSError: downloaded = False
    reason = None if runtime and downloaded else ('Download the local MobileSAM model.' if runtime else
        'Local segmentation runtime is unavailable. CLI users can install seamstress-video[segmentation].')
    return {'modelId': MODEL_ID, 'name': 'MobileSAM', 'downloadBytes': DOWNLOAD_BYTES,
            'runtimeAvailable': runtime, 'downloaded': downloaded, 'available': runtime and downloaded,
            'reason': reason, 'provider': 'CPUExecutionProvider',
            'license': 'Apache-2.0 upstream; MIT ONNX exporter', 'publisher': PUBLISHER,
            'revision': REVISION}


def available(model_dir=None):
    """Purely local capability check; never installs or downloads anything."""
    return model_status(model_dir)['available']


class _ModelRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, response, code, message, headers, newurl):
        parsed = urllib.parse.urlparse(newurl)
        host = parsed.hostname or ''
        if parsed.scheme != 'https' or parsed.username or parsed.password or not (
            host == 'huggingface.co' or host.endswith('.huggingface.co') or host.endswith('.hf.co')):
            raise SegmentationError('Model download redirected outside the trusted publisher hosts')
        return super().redirect_request(request, response, code, message, headers, newurl)


def setup_model(model_dir=None, *, allow_download=False, source_dir=None, progress=None, cancelled=None):
    """Download pinned files or import identical files from a local directory.

    Each file is verified before an atomic, no-overwrite publication. An
    interrupted download leaves no partial model. Existing mismatched files
    are preserved and reported instead of silently replaced.
    """
    _check(cancelled)
    directory = _directory(model_dir)
    if source_dir is None and allow_download is not True:
        raise SegmentationError('Downloading MobileSAM requires an explicit request')
    source = Path(source_dir).expanduser().resolve() if source_dir is not None else None
    if source and not source.is_dir(): raise SegmentationError('Choose a directory containing the pinned ONNX files')
    directory.mkdir(parents=True, exist_ok=True)
    done = 0
    def emit(message, fraction):
        if progress: progress({'stage': 'reconstruct', 'fraction': fraction, 'message': message})
    for name, record in FILES.items():
        _check(cancelled)
        target = directory/name
        if target.exists():
            if not _verified(target, record):
                raise SegmentationError(f'Existing {name} failed verification; choose a clean model directory')
            done += record['size']; emit('Verified '+name, done/DOWNLOAD_BYTES); continue
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(prefix='.model-', dir=directory, delete=False) as output:
                temporary = Path(output.name)
                if source:
                    stream = (source/name).open('rb')
                else:
                    url = f'{PUBLISHER}/resolve/{REVISION}/{name}'
                    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _ModelRedirect())
                    stream = opener.open(urllib.request.Request(url, headers={'User-Agent':'Seamstress local segmentation'}), timeout=45)
                count = 0
                with stream:
                    while True:
                        _check(cancelled)
                        block = stream.read(256*1024)
                        if not block: break
                        count += len(block)
                        if count > record['size']: raise SegmentationError('Model download exceeded its pinned size')
                        output.write(block)
                        emit('Downloading MobileSAM locally' if not source else 'Importing verified MobileSAM files',
                             (done+count)/DOWNLOAD_BYTES)
                output.flush(); os.fsync(output.fileno())
            _check(cancelled)
            if not _verified(temporary, record): raise SegmentationError('Model bytes do not match the pinned SHA-256')
            try: os.link(temporary, target)
            except FileExistsError:
                if not _verified(target, record): raise SegmentationError('Another model setup wrote an unverified file') from None
            done += record['size']
        except (OSError, ValueError) as error:
            if isinstance(error, InterruptedError): raise
            raise SegmentationError('Model download or import failed; retry the explicit setup operation') from None
        finally:
            if temporary: temporary.unlink(missing_ok=True)
    notice = ('MobileSAM optional local segmentation\n\n'
              f'Upstream: {UPSTREAM} (Apache License 2.0)\n'
              f'ONNX exporter: {PUBLISHER} (publisher declares MIT)\n'
              f'Pinned revision: {REVISION}\n'
              'The model is used locally, without uploading user images.\n')
    (directory/'NOTICE.txt').write_text(notice)
    license_path = Path(__file__).parent/'resources/MobileSAM-LICENSE.txt'
    if license_path.is_file(): shutil.copyfile(license_path, directory/'MobileSAM-LICENSE.txt')
    (directory/'provenance.json').write_text(json.dumps({'modelId':MODEL_ID,'publisher':PUBLISHER,
        'revision':REVISION,'upstream':UPSTREAM,'files':FILES},indent=2))
    emit('MobileSAM downloaded and verified', 1.)
    return model_status(directory)


def _prompts(width, height, points, box):
    if points is None: points = []
    if not isinstance(points, (list, tuple)) or len(points) > 64:
        raise ValueError('Use at most 64 foreground/background points')
    coords, labels = [], []
    for point in points:
        if not isinstance(point, dict) or set(point)-{'x','y','label'}:
            raise ValueError('Each prompt point needs x, y, and label')
        x,y,label=point.get('x'),point.get('y'),point.get('label')
        if isinstance(x,bool) or isinstance(y,bool) or not isinstance(x,(int,float)) or not isinstance(y,(int,float)):
            raise ValueError('Prompt coordinates must be finite image pixels')
        if not np.isfinite([x,y]).all() or not (0 <= x < width and 0 <= y < height) or type(label) is not int or label not in (0,1):
            raise ValueError('Prompt points must lie inside the frame with labels zero or one')
        coords.append([x,y]); labels.append(label)
    if box is not None:
        if not isinstance(box,(list,tuple)) or len(box)!=4 or any(isinstance(v,bool) or not isinstance(v,(int,float)) for v in box):
            raise ValueError('Box must be [left, top, right, bottom] in image pixels')
        x0,y0,x1,y1=box
        if not np.isfinite(box).all() or not (0<=x0<x1<=width and 0<=y0<y1<=height):
            raise ValueError('Prompt box must be inside the frame and have positive size')
        coords += [[x0,y0],[x1,y1]]; labels += [2,3]
    elif coords:
        # SAM distinguishes an absent box with the padded not-a-point token.
        coords.append([0,0]); labels.append(-1)
    if not coords: raise ValueError('Choose a foreground point or a subject box')
    return np.asarray(coords,np.float32),np.asarray(labels,np.float32)


def predict_mask(rgb, *, points=None, box=None, model_dir=None, cancelled=None):
    """Return a binary native-size float mask from point/box prompts on CPU.

    Mask proposal confidence is advisory. Existing mask tracking propagates the
    seed; this function never changes foreground drawings or accepted footage.
    """
    _check(cancelled)
    rgb=np.asarray(rgb)
    if rgb.dtype!=np.uint8 or rgb.ndim!=3 or rgb.shape[2]!=3 or min(rgb.shape[:2])<1:
        raise ValueError('Segmentation input must be a nonempty RGB uint8 frame')
    height,width=rgb.shape[:2]
    if max(height,width)>8192 or height*width>33_554_432: raise ValueError('Segmentation frame is too large')
    coords,labels=_prompts(width,height,points,box)
    directory=_directory(model_dir)
    status=model_status(directory)
    if not status['available']: raise SegmentationError(status['reason'])
    import onnxruntime as ort
    options=ort.SessionOptions(); options.intra_op_num_threads=2; options.inter_op_num_threads=1
    options.execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL
    options.log_severity_level=3
    # Loading only verified pinned graphs avoids arbitrary imported ONNX code
    # or external tensor files. Keep execution on the portable CPU provider.
    encoder=ort.InferenceSession(str(directory/'mobile_sam_image_encoder.onnx'),sess_options=options,
                                providers=['CPUExecutionProvider'])
    decoder=ort.InferenceSession(str(directory/'sam_mask_decoder_single.onnx'),sess_options=options,
                                providers=['CPUExecutionProvider'])
    scale=1024/max(height,width)
    resized_width,resized_height=int(width*scale+.5),int(height*scale+.5)
    image=np.asarray(Image.fromarray(rgb).resize((resized_width,resized_height),Image.Resampling.BILINEAR),np.float32)
    # This pinned encoder includes channel normalization and zero padding.
    _check(cancelled)
    embedding=encoder.run(['image_embeddings'],{'input_image':image})[0]
    _check(cancelled)
    coords[:,0]*=resized_width/width; coords[:,1]*=resized_height/height
    masks,scores,_=decoder.run(None,{'image_embeddings':embedding,
        'point_coords':coords[None], 'point_labels':labels[None],
        'mask_input':np.zeros((1,1,256,256),np.float32),
        'has_mask_input':np.zeros((1,),np.float32),'orig_im_size':np.asarray([height,width],np.float32)})
    _check(cancelled)
    logits=np.asarray(masks)[0,0]
    score=float(np.asarray(scores).reshape(-1)[0])
    if logits.shape!=(height,width) or not np.isfinite(logits).all() or not np.isfinite(score):
        raise SegmentationError('Local segmentation returned an invalid mask')
    return {'mask':(logits>0).astype(np.float32),'score':float(np.clip(score,0,1)),
            'model':{'id':MODEL_ID,'revision':REVISION,'provider':'CPUExecutionProvider',
                     'publisher':PUBLISHER,'files':{k:v['sha256'] for k,v in FILES.items()},
                     'points':points or [],'box':box}}
