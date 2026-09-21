"""Immutable, source-bound assets for native RGB layer reconstruction."""
from __future__ import annotations

import copy
import hashlib
import json
import math
import re
import shutil
import uuid
from pathlib import Path

import cv2
import numpy as np

from ..media import probe

SCHEMA = 'seamstress.reconstruction/v1'


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def rgb_hash(rgb):
    return hashlib.sha256(np.ascontiguousarray(rgb).tobytes()).hexdigest()


def check_cancel(cancelled):
    if cancelled and cancelled():
        raise InterruptedError('Reconstruction cancelled')


def report(progress, fraction, message):
    if progress:
        progress({'fraction': float(fraction), 'progress': float(fraction), 'message': message})


def fresh_directory(path):
    path = Path(path).expanduser().resolve()
    # An exclusive directory is also the concurrent-job reservation. Never reuse
    # an incomplete job: callers choose a new revision after any failure.
    path.mkdir(parents=True, exist_ok=False)
    return path


def asset_path(bundle, name):
    if not isinstance(name, str) or not name or Path(name).is_absolute():
        raise ValueError('Bundle assets must use relative paths')
    root = Path(bundle['_root']).resolve()
    path = (root / name).resolve()
    if not path.is_relative_to(root) or path == root:
        raise ValueError('Bundle asset escapes its directory')
    if name not in bundle['assets']:
        raise ValueError(f'Unregistered bundle asset: {name}')
    return path


def register(bundle, path):
    path = Path(path).resolve()
    name = str(path.relative_to(bundle['_root']))
    bundle['assets'][name] = sha256(path)
    return name


def copy_asset(bundle, path, name):
    target = (Path(bundle['_root']) / name).resolve()
    if Path(name).is_absolute() or not target.is_relative_to(Path(bundle['_root']).resolve()):
        raise ValueError('Asset copy escapes bundle directory')
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(path, target)
    return register(bundle, target)


def save_rgb(bundle, name, rgb):
    from PIL import Image
    path = (Path(bundle['_root']) / name).resolve()
    if Path(name).is_absolute() or not path.is_relative_to(Path(bundle['_root']).resolve()):
        raise ValueError('Image asset escapes bundle directory')
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(rgb, np.uint8)).save(path)
    return register(bundle, path)


def matrix(value):
    m = np.asarray(value, dtype=np.float64)
    if m.shape != (3, 3) or not np.isfinite(m).all() or not np.allclose(m[2], [0, 0, 1], atol=1e-8):
        raise ValueError('Layer motion must be a finite affine 3x3 matrix')
    a = m[:2, :2]
    gram = a.T @ a
    scale2 = np.trace(gram) / 2
    if np.linalg.det(a) <= 0 or scale2 < 1e-8 or not np.allclose(gram, np.eye(2)*scale2, rtol=1e-5, atol=1e-7):
        raise ValueError('Layer motion must be a similarity: translation, rotation and uniform scale only')
    return m


def layer_id(value):
    if not isinstance(value,str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}',value):
        raise ValueError('Layer ID must contain only letters, digits, underscores or hyphens')
    return value


def new_bundle(source, frame, output_dir, support, baseline_plan=None):
    source = Path(source).expanduser().resolve()
    metadata = probe(source)
    if type(frame) is not int or not 0 < frame < metadata['frame_count']:
        raise ValueError('Selected seam must be an incoming source frame inside the video')
    start, end = support
    if any(type(v) is not int for v in (start, end)) or not 0 <= start < frame < end <= metadata['frame_count']:
        raise ValueError('Reconstruction support must span the seam inside the source')
    if metadata.get('color_transfer') in ('smpte2084', 'arib-std-b67'):
        raise ValueError('Reconstruction currently supports SDR RGB, not HDR')
    source_hash = sha256(source)
    baseline = None
    if baseline_plan is not None:
        from ..conform import validate_conform_plan
        p = Path(baseline_plan).expanduser().resolve()
        data = json.loads(p.read_text())
        validate_conform_plan(data, metadata)
        if data['source_sha256'] != source_hash:
            raise ValueError('Reconstruction baseline belongs to a different source')
        baseline = {'path': str(p), 'sha256': sha256(p)}
    root = fresh_directory(output_dir)
    bundle = {'schema': SCHEMA, 'id': uuid.uuid4().hex, 'source': {**metadata, 'sha256': source_hash},
              'baseline': baseline, 'frame': frame, 'support': {'start': start, 'end': end},
              'assets': {}, 'frames': {}, 'layers': [],
              'qa': {'status': 'needs-review', 'auto_eligible': False, 'issues': [], 'metrics': {}},
              'provenance': {'same_frame_foreground': True, 'coordinate_space': 'source-native-rgb'},
              'compositor': {'edge_band': 0, 'gap_inpaint_radius': 0}, '_root': str(root)}
    if baseline:
        bundle['baseline']['asset'] = copy_asset(bundle, baseline['path'], 'baseline.plan.json')
    return bundle


def freeze(bundle):
    root = Path(bundle['_root'])
    if (root/'READY.json').exists():
        raise FileExistsError('Bundle is already immutable')
    clean = {key: value for key, value in bundle.items() if not key.startswith('_')}
    path = root/'manifest.json'
    path.write_text(json.dumps(clean, indent=2, allow_nan=False)+'\n')
    # Structural and asset validation precedes publication of READY.
    validate(bundle, verify=True, verify_source=False)
    (root/'READY.json').write_text(json.dumps({'schema': SCHEMA, 'manifest_sha256': sha256(path)})+'\n')
    return path


def validate(bundle, *, verify=True, verify_source=True):
    if bundle.get('schema') != SCHEMA:
        raise ValueError('Unsupported reconstruction bundle schema')
    source = bundle['source']
    w, h, count = source['width'], source['height'], source['frame_count']
    if any(type(v) is not int or v <= 0 for v in (w, h, count)):
        raise ValueError('Invalid source dimensions/frame count')
    from fractions import Fraction
    fps = float(Fraction(source['fps_fraction']))
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError('Invalid source cadence')
    start, end = bundle['support']['start'], bundle['support']['end']
    if any(type(v) is not int for v in (start, end, bundle['frame'])) or not 0 <= start < bundle['frame'] < end <= count:
        raise ValueError('Invalid reconstruction support')
    if verify_source and sha256(source['path']) != source['sha256']:
        raise ValueError('Reconstruction source hash mismatch')
    if not isinstance(bundle['assets'], dict):
        raise ValueError('Bundle asset registry is required')
    for name, digest in bundle['assets'].items():
        path = asset_path(bundle, name)
        if not path.is_file() or (verify and sha256(path) != digest):
            raise ValueError(f'Reconstruction asset hash mismatch: {name}')
    baseline = bundle.get('baseline')
    if baseline and sha256(asset_path(bundle, baseline['asset'])) != baseline['sha256']:
        raise ValueError('Frozen reconstruction baseline hash mismatch')
    if set(bundle['frames']) != {str(n) for n in range(start, end)}:
        raise ValueError('Bundle must cover every supported source frame exactly')
    ids = [layer_id(layer['id']) for layer in bundle['layers']]
    if len(set(ids)) != len(ids) or not ids:
        raise ValueError('Layer IDs must be unique and nonempty')
    if sum(layer['kind'] == 'foreground' for layer in bundle['layers']) != 1:
        raise ValueError('One same-frame foreground layer is required')
    if not any(layer['kind'] == 'background' for layer in bundle['layers']):
        raise ValueError('At least one background layer is required')
    config=bundle.get('compositor',{})
    band=config.get('edge_band',0);radius=config.get('gap_inpaint_radius',0);threshold=config.get('gap_threshold',.7)
    if type(band) is not int or not 0<=band<=max(1,int(min(w,h)*.1)):
        raise ValueError('Edge extrapolation is limited to ten percent of the smaller source dimension')
    if not isinstance(radius,(int,float)) or not math.isfinite(radius) or not 0<=radius<=8:
        raise ValueError('Gap interpolation radius must be between zero and eight native pixels')
    if not isinstance(threshold,(int,float)) or not math.isfinite(threshold) or not .5<=threshold<=1:
        raise ValueError('Invalid background coverage threshold')
    from PIL import Image
    for layer in bundle['layers']:
        if layer['kind'] not in ('foreground', 'background'):
            raise ValueError('Unknown layer type')
        if set(layer['matrices']) != set(bundle['frames']) or set(layer['mask_by_frame']) != set(bundle['frames']):
            raise ValueError('Each layer must exactly cover the declared frame support')
        for n in range(start, end):
            matrix(layer['matrices'][str(n)])
            mask = np.asarray(Image.open(asset_path(bundle, layer['mask_by_frame'][str(n)])))
            if mask.shape != (h, w):
                raise ValueError('Layer masks must match native source dimensions')
    for key, row in bundle['frames'].items():
        if row.get('foreground_source_frame', int(key)) != int(key):
            raise ValueError('Foreground must come from the same source frame')
        if not 0 <= row.get('residual_strength', 1.) <= 1:
            raise ValueError('Invalid reconstruction residual strength')
        plate = asset_path(bundle, row['plate'])
        with Image.open(asset_path(bundle,row['source'])) as image:
            rgb = np.asarray(image.convert('RGB'))
        if rgb.shape != (h,w,3) or rgb_hash(rgb) != row.get('source_rgb_sha256'):
            raise ValueError(f'Source RGB asset mismatch at frame {key}')
        with Image.open(plate) as image:
            if image.size != (w, h):
                raise ValueError('Plate dimensions do not match native source')
        with np.load(asset_path(bundle, row['matte']), allow_pickle=False) as data:
            arrays={name:data[name] for name in ('alpha','premultiplied','emission')}
            for name, shape, lo, hi in [('alpha', (h,w), 0,1), ('premultiplied',(h,w,3),0,255), ('emission',(h,w,3),0,255)]:
                v = arrays[name]
                if v.shape != shape or not np.isfinite(v).all() or v.min() < lo-1e-4 or v.max() > hi+1e-4:
                    raise ValueError(f'Invalid matte {name} at frame {key}')
            if str(data['background_sha256']) != bundle['assets'][row['plate']]:
                raise ValueError(f'Matte/plate provenance mismatch at frame {key}')
            core=arrays['alpha']>=.99999
            if core.any() and np.max(np.abs((arrays['premultiplied']+arrays['emission'])[core]-rgb.astype(np.float32)[core]))>.05:
                raise ValueError(f'Opaque foreground core differs from same-frame source RGB at {key}')
    return bundle


def load_bundle(manifest, *, verify=True):
    if isinstance(manifest, dict):
        return manifest
    path = Path(manifest).expanduser().resolve()
    if path.is_dir():
        path = path/'manifest.json'
    ready = path.parent/'READY.json'
    if not ready.is_file() or json.loads(ready.read_text()).get('manifest_sha256') != sha256(path):
        raise ValueError('Reconstruction bundle is incomplete or manifest was modified')
    bundle = json.loads(path.read_text())
    bundle['_root'] = str(path.parent)
    bundle['_manifest'] = str(path)
    return validate(bundle, verify=verify, verify_source=verify)


def summary(manifest):
    b = load_bundle(manifest)
    n = str(b['frame'])
    return {'manifestPath': str(Path(b['_root'])/'manifest.json'),
            'manifestSha256':sha256(Path(b['_root'])/'manifest.json'), 'id': b['id'], 'frame':b['frame'],
            'status': b['qa']['status'], 'sourceFrame': b['frame'],
            'reachFrames': b.get('options',{}).get('reachFrames'),
            'motionStrength': b.get('options',{}).get('motionStrength'),
            'segmentation': b.get('options',{}).get('segmentation'),
            'width': b['source']['width'], 'height': b['source']['height'],
            'startFrame': b['support']['start'], 'endFrame': b['support']['end'],
            'sourceFramePath': str(asset_path(b, b['frames'][n]['source'])),
            'layers': [{'id': l['id'], 'name': l.get('name',l['id']), 'role': l['kind'],
                        'maskPreviewPath': str(asset_path(b,l['mask_by_frame'][n])),
                        'keyframes': l.get('keyframes',[]), 'confidence': l.get('confidence',0.)}
                       for l in b['layers']],
            'previews': [{'frame': int(k), 'path': str(asset_path(b,r['source']))}
                         for k,r in b['frames'].items() if int(k) in {b['frame']-1,b['frame'],b['support']['start'],b['support']['end']-1}],
            'frames': [{'frame':int(k),'sourceFramePath':str(asset_path(b,r['source'])),
                        'masks':{l['id']:str(asset_path(b,l['mask_by_frame'][k])) for l in b['layers']}}
                       for k,r in b['frames'].items()],
            'issues': list(dict.fromkeys([*b['qa'].get('blocking_errors',[]),*b['qa']['issues']])),
            'metrics': b['qa']['metrics'], 'qa':copy.deepcopy(b['qa']),
            'needsBackgroundFill':bool(b['qa']['metrics'].get('unrecovered_background_pixels',0)), 'canRender': True,
            'canAccept': not b['qa'].get('blocking_errors'), 'autoEligible': b['qa'].get('auto_eligible',False)}
