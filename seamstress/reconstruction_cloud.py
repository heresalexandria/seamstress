"""Opt-in, bounded OpenAI background anchors; never character tween frames.

Images/edit contract checked against official documentation on 2026-09-21:
https://developers.openai.com/api/reference/resources/images/methods/edit
Masks are guidance to the service, so the final composite enforces them locally.
No SDK, shell command, environment-key discovery, or configurable endpoint is used.
"""
from __future__ import annotations

import base64
from contextlib import contextmanager
import hashlib
import io
import json
import math
import os
from pathlib import Path
import queue
import re
import shutil
import threading
import time
import urllib.error
import urllib.request
import uuid

import numpy as np
from PIL import Image

ENDPOINT = 'https://api.openai.com/v1/images/edits'
MODEL = 'gpt-image-2'
PROMPT = (
    'Create a clean background plate for this exact video frame. Replace only '
    'the transparent masked regions with the scenery that would be visible '
    'behind the removed foreground subjects. Continue existing lines, buildings, '
    'perspective, textures, colors, lighting and illustration style. Preserve '
    'camera position and all unmasked content. Do not add people, characters, '
    'limbs, hair, flames, motion blur, text or new objects. Do not change framing '
    'or color grade. Return one background image, not a transition or tween.'
)


class CloudError(RuntimeError):
    """A public, credential-safe provider failure."""


class RateLimitError(CloudError):
    pass


class ProviderRejectedError(CloudError):
    """A definitive non-chargeable rejection, safe to try after fixing settings."""


class RequestBudget:
    """A job-wide attempt cap, persisted *before* sending any chargeable request.

    Share one instance between all anchor calls in a job. An interrupted job
    resumes its existing count by reopening the same ledger. A request cap is
    not a dollar-price guarantee; account billing remains with the API provider.
    """

    def __init__(self, max_requests=1, *, ledger_path=None):
        if type(max_requests) is not int or not 1 <= max_requests <= 8:
            raise ValueError('AI request limit must be between 1 and 8')
        self.max_requests = max_requests
        self.ledger_path = Path(ledger_path) if ledger_path is not None else None
        self._lock = threading.Lock()
        self.attempts = []
        self._load()

    def _load(self):
        if self.ledger_path and self.ledger_path.exists():
            data = json.loads(self.ledger_path.read_text())
            if data.get('schema') != 1 or data.get('max_requests') != self.max_requests:
                raise CloudError('AI request ledger does not match this job budget')
            self.attempts = data.get('attempts', [])
            if not isinstance(self.attempts, list) or len(self.attempts) > self.max_requests:
                raise CloudError('Invalid AI request ledger')

    @contextmanager
    def _exclusive(self):
        with self._lock:
            lock = None
            if self.ledger_path:
                self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
                lock = self.ledger_path.with_name(self.ledger_path.name + '.lock')
                try: lock.mkdir()
                except FileExistsError:
                    raise CloudError('AI request ledger is in use or was interrupted; no request was sent') from None
            try:
                self._load()
                yield
            finally:
                if lock: lock.rmdir()

    @property
    def remaining(self):
        return max(0, self.max_requests - len(self.attempts))

    def _save(self):
        if self.ledger_path is not None:
            self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
            _atomic_json(self.ledger_path, {'schema': 1, 'max_requests': self.max_requests,
                                           'attempts': self.attempts})

    def reserve(self, cache_key):
        with self._exclusive():
            if not self.remaining:
                raise CloudError('AI request budget exhausted; no further requests were sent')
            attempt = {'id': uuid.uuid4().hex, 'cache_key': cache_key,
                       'status': 'pending', 'started_at': int(time.time())}
            self.attempts.append(attempt)
            self._save()
            return attempt['id']

    def finish(self, attempt_id, status):
        with self._exclusive():
            for attempt in self.attempts:
                if attempt['id'] == attempt_id:
                    attempt['status'] = status
                    self._save()
                    return


def _atomic_json(path, data):
    path = Path(path)
    temporary = path.with_name(f'.{path.name}.{uuid.uuid4().hex}')
    try:
        with temporary.open('x') as stream:
            json.dump(data, stream, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _check_cancelled(cancelled):
    if cancelled and cancelled():
        raise InterruptedError('Background generation cancelled')


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _png(array):
    stream = io.BytesIO()
    Image.fromarray(array).save(stream, format='PNG')
    return stream.getvalue()


def _api_images(source, mask):
    """Pad extreme aspect ratios and fit documented API resolution constraints."""
    height, width = source.shape[:2]
    padded_width = max(width, math.ceil(height / 3))
    padded_height = max(height, math.ceil(width / 3))
    padded = np.pad(source, ((0, padded_height-height), (0, padded_width-width), (0, 0)), mode='edge')
    padded_mask = np.pad(mask, ((0, padded_height-height), (0, padded_width-width)))
    scale = min(1., math.sqrt(3_686_400 / (padded_width * padded_height)), 2560 / max(padded_width, padded_height))
    scale = max(scale, math.sqrt(655_360 / (padded_width * padded_height)))
    request_width = math.ceil(padded_width * scale / 16) * 16
    request_height = math.ceil(padded_height * scale / 16) * 16
    if request_width > 3 * request_height: request_height = math.ceil(request_width / 48) * 16
    if request_height > 3 * request_width: request_width = math.ceil(request_height / 48) * 16
    size = (request_width, request_height)
    image = np.asarray(Image.fromarray(padded).resize(size, Image.Resampling.LANCZOS))
    # API alpha 0 means editable, alpha 255 means protected.
    alpha = np.asarray(Image.fromarray(np.where(padded_mask > 0, 0, 255).astype(np.uint8))
                       .resize(size, Image.Resampling.NEAREST))
    rgba = np.full((request_height, request_width, 4), 255, dtype=np.uint8)
    rgba[..., 3] = alpha
    return _png(image), _png(rgba), {'request_width': request_width, 'request_height': request_height,
                                    'padded_width': padded_width, 'padded_height': padded_height}


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *_args, **_kwargs):
        raise CloudError('Image service redirected the request; no credentials were forwarded')


def _https_edit(*, payload, api_key, cancelled=None):
    """Fixed HTTPS endpoint; no redirects, remote image URLs, or error-body logs."""
    result = queue.Queue(maxsize=1)

    def request():
        try:
            req = urllib.request.Request(ENDPOINT, data=json.dumps(payload).encode(), method='POST',
                headers={'Authorization': f'Bearer {api_key}', 'Content-Type': 'application/json'})
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect())
            try:
                with opener.open(req, timeout=240) as response:
                    raw = response.read(64 * 1024 * 1024 + 1)
                    if len(raw) > 64 * 1024 * 1024:
                        raise CloudError('Image service response exceeded its size limit')
                    data = json.loads(raw)
                    result.put((True, {'data': data.get('data'), 'request_id': response.headers.get('x-request-id')}))
            except urllib.error.HTTPError as error:
                # Parse only error codes; service messages can echo prompt/input.
                code = None
                try: code = json.loads(error.read(8192)).get('error', {}).get('code')
                except Exception: pass
                if error.code == 429 and code not in ('insufficient_quota', 'billing_hard_limit_reached'):
                    raise RateLimitError('OpenAI rate limit reached') from None
                if code in ('insufficient_quota', 'billing_hard_limit_reached'):
                    raise ProviderRejectedError('OpenAI quota or billing limit reached; check your API account') from None
                if error.code in (401, 403):
                    raise ProviderRejectedError('OpenAI rejected the key or model access; check AI settings and your API account') from None
                if 400 <= error.code < 500:
                    raise ProviderRejectedError(f'OpenAI rejected the image request (HTTP {error.code}); no retry was sent') from None
                raise CloudError(f'OpenAI image request failed (HTTP {error.code}); no automatic retry was sent') from None
        except BaseException as error:
            # Never transfer a raw transport exception: it can contain headers.
            if not isinstance(error, CloudError):
                error = CloudError('Image request failed or timed out; it may have been processed. No automatic retry was sent')
            result.put((False, error))

    thread = threading.Thread(target=request, daemon=True)
    thread.start()
    while True:
        _check_cancelled(cancelled)
        try: ok, value = result.get(timeout=.1)
        except queue.Empty: continue
        _check_cancelled(cancelled)
        if not ok: raise value
        return value


def _cache_result(directory, identity, cache_hit):
    try:
        provenance = json.loads((directory / 'provenance.json').read_text())
        if provenance.get('identity') != identity: raise ValueError()
        if set(provenance['files']) != {'background.png', 'generated.png', 'source.png', 'editable-mask.png'}:
            raise ValueError()
        for filename, expected in provenance['files'].items():
            if _sha((directory / filename).read_bytes()) != expected: raise ValueError()
    except (OSError, ValueError, KeyError):
        raise CloudError('Cached background assets failed integrity validation; no new request was sent') from None
    return {'image_path': str(directory / 'background.png'), 'generated_path': str(directory / 'generated.png'),
            'provenance_path': str(directory / 'provenance.json'), 'cache_hit': cache_hit,
            'request_id': provenance.get('request_id')}


def generate_background_anchor(source, editable_mask, cache_dir, *, api_key=None, allow_upload=False,
                               budget=None, quality='medium', cancelled=None, transport=None):
    """Generate one native-sized anchor, preserving pixels outside editable_mask.

    ``source`` is RGB uint8; ``editable_mask`` is boolean or finite floats 0..1.
    The source frame and mask are uploaded only after explicit ``allow_upload``.
    Injected transports accept payload/api_key/cancelled and return the same
    base64 ``data`` shape as OpenAI plus an optional safe request_id.
    """
    _check_cancelled(cancelled)
    if allow_upload is not True:
        raise CloudError('AI background fill requires explicit upload permission')
    source = np.asarray(source)
    mask = np.asarray(editable_mask)
    if source.dtype != np.uint8 or source.ndim != 3 or source.shape[2] != 3 or min(source.shape[:2]) < 1:
        raise ValueError('Background source must be a nonempty RGB uint8 image')
    if max(source.shape[:2]) > 8192 or source.shape[0] * source.shape[1] > 33_554_432:
        raise ValueError('Background source is too large')
    if mask.shape != source.shape[:2] or not np.issubdtype(mask.dtype, np.number) and mask.dtype != np.bool_:
        raise ValueError('Editable mask must match the source dimensions')
    if not np.isfinite(mask).all() or mask.min() < 0 or mask.max() > 1 or not np.any(mask > 0):
        raise ValueError('Editable mask must contain finite values from zero to one and an editable region')
    mask = mask.astype(np.float32)
    if quality not in ('low', 'medium', 'high'):
        raise ValueError('Background quality must be low, medium, or high')
    identity = {'schema': 1, 'model': MODEL, 'quality': quality, 'prompt': PROMPT,
                'width': int(source.shape[1]), 'height': int(source.shape[0]),
                'source_sha256': _sha(source.tobytes()), 'mask_sha256': _sha(mask.tobytes())}
    cache_key = _sha(json.dumps(identity, sort_keys=True).encode())
    cache_root = Path(cache_dir).expanduser().resolve()
    directory = cache_root / cache_key
    if directory.exists(): return _cache_result(directory, identity, True)
    if not isinstance(api_key, str) or not re.fullmatch(r'sk-[A-Za-z0-9_-]{16,4093}', api_key.strip()):
        raise CloudError('Add an OpenAI API key before generating a background')
    api_key = api_key.strip()
    if budget is None: budget = RequestBudget()
    if not isinstance(budget, RequestBudget): raise TypeError('Share a RequestBudget across this generation job')
    if not budget.remaining: raise CloudError('AI request budget exhausted; no further requests were sent')
    image_bytes, mask_bytes, sizing = _api_images(source, mask)
    payload = {'model': MODEL, 'prompt': PROMPT, 'quality': quality, 'n': 1, 'output_format': 'png',
               'size': f"{sizing['request_width']}x{sizing['request_height']}",
               'images': [{'image_url': 'data:image/png;base64,' + base64.b64encode(image_bytes).decode()}],
               'mask': {'image_url': 'data:image/png;base64,' + base64.b64encode(mask_bytes).decode()}}
    cache_root.mkdir(parents=True, exist_ok=True)
    claim = cache_root / f'.pending-{cache_key}'
    try: claim.mkdir()
    except FileExistsError:
        raise CloudError('A matching background request is running or was interrupted; no duplicate request was sent') from None
    uncertain = False
    attempt_id = None
    try:
        while True:
            _check_cancelled(cancelled)
            attempt_id = budget.reserve(cache_key)
            _atomic_json(claim / 'request.json', {'attempt_id': attempt_id, 'identity': identity, 'status': 'pending'})
            uncertain = True
            try:
                response = (transport or _https_edit)(payload=payload, api_key=api_key, cancelled=cancelled)
                break
            except RateLimitError:
                budget.finish(attempt_id, 'rate_limited')
                uncertain = False
                if not budget.remaining: raise CloudError('OpenAI rate limit reached; the request budget is exhausted') from None
                for _ in range(20):
                    _check_cancelled(cancelled)
                    time.sleep(.1)
            except (InterruptedError, KeyboardInterrupt):
                budget.finish(attempt_id, 'cancelled_outcome_unknown')
                raise
            except ProviderRejectedError:
                budget.finish(attempt_id, 'rejected')
                uncertain = False
                raise
            except CloudError:
                budget.finish(attempt_id, 'failed_outcome_unknown')
                raise
            except Exception:
                budget.finish(attempt_id, 'failed_outcome_unknown')
                raise CloudError('Background provider failed; no automatic retry was sent') from None
        _check_cancelled(cancelled)
        # A received response is never re-requested automatically, even if invalid.
        budget.finish(attempt_id, 'received')
        try:
            entries = response['data']
            if len(entries) != 1: raise ValueError()
            raw = base64.b64decode(entries[0]['b64_json'], validate=True)
            if len(raw) > 32 * 1024 * 1024: raise ValueError()
            with Image.open(io.BytesIO(raw)) as generated:
                if generated.size != (sizing['request_width'], sizing['request_height']): raise ValueError()
                mapped = generated.convert('RGB').resize((sizing['padded_width'], sizing['padded_height']), Image.Resampling.LANCZOS)
                generated_native = np.asarray(mapped)[:source.shape[0], :source.shape[1]].copy()
        except Exception:
            raise CloudError('Image service returned an invalid image or unexpected dimensions; no retry was sent') from None
        final = source.copy()
        active = mask > 0
        blended = np.rint(source.astype(np.float32) * (1-mask[..., None]) + generated_native * mask[..., None]).clip(0, 255).astype(np.uint8)
        final[active] = blended[active]
        files = {'background.png': _png(final), 'generated.png': _png(generated_native),
                 'source.png': _png(source), 'editable-mask.png': _png(np.rint(mask*255).astype(np.uint8))}
        for filename, data in files.items(): (claim / filename).write_bytes(data)
        request_id = response.get('request_id')
        if not isinstance(request_id, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,200}', request_id) or request_id.startswith('sk-'):
            request_id = None
        provenance = {'schema': 1, 'identity': identity, 'sizing': sizing, 'provider': 'openai',
                      'endpoint': ENDPOINT, 'request_id': request_id, 'attempt_id': attempt_id,
                      'mask_enforced_locally': True, 'edited_pixels': int(active.sum()),
                      'outside_mask_unchanged': bool(np.array_equal(final[~active], source[~active])),
                      'files': {filename: _sha(data) for filename, data in files.items()}}
        _atomic_json(claim / 'provenance.json', provenance)
        _check_cancelled(cancelled)
        (claim / 'request.json').unlink()
        os.rename(claim, directory)
        uncertain = False
        budget.finish(attempt_id, 'complete')
        return _cache_result(directory, identity, False)
    finally:
        # Keep uncertain claims after cancellation/timeouts to stop accidental
        # duplicate charges on a resume. No credentials are written into them.
        if not uncertain and claim.exists(): shutil.rmtree(claim)
