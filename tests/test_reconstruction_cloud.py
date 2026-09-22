import base64
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import urllib.error

import numpy as np
from PIL import Image

from seamstress.reconstruction_cloud import (CloudError, RateLimitError, RequestBudget,
    _api_images, _https_edit, _NoRedirect, generate_background_anchor)

KEY = 'sk-unit_test_secret_1234567890'


class BackgroundCloudTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = np.random.default_rng(842).integers(0, 200, (16, 32, 3), dtype=np.uint8)
        self.mask = np.zeros((16, 32), np.float32)
        self.mask[4:12, 8:24] = 1
        self.calls = []

    def provider(self, **kwargs):
        self.calls.append(kwargs)
        width, height = map(int, kwargs['payload']['size'].split('x'))
        stream = io.BytesIO()
        Image.new('RGB', (width, height), (231, 119, 57)).save(stream, format='PNG')
        return {'data': [{'b64_json': base64.b64encode(stream.getvalue()).decode()}], 'request_id': 'req-test'}

    def generate(self, **kwargs):
        return generate_background_anchor(self.source, self.mask, self.root/'cache',
            **{'allow_upload': True, 'api_key': KEY, 'transport': self.provider, **kwargs})

    def test_opt_in_and_key_are_required_without_any_network_call(self):
        for kwargs in ({'allow_upload': False}, {'api_key': None}, {'api_key': 'bad'}):
            with self.subTest(kwargs=kwargs), self.assertRaises(CloudError): self.generate(**kwargs)
        self.assertEqual(self.calls, [])
        self.assertFalse((self.root/'cache').exists())

    def test_generated_pixels_are_bounded_and_cache_preserves_hashes(self):
        budget = RequestBudget(1, ledger_path=self.root/'ledger.json')
        result = self.generate(budget=budget)
        final = np.asarray(Image.open(result['image_path']))
        np.testing.assert_array_equal(final[self.mask == 0], self.source[self.mask == 0])
        np.testing.assert_array_equal(final[self.mask == 1], np.tile([231, 119, 57], (128, 1)))
        self.assertEqual(final.shape, self.source.shape)
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(budget.remaining, 0)
        reused = self.generate(api_key=None, budget=budget)
        self.assertTrue(reused['cache_hit'])
        self.assertEqual(len(self.calls), 1)
        for path in self.root.rglob('*'):
            if path.is_file(): self.assertNotIn(KEY.encode(), path.read_bytes())
        provenance = json.loads(Path(result['provenance_path']).read_text())
        self.assertTrue(provenance['outside_mask_unchanged'])
        self.assertEqual(provenance['identity']['width'], 32)
        Path(result['image_path']).write_bytes(b'corrupt')
        with self.assertRaisesRegex(CloudError, 'integrity'): self.generate()
        self.assertEqual(len(self.calls), 1)

    def test_request_budget_is_shared_across_anchors_and_survives_restart(self):
        ledger = self.root/'ledger.json'
        self.generate(budget=RequestBudget(1, ledger_path=ledger))
        self.source = self.source.copy(); self.source[0, 0] = 0
        with self.assertRaisesRegex(CloudError, 'budget exhausted'):
            self.generate(budget=RequestBudget(1, ledger_path=ledger))
        self.assertEqual(len(self.calls), 1)
        with self.assertRaisesRegex(CloudError, 'does not match'): RequestBudget(2, ledger_path=ledger)
        for invalid in (True, 0, 9, 1.5):
            with self.assertRaises(ValueError): RequestBudget(invalid)

    def test_rate_limit_retries_count_against_total_budget(self):
        calls = []
        def limited(**kwargs):
            calls.append(1)
            if len(calls) == 1: raise RateLimitError('limited')
            return self.provider(**kwargs)
        budget = RequestBudget(2, ledger_path=self.root/'ledger.json')
        with patch('seamstress.reconstruction_cloud.time.sleep'):
            self.generate(budget=budget, transport=limited)
        self.assertEqual(len(calls), 2)
        self.assertEqual([a['status'] for a in budget.attempts], ['rate_limited', 'complete'])

    def test_default_budget_never_retries_a_rate_limit(self):
        calls = []
        def limited(**_):
            calls.append(1)
            raise RateLimitError('limited')
        with self.assertRaisesRegex(CloudError, 'rate limit'): self.generate(transport=limited)
        self.assertEqual(len(calls), 1)

    def test_uncertain_provider_failure_is_redacted_and_not_retried_on_resume(self):
        def broken(**_): raise RuntimeError('Authorization: Bearer '+KEY)
        with self.assertRaises(CloudError) as raised: self.generate(transport=broken)
        self.assertNotIn(KEY, str(raised.exception))
        with self.assertRaisesRegex(CloudError, 'duplicate request'): self.generate()
        self.assertEqual(len(self.calls), 0)

    def test_cancel_before_request_and_after_response_does_not_publish(self):
        with self.assertRaises(InterruptedError): self.generate(cancelled=lambda: True)
        self.assertEqual(self.calls, [])
        cancelled = [False]
        def stop_after_response(**kwargs):
            response = self.provider(**kwargs)
            cancelled[0] = True
            return response
        with self.assertRaises(InterruptedError):
            self.generate(cancelled=lambda: cancelled[0], transport=stop_after_response)
        self.assertFalse(any((self.root/'cache').glob('*/background.png')))
        with self.assertRaisesRegex(CloudError, 'duplicate'): self.generate()

    def test_bad_provider_dimensions_fail_without_retry(self):
        def bad(**kwargs):
            kwargs['payload']['size'] = '16x16'
            return self.provider(**kwargs)
        with self.assertRaisesRegex(CloudError, 'unexpected dimensions'): self.generate(transport=bad)
        self.assertEqual(len(self.calls), 1)

    def test_soft_mask_blending_and_api_alpha_direction(self):
        self.mask[4, 8] = .5
        result = self.generate()
        final = np.asarray(Image.open(result['image_path']))
        np.testing.assert_array_equal(final[4, 8], np.rint(self.source[4, 8]*.5 + np.array([231,119,57])*.5))
        payload = self.calls[0]['payload']
        self.assertEqual(payload['n'], 1)
        mask_data = base64.b64decode(payload['mask']['image_url'].split(',')[1])
        alpha = np.asarray(Image.open(io.BytesIO(mask_data)))[:, :, 3]
        self.assertEqual(alpha[0, 0], 255)
        self.assertEqual(alpha[alpha.shape[0]//2, alpha.shape[1]//2], 0)

    def test_api_resolutions_fit_portrait_landscape_and_extreme_aspects(self):
        for height, width in [(720,1280),(1080,1920),(2160,3840),(40,1000),(1000,40),(1,1)]:
            with self.subTest(size=(width,height)):
                source = np.zeros((height,width,3),np.uint8)
                _, _, sizing = _api_images(source,np.ones((height,width),np.float32))
                rw,rh=sizing['request_width'],sizing['request_height']
                self.assertEqual(rw%16,0); self.assertEqual(rh%16,0)
                self.assertLessEqual(max(rw,rh)/min(rw,rh),3)
                self.assertGreaterEqual(rw*rh,655360)
                self.assertLessEqual(rw*rh,8294400)

    def test_invalid_masks_never_upload(self):
        for mask in [np.zeros((16,32)),np.ones((32,16)),np.full((16,32),np.nan),np.full((16,32),2)]:
            with self.assertRaises(ValueError):
                generate_background_anchor(self.source,mask,self.root,api_key=KEY,allow_upload=True,transport=self.provider)
        self.assertFalse(self.calls)


class TransportTests(unittest.TestCase):
    def test_http_quota_and_auth_messages_never_echo_response_secret(self):
        for status,code,expected in [(429,'insufficient_quota','quota'),(401,'invalid_api_key','rejected'),(500,None,'HTTP 500')]:
            error = urllib.error.HTTPError('https://api.openai.com/v1/images/edits',status,'bad',{},
                io.BytesIO(json.dumps({'error':{'code':code,'message':KEY}}).encode()))
            with patch('urllib.request.build_opener') as build:
                build.return_value.open.side_effect=error
                with self.assertRaisesRegex(CloudError,expected) as raised:
                    _https_edit(payload={},api_key=KEY)
                self.assertNotIn(KEY,str(raised.exception))

    def test_redirects_never_forward_authorization(self):
        with self.assertRaisesRegex(CloudError,'no credentials'):
            _NoRedirect().redirect_request(None,None,307,'redirect',{},'https://example.org')


if __name__=='__main__': unittest.main()
