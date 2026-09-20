"""Local color safety, exact prototype evaluation, and renderer integration."""
from __future__ import annotations

import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from seamstress import conform
from seamstress.design import ALGORITHM as DESIGN_ALGORITHM, build_conform_plan
from seamstress.local_color import (ALGORITHM, apply_image, apply_samples,
                                   apply_local_color_at, local_color_at,
                                   local_color_provenance, prepare_local_color_curves,
                                   validate_local_color_curves)


def model(coefficients=(12, -8, 4), limit=18):
    return {'mode': 'hybrid', 'feature_scale': [255, 255, 255, 1, 1],
            'centers': [[.5, .5, .5, .5, .5]],
            'coefficients': [list(coefficients)], 'limit': limit}


def curve(frame=9, before=4, after=4):
    return {'frame': frame, 'support_before': before, 'support_after': after,
            'left': model(), 'right': model((-6, 10, -4))}


def reference_samples(rgb, xy, m):
    """Independent literal reference of the audited research prototype."""
    c = rgb.astype(np.float32)
    values = np.concatenate([c, xy], axis=1).astype(np.float32)/np.array(m['feature_scale'], np.float32)
    centers = np.array(m['centers'], np.float32)
    distances = np.maximum(np.sum(values*values, axis=1)[:, None]
                           + np.sum(centers*centers, axis=1)[None, :]-2*values@centers.T, 0)
    nearest = distances.min(axis=1)
    basis = np.exp(-.5*(distances-nearest[:, None]))
    basis /= np.maximum(basis.sum(axis=1)[:, None], 1e-9)
    basis *= np.exp(-np.maximum(nearest-3, 0)/3)[:, None]
    if m.get('response') == 'directional-gamut-v2':
        # The research directional response retains float64 coefficients and
        # response arithmetic after its original float32 RBF basis.
        raw = basis@np.array(m['coefficients'])
        c = c.astype(float)
        low = np.mean(c*c, axis=1, keepdims=True)
        high = np.mean((255-c)**2, axis=1, keepdims=True)
        gate = (1-np.exp(-low/64))*(1-np.exp(-high/64))
        offset = m['limit']*np.tanh(raw/m['limit'])*gate
        available = np.where(offset >= 0, 255-c, c)
        return c+available*np.tanh(offset/np.maximum(available, 1e-12))
    raw = basis@np.array(m['coefficients'], np.float32)
    return c+m['limit']*np.tanh(raw/m['limit'])*4*(c/255)*(1-c/255)


class LocalColorTests(unittest.TestCase):
    def test_single_center_correction_has_independent_channels_and_exact_headroom(self):
        rgb = np.array([[128, 100, 210]], np.float32)
        actual = apply_samples(rgb, np.array([[.5, .5]]), model())
        expected = rgb+18*np.tanh(np.array([[12, -8, 4]], np.float32)/18)*4*(rgb/255)*(1-rgb/255)
        np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-5)
        self.assertGreater(actual[0, 0], rgb[0, 0])
        self.assertLess(actual[0, 1], rgb[0, 1])

    def test_confidence_tapers_unrepresented_colors_positions_toward_identity(self):
        m = model(); m['centers'] = [[.5, .5, .5, 3, .5]]
        rgb = np.array([[127.5, 127.5, 127.5]], np.float32)
        # Feature distance = 2.5**2. Confidence acts before the tanh limiter.
        confidence = np.exp(-(6.25-3)/3)
        expected = rgb+18*np.tanh(np.array([[12, -8, 4]])*confidence/18)
        np.testing.assert_allclose(apply_samples(rgb, np.array([[.5, .5]]), m), expected, atol=2e-5)

    def test_quintic_support_uses_original_frame_indices_and_two_full_strength_anchors(self):
        c = curve(); image = np.full((5, 7, 3), 128, np.uint8)
        prepared = prepare_local_color_curves([c], 18)
        for index in (0, 4, 13, 17):
            self.assertIsNone(local_color_at(index, prepared))
            self.assertIs(apply_local_color_at(image, index, prepared), image)
        for index, side in ((8, 'left'), (9, 'right')):
            m, weight = local_color_at(index, [c])
            self.assertIs(m, c[side]); self.assertEqual(weight, 1)
            np.testing.assert_array_equal(apply_local_color_at(image, index, prepared), apply_image(image, c[side]))
        self.assertEqual(local_color_at(6, [c])[1], .5)
        self.assertEqual(local_color_at(11, [c])[1], .5)
        # The final support may extend beyond the video, matching grade LUTs.
        validate_local_color_curves([curve(16, 4, 8)], 18)

    def test_zero_strength_and_identity_models_return_original_without_rounding(self):
        image = np.arange(4*9*3, dtype=np.uint8).reshape(4, 9, 3)
        self.assertIs(apply_image(image, model(), weight=0), image)
        self.assertIs(apply_image(image, model((0, 0, 0))), image)
        self.assertIs(apply_local_color_at(image, 8, []), image)

    def test_black_white_and_all_channel_levels_remain_in_range_at_maximum_limit(self):
        levels = np.repeat(np.arange(256, dtype=np.float32)[:, None], 3, axis=1)
        for sign in (-1, 1):
            result = apply_samples(levels, np.full((256, 2), .5), model((sign*1e12,)*3, 63.75))
            self.assertGreaterEqual(float(result.min()), 0)
            self.assertLessEqual(float(result.max()), 255)
            np.testing.assert_array_equal(result[[0, -1]], levels[[0, -1]])

    def test_native_xy_and_chunking_match_independent_prototype(self):
        rng = np.random.default_rng(26)
        image = rng.integers(0, 256, (19, 31, 3), dtype=np.uint8)
        m = {'mode': 'hybrid', 'feature_scale': [48, 48, 48, .2, .2],
             'centers': rng.uniform(0, 5, (17, 5)).tolist(),
             'coefficients': rng.normal(0, 9, (17, 3)).tolist(), 'limit': 18}
        y, x = np.indices(image.shape[:2])
        xy = np.column_stack((x.ravel()/30, y.ravel()/18))
        expected = np.rint(reference_samples(image.reshape(-1, 3), xy, m)).astype(np.uint8).reshape(image.shape)
        for size in (7, 43, 32768):
            np.testing.assert_array_equal(apply_image(image, m, chunk_size=size), expected)
        # A one-pixel dimension uses coordinate zero, avoiding division by zero.
        self.assertTrue(np.isfinite(apply_image(image[:1], m)).all())

    def test_malformed_models_and_supports_fail_before_render(self):
        invalid_models = []
        for key, value in (('mode', 'spatial'), ('feature_scale', [48]*4),
                           ('feature_scale', [48, 48, 48, 0, 1]),
                           ('feature_scale', [48, 48, 48, float('nan'), 1]),
                           ('feature_scale', [48, 48, 48, 1e-30, 1]),
                           ('feature_scale', ['48', '48', '48', '1', '1']),
                           ('feature_scale', [True]*5),
                           ('centers', []), ('centers', [[1, 2]]),
                           ('centers', [[0]*5]*513), ('coefficients', [[1, 2]]),
                           ('coefficients', np.array([[1+2j, 0, 0]])),
                           ('coefficients', [[float('inf'), 0, 0]]),
                           ('limit', 0), ('limit', 64), ('limit', True),
                           ('limit', 1e-300), ('limit', 10**1000)):
            m = model(); m[key] = value; invalid_models.append(m)
        for m in invalid_models:
            c = curve(); c['left'] = m
            with self.subTest(model=m), self.assertRaises(ValueError):
                validate_local_color_curves([c], 18)
        for curves in (None, [{}], [curve(0)], [curve(18)], [curve(True)],
                       [curve(before=0)], [curve(before=9)], [curve(after=1.5)],
                       [curve(), curve(12, 2, 2)], [curve(), curve()]):
            with self.subTest(curves=curves), self.assertRaises(ValueError):
                validate_local_color_curves(curves, 18)
        for weight in (-1, 1.01, float('nan'), 10**1000):
            with self.assertRaises(ValueError):
                apply_image(np.zeros((2, 2, 3), np.uint8), model(), weight)

    def test_provenance_changes_with_model_and_support_but_not_dictionary_key_order(self):
        curves = [curve()]
        first = local_color_provenance(curves)
        self.assertEqual(first['algorithm'], ALGORITHM)
        self.assertEqual(first, local_color_provenance(json.loads(json.dumps(curves, sort_keys=True))))
        changed = copy.deepcopy(curves); changed[0]['support_after'] += 1
        self.assertNotEqual(first['curves_sha256'], local_color_provenance(changed)['curves_sha256'])
        changed = copy.deepcopy(curves); changed[0]['left']['coefficients'][0][0] += 1
        self.assertNotEqual(first['models'][0]['left_sha256'], local_color_provenance(changed)['models'][0]['left_sha256'])

    def test_explicit_v1_matches_omitted_response_exactly_and_unknown_responses_are_rejected(self):
        image = np.random.default_rng(201).integers(0, 256, (31, 43, 3), dtype=np.uint8)
        implicit = model(); explicit = dict(implicit, response='headroom-v1')
        np.testing.assert_array_equal(apply_image(image, implicit), apply_image(image, explicit))
        for response in ('v2', '', None, 2, True, {}, []):
            c = curve(); c['left']['response'] = response
            with self.subTest(response=response), self.assertRaisesRegex(ValueError, 'response'):
                validate_local_color_curves([c], 18)

    def test_directional_response_preserves_true_endpoints_but_can_move_saturated_channels(self):
        rgb = np.array([[0, 0, 0], [255, 255, 255], [0, 80, 160], [255, 80, 160]], np.float32)
        xy = np.full((len(rgb), 2), .5)
        positive = dict(model((20, 20, 20), 63.75), response='directional-gamut-v2')
        negative = dict(model((-20, -20, -20), 63.75), response='directional-gamut-v2')
        for m in (positive, negative):
            np.testing.assert_array_equal(apply_samples(rgb, xy, m)[:2], rgb[:2])
        self.assertGreater(apply_samples(rgb, xy, positive)[2, 0], 0)
        self.assertLess(apply_samples(rgb, xy, negative)[3, 0], 255)
        self.assertEqual(apply_samples(rgb, xy, model((20,)*3, 63.75))[2, 0], 0)
        self.assertEqual(apply_samples(rgb, xy, model((-20,)*3, 63.75))[3, 0], 255)
        levels = [0, 1, 10, 64, 128, 192, 245, 254, 255]
        colors = np.array(np.meshgrid(levels, levels, levels)).reshape(3, -1).T
        for sign in (-1, 1):
            m = dict(model((sign*1e12, -sign*1e12, sign*1e12), 63.75), response='directional-gamut-v2')
            corrected = apply_samples(colors, np.full((len(colors), 2), .5), m)
            self.assertTrue(np.isfinite(corrected).all())
            self.assertGreaterEqual(float(corrected.min()), 0)
            self.assertLessEqual(float(corrected.max()), 255)

    def test_directional_response_quantized_parity_with_research_at_native_xy_and_chunk_boundaries(self):
        rng = np.random.default_rng(291)
        image = rng.integers(0, 256, (79, 103, 3), dtype=np.uint8)
        image[0] = [0, 0, 0]; image[1] = [255, 255, 255]
        image[2] = [0, 50, 250]; image[3] = [255, 130, 50]
        m = {'mode': 'hybrid', 'response': 'directional-gamut-v2',
             'feature_scale': [48, 48, 48, .2, .2],
             'centers': rng.uniform(0, 5, (31, 5)).tolist(),
             'coefficients': rng.normal(0, 75, (31, 3)).tolist(), 'limit': 63.75}
        y, x = np.indices(image.shape[:2]); xy = np.column_stack((x.ravel()/102, y.ravel()/78))
        expected = reference_samples(image.reshape(-1, 3), xy, m)
        np.testing.assert_allclose(apply_samples(image.reshape(-1, 3), xy, m), expected, rtol=0, atol=1e-12)
        quantized = np.rint(expected).astype(np.uint8).reshape(image.shape)
        np.testing.assert_array_equal(apply_image(image, m), quantized)
        # Very small batches can choose a different float32 BLAS reduction for
        # the RBF features. Only rare rounding-boundary pixels may differ.
        for size in (7, 43):
            difference = abs(apply_image(image, m, chunk_size=size).astype(int)-quantized)
            self.assertLessEqual(int(difference.max()), 1)
            self.assertLess(np.count_nonzero(difference)/difference.size, .001)

    def test_directional_support_and_mixed_response_provenance_are_explicit(self):
        c = curve(); c['right']['response'] = 'directional-gamut-v2'
        prepared = prepare_local_color_curves([c], 18)
        image = np.full((5, 7, 3), [0, 100, 200], np.uint8)
        for index in (4, 13):
            self.assertIs(apply_local_color_at(image, index, prepared), image)
        np.testing.assert_array_equal(apply_local_color_at(image, 9, prepared), apply_image(image, c['right']))
        np.testing.assert_array_equal(apply_local_color_at(image, 11, prepared), apply_image(image, c['right'], weight=.5))
        provenance = local_color_provenance([c])
        self.assertEqual(provenance['algorithm'], 'hybrid-rbf-mixed-responses-v2')
        self.assertEqual(provenance['models'][0]['left_response'], 'headroom-v1')
        self.assertEqual(provenance['models'][0]['right_response'], 'directional-gamut-v2')
        changed = copy.deepcopy(c); changed['right'].pop('response')
        self.assertNotEqual(provenance['models'][0]['right_sha256'], local_color_provenance([changed])['models'][0]['right_sha256'])


class LocalColorOrchestrationTests(unittest.TestCase):
    def setUp(self):
        self.metadata = {'width': 16, 'height': 12, 'frame_count': 18,
                         'frame_count_estimated': False,
                         'fps_fraction': '24000/1001', 'fps': 24000/1001, 'has_audio': False}
        self.frames = np.array([np.full((12, 16, 3), [55+i*4, 100, 160-i*2], np.uint8) for i in range(18)])
        self.plan = {'schema_version': 3, 'method': 'source_conform', 'source': self.metadata,
                     'source_sha256': hashlib.sha256(b'unchanged source').hexdigest(),
                     'segments': [{'start': 0, 'end': 18, 'matrix': np.eye(3).tolist(),
                                   'gain': [1, 1, 1], 'bias': [0, 0, 0]}]}

    def render_recorded(self, recipe, start, end):
        written = []
        class Writer:
            def __init__(self, *args, **kwargs): self.frames_written = 0
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def write(self, frame):
                written.append(frame.copy()); self.frames_written += 1
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); source = root/'source.mp4'; source.write_bytes(b'unchanged source')
            plan = root/'plan.json'; plan.write_text(json.dumps(recipe)); output = root/'output.mp4'
            def recorded_metadata(path):
                return {**self.metadata, 'frame_count': end-start} if Path(path).name == 'complete.mp4' else self.metadata
            with patch.object(conform, 'probe', side_effect=recorded_metadata), \
                 patch.object(conform, 'VideoWriter', Writer), \
                 patch.object(conform, 'iter_frames', return_value=iter(self.frames[start:end])) as decode, \
                 patch.object(conform, '_run', return_value=SimpleNamespace(stdout=b'ffmpeg version test\n')), \
                 patch.object(conform, '_tool', return_value='ffmpeg'), \
                 patch.object(conform, 'mux_audio', side_effect=lambda a, b, c: c.write_bytes(b'encoded')), \
                 contextlib.redirect_stdout(io.StringIO()):
                report = conform.render_conform(source, plan, output, start_frame=start, end_frame=end)
            decode.assert_called_once_with(source, start, end-start)
            self.assertEqual(source.read_bytes(), b'unchanged source')
        return np.stack(written), report

    def test_preview_applies_local_color_after_lut_at_original_indices_and_keeps_other_frames(self):
        self.plan['local_color_curves'] = [curve()]
        levels = np.repeat(np.arange(256)[:, None], 3, axis=1)
        table = (255*(levels/255)**1.2).tolist()
        self.plan['grade_curves'] = [{'frame': 9, 'support_before': 4, 'support_after': 4,
                                      'left_lut': table, 'right_lut': table}]
        actual, report = self.render_recorded(self.plan, 4, 15)
        expected = []
        for index in range(4, 15):
            frame = self.frames[index]
            lut = conform.tone_lut_at(index, self.plan['grade_curves'])
            if lut is not None:
                frame = np.stack([lut[frame[..., c], c] for c in range(3)], axis=-1).round().astype(np.uint8)
            active = local_color_at(index, self.plan['local_color_curves'])
            if active:
                m, weight = active
                # A single broad center has confidence 1 for all fixture RGBXY.
                floating = frame.astype(np.float32)
                corrected = floating+weight*18*np.tanh(np.array(m['coefficients'], np.float32)/18)*4*(floating/255)*(1-floating/255)
                frame = corrected.round().astype(np.uint8)
            expected.append(frame)
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(actual[[0, -2, -1]], self.frames[[4, 13, 14]])
        self.assertEqual((report['source_start_frame'], report['source_end_frame_exclusive']), (4, 15))
        self.assertEqual(report['local_color_curve_count'], 1)
        self.assertEqual(report['local_color_model_provenance'], local_color_provenance(self.plan['local_color_curves']))

    def test_absent_local_models_never_enter_new_render_branch(self):
        with patch.object(conform, 'apply_local_color_at', side_effect=AssertionError('Unexpected local-color call')):
            actual, report = self.render_recorded(self.plan, 0, 18)
        np.testing.assert_array_equal(actual, self.frames)
        self.assertEqual(report['local_color_curve_count'], 0)
        self.assertNotIn('local_color_model_provenance', report)

    def test_renderer_honors_directional_response_on_both_sides_of_preview_cut(self):
        c = curve()
        for side in ('left', 'right'):
            c[side]['response'] = 'directional-gamut-v2'
        self.plan['local_color_curves'] = [c]
        actual, report = self.render_recorded(self.plan, 7, 11)
        y, x = np.indices(self.frames.shape[1:3]); xy = np.column_stack((x.ravel()/15, y.ravel()/11))
        expected = []
        for index in range(7, 11):
            frame = self.frames[index]; m, weight = local_color_at(index, [c])
            floating = reference_samples(frame.reshape(-1, 3), xy, m).reshape(frame.shape)
            expected.append(np.rint(frame+weight*(floating-frame)).astype(np.uint8))
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(report['local_color_model_provenance']['algorithm'], 'hybrid-rbf-directional-gamut-v2')

    def test_schema_rejects_invalid_local_model(self):
        self.plan['local_color_curves'] = [curve()]
        self.plan['local_color_curves'][0]['right']['coefficients'] = []
        with self.assertRaisesRegex(ValueError, 'coefficients'):
            conform.validate_conform_plan(self.plan, self.metadata)

    def test_invalid_provenance_metadata_fails_before_encoder_or_output_publication(self):
        self.plan['local_color_curves'] = [curve()]
        self.plan['local_color_curves'][0]['left']['fit_metadata'] = {'error': float('nan')}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); source = root/'source.mp4'; source.write_bytes(b'unchanged source')
            plan = root/'plan.json'; plan.write_text(json.dumps(self.plan)); output = root/'output.mp4'
            with patch.object(conform, 'probe', return_value=self.metadata), \
                 patch.object(conform, 'VideoWriter') as writer, \
                 patch.object(conform, '_run', return_value=SimpleNamespace(stdout=b'ffmpeg version test\n')), \
                 patch.object(conform, '_tool', return_value='ffmpeg'):
                with self.assertRaisesRegex(ValueError, 'JSON compliant'):
                    conform.render_conform(source, plan, output)
                writer.assert_not_called()
            self.assertFalse(output.exists())
            self.assertFalse(output.with_suffix('.repair.json').exists())

    def test_design_passthrough_is_optional_and_does_not_change_geometry(self):
        metadata = {'width': 80, 'height': 60, 'frame_count': 96, 'fps_fraction': '24000/1001'}
        calibration = {'schema_version': 1, 'method': 'source_conform_calibration',
                       'algorithm': DESIGN_ALGORITHM, 'source': metadata, 'source_sha256': 'a'*64,
                       'parameters': {'geometry_support': 16, 'rate_support': 4, 'source_margin_pixels': 2,
                                      'max_view_crop_fraction_total_dimension': .15},
                       'cuts': [{'frame': 40, 'right_to_left_matrix': np.eye(3).tolist(),
                                 'pre_rate': [0]*4, 'post_rate': [0]*4, 'ease_rate': False}]}
        baseline = build_conform_plan(calibration, metadata)
        self.assertNotIn('local_color_curves', baseline)
        calibration['local_color_curves'] = [curve(40, 16, 16)]
        revised = build_conform_plan(calibration, metadata)
        self.assertEqual(revised.pop('local_color_curves'), calibration['local_color_curves'])
        self.assertEqual(revised, baseline)
        calibration['local_color_curves'][0]['left']['limit'] = 100
        with self.assertRaisesRegex(ValueError, 'limit'):
            build_conform_plan(calibration, metadata)


if __name__ == '__main__':
    unittest.main()
