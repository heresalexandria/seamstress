"""Layer composition precedes the frozen grade/crop, including scaled previews."""
import copy
from pathlib import Path
import shutil
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from seamstress import conform, projects, reconstruction as engine
from seamstress.media import VideoWriter
from seamstress.reconstruction_render import entry_for_manifest
from tests import test_reconstruction_workflow as workflow_fixture


@unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'), 'FFmpeg required')
class ReconstructionRenderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        workflow_fixture.ReconstructionWorkflowTests.setUpClass()

    @classmethod
    def tearDownClass(cls):
        workflow_fixture.ReconstructionWorkflowTests.tearDownClass()

    def setUp(self):
        self.fixture = workflow_fixture.ReconstructionWorkflowTests(); self.fixture.setUp()
        self.recipe = copy.deepcopy(self.fixture.baseline)
        identity = np.repeat(np.arange(256, dtype=float)[:, None], 3, axis=1)
        lut = identity.copy(); lut[:, 0] = 255*(identity[:, 0]/255)**.92
        self.recipe['grade_curves'] = [{'frame': 36, 'support_before': 3, 'support_after': 3,
                                      'left_lut': lut.tolist(), 'right_lut': lut.tolist()}]
        projects.atomic_json(self.fixture.plan_path, self.recipe)

    def tearDown(self):
        self.fixture.tearDown()

    def capture(self, plan, output, **options):
        frames = []
        class CapturingWriter(VideoWriter):
            def write(self, picture):
                frames.append(picture.copy())
                super().write(picture)
        with patch.object(conform, 'VideoWriter', CapturingWriter):
            conform.render_conform(self.fixture.source, plan, output, crf=0, **options)
        return frames

    def expected(self, raw, number, *, resize=None):
        scale = np.eye(3)
        if resize:
            raw = cv2.resize(raw, resize, interpolation=cv2.INTER_AREA)
            sx, sy = resize[0]/96, resize[1]/64
            scale = np.array([[sx, 0, (sx-1)/2], [0, sy, (sy-1)/2], [0, 0, 1]])
        segment = next(row for row in self.recipe['segments'] if row['start'] <= number < row['end'])
        matrix = scale @ np.asarray(self.recipe['view_matrix']) @ np.asarray(self.recipe['frame_matrices'][number]) @ np.linalg.inv(scale)
        picture, _, _ = conform.conform_frame(raw, matrix, np.asarray(segment['gain']), np.asarray(segment['bias']),
            self.recipe.get('edge_extension_pixels', 0)*max(scale[0, 0], scale[1, 1])+(1 if resize else 0))
        lut = conform.tone_lut_at(number, self.recipe['grade_curves'])
        return cv2.LUT(picture, lut[:, None, :]).round().astype(np.uint8) if lut is not None else picture

    def test_no_option_render_does_not_invoke_layers_and_native_composition_is_graded_once(self):
        fixture = self.fixture
        with patch('seamstress.reconstruction_render.FrameReconstruction', side_effect=AssertionError('Default renderer must not instantiate layers')):
            baseline = self.capture(fixture.plan_path, fixture.root/'baseline.mp4')
        self.assertEqual(len(baseline), 48)
        for number, picture in enumerate(baseline):
            np.testing.assert_array_equal(picture, self.expected(fixture.frames[number], number))
        manifest = fixture.authored(36)
        layered = copy.deepcopy(self.recipe); layered['reconstructions'] = [entry_for_manifest(manifest, layered)]
        path = fixture.root/'layered.plan.json'; projects.atomic_json(path, layered)
        actual = self.capture(path, fixture.root/'layered.mp4')
        bundle = engine.load_bundle(manifest)
        for number, picture in enumerate(actual):
            native = engine.apply_frame(bundle, number, fixture.frames[number])
            np.testing.assert_array_equal(picture, self.expected(native, number))
            if number != 36:
                np.testing.assert_array_equal(picture, baseline[number])
        self.assertFalse(np.array_equal(actual[36], baseline[36]))
        # Nonidentity geometry and grade make applying the frozen context a
        # second time observably wrong, not an identity-only false positive.
        self.assertGreater(np.abs(actual[36].astype(float)-self.expected(actual[36], 36)).mean(), 1.)

    def test_scaled_preview_composes_source_native_layers_before_resizing_and_grade(self):
        fixture = self.fixture; manifest = fixture.authored(36)
        layered = copy.deepcopy(self.recipe); layered['reconstructions'] = [entry_for_manifest(manifest, layered)]
        path = fixture.root/'preview.plan.json'; projects.atomic_json(path, layered)
        native_inputs = []; original = engine.apply_frame
        def record(bundle, number, rgb):
            native_inputs.append((number, rgb.copy()))
            return original(bundle, number, rgb)
        with patch.object(engine, 'apply_frame', side_effect=record):
            actual = self.capture(path, fixture.root/'preview.mp4', start_frame=33, end_frame=40, preview_width=48)
        self.assertEqual([number for number, _ in native_inputs], list(range(33, 40)))
        for number, rgb in native_inputs:
            self.assertEqual(rgb.shape, (64, 96, 3))
            np.testing.assert_array_equal(rgb, fixture.frames[number])
        bundle = engine.load_bundle(manifest)
        for number, picture in enumerate(actual, 33):
            self.assertEqual(picture.shape, (32, 48, 3))
            native = original(bundle, number, fixture.frames[number])
            np.testing.assert_array_equal(picture, self.expected(native, number, resize=(48, 32)))
