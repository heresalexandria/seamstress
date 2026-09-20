"""Whole-timeline suggestions must be supported by edits, never a timer."""
from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from seamstress import detection


def drawing(seed=7, height=96, width=160):
    rng = np.random.default_rng(seed)
    image = np.full((height, width, 3), (35, 74, 121), np.uint8)
    for _ in range(24):
        x, y = rng.integers(0, width-12), rng.integers(0, height-12)
        color = tuple(int(x) for x in rng.integers(30, 225, 3))
        cv2.rectangle(image, (x, y), (x+int(rng.integers(5, 28)), y+int(rng.integers(5, 23))), color, -1)
    cv2.circle(image, (width//2, height//2), 15, (188, 134, 74), -1)
    cv2.circle(image, (width//2, height//2), 15, (12, 14, 17), 2)
    return image


def moving_frames(count=216, fps=12, mode='pan', edits=(), portrait=False):
    height, width = (160, 96) if portrait else (96, 160)
    image = drawing(height=height, width=width)
    other = drawing(seed=91, height=height, width=width)
    frames = []
    for index in range(count):
        position = index
        if mode == 'on_twos':
            position = index//2*2
        if mode == 'hold':
            position = min(index, 78) if index < 90 else index-12
        shift = position*.22
        base = other if mode == 'hard_cut' and index >= 8*fps else image
        if mode == 'hard_cut' and index >= 8*fps:
            base = np.flip(base, axis=1).copy()
        frame = cv2.warpAffine(base, np.float32([[1, 0, shift], [0, 1, 0]]),
                               (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        if mode == 'reframe' and index >= 8*fps:
            frame = cv2.warpAffine(frame, np.float32([[1, 0, 6], [0, 1, 0]]),
                                   (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        if mode == 'fade':
            frame = np.rint(frame.astype(float)*(.65+.3*index/(count-1))).astype(np.uint8)
        if mode == 'subtle':
            step = sum(index >= boundary for boundary in edits)
            # Modest material-dependent grading plus a small continuation reframe.
            gains = np.array([1+.023*step, 1-.01*step, 1+.012*step])
            frame = cv2.warpAffine(frame, np.float32([[1, 0, step*.8], [0, 1, step*.3]]),
                                   (width, height), borderMode=cv2.BORDER_WRAP)
            frame = np.rint(np.clip(frame*gains+[step, 0, -step], 0, 255)).astype(np.uint8)
        frames.append(frame)
    return frames


class DetectionTests(unittest.TestCase):
    def run_scan(self, frames, fps=12, **kwargs):
        h, w = frames[0].shape[:2]
        meta = {'path': '/unused/video.mp4', 'width': w, 'height': h,
                'frame_count': len(frames), 'frame_count_estimated': False,
                'fps': fps, 'fps_fraction': str(fps)+'/1', 'duration': len(frames)/fps,
                'has_audio': False, 'codec_name': 'h264', 'pix_fmt': 'yuv420p'}
        closed = []

        def decode(path, size=None):
            try:
                for image in frames:
                    yield cv2.resize(image, size, interpolation=cv2.INTER_AREA) if size else image
            finally:
                closed.append(True)

        try:
            with patch.object(detection, 'probe', return_value=meta), patch.object(detection, 'iter_frames', side_effect=decode):
                result = detection.detect_seams(Path('/unused/video.mp4'), **kwargs)
        finally:
            self.last_decoder_closed = bool(closed)
        self.assertTrue(closed)
        return result

    def test_arbitrary_eight_second_cut_without_interval_hint(self):
        result = self.run_scan(moving_frames(mode='hard_cut'), options={'interval_hints': []})
        self.assertIn(96, [s['frame'] for s in result['seams']])
        seam = next(s for s in result['seams'] if s['frame'] == 96)
        self.assertEqual(seam['time'], 8.)
        self.assertEqual(seam['origin'], 'detected')
        self.assertTrue(seam['enabled'])
        self.assertTrue(0 <= seam['confidence'] <= 1)
        self.assertTrue(seam['review_advisories'])

    def test_regular_pan_fade_cadence_and_short_hold_do_not_create_edits(self):
        for mode in ['pan', 'fade', 'on_twos', 'hold']:
            with self.subTest(mode=mode):
                result = self.run_scan(moving_frames(mode=mode))
                self.assertEqual(result['seams'], [], [(s['frame'], s['score']) for s in result['seams']])

    def test_arbitrary_rigid_reframe_without_color_change(self):
        result = self.run_scan(moving_frames(mode='reframe'), options={'interval_hints': []})
        self.assertIn(96, [s['frame'] for s in result['seams']])

    def test_unrelated_scene_cut_is_explicitly_flagged_for_review(self):
        frames = [np.full((96, 160, 3), (170, 30, 40) if i < 96 else (20, 170, 140), np.uint8)
                  for i in range(156)]
        result = self.run_scan(frames, options={'interval_hints': []})
        seam = next(s for s in result['seams'] if s['frame'] == 96)
        self.assertEqual(seam['classification'], 'scene_cut')
        self.assertTrue(any('unrelated' in x for x in seam['review_advisories']))

    def test_subtle_periodic_continuations_need_actual_visual_evidence(self):
        cuts = [120, 240]
        result = self.run_scan(moving_frames(count=324, mode='subtle', edits=cuts),
                               options={'interval_hints': [10]})
        found = [s['frame'] for s in result['seams']]
        for cut in cuts:
            self.assertTrue(any(abs(n-cut) <= 1 for n in found), (cut, found))
        self.assertTrue(all(any(abs(n-cut) <= 1 for cut in cuts) for n in found), found)

    def test_interval_hints_do_not_force_static_boundaries(self):
        frames = [drawing() for _ in range(192)]
        result = self.run_scan(frames, options={'interval_hints': [5, 10, 15]})
        self.assertEqual(result['seams'], [])
        self.assertGreater(result['diagnostics']['periodic_search_pair_count'], 0)

    def test_portrait_proxy_preserves_aspect_ratio_and_reports_progress(self):
        events = []
        frames = moving_frames(count=24, portrait=True)
        result = self.run_scan(frames, options={'scan_width': 80, 'interval_hints': []}, progress=events.append)
        self.assertEqual(result['diagnostics']['proxy_size'], [48, 80])
        self.assertEqual(events[-1]['fraction'], 1.)
        self.assertEqual(events[-1]['stage'], 'complete')
        self.assertTrue(all(0 <= p['fraction'] <= 1 for p in events))

    def test_cancellation_closes_decoder_and_removes_temporary_cache(self):
        frames = moving_frames(count=96)
        call = [0]
        def cancelled():
            call[0] += 1
            return call[0] > 15
        with tempfile.TemporaryDirectory() as root:
            original = detection.tempfile.TemporaryDirectory
            with patch.object(detection.tempfile, 'TemporaryDirectory', side_effect=lambda **kw: original(dir=root, **kw)):
                with self.assertRaises(detection.DetectionCancelled):
                    self.run_scan(frames, cancelled=cancelled)
            self.assertTrue(self.last_decoder_closed)
            self.assertEqual(list(Path(root).iterdir()), [])

    def test_invalid_options_rejected_before_decoding(self):
        for options in [{'sensitivity': float('nan')}, {'interval_hints': [0]}, {'scan_width': True}, {'min_spacing': -1}]:
            with self.subTest(options=options), patch.object(detection, 'probe') as probe:
                with self.assertRaises(ValueError):
                    detection.detect_seams(Path('/unused.mp4'), options=options)
                probe.assert_not_called()


if __name__ == '__main__':
    unittest.main()
