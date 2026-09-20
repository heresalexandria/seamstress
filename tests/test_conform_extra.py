"""Conform orchestration with real fractional-rate video and copied AAC audio."""
from __future__ import annotations

import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from seamstress import conform
from seamstress.media import VideoWriter, mux_audio, probe, read_frames


ROOT = Path(__file__).resolve().parents[1]
CURVES = ROOT / "research/segment-color/protected-midpoint-curves.json"


def run(*args):
    return subprocess.run(list(args), check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg integration requires ffmpeg and ffprobe")
class ConformIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix="seamstress-conform-test-")
        cls.root = Path(cls.temp.name)
        video, audio = cls.root/"pictures.mp4", cls.root/"audio.m4a"
        with VideoWriter(video, 64, 48, "24000/1001", crf=12, preset="ultrafast") as writer:
            for i in range(18):
                frame = np.full((48, 64, 3), [20+i*4, 70, 140-i*3], np.uint8)
                frame[12:28, 2+i*2:12+i*2] = [210, 190, 40]
                frame[35:41, 5:50] = [8, 8, 8]
                writer.write(frame)
        run("ffmpeg", "-v", "error", "-nostdin", "-n", "-f", "lavfi", "-i",
            "sine=frequency=523:sample_rate=48000:duration=0.8", "-c:a", "aac", "-b:a", "96k", str(audio))
        cls.source = cls.root/"source.mp4"
        mux_audio(video, audio, cls.source)
        cls.metadata = probe(cls.source)
        cls.source_hash = digest(cls.source)
        cls.decoded = read_frames(cls.source, 0, 18)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def plan(self):
        return {"schema_version": 3, "method": "source_conform", "source": self.metadata,
                "source_sha256": self.source_hash,
                "segments": [{"start": 0, "end": 9, "matrix": np.eye(3).tolist(), "gain": [1,1,1], "bias": [0,0,0]},
                             {"start": 9, "end": 18, "matrix": np.eye(3).tolist(), "gain": [1,1,1], "bias": [0,0,0]}],
                "frame_matrices": [np.eye(3).tolist() for _ in range(18)]}

    def write_plan(self, plan, name):
        path = self.root/f"{name}.json"
        path.write_text(json.dumps(plan))
        return path

    def audio_packets(self, path):
        value = run("ffprobe", "-v", "error", "-select_streams", "a:0", "-show_packets",
                    "-show_data_hash", "sha256", "-show_entries", "packet=pts,dts,duration,size,data_hash",
                    "-of", "json", str(path))
        return json.loads(value.stdout)["packets"]

    def test_cli_real_grade_render_preserves_rate_count_source_and_audio_packets(self):
        if not CURVES.is_file():
            self.skipTest("Source-specific research LUT artifact is unavailable")
        curve = copy.deepcopy(json.loads(CURVES.read_text())[0])
        curve.update(frame=9, support_before=4, support_after=4)
        plan = self.plan(); plan["grade_curves"] = [curve]
        output = self.root/"cli-graded.mp4"
        recipe = self.write_plan(plan, "cli-graded")
        result = run(sys.executable, "-m", "seamstress", "conform", str(self.source),
                     "--plan", str(recipe), "--output", str(output), "--crf", "12")
        self.assertIn(b'"protected_grade_curve_count": 1', result.stdout)
        rendered = probe(output)
        self.assertEqual((rendered["frame_count"], rendered["fps_fraction"]), (18, "24000/1001"))
        self.assertEqual((rendered["width"], rendered["height"]), (64, 48))
        self.assertTrue(rendered["has_audio"])
        # Packet hashes verify compressed audio content; timestamps/durations
        # verify it was not silently shifted while remuxing.
        self.assertEqual(self.audio_packets(self.source), self.audio_packets(output))
        self.assertEqual(digest(self.source), self.source_hash)
        sidecar = json.loads(output.with_suffix(".repair.json").read_text())
        self.assertEqual((sidecar["synthesized_frames"], sidecar["frame_mapping"]), (0, "identity"))
        self.assertEqual(sidecar["plan_sha256"], digest(recipe))
        self.assertEqual(sidecar["renderer"], "seamstress.conform")
        self.assertTrue(sidecar["renderer_version"])
        self.assertTrue(sidecar["ffmpeg_version"].startswith("ffmpeg version "))
        # The actual output must show the loaded grade, rather than merely
        # recording a curve count in metadata and ignoring the LUT.
        encoded = read_frames(output, 9, 1)[0]
        lut = np.array(curve["right_lut"])
        expected = np.stack([lut[self.decoded[9, ..., c], c] for c in range(3)], axis=-1).round()
        corrected_error = np.mean(abs(encoded.astype(float)-expected))
        identity_error = np.mean(abs(encoded.astype(float)-self.decoded[9]))
        self.assertLess(corrected_error, identity_error)

    def test_renderer_passes_each_identity_source_frame_unchanged_to_encoder(self):
        written = []

        class RecordingWriter(VideoWriter):
            def write(self, frame):
                written.append(frame.copy())
                super().write(frame)

        recipe = self.write_plan(self.plan(), "identity")
        with patch.object(conform, "VideoWriter", RecordingWriter), contextlib.redirect_stdout(io.StringIO()):
            conform.render_conform(self.source, recipe, self.root/"identity-output.mp4", crf=12)
        np.testing.assert_array_equal(np.stack(written), self.decoded)
        self.assertEqual(digest(self.source), self.source_hash)

    def test_output_collisions_and_wrong_source_fail_before_encoder(self):
        recipe = self.write_plan(self.plan(), "guards")
        sentinel = self.root/"occupied.mp4"; sentinel.write_bytes(b"keep existing output")
        sidecar_output = self.root/"sidecar-collision.mp4"
        sidecar_output.with_suffix(".repair.json").write_text("keep existing sidecar")
        with patch.object(conform, "VideoWriter") as writer:
            for destination in (self.source, sentinel, sidecar_output):
                with self.assertRaisesRegex(ValueError, "new output"):
                    conform.render_conform(self.source, recipe, destination)
            changed = self.plan(); changed["source_sha256"] = "0"*64
            wrong_recipe = self.write_plan(changed, "wrong-source")
            with self.assertRaisesRegex(ValueError, "fingerprint"):
                conform.render_conform(self.source, wrong_recipe, self.root/"wrong-source.mp4")
            writer.assert_not_called()
        self.assertEqual(sentinel.read_bytes(), b"keep existing output")
        self.assertEqual(sidecar_output.with_suffix(".repair.json").read_text(), "keep existing sidecar")
        self.assertEqual(digest(self.source), self.source_hash)

    def test_cli_collision_returns_failure_without_modifying_source(self):
        recipe = self.write_plan(self.plan(), "cli-collision")
        result = subprocess.run([sys.executable, "-m", "seamstress", "conform", str(self.source),
                                 "--plan", str(recipe), "--output", str(self.source)],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.assertEqual(result.returncode, 1)
        self.assertIn(b"output must differ", result.stderr)
        self.assertEqual(digest(self.source), self.source_hash)

    def test_preview_cli_keeps_original_indices_and_trims_audio_to_the_same_range(self):
        plan = self.plan()
        # The selected window crosses a segment and grade boundary. A renderer
        # that accidentally renumbers its LUT lookup from zero will differ.
        levels = np.repeat(np.arange(256)[:, None], 3, axis=1)
        table = (255*(levels/255)**1.18).tolist()
        plan["grade_curves"] = [{"frame": 9, "support_before": 4, "support_after": 4,
                                 "left_lut": table, "right_lut": table}]
        recipe = self.write_plan(plan, "preview")
        output = self.root/"preview-output.mp4"
        run(sys.executable, "-m", "seamstress", "conform", str(self.source), "--plan", str(recipe),
            "--output", str(output), "--start-frame", "6", "--end-frame", "14", "--crf", "12")
        metadata = probe(output)
        self.assertEqual((metadata["frame_count"], metadata["fps_fraction"]), (8, "24000/1001"))
        sidecar = json.loads(output.with_suffix(".repair.json").read_text())
        self.assertEqual((sidecar["source_start_frame"], sidecar["source_end_frame_exclusive"]), (6, 14))
        self.assertTrue(sidecar["preview"])
        self.assertEqual(sidecar["frame_mapping"], "contiguous_original_frames")
        self.assertEqual(sidecar["audio_mode"], "trimmed and encoded as AAC")
        actual = read_frames(output, 0, 8)
        expected = []
        for source_index in range(6, 14):
            frame = self.decoded[source_index]
            lut = conform.tone_lut_at(source_index, plan["grade_curves"])
            expected.append(frame if lut is None else np.stack([lut[frame[..., c], c] for c in range(3)], axis=-1).round())
        self.assertLess(np.mean(abs(actual.astype(float)-np.stack(expected))), 1.5)
        # Compare decoded waveform phase within the trimmed interval. 48 kHz
        # contains exactly 2002 samples per 24000/1001 source frame.
        def pcm(path):
            data = run("ffmpeg", "-v", "error", "-i", str(path), "-map", "0:a:0",
                       "-ac", "1", "-ar", "48000", "-f", "f32le", "pipe:1").stdout
            return np.frombuffer(data, dtype="<f4")
        source_audio, output_audio = pcm(self.source), pcm(output)
        expected_audio = source_audio[6*2002:14*2002]
        length = min(len(output_audio), len(expected_audio))
        self.assertGreater(np.corrcoef(output_audio[2048:length-1024], expected_audio[2048:length-1024])[0, 1], .995)
        self.assertLess(abs(len(output_audio)-8*2002), 1025)
        self.assertNotEqual(self.audio_packets(self.source), self.audio_packets(output))
        self.assertEqual(digest(self.source), self.source_hash)

    def test_preview_identity_passes_only_requested_original_frames_to_encoder(self):
        written = []

        class RecordingWriter(VideoWriter):
            def write(self, frame):
                written.append(frame.copy())
                super().write(frame)

        recipe = self.write_plan(self.plan(), "preview-identity")
        with patch.object(conform, "VideoWriter", RecordingWriter), contextlib.redirect_stdout(io.StringIO()):
            conform.render_conform(self.source, recipe, self.root/"preview-identity.mp4", crf=12, start_frame=10, end_frame=16)
        np.testing.assert_array_equal(np.stack(written), self.decoded[10:16])

    def test_invalid_preview_ranges_fail_before_encoding(self):
        recipe = self.write_plan(self.plan(), "preview-ranges")
        with patch.object(conform, "VideoWriter") as writer:
            for start, end in ((-1, 5), (5, 5), (9, 7), (0, 19), (True, 5), (0, 1.5)):
                with self.subTest(start=start, end=end), self.assertRaisesRegex(ValueError, "Preview range"):
                    conform.render_conform(self.source, recipe, self.root/"invalid-range.mp4", start_frame=start, end_frame=end)
            writer.assert_not_called()
        self.assertFalse((self.root/"invalid-range.mp4").exists())


class ProtectedCurveArtifactTests(unittest.TestCase):
    @unittest.skipUnless(CURVES.is_file(), "Source-specific research LUT artifact is unavailable")
    def test_all_source_grade_curves_validate_and_keep_documented_endpoint_strength(self):
        curves = json.loads(CURVES.read_text())
        meta = {"width": 1280, "height": 720, "frame_count": 3347, "fps_fraction": "24000/1001"}
        plan = {"schema_version": 3, "method": "source_conform", "source": meta, "source_sha256": "a"*64,
                "segments": [{"start": 0, "end": 3347, "matrix": np.eye(3).tolist(), "gain": [1,1,1], "bias": [0,0,0]}],
                "grade_curves": curves}
        conform.validate_conform_plan(plan, meta)
        for curve in curves:
            for frame, side in ((curve["frame"]-1, "left_lut"), (curve["frame"], "right_lut")):
                np.testing.assert_allclose(conform.tone_lut_at(frame, curves), curve[side], atol=2e-5)
            self.assertIsNone(conform.tone_lut_at(curve["frame"]-1-curve["support_before"], curves))
            self.assertIsNone(conform.tone_lut_at(curve["frame"]+curve["support_after"], curves))


if __name__ == "__main__":
    unittest.main()
