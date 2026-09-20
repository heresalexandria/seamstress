"""Renderer orchestration and model-download integrity without neural inference."""
from __future__ import annotations

import copy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch
import zipfile

import numpy as np

from seamstress import bridge, model_setup


class BridgeOrchestrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "original.mp4"
        self.source.write_bytes(b"synthetic source identity")
        self.output = self.root / "output.mp4"
        self.metadata = {"width": 64, "height": 48, "fps": 24.,
                         "fps_fraction": "24/1", "frame_count": 40}
        self.frames = [np.full((48, 64, 3), i, np.uint8) for i in range(40)]
        self.plan = {
            "schema_version": 2, "method": "rife_bridge", "source": self.metadata.copy(),
            "source_sha256": bridge.fingerprint(self.source),
            "seams": [{"frame": 9, "time": 9/24, "bridge_start": 6, "bridge_end": 12,
                       "camera_control": False},
                      {"frame": 23, "time": 23/24, "bridge_start": 20, "bridge_end": 26,
                       "camera_control": False}],
        }

    def render(self, *, camera=False, malformed=False):
        for seam in self.plan["seams"]:
            seam["camera_control"] = camera
        plan_path = self.root / "plan.json"
        plan_path.write_text(json.dumps(self.plan))
        written, handles = [], []

        class Writer:
            def __init__(self, *args, **kwargs):
                pass
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass
            def write(self, frame):
                written.append(frame.copy())

        class Model:
            device = "cpu"
            def __init__(self, *args, **kwargs):
                pass
            def synthesize(self, left, right, progress):
                result = [left.copy()] + [np.full_like(left, 220) for _ in progress[1:-1]] + [right.copy()]
                if malformed:
                    result.insert(-1, np.full_like(left, 221))
                return result

        def camera_synthesis(model, left, right, before, after, progress):
            handles.append(([int(x[0, 0, 0]) for x in before],
                            [int(x[0, 0, 0]) for x in after],
                            int(left[0, 0, 0]), int(right[0, 0, 0])))
            return model.synthesize(left, right, progress), {"test_camera": True}

        def fake_mux(video, source, output):
            Path(output).write_bytes(b"finished encoded container")

        with patch.object(bridge, "probe", return_value=self.metadata), \
             patch.object(bridge, "read_frames", side_effect=lambda p, start, count: np.stack(self.frames[start:start+count])), \
             patch.object(bridge, "iter_frames", side_effect=lambda p: iter(self.frames)), \
             patch.object(bridge, "RifeModel", Model), \
             patch.object(bridge, "VideoWriter", Writer), \
             patch.object(bridge, "mux_audio", side_effect=fake_mux), \
             patch("seamstress.camera_bridge.synthesize_camera_bridge", side_effect=camera_synthesis):
            report = bridge.render_bridges(self.source, plan_path, self.output, self.root / "unused.pkl")
        return written, handles, report

    def test_only_bridge_interiors_are_replaced_and_count_is_preserved(self):
        written, _, report = self.render()
        changed = {7, 8, 9, 10, 11, 21, 22, 23, 24, 25}
        self.assertEqual(len(written), len(self.frames))
        for index, frame in enumerate(written):
            expected = np.full_like(frame, 220) if index in changed else self.frames[index]
            np.testing.assert_array_equal(frame, expected)
        self.assertEqual(self.source.read_bytes(), b"synthetic source identity")
        self.assertEqual(self.output.read_bytes(), b"finished encoded container")
        sidecar = json.loads(self.output.with_suffix(".repair.json").read_text())
        self.assertEqual([s["synthesized_frames"] for s in sidecar["seams"]], [5, 5])
        self.assertEqual(report["frame_count"], 40)

    def test_camera_receives_chronological_handles_excluding_endpoints(self):
        written, handles, _ = self.render(camera=True)
        self.assertEqual(handles, [([2, 3, 4, 5], [13, 14, 15, 16], 6, 12),
                                   ([16, 17, 18, 19], [27, 28, 29, 30], 20, 26)])
        self.assertEqual(len(written), 40)
        for index in (6, 12, 20, 26):
            np.testing.assert_array_equal(written[index], self.frames[index])

    def test_wrong_synthesis_count_is_rejected_before_output_publication(self):
        with self.assertRaises(RuntimeError):
            self.render(malformed=True)
        self.assertFalse(self.output.exists())

    def test_existing_sidecar_is_preserved_before_model_load(self):
        sidecar = self.output.with_suffix(".repair.json")
        sidecar.write_text("existing report")
        with patch.object(bridge, "probe", return_value=self.metadata), patch.object(bridge, "RifeModel") as model:
            with self.assertRaisesRegex(ValueError, "new output"):
                bridge.render_bridges(self.source, self.root / "missing.json", self.output, self.root / "unused.pkl")
            model.assert_not_called()
        self.assertEqual(sidecar.read_text(), "existing report")

    def test_changed_source_fingerprint_is_rejected_before_model_load(self):
        self.plan["source_sha256"] = "0" * 64
        path = self.root / "plan.json"
        path.write_text(json.dumps(self.plan))
        with patch.object(bridge, "probe", return_value=self.metadata), patch.object(bridge, "RifeModel") as model:
            with self.assertRaisesRegex(ValueError, "fingerprint"):
                bridge.render_bridges(self.source, path, self.output, self.root / "unused.pkl")
            model.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_camera_windows_require_source_handles_on_both_sides(self):
        for start, end, frame in ((0, 12, 9), (20, 39, 23)):
            with self.subTest(start=start, end=end):
                plan = copy.deepcopy(self.plan)
                plan["seams"] = [{"frame": frame, "time": frame/24,
                                  "bridge_start": start, "bridge_end": end, "camera_control": True}]
                with self.assertRaises(ValueError):
                    bridge.validate_bridge_plan(plan, self.metadata)

    def test_schema_rejects_bad_indices_timing_and_camera_flags(self):
        for field, value in (("bridge_start", True), ("bridge_end", 12.5),
                             ("time", float("nan")), ("camera_control", "false"),
                             ("start_slope", -1), ("end_slope", float("inf"))):
            with self.subTest(field=field, value=value):
                plan = copy.deepcopy(self.plan)
                plan["seams"][0][field] = value
                with self.assertRaises(ValueError):
                    bridge.validate_bridge_plan(plan, self.metadata)


class ModelSetupTests(unittest.TestCase):
    def archive(self, content):
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr("train_log/flownet.pkl", content)
        return buffer.getvalue()

    def test_verified_model_download_and_existing_reuse(self):
        content = b"test checkpoint bytes; never deserialized"
        digest = hashlib.sha256(content).hexdigest()
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "weights" / "flownet.pkl"
            with patch.object(model_setup, "WEIGHTS_SHA256", digest), \
                 patch.object(model_setup.urllib.request, "urlopen", return_value=io.BytesIO(self.archive(content))) as download:
                first = model_setup.setup_model(output)
                second = model_setup.setup_model(output)
            self.assertEqual(output.read_bytes(), content)
            self.assertEqual(first["status"], "downloaded and verified")
            self.assertEqual(second["status"], "already verified")
            self.assertEqual(download.call_count, 1)

    def test_wrong_download_checksum_never_publishes_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "weights" / "flownet.pkl"
            with patch.object(model_setup, "WEIGHTS_SHA256", "0" * 64), \
                 patch.object(model_setup.urllib.request, "urlopen", return_value=io.BytesIO(self.archive(b"wrong checkpoint"))):
                with self.assertRaisesRegex(ValueError, "checksum"):
                    model_setup.setup_model(output)
            self.assertFalse(output.exists())

    def test_existing_wrong_checkpoint_is_never_replaced(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "flownet.pkl"
            output.write_bytes(b"keep existing")
            with patch.object(model_setup.urllib.request, "urlopen") as download:
                with self.assertRaisesRegex(ValueError, "checksum"):
                    model_setup.setup_model(output)
                download.assert_not_called()
            self.assertEqual(output.read_bytes(), b"keep existing")


class RifeInputGuardTests(unittest.TestCase):
    def test_invalid_progress_is_rejected_before_neural_inference(self):
        model = object.__new__(bridge.RifeModel)
        model.torch = MagicMock()
        model.device = "cpu"
        model.model = MagicMock()
        image = np.zeros((64, 64, 3), np.uint8)
        for progress in ([], [0], [0, .5], [.1, .5, 1], [0, float("nan"), 1], [0, .7, .6, 1], [0, 0, 1]):
            with self.subTest(progress=progress):
                with self.assertRaises(ValueError):
                    model.synthesize(image, image, progress)
        model.model.assert_not_called()


if __name__ == "__main__":
    unittest.main()
