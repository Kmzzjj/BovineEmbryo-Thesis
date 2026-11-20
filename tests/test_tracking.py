import csv
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock, patch

import cv2
import numpy as np

from embryo_imaging import paths, tracking


class TrackingTest(unittest.TestCase):
    def test_identity_tools_resolve_thesis_image_names(self):
        filename = "demo-01_P2_FP3_70.jpg"
        key = paths.parse_from_filename(filename)
        self.assertEqual(key, ("demo-01", 2, 3, 70))
        self.assertEqual(
            paths.big_image_path(Path("images"), *key),
            Path("images/MTL-0034-demo-01") / filename,
        )

    def test_groups_and_sorts_frames(self):
        with TemporaryDirectory() as temp:
            run = Path(temp)
            for name in (
                "run_P4_FP3_10.jpg",
                "run_P4_FP3_2_I1.JPEG",
                "run_P5_FP3_1.jpg",
                "run_P4_FP2_1.jpg",
                "notes.txt",
            ):
                (run / name).touch()

            groups = tracking.group_frames(run, "FP3")
            self.assertEqual(list(groups), ["P4_FP3", "P5_FP3"])
            self.assertEqual(
                [frame.name for frame in groups["P4_FP3"]],
                ["run_P4_FP3_2_I1.JPEG", "run_P4_FP3_10.jpg"],
            )

    def test_incomplete_sequence_can_be_retried(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            frames = [root / f"run_P1_FP3_{time}.jpg" for time in (1, 2)]
            for frame in frames:
                self.assertTrue(cv2.imwrite(str(frame), np.zeros((24, 24, 3), dtype=np.uint8)))

            output = root / "tracks"
            result = SimpleNamespace(boxes=None)
            torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
            with patch.dict(sys.modules, torch=torch):
                model = SimpleNamespace(track=lambda **kwargs: iter([result]))
                with self.assertRaisesRegex(RuntimeError, "1 results for 2 frames"):
                    tracking.track_sequence(
                        model, frames, output, "cpu", save_video=False, overwrite=False
                    )
                self.assertFalse((output / "tracks.csv").exists())

                model.track = lambda **kwargs: iter([result, result])
                tracking.track_sequence(
                    model, frames, output, "cpu", save_video=False, overwrite=False
                )

            with (output / "tracks.csv").open(newline="", encoding="utf-8") as stream:
                self.assertEqual(list(csv.reader(stream)), [list(tracking.TRACK_COLUMNS)])
            self.assertFalse(list(output.glob(".tracking-*")))


    def test_unassigned_detections_do_not_become_one_shared_identity(self):
        def tensor(values):
            array = np.asarray(values)
            result = SimpleNamespace(cpu=lambda: result, numpy=lambda: array, int=lambda: result)
            return result

        class Boxes:
            cls = tensor([0, 0])
            conf = tensor([0.95, 0.90])
            xyxy = tensor([[0, 0, 20, 20], [20, 0, 40, 20]])
            id = None

            def __len__(self):
                return 2

        boxes = Boxes()
        local_ids = {}
        self.assertEqual(tracking.track_rows(boxes, local_ids), [])
        boxes.id = tensor([-1, 8])
        rows = tracking.track_rows(boxes, local_ids)
        self.assertEqual(len(rows), 1)
        self.assertEqual(local_ids, {8: 0})
        self.assertEqual(rows[0][0], 0)

    def test_each_well_starts_with_fresh_tracker_state(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            frame = root / "run_P1_FP3_1.jpg"
            self.assertTrue(cv2.imwrite(str(frame), np.zeros((24, 24, 3), dtype=np.uint8)))
            tracker = SimpleNamespace(reset=Mock())
            model = SimpleNamespace(predictor=SimpleNamespace(trackers=[tracker]))

            def track(**kwargs):
                self.assertEqual(tracker.reset.call_count, model.calls + 1)
                model.calls += 1
                return iter([SimpleNamespace(boxes=None)])

            model.calls = 0
            model.track = track
            torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
            with patch.dict(sys.modules, torch=torch):
                for well in ("P1_FP3", "P2_FP3"):
                    tracking.track_sequence(model, [frame], root / well, "cpu", save_video=False, overwrite=False)
            self.assertEqual(tracker.reset.call_count, 2)

    def test_failed_overwrite_preserves_completed_csv(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            frames = [root / f"run_P1_FP3_{time}.jpg" for time in (1, 2)]
            for frame in frames:
                self.assertTrue(cv2.imwrite(str(frame), np.zeros((24, 24, 3), dtype=np.uint8)))
            output = root / "tracks"
            output.mkdir()
            previous = b"previous completed output\n"
            (output / "tracks.csv").write_bytes(previous)
            model = SimpleNamespace(track=lambda **kwargs: iter([SimpleNamespace(boxes=None)]))
            torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
            with patch.dict(sys.modules, torch=torch):
                with self.assertRaisesRegex(RuntimeError, "1 results for 2 frames"):
                    tracking.track_sequence(model, frames, output, "cpu", save_video=False, overwrite=True)
            self.assertEqual((output / "tracks.csv").read_bytes(), previous)
