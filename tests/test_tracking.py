import csv
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

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
