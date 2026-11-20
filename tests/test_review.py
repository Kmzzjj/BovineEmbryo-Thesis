import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
from PyQt5.QtCore import QLockFile

from embryo_imaging.review import csv_version, find_center_img_path, save_label_csv


class ReviewTest(unittest.TestCase):
    def test_resolves_paths_in_manifest_and_crop_directory(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            crop_dir = root / "centre crops"
            crop_dir.mkdir()
            image = crop_dir / "example_center_320.jpg"
            image.touch()

            self.assertEqual(
                find_center_img_path(
                    {"seq_id": "other", "center_img_path": str(image.relative_to(root))},
                    root,
                    crop_dir,
                ),
                str(image),
            )
            self.assertEqual(find_center_img_path({"seq_id": "example"}, root, crop_dir), str(image))

    def test_backs_up_previous_csv_and_rejects_stale_or_locked_save(self):
        with TemporaryDirectory() as temp:
            path = Path(temp) / "labels.csv"
            previous = b"seq_id,label\nexample,A\n"
            path.write_bytes(previous)
            version = csv_version(path)
            labels = pd.DataFrame({"seq_id": ["example"], "label": ["B"]})

            lock = QLockFile(str(path.with_suffix(".csv.lock")))
            self.assertTrue(lock.tryLock(0))
            try:
                with self.assertRaises(OSError):
                    save_label_csv(labels, path, version)
            finally:
                lock.unlock()
            self.assertEqual(path.read_bytes(), previous)

            backup = save_label_csv(labels, path, version)
            self.assertEqual(backup.read_bytes(), previous)
            self.assertEqual(pd.read_csv(path).loc[0, "label"], "B")
            with self.assertRaises(ValueError):
                save_label_csv(labels, path, version)
            self.assertEqual(len(list(path.parent.glob("labels.bak_*.csv"))), 1)
