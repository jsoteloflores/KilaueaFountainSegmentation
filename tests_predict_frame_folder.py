"""
tests_predict_frame_folder.py — Non-GUI tests for predict_frame_folder_gui.py
==============================================================================
Run with:
    python tests_predict_frame_folder.py
or:
    python -m pytest tests_predict_frame_folder.py -v
"""

from __future__ import annotations

import csv
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

# Make sure the module is importable from the same directory as this file.
sys.path.insert(0, str(Path(__file__).parent))

from predict_frame_folder_gui import (
    SUPPORTED_EXTENSIONS,
    collect_input_frames,
    output_mask_path_for_frame,
    output_overlay_path_for_frame,
    output_prob_path_for_frame,
    save_binary_mask,
    save_overlay,
    save_probability_map,
    write_run_summary,
    write_config_json,
    preprocess_image,
    _restore_prob,
    ResizePadMeta,
    DEFAULT_THRESHOLD,
    DEFAULT_INPUT_SIZE,
    SUMMARY_FIELDS,
)


# ---------------------------------------------------------------------------
# Helper: create a small test image on disk
# ---------------------------------------------------------------------------
def _write_frame(folder: Path, name: str, h: int = 60, w: int = 80) -> Path:
    p = folder / name
    img = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
    cv2.imwrite(str(p), img)
    return p


# ===========================================================================
# 1. Mask naming
# ===========================================================================
class TestMaskNaming(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.in_dir  = self.root / "frames"
        self.out_dir = self.root / "masks"
        self.in_dir.mkdir()
        self.out_dir.mkdir()

    def tearDown(self):
        self.tmp.cleanup()

    def test_png_stem_mask_suffix(self):
        frame = self.in_dir / "frame_000123.png"
        result = output_mask_path_for_frame(frame, self.in_dir, self.out_dir)
        self.assertEqual(result.name, "frame_000123_mask.png")

    def test_jpg_stem_mask_suffix(self):
        frame = self.in_dir / "EP12_camB_frame_001250.jpg"
        result = output_mask_path_for_frame(frame, self.in_dir, self.out_dir)
        self.assertEqual(result.name, "EP12_camB_frame_001250_mask.png")

    def test_tif_stem_mask_suffix(self):
        frame = self.in_dir / "videoA_000004.tif"
        result = output_mask_path_for_frame(frame, self.in_dir, self.out_dir)
        self.assertEqual(result.name, "videoA_000004_mask.png")

    def test_output_always_png(self):
        for ext in [".jpg", ".jpeg", ".tif", ".tiff", ".bmp"]:
            frame = self.in_dir / f"img{ext}"
            result = output_mask_path_for_frame(frame, self.in_dir, self.out_dir)
            self.assertEqual(result.suffix, ".png",
                             f"Expected .png output for input {ext}")

    def test_overlay_naming(self):
        frame = self.in_dir / "frame_000123.png"
        ov_dir = self.root / "overlays"
        result = output_overlay_path_for_frame(frame, self.in_dir, ov_dir)
        self.assertEqual(result.name, "frame_000123_overlay.png")

    def test_prob_naming(self):
        frame = self.in_dir / "frame_000123.png"
        pb_dir = self.root / "probs"
        result = output_prob_path_for_frame(frame, self.in_dir, pb_dir)
        self.assertEqual(result.name, "frame_000123_prob.png")


# ===========================================================================
# 2. Existing _mask input skip
# ===========================================================================
class TestMaskInputSkip(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.in_dir = Path(self.tmp.name) / "frames"
        self.in_dir.mkdir()

    def tearDown(self):
        self.tmp.cleanup()

    def _write(self, name: str) -> Path:
        return _write_frame(self.in_dir, name)

    def test_mask_files_skipped(self):
        self._write("frame_000123.png")
        self._write("frame_000123_mask.png")
        frames = collect_input_frames(self.in_dir)
        names = [f.name for f in frames]
        self.assertIn("frame_000123.png", names)
        self.assertNotIn("frame_000123_mask.png", names)

    def test_all_mask_files_skipped(self):
        self._write("a_mask.png")
        self._write("b_mask.jpg")
        frames = collect_input_frames(self.in_dir)
        self.assertEqual(frames, [])

    def test_non_mask_files_included(self):
        self._write("frame_001.png")
        self._write("frame_002.jpg")
        frames = collect_input_frames(self.in_dir)
        self.assertEqual(len(frames), 2)

    def test_hidden_files_skipped(self):
        self._write(".hidden.png")
        self._write("visible.png")
        frames = collect_input_frames(self.in_dir)
        names = [f.name for f in frames]
        self.assertNotIn(".hidden.png", names)
        self.assertIn("visible.png", names)

    def test_unsupported_extension_skipped(self):
        (self.in_dir / "file.mp4").write_text("dummy")
        (self.in_dir / "file.txt").write_text("dummy")
        self._write("frame.png")
        frames = collect_input_frames(self.in_dir)
        self.assertEqual(len(frames), 1)


# ===========================================================================
# 3. Output shape
# ===========================================================================
class TestOutputShape(unittest.TestCase):
    def test_restore_returns_original_shape(self):
        h, w = 480, 640
        # simulate a prob map from the model (same size as input canvas)
        input_size = 512
        img_rgb = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
        chw, meta = preprocess_image(img_rgb, input_size)

        # fake prob canvas (same spatial size as model input)
        prob_canvas = np.random.rand(input_size, input_size).astype(np.float32)
        prob_orig = _restore_prob(prob_canvas, meta)

        self.assertEqual(prob_orig.shape, (h, w),
                         f"Expected ({h},{w}), got {prob_orig.shape}")

    def test_mask_matches_original_shape(self):
        h, w = 360, 720
        img_rgb = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
        input_size = 512
        chw, meta = preprocess_image(img_rgb, input_size)
        prob_canvas = np.ones((input_size, input_size), dtype=np.float32) * 0.9
        prob_orig = _restore_prob(prob_canvas, meta)
        mask = (prob_orig >= 0.5).astype(np.uint8) * 255
        self.assertEqual(mask.shape, (h, w))


# ===========================================================================
# 4. Binary values
# ===========================================================================
class TestBinaryValues(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.out_dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_saved_mask_only_0_and_255(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[20:50, 30:70] = 255
        p = self.out_dir / "frame_mask.png"
        save_binary_mask(mask, p)

        loaded = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        unique_vals = set(np.unique(loaded).tolist())
        self.assertTrue(unique_vals.issubset({0, 255}),
                        f"Unexpected values: {unique_vals}")

    def test_saved_mask_is_single_channel(self):
        mask = np.zeros((50, 60), dtype=np.uint8)
        p = self.out_dir / "single_channel_mask.png"
        save_binary_mask(mask, p)

        loaded = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        # single-channel PNG has 2 dimensions
        self.assertEqual(len(loaded.shape), 2,
                         f"Expected 2D array, got shape {loaded.shape}")

    def test_all_background(self):
        mask = np.zeros((40, 40), dtype=np.uint8)
        p = self.out_dir / "all_bg_mask.png"
        save_binary_mask(mask, p)
        loaded = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        self.assertTrue((loaded == 0).all())

    def test_all_foreground(self):
        mask = np.full((40, 40), 255, dtype=np.uint8)
        p = self.out_dir / "all_fg_mask.png"
        save_binary_mask(mask, p)
        loaded = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        self.assertTrue((loaded == 255).all())


# ===========================================================================
# 5. Run summary creation
# ===========================================================================
class TestRunSummary(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.out_dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def _make_rows(self) -> list:
        return [
            {
                "input_path": "/frames/frame_001.png",
                "output_mask_path": "/masks/frame_001_mask.png",
                "output_overlay_path": "",
                "output_probability_path": "",
                "width": 640,
                "height": 480,
                "threshold": 0.5,
                "positive_pixels": 1234,
                "positive_fraction": "0.004012",
                "status": "ok",
                "error_message": "",
            },
            {
                "input_path": "/frames/bad.png",
                "output_mask_path": "",
                "output_overlay_path": "",
                "output_probability_path": "",
                "width": "",
                "height": "",
                "threshold": 0.5,
                "positive_pixels": "",
                "positive_fraction": "",
                "status": "failed",
                "error_message": "could not read image",
            },
        ]

    def test_csv_exists(self):
        csv_path = self.out_dir / "prediction_run_summary.csv"
        write_run_summary(self._make_rows(), csv_path)
        self.assertTrue(csv_path.exists())

    def test_csv_has_correct_columns(self):
        csv_path = self.out_dir / "prediction_run_summary.csv"
        write_run_summary(self._make_rows(), csv_path)
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            self.assertEqual(reader.fieldnames, SUMMARY_FIELDS)

    def test_csv_row_count(self):
        csv_path = self.out_dir / "prediction_run_summary.csv"
        rows = self._make_rows()
        write_run_summary(rows, csv_path)
        with open(csv_path) as f:
            data = list(csv.DictReader(f))
        self.assertEqual(len(data), len(rows))

    def test_json_exists(self):
        json_path = self.out_dir / "prediction_config.json"
        cfg = {"checkpoint_path": "/ckpt.pth", "threshold": 0.5}
        write_config_json(cfg, json_path)
        self.assertTrue(json_path.exists())

    def test_json_is_valid(self):
        json_path = self.out_dir / "prediction_config.json"
        cfg = {"model_name": "Unet", "threshold": 0.5, "batch_size": 8}
        write_config_json(cfg, json_path)
        with open(json_path) as f:
            loaded = json.load(f)
        self.assertEqual(loaded["model_name"], "Unet")
        self.assertAlmostEqual(loaded["threshold"], 0.5)


# ===========================================================================
# 6. Existing output warning logic
# ===========================================================================
class TestExistingOutputWarning(unittest.TestCase):
    """
    The overwrite-warning logic lives in the GUI callback, but the count
    can be tested by checking the glob used inside the GUI.
    """
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.out_dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def _count_existing_masks(self, folder: Path) -> int:
        return len(list(folder.glob("*_mask.png")))

    def test_empty_folder_no_warning(self):
        n = self._count_existing_masks(self.out_dir)
        self.assertEqual(n, 0)

    def test_folder_with_masks_triggers_warning(self):
        (self.out_dir / "frame_001_mask.png").write_bytes(b"")
        (self.out_dir / "frame_002_mask.png").write_bytes(b"")
        n = self._count_existing_masks(self.out_dir)
        self.assertGreater(n, 0)

    def test_non_mask_files_not_counted(self):
        (self.out_dir / "frame_001.png").write_bytes(b"")
        (self.out_dir / "prediction_config.json").write_text("{}")
        n = self._count_existing_masks(self.out_dir)
        self.assertEqual(n, 0)


# ===========================================================================
# 7. Missing checkpoint error
# ===========================================================================
class TestMissingCheckpoint(unittest.TestCase):
    def test_missing_checkpoint_raises(self):
        """load_checkpoint_and_model should raise for a non-existent path."""
        if not _torch_available():
            self.skipTest("PyTorch not available")
        import torch
        from predict_frame_folder_gui import load_checkpoint_and_model
        with self.assertRaises(Exception):
            load_checkpoint_and_model(
                Path("/does/not/exist/model.pth"),
                torch.device("cpu"),
            )

    def test_cli_missing_checkpoint_returns_nonzero(self):
        from predict_frame_folder_gui import main_cli
        ret = main_cli([
            "--checkpoint", "/no/such/file.pth",
            "--input-folder", "/tmp",
            "--output-folder", "/tmp/out",
        ])
        self.assertNotEqual(ret, 0)


# ===========================================================================
# 8. Missing input folder error
# ===========================================================================
class TestMissingInputFolder(unittest.TestCase):
    def test_collect_empty_folder(self):
        with tempfile.TemporaryDirectory() as tmp:
            frames = collect_input_frames(Path(tmp))
            self.assertEqual(frames, [])

    def test_cli_missing_input_folder_returns_nonzero(self):
        from predict_frame_folder_gui import main_cli
        ret = main_cli([
            "--checkpoint", "/tmp/fake.pth",
            "--input-folder", "/does/not/exist/frames",
            "--output-folder", "/tmp/out",
        ])
        self.assertNotEqual(ret, 0)


# ===========================================================================
# 9. Natural sort
# ===========================================================================
class TestNaturalSort(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.in_dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_natural_order_numbers(self):
        for name in ["frame_10.png", "frame_2.png", "frame_1.png"]:
            _write_frame(self.in_dir, name)
        frames = collect_input_frames(self.in_dir)
        names = [f.name for f in frames]
        self.assertEqual(names, ["frame_1.png", "frame_2.png", "frame_10.png"])


# ===========================================================================
# 10. Preprocessing consistency
# ===========================================================================
class TestPreprocessing(unittest.TestCase):
    def test_chw_shape(self):
        h, w = 100, 150
        img = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
        chw, meta = preprocess_image(img, 256)
        self.assertEqual(chw.shape, (3, 256, 256))

    def test_meta_stores_orig_dims(self):
        h, w = 123, 456
        img = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
        _, meta = preprocess_image(img, 512)
        self.assertEqual(meta.orig_h, h)
        self.assertEqual(meta.orig_w, w)

    def test_roundtrip_shape(self):
        """After restore, shape matches original."""
        for (h, w) in [(100, 200), (480, 640), (1080, 1920)]:
            img = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
            sz = 256
            _, meta = preprocess_image(img, sz)
            fake_prob = np.random.rand(sz, sz).astype(np.float32)
            restored = _restore_prob(fake_prob, meta)
            self.assertEqual(restored.shape, (h, w),
                             f"Failed for ({h},{w})")


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
def _torch_available() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Overlay / probability save tests
# ---------------------------------------------------------------------------
class TestOptionalSaves(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.out_dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_overlay_is_written(self):
        h, w = 60, 80
        img = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[10:30, 20:60] = 255
        p = self.out_dir / "test_overlay.png"
        save_overlay(img, mask, p)
        self.assertTrue(p.exists())
        loaded = cv2.imread(str(p))
        self.assertEqual(loaded.shape[:2], (h, w))

    def test_prob_map_saved_as_grayscale(self):
        prob = np.random.rand(50, 70).astype(np.float32)
        p = self.out_dir / "test_prob.png"
        save_probability_map(prob, p)
        self.assertTrue(p.exists())
        loaded = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        self.assertEqual(loaded.shape, (50, 70))
        self.assertTrue(loaded.min() >= 0 and loaded.max() <= 255)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    unittest.main(verbosity=2)
