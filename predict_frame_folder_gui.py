"""
predict_frame_folder_gui.py — Lava Fountain Frame-Folder Mask Predictor
========================================================================
GUI utility for KilaueaFountainSegmentation that takes a trained model
checkpoint and a folder of extracted frame images, runs 2D model inference
on every frame, and writes predicted binary masks in the exact filename
format expected by FountainLabeller:

    input:   <stem>.<ext>
    output:  <stem>_mask.png

Intended workflow:
    frames folder
        → this tool (model inference)
        → output mask folder  (<stem>_mask.png for each frame)
        → copy into FountainLabeller dataset
        → review / correct masks in FountainLabeller

CLI usage:
    python predict_frame_folder_gui.py \\
        --checkpoint /path/best_model.pth \\
        --input-folder /path/frames \\
        --output-folder /path/predicted_masks \\
        --threshold 0.5 \\
        --batch-size 8 \\
        --save-overlays

GUI usage:
    python predict_frame_folder_gui.py
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import threading
import traceback
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Optional heavy imports — only needed during inference, not for pure tests
# ---------------------------------------------------------------------------
try:
    import torch
    import torch.nn as nn
    import segmentation_models_pytorch as smp
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

# ---------------------------------------------------------------------------
# Constants / defaults matching lava_fountain_segmentation_colab.py
# ---------------------------------------------------------------------------
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

DEFAULT_MODEL_NAME   = "Unet"
DEFAULT_ENCODER_NAME = "efficientnet-b3"
DEFAULT_INPUT_SIZE   = 1536
DEFAULT_THRESHOLD    = 0.50
DEFAULT_BATCH_SIZE   = 8

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

SMP_MODELS: Dict[str, Any] = {}
if _TORCH_AVAILABLE:
    SMP_MODELS = {
        "unet":         smp.Unet,
        "unetplusplus": smp.UnetPlusPlus,
        "deeplabv3plus":smp.DeepLabV3Plus,
        "fpn":          smp.FPN,
        "manet":        smp.MAnet,
    }

TARGET_DEFINITION = "active_rising_lava_fountain"


# ---------------------------------------------------------------------------
# Natural sort helper
# ---------------------------------------------------------------------------
def _natural_key(path: Path) -> list:
    return [int(c) if c.isdigit() else c.lower()
            for c in re.split(r"(\d+)", path.name)]


# ---------------------------------------------------------------------------
# Frame collection
# ---------------------------------------------------------------------------
def collect_input_frames(
    input_folder: Path,
    recursive: bool = False,
) -> List[Path]:
    """Return sorted list of supported image paths, skipping *_mask files."""
    pattern = "**/*" if recursive else "*"
    files: List[Path] = []
    for p in input_folder.glob(pattern):
        if p.is_dir():
            continue
        if p.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        if p.name.startswith("."):
            continue
        if p.stem.endswith("_mask"):
            continue
        files.append(p)
    return sorted(files, key=_natural_key)


# ---------------------------------------------------------------------------
# Output-path helper — fixed naming convention
# ---------------------------------------------------------------------------
def output_mask_path_for_frame(
    frame_path: Path,
    input_root: Path,
    output_root: Path,
) -> Path:
    """Return <output_root>/<relative_path>/<stem>_mask.png."""
    rel = frame_path.relative_to(input_root)
    return output_root / rel.parent / f"{rel.stem}_mask.png"


def output_overlay_path_for_frame(frame_path: Path, input_root: Path, overlay_root: Path) -> Path:
    rel = frame_path.relative_to(input_root)
    return overlay_root / rel.parent / f"{rel.stem}_overlay.png"


def output_prob_path_for_frame(frame_path: Path, input_root: Path, prob_root: Path) -> Path:
    rel = frame_path.relative_to(input_root)
    return prob_root / rel.parent / f"{rel.stem}_prob.png"


# ---------------------------------------------------------------------------
# Preprocessing (mirrors lava_fountain_segmentation_colab.py)
# ---------------------------------------------------------------------------
@dataclass
class ResizePadMeta:
    orig_h: int
    orig_w: int
    pad_top: int
    pad_left: int
    resized_h: int
    resized_w: int


def _resize_and_pad(image_rgb: np.ndarray, target_size: int) -> Tuple[np.ndarray, ResizePadMeta]:
    """Aspect-preserving resize + center-pad to square canvas."""
    orig_h, orig_w = image_rgb.shape[:2]
    scale = min(target_size / orig_w, target_size / orig_h)
    new_w = max(1, int(round(orig_w * scale)))
    new_h = max(1, int(round(orig_h * scale)))

    img_resized = cv2.resize(image_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    pad_top    = (target_size - new_h) // 2
    pad_bottom = target_size - new_h - pad_top
    pad_left   = (target_size - new_w) // 2
    pad_right  = target_size - new_w - pad_left

    canvas = cv2.copyMakeBorder(
        img_resized, pad_top, pad_bottom, pad_left, pad_right,
        borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0))

    meta = ResizePadMeta(
        orig_h=orig_h, orig_w=orig_w,
        pad_top=pad_top, pad_left=pad_left,
        resized_h=new_h, resized_w=new_w,
    )
    return canvas, meta


def _normalize_image(img_uint8: np.ndarray) -> np.ndarray:
    """[0,255] uint8 HWC → float32 CHW with ImageNet normalisation."""
    img = img_uint8.astype(np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    return np.ascontiguousarray(img.transpose(2, 0, 1))


def _restore_prob(canvas_2d: np.ndarray, meta: ResizePadMeta) -> np.ndarray:
    """Remove padding and resize float32 prob map back to original frame size."""
    cropped = canvas_2d[
        meta.pad_top : meta.pad_top + meta.resized_h,
        meta.pad_left : meta.pad_left + meta.resized_w,
    ]
    restored = cv2.resize(
        cropped, (meta.orig_w, meta.orig_h), interpolation=cv2.INTER_LINEAR
    )
    return restored


def preprocess_image(image_rgb: np.ndarray, input_size: int) -> Tuple[np.ndarray, ResizePadMeta]:
    """Return (CHW float32 tensor-ready array, pad metadata)."""
    canvas, meta = _resize_and_pad(image_rgb, input_size)
    return _normalize_image(canvas), meta


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_checkpoint_and_model(
    checkpoint_path: Path,
    device: "torch.device",
    log_fn=print,
) -> Tuple["nn.Module", Dict[str, Any], List[str]]:
    """
    Load checkpoint and reconstruct model.

    Returns (model, info_dict, warnings_list).
    info_dict keys: input_size, threshold, model_name, encoder_name,
                    normalization, target_definition, in_channels.
    """
    if not _TORCH_AVAILABLE:
        raise RuntimeError("PyTorch / segmentation-models-pytorch not installed.")

    ckpt = torch.load(str(checkpoint_path), map_location=device)

    warnings: List[str] = []
    cfg = ckpt.get("config", {})

    # --- pull metadata with fallbacks ---
    model_name   = ckpt.get("model_name",   cfg.get("model_name",   DEFAULT_MODEL_NAME))
    encoder_name = ckpt.get("encoder_name", cfg.get("encoder_name", DEFAULT_ENCODER_NAME))
    in_channels  = cfg.get("in_channels",   3)
    classes      = cfg.get("classes",       1)
    input_size   = ckpt.get("input_size",   cfg.get("input_size",   DEFAULT_INPUT_SIZE))
    threshold    = ckpt.get("threshold",    cfg.get("threshold",    DEFAULT_THRESHOLD))
    normalization = ckpt.get("normalization", "imagenet")
    target_def   = ckpt.get("target_definition", None)

    has_full_meta = all(k in ckpt for k in ("model_name", "encoder_name", "input_size", "threshold"))
    if not has_full_meta:
        warnings.append(
            "Warning: checkpoint does not contain full metadata. "
            "Using fallback model settings."
        )

    if target_def is not None and target_def != TARGET_DEFINITION:
        warnings.append(
            f"Warning: checkpoint target_definition is '{target_def}', "
            f"not '{TARGET_DEFINITION}'. "
            "These masks may not match the current FountainLabeller class definition."
        )

    model_cls = SMP_MODELS.get(model_name.lower())
    if model_cls is None:
        raise ValueError(
            f"Unsupported model_name '{model_name}'. "
            f"Options: {sorted(SMP_MODELS.keys())}"
        )

    model = model_cls(
        encoder_name=encoder_name,
        encoder_weights=None,
        in_channels=in_channels,
        classes=classes,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    info: Dict[str, Any] = {
        "input_size":        input_size,
        "threshold":         threshold,
        "model_name":        model_name,
        "encoder_name":      encoder_name,
        "normalization":     normalization,
        "target_definition": target_def or TARGET_DEFINITION,
        "in_channels":       in_channels,
    }
    return model, info, warnings


def auto_detect_device() -> "torch.device":
    if not _TORCH_AVAILABLE:
        raise RuntimeError("PyTorch not installed.")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Mask / overlay / probability saving
# ---------------------------------------------------------------------------
def save_binary_mask(mask_hw: np.ndarray, output_path: Path) -> None:
    """Save uint8 {0,255} single-channel mask as PNG."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), mask_hw)


def save_overlay(image_rgb: np.ndarray, mask_hw: np.ndarray, output_path: Path) -> None:
    """Blend mask over image for human inspection."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    overlay = image_rgb.copy()
    where = mask_hw > 127
    # semi-transparent orange tint on predicted regions
    overlay[where, 0] = np.clip(overlay[where, 0].astype(np.int32) + 80, 0, 255)
    overlay[where, 1] = np.clip(overlay[where, 1].astype(np.int32) - 20, 0, 255)
    overlay[where, 2] = np.clip(overlay[where, 2].astype(np.int32) - 20, 0, 255)
    blended = cv2.addWeighted(image_rgb, 0.55, overlay, 0.45, 0)
    cv2.imwrite(str(output_path), cv2.cvtColor(blended, cv2.COLOR_RGB2BGR))


def save_probability_map(prob_hw: np.ndarray, output_path: Path) -> None:
    """Save probability [0,1] as 8-bit grayscale PNG."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    prob8 = np.clip(prob_hw * 255.0, 0, 255).astype(np.uint8)
    cv2.imwrite(str(output_path), prob8)


# ---------------------------------------------------------------------------
# Run-summary helpers
# ---------------------------------------------------------------------------
SUMMARY_FIELDS = [
    "input_path", "output_mask_path", "output_overlay_path",
    "output_probability_path", "width", "height", "threshold",
    "positive_pixels", "positive_fraction", "status", "error_message",
]


def write_run_summary(rows: List[Dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_config_json(config: Dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(config, f, indent=2, default=str)


# ---------------------------------------------------------------------------
# Core prediction job (no Tkinter dependency)
# ---------------------------------------------------------------------------
@dataclass
class PredictionConfig:
    checkpoint_path: Path
    input_folder: Path
    output_mask_folder: Path
    output_overlay_folder: Optional[Path] = None
    output_probability_folder: Optional[Path] = None
    save_overlays: bool = False
    save_probabilities: bool = False
    threshold: Optional[float] = None        # None → use checkpoint value
    batch_size: int = DEFAULT_BATCH_SIZE
    device_str: str = "auto"
    recursive: bool = False


def run_prediction_job(
    config: PredictionConfig,
    progress_fn=None,   # callable(str) for log messages
    cancel_fn=None,     # callable() → bool; returns True if cancelled
) -> Dict[str, Any]:
    """
    Run inference over all frames and write outputs.

    progress_fn: receives status strings (can be None).
    cancel_fn:   returns True if the user has cancelled (can be None).

    Returns a summary dict with keys: succeeded, failed, cancelled, rows.
    """
    def log(msg: str) -> None:
        if progress_fn:
            progress_fn(msg)

    def is_cancelled() -> bool:
        return cancel_fn() if cancel_fn else False

    if not _TORCH_AVAILABLE:
        raise RuntimeError(
            "PyTorch and segmentation-models-pytorch are required for inference."
        )

    # --- device ---
    if config.device_str == "auto":
        device = auto_detect_device()
    else:
        device = torch.device(config.device_str)
    log(f"Detected device: {device}")

    # --- load model ---
    log(f"Loading checkpoint: {config.checkpoint_path.name}")
    model, info, warns = load_checkpoint_and_model(config.checkpoint_path, device, log)
    for w in warns:
        log(w)

    input_size  = info["input_size"]
    threshold   = config.threshold if config.threshold is not None else info["threshold"]
    log(f"Using threshold: {threshold:.4f}")
    log(f"Model: {info['model_name']} / {info['encoder_name']} / input_size={input_size}")

    # --- collect frames ---
    frames = collect_input_frames(config.input_folder, recursive=config.recursive)
    if not frames:
        raise ValueError(
            f"No supported images found in {config.input_folder}. "
            "Check the path and supported extensions: "
            + ", ".join(sorted(SUPPORTED_EXTENSIONS))
        )
    log(f"Found {len(frames)} input frames.")

    # --- overwrite check (logged by caller / GUI) ---
    config.output_mask_folder.mkdir(parents=True, exist_ok=True)

    log(f"Writing masks to: {config.output_mask_folder}")

    run_ts = datetime.now(timezone.utc).isoformat()
    rows: List[Dict] = []
    succeeded = 0
    failed    = 0

    # --- batch loop ---
    batch_imgs: List[np.ndarray] = []   # CHW float32
    batch_metas: List[ResizePadMeta]    = []
    batch_paths: List[Path]             = []
    batch_rgb: List[np.ndarray]         = []  # HxWx3 uint8 for overlay

    def flush_batch():
        nonlocal succeeded, failed
        if not batch_imgs:
            return

        # stack → tensor
        tensor = torch.from_numpy(np.stack(batch_imgs, axis=0)).float().to(device)
        with torch.no_grad():
            logits = model(tensor)              # [B, 1, H, W]
            probs_t = torch.sigmoid(logits).squeeze(1).cpu().numpy()  # [B, H, W]

        for i, (frame_path, meta, prob_canvas) in enumerate(
            zip(batch_paths, batch_metas, probs_t)
        ):
            orig_rgb = batch_rgb[i]
            try:
                prob_orig = _restore_prob(prob_canvas, meta)  # HxW float32
                mask_orig = (prob_orig >= threshold).astype(np.uint8) * 255

                assert mask_orig.shape == (meta.orig_h, meta.orig_w), \
                    f"Shape mismatch: {mask_orig.shape} vs ({meta.orig_h},{meta.orig_w})"

                mask_path = output_mask_path_for_frame(
                    frame_path, config.input_folder, config.output_mask_folder
                )
                save_binary_mask(mask_orig, mask_path)

                overlay_path_str = ""
                prob_path_str    = ""

                if config.save_overlays and config.output_overlay_folder:
                    ov_path = output_overlay_path_for_frame(
                        frame_path, config.input_folder, config.output_overlay_folder
                    )
                    save_overlay(orig_rgb, mask_orig, ov_path)
                    overlay_path_str = str(ov_path)

                if config.save_probabilities and config.output_probability_folder:
                    pb_path = output_prob_path_for_frame(
                        frame_path, config.input_folder, config.output_probability_folder
                    )
                    save_probability_map(prob_orig, pb_path)
                    prob_path_str = str(pb_path)

                pos_px = int(mask_orig.astype(bool).sum())
                pos_frac = pos_px / max(meta.orig_h * meta.orig_w, 1)

                rows.append({
                    "input_path":             str(frame_path),
                    "output_mask_path":       str(mask_path),
                    "output_overlay_path":    overlay_path_str,
                    "output_probability_path": prob_path_str,
                    "width":                  meta.orig_w,
                    "height":                 meta.orig_h,
                    "threshold":              threshold,
                    "positive_pixels":        pos_px,
                    "positive_fraction":      f"{pos_frac:.6f}",
                    "status":                 "ok",
                    "error_message":          "",
                })
                succeeded += 1

            except Exception as exc:
                rows.append({
                    "input_path":             str(frame_path),
                    "output_mask_path":       "",
                    "output_overlay_path":    "",
                    "output_probability_path": "",
                    "width":                  meta.orig_w,
                    "height":                 meta.orig_h,
                    "threshold":              threshold,
                    "positive_pixels":        "",
                    "positive_fraction":      "",
                    "status":                 "failed",
                    "error_message":          str(exc),
                })
                failed += 1
                log(f"  Failed post-processing for {frame_path.name}: {exc}")

        batch_imgs.clear()
        batch_metas.clear()
        batch_paths.clear()
        batch_rgb.clear()

    total = len(frames)
    for idx, frame_path in enumerate(frames):
        if is_cancelled():
            flush_batch()
            log("Prediction cancelled by user.")
            log("Partial outputs remain in output folder.")
            break

        # read image
        try:
            img_bgr = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
            if img_bgr is None:
                raise IOError(f"Could not read image: {frame_path}")
            image_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        except Exception as exc:
            rows.append({
                "input_path":             str(frame_path),
                "output_mask_path":       "",
                "output_overlay_path":    "",
                "output_probability_path": "",
                "width":                  "",
                "height":                 "",
                "threshold":              threshold,
                "positive_pixels":        "",
                "positive_fraction":      "",
                "status":                 "failed",
                "error_message":          f"could not read image: {exc}",
            })
            failed += 1
            log(f"  Skipping unreadable frame: {frame_path.name}")
            continue

        chw, meta = preprocess_image(image_rgb, input_size)
        batch_imgs.append(chw)
        batch_metas.append(meta)
        batch_paths.append(frame_path)
        batch_rgb.append(image_rgb)

        if len(batch_imgs) >= config.batch_size:
            flush_batch()

        if (idx + 1) % 100 == 0 or (idx + 1) == total:
            log(f"Processed {idx + 1} / {total} frames...")

    flush_batch()

    # --- write summaries ---
    summary_csv  = config.output_mask_folder / "prediction_run_summary.csv"
    summary_json = config.output_mask_folder / "prediction_config.json"

    write_run_summary(rows, summary_csv)

    json_cfg: Dict[str, Any] = {
        "checkpoint_path":          str(config.checkpoint_path),
        "input_folder":             str(config.input_folder),
        "output_mask_folder":       str(config.output_mask_folder),
        "output_overlay_folder":    str(config.output_overlay_folder) if config.output_overlay_folder else "",
        "output_probability_folder": str(config.output_probability_folder) if config.output_probability_folder else "",
        "save_overlays":            config.save_overlays,
        "save_probabilities":       config.save_probabilities,
        "threshold":                threshold,
        "batch_size":               config.batch_size,
        "device":                   str(device),
        "target_definition":        info["target_definition"],
        "model_name":               info["model_name"],
        "encoder_name":             info["encoder_name"],
        "input_size":               input_size,
        "normalization":            info["normalization"],
        "created_at":               run_ts,
    }
    write_config_json(json_cfg, summary_json)

    cancelled = is_cancelled()
    msg = (
        f"Finished. {succeeded} succeeded, {failed} failed."
        if not cancelled
        else f"Cancelled. {succeeded} succeeded, {failed} failed before cancel."
    )
    if failed:
        msg += f" See prediction_run_summary.csv for details."
    log(msg)

    return {"succeeded": succeeded, "failed": failed, "cancelled": cancelled, "rows": rows}


# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------
def launch_gui() -> None:
    """Launch the Tkinter GUI for the frame-folder predictor."""
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox

    root = tk.Tk()
    root.title("Lava Fountain Frame-Folder Predictor")
    root.resizable(True, True)

    # ---- variables ----
    var_checkpoint   = tk.StringVar()
    var_input_folder = tk.StringVar()
    var_output_mask  = tk.StringVar()
    var_overlay_folder = tk.StringVar()
    var_prob_folder  = tk.StringVar()

    var_save_overlays = tk.BooleanVar(value=False)
    var_save_probs    = tk.BooleanVar(value=False)
    var_recursive     = tk.BooleanVar(value=False)

    var_threshold  = tk.StringVar(value=f"{DEFAULT_THRESHOLD:.2f}")
    var_batch_size = tk.StringVar(value=str(DEFAULT_BATCH_SIZE))
    var_device     = tk.StringVar(value="auto")

    _cancel_flag = [False]
    _running     = [False]

    # ---- helpers ----
    def browse_file(var: tk.StringVar, title: str, filetypes=None):
        path = filedialog.askopenfilename(title=title, filetypes=filetypes or [("All", "*.*")])
        if path:
            var.set(path)
            # auto-fill threshold / model info from checkpoint
            if var is var_checkpoint:
                _try_prefill_from_checkpoint(path)

    def browse_dir(var: tk.StringVar, title: str):
        path = filedialog.askdirectory(title=title)
        if path:
            var.set(path)

    def _try_prefill_from_checkpoint(path: str) -> None:
        if not _TORCH_AVAILABLE:
            return
        try:
            ckpt = torch.load(path, map_location="cpu")
            cfg  = ckpt.get("config", {})
            thr  = ckpt.get("threshold", cfg.get("threshold", None))
            if thr is not None:
                var_threshold.set(f"{float(thr):.4f}")
        except Exception:
            pass

    def _auto_overlay_folder() -> Optional[Path]:
        out = var_output_mask.get().strip()
        custom = var_overlay_folder.get().strip()
        if custom:
            return Path(custom)
        if out:
            return Path(out + "_overlays")
        return None

    def _auto_prob_folder() -> Optional[Path]:
        out = var_output_mask.get().strip()
        custom = var_prob_folder.get().strip()
        if custom:
            return Path(custom)
        if out:
            return Path(out + "_probabilities")
        return None

    def log(msg: str) -> None:
        txt_log.config(state="normal")
        txt_log.insert("end", msg + "\n")
        txt_log.see("end")
        txt_log.config(state="disabled")
        root.update_idletasks()

    def _check_existing_masks(output_folder: Path) -> int:
        existing = list(output_folder.glob("*_mask.png"))
        return len(existing)

    def _on_run():
        if _running[0]:
            return

        # --- validate ---
        ckpt_str  = var_checkpoint.get().strip()
        in_str    = var_input_folder.get().strip()
        out_str   = var_output_mask.get().strip()

        if not ckpt_str:
            messagebox.showerror("Missing input", "Please select a model checkpoint.")
            return
        if not in_str:
            messagebox.showerror("Missing input", "Please select an input frame folder.")
            return
        if not out_str:
            messagebox.showerror("Missing input", "Please select an output mask folder.")
            return

        ckpt_path = Path(ckpt_str)
        if not ckpt_path.is_file():
            messagebox.showerror("File not found", f"Checkpoint not found:\n{ckpt_path}")
            return

        in_path = Path(in_str)
        if not in_path.is_dir():
            messagebox.showerror("Folder not found", f"Input folder not found:\n{in_path}")
            return

        out_path = Path(out_str)

        # overwrite check
        if out_path.exists():
            n = _check_existing_masks(out_path)
            if n > 0:
                answer = messagebox.askyesno(
                    "Existing masks detected",
                    f"Output folder already contains {n} *_mask.png files.\n"
                    "Existing masks may be overwritten.\n\n"
                    "Continue and overwrite existing masks?"
                )
                if not answer:
                    return

        try:
            threshold = float(var_threshold.get())
        except ValueError:
            messagebox.showerror("Invalid value", "Threshold must be a number.")
            return

        try:
            batch_size = int(var_batch_size.get())
            if batch_size < 1:
                raise ValueError
        except ValueError:
            messagebox.showerror("Invalid value", "Batch size must be a positive integer.")
            return

        save_overlays = var_save_overlays.get()
        save_probs    = var_save_probs.get()
        recursive     = var_recursive.get()
        device_str    = var_device.get().strip() or "auto"

        ov_folder   = _auto_overlay_folder() if save_overlays else None
        prob_folder = _auto_prob_folder()    if save_probs    else None

        cfg = PredictionConfig(
            checkpoint_path=ckpt_path,
            input_folder=in_path,
            output_mask_folder=out_path,
            output_overlay_folder=ov_folder,
            output_probability_folder=prob_folder,
            save_overlays=save_overlays,
            save_probabilities=save_probs,
            threshold=threshold,
            batch_size=batch_size,
            device_str=device_str,
            recursive=recursive,
        )

        _cancel_flag[0] = False
        _running[0] = True
        btn_run.config(state="disabled")
        btn_cancel.config(state="normal")

        txt_log.config(state="normal")
        txt_log.delete("1.0", "end")
        txt_log.config(state="disabled")

        def job():
            try:
                run_prediction_job(
                    cfg,
                    progress_fn=log,
                    cancel_fn=lambda: _cancel_flag[0],
                )
            except Exception as exc:
                log(f"ERROR: {exc}")
                log(traceback.format_exc())
            finally:
                _running[0] = False
                root.after(0, lambda: btn_run.config(state="normal"))
                root.after(0, lambda: btn_cancel.config(state="disabled"))

        threading.Thread(target=job, daemon=True).start()

    def _on_cancel():
        _cancel_flag[0] = True
        log("Cancellation requested...")

    # ---- layout ----
    pad = dict(padx=8, pady=4)

    # --- checkpoint ---
    frm_ckpt = ttk.LabelFrame(root, text="Model checkpoint")
    frm_ckpt.pack(fill="x", **pad)
    ttk.Entry(frm_ckpt, textvariable=var_checkpoint, width=60).pack(side="left", fill="x", expand=True, padx=4, pady=4)
    ttk.Button(frm_ckpt, text="Browse…", command=lambda: browse_file(
        var_checkpoint, "Select model checkpoint",
        filetypes=[("PyTorch checkpoint", "*.pth *.pt"), ("All", "*.*")]
    )).pack(side="right", padx=4, pady=4)

    # --- input folder ---
    frm_in = ttk.LabelFrame(root, text="Input frame folder")
    frm_in.pack(fill="x", **pad)
    ttk.Entry(frm_in, textvariable=var_input_folder, width=60).pack(side="left", fill="x", expand=True, padx=4, pady=4)
    ttk.Button(frm_in, text="Browse…", command=lambda: browse_dir(var_input_folder, "Select input frame folder")).pack(side="right", padx=4, pady=4)

    # --- output mask folder ---
    frm_out = ttk.LabelFrame(root, text="Output mask folder")
    frm_out.pack(fill="x", **pad)
    ttk.Entry(frm_out, textvariable=var_output_mask, width=60).pack(side="left", fill="x", expand=True, padx=4, pady=4)
    ttk.Button(frm_out, text="Browse…", command=lambda: browse_dir(var_output_mask, "Select output mask folder")).pack(side="right", padx=4, pady=4)

    # --- optional outputs ---
    frm_opt = ttk.LabelFrame(root, text="Optional outputs")
    frm_opt.pack(fill="x", **pad)

    ttk.Checkbutton(frm_opt, text="Save overlay previews", variable=var_save_overlays).grid(row=0, column=0, sticky="w", padx=6, pady=2)
    ttk.Entry(frm_opt, textvariable=var_overlay_folder, width=45).grid(row=0, column=1, padx=4, pady=2, sticky="ew")
    ttk.Button(frm_opt, text="Browse…", command=lambda: browse_dir(var_overlay_folder, "Select overlay output folder")).grid(row=0, column=2, padx=4, pady=2)

    ttk.Checkbutton(frm_opt, text="Save probability maps", variable=var_save_probs).grid(row=1, column=0, sticky="w", padx=6, pady=2)
    ttk.Entry(frm_opt, textvariable=var_prob_folder, width=45).grid(row=1, column=1, padx=4, pady=2, sticky="ew")
    ttk.Button(frm_opt, text="Browse…", command=lambda: browse_dir(var_prob_folder, "Select probability output folder")).grid(row=1, column=2, padx=4, pady=2)

    ttk.Checkbutton(frm_opt, text="Include subfolders (recursive)", variable=var_recursive).grid(row=2, column=0, sticky="w", padx=6, pady=2)
    frm_opt.columnconfigure(1, weight=1)

    # --- inference settings ---
    frm_cfg = ttk.LabelFrame(root, text="Inference settings")
    frm_cfg.pack(fill="x", **pad)

    lbl_kwargs = dict(anchor="e")
    ttk.Label(frm_cfg, text="Threshold:", **lbl_kwargs).grid(row=0, column=0, padx=6, pady=3, sticky="e")
    ttk.Entry(frm_cfg, textvariable=var_threshold, width=10).grid(row=0, column=1, padx=4, pady=3, sticky="w")

    ttk.Label(frm_cfg, text="Batch size:", **lbl_kwargs).grid(row=0, column=2, padx=6, pady=3, sticky="e")
    ttk.Entry(frm_cfg, textvariable=var_batch_size, width=8).grid(row=0, column=3, padx=4, pady=3, sticky="w")

    ttk.Label(frm_cfg, text="Device:", **lbl_kwargs).grid(row=0, column=4, padx=6, pady=3, sticky="e")
    ttk.Entry(frm_cfg, textvariable=var_device, width=12).grid(row=0, column=5, padx=4, pady=3, sticky="w")
    ttk.Label(frm_cfg, text="(auto / cuda / cpu / mps)").grid(row=0, column=6, padx=2, pady=3, sticky="w")

    # --- buttons ---
    frm_btn = ttk.Frame(root)
    frm_btn.pack(fill="x", **pad)
    btn_run = ttk.Button(frm_btn, text="Run Prediction", command=_on_run)
    btn_run.pack(side="left", padx=4, pady=4)
    btn_cancel = ttk.Button(frm_btn, text="Cancel", command=_on_cancel, state="disabled")
    btn_cancel.pack(side="left", padx=4, pady=4)

    # --- log ---
    frm_log = ttk.LabelFrame(root, text="Log")
    frm_log.pack(fill="both", expand=True, **pad)
    txt_log = tk.Text(frm_log, state="disabled", wrap="word", height=18)
    scrollbar = ttk.Scrollbar(frm_log, command=txt_log.yview)
    txt_log.config(yscrollcommand=scrollbar.set)
    scrollbar.pack(side="right", fill="y")
    txt_log.pack(fill="both", expand=True, padx=2, pady=2)

    if not _TORCH_AVAILABLE:
        log("WARNING: PyTorch / segmentation-models-pytorch not found.")
        log("Install them before running inference.")

    root.mainloop()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_cli_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Lava Fountain Frame-Folder Mask Predictor (CLI)"
    )
    p.add_argument("--checkpoint",    required=True,  help="Path to .pth checkpoint")
    p.add_argument("--input-folder",  required=True,  help="Folder of input frames")
    p.add_argument("--output-folder", required=True,  help="Output mask folder")
    p.add_argument("--overlay-folder", default=None,  help="Output overlay folder (optional)")
    p.add_argument("--prob-folder",    default=None,  help="Output probability folder (optional)")
    p.add_argument("--threshold",  type=float, default=None,
                   help="Threshold override (default: from checkpoint or 0.50)")
    p.add_argument("--batch-size", type=int,   default=DEFAULT_BATCH_SIZE)
    p.add_argument("--device",     default="auto", help="auto / cuda / cpu / mps")
    p.add_argument("--save-overlays",     action="store_true")
    p.add_argument("--save-probabilities", action="store_true")
    p.add_argument("--recursive",  action="store_true",
                   help="Process images in subfolders too")
    return p


def main_cli(argv: Optional[List[str]] = None) -> int:
    parser = build_cli_parser()
    args = parser.parse_args(argv)

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_file():
        print(f"ERROR: Checkpoint not found: {ckpt_path}", file=sys.stderr)
        return 1

    in_path = Path(args.input_folder)
    if not in_path.is_dir():
        print(f"ERROR: Input folder not found: {in_path}", file=sys.stderr)
        return 1

    out_path = Path(args.output_folder)

    ov_folder = Path(args.overlay_folder) if args.overlay_folder else (
        Path(str(out_path) + "_overlays") if args.save_overlays else None
    )
    prob_folder = Path(args.prob_folder) if args.prob_folder else (
        Path(str(out_path) + "_probabilities") if args.save_probabilities else None
    )

    cfg = PredictionConfig(
        checkpoint_path=ckpt_path,
        input_folder=in_path,
        output_mask_folder=out_path,
        output_overlay_folder=ov_folder,
        output_probability_folder=prob_folder,
        save_overlays=args.save_overlays,
        save_probabilities=args.save_probabilities,
        threshold=args.threshold,
        batch_size=args.batch_size,
        device_str=args.device,
        recursive=args.recursive,
    )

    try:
        result = run_prediction_job(cfg, progress_fn=print)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    return 0 if result["failed"] == 0 else 2


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) > 1:
        sys.exit(main_cli())
    else:
        launch_gui()
