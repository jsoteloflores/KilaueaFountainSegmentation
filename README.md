# KilaueaFountainSegmentation

Binary segmentation of **visible airborne incandescent lava fountain material** inside a
target-fountain analysis ROI.

```text
RGB ROI image  ->  binary mask (255 = positive lava, 0 = background)
```

Pixels outside the ROI are not part of the sample and do not affect training.

## Files

| File | Purpose |
| --- | --- |
| [lava_fountain_segmentation_colab.ipynb](lava_fountain_segmentation_colab.ipynb) | Single self-contained Colab notebook — upload to Google Colab and run top to bottom. |
| [lava_fountain_segmentation_colab.py](lava_fountain_segmentation_colab.py) | Identical source in Jupyter "percent" format (readable/diffable; opens as cells in VS Code). |

The `.py` is the source of truth; the `.ipynb` is generated from it.

## Quick start (Google Colab)

1. Runtime ▸ Change runtime type ▸ **GPU** (A100 recommended for `input_size=2048`).
2. Upload `lava_fountain_segmentation_colab.ipynb` (File ▸ Upload notebook).
3. In **Section 3 (CONFIG)** set:
   - `dataset_root` — folder containing `metadata/frames.csv`
   - `run_root` — where run outputs are written
   - `input_size` — `768` debug · `1536` recommended · `2048` A100
   - `batch_size` — start at `2`
4. Run all cells. A live training graph updates every epoch.

## Expected dataset layout

```text
dataset_root/
  metadata/frames.csv          # required (never modified by the script)
  metadata/split.csv           # optional; auto-generated if absent
  images/all/<sample_id>.png
  masks/all/<sample_id>_mask.png
```

Required `frames.csv` columns: `sample_id`, `image_path`, `mask_path`. Optional columns
(`episode_id`, `camera_id`, `frame_index`, `time_seconds`, `label_status`,
`lighting_condition`, `contains_smoke/tephra/base_glow`, ROI/size columns,
`mask_positive_fraction`) are used when present and handled gracefully when absent.
Image/mask pairs may be any size (up to 4K); each mask must match its own image's
dimensions. Masks are canonicalized to binary `{0, 255}`.

## Modes (`CONFIG['mode']`)

- `train` — fresh training, then threshold sweep, overlays, and inference on all frames.
- `resume_train` — continue from `CONFIG['checkpoint_path']`.
- `evaluate` — load a checkpoint, sweep thresholds, export QC overlays.
- `infer_all` — load a checkpoint and predict on every frame in `frames.csv`.

## Outputs

Each run writes a timestamped folder under `run_root`:

```text
runs/lava_unet_<encoder>_<timestamp>/
  best_model.pth  last_model.pth  config.json  split.csv
  metrics_history.csv  threshold_report.csv  training_curves.png
  validation_overlays/  test_overlays/
  predictions_all_frames/
    masks/            <sample_id>_pred_mask.png   (0/255, original ROI size)
    probabilities/    <sample_id>_prob.png
    overlays/         <sample_id>_overlay.png
    comparisons/      <sample_id>_gt_vs_pred.png
    full_frame_masks/ (when ROI metadata is present)
    for_labeller/masks/all/<sample_id>_mask.png   (drop-in for FountainLabeller)
```

## Active-learning loop

```text
label frames  ->  train  ->  predict all frames  ->  export draft masks
   ^                                                        |
   +------------ refine in FountainLabeller  <--------------+
```

Predicted masks are binary `0/255` PNGs that exactly match the original image dimensions, so
they can be loaded straight back into FountainLabeller for correction. Add frames and rerun
(`train`) or continue (`resume_train`).

## Model

Pretrained-encoder U-Net via `segmentation_models_pytorch` (default
`Unet` + `efficientnet-b3`, ImageNet weights). Aspect ratio is preserved via resize + square
padding; padded pixels are excluded from the masked BCE + Dice loss and all metrics. Also
selectable: `UnetPlusPlus`, `DeepLabV3Plus`, `FPN`, `MAnet`.