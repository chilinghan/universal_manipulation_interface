# Dataset Preparation Pipeline

This guide describes the steps to convert raw demonstration videos into a **LeRobot v3 dataset**.

Pipeline overview:

1. **Transcode videos** (crop + resize)
2. **Match raw videos to Zarr frames**
3. **Convert dataset to LeRobot v3 format**

---
# 1. Transcode Videos

Transcode raw videos to square 1920×1920 resolution.

Transformation:

- Input assumed **2704×2028**
- **Center crop → 2028×2028**
- **Resize → 1920×1920**
- Decode using **`hevc_cuvid`**
- Encode using **`av1_nvenc`**

## Example Demo Session

```bash
python run_slam_pipeline.py example_demo_session
python scripts/transcode_videos.py \
  --input-dir ./example_demo_session/raw_videos \
  --output-dir ./example_demo_session/raw_videos_transcoded \
  --glob "**/*.mp4" \
  --num-workers 8
```

## Cup in the Wild Dataset

```bash
python scripts/transcode_videos.py \
  --input-dir data/cup_in_the_wild_mp4s \
  --output-dir data/cup_in_the_wild_mp4s_transcoded \
  --glob '**/*.mp4' \
  --resize-size (1080, 1080) \
  --num-workers 8
```

---

# 2. Match Zarr Frames to Raw Videos

This step aligns frames from the Zarr dataset with frames from the transcoded videos.

Parameters:

* **`--mse-threshold`**
  Maximum pixel MSE allowed for frame matching.

* **`--frame-count-tolerance`**
  Allowed mismatch in frame counts.

## Example Demo Session

```bash
python scripts/match_raw_videos_to_zarr.py \
  --zarr ./example_demo_session/dataset.zarr.zip \
  --raw-video-dir ./example_demo_session/raw_videos_transcoded/ \
  --output ./example_demo_session/matched_data.zarr.zip
```

## Cup in the Wild Dataset

```bash
python scripts/match_raw_videos_to_zarr.py \
  --zarr ./data/cup_in_the_wild.zarr.zip \
  --raw-video-dir ./data/cup_in_the_wild_mp4s_transcoded \
  --output ./data/cup_in_the_wild_matched_data.zarr.zip \
  --debug-dir ./data/cup_in_the_wild_debug
```

---

# 3. Convert to LeRobot v3 Dataset

Reorganize the Zarr dataset into **LeRobot v3 format**.

## Example Demo Session

```bash
python scripts/reorganize_zarr_for_lerobot_v3.py \
  --input ./example_demo_session/matched_data.zarr.zip \
  --output-dir ./example_demo_session/lerobot_v3_dataset \
  --repo-id local/example_demo_session \
  --video-mode symlink \
  --overwrite
```

## Cup in the Wild Dataset

```bash
python scripts/reorganize_zarr_for_lerobot_v3.py \
  --input ./data/cup_in_the_wild_matched_data.zarr.zip \
  --output-dir ./data/cup_in_the_wild_lerobot_v3_dataset \
  --repo-id local/cup_in_the_wild \
  --video-mode symlink \
  --overwrite
```

---

# Requirements

`reorganize_zarr_for_lerobot_v3.py` requires both:

* `zarr`
* `pyarrow`

These must be installed **in the same Python environment**.

Example:

```bash
pip install zarr pyarrow
```

---

# Final Output

After completing all steps, the dataset will be organized as a **LeRobot v3-compatible dataset** ready for training or uploading.