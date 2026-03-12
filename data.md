# Dataset Preparation Pipeline

This pipeline aligns raw videos to processed zarr frames, then transcodes aligned source segments, then converts to LeRobot v3.

## 1. Match Raw Videos to Zarr Frames

This step matches each zarr episode/camera to the best raw video segment by MSE, while accounting for frame-window offsets produced by `scripts_slam_pipeline/06_generate_dataset_plan.py`.

Outputs:
- matched zarr containing filtered episodes
- per-camera metadata:
  - `matched_raw_video_paths_camera{idx}`
  - `matched_raw_video_start_frame_camera{idx}`
  - `matched_raw_video_n_frames_camera{idx}`
  - `matched_raw_video_mse_camera{idx}`

```bash
python scripts/match_raw_videos_to_zarr.py \
  --zarr <processed_dataset.zarr.zip> \
  --raw-video-dir <raw_video_root_dir> \
  --video-glob "**/*.mp4" \
  --output <matched_dataset.zarr.zip> \
  --debug-dir <optional_debug_dir>
```

Notes:
- Use `--video-glob` to choose which files under `--raw-video-dir` are considered.
- Use `--no-mirror` / `--mirror-swap` if your UMI preprocessing requires it.

## 2. Transcode Matched Segments

This step reads trim ranges from the matched zarr metadata and transcodes only matched segments.

Filter order per output clip:
1. trim to matched `[start_frame, start_frame + n_frames)`
2. center crop
3. resize

```bash
python scripts/transcode_videos.py \
  --input-dir <raw_video_root_dir> \
  --video-glob "**/*.mp4" \
  --match-zarr <matched_dataset.zarr.zip> \
  --output-dir <transcoded_video_root_dir> \
  --crop-size 2028 2028 \
  --resize-size 1920 1920 \
  --num-workers 8
```

Notes:
- Paths are matched by absolute normalized path, so keep raw videos in the same location between step 1 and step 2.
- `tag_detection.pkl` is trimmed and corner coordinates are transformed with the same crop+resize parameters.

## 3. Convert to LeRobot v3

```bash
python scripts/reorganize_zarr_for_lerobot_v3.py \
  --input <matched_dataset.zarr.zip> \
  --output-dir <lerobot_output_dir> \
  --repo-id <repo_id> \
  --video-mode symlink \
  --overwrite
```

## Requirements

Install required packages in the same Python environment used to run the scripts.

```bash
pip install zarr pyarrow
```

## Example Commands

### example_demo_session

```bash
python run_slam_pipeline.py example_demo_session

python scripts/match_raw_videos_to_zarr.py \
  --zarr ./example_demo_session/dataset.zarr.zip \
  --raw-video-dir ./example_demo_session/raw_videos \
  --video-glob "*.mp4" \
  --output ./example_demo_session/matched_data.zarr.zip \
  --debug-dir ./example_demo_session/debug

python scripts/transcode_videos.py \
  --input-dir ./example_demo_session/raw_videos \
  --video-glob "*.mp4" \
  --match-zarr ./example_demo_session/matched_data.zarr.zip \
  --output-dir ./example_demo_session/raw_videos_transcoded \
  --crop-size 2028 2028 \
  --resize-size 1920 1920 \
  --num-workers 8

python scripts/reorganize_zarr_for_lerobot_v3.py \
  --input ./example_demo_session/matched_data.zarr.zip \
  --output-dir ./example_demo_session/lerobot_v3_dataset \
  --repo-id local/example_demo_session \
  --video-mode symlink \
  --overwrite
```

### cup_in_the_wild_mp4s
/home/chilingh/projects/universal_manipulation_interface/data/cup_in_the_wild_mp4s/20240110_allen_front_door_facing_shriram


```bash
python scripts/match_raw_videos_to_zarr.py \
  --zarr ./data/cup_in_the_wild.zarr.zip \
  --raw-video-dir ./data/cup_in_the_wild_mp4s \
  --video-glob "**/*.mp4" \
  --output ./data/cup_in_the_wild_matched_data.zarr.zip \
  --debug-dir ./data/cup_in_the_wild_debug

python scripts/transcode_videos.py \
  --input-dir ./data/cup_in_the_wild_mp4s \
  --video-glob "**/*.mp4" \
  --match-zarr ./data/cup_in_the_wild_matched_data.zarr.zip \
  --output-dir ./data/cup_in_the_wild_mp4s_transcoded \
  --crop-size 2028 2028 \
  --resize-size 720 720 \
  --num-workers 8

python scripts/reorganize_zarr_for_lerobot_v3.py \
  --input ./data/cup_in_the_wild_matched_data.zarr.zip \
  --output-dir ./data/cup_in_the_wild_lerobot_v3_dataset \
  --repo-id local/cup_in_the_wild \
  --video-mode symlink \
  --overwrite
```
