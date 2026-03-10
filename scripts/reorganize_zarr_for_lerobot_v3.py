#!/usr/bin/env python3
"""Convert processed_matched zarr into official LeRobot v3 on-disk dataset layout.

Expected input metadata keys in zarr:
- matched_raw_video_paths_camera{idx}

Output layout (LeRobot v3):
- meta/info.json
- meta/tasks.parquet
- meta/episodes/chunk-000/file-000.parquet
- data/chunk-XXX/file-YYY.parquet
- videos/{video_key}/chunk-XXX/file-YYY.mp4
"""

import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

import json
import pathlib
import re
import shutil
import subprocess
from typing import Dict, List, Tuple

import av
import click
import numpy as np
import pandas as pd
import zarr

from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs
import pyarrow

register_codecs()

CAMERA_RGB_RE = re.compile(r"^camera\d+_rgb$")


def open_rb(path: pathlib.Path, mode: str):
    if path.suffix == ".zip":
        store = zarr.ZipStore(str(path), mode=mode)
        root = zarr.group(store=store)
        return store, ReplayBuffer.create_from_group(root)
    root = zarr.open(str(path), mode=mode)
    return None, ReplayBuffer.create_from_group(root)


def _episode_slices(episode_ends: np.ndarray) -> List[slice]:
    starts = np.concatenate([[0], episode_ends[:-1]])
    return [slice(int(s), int(e)) for s, e in zip(starts, episode_ends)]


def _safe_symlink(src: pathlib.Path, dst: pathlib.Path, overwrite: bool):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        if overwrite:
            dst.unlink()
        else:
            return
    rel = os.path.relpath(src, start=dst.parent)
    dst.symlink_to(rel)


def _copy_or_link_video(
    src: pathlib.Path,
    dst: pathlib.Path,
    mode: str,
    num_frames: int,
    overwrite: bool,
):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        if overwrite:
            if dst.is_symlink() or dst.is_file():
                dst.unlink()
        else:
            return

    if mode == "symlink":
        _safe_symlink(src, dst, overwrite=True)
        return

    if mode == "copy":
        shutil.copy2(src, dst)
        return

    if mode == "clip":
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(src),
            "-frames:v",
            str(int(num_frames)),
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "18",
            "-an",
            str(dst),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg clip failed for {src} -> {dst}: {proc.stderr.strip()}")
        return

    raise ValueError(f"Unknown video mode: {mode}")


def _to_lerobot_dtype(np_dtype) -> str:
    dt = np.dtype(np_dtype)
    if dt.kind == "f":
        return "float32" if dt.itemsize <= 4 else "float64"
    if dt.kind in ["i", "u"]:
        return "int64" if dt.itemsize >= 8 else "int32"
    if dt.kind == "b":
        return "bool"
    return "float32"


def _infer_fps_from_video(video_path: pathlib.Path) -> float:
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        afr = stream.average_rate
        if afr is not None:
            return float(afr)
        rfr = stream.base_rate
        if rfr is not None:
            return float(rfr)
    raise RuntimeError(f"Cannot infer fps from video: {video_path}")


def _normalize_meta_path(value) -> str:
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8")
    return str(value)


def _cast_np_to_feature_dtype(arr: np.ndarray, feature_dtype: str) -> np.ndarray:
    if feature_dtype == "float32":
        return arr.astype(np.float32, copy=False)
    if feature_dtype == "float64":
        return arr.astype(np.float64, copy=False)
    if feature_dtype == "int64":
        return arr.astype(np.int64, copy=False)
    if feature_dtype == "int32":
        return arr.astype(np.int32, copy=False)
    if feature_dtype == "bool":
        return arr.astype(bool, copy=False)
    return arr


def _build_features(
    rb: ReplayBuffer,
    lowdim_keys: List[str],
    camera_indices: List[int],
) -> Dict[str, dict]:
    # LeRobot default features.
    features = {
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None},
        "index": {"dtype": "int64", "shape": [1], "names": None},
        "task_index": {"dtype": "int64", "shape": [1], "names": None},
    }

    for key in sorted(lowdim_keys):
        arr = rb.data[key]
        shape = list(arr.shape[1:]) if len(arr.shape) > 1 else [1]
        if len(shape) == 0:
            shape = [1]
        names = None
        if len(shape) == 1:
            names = [f"{key}_{i}" for i in range(shape[0])]
        features[key] = {
            "dtype": _to_lerobot_dtype(arr.dtype),
            "shape": shape,
            "names": names,
        }

    for cam_idx in camera_indices:
        rgb_key = f"camera{cam_idx}_rgb"
        if rgb_key not in rb.data:
            continue
        arr = rb.data[rgb_key]
        h, w, c = map(int, arr.shape[1:4])
        vid_key = f"observation.images.camera{cam_idx}"
        features[vid_key] = {
            "dtype": "video",
            "shape": [h, w, c],
            "names": ["height", "width", "channels"],
        }

    return features


@click.command()
@click.option("--input", "input_path", required=True, type=str, help="processed_matched zarr path")
@click.option("--output-dir", required=True, type=str, help="LeRobot v3 dataset root directory")
@click.option("--repo-id", default="local/umi_dataset", show_default=True, type=str)
@click.option("--video-mode", type=click.Choice(["symlink", "copy", "clip"]), default="symlink", show_default=True)
@click.option("--overwrite", is_flag=True, default=False)
@click.option("--chunks-size", default=1000, show_default=True, type=int)
@click.option("--robot-type", default=None, type=str)
def main(
    input_path: str,
    output_dir: str,
    repo_id: str,
    video_mode: str,
    overwrite: bool,
    chunks_size: int,
    robot_type: str,
):
    src_path = pathlib.Path(os.path.expanduser(input_path)).absolute()
    out_root = pathlib.Path(os.path.expanduser(output_dir)).absolute()

    if out_root.exists() and any(out_root.iterdir()) and not overwrite:
        raise click.ClickException(f"Output dir not empty: {out_root}. Use --overwrite to allow updates.")

    (out_root / "meta" / "episodes").mkdir(parents=True, exist_ok=True)
    (out_root / "data").mkdir(parents=True, exist_ok=True)
    (out_root / "videos").mkdir(parents=True, exist_ok=True)

    src_store, src_rb = open_rb(src_path, mode="r")
    try:
        episode_ends = src_rb.episode_ends[:]
        ep_slices = _episode_slices(episode_ends)
        n_episodes = len(ep_slices)

        lowdim_keys = [k for k in src_rb.data.keys() if not CAMERA_RGB_RE.match(k)]
        if not lowdim_keys:
            raise click.ClickException("No low-dim keys found in input zarr")

        cam_indices = []
        for mk in src_rb.meta.keys():
            if mk.startswith("matched_raw_video_paths_camera"):
                idx = mk[len("matched_raw_video_paths_camera") :]
                if idx.isdigit():
                    cam_indices.append(int(idx))
        cam_indices = sorted(cam_indices)
        if not cam_indices:
            raise click.ClickException("Missing matched_raw_video_paths_camera* in zarr meta")

        matched_paths = {}
        for cam_idx in cam_indices:
            k = f"matched_raw_video_paths_camera{cam_idx}"
            vals = src_rb.meta[k][:]
            matched_paths[cam_idx] = [_normalize_meta_path(x) for x in vals.tolist()]
            if len(matched_paths[cam_idx]) != n_episodes:
                raise click.ClickException(
                    f"Meta {k} length {len(matched_paths[cam_idx])} != n_episodes {n_episodes}"
                )
        if n_episodes == 0:
            raise click.ClickException("Input zarr has zero episodes after matching/filtering.")

        first_cam = cam_indices[0]
        first_video = pathlib.Path(matched_paths[first_cam][0]).expanduser().absolute()
        if not first_video.is_file():
            raise click.ClickException(f"Matched video path does not exist: {first_video}")
        fps = _infer_fps_from_video(first_video)
        print(f"Inferred fps from {first_video}: {fps:.6f}")

        features = _build_features(src_rb, lowdim_keys, cam_indices)

        # task table
        tasks_df = pd.DataFrame({"task_index": [0]}, index=pd.Index(["default"], name="task"))
        tasks_df.to_parquet(out_root / "meta" / "tasks.parquet")

        # Per-episode data parquet and episode metadata rows.
        global_index = 0

        for ep_idx, sl in enumerate(ep_slices):
            ep_len = int(sl.stop - sl.start)
            chunk_idx = ep_idx // chunks_size
            file_idx = ep_idx % chunks_size

            # data parquet path
            data_rel = pathlib.Path("data") / f"chunk-{chunk_idx:03d}" / f"file-{file_idx:03d}.parquet"
            data_abs = out_root / data_rel
            data_abs.parent.mkdir(parents=True, exist_ok=True)

            frame_index = np.arange(ep_len, dtype=np.int64)
            timestamp = frame_index.astype(np.float32) / float(fps)
            index = np.arange(global_index, global_index + ep_len, dtype=np.int64)

            data_dict = {
                "timestamp": timestamp,
                "frame_index": frame_index,
                "episode_index": np.full(ep_len, ep_idx, dtype=np.int64),
                "index": index,
                "task_index": np.zeros(ep_len, dtype=np.int64),
            }
            for key in lowdim_keys:
                arr = src_rb.data[key][sl]
                feature_dtype = features[key]["dtype"]
                arr = _cast_np_to_feature_dtype(np.asarray(arr), feature_dtype)

                # shape [1] in LeRobot schema must be stored as scalar column.
                if arr.ndim == 1:
                    data_dict[key] = arr
                elif arr.ndim == 2 and arr.shape[1] == 1:
                    data_dict[key] = arr[:, 0]
                else:
                    # Store per-frame tensors as nested Python lists so parquet can encode them.
                    data_dict[key] = [x.tolist() for x in arr]

            pd.DataFrame(data_dict).to_parquet(data_abs)

            ep_row = {
                "episode_index": ep_idx,
                "tasks": ["default"],
                "length": ep_len,
                "dataset_from_index": int(global_index),
                "dataset_to_index": int(global_index + ep_len),
                "data/chunk_index": int(chunk_idx),
                "data/file_index": int(file_idx),
                "meta/episodes/chunk_index": int(chunk_idx),
                "meta/episodes/file_index": int(file_idx),
            }

            for cam_idx in cam_indices:
                src_vid = pathlib.Path(matched_paths[cam_idx][ep_idx]).expanduser().absolute()
                if not src_vid.is_file():
                    raise click.ClickException(f"Matched video path does not exist: {src_vid}")
                vid_key = f"observation.images.camera{cam_idx}"
                vid_rel = pathlib.Path("videos") / vid_key / f"chunk-{chunk_idx:03d}" / f"file-{file_idx:03d}.mp4"
                vid_abs = out_root / vid_rel
                _copy_or_link_video(src=src_vid, dst=vid_abs, mode=video_mode, num_frames=ep_len, overwrite=overwrite)

                ep_row[f"videos/{vid_key}/chunk_index"] = int(chunk_idx)
                ep_row[f"videos/{vid_key}/file_index"] = int(file_idx)
                ep_row[f"videos/{vid_key}/from_timestamp"] = 0.0
                ep_row[f"videos/{vid_key}/to_timestamp"] = float(ep_len) / float(fps)

            # Keep episode parquet chunking/file indexing aligned with data parquet.
            episodes_path = (
                out_root
                / "meta"
                / "episodes"
                / f"chunk-{chunk_idx:03d}"
                / f"file-{file_idx:03d}.parquet"
            )
            episodes_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame([ep_row]).to_parquet(episodes_path)
            global_index += ep_len

        info = {
            "codebase_version": "v3.0",
            "robot_type": robot_type,
            "total_episodes": int(n_episodes),
            "total_frames": int(global_index),
            "total_tasks": 1,
            "chunks_size": int(chunks_size),
            "data_files_size_in_mb": 100,
            "video_files_size_in_mb": 200,
            "fps": float(fps),
            "splits": {"train": f"0:{n_episodes}"},
            "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
            "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
            "features": features,
            "repo_id": repo_id,
        }
        (out_root / "meta" / "info.json").write_text(json.dumps(info, indent=2))

        # Optional stats; loader accepts missing stats, but writing helps compatibility.
        stats = {}
        for key in [k for k in features.keys() if k in src_rb.data.keys() and not CAMERA_RGB_RE.match(k)]:
            arr = src_rb.data[key][:]
            flat = arr.reshape(arr.shape[0], -1) if arr.ndim > 1 else arr[:, None]
            stats[key] = {
                "mean": np.mean(flat, axis=0).tolist(),
                "std": np.std(flat, axis=0).tolist(),
                "min": np.min(flat, axis=0).tolist(),
                "max": np.max(flat, axis=0).tolist(),
            }
        (out_root / "meta" / "stats.json").write_text(json.dumps(stats, indent=2))

        print(f"Converted {n_episodes} episodes to LeRobot v3 format at: {out_root}")
    finally:
        if src_store is not None:
            src_store.close()


if __name__ == "__main__":
    main()
