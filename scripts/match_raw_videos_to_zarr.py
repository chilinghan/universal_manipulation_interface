#!/usr/bin/env python3
"""Match raw videos to processed zarr episodes and save matched paths into metadata.

Workflow:
1) Compare frame counts: zarr episode lengths vs raw videos.
2) Apply the same UMI image processing to anchor raw-video frames.
3) For each zarr episode/camera, match candidates with same frame count.
4) Resolve ambiguity with first/mid/last-frame MSE.
5) Drop episodes with missing/low-confidence matches.
6) Save matched paths + MSE to zarr meta.
"""

import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

import json
import pathlib
import pickle
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import av
import click
import numpy as np
import zarr

from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs
from umi.common.video_preprocess import make_umi_image_processor

register_codecs()


@dataclass
class RawVideoInfo:
    path: str
    n_frames: int
    width: int
    height: int
    first_frame_processed: Optional[np.ndarray] = None


def _open_replay_buffer(zarr_path: pathlib.Path, mode: str):
    if zarr_path.suffix == ".zip":
        store = zarr.ZipStore(str(zarr_path), mode=mode)
        root = zarr.group(store=store)
        return store, ReplayBuffer.create_from_group(root)
    root = zarr.open(str(zarr_path), mode=mode)
    return None, ReplayBuffer.create_from_group(root)


def _count_video_frames(path: str) -> Tuple[int, int, int]:
    with av.open(path) as container:
        stream = container.streams.video[0]
        width = int(stream.width)
        height = int(stream.height)
        n_frames = int(stream.frames or 0)
        if n_frames > 0:
            return n_frames, width, height

        n = 0
        for _ in container.decode(stream):
            n += 1
        return n, width, height


def _read_frame_rgb_at(path: str, frame_idx: int) -> np.ndarray:
    if frame_idx < 0:
        raise ValueError(f"frame_idx must be >= 0, got {frame_idx}")
    with av.open(path) as container:
        stream = container.streams.video[0]
        for i, frame in enumerate(container.decode(stream)):
            if i == frame_idx:
                return frame.to_ndarray(format="rgb24")
    raise RuntimeError(f"No frame {frame_idx} decoded from {path}")


def _infer_camera_indices(rb: ReplayBuffer) -> List[int]:
    cam_idxs = []
    for key in rb.data.keys():
        if key.startswith("camera") and key.endswith("_rgb"):
            mid = key[len("camera") : -len("_rgb")]
            if mid.isdigit():
                cam_idxs.append(int(mid))
    return sorted(cam_idxs)


def _episode_lengths(episode_ends: np.ndarray) -> np.ndarray:
    starts = np.concatenate([[0], episode_ends[:-1]])
    return episode_ends - starts


def _as_unicode_array(values: Sequence[str]) -> np.ndarray:
    maxlen = max([len(v) for v in values], default=1)
    return np.asarray(values, dtype=f"<U{maxlen}")


def _copy_filtered_episodes(
    src_rb: ReplayBuffer,
    keep_episode_indices: List[int],
    out_path: pathlib.Path,
    extra_meta: Dict[str, np.ndarray],
):
    src_data_arrays = {k: src_rb.data[k] for k in src_rb.data.keys()}
    chunks = {k: src_data_arrays[k].chunks for k in src_data_arrays.keys()}
    compressors = {k: src_data_arrays[k].compressor for k in src_data_arrays.keys()}

    # Build in memory first. ZipStore doesn't support rename(), but ReplayBuffer.add_episode
    # may call rechunk_recompress_array -> group.move() -> rename().
    out_rb = ReplayBuffer.create_empty_zarr(storage=zarr.MemoryStore())

    for ep_idx in keep_episode_indices:
        sl = src_rb.get_episode_slice(ep_idx)
        episode_data = {k: src_rb.data[k][sl] for k in src_rb.data.keys()}
        out_rb.add_episode(episode_data, chunks=chunks, compressors=compressors)

    inherited_meta = {}
    for k in src_rb.meta.keys():
        if k == "episode_ends":
            continue
        arr = src_rb.meta[k]
        inherited_meta[k] = np.array(arr) if arr.shape == () else arr[:]

    if inherited_meta:
        out_rb.update_meta(inherited_meta)
    if extra_meta:
        safe_meta = {}
        for k, v in extra_meta.items():
            arr = np.asarray(v)
            # zarr array(..., chunks=value.shape) fails for any zero-sized dim.
            if arr.ndim > 0 and any(d == 0 for d in arr.shape):
                print(f"Skipping empty meta array: {k} shape={arr.shape}")
                continue
            safe_meta[k] = arr
        if safe_meta:
            out_rb.update_meta(safe_meta)

    if out_path.suffix == ".zip":
        out_store = zarr.ZipStore(str(out_path), mode="w")
        try:
            out_rb.save_to_store(out_store)
        finally:
            out_store.close()
    else:
        out_root = zarr.open(str(out_path), mode="w")
        out_rb.save_to_store(out_root.store)


@click.command()
@click.option("--zarr", "zarr_path", required=True, type=str, help="Input processed zarr path (dir or .zip)")
@click.option("--raw-video-dir", required=True, type=str, help="Directory containing raw videos")
@click.option("--output", required=True, type=str, help="Output zarr path with matched metadata and dropped episodes")
@click.option("--video-glob", default="**/*.mp4", show_default=True, type=str)
@click.option("--no-mirror", is_flag=True, default=False)
@click.option("--mirror-swap", is_flag=True, default=False)
@click.option("--debug-dir", default=None, type=str, help="Optional directory to write debug matched frames")
def main(
    zarr_path: str,
    raw_video_dir: str,
    output: str,
    video_glob: str,
    no_mirror: bool,
    mirror_swap: bool,
    debug_dir: Optional[str],
):
    zarr_path = pathlib.Path(os.path.expanduser(zarr_path)).absolute()
    raw_video_dir = pathlib.Path(os.path.expanduser(raw_video_dir)).absolute()
    output = pathlib.Path(os.path.expanduser(output)).absolute()
    debug_dir_path = None
    if debug_dir:
        debug_dir_path = pathlib.Path(os.path.expanduser(debug_dir)).absolute()
        debug_dir_path.mkdir(parents=True, exist_ok=True)

    store, rb = _open_replay_buffer(zarr_path, mode="r")
    try:
        camera_indices = _infer_camera_indices(rb)
        if not camera_indices:
            raise click.ClickException("No camera*_rgb arrays found in zarr data")

        episode_ends = rb.episode_ends[:]
        ep_lengths = _episode_lengths(episode_ends)
        n_episodes = len(ep_lengths)

        zarr_starts = np.concatenate([[0], episode_ends[:-1]])

        print(f"Found {n_episodes} episodes in zarr")
        print(f"Cameras in zarr: {camera_indices}")

        raw_paths = sorted(raw_video_dir.glob(video_glob))
        raw_paths = [p for p in raw_paths if p.is_file() and p.suffix.lower() in {".mp4", ".mov", ".mkv", ".avi"}]
        if not raw_paths:
            raise click.ClickException(f"No videos found under {raw_video_dir} matching {video_glob}")

        raw_infos: List[RawVideoInfo] = []
        for p in raw_paths:
            n_frames, w, h = _count_video_frames(str(p))
            raw_infos.append(RawVideoInfo(path=str(p), n_frames=n_frames, width=w, height=h))

        frame_count_hist: Dict[int, int] = {}
        for r in raw_infos:
            frame_count_hist[r.n_frames] = frame_count_hist.get(r.n_frames, 0) + 1

        print(f"Indexed {len(raw_infos)} raw videos")
        print(f"Unique raw frame counts: {len(frame_count_hist)}")

        processors: Dict[Tuple[int, int, int, int], callable] = {}
        processed_frame_cache: Dict[Tuple[str, int, int, int], np.ndarray] = {}

        def preprocess_frame_at(info: RawVideoInfo, out_w: int, out_h: int, frame_idx: int) -> np.ndarray:
            cache_key = (info.path, out_w, out_h, frame_idx)
            if cache_key in processed_frame_cache:
                return processed_frame_cache[cache_key]

            key = (info.width, info.height, out_w, out_h)
            if key not in processors:
                processors[key] = make_umi_image_processor(
                    in_res=(info.width, info.height),
                    out_res=(out_w, out_h),
                    no_mirror=no_mirror,
                    mirror_swap=mirror_swap,
                    fisheye_converter=None,
                )
            rgb = _read_frame_rgb_at(info.path, frame_idx=frame_idx)
            
            out = processors[key](rgb, None) # don't inpaint tag
            processed_frame_cache[cache_key] = out
            return out

        keep_eps: List[int] = []
        dropped_eps: List[int] = []
        matched_paths_by_cam: Dict[int, List[str]] = {c: [] for c in camera_indices}
        matched_mse_by_cam: Dict[int, List[float]] = {c: [] for c in camera_indices}

        by_frame_count: Dict[int, List[RawVideoInfo]] = {}
        for info in raw_infos:
            by_frame_count.setdefault(info.n_frames, []).append(info)

        print(f"Frame count distribution: {sorted(list(by_frame_count.keys()))}")
        print(f"Zarr episode lengths: {sorted(set(ep_lengths))}")
        for ep_idx in range(n_episodes):
            length = int(ep_lengths[ep_idx])
            start_idx = int(zarr_starts[ep_idx])
            
            candidates = by_frame_count.get(length, [])
            if not candidates:
                dropped_eps.append(ep_idx)
                continue

            assigned = {}
            used_paths = set()
            failed = False

            print(f"Processing Zarr episode {ep_idx} with length {length}, candidates: {len(candidates)}")
            for cam_idx in camera_indices:
                key = f"camera{cam_idx}_rgb"
                anchor_offsets = sorted(set([0, max(length - 1, 0)]))
                zimgs = {ao: rb.data[key][start_idx + ao] for ao in anchor_offsets}
                zimg0 = zimgs[anchor_offsets[0]]
                out_h, out_w = int(zimg0.shape[0]), int(zimg0.shape[1])

                scored = []
                for c in candidates:
                    if c.path in used_paths:
                        continue
                    if c.n_frames <= anchor_offsets[-1]:
                        continue
                    mses = []
                    ok = True
                    for ao in anchor_offsets:
                        try:
                            proc = preprocess_frame_at(c, out_w=out_w, out_h=out_h, frame_idx=ao)
                        except Exception as e:
                            ok = False
                            break
                        zimg = zimgs[ao]
                        mse = float(np.mean((proc.astype(np.float32) - zimg.astype(np.float32)) ** 2))
                        mses.append(mse)
                    if not ok or len(mses) == 0:
                        continue
                    mean_mse = float(np.mean(mses))
                    scored.append((mean_mse, c.path))

                if not scored:
                    failed = True
                    break

                scored.sort(key=lambda x: x[0])
                best_mse, best_path = scored[0]
                second_mse, second_path = (scored[1] if len(scored) > 1 else (None, None))
                if debug_dir_path is not None:
                    with debug_dir_path.joinpath("match_debug.json").open("a") as f:
                        json.dump({
                            "episode_index": ep_idx,
                            "camera_index": cam_idx,
                            "zarr_length": length,
                            "best_path": best_path,
                            "best_mse": best_mse,
                            "best_length": next((c.n_frames for c in candidates if c.path == best_path), None),
                            "second_path": second_path,
                            "second_mse": second_mse,
                            "second_length": next((c.n_frames for c in candidates if c.path == second_path), None) if second_path else None,
                        }, f, indent=2)

                assigned[cam_idx] = (best_path, best_mse)
                used_paths.add(best_path)

            if failed:
                dropped_eps.append(ep_idx)
                continue

            keep_eps.append(ep_idx)
            for cam_idx in camera_indices:
                p, mse = assigned[cam_idx]
                matched_paths_by_cam[cam_idx].append(p)
                matched_mse_by_cam[cam_idx].append(mse)

        print(f"Matched episodes: {len(keep_eps)} / {n_episodes}")
        print(f"Dropped episodes: {len(dropped_eps)}")

        extra_meta = {
            "num_source_episodes_total": np.asarray(n_episodes, dtype=np.int64),
            "num_source_episodes_kept": np.asarray(len(keep_eps), dtype=np.int64),
            "num_source_episodes_dropped": np.asarray(len(dropped_eps), dtype=np.int64),
        }
        if len(keep_eps) > 0:
            extra_meta["source_episode_idx"] = np.asarray(keep_eps, dtype=np.int64)
        for cam_idx in camera_indices:
            if len(matched_paths_by_cam[cam_idx]) > 0:
                extra_meta[f"matched_raw_video_paths_camera{cam_idx}"] = _as_unicode_array(matched_paths_by_cam[cam_idx])
                extra_meta[f"matched_raw_video_mse_camera{cam_idx}"] = np.asarray(matched_mse_by_cam[cam_idx], dtype=np.float32)

        _copy_filtered_episodes(
            src_rb=rb,
            keep_episode_indices=keep_eps,
            out_path=output,
            extra_meta=extra_meta,
        )

        print(f"Wrote matched zarr: {output}")
    finally:
        if store is not None:
            store.close()


if __name__ == "__main__":
    main()
