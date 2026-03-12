#!/usr/bin/env python3
"""Match raw videos to processed zarr episodes and write alignment metadata.

Pipeline intent:
1) Reproduce dataset-plan camera alignment windows from raw videos.
2) For each zarr episode/camera, search start offsets inside each raw window.
3) Pick the candidate with lowest mean MSE on anchor frames.
4) Keep only fully matched episodes and save matched path/start/n_frames per camera.
"""

import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

import json
import pathlib
from dataclasses import dataclass
from typing import Dict, List

import av
import click
import numpy as np
import zarr
from tqdm import tqdm

from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs
from umi.common.cv_util import make_umi_image_processor
from umi.common.raw_video_util import collect_alignment_windows, count_video_frames, normalize_video_path, resolve_video_paths

register_codecs()
av.logging.set_level(av.logging.ERROR)

@dataclass
class RawVideoInfo:
    path: str
    n_frames: int
    width: int
    height: int
    # window_start_frame: int
    # window_n_frames: int


def _open_replay_buffer(zarr_path: pathlib.Path, mode: str):
    if zarr_path.suffix == ".zip":
        store = zarr.ZipStore(str(zarr_path), mode=mode)
        root = zarr.group(store=store)
        return store, ReplayBuffer.create_from_group(root)
    root = zarr.open(str(zarr_path), mode=mode)
    return None, ReplayBuffer.create_from_group(root)


def _read_preprocessed_frame(info: RawVideoInfo, frame_idx: int, processor) -> np.ndarray:
    if frame_idx < 0:
        raise ValueError(f"frame_idx must be >= 0, got {frame_idx}")
    with av.open(info.path) as container:
        stream = container.streams.video[0]
        for i, frame in enumerate(container.decode(stream)):
            if i == frame_idx:
                return processor(frame.to_ndarray(format="rgb24"), None)
    raise RuntimeError(f"No frame {frame_idx} decoded from {info.path}")


def _read_preprocessed_frame_with_seek(info: RawVideoInfo, frame_idx: int, processor) -> np.ndarray:
    with av.open(info.path) as container:
        stream = container.streams.video[0]

        fps = float(stream.average_rate)
        time_base = float(stream.time_base)

        target_pts = int((frame_idx / fps) / time_base)
        container.seek(max(target_pts, 0), stream=stream, backward=True)

        for frame in container.decode(stream):
            if frame.pts is None:
                continue

            decoded_idx = int(round(float(frame.pts) * time_base * fps))
            if decoded_idx == frame_idx:
                rgb = frame.to_ndarray(format="rgb24")
                return processor(rgb, None)

            if decoded_idx > frame_idx:
                break

    raise RuntimeError(f"Frame {frame_idx} not found in {info.path}")


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


def _as_unicode_array(values: List[str]) -> np.ndarray:
    maxlen = max([len(v) for v in values], default=1)
    return np.asarray(values, dtype=f"<U{maxlen}")


def _copy_filtered_episodes(src_rb: ReplayBuffer, keep_episode_indices: List[int], out_path: pathlib.Path, extra_meta: Dict[str, np.ndarray]):
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
@click.option("--debug-dir", default=None, type=str, help="Optional directory to write jsonl debug rows")
def main(
    zarr_path: str,
    raw_video_dir: str,
    output: str,
    video_glob: str,
    no_mirror: bool,
    mirror_swap: bool,
    debug_dir,
):
    zarr_path = pathlib.Path(os.path.expanduser(zarr_path)).absolute()
    raw_video_dir = pathlib.Path(os.path.expanduser(raw_video_dir)).absolute()
    output = pathlib.Path(os.path.expanduser(output)).absolute()
    debug_dir_path = None
    if debug_dir:
        debug_dir_path = pathlib.Path(os.path.expanduser(debug_dir)).absolute()
        debug_dir_path.mkdir(parents=True, exist_ok=True)
        debug_file = debug_dir_path / "match_debug.jsonl"
        if debug_file.exists():
            debug_file.unlink()

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

        try:
            raw_paths, _ = resolve_video_paths(raw_video_dir, glob_pattern=video_glob)
        except FileNotFoundError as exc:
            raise click.ClickException(str(exc))
        print(f"Resolved {len(raw_paths)} videos from {raw_video_dir}")

        # frame_windows = collect_alignment_windows(raw_paths, glob_pattern=video_glob)
        # if frame_windows:
        #     print(
        #         "Using dataset-plan style alignment windows "
        #         f"(includes end-frame -1 behavior) for {len(frame_windows)} videos"
        #     )
        # else:
        #     print("No alignment windows resolved; falling back to full video ranges")
        print("Alignment windows disabled; using full video ranges")

        raw_infos: List[RawVideoInfo] = []
        for p in raw_paths:
            n_frames, width, height = count_video_frames(str(p))
            # window_start, window_n_frames = frame_windows.get(p, (0, n_frames))
            raw_infos.append(
                RawVideoInfo(
                    path=normalize_video_path(p),
                    n_frames=int(n_frames),
                    width=int(width),
                    height=int(height),
                    # window_start_frame=int(window_start),
                    # window_n_frames=n_frames,
                )
            )

        by_window_frame_count: Dict[int, List[RawVideoInfo]] = {}
        for info in raw_infos:
            by_window_frame_count.setdefault(info.n_frames, []).append(info)

        processors = {}

        keep_eps: List[int] = []
        dropped_eps: List[int] = []
        matched_paths_by_cam: Dict[int, List[str]] = {c: [] for c in camera_indices}
        matched_mse_by_cam: Dict[int, List[float]] = {c: [] for c in camera_indices}
        matched_start_by_cam: Dict[int, List[int]] = {c: [] for c in camera_indices}

        print(f"Window frame counts: {list(by_window_frame_count.keys())}")
        for ep_idx in tqdm(range(n_episodes), desc="Matching episodes"):
            length = int(ep_lengths[ep_idx])
            start_idx = int(zarr_starts[ep_idx])

            assigned = {}
            used_paths = set()
            failed = False

            print(f"Matching episode {ep_idx}/{n_episodes - 1} (length={length})")
            candidate_lengths = range(length + 1, length + 5)
            candidates = []
            for candidate_length in candidate_lengths:
                candidates.extend(by_window_frame_count.get(candidate_length, []))
            print(f"Found {len(candidates)} candidate videos with matching window frame count")
            if not candidates:
                dropped_eps.append(ep_idx)
                continue

            for cam_idx in camera_indices:
                key = f"camera{cam_idx}_rgb"
                zimg = rb.data[key][start_idx + max(length - 1, 0)]
                out_h, out_w = int(zimg.shape[0]), int(zimg.shape[1])

                scored = []
                for c in candidates:
                    if c.path in used_paths:
                        continue
                    end_frame_idx = c.n_frames - 1
                    best_start_frame = end_frame_idx - length + 1
                    if best_start_frame < 0:
                        continue
                    proc_key = (c.width, c.height, out_w, out_h)
                    if proc_key not in processors:
                        processors[proc_key] = make_umi_image_processor(
                            in_res=(c.width, c.height),
                            out_res=(out_w, out_h),
                            no_mirror=no_mirror,
                            mirror_swap=mirror_swap,
                            fisheye_converter=None,
                        )

                    try:
                        proc = _read_preprocessed_frame_with_seek(
                            info=c,
                            frame_idx=end_frame_idx,
                            processor=processors[proc_key],
                        )
                    except Exception:
                        continue

                    mse = float(np.mean((proc.astype(np.float32) - zimg.astype(np.float32)) ** 2))
                    scored.append((mse, c.path, best_start_frame, end_frame_idx))

                if not scored:
                    failed = True
                    break

                scored.sort(key=lambda x: x[0])
                best_mse, best_path, best_start_frame, best_end_frame = scored[0]
                second = scored[1] if len(scored) > 1 else None

                if debug_dir_path is not None:
                    with debug_file.open("a") as f:
                        f.write(
                            json.dumps(
                                {
                                    "episode_index": ep_idx,
                                    "camera_index": cam_idx,
                                    "zarr_length": length,
                                    "best_path": best_path,
                                    "best_mse": best_mse,
                                    "best_start_frame": best_start_frame,
                                    "best_end_frame": best_end_frame,
                                    "second_path": second[1] if second else None,
                                    "second_mse": second[0] if second else None,
                                }
                            )
                            + "\n"
                        )


                assigned[cam_idx] = (best_path, best_mse, best_start_frame)
                used_paths.add(best_path)

            if failed:
                dropped_eps.append(ep_idx)
                continue

            keep_eps.append(ep_idx)
            for cam_idx in camera_indices:
                p, mse, start_frame = assigned[cam_idx]
                matched_paths_by_cam[cam_idx].append(p)
                matched_mse_by_cam[cam_idx].append(mse)
                matched_start_by_cam[cam_idx].append(start_frame)

        print(f"Matched episodes: {len(keep_eps)} / {n_episodes}")
        print(f"Dropped episodes: {len(dropped_eps)}")

        extra_meta: Dict[str, np.ndarray] = {
            "num_source_episodes_total": np.asarray(n_episodes, dtype=np.int64),
            "num_source_episodes_kept": np.asarray(len(keep_eps), dtype=np.int64),
            "num_source_episodes_dropped": np.asarray(len(dropped_eps), dtype=np.int64),
        }
        if keep_eps:
            extra_meta["source_episode_idx"] = np.asarray(keep_eps, dtype=np.int64)

        for cam_idx in camera_indices:
            if matched_paths_by_cam[cam_idx]:
                extra_meta[f"matched_raw_video_paths_camera{cam_idx}"] = _as_unicode_array(matched_paths_by_cam[cam_idx])
                extra_meta[f"matched_raw_video_mse_camera{cam_idx}"] = np.asarray(matched_mse_by_cam[cam_idx], dtype=np.float32)
                extra_meta[f"matched_raw_video_start_frame_camera{cam_idx}"] = np.asarray(matched_start_by_cam[cam_idx], dtype=np.int64)

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
