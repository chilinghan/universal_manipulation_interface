#!/usr/bin/env python3
"""Match raw videos to processed zarr episodes using aligned-window boundaries."""

import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

import json
import pathlib
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import av
import click
import numpy as np
import zarr
from tqdm import tqdm
import cv2

from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs
from umi.common.cv_util import make_umi_image_processor
from umi.common.raw_video_util import collect_alignment_windows, count_video_frames, normalize_video_path, resolve_video_paths

register_codecs()
av.logging.set_level(av.logging.ERROR)

@dataclass
class RawVideoInfo:
    path: str
    width: int
    height: int
    window_start_frame: int
    window_n_frames: int


@dataclass
class CandidateScore:
    mean_mse: float
    path: str
    start_frame: int
    n_frames: int
    start_mse: float
    end_mse: float


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


def _prepare_debug_file(debug_dir: Optional[str]) -> Optional[pathlib.Path]:
    if not debug_dir:
        return None

    debug_dir_path = pathlib.Path(os.path.expanduser(debug_dir)).absolute()
    debug_dir_path.mkdir(parents=True, exist_ok=True)
    debug_file = debug_dir_path / "match_debug.jsonl"
    if debug_file.exists():
        debug_file.unlink()
    return debug_file


def _append_debug_row(debug_file: Optional[pathlib.Path], row: Dict) -> None:
    if debug_file is None:
        return
    with debug_file.open("a") as f:
        f.write(json.dumps(row) + "\n")


def _to_debug_image(img: np.ndarray) -> np.ndarray:
    arr = np.asarray(img)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 3 and arr.shape[2] == 3:
        arr = arr[:, :, ::-1]
    return arr


def _write_debug_image(debug_dir: Optional[pathlib.Path], name: str, img: np.ndarray) -> None:
    if debug_dir is None:
        return
    out_path = debug_dir / name
    cv2.imwrite(str(out_path), _to_debug_image(img))


def _collect_raw_infos(
    raw_paths: List[pathlib.Path],
    video_glob: str,
) -> Tuple[List[RawVideoInfo], int]:
    frame_windows = collect_alignment_windows(raw_paths, glob_pattern=video_glob)
    if not frame_windows:
        raise click.ClickException(
            "No alignment windows resolved. The aligned-window matcher requires "
            "dataset-plan style window metadata."
        )

    raw_infos: List[RawVideoInfo] = []
    skipped_no_window = 0
    for path in raw_paths:
        window = frame_windows.get(path)
        if window is None:
            skipped_no_window += 1
            continue

        _, width, height = count_video_frames(str(path))
        window_start_frame, window_n_frames = window
        if window_n_frames <= 0:
            skipped_no_window += 1
            continue

        raw_infos.append(
            RawVideoInfo(
                path=normalize_video_path(path),
                width=int(width),
                height=int(height),
                window_start_frame=int(window_start_frame),
                window_n_frames=int(window_n_frames),
            )
        )

    return raw_infos, skipped_no_window


def _score_candidate(
    info: RawVideoInfo,
    processor,
    zimg_start: np.ndarray,
    zimg_end: np.ndarray,
    length: int,
    debug_dir: Optional[pathlib.Path] = None,
    debug_prefix: str = "",
) -> CandidateScore:
    if info.window_n_frames != length:
        raise ValueError(
            f"expected window_n_frames == {length}, got {info.window_n_frames} for {info.path}"
        )

    zimg_start_f = zimg_start.astype(np.float32)
    zimg_end_f = zimg_end.astype(np.float32)
    start_frame = info.window_start_frame
    end_frame = start_frame + length - 1

    raw_start = _read_preprocessed_frame(
        info=info,
        frame_idx=start_frame,
        processor=processor,
    ).astype(np.float32)
    raw_end = _read_preprocessed_frame(
        info=info,
        frame_idx=end_frame,
        processor=processor,
    ).astype(np.float32)
    if debug_dir is not None:
        prefix = debug_prefix or pathlib.Path(info.path).stem
        _write_debug_image(debug_dir, f"{prefix}_raw_start.png", raw_start)
        _write_debug_image(debug_dir, f"{prefix}_raw_end.png", raw_end)
        _write_debug_image(debug_dir, f"{prefix}_zarr_start.png", zimg_start)
        _write_debug_image(debug_dir, f"{prefix}_zarr_end.png", zimg_end)

    start_mse = float(np.mean((raw_start - zimg_start_f) ** 2))
    end_mse = float(np.mean((raw_end - zimg_end_f) ** 2))
    mean_mse = 0.5 * (start_mse + end_mse)
    return CandidateScore(
        mean_mse=mean_mse,
        path=info.path,
        start_frame=start_frame,
        n_frames=length,
        start_mse=start_mse,
        end_mse=end_mse,
    )


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
@click.option("--mse-threshold", default=500.0, show_default=True, type=float, help="Drop episodes whose best boundary mean MSE is at or above this threshold")
def main(
    zarr_path: str,
    raw_video_dir: str,
    output: str,
    video_glob: str,
    no_mirror: bool,
    mirror_swap: bool,
    debug_dir,
    mse_threshold: float,
):
    zarr_path = pathlib.Path(os.path.expanduser(zarr_path)).absolute()
    raw_video_dir = pathlib.Path(os.path.expanduser(raw_video_dir)).absolute()
    output = pathlib.Path(os.path.expanduser(output)).absolute()
    debug_file = _prepare_debug_file(debug_dir)
    debug_dir_path = debug_file.parent if debug_file is not None else None

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

        raw_infos, skipped_no_window = _collect_raw_infos(raw_paths, video_glob)
        print(
            "Using dataset-plan style alignment windows "
            f"(includes end-frame -1 behavior) for {len(raw_infos)} videos"
        )
        print(f"Candidate videos with usable alignment windows: {len(raw_infos)}")
        if skipped_no_window > 0:
            print(f"Skipped {skipped_no_window} videos without usable alignment windows")

        processors = {}

        keep_eps: List[int] = []
        dropped_eps: List[int] = []
        matched_paths_by_cam: Dict[int, List[str]] = {c: [] for c in camera_indices}
        matched_start_by_cam: Dict[int, List[int]] = {c: [] for c in camera_indices}
        matched_n_frames_by_cam: Dict[int, List[int]] = {c: [] for c in camera_indices}

        for ep_idx in tqdm(range(n_episodes), desc="Matching episodes"):
            length = int(ep_lengths[ep_idx])
            start_idx = int(zarr_starts[ep_idx])

            assigned = {}
            used_paths = set()
            failed = False

            candidates = [info for info in raw_infos if info.window_n_frames == length]
            if not candidates:
                dropped_eps.append(ep_idx)
                continue

            for cam_idx in camera_indices:
                key = f"camera{cam_idx}_rgb"
                zimg_start = rb.data[key][start_idx]
                zimg_end = rb.data[key][start_idx + max(length - 1, 0)]
                scored = []
                out_res = (int(zimg_end.shape[1]), int(zimg_end.shape[0]))
                for info in candidates:
                    if info.path in used_paths:
                        continue

                    proc_key = (info.width, info.height, out_res[0], out_res[1])
                    if proc_key not in processors:
                        processors[proc_key] = make_umi_image_processor(
                            in_res=(info.width, info.height),
                            out_res=out_res,
                            no_mirror=no_mirror,
                            mirror_swap=mirror_swap,
                            fisheye_converter=None,
                        )

                    try:
                        score = _score_candidate(
                            info=info,
                            processor=processors[proc_key],
                            zimg_start=zimg_start,
                            zimg_end=zimg_end,
                            length=length,
                            debug_dir=debug_dir_path,
                            debug_prefix=(
                                f"ep{ep_idx:04d}_cam{cam_idx}_"
                                f"{pathlib.Path(info.path).stem}_start{info.window_start_frame}"
                            ),
                        )
                    except Exception:
                        continue

                    scored.append(score)

                if not scored:
                    failed = True
                    break

                scored.sort(key=lambda x: x.mean_mse)
                best = scored[0]
                second = scored[1] if len(scored) > 1 else None
                if best.mean_mse >= mse_threshold:
                    failed = True
                    break

                _append_debug_row(
                    debug_file,
                    {
                        "episode_index": ep_idx,
                        "camera_index": cam_idx,
                        "zarr_length": length,
                        "best_path": best.path,
                        "best_mse": best.mean_mse,
                        "best_start_mse": best.start_mse,
                        "best_end_mse": best.end_mse,
                        "best_start_frame": best.start_frame,
                        "best_n_frames": best.n_frames,
                        "second_path": second.path if second else None,
                        "second_mse": second.mean_mse if second else None,
                    },
                ) 

                assigned[cam_idx] = (best.path, best.start_frame, best.n_frames)
                used_paths.add(best.path)

            if failed:
                dropped_eps.append(ep_idx)
                continue

            keep_eps.append(ep_idx)
            for cam_idx in camera_indices:
                p, start_frame, n_frames = assigned[cam_idx]
                matched_paths_by_cam[cam_idx].append(p)
                matched_start_by_cam[cam_idx].append(start_frame)
                matched_n_frames_by_cam[cam_idx].append(n_frames)

        print(f"Matched episodes: {len(keep_eps)} / {n_episodes}")
        print(f"Dropped episodes: {len(dropped_eps)}")

        extra_meta: Dict[str, np.ndarray] = {}
        for cam_idx in camera_indices:
            if matched_paths_by_cam[cam_idx]:
                extra_meta[f"matched_raw_video_paths_camera{cam_idx}"] = _as_unicode_array(matched_paths_by_cam[cam_idx])
                extra_meta[f"matched_raw_video_start_frame_camera{cam_idx}"] = np.asarray(matched_start_by_cam[cam_idx], dtype=np.int64)
                extra_meta[f"matched_raw_video_n_frames_camera{cam_idx}"] = np.asarray(matched_n_frames_by_cam[cam_idx], dtype=np.int64)

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
