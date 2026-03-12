#!/usr/bin/env python3
"""Transcode matched videos: trim -> center crop -> resize."""

import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

import pathlib
import pickle
import re
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

import click
import numpy as np
import zarr
from tqdm import tqdm

from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs
from diffusion_policy.common.replay_buffer import ReplayBuffer
from umi.common.raw_video_util import normalize_video_path, resolve_video_paths, safe_relative_path

register_codecs()

MATCHED_PATHS_RE = re.compile(r"^matched_raw_video_paths_camera(\d+)$")


def _open_replay_buffer(zarr_path: pathlib.Path, mode: str):
    if zarr_path.suffix == ".zip":
        store = zarr.ZipStore(str(zarr_path), mode=mode)
        root = zarr.group(store=store)
        return store, ReplayBuffer.create_from_group(root)
    root = zarr.open(str(zarr_path), mode=mode)
    return None, ReplayBuffer.create_from_group(root)


def _load_trim_ranges_from_matched_zarr(match_zarr: pathlib.Path) -> Dict[str, Tuple[int, int]]:
    store, rb = _open_replay_buffer(match_zarr, mode="r")
    try:
        trim_by_path: Dict[str, Tuple[int, int]] = {}

        cameras = []
        for key in rb.meta.keys():
            m = MATCHED_PATHS_RE.match(key)
            if m:
                cameras.append(int(m.group(1)))
        cameras = sorted(cameras)
        if not cameras:
            raise click.ClickException("No matched_raw_video_paths_camera* keys found in --match-zarr metadata.")

        conflicts = []
        for cam_idx in cameras:
            path_key = f"matched_raw_video_paths_camera{cam_idx}"
            start_key = f"matched_raw_video_start_frame_camera{cam_idx}"
            n_key = f"matched_raw_video_n_frames_camera{cam_idx}"

            if start_key not in rb.meta or n_key not in rb.meta:
                raise click.ClickException(f"Missing {start_key} or {n_key} in --match-zarr metadata.")

            paths = rb.meta[path_key][:].tolist()
            starts = np.asarray(rb.meta[start_key][:], dtype=np.int64)
            n_frames = np.asarray(rb.meta[n_key][:], dtype=np.int64)
            if len(paths) != len(starts) or len(paths) != len(n_frames):
                raise click.ClickException(
                    f"Metadata length mismatch for camera {cam_idx}: "
                    f"paths={len(paths)}, starts={len(starts)}, n_frames={len(n_frames)}"
                )

            for raw_path, start, n in zip(paths, starts, n_frames):
                norm_path = normalize_video_path(raw_path)
                trim = (int(start), int(n))
                prev = trim_by_path.get(norm_path)
                if prev is None:
                    trim_by_path[norm_path] = trim
                elif prev != trim:
                    conflicts.append((norm_path, prev, trim))

        if conflicts:
            msg = "; ".join([f"{p}: {a} vs {b}" for p, a, b in conflicts[:5]])
            raise click.ClickException(
                "Conflicting trim ranges found for the same raw path in --match-zarr metadata. "
                f"Examples: {msg}"
            )

        return trim_by_path
    finally:
        if store is not None:
            store.close()


def _build_vf(crop_size: tuple, resize_size: tuple, start_frame: int = 0, n_frames: Optional[int] = None) -> str:
    filters = []
    if start_frame > 0 or n_frames is not None:
        trim_parts = [f"start_frame={start_frame}"]
        if n_frames is not None:
            trim_parts.append(f"end_frame={start_frame + n_frames}")
        filters.append(f"trim={':'.join(trim_parts)}")
        filters.append("setpts=PTS-STARTPTS")

    # Pipeline order: center-crop first, then resize.
    filters.append(f"crop={crop_size[0]}:{crop_size[1]}:(iw-{crop_size[0]})/2:(ih-{crop_size[1]})/2")
    filters.append(f"scale={resize_size[0]}:{resize_size[1]}:flags=lanczos")
    return ",".join(filters)


def build_ffmpeg_cmd_gpu(
    inp: pathlib.Path,
    out: pathlib.Path,
    crop_size: tuple,
    resize_size: tuple,
    start_frame: int = 0,
    n_frames: Optional[int] = None,
) -> List[str]:
    vf = _build_vf(crop_size=crop_size, resize_size=resize_size, start_frame=start_frame, n_frames=n_frames)
    return [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-hwaccel",
        "cuda",
        "-c:v",
        "hevc_cuvid",
        "-i",
        str(inp),
        "-vf",
        vf,
        "-c:v",
        "av1_nvenc",
        "-cq",
        "30",
        "-b:v",
        "0",
        "-preset",
        "p5",
        "-pix_fmt",
        "yuv420p",
        "-an",
        str(out),
    ]


def build_ffmpeg_cmd_cpu(
    inp: pathlib.Path,
    out: pathlib.Path,
    crop_size: tuple,
    resize_size: tuple,
    start_frame: int = 0,
    n_frames: Optional[int] = None,
) -> List[str]:
    vf = _build_vf(crop_size=crop_size, resize_size=resize_size, start_frame=start_frame, n_frames=n_frames)
    return [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(inp),
        "-vf",
        vf,
        "-c:v",
        "libx264",
        "-crf",
        "18",
        "-preset",
        "veryfast",
        "-pix_fmt",
        "yuv420p",
        "-an",
        str(out),
    ]


def _transform_corners(corners: np.ndarray, in_w: int, in_h: int, crop_size: tuple, resize_size: tuple) -> Optional[np.ndarray]:
    crop_x0 = (in_w - crop_size[0]) / 2.0
    crop_y0 = (in_h - crop_size[1]) / 2.0
    sx = float(resize_size[0]) / float(crop_size[0])
    sy = float(resize_size[1]) / float(crop_size[1])

    pts = np.asarray(corners, dtype=np.float32).reshape(-1, 2).copy()
    pts[:, 0] = (pts[:, 0] - crop_x0) * sx
    pts[:, 1] = (pts[:, 1] - crop_y0) * sy

    in_x = np.logical_and(pts[:, 0] >= 0.0, pts[:, 0] < float(resize_size[0]))
    in_y = np.logical_and(pts[:, 1] >= 0.0, pts[:, 1] < float(resize_size[1]))
    if not bool(np.all(np.logical_and(in_x, in_y))):
        return None
    return pts


def _rescale_tag_detection_pkl(
    src_pkl: pathlib.Path,
    dst_pkl: pathlib.Path,
    in_w: int,
    in_h: int,
    crop_size: tuple,
    resize_size: tuple,
    start_frame: int,
    n_frames: int,
):
    dets = pickle.load(src_pkl.open("rb"))
    if not isinstance(dets, list):
        shutil.copy2(src_pkl, dst_pkl)
        return

    dets = dets[start_frame : start_frame + n_frames]

    out_dets = []
    for frame in dets:
        if not isinstance(frame, dict):
            out_dets.append(frame)
            continue
        out_frame = dict(frame)
        tag_dict = frame.get("tag_dict", {})
        if not isinstance(tag_dict, dict):
            out_dets.append(out_frame)
            continue

        out_tag_dict = {}
        for tag_id, tag in tag_dict.items():
            if not isinstance(tag, dict):
                continue
            corners = tag.get("corners", None)
            if corners is None:
                continue
            mapped = _transform_corners(corners, in_w=in_w, in_h=in_h, crop_size=crop_size, resize_size=resize_size)
            if mapped is None:
                continue
            out_tag = dict(tag)
            out_tag["corners"] = mapped
            out_tag_dict[tag_id] = out_tag
        out_frame["tag_dict"] = out_tag_dict
        out_dets.append(out_frame)

    with dst_pkl.open("wb") as f:
        pickle.dump(out_dets, f)


@click.command()
@click.option("--input-dir", required=True, type=str, help="Root directory containing raw videos.")
@click.option("--output-dir", required=True, type=str, help="Directory to write transcoded videos.")
@click.option("--match-zarr", required=True, type=str, help="Matched zarr with matched_raw_video_* metadata.")
@click.option("--video-glob", default="**/*.mp4", show_default=True, type=str, help="Glob pattern under --input-dir.")
@click.option("--num-workers", default=8, show_default=True, type=int)
@click.option("--crop-size", default=(2028, 2028), show_default=True, type=(int, int))
@click.option("--resize-size", default=(1920, 1920), show_default=True, type=(int, int))
@click.option("--cpu-only", is_flag=True, default=False, help="Use software transcoding (libx264) only")
def main(
    input_dir: str,
    output_dir: str,
    match_zarr: str,
    video_glob: str,
    num_workers: int,
    crop_size: tuple,
    resize_size: tuple,
    cpu_only: bool,
):
    out_dir = pathlib.Path(os.path.expanduser(output_dir)).absolute()
    out_dir.mkdir(parents=True, exist_ok=True)
    if any(out_dir.iterdir()):
        click.confirm(f"Output directory {out_dir} is not empty. Overwrite existing outputs?", abort=True)

    try:
        videos, input_root = resolve_video_paths(input_dir, glob_pattern=video_glob)
    except FileNotFoundError as exc:
        raise click.ClickException(str(exc))

    match_zarr_path = pathlib.Path(os.path.expanduser(match_zarr)).absolute()
    trim_by_path = _load_trim_ranges_from_matched_zarr(match_zarr_path)
    print(f"Loaded trim metadata from {match_zarr_path}")
    print(f"Trim ranges for {len(trim_by_path)} raw videos")

    jobs = []
    tag_pkl_jobs = []
    skipped_unmatched = 0

    for inp in videos:
        trim = trim_by_path.get(normalize_video_path(inp))
        if trim is None:
            skipped_unmatched += 1
            continue

        start_frame, n_frames = trim
        rel = safe_relative_path(inp, input_root)
        out = (out_dir / rel).with_suffix(".mp4")
        out.parent.mkdir(parents=True, exist_ok=True)
        jobs.append((inp, out, start_frame, n_frames))

        src_pkl = inp.parent / "tag_detection.pkl"
        dst_pkl = out.parent / "tag_detection.pkl"
        if src_pkl.is_file():
            tag_pkl_jobs.append((src_pkl, dst_pkl, inp, start_frame, n_frames))

    print(f"Resolved {len(videos)} videos from {input_dir}")
    print(f"Skipped {skipped_unmatched} videos without match metadata")
    print(f"Transcoding {len(jobs)} videos with {num_workers} workers")
    if not jobs:
        raise click.ClickException("No transcode jobs were created.")

    def run_one(inp: pathlib.Path, out: pathlib.Path, start_frame: int, n_frames: int):
        if not cpu_only:
            gpu_cmd = build_ffmpeg_cmd_gpu(inp, out, crop_size, resize_size, start_frame=start_frame, n_frames=n_frames)
            proc = subprocess.run(gpu_cmd, capture_output=True, text=True)
            if proc.returncode == 0:
                return inp, out, 0, ""
            gpu_err = proc.stderr.strip()
        else:
            gpu_err = ""

        cpu_cmd = build_ffmpeg_cmd_cpu(inp, out, crop_size, resize_size, start_frame=start_frame, n_frames=n_frames)
        proc = subprocess.run(cpu_cmd, capture_output=True, text=True)
        if proc.returncode == 0:
            return inp, out, 0, gpu_err

        err = proc.stderr.strip()
        if gpu_err:
            err = f"GPU attempt failed:\n{gpu_err}\nCPU fallback failed:\n{err}"
        return inp, out, proc.returncode, err

    n_ok = 0
    failures = []
    successful_inputs = set()

    with ThreadPoolExecutor(max_workers=num_workers) as ex:
        futs = [ex.submit(run_one, inp, out, start_frame, n_frames) for inp, out, start_frame, n_frames in jobs]
        with tqdm(total=len(futs), desc="Transcoding videos") as pbar:
            for fut in as_completed(futs):
                inp, out, code, err = fut.result()
                if code == 0:
                    n_ok += 1
                    successful_inputs.add(str(inp))
                else:
                    failures.append((str(inp), str(out), err))
                pbar.update(1)

    print(f"Done: {n_ok}/{len(jobs)} succeeded")

    rescaled_pkls = 0
    for src_pkl, dst_pkl, inp, start_frame, n_frames in tqdm(tag_pkl_jobs, desc="Rescaling tag_detection.pkl"):
        if str(inp) not in successful_inputs:
            continue

        probe_cmd = [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-of",
            "csv=p=0:s=x",
            str(inp),
        ]
        probe = subprocess.run(probe_cmd, capture_output=True, text=True)
        if probe.returncode != 0:
            raise click.ClickException(f"ffprobe failed for {inp}: {probe.stderr.strip()}")

        wh = probe.stdout.strip().split("x")
        if len(wh) != 2:
            raise click.ClickException(f"Unexpected ffprobe size output for {inp}: {probe.stdout.strip()}")
        in_w, in_h = int(wh[0]), int(wh[1])

        _rescale_tag_detection_pkl(
            src_pkl,
            dst_pkl,
            in_w=in_w,
            in_h=in_h,
            crop_size=crop_size,
            resize_size=resize_size,
            start_frame=start_frame,
            n_frames=n_frames,
        )
        rescaled_pkls += 1

    print(f"Wrote transformed tag_detection.pkl files: {rescaled_pkls}")

    if failures:
        print(f"Failures: {len(failures)}")
        for i, (inp, out, err) in enumerate(failures[:20]):
            print(f"[{i}] {inp} -> {out}")
            print(err)
        raise click.ClickException("Some transcodes failed")


if __name__ == "__main__":
    try:
        main()
    finally:
        os.system("stty sane")
