#!/usr/bin/env python3
"""GPU transcode videos for LeRobot V3: center-crop to square then resize.

Default transform:
- input assumed 2704x2028
- center crop to 2028x2028
- resize to 1920x1920 with lanczos
- decode with hevc_cuvid
- encode with av1_nvenc
"""

import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

import pathlib
import subprocess
import shutil
import pickle
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

import av
import numpy as np
from tqdm import tqdm

import click
from exiftool import ExifToolHelper

from umi.common.timecode_util import mp4_get_start_datetime


def _build_vf(crop_size: tuple, resize_size: tuple, start_frame: int = 0, n_frames: Optional[int] = None) -> str:
    filters = []
    if start_frame > 0 or n_frames is not None:
        trim_parts = [f"start_frame={start_frame}"]
        if n_frames is not None:
            trim_parts.append(f"end_frame={start_frame + n_frames}")
        filters.append(f"trim={':'.join(trim_parts)}")
        filters.append("setpts=PTS-STARTPTS")
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
    cmd = [
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
    return cmd


def build_ffmpeg_cmd_cpu(
    inp: pathlib.Path,
    out: pathlib.Path,
    crop_size: tuple,
    resize_size: tuple,
    start_frame: int = 0,
    n_frames: Optional[int] = None,
) -> List[str]:
    vf = _build_vf(crop_size=crop_size, resize_size=resize_size, start_frame=start_frame, n_frames=n_frames)
    cmd = [
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
    return cmd


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
    start_frame: int = 0,
    n_frames: Optional[int] = None,
):
    dets = pickle.load(src_pkl.open("rb"))
    if not isinstance(dets, list):
        # Keep original structure for unexpected formats.
        shutil.copy2(src_pkl, dst_pkl)
        return

    end_frame = None if n_frames is None else (start_frame + n_frames)
    dets = dets[start_frame:end_frame]

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


def _collect_alignment_windows(videos: List[pathlib.Path]) -> Dict[pathlib.Path, Tuple[int, int]]:
    demo_videos = [p for p in videos if p.suffix.lower() == ".mp4"]
    if not demo_videos:
        return {}

    rows = []
    with ExifToolHelper() as et:
        for mp4_path in demo_videos:
            meta = list(et.get_metadata(str(mp4_path)))[0]
            cam_serial = meta["QuickTime:CameraSerialNumber"]
            start_date = mp4_get_start_datetime(str(mp4_path))
            start_timestamp = start_date.timestamp()
            with av.open(str(mp4_path), "r") as container:
                stream = container.streams.video[0]
                n_frames = int(stream.frames)
                fps = stream.average_rate
            duration_sec = float(n_frames / fps)
            rows.append(
                {
                    "path": mp4_path,
                    "camera_serial": cam_serial,
                    "start_timestamp": start_timestamp,
                    "end_timestamp": start_timestamp + duration_sec,
                    "fps": fps,
                }
            )

    if not rows:
        return {}

    video_meta = rows
    serials = sorted({row["camera_serial"] for row in video_meta})
    n_cameras = len(serials)
    events = []
    for idx, row in enumerate(video_meta):
        events.append({"vid_idx": idx, "camera_serial": row["camera_serial"], "t": row["start_timestamp"], "is_start": True})
        events.append({"vid_idx": idx, "camera_serial": row["camera_serial"], "t": row["end_timestamp"], "is_start": False})
    events.sort(key=lambda x: x["t"])

    demo_groups = []
    on_videos = set()
    on_cameras = set()
    t_demo_start = None
    for event in events:
        if event["is_start"]:
            on_videos.add(event["vid_idx"])
            on_cameras.add(event["camera_serial"])
        else:
            on_videos.remove(event["vid_idx"])
            on_cameras.remove(event["camera_serial"])

        if len(on_cameras) == n_cameras:
            t_demo_start = event["t"]
        elif t_demo_start is not None:
            demo_groups.append(
                {
                    "video_idxs": sorted(on_videos | {event["vid_idx"]}),
                    "start_timestamp": t_demo_start,
                    "end_timestamp": event["t"],
                }
            )
            t_demo_start = None

    frame_windows: Dict[pathlib.Path, Tuple[int, int]] = {}
    for group in demo_groups:
        selected = [video_meta[i] for i in group["video_idxs"]]
        selected.sort(key=lambda row: row["camera_serial"])
        dt = None
        alignment_costs = []
        for row in selected:
            dt = 1 / row["fps"]
            this_alignment_cost = []
            for other_row in selected:
                diff = other_row["start_timestamp"] - row["start_timestamp"]
                this_alignment_cost.append(diff % dt)
            alignment_costs.append(this_alignment_cost)
        align_cam_idx = int(np.argmin([sum(x) for x in alignment_costs]))

        start_timestamp = group["start_timestamp"]
        end_timestamp = group["end_timestamp"]
        align_video_start = selected[align_cam_idx]["start_timestamp"]
        start_timestamp += dt - ((start_timestamp - align_video_start) % dt)

        n_frames = int((end_timestamp - start_timestamp) / dt)
        cam_start_frame_idxs = []
        for row in selected:
            video_start_frame = math.ceil((start_timestamp - row["start_timestamp"]) / dt)
            video_n_frames = math.floor((row["end_timestamp"] - start_timestamp) / dt) - 1
            if video_start_frame < 0:
                video_n_frames += video_start_frame
                video_start_frame = 0
            cam_start_frame_idxs.append(video_start_frame)
            n_frames = min(n_frames, video_n_frames)

        if n_frames <= 0:
            continue

        for row, start_frame in zip(selected, cam_start_frame_idxs):
            frame_windows[row["path"]] = (int(start_frame), int(n_frames))

    return frame_windows


@click.command()
@click.option("--input-dir", required=True, type=str)
@click.option("--output-dir", required=True, type=str)
@click.option("--glob", "glob_pattern", default="**/*.mp4", show_default=True)
@click.option("--num-workers", default=8, show_default=True, type=int)
@click.option("--crop-size", default=(2028, 2028), show_default=True, type=(int, int))
@click.option("--resize-size", default=(1920, 1920), show_default=True, type=(int, int))
@click.option("--cpu-only", is_flag=True, default=False, help="Use software transcoding (libx264) only")
def main(input_dir: str, output_dir: str, glob_pattern: str, num_workers: int, crop_size: tuple, resize_size: tuple, cpu_only: bool):
    in_dir = pathlib.Path(os.path.expanduser(input_dir)).absolute()
    out_dir = pathlib.Path(os.path.expanduser(output_dir)).absolute()
    out_dir.mkdir(parents=True, exist_ok=True)
    if any(out_dir.iterdir()):
        click.confirm(f"Output directory {out_dir} is not empty. Overwrite existing outputs?", abort=True)

    glob_patterns = [glob_pattern]
    if ".mp4" in glob_pattern:
        glob_patterns.append(glob_pattern.replace(".mp4", ".MP4"))
    videos = []
    seen = set()
    for pat in glob_patterns:
        for p in in_dir.glob(pat):
            if p.is_file() and p not in seen:
                videos.append(p)
                seen.add(p)
    videos = sorted(videos)
    if not videos:
        raise click.ClickException(f"No videos found in {in_dir} with pattern {glob_pattern}")

    frame_windows = _collect_alignment_windows(videos)
    if frame_windows:
        print(f"Applying aligned frame windows to {len(frame_windows)} demo videos")

    jobs = []
    tag_pkl_jobs = []
    for inp in videos:
        rel = inp.relative_to(in_dir)
        out = out_dir / rel
        out = out.with_suffix(".mp4")
        out.parent.mkdir(parents=True, exist_ok=True)
        start_frame, n_frames = frame_windows.get(inp, (0, None))
        jobs.append((inp, out, start_frame, n_frames))
        src_pkl = inp.parent / "tag_detection.pkl"
        dst_pkl = out.parent / "tag_detection.pkl"
        if src_pkl.is_file():
            tag_pkl_jobs.append((src_pkl, dst_pkl, inp, start_frame, n_frames))

    print(f"Transcoding {len(jobs)} videos with {num_workers} workers")

    def run_one(inp: pathlib.Path, out: pathlib.Path, start_frame: int, n_frames: Optional[int]):
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
    for src_pkl, dst_pkl, inp, start_frame, n_frames in tqdm(tag_pkl_jobs, desc="Rescaling tag detection pkl files"):
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
