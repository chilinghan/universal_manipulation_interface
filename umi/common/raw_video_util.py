import collections
import math
import os
import pathlib
from typing import Dict, Iterable, List, Optional, Set, Tuple, Union

import av
import numpy as np
from exiftool import ExifToolHelper

from umi.common.timecode_util import mp4_get_start_datetime

VIDEO_SUFFIXES: Set[str] = {".mp4", ".mov", ".mkv", ".avi"}
PathLike = Union[str, pathlib.Path]


def normalize_video_path(path: PathLike) -> str:
    return str(pathlib.Path(os.path.expanduser(str(path))).absolute())


def safe_relative_path(path: pathlib.Path, base: pathlib.Path) -> pathlib.Path:
    try:
        return path.relative_to(base)
    except ValueError:
        return pathlib.Path(path.name)


def resolve_video_paths(
    input_path: PathLike,
    glob_pattern: str = "**/*.mp4",
    valid_suffixes: Optional[Iterable[str]] = None,
) -> Tuple[List[pathlib.Path], pathlib.Path]:
    """Resolve video file paths from a root directory and glob pattern."""
    suffixes = {s.lower() for s in (valid_suffixes or VIDEO_SUFFIXES)}
    root = pathlib.Path(os.path.expanduser(str(input_path))).absolute()

    patterns = [glob_pattern]
    if ".mp4" in glob_pattern:
        patterns.append(glob_pattern.replace(".mp4", ".MP4"))

    paths: List[pathlib.Path] = []
    for pat in patterns:
        for v in root.glob(pat):
            if v.is_file() and v.suffix.lower() in suffixes:
                paths.append(v.absolute())

    resolved = sorted(set(paths))
    if not resolved:
        raise FileNotFoundError(f"No videos found for {root} with glob {glob_pattern}")
    return resolved, root


def count_video_frames(path: str) -> Tuple[int, int, int]:
    """Return (n_frames, width, height), decoding if frame count is unavailable."""
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


def _collect_alignment_windows_single_group(group_rows: List[dict]) -> Dict[pathlib.Path, Tuple[int, int]]:
    """Stage-06 alignment logic on one session group of videos."""
    serial_count = collections.Counter([row["camera_serial"] for row in group_rows])
    n_cameras = len(serial_count)
    if n_cameras == 0:
        return {}

    events = []
    for vid_idx, row in enumerate(group_rows):
        events.append(
            {
                "vid_idx": vid_idx,
                "camera_serial": row["camera_serial"],
                "t": row["start_timestamp"],
                "is_start": True,
            }
        )
        events.append(
            {
                "vid_idx": vid_idx,
                "camera_serial": row["camera_serial"],
                "t": row["end_timestamp"],
                "is_start": False,
            }
        )
    events = sorted(events, key=lambda x: x["t"])

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

        assert len(on_videos) == len(on_cameras)

        if len(on_cameras) == n_cameras:
            t_demo_start = event["t"]
        elif t_demo_start is not None:
            assert not event['is_start']
            
            t_start = t_demo_start
            t_end = event['t']
            
            # undo state update to get full set of videos
            demo_vid_idxs = set(on_videos)
            demo_vid_idxs.add(event['vid_idx'])
            
            demo_groups.append({
                "video_idxs": sorted(demo_vid_idxs),
                "start_timestamp": t_start,
                "end_timestamp": t_end
            })
            t_demo_start = None

    frame_windows: Dict[pathlib.Path, Tuple[int, int]] = {}
    for group in demo_groups:
        selected = [group_rows[i] for i in group["video_idxs"]]
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
        align_cam_idx = np.argmin([sum(x) for x in alignment_costs])

        start_timestamp = float(group["start_timestamp"])
        end_timestamp = float(group["end_timestamp"])
        align_video_start = float(selected[align_cam_idx]["start_timestamp"])
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


def _group_key_from_glob(path: pathlib.Path, glob_pattern: Optional[str]) -> str:
    if not glob_pattern:
        return str(path.parent)
    depth = max(glob_pattern.count("**/") - 1, 0)
    parent = path.parent
    for _ in range(depth):
        if parent.parent == parent:
            break
        parent = parent.parent
    return str(parent)


def collect_alignment_windows(
    videos: List[pathlib.Path],
    glob_pattern: Optional[str] = None,
) -> Dict[pathlib.Path, Tuple[int, int]]:
    """Compute per-video alignment windows close to 06_generate_dataset_plan.py.

    Difference from stage-06 script:
    - Stage-06 runs per project/session directory.
    - Here we may receive many sessions at once (e.g. cup_in_the_wild root), so we
      run the same logic per parent folder and merge the results.
    """
    demo_videos = [p.absolute() for p in videos if p.suffix.lower() == ".mp4"]
    if not demo_videos:
        return {}

    rows = []
    with ExifToolHelper() as et:
        for mp4_path in demo_videos:
            try:
                meta = list(et.get_metadata(str(mp4_path)))[0]
                cam_serial = str(meta["QuickTime:CameraSerialNumber"])
                start_date = mp4_get_start_datetime(str(mp4_path))
                start_timestamp = start_date.timestamp()
                with av.open(str(mp4_path), "r") as container:
                    stream = container.streams.video[0]
                    n_frames = int(stream.frames)
                    fps = stream.average_rate
                duration_sec = float(n_frames / fps)
            except Exception as exc:
                print(f"Skipping alignment window for {mp4_path}: {exc}")
                continue

            rows.append(
                {
                    "path": mp4_path,
                    "group_key": _group_key_from_glob(mp4_path, glob_pattern),
                    "camera_serial": cam_serial,
                    "start_timestamp": float(start_timestamp),
                    "end_timestamp": float(start_timestamp + duration_sec),
                    "fps": fps,
                }
            )

    if not rows:
        return {}

    rows_by_group = collections.defaultdict(list)
    for row in rows:
        rows_by_group[row["group_key"]].append(row)

    frame_windows: Dict[pathlib.Path, Tuple[int, int]] = {}
    for _, group_rows in rows_by_group.items():
        frame_windows.update(_collect_alignment_windows_single_group(group_rows))

    return frame_windows
