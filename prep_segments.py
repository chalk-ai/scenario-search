#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy==2.3.3", "pyarrow==21.0.0"]
# ///
"""Cut each clip into 10-second segments, transcode them for browsers, and extract frames.

    ./prep_segments.py --raw raw/ --out out/      # needs ffmpeg and ffprobe on PATH

For clip `factory051_worker003_00017.mp4` and start 60 s this writes, under out/:
  clips/w003_c017_t060.mp4         640 px h264, 24 fps, faststart (browsers cannot play the source h265)
  thumbs/w003_c017_t060.jpg        320 px middle frame
  frames/w003_c017_t060_<k>.jpg    four 384 px frames at 1.25/3.75/6.25/8.75 s, embedded by the model
and one row of data/segments_meta.parquet: worker, clip, start, duration and a camera-motion score.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

SEGMENT_S = 10
STARTS = (0, 30, 60, 90, 120, 150)
FRAME_OFFSETS = (1.25, 3.75, 6.25, 8.75)


def run(cmd: list[str]) -> bytes:
    return subprocess.run(cmd, check=True, capture_output=True).stdout


def duration(src: Path) -> float:
    out = run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(src)])
    return float(out.strip() or 0)


def segment(job: tuple[Path, int], out: Path) -> dict[str, object]:
    src, start = job
    _, worker, clip = src.stem.split("_")
    seg_id = f"w{worker[-3:]}_c{int(clip):03d}_t{start:03d}"
    mp4 = out / "clips" / f"{seg_id}.mp4"
    if not mp4.exists():
        run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-ss",
                str(start),
                "-t",
                str(SEGMENT_S),
                "-i",
                str(src),
                "-vf",
                "scale=640:-2,fps=24",
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "30",
                "-pix_fmt",
                "yuv420p",
                "-an",
                "-movflags",
                "+faststart",
                str(mp4),
            ]
        )
    thumb = out / "thumbs" / f"{seg_id}.jpg"
    if not thumb.exists():
        run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-ss",
                str(SEGMENT_S / 2),
                "-i",
                str(mp4),
                "-frames:v",
                "1",
                "-vf",
                "scale=320:-2",
                "-q:v",
                "4",
                str(thumb),
            ]
        )
    for k, off in enumerate(FRAME_OFFSETS):
        frame = out / "frames" / f"{seg_id}_{k}.jpg"
        if not frame.exists():
            run(
                [
                    "ffmpeg",
                    "-y",
                    "-loglevel",
                    "error",
                    "-ss",
                    str(start + off),
                    "-i",
                    str(src),
                    "-frames:v",
                    "1",
                    "-vf",
                    "scale=-2:384",
                    "-q:v",
                    "3",
                    str(frame),
                ]
            )
    # Camera motion: mean absolute difference between consecutive 64x36 grayscale frames at 4 fps.
    raw = run(
        ["ffmpeg", "-loglevel", "error", "-i", str(mp4), "-vf", "fps=4,scale=64:36,format=gray", "-f", "rawvideo", "-"]
    )
    frames = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 36, 64).astype(np.float32)
    motion = float(np.abs(np.diff(frames, axis=0)).mean()) if len(frames) > 1 else 0.0
    return {
        "segment_id": seg_id,
        "worker_id": f"worker_{worker[-3:]}",
        "clip_index": int(clip),
        "start_s": start,
        "duration_s": SEGMENT_S,
        "motion": round(motion, 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, default=Path("raw"))
    parser.add_argument("--out", type=Path, default=Path("out"))
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    for tool in ("ffmpeg", "ffprobe"):
        if shutil.which(tool) is None:
            raise SystemExit(f"{tool} is not on PATH")
    for d in ("clips", "thumbs", "frames", "data"):
        (args.out / d).mkdir(parents=True, exist_ok=True)
    sources = sorted(args.raw.glob("factory*_worker*_*.mp4"))
    with ProcessPoolExecutor(args.workers) as pool:
        durations = dict(zip(sources, pool.map(duration, sources)))
        # Some source clips run short of 3 minutes; only whole 10-second segments are kept.
        jobs = [(src, s) for src in sources for s in STARTS if s + SEGMENT_S <= durations[src] - 0.5]
        rows = list(pool.map(partial(segment, out=args.out), jobs, chunksize=4))
    rows.sort(key=lambda r: str(r["segment_id"]))
    table = pa.table(
        {
            "segment_id": [r["segment_id"] for r in rows],
            "worker_id": [r["worker_id"] for r in rows],
            "clip_index": pa.array([r["clip_index"] for r in rows], pa.int32()),
            "start_s": pa.array([r["start_s"] for r in rows], pa.int32()),
            "duration_s": pa.array([r["duration_s"] for r in rows], pa.int32()),
            "motion": pa.array([r["motion"] for r in rows], pa.float64()),
        }
    )
    pq.write_table(table, args.out / "data" / "segments_meta.parquet")
    print(f"{len(rows)} segments from {len(sources)} clips")


if __name__ == "__main__":
    main()
