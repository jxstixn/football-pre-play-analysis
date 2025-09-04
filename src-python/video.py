import os
import asyncio
import re
from typing import Callable, List, Tuple, Awaitable

from pydantic import BaseModel

# ---- Parameters container (shared with request model but decoupled here) ----
class Params(BaseModel):
    file_path: str
    output_dir: str
    scene_threshold: float = 0.25
    min_play_seconds: float = 3.0
    max_clips: int = 0
    snap_at: float = 0.5
    snap_step: float = 0.4
    num_snap_frames: int = 5

# ----------------------- Async helpers for ffmpeg ----------------------
async def run_cmd(*cmd: str, stdout=None, stderr=None):
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=stdout or asyncio.subprocess.PIPE,
        stderr=stderr or asyncio.subprocess.PIPE,
    )
    return proc

async def ffprobe_duration(path: str) -> float:
    proc = await run_cmd(
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=nk=1:nw=1", path,
    )
    out, _ = await proc.communicate()
    return float(out.decode().strip())

async def iter_showinfo_scene_changes(path: str, thr: float):
    """
    Yields scene-cut times (seconds) by parsing ffmpeg showinfo output incrementally.
    """
    cmd = [
        "ffmpeg", "-v", "info", "-i", path,
        "-vf", f"select='gt(scene,{thr})',showinfo", "-f", "null", "-",
    ]
    proc = await run_cmd(*cmd)
    cuts: List[float] = []
    pattern = re.compile(r"pts_time:(\d+\.\d+)")
    assert proc.stderr is not None
    async for raw in proc.stderr:  # type: ignore
        line = raw.decode(errors="ignore")
        m = pattern.search(line)
        if m:
            t = float(m.group(1))
            cuts.append(t)
            yield ("cut", t)
    await proc.wait()
    yield ("done", cuts)

async def ffmpeg_cut_with_progress(src: str, start: float, end: float, out_path: str):
    """
    Cut [start, end) using stream copy and parse -progress output for percent.
    Yields percent (0..1).
    """
    dur = max(0.000001, end - start)
    # -nostats keeps output clean; -progress pipe:1 prints key=value lines
    cmd = [
        "ffmpeg", "-y", "-nostdin",
        "-ss", str(start), "-t", str(dur), "-i", src,
        "-c", "copy",
        "-v", "error",
        "-progress", "pipe:1",
        out_path,
    ]
    proc = await run_cmd(*cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    assert proc.stdout is not None
    async for raw in proc.stdout:  # type: ignore
        line = raw.decode(errors="ignore").strip()
        if line.startswith("out_time_ms="):
            ms = int(line.split("=", 1)[1])
            pct = min(1.0, ms / (dur * 1_000_000.0))  # <-- fixed scaling
            yield pct
    await proc.wait()
    yield 1.0

async def ffmpeg_snap_frames(clip_path: str, snaps_dir: str, first_at: float, step: float, count: int):
    base = os.path.splitext(os.path.basename(clip_path))[0]
    for i in range(count):
        t = first_at + i * step
        out_jpg = os.path.join(snaps_dir, f"{base}_snap{i+1}.jpg")
        proc = await run_cmd(
            "ffmpeg", "-y", "-nostdin", "-ss", f"{t}", "-i", clip_path,
            "-frames:v", "1", out_jpg,
        )
        await proc.wait()
        yield i + 1  # how many produced so far

# --------------------------- Orchestration -----------------------------
async def process_video_job(job_id: str, params: Params, emit: Callable[[dict], Awaitable[None]]):
    """
    Orchestrates the whole pipeline and emits structured progress events.
    """
    src = params.file_path
    out_dir = params.output_dir
    plays_dir = os.path.join(out_dir, "plays")
    snaps_dir = os.path.join(out_dir, "snaps")
    os.makedirs(plays_dir, exist_ok=True)
    os.makedirs(snaps_dir, exist_ok=True)

    duration = await ffprobe_duration(src)
    await emit({"type": "meta", "duration": duration})

    # ---- Stage 1: Scene detection ----
    cuts: List[float] = []
    async for kind, val in iter_showinfo_scene_changes(src, params.scene_threshold):
        if kind == "cut":
            t = float(val)
            cuts.append(t)
            await emit({
                "type": "scene_detect",
                "found": len(cuts),
                "last_time": t,
                "approx_percent": round(min(1.0, t / max(0.000001, duration)), 4)
            })
        elif kind == "done":
            # finalize list from generator
            cuts = list(val)

    times = [0.0] + cuts + [duration]

    # Build clip ranges and filter by min duration / max clips
    ranges: List[Tuple[float, float]] = []
    for s, e in zip(times[:-1], times[1:]):
        if (e - s) >= params.min_play_seconds:
            ranges.append((s, e))
    if params.max_clips and params.max_clips > 0:
        ranges = ranges[: params.max_clips]

    await emit({"type": "clips_planned", "count": len(ranges)})

    # ---- Stage 2: Cut clips with progress ----
    total_clips = max(1, len(ranges))
    completed_clips = 0

    for idx, (s, e) in enumerate(ranges, start=1):
        out_path = os.path.join(plays_dir, f"play_{idx:03}.mp4")
        await emit({
            "type": "clip_start",
            "index": idx,
            "of": len(ranges),
            "start": round(s, 3),
            "end": round(e, 3),
            "path": out_path,
        })
        async for pct in ffmpeg_cut_with_progress(src, s, e, out_path):
            # overall approx: 10% (scene) + 80% (cuts) + 10% (snaps)
            overall = 0.10 + 0.80 * (((idx - 1) + pct) / total_clips)
            await emit({
                "type": "clip_progress",
                "index": idx,
                "of": len(ranges),
                "percent": round(pct, 4),
                "overall_estimate": round(min(0.99, overall), 4),
            })
        completed_clips += 1
        await emit({"type": "clip_done", "index": idx})

        # ---- Stage 3: snapshots for this clip ----
        produced = 0
        async for produced in ffmpeg_snap_frames(out_path, snaps_dir, params.snap_at, params.snap_step, params.num_snap_frames):
            # overall approx includes last 10% for snapshots proportionally across clips
            snaps_overall = 0.10 * ((completed_clips - 1) / total_clips + (produced / max(1, params.num_snap_frames)) / total_clips)
            await emit({
                "type": "snap_progress",
                "index": idx,
                "produced": produced,
                "of": params.num_snap_frames,
                "overall_estimate": round(min(0.999, 0.90 + snaps_overall), 4),
            })
        await emit({"type": "snap_done", "index": idx, "produced": produced})

    await emit({
        "type": "summary",
        "clips": len(ranges),
        "snaps": len(ranges) * params.num_snap_frames,
        "plays_dir": plays_dir,
        "snaps_dir": snaps_dir,
        "overall_estimate": 1.0,
    })
