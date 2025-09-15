from typing import Any, Dict, Optional
from pydantic import BaseModel, Field


class CutVideoInput(BaseModel):
    file_path: str = Field(..., description="Absolute or container-local path to source video")
    output_dir: str = Field(..., description="Directory where clips/snaps will be written")
    scene_threshold: float = Field(0.25, description="ffmpeg scene-change sensitivity")
    min_play_seconds: float = Field(3, description="Ignore clips shorter than this")
    max_clips: int = Field(10, description="0 = no limit; otherwise stop after N clips")
    snap_at: float = Field(0.5, description="First snapshot (s) into each clip")
    snap_step: float = Field(0.4, description="Spacing (s) between snapshots")
    num_snap_frames: int = Field(3, description="Snapshots per clip")


class AnalyzeSnapInput(BaseModel):
    file_path: str = Field(..., description="Absolute or container-local path to source media")
    output_dir: str = Field(..., description="Directory where analysis/snap outputs will be written")


class JobStatus(BaseModel):
    job_id: str
    done: bool
    error: Optional[str] = None
    last_event: Optional[Dict[str, Any]] = None
