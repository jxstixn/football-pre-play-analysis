import os
import time
import uuid
import asyncio
from typing import Dict, Any, List, Optional, Set

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Body
from pydantic import BaseModel, Field
from uvicorn import Config, Server
import signal, threading

from video import process_video_job

# ---------------------------- Config ---------------------------------
API_PORT = int(os.getenv("PORT_API", "8008"))
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",")

# ---------------------------- App ------------------------------------
app = FastAPI(title="Video Cutter API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ------------------------- Job manager --------------------------------
class Job:
    def __init__(self, job_id: str):
        self.id = job_id
        self.subscribers: Set[WebSocket] = set()
        self.last_event: Optional[Dict[str, Any]] = None
        self.done: bool = False
        self.error: Optional[str] = None
        self.task: Optional[asyncio.Task] = None


class JobManager:
    def __init__(self):
        self.jobs: Dict[str, Job] = {}
        self.lock = asyncio.Lock()

    async def create(self) -> Job:
        async with self.lock:
            job_id = str(uuid.uuid4())
            job = Job(job_id)
            self.jobs[job_id] = job
            return job

    def get(self, job_id: str) -> Optional[Job]:
        return self.jobs.get(job_id)

    async def remove(self, job_id: str) -> None:
        async with self.lock:
            self.jobs.pop(job_id, None)

    async def broadcast(self, job_id: str, event: Dict[str, Any]):
        job = self.get(job_id)
        if not job:
            return
        event = {"job_id": job_id, **event}
        job.last_event = event
        dead: List[WebSocket] = []
        for ws in list(job.subscribers):
            try:
                await ws.send_json(event)
            except Exception:
                dead.append(ws)
        for ws in dead:
            try:
                job.subscribers.remove(ws)
            except KeyError:
                pass


job_manager = JobManager()


# --------------------------- Schemas ----------------------------------
class CutVideoInput(BaseModel):
    file_path: str = Field(..., description="Absolute or container-local path to source video")
    output_dir: str = Field(..., description="Directory where clips/snaps will be written")
    scene_threshold: float = Field(0.25, description="ffmpeg scene-change sensitivity")
    min_play_seconds: float = Field(3, description="Ignore clips shorter than this")
    max_clips: int = Field(10, description="0 = no limit; otherwise stop after N clips")
    snap_at: float = Field(0.5, description="First snapshot (s) into each clip")
    snap_step: float = Field(0.4, description="Spacing (s) between snapshots")
    num_snap_frames: int = Field(3, description="Snapshots per clip")


class JobStatus(BaseModel):
    job_id: str
    done: bool
    error: Optional[str] = None
    last_event: Optional[Dict[str, Any]] = None


# ---------------------------- REST ------------------------------------
@app.post("/v1/cut_video", response_model=JobStatus)
async def cut_video(payload: CutVideoInput = Body(...)):
    # Validate paths early
    if not os.path.exists(payload.file_path):
        raise HTTPException(status_code=400, detail=f"File not found: {payload.file_path}")
    os.makedirs(payload.output_dir, exist_ok=True)

    job = await job_manager.create()

    async def run():
        try:
            await job_manager.broadcast(job.id, {"type": "started"})
            await process_video_job(
                job_id=job.id,
                params=payload,
                emit=lambda ev: job_manager.broadcast(job.id, ev),
            )
            await job_manager.broadcast(job.id, {"type": "completed"})
            job.done = True
        except asyncio.CancelledError:
            await job_manager.broadcast(job.id, {"type": "cancelled"})
            job.done = True
            raise
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            job.error = str(e) or tb
            await job_manager.broadcast(job.id, {"type": "error", "message": (str(e) or tb)})
            job.done = True

    job.task = asyncio.create_task(run())

    return JobStatus(job_id=job.id, done=False, error=None, last_event=None)


@app.get("/v1/cut_video/{job_id}", response_model=JobStatus)
async def get_job(job_id: str):
    job = job_manager.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return JobStatus(job_id=job.id, done=job.done, error=job.error, last_event=job.last_event)


@app.post("/v1/cut_video/{job_id}/cancel")
async def cancel_job(job_id: str):
    job = job_manager.get(job_id)
    if not job or not job.task:
        raise HTTPException(status_code=404, detail="Job not found or not running")
    job.task.cancel()
    return {"ok": True}


# --------------------------- WebSocket --------------------------------
@app.websocket("/ws/progress/{job_id}")
async def ws_progress(websocket: WebSocket, job_id: str):
    # Basic origin check (optional; adjust for your deployment)
    await websocket.accept()
    job = job_manager.get(job_id)
    if not job:
        await websocket.send_json({"type": "error", "message": "Unknown job_id"})
        await websocket.close(code=4404)
        return

    job.subscribers.add(websocket)
    # Send last event so late subscribers see state
    if job.last_event:
        await websocket.send_json(job.last_event)

    try:
        while True:
            # Keep the connection alive; we don't expect client messages.
            await websocket.receive_text()
    except WebSocketDisconnect:
        # client left; that's fine
        try:
            job.subscribers.remove(websocket)
        except KeyError:
            pass


# ----------------------- Programmatic server control ------------------------
server_instance: Server | None = None
_server_thread: threading.Thread | None = None


def start_api_server(host: str = "0.0.0.0", port: int = API_PORT, log_level: str = "info") -> bool:
    """Start Uvicorn in a background thread (singleton). Returns True if started."""
    global server_instance, _server_thread
    if server_instance is not None and _server_thread and _server_thread.is_alive():
        print("[sidecar] Server already running", flush=True)
        return False

    def _run():
        try:
            config = Config(app, host=host, port=port, log_level=log_level)
            srv = Server(config)
            # expose globally so stop_api_server can signal it
            globals()["server_instance"] = srv
            asyncio.run(srv.serve())
        except Exception as e:
            print(f"[sidecar] Server failed: {e}", flush=True)
        finally:
            globals()["server_instance"] = None

    t = threading.Thread(target=_run, name="uvicorn-sidecar", daemon=True)
    _server_thread = t
    t.start()
    print(f"[sidecar] Starting API server on {host}:{port}...", flush=True)
    return True


def stop_api_server(force: bool = False) -> None:
    """Signal the server to exit; if force=True, sends SIGINT to the process."""
    global server_instance, _server_thread
    if force:
        try:
            os.kill(os.getpid(), signal.SIGINT)
        except Exception:
            pass
        return
    if server_instance is not None:
        server_instance.should_exit = True
    if _server_thread and _server_thread.is_alive():
        _server_thread.join(timeout=5)
    print("[sidecar] Server stopped", flush=True)


if __name__ == "__main__":
    # Running as a standalone script
    start_api_server(port=API_PORT)
    # Keep the main thread alive so the daemon thread can serve
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        stop_api_server()
