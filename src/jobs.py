"""Background analysis jobs for the web app.

Analysis used to run inside the POST handler, so the browser sat on a blocked
request with no progress and no way to cancel; a long video simply timed out.
Jobs now run on a small worker pool and the page polls for progress.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

from .camera_source import CameraError
from .config import PipelineConfig
from .recorder import IncidentRecord
from .security_camera import Progress, SmartSecurityCamera

LOGGER = logging.getLogger(__name__)

IncidentHook = Callable[[str, IncidentRecord], None]

QUEUED = "queued"
RUNNING = "running"
DONE = "done"
FAILED = "failed"
CANCELLED = "cancelled"


@dataclass
class Job:
    """One video analysis run."""

    job_id: str
    source_name: str
    source_path: Path
    output_dir: Path
    config: PipelineConfig
    status: str = QUEUED
    percent: int = 0
    frames_done: int = 0
    frames_total: int = 0
    fps: float = 0.0
    duration_s: float = 0.0
    """Length of the source, so the page can plot where incidents fall."""
    incidents: List[Dict[str, object]] = field(default_factory=list)
    error: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    finished_at: Optional[float] = None
    stop_event: threading.Event = field(default_factory=threading.Event)

    def to_dict(self) -> Dict[str, object]:
        return {
            "job_id": self.job_id,
            "status": self.status,
            "percent": self.percent,
            "frames_done": self.frames_done,
            "frames_total": self.frames_total,
            "fps": round(self.fps, 2),
            "duration_s": round(self.duration_s, 2),
            "incidents": self.incidents,
            "incidents_detected": len(self.incidents),
            "source_name": self.source_name,
            "sensitivity": self.config.sensitivity,
            "error": self.error,
            "elapsed_s": round(
                (self.finished_at or time.time()) - self.created_at, 1
            ),
        }


class JobManager:
    """Runs analysis jobs on a bounded pool and keeps their status."""

    def __init__(
        self,
        job_root: Path,
        max_workers: int = 2,
        history: int = 50,
        on_incident: Optional[IncidentHook] = None,
    ) -> None:
        self.job_root = Path(job_root)
        self.history = history
        self.on_incident = on_incident
        self._pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="analysis")
        self._jobs: Dict[str, Job] = {}
        self._lock = threading.Lock()

    # -- API ---------------------------------------------------------------
    def submit(self, source_path: Path, source_name: str, config: PipelineConfig) -> Job:
        job_id = uuid.uuid4().hex[:12]
        job = Job(
            job_id=job_id,
            source_name=source_name,
            source_path=Path(source_path),
            output_dir=self.job_root / job_id / "incidents",
            config=config,
        )
        with self._lock:
            self._jobs[job_id] = job
            self._trim()
        self._pool.submit(self._run, job)
        LOGGER.info("Job %s queued for %s", job_id, source_name)
        return job

    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(job_id)

    def cancel(self, job_id: str) -> bool:
        job = self.get(job_id)
        if job is None or job.status in {DONE, FAILED, CANCELLED}:
            return False
        job.stop_event.set()
        return True

    def list(self) -> List[Dict[str, object]]:
        with self._lock:
            jobs = sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)
        return [job.to_dict() for job in jobs]

    def shutdown(self) -> None:
        for job in list(self._jobs.values()):
            job.stop_event.set()
        self._pool.shutdown(wait=False)

    # -- internals ---------------------------------------------------------
    def _trim(self) -> None:
        if len(self._jobs) <= self.history:
            return
        finished = sorted(
            (j for j in self._jobs.values() if j.status in {DONE, FAILED, CANCELLED}),
            key=lambda j: j.finished_at or j.created_at,
        )
        for job in finished[: len(self._jobs) - self.history]:
            self._jobs.pop(job.job_id, None)

    def _run(self, job: Job) -> None:
        job.status = RUNNING
        camera = SmartSecurityCamera(
            job.source_path, job.output_dir, job.config, source_label=job.source_name
        )

        def report(progress: Progress) -> None:
            job.frames_done = progress.frames_done
            job.frames_total = progress.frames_total
            job.percent = progress.percent

        try:
            records = camera.process(progress=report, stop_event=job.stop_event)
            if camera.pipeline is not None:
                job.fps = camera.pipeline.fps
                frames = job.frames_total or camera.pipeline.frames_processed
                job.duration_s = frames / job.fps if job.fps else 0.0
            for record in records:
                if self.on_incident is not None:
                    self.on_incident(job.job_id, record)
                job.incidents.append(
                    {
                        "index": record.index,
                        "filename": record.filename,
                        "start_time_s": record.start_time_s,
                        "end_time_s": record.end_time_s,
                        "duration_s": record.duration_s,
                        "max_objects": record.max_objects,
                        "media_url": f"/media/{job.job_id}/{record.filename}",
                        "view_url": f"/view/{job.job_id}/{record.filename}",
                        "thumbnail_url": (
                            f"/media/{job.job_id}/{record.thumbnail}"
                            if record.thumbnail
                            else None
                        ),
                    }
                )
            job.status = CANCELLED if job.stop_event.is_set() else DONE
            job.percent = 100 if job.status == DONE else job.percent
        except CameraError as exc:
            job.status = FAILED
            job.error = str(exc)
            LOGGER.warning("Job %s failed: %s", job.job_id, exc)
        except Exception as exc:  # pragma: no cover - defensive
            job.status = FAILED
            job.error = f"Unexpected error: {exc}"
            LOGGER.exception("Job %s crashed", job.job_id)
        finally:
            job.finished_at = time.time()
            self._cleanup_source(job)

    @staticmethod
    def _cleanup_source(job: Job) -> None:
        """Delete the uploaded/downloaded source once analysis is finished."""
        try:
            path = job.source_path
            if path.exists():
                path.unlink()
            parent = path.parent
            if parent.exists() and parent.name.startswith(("upload_", "download_")):
                if not any(parent.iterdir()):
                    parent.rmdir()
        except OSError:  # pragma: no cover - best effort
            LOGGER.debug("Could not clean up %s", job.source_path)
