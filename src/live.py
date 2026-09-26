"""Live camera session management for the web app.

The first version kept the camera in a module-level dict, released it from the
HTTP handler while the worker thread was still reading from it, and streamed
frames from a ``while True`` loop that spun at 100% CPU whenever a frame was
not ready.  This module replaces that with one object that owns the thread,
the capture and the published frame:

* start/stop are serialised by a lock and stopping *waits* for the worker to
  exit, so the capture is only ever released by the thread that uses it;
* viewers block on a condition variable until a new frame is published, so an
  idle stream costs nothing and never sends the same frame twice;
* the same :class:`SecurityPipeline` used for files does the detection.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

import cv2
import numpy as np

from .camera_source import CameraError, VideoSource, open_camera
from .config import PipelineConfig
from .pipeline import SecurityPipeline
from .recorder import IncidentRecord

LOGGER = logging.getLogger(__name__)

IncidentHook = Callable[[str, IncidentRecord], None]


@dataclass
class LiveStats:
    """A snapshot of the session, safe to serialise to JSON."""

    active: bool = False
    job_id: Optional[str] = None
    device_index: int = 0
    backend: str = ""
    width: int = 0
    height: int = 0
    source_fps: float = 0.0
    measured_fps: float = 0.0
    frames: int = 0
    incidents: int = 0
    recording: bool = False
    alert: bool = False
    warming_up: bool = False
    tracked_objects: int = 0
    started_at: Optional[float] = None
    error: Optional[str] = None
    sensitivity: str = "balanced"

    @property
    def uptime_s(self) -> float:
        return 0.0 if self.started_at is None else max(0.0, time.time() - self.started_at)

    def to_dict(self) -> Dict[str, object]:
        data = {k: v for k, v in self.__dict__.items()}
        data["uptime_s"] = round(self.uptime_s, 1)
        data["measured_fps"] = round(self.measured_fps, 1)
        data["source_fps"] = round(self.source_fps, 1)
        return data


class LiveSession:
    """Owns the camera thread, the detection pipeline and the preview frame."""

    def __init__(self, job_root: Path, on_incident: Optional[IncidentHook] = None) -> None:
        self.job_root = Path(job_root)
        self.on_incident = on_incident

        self._control_lock = threading.Lock()
        self._frame_lock = threading.Condition()
        self._frame_jpeg: Optional[bytes] = None
        self._frame_seq = 0

        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._stats = LiveStats()
        self._stats_lock = threading.Lock()

    # -- state -------------------------------------------------------------
    @property
    def active(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def stats(self) -> Dict[str, object]:
        with self._stats_lock:
            snapshot = LiveStats(**self._stats.__dict__)
        snapshot.active = self.active
        return snapshot.to_dict()

    def _update_stats(self, **fields: object) -> None:
        with self._stats_lock:
            for key, value in fields.items():
                setattr(self._stats, key, value)

    # -- lifecycle ---------------------------------------------------------
    def start(self, job_id: str, config: PipelineConfig, record: bool = True) -> Dict[str, object]:
        """Open the camera and start the worker thread."""
        with self._control_lock:
            if self.active:
                raise RuntimeError("The live camera is already running.")

            camera = config.camera
            source = open_camera(
                index=camera.device_index,
                width=camera.width,
                height=camera.height,
                target_fps=camera.target_fps,
                warmup_reads=camera.warmup_reads,
            )

            output_dir = self.job_root / job_id / "incidents" if record else None
            if output_dir is not None:
                output_dir.mkdir(parents=True, exist_ok=True)

            pipeline = SecurityPipeline(
                config,
                fps=source.fps,
                frame_size=source.size,
                output_dir=output_dir,
                source_label=f"Camera {camera.device_index}",
                live=True,
                on_incident=lambda record_: self._handle_incident(job_id, record_),
            )

            self._stop.clear()
            with self._stats_lock:
                self._stats = LiveStats(
                    active=True,
                    job_id=job_id,
                    device_index=camera.device_index,
                    backend=source.backend,
                    width=source.width,
                    height=source.height,
                    source_fps=source.fps,
                    measured_fps=source.fps,
                    started_at=time.time(),
                    sensitivity=config.sensitivity,
                )

            self._thread = threading.Thread(
                target=self._run,
                args=(source, pipeline),
                name=f"live-camera-{job_id}",
                daemon=True,
            )
            self._thread.start()

            return {
                "job_id": job_id,
                "device": camera.device_index,
                "backend": source.backend,
                "resolution": f"{source.width}x{source.height}",
                "fps": round(source.fps, 1),
                "recording": bool(output_dir),
                "sensitivity": config.sensitivity,
            }

    def stop(self, timeout: float = 5.0) -> Dict[str, object]:
        """Ask the worker to finish and wait for it, so nothing is left open."""
        with self._control_lock:
            thread = self._thread
            self._stop.set()

        if thread is not None:
            thread.join(timeout=timeout)
            if thread.is_alive():
                LOGGER.warning("Live camera thread did not stop within %.1fs", timeout)

        with self._control_lock:
            if thread is not None and not thread.is_alive():
                self._thread = None

        with self._frame_lock:
            self._frame_jpeg = None
            self._frame_seq += 1
            self._frame_lock.notify_all()

        self._update_stats(active=False, recording=False, alert=False)
        return self.stats()

    # -- worker ------------------------------------------------------------
    def _run(self, source: VideoSource, pipeline: SecurityPipeline) -> None:
        started = time.perf_counter()
        frame_times: List[float] = []
        reconnects = 0

        try:
            while not self._stop.is_set():
                ok, frame = source.read()
                if not ok:
                    if source.consecutive_failures < 15:
                        time.sleep(0.02)
                        continue
                    if reconnects < pipeline.config.camera.max_reconnect_attempts:
                        reconnects += 1
                        LOGGER.warning(
                            "Camera stopped delivering frames; reconnect attempt %d",
                            reconnects,
                        )
                        source = self._reconnect(source, pipeline.config)
                        if source is None:
                            break
                        continue
                    self._update_stats(error="The camera stopped delivering frames.")
                    break

                now = time.perf_counter()
                frame_times.append(now)
                if len(frame_times) > 30:
                    frame_times.pop(0)
                measured = (
                    (len(frame_times) - 1) / (frame_times[-1] - frame_times[0])
                    if len(frame_times) > 5 and frame_times[-1] > frame_times[0]
                    else source.fps
                )

                result = pipeline.process_frame(
                    frame, timestamp_s=now - started, measured_fps=measured
                )
                self._publish(result.annotated)
                self._update_stats(
                    frames=result.frame_idx,
                    incidents=pipeline.incident_count,
                    recording=result.recording,
                    alert=result.alert,
                    warming_up=result.warming_up,
                    tracked_objects=len(result.tracks),
                    measured_fps=measured,
                )
        except Exception as exc:  # pragma: no cover - worker must not die silently
            LOGGER.exception("Live camera thread failed")
            self._update_stats(error=str(exc))
        finally:
            pipeline.close()
            source.release()
            self._cleanup_empty_output(pipeline)
            self._update_stats(
                active=False,
                recording=False,
                alert=False,
                incidents=pipeline.incident_count,
            )
            LOGGER.info(
                "Live session finished: %d frames, %d incidents",
                pipeline.frames_processed,
                pipeline.incident_count,
            )

    def _reconnect(self, source: VideoSource, config: PipelineConfig) -> Optional[VideoSource]:
        source.release()
        time.sleep(0.5)
        try:
            return open_camera(
                index=config.camera.device_index,
                width=config.camera.width,
                height=config.camera.height,
                target_fps=config.camera.target_fps,
                warmup_reads=config.camera.warmup_reads,
            )
        except CameraError as exc:
            LOGGER.error("Reconnect failed: %s", exc)
            self._update_stats(error=str(exc))
            return None

    @staticmethod
    def _cleanup_empty_output(pipeline: SecurityPipeline) -> None:
        """Remove the session folder when the session recorded nothing."""
        recorder = pipeline.recorder
        if recorder is None or pipeline.incident_count:
            return
        directory = Path(recorder.output_dir)
        for candidate in (directory, directory.parent):
            try:
                if candidate.exists() and not any(candidate.iterdir()):
                    candidate.rmdir()
            except OSError:
                return

    def _handle_incident(self, job_id: str, record: IncidentRecord) -> None:
        if self.on_incident is not None:
            self.on_incident(job_id, record)

    # -- preview -----------------------------------------------------------
    def _publish(self, frame: np.ndarray) -> None:
        ok, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ok:
            return
        with self._frame_lock:
            self._frame_jpeg = buffer.tobytes()
            self._frame_seq += 1
            self._frame_lock.notify_all()

    def frames(self, timeout: float = 2.0):
        """Yield MJPEG parts, one per newly published frame."""
        last_seq = -1
        while self.active:
            with self._frame_lock:
                if self._frame_seq == last_seq:
                    self._frame_lock.wait(timeout=timeout)
                if self._frame_seq == last_seq or self._frame_jpeg is None:
                    continue
                last_seq = self._frame_seq
                payload = self._frame_jpeg

            yield (
                b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                + str(len(payload)).encode()
                + b"\r\n\r\n"
                + payload
                + b"\r\n"
            )

    def snapshot(self) -> Optional[bytes]:
        with self._frame_lock:
            return self._frame_jpeg
