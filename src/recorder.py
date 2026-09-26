"""Incident recording: codec selection, clip writing, metadata and thumbnails."""

from __future__ import annotations

import json
import logging
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .nativelog import configure_quiet_opencv, suppressed_native_stderr

LOGGER = logging.getLogger(__name__)

# Candidate (fourcc, extension) pairs in preference order.  H.264 in MP4 and
# VP8/VP9 in WebM play natively in browsers; MPEG-4 Part 2 ("mp4v"/FMP4) does
# not, which is why the old build produced clips that only VLC could open.
_CODEC_CANDIDATES: Tuple[Tuple[str, str], ...] = (
    ("avc1", ".mp4"),
    ("H264", ".mp4"),
    ("VP80", ".webm"),
    ("mp4v", ".mp4"),
    ("MJPG", ".avi"),
)

BROWSER_SAFE_FOURCC = {"avc1", "h264", "H264", "X264", "VP80", "VP90", "vp09"}

_probe_cache: Optional[Tuple[str, str]] = None


def _open_writer(path: Path, fourcc: str, fps: float, size: Tuple[int, int]):
    """Create a VideoWriter without leaking encoder-probe noise to stderr."""
    configure_quiet_opencv()
    with suppressed_native_stderr():
        writer = cv2.VideoWriter(
            str(path), cv2.VideoWriter_fourcc(*fourcc), float(fps), size
        )
    return writer


def probe_codec(fps: float = 25.0, size: Tuple[int, int] = (320, 240)) -> Tuple[str, str]:
    """Find the best (fourcc, extension) this OpenCV build can actually write.

    The result is cached: probing writes throwaway files and is not free.
    """
    global _probe_cache
    if _probe_cache is not None:
        return _probe_cache

    frame = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    cv2.rectangle(frame, (10, 10), (size[0] - 10, size[1] - 10), (255, 255, 255), -1)

    with tempfile.TemporaryDirectory(prefix="codec_probe_") as tmp:
        for fourcc, ext in _CODEC_CANDIDATES:
            path = Path(tmp) / f"probe{ext}"
            writer = _open_writer(path, fourcc, fps, size)
            if not writer.isOpened():
                writer.release()
                continue
            for _ in range(5):
                writer.write(frame)
            writer.release()

            if not path.exists() or path.stat().st_size < 512:
                continue
            with suppressed_native_stderr():
                cap = cv2.VideoCapture(str(path))
                readable = cap.isOpened() and cap.read()[0]
                cap.release()
            if readable:
                _probe_cache = (fourcc, ext)
                LOGGER.info("Using %s%s for incident clips", fourcc, ext)
                return _probe_cache

    LOGGER.warning("No verified codec found; falling back to mp4v")
    _probe_cache = ("mp4v", ".mp4")
    return _probe_cache


def reset_codec_cache() -> None:
    """Testing hook."""
    global _probe_cache
    _probe_cache = None


@dataclass
class IncidentRecord:
    """Metadata describing one saved incident."""

    index: int
    filename: str
    path: str
    start_time_s: float
    end_time_s: float
    duration_s: float
    start_frame: int
    end_frame: int
    frame_count: int
    fps: float
    max_objects: int
    track_ids: List[int] = field(default_factory=list)
    labels: List[str] = field(default_factory=list)
    """Distinct classes seen during the incident (empty without a classifier)."""
    thumbnail: Optional[str] = None
    started_at: Optional[str] = None
    source: Optional[str] = None
    sensitivity: str = "balanced"

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


class IncidentRecorder:
    """Writes one incident clip plus a JPEG thumbnail and a JSON sidecar.

    Short incidents are deleted on close, so a single noisy frame burst does
    not leave a 200 ms clip behind.
    """

    def __init__(
        self,
        output_dir: Path,
        fps: float,
        frame_size: Tuple[int, int],
        *,
        source: Optional[str] = None,
        sensitivity: str = "balanced",
        min_duration_s: float = 0.0,
        write_metadata: bool = True,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.fps = fps if fps and fps > 0 else 25.0
        self.frame_size = frame_size
        self.source = source
        self.sensitivity = sensitivity
        self.min_duration_s = min_duration_s
        self.write_metadata = write_metadata

        self.fourcc, self.extension = probe_codec(self.fps, frame_size)
        self._writer: Optional[cv2.VideoWriter] = None
        self._index = 0
        self._frames_written = 0
        self._path: Optional[Path] = None
        self._start_time_s = 0.0
        self._start_frame = 0
        self._started_at: Optional[str] = None
        self._max_objects = 0
        self._track_ids: set = set()
        self._labels: set = set()
        self._best_frame: Optional[np.ndarray] = None
        self._best_score = -1

    # -- state -------------------------------------------------------------
    @property
    def is_recording(self) -> bool:
        return self._writer is not None

    @property
    def frames_written(self) -> int:
        return self._frames_written

    # -- lifecycle ---------------------------------------------------------
    def start(self, index: int, start_time_s: float, start_frame: int) -> Optional[Path]:
        """Open a writer for incident ``index``; returns the clip path."""
        if self._writer is not None:
            self.abort()

        self._index = index
        filename = f"incident_{index:03d}{self.extension}"
        path = self.output_dir / filename
        writer = _open_writer(path, self.fourcc, self.fps, self.frame_size)
        if not writer.isOpened():
            writer.release()
            LOGGER.error("Could not open a video writer for %s", path)
            return None

        self._writer = writer
        self._path = path
        self._frames_written = 0
        self._start_time_s = start_time_s
        self._start_frame = start_frame
        self._started_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self._max_objects = 0
        self._track_ids = set()
        self._labels = set()
        self._best_frame = None
        self._best_score = -1
        return path

    def write(
        self,
        frame: np.ndarray,
        object_count: int = 0,
        track_ids: Sequence[int] = (),
        labels: Sequence[str] = (),
    ) -> None:
        """Append one annotated frame to the open clip."""
        if self._writer is None:
            return
        if (frame.shape[1], frame.shape[0]) != self.frame_size:
            frame = cv2.resize(frame, self.frame_size)
        self._writer.write(frame)
        self._frames_written += 1
        self._max_objects = max(self._max_objects, object_count)
        self._track_ids.update(int(t) for t in track_ids)
        self._labels.update(str(label) for label in labels if label)

        # Keep the busiest frame as the thumbnail.
        if object_count > self._best_score:
            self._best_score = object_count
            self._best_frame = frame.copy()

    def finish(self, end_time_s: float, end_frame: int) -> Optional[IncidentRecord]:
        """Close the clip, dropping it when it is too short to be useful."""
        if self._writer is None or self._path is None:
            return None

        self._writer.release()
        self._writer = None
        path = self._path
        self._path = None

        duration = max(0.0, end_time_s - self._start_time_s)
        if duration + 1e-6 < self.min_duration_s or self._frames_written == 0:
            LOGGER.info("Discarding %s (%.2fs is below the minimum)", path.name, duration)
            path.unlink(missing_ok=True)
            return None

        thumbnail = None
        if self._best_frame is not None:
            thumb_path = path.with_suffix(".jpg")
            if cv2.imwrite(str(thumb_path), self._best_frame, [cv2.IMWRITE_JPEG_QUALITY, 85]):
                thumbnail = thumb_path.name

        record = IncidentRecord(
            index=self._index,
            filename=path.name,
            path=str(path),
            start_time_s=round(self._start_time_s, 2),
            end_time_s=round(end_time_s, 2),
            duration_s=round(duration, 2),
            start_frame=self._start_frame,
            end_frame=end_frame,
            frame_count=self._frames_written,
            fps=round(self.fps, 3),
            max_objects=self._max_objects,
            track_ids=sorted(self._track_ids),
            labels=sorted(self._labels),
            thumbnail=thumbnail,
            started_at=self._started_at,
            source=self.source,
            sensitivity=self.sensitivity,
        )

        if self.write_metadata:
            meta_path = path.with_suffix(".json")
            meta_path.write_text(json.dumps(record.to_dict(), indent=2), encoding="utf-8")

        return record

    def abort(self) -> None:
        """Close and delete the in-progress clip."""
        if self._writer is not None:
            self._writer.release()
            self._writer = None
        if self._path is not None:
            self._path.unlink(missing_ok=True)
            self._path = None
        self._frames_written = 0

    def close(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None
