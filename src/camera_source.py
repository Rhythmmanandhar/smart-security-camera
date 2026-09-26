"""Video sources: files and physical cameras.

Opening a webcam reliably on Windows is the part of this project that broke
most often, so it is isolated here:

* the DirectShow backend is tried before Media Foundation, because MSMF can
  take 5-10 seconds to open a device and fails outright on some drivers;
* a device is only reported as present once it has actually delivered a frame,
  instead of trusting ``VideoCapture.isOpened()``, which returns True for
  phantom devices;
* the first frames are discarded while auto-exposure settles, and the true
  frame rate is *measured* rather than taken from the driver, which commonly
  reports 30 fps while delivering 10 - the reason live clips used to play back
  at triple speed;
* reads that fail are retried and the device is reopened before the session is
  declared dead.
"""

from __future__ import annotations

import logging
import platform
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .nativelog import suppressed_native_stderr

LOGGER = logging.getLogger(__name__)

IS_WINDOWS = platform.system() == "Windows"

# Backend preference per platform, most reliable first.
if IS_WINDOWS:
    _BACKENDS: Tuple[Tuple[int, str], ...] = (
        (cv2.CAP_DSHOW, "DirectShow"),
        (cv2.CAP_MSMF, "Media Foundation"),
        (cv2.CAP_ANY, "default"),
    )
else:
    _BACKENDS = ((cv2.CAP_ANY, "default"),)

FPS_MIN = 1.0
FPS_MAX = 120.0
DEFAULT_FPS = 25.0


class CameraError(RuntimeError):
    """Raised with an actionable message when a camera cannot be used."""


def sanitise_fps(raw_fps: float, fallback: float = DEFAULT_FPS) -> float:
    """Clamp a reported frame rate into a believable range.

    Containers regularly report 0, NaN or 1000 fps; writing clips at those
    rates produces files that play at the wrong speed or not at all.
    """
    try:
        fps = float(raw_fps)
    except (TypeError, ValueError):
        return fallback
    if not np.isfinite(fps) or fps < FPS_MIN or fps > FPS_MAX:
        return fallback
    return fps


@dataclass
class DeviceInfo:
    index: int
    name: str
    width: int
    height: int
    backend: str

    def to_dict(self) -> Dict[str, object]:
        return {
            "index": self.index,
            "name": self.name,
            "width": self.width,
            "height": self.height,
            "backend": self.backend,
        }


def _try_open(index: int, backend: int) -> Optional[cv2.VideoCapture]:
    """Open a device quietly - absent indices make OpenCV shout on stderr."""
    with suppressed_native_stderr():
        cap = cv2.VideoCapture(index, backend)
        opened = cap.isOpened()
        if not opened:
            cap.release()
    return cap if opened else None


def list_devices(max_index: int = 5) -> List[DeviceInfo]:
    """Return the cameras that actually deliver frames.

    Probing is genuinely slow (a second or so per absent index on some
    drivers), so callers should cache the result - :mod:`app` does.
    """
    devices: List[DeviceInfo] = []
    for index in range(max_index):
        for backend, backend_name in _BACKENDS:
            cap = _try_open(index, backend)
            if cap is None:
                continue
            ok, frame = cap.read()
            if ok and frame is not None:
                height, width = frame.shape[:2]
                devices.append(
                    DeviceInfo(
                        index=index,
                        name=f"Camera {index}",
                        width=int(width),
                        height=int(height),
                        backend=backend_name,
                    )
                )
                cap.release()
                break
            cap.release()
    return devices


class VideoSource:
    """A frame source with a known size and a trustworthy frame rate."""

    def __init__(
        self,
        capture: cv2.VideoCapture,
        *,
        fps: float,
        width: int,
        height: int,
        is_live: bool,
        label: str,
        backend: str = "",
        frame_count: int = 0,
    ) -> None:
        self.capture = capture
        self.fps = fps
        self.width = width
        self.height = height
        self.is_live = is_live
        self.label = label
        self.backend = backend
        self.frame_count = frame_count
        self._failed_reads = 0

    @property
    def size(self) -> Tuple[int, int]:
        return (self.width, self.height)

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        ok, frame = self.capture.read()
        if ok and frame is not None:
            self._failed_reads = 0
            return True, frame
        self._failed_reads += 1
        return False, None

    @property
    def consecutive_failures(self) -> int:
        return self._failed_reads

    def release(self) -> None:
        try:
            self.capture.release()
        except Exception:  # pragma: no cover - defensive
            pass


def open_file(path: Path) -> VideoSource:
    """Open a video file, validating it before anything else runs."""
    path = Path(path)
    if not path.exists():
        raise CameraError(f"Input video not found: {path}")
    if path.stat().st_size == 0:
        raise CameraError(f"Input video is empty: {path.name}")

    with suppressed_native_stderr():
        capture = cv2.VideoCapture(str(path))
        opened = capture.isOpened()
    if not opened:
        capture.release()
        raise CameraError(
            f"Could not decode {path.name}. The file may be corrupt or use an "
            "unsupported codec."
        )

    ok, frame = capture.read()
    if not ok or frame is None:
        capture.release()
        raise CameraError(f"{path.name} contains no readable video frames.")
    # Rewind so the caller still sees frame 1.
    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)

    height, width = frame.shape[:2]
    fps = sanitise_fps(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    return VideoSource(
        capture,
        fps=fps,
        width=int(width),
        height=int(height),
        is_live=False,
        label=path.name,
        frame_count=max(0, frame_count),
    )


def open_camera(
    index: int = 0,
    width: int = 1280,
    height: int = 720,
    target_fps: float = 30.0,
    warmup_reads: int = 5,
    measure_fps: bool = True,
) -> VideoSource:
    """Open a physical camera, warm it up and measure its real frame rate."""
    last_error = "no camera backend accepted the device"

    for backend, backend_name in _BACKENDS:
        capture = _try_open(index, backend)
        if capture is None:
            last_error = f"device {index} could not be opened"
            continue

        capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        capture.set(cv2.CAP_PROP_FPS, target_fps)
        # A 1-frame buffer keeps the live preview close to real time.
        try:
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:  # pragma: no cover - unsupported on some backends
            pass

        frame = None
        for _ in range(max(1, warmup_reads)):
            ok, candidate = capture.read()
            if ok and candidate is not None:
                frame = candidate
        if frame is None:
            capture.release()
            last_error = (
                f"device {index} opened with {backend_name} but returned no frames "
                "(it may be in use by another application, or blocked by Windows "
                "camera privacy settings)"
            )
            continue

        actual_h, actual_w = frame.shape[:2]
        fps = sanitise_fps(capture.get(cv2.CAP_PROP_FPS), fallback=target_fps)
        if measure_fps:
            measured = _measure_fps(capture)
            if measured is not None:
                LOGGER.info(
                    "Camera %d reports %.1f fps, measured %.1f fps", index, fps, measured
                )
                fps = measured

        LOGGER.info(
            "Camera %d opened with %s at %dx%d @ %.1f fps",
            index,
            backend_name,
            actual_w,
            actual_h,
            fps,
        )
        return VideoSource(
            capture,
            fps=fps,
            width=int(actual_w),
            height=int(actual_h),
            is_live=True,
            label=f"Camera {index}",
            backend=backend_name,
        )

    raise CameraError(
        f"Could not start camera {index}: {last_error}. "
        "Close any other app using the camera (Teams, Zoom, the Camera app), "
        "check Settings > Privacy > Camera, then try again."
    )


def _measure_fps(capture: cv2.VideoCapture, samples: int = 12) -> Optional[float]:
    """Time a short burst of reads to find the delivered frame rate."""
    start = time.perf_counter()
    read = 0
    for _ in range(samples):
        ok, _frame = capture.read()
        if not ok:
            break
        read += 1
    elapsed = time.perf_counter() - start
    if read < 4 or elapsed <= 0:
        return None
    return sanitise_fps(read / elapsed, fallback=DEFAULT_FPS)
