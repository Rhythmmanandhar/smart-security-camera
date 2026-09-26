"""Public API and command-line interface for the smart security camera.

``SmartSecurityCamera`` is the façade used by the web app, the CLI and the
tests.  It owns a :class:`~src.pipeline.SecurityPipeline` and feeds it frames
from either a video file or a physical camera.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from threading import Event
from typing import Callable, List, Optional, Sequence

import cv2

from .camera_source import (
    CameraError,
    VideoSource,
    list_devices,
    open_camera,
    open_file,
    sanitise_fps,
)
from .config import (
    SENSITIVITY_LEVELS,
    CameraConfig,
    PipelineConfig,
    build_config,
    normalise_sensitivity,
)
from .geometry import MotionBox, NormRect
from .pipeline import FrameResult, SecurityPipeline
from .recorder import IncidentRecord
from .tracker import CentroidTracker, Track

LOGGER = logging.getLogger(__name__)

# Re-exported so callers can import the whole public API from one module.
__all__ = [
    "SmartSecurityCamera",
    "Progress",
    "SecurityPipeline",
    "FrameResult",
    "IncidentRecord",
    "CentroidTracker",
    "Track",
    "MotionBox",
    "CameraError",
    "build_config",
    "normalise_sensitivity",
    "list_devices",
    "sanitise_fps",
    "main",
]

ProgressCallback = Callable[["Progress"], None]


@dataclass
class Progress:
    """Progress of a file analysis run."""

    frames_done: int
    frames_total: int
    incidents: int
    elapsed_s: float

    @property
    def fraction(self) -> float:
        if self.frames_total <= 0:
            return 0.0
        return min(1.0, self.frames_done / float(self.frames_total))

    @property
    def percent(self) -> int:
        return int(round(self.fraction * 100))


class SmartSecurityCamera:
    """Analyse a video file or a live camera and save incident clips."""

    def __init__(
        self,
        video_path: Optional[Path] = None,
        output_dir: Optional[Path] = None,
        config: Optional[PipelineConfig] = None,
        *,
        live: bool = False,
        device_index: Optional[int] = None,
        source_label: Optional[str] = None,
    ) -> None:
        self.video_path = Path(video_path) if video_path else None
        self.output_dir = Path(output_dir) if output_dir else None
        self.config = config or build_config()
        self.live = live
        # Uploads are stored under a temporary name, so the caller can pass the
        # name the operator recognises; it is burnt into the evidence overlay.
        self.source_label = source_label
        if device_index is not None:
            self.config = replace(
                self.config,
                camera=replace(self.config.camera, device_index=device_index),
            )
        self.pipeline: Optional[SecurityPipeline] = None

    # -- sources -----------------------------------------------------------
    def _open_source(self) -> VideoSource:
        if self.live:
            camera: CameraConfig = self.config.camera
            return open_camera(
                index=camera.device_index,
                width=camera.width,
                height=camera.height,
                target_fps=camera.target_fps,
                warmup_reads=camera.warmup_reads,
            )
        if self.video_path is None:
            raise CameraError("No input video was provided.")
        return open_file(self.video_path)

    # -- processing --------------------------------------------------------
    def process(
        self,
        progress: Optional[ProgressCallback] = None,
        stop_event: Optional[Event] = None,
        preview: bool = False,
        on_frame: Optional[Callable[[FrameResult], None]] = None,
    ) -> List[IncidentRecord]:
        """Run detection over the whole source and return the incidents found.

        ``stop_event`` makes the run cancellable, which is what lets the web UI
        abort a long analysis instead of leaving a thread burning CPU.
        """
        source = self._open_source()
        started = time.perf_counter()
        frames_total = source.frame_count

        self.pipeline = SecurityPipeline(
            self.config,
            fps=source.fps,
            frame_size=source.size,
            output_dir=self.output_dir,
            source_label=self.source_label or source.label,
            live=source.is_live,
        )

        measured_fps: Optional[float] = source.fps if source.is_live else None
        window = "Smart Security Camera - press q to quit"
        cancelled = False

        try:
            while True:
                if stop_event is not None and stop_event.is_set():
                    cancelled = True
                    break

                ok, frame = source.read()
                if not ok:
                    if source.is_live and source.consecutive_failures < 30:
                        # A dropped USB frame is normal; keep going briefly.
                        time.sleep(0.01)
                        continue
                    break

                timestamp = (
                    time.perf_counter() - started if source.is_live else None
                )
                result = self.pipeline.process_frame(
                    frame, timestamp_s=timestamp, measured_fps=measured_fps
                )

                if on_frame is not None:
                    on_frame(result)

                if preview:
                    cv2.imshow(window, result.annotated)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                if progress is not None and result.frame_idx % 10 == 0:
                    progress(
                        Progress(
                            frames_done=result.frame_idx,
                            frames_total=frames_total,
                            incidents=self.pipeline.incident_count,
                            elapsed_s=time.perf_counter() - started,
                        )
                    )
        finally:
            incidents = self.pipeline.close()
            source.release()
            if preview:
                cv2.destroyAllWindows()

        if progress is not None:
            progress(
                Progress(
                    frames_done=self.pipeline.frames_processed,
                    frames_total=frames_total or self.pipeline.frames_processed,
                    incidents=len(incidents),
                    elapsed_s=time.perf_counter() - started,
                )
            )
        if cancelled:
            LOGGER.info("Analysis cancelled after %d frames", self.pipeline.frames_processed)
        return incidents


# ---------------------------------------------------------------------------
# Command line interface
# ---------------------------------------------------------------------------


def _parse_roi(values: Optional[Sequence[float]]) -> Optional[NormRect]:
    if not values:
        return None
    if len(values) != 4:
        raise argparse.ArgumentTypeError("--roi needs exactly four values")
    x1, y1, x2, y2 = (float(v) for v in values)
    for value in (x1, y1, x2, y2):
        if not 0.0 <= value <= 1.0:
            raise argparse.ArgumentTypeError("--roi values must be between 0 and 1")
    return (x1, y1, x2, y2)


def _parse_zone(raw: str) -> NormRect:
    parts = [p for p in raw.replace(",", " ").split() if p]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError(
            "--ignore-zone needs four numbers, e.g. '0.7 0 1 0.1'"
        )
    return tuple(float(p) for p in parts)  # type: ignore[return-value]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="security-camera",
        description="Detect unauthorised motion in a video file or a live camera feed.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    source = parser.add_argument_group("source")
    source.add_argument("--input", type=Path, help="Path to a video file")
    source.add_argument("--live", action="store_true", help="Use a camera instead of a file")
    source.add_argument("--device", type=int, default=0, help="Camera device index")
    source.add_argument(
        "--list-cameras",
        action="store_true",
        help="List the cameras that are available and exit",
    )

    output = parser.add_argument_group("output")
    output.add_argument(
        "--output-dir", type=Path, default=Path("outputs/incidents"),
        help="Directory for incident clips",
    )
    output.add_argument(
        "--no-record", action="store_true", help="Detect only; do not write clips"
    )
    output.add_argument(
        "--preview", action="store_true", help="Show an annotated preview window"
    )

    tuning = parser.add_argument_group("detection tuning")
    tuning.add_argument(
        "--sensitivity", choices=SENSITIVITY_LEVELS, default="balanced",
        help="Preset controlling how easily motion raises an alarm",
    )
    tuning.add_argument(
        "--min-area-ratio", type=float,
        help="Minimum blob size as a fraction of frame area (overrides the preset)",
    )
    tuning.add_argument("--process-width", type=int, help="Analysis width in pixels")
    tuning.add_argument("--warmup-frames", type=int, help="Frames used to learn the background")
    tuning.add_argument(
        "--roi", nargs=4, type=float, metavar=("X1", "Y1", "X2", "Y2"),
        help="Monitor only this normalised region, e.g. --roi 0.2 0.2 0.9 0.9",
    )
    tuning.add_argument(
        "--ignore-zone", action="append", type=_parse_zone, default=[],
        metavar="X1,Y1,X2,Y2",
        help="Never alarm inside this normalised region (repeatable)",
    )
    tuning.add_argument(
        "--classifier", choices=["none", "hog", "yolo"], default="none",
        help="Confirm what moved before alerting (yolo needs `pip install ultralytics`)",
    )
    tuning.add_argument(
        "--detect-objects", action="store_true",
        help=(
            "Let YOLO find the objects instead of vetting motion blobs: one "
            "labelled box per person or vehicle, including distant ones. "
            "Implies --classifier yolo"
        ),
    )
    tuning.add_argument(
        "--detector-size", type=int, default=None,
        help="Object detector input size; larger finds smaller/further objects",
    )
    tuning.add_argument(
        "--classes", default=None,
        help="Comma-separated classes that may raise an alarm, e.g. person,car",
    )
    tuning.add_argument(
        "--classifier-confidence", type=float, default=None,
        help="Minimum classifier confidence (0-1)",
    )
    tuning.add_argument(
        "--require-person", action="store_true",
        help="Shorthand for --classifier hog",
    )
    tuning.add_argument(
        "--no-tracking", action="store_true", help="Disable track IDs and paths in the overlay"
    )

    timing = parser.add_argument_group("incident timing")
    timing.add_argument("--min-motion-seconds", type=float, help="Motion needed to open an incident")
    timing.add_argument("--pre-event-seconds", type=float, help="Footage kept before the trigger")
    timing.add_argument("--post-event-seconds", type=float, help="Quiet time before an incident ends")
    timing.add_argument("--min-incident-seconds", type=float, help="Discard clips shorter than this")
    timing.add_argument("--max-incident-seconds", type=float, help="Split clips longer than this")
    # Frame-count equivalents kept for backwards compatibility with v1 scripts.
    timing.add_argument("--min-motion-frames", type=int, help=argparse.SUPPRESS)
    timing.add_argument("--pre-event-frames", type=int, help=argparse.SUPPRESS)
    timing.add_argument("--post-event-frames", type=int, help=argparse.SUPPRESS)

    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Logging verbosity",
    )
    return parser


def config_from_args(args: argparse.Namespace) -> PipelineConfig:
    """Translate parsed CLI arguments into a :class:`PipelineConfig`."""
    return build_config(
        args.sensitivity,
        detection={
            "min_area_ratio": getattr(args, "min_area_ratio", None),
            "process_width": getattr(args, "process_width", None),
            "warmup_frames": getattr(args, "warmup_frames", None),
            "roi": _parse_roi(getattr(args, "roi", None)),
            "ignore_zones": tuple(getattr(args, "ignore_zone", []) or ()) or None,
        },
        incident={
            "min_motion_seconds": getattr(args, "min_motion_seconds", None),
            "pre_event_seconds": getattr(args, "pre_event_seconds", None),
            "post_event_seconds": getattr(args, "post_event_seconds", None),
            "min_incident_seconds": getattr(args, "min_incident_seconds", None),
            "max_incident_seconds": getattr(args, "max_incident_seconds", None),
            "min_motion_frames": getattr(args, "min_motion_frames", None),
            "pre_event_frames": getattr(args, "pre_event_frames", None),
            "post_event_frames": getattr(args, "post_event_frames", None),
        },
        camera={"device_index": getattr(args, "device", None)},
        classifier={
            "kind": _classifier_kind(args),
            "mode": "detect" if getattr(args, "detect_objects", False) else None,
            "classes": _parse_classes(getattr(args, "classes", None)),
            "confidence": getattr(args, "classifier_confidence", None),
            "image_size": getattr(args, "detector_size", None),
            "interval_seconds": 0.0 if getattr(args, "detect_objects", False) else None,
        },
        record_incidents=not getattr(args, "no_record", False),
        draw_tracks=not getattr(args, "no_tracking", False),
    )


def _classifier_kind(args: argparse.Namespace) -> str:
    """``--require-person`` is the v1 spelling of ``--classifier hog``."""
    kind = getattr(args, "classifier", "none") or "none"
    if getattr(args, "detect_objects", False):
        return "yolo"
    if kind == "none" and getattr(args, "require_person", False):
        return "hog"
    return kind


def _parse_classes(raw: Optional[str]):
    if not raw:
        return None
    names = tuple(name.strip().lower() for name in raw.split(",") if name.strip())
    return names or None


class ProgressPrinter:
    """A single-line progress indicator that log records can write over.

    The indicator is redrawn in place with a carriage return, so any log record
    emitted mid-run must erase it first - otherwise the two overwrite each
    other, which is what the first version did.
    """

    def __init__(self, stream=None) -> None:
        self.stream = stream or sys.stderr
        self._width = 0

    def update(self, progress: "Progress") -> None:
        if progress.frames_total:
            line = (
                f"  {progress.percent:3d}%  frame {progress.frames_done}/"
                f"{progress.frames_total}  incidents: {progress.incidents}"
            )
        else:
            line = f"  frame {progress.frames_done}  incidents: {progress.incidents}"
        self._width = max(self._width, len(line))
        self.stream.write("\r" + line.ljust(self._width))
        self.stream.flush()

    def clear(self) -> None:
        if self._width:
            self.stream.write("\r" + " " * self._width + "\r")
            self.stream.flush()
            self._width = 0


class ProgressAwareHandler(logging.StreamHandler):
    """Stream handler that clears the progress line before each record."""

    def __init__(self, printer: ProgressPrinter) -> None:
        super().__init__(stream=sys.stderr)
        self.printer = printer

    def emit(self, record: logging.LogRecord) -> None:
        self.printer.clear()
        super().emit(record)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    printer = ProgressPrinter()
    handler = ProgressAwareHandler(printer)
    handler.setFormatter(
        logging.Formatter("%(asctime)s  %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
    )
    logging.basicConfig(level=getattr(logging, args.log_level), handlers=[handler])

    if args.list_cameras:
        devices = list_devices()
        if not devices:
            print("No cameras detected.")
            return 1
        print("Available cameras:")
        for device in devices:
            print(f"  [{device.index}] {device.width}x{device.height} via {device.backend}")
        return 0

    if not args.live and args.input is None:
        print("Provide --input <video> or --live (see --help).", file=sys.stderr)
        return 2
    if args.live and args.input is not None:
        LOGGER.warning("--input is ignored when --live is used")

    config = config_from_args(args)
    camera = SmartSecurityCamera(
        video_path=None if args.live else args.input,
        output_dir=args.output_dir,
        config=config,
        live=args.live,
    )

    print("Smart Security Camera")
    print(f"  Source      : {'camera ' + str(args.device) if args.live else args.input}")
    print(f"  Output      : {args.output_dir if config.record_incidents else 'detection only'}")
    print(f"  Sensitivity : {config.sensitivity}")
    classifier = config.classifier
    role = "detecting objects" if classifier.detects else "confirming motion"
    print(
        f"  Detection   : {'motion + ' + classifier.kind if classifier.enabled else 'motion'}"
        + (f" - {role}" if classifier.enabled else "")
    )
    if classifier.detects:
        print(f"  Classes     : {', '.join(classifier.classes)}")
    # Flushed so the header still precedes the progress line (stderr) when
    # stdout is redirected to a file or a pipe.
    print(flush=True)

    try:
        incidents = camera.process(
            progress=None if args.live else printer.update,
            preview=args.preview or args.live,
        )
    except CameraError as exc:
        printer.clear()
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        printer.clear()
        print("Interrupted.")
        return 130

    printer.clear()
    print()
    pipeline = camera.pipeline
    frames = pipeline.frames_processed if pipeline else 0
    alert_frames = pipeline.alert_frames if pipeline else 0

    if not config.record_incidents:
        share = (alert_frames / frames * 100) if frames else 0.0
        print(
            f"Detection finished (recording disabled). "
            f"{alert_frames} of {frames} frames showed unauthorised motion ({share:.0f}%)."
        )
        return 0

    print(f"Processing finished. Incidents: {len(incidents)}")
    for incident in incidents:
        print(
            f"  {incident.index:02d}. {incident.filename}  "
            f"start={incident.start_time_s:.2f}s  end={incident.end_time_s:.2f}s  "
            f"duration={incident.duration_s:.2f}s  objects={incident.max_objects}"
            + (f"  [{', '.join(incident.labels)}]" if incident.labels else "")
        )
    if not incidents:
        print("  No unauthorised motion was detected.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
