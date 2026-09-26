"""The detection pipeline shared by file analysis and the live camera.

Having one engine matters: the first version of this project reimplemented
detection, tracking and clipping inside the Flask live-camera thread, so live
monitoring quietly used different (and worse) rules than file analysis, and
every fix had to be made twice.
"""

from __future__ import annotations

import datetime as dt
import logging
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Deque, List, Optional, Sequence, Tuple

import numpy as np

from .annotator import OverlayState, draw_overlay, media_timestamp, wall_clock_timestamp
from .config import PipelineConfig
from .classifier import ObjectDetection, build_classifier
from .detector import MotionDetector
from .geometry import MotionBox, intersection_area, iou
from .recorder import IncidentRecord, IncidentRecorder
from .tracker import CentroidTracker, Track

LOGGER = logging.getLogger(__name__)

IncidentCallback = Callable[[IncidentRecord], None]


@dataclass
class FrameResult:
    """The outcome of analysing a single frame."""

    frame_idx: int
    timestamp_s: float
    annotated: np.ndarray
    alert: bool
    tracks: List[Track] = field(default_factory=list)
    alerting_ids: List[int] = field(default_factory=list)
    boxes: List[MotionBox] = field(default_factory=list)
    warming_up: bool = False
    scene_change: bool = False
    recording: bool = False
    incident_index: Optional[int] = None
    foreground_ratio: float = 0.0


class SecurityPipeline:
    """Motion detection, tracking, alerting and incident recording."""

    def __init__(
        self,
        config: PipelineConfig,
        fps: float,
        frame_size: Tuple[int, int],
        output_dir: Optional[Path] = None,
        *,
        source_label: str = "",
        live: bool = False,
        on_incident: Optional[IncidentCallback] = None,
    ) -> None:
        self.config = config
        self.fps = fps if fps and fps > 0 else 25.0
        self.frame_size = frame_size
        self.source_label = source_label
        self.live = live
        self.on_incident = on_incident

        self.detector = MotionDetector(config.detection, frame_size, fps=self.fps)
        self.tracker = CentroidTracker(config.tracking, frame_size)

        self.classifier = build_classifier(
            config.classifier.kind,
            allowed_classes=config.classifier.classes,
            confidence=config.classifier.confidence,
            image_size=config.classifier.image_size,
        )
        self.classify_interval = config.classifier.interval_frames(self.fps)
        self.new_object_frames = config.classifier.new_object_frames(self.fps)

        # In "detect" mode the model supplies the boxes; motion only decides
        # which of the objects it found are worth alarming about.
        self.object_mode = config.classifier.detects and hasattr(
            self.classifier, "detect_objects"
        )
        self._objects: List[ObjectDetection] = []
        self._objects_frame = -1

        if self.classifier is not None:
            LOGGER.info(
                "Classifier enabled: %s (%s)",
                self.classifier.describe(),
                "detecting objects" if self.object_mode else "confirming motion",
            )

        timings = config.incident.frames(self.fps)
        self.min_motion_frames = timings["min_motion"]
        self.post_event_frames = timings["post_event"]
        self.pre_event_frames = timings["pre_event"]
        self.max_incident_frames = timings["max_incident"]

        self._pre_buffer: Deque[np.ndarray] = deque(maxlen=max(1, self.pre_event_frames))
        self._motion_streak = 0
        self._silent_streak = 0
        self._frame_idx = 0
        self._incident_counter = 0
        self._incident_start_s = 0.0
        self._incident_start_frame = 0
        self._last_timestamp_s = 0.0
        self._last_incident_end_s = 0.0
        self._last_incident_end_frame = 0

        self.alert_frames = 0
        """Frames flagged as unauthorised motion, recorded or not."""

        self.incidents: List[IncidentRecord] = []
        self.recorder: Optional[IncidentRecorder] = None
        if config.record_incidents and output_dir is not None:
            self.recorder = IncidentRecorder(
                Path(output_dir),
                fps=self.fps,
                frame_size=frame_size,
                source=source_label,
                sensitivity=config.sensitivity,
                min_duration_s=config.incident.min_incident_seconds,
            )

        self._started_wall = dt.datetime.now()

    # -- properties --------------------------------------------------------
    @property
    def frames_processed(self) -> int:
        return self._frame_idx

    @property
    def incident_count(self) -> int:
        return len(self.incidents)

    @property
    def is_recording(self) -> bool:
        return self.recorder is not None and self.recorder.is_recording

    # -- main entry point --------------------------------------------------
    def process_frame(
        self,
        frame: np.ndarray,
        timestamp_s: Optional[float] = None,
        measured_fps: Optional[float] = None,
    ) -> FrameResult:
        """Analyse one frame, update the incident state machine and annotate."""
        self._frame_idx += 1
        frame_idx = self._frame_idx
        now_s = (
            timestamp_s
            if timestamp_s is not None
            else (frame_idx - 1) / self.fps
        )

        self._last_timestamp_s = now_s

        detection = self.detector.detect(frame, motion_expected=self._motion_streak > 0)

        if self.object_mode:
            visible, candidates = self._track_objects(frame, detection, frame_idx)
        else:
            tracks = self.tracker.update(detection.boxes, frame_idx)
            visible = [t for t in tracks if t.missing == 0]

            # Only tracks that are confirmed *and* actually travelling are worth
            # classifying, which keeps the expensive stage off idle frames.
            candidates = [
                track
                for track in visible
                if self.tracker.is_confirmed(track) and self.tracker.is_moving(track)
            ]
            self._classify(frame, candidates, frame_idx)

        alerting_ids = [
            track.track_id
            for track in candidates
            if self.object_mode or self.classifier is None or track.label is not None
        ]

        alert = bool(alerting_ids)
        if alert:
            self.alert_frames += 1
            self._motion_streak += 1
            self._silent_streak = 0
        else:
            self._silent_streak += 1
            self._motion_streak = 0

        annotated = self._annotate(
            frame, detection.boxes, visible, alerting_ids, frame_idx, now_s, alert,
            detection.warming_up, detection.scene_change, measured_fps,
        )

        self._update_recording(
            annotated,
            now_s,
            frame_idx,
            alert,
            len(alerting_ids),
            alerting_ids,
            labels=[t.label for t in candidates if t.label and t.label != "motion"],
        )

        # The pre-roll buffer is filled *after* the trigger check, so the frame
        # that opens an incident is written exactly once.
        if not self.is_recording and self.pre_event_frames > 0:
            self._pre_buffer.append(annotated)

        return FrameResult(
            frame_idx=frame_idx,
            timestamp_s=now_s,
            annotated=annotated,
            alert=alert,
            tracks=visible,
            alerting_ids=alerting_ids,
            boxes=detection.boxes,
            warming_up=detection.warming_up,
            scene_change=detection.scene_change,
            recording=self.is_recording,
            incident_index=self._incident_counter if self.is_recording else None,
            foreground_ratio=detection.foreground_ratio,
        )

    def close(self) -> List[IncidentRecord]:
        """Finish any open incident and return every incident recorded."""
        if self.recorder is not None and self.recorder.is_recording:
            self._finish_incident(self._last_timestamp_s, self._frame_idx)
        if self.recorder is not None:
            self.recorder.close()
        return self.incidents

    # -- object-detection mode ---------------------------------------------
    def _track_objects(self, frame, detection, frame_idx):
        """Track what the object detector sees, and alarm on the ones moving.

        The detector is the expensive part, so it runs on an interval; between
        runs the existing tracks are left untouched (their history must hold
        real observations, or a standing person would appear to drift) and only
        the drawn box is extrapolated from the track's velocity.
        """
        due = (
            self._objects_frame < 0
            or frame_idx - self._objects_frame >= max(1, self.classify_interval)
        )

        if due:
            try:
                self._objects = self.classifier.detect_objects(frame)
            except Exception:  # pragma: no cover - a model error must not stop capture
                LOGGER.exception("Object detection failed on frame %d", frame_idx)
                self._objects = []
            self._objects_frame = frame_idx

            # Objects the model named, plus anything that moved and was not
            # covered by one of them.  A detector that has never seen a
            # warehouse worker bent over a crate should not make that movement
            # disappear: an unnamed moving blob is still reported, just without
            # a class.
            boxes = self._merge_detections(self._objects, detection.boxes)

            tracks = self.tracker.update(boxes, frame_idx)
            self._label_tracks(tracks)
            for track in tracks:
                track.display_box = None
        else:
            tracks = list(self.tracker.tracks.values())
            for track in tracks:
                self._extrapolate(track, frame_idx)

        visible = [t for t in tracks if t.missing == 0]

        # A parked car and an empty chair are objects too, so an object has to
        # earn the alarm: either it is moving, or it has only just arrived and
        # has not yet had the chance to move.
        candidates = [
            track
            for track in visible
            if self.tracker.is_confirmed(track)
            and (
                self.tracker.is_moving(track)
                or frame_idx - track.first_frame <= self.new_object_frames
            )
        ]
        return visible, candidates

    @staticmethod
    def _merge_detections(objects, motion_boxes):
        """Object boxes win; motion is added only where nothing was named.

        The overlap test has to work both ways.  A motion blob is often just
        part of a person the model already boxed (the moving arm), and
        sometimes it is larger than the model's box (a person plus the trolley
        they are pushing).  Either way it is the same object, and adding it
        would put a second rectangle and a second id on one body - the
        duplicate this mode exists to remove.
        """
        kept = [obj.box for obj in objects]

        for box in motion_boxes:
            area = float(box.w * box.h) or 1.0
            duplicate = False
            for existing in kept:
                shared = intersection_area(box.rect, existing.rect)
                if shared <= 0:
                    continue
                existing_area = float(existing.w * existing.h) or 1.0
                if (
                    iou(box, existing) >= 0.15
                    or shared / area >= 0.35
                    or shared / existing_area >= 0.35
                ):
                    duplicate = True
                    break
            if not duplicate:
                kept.append(box)

        return kept

    def _label_tracks(self, tracks) -> None:
        """Give every track the class of the detection it was matched to."""
        for track in tracks:
            best, best_overlap = None, 0.0
            for obj in self._objects:
                overlap = iou(track.box, obj.box)
                if overlap > best_overlap:
                    best, best_overlap = obj, overlap
            if best is not None and best_overlap >= 0.3:
                track.label = best.label
                track.label_confidence = best.confidence
                track.label_frame = self._frame_idx
                track.classified = True

    @staticmethod
    def _extrapolate(track, frame_idx: int) -> None:
        """Slide the drawn box along the track's velocity between detections."""
        ahead = max(0, frame_idx - track.last_frame)
        if ahead == 0:
            track.display_box = None
            return
        vx, vy = track.velocity
        track.display_box = MotionBox(
            int(round(track.box.x + vx * ahead)),
            int(round(track.box.y + vy * ahead)),
            track.box.w,
            track.box.h,
            track.box.area,
        )

    # -- internals ---------------------------------------------------------
    def _classify(
        self, frame: np.ndarray, tracks: Sequence[Track], frame_idx: int
    ) -> None:
        """Refresh the class of any track whose verdict is missing or stale.

        Classification is the expensive stage, so a verdict is reused until it
        ages out; a track that was recognised as a person keeps alarming while
        it is being re-checked.
        """
        if self.classifier is None or not tracks:
            return

        due = [
            track
            for track in tracks
            if not track.classified
            or frame_idx - track.label_frame >= self.classify_interval
        ]
        if not due:
            return

        try:
            results = self.classifier.classify(frame, [track.box for track in due])
        except Exception:  # pragma: no cover - a model error must not stop capture
            LOGGER.exception("Classification failed; treating this frame as unclassified")
            return

        for track, result in zip(due, results):
            track.label = result.label if result else None
            track.label_confidence = result.confidence if result else 0.0
            track.label_frame = frame_idx
            track.classified = True

    def _annotate(
        self,
        frame: np.ndarray,
        boxes: Sequence[MotionBox],
        tracks: Sequence[Track],
        alerting_ids: Sequence[int],
        frame_idx: int,
        now_s: float,
        alert: bool,
        warming_up: bool,
        scene_change: bool,
        measured_fps: Optional[float],
    ) -> np.ndarray:
        if self.live:
            timestamp = wall_clock_timestamp()
        else:
            timestamp = media_timestamp(frame_idx, self.fps)

        elapsed = 0.0
        if self.is_recording:
            elapsed = max(0.0, now_s - self._incident_start_s)

        state = OverlayState(
            frame_idx=frame_idx,
            fps=self.fps,
            timestamp=timestamp,
            alert=alert,
            warming_up=warming_up,
            scene_change=scene_change,
            recording=self.is_recording,
            incident_index=self._incident_counter if self.is_recording else None,
            incident_elapsed_s=elapsed,
            incident_count=len(self.incidents),
            measured_fps=measured_fps,
            source_label=self.source_label,
        )
        return draw_overlay(
            frame,
            boxes,
            tracks,
            alerting_ids,
            state,
            roi_rect=self.detector.roi_rect,
            ignore_rects=self.detector.ignore_rects,
            draw_tracks=self.config.draw_tracks,
        )

    def _update_recording(
        self,
        annotated: np.ndarray,
        now_s: float,
        frame_idx: int,
        alert: bool,
        object_count: int,
        track_ids: Sequence[int],
        labels: Sequence[str] = (),
    ) -> None:
        if self.recorder is None:
            return

        if not self.recorder.is_recording:
            if self._motion_streak >= self.min_motion_frames:
                self._start_incident(now_s, frame_idx)
            else:
                return

        if self.recorder.is_recording:
            self.recorder.write(
                annotated,
                object_count=object_count,
                track_ids=track_ids,
                labels=labels,
            )

            too_long = self.recorder.frames_written >= self.max_incident_frames
            quiet_enough = self._silent_streak >= self.post_event_frames
            if quiet_enough or too_long:
                if too_long:
                    LOGGER.info(
                        "Incident %03d reached the maximum length; splitting",
                        self._incident_counter,
                    )
                self._finish_incident(now_s, frame_idx)

    def _start_incident(self, now_s: float, frame_idx: int) -> None:
        assert self.recorder is not None
        self._incident_counter += 1
        # Credit the incident to the frame where motion actually began, but
        # never reach back past the previous incident: when a long event is
        # split at the maximum length, the motion streak still covers the part
        # already recorded, which would make every segment claim the same
        # start time and report a duration that keeps growing.
        start_s = max(0.0, now_s - (self._motion_streak - 1) / self.fps)
        start_s = max(start_s, self._last_incident_end_s)
        start_frame = max(1, frame_idx - (self._motion_streak - 1))
        start_frame = max(start_frame, self._last_incident_end_frame)
        path = self.recorder.start(self._incident_counter, start_s, start_frame)
        if path is None:
            self._incident_counter -= 1
            return

        self._incident_start_s = start_s
        self._incident_start_frame = start_frame
        for buffered in self._pre_buffer:
            self.recorder.write(buffered)
        self._pre_buffer.clear()
        LOGGER.info("Incident %03d started at %.2fs", self._incident_counter, start_s)

    def _finish_incident(self, now_s: float, frame_idx: int) -> None:
        assert self.recorder is not None
        record = self.recorder.finish(now_s, frame_idx)
        self._last_incident_end_s = now_s
        self._last_incident_end_frame = frame_idx
        if record is None:
            # Too short to keep: reuse the number for the next incident.
            self._incident_counter = max(0, self._incident_counter - 1)
            return

        self.incidents.append(record)
        LOGGER.info(
            "Incident %03d saved (%.2fs, %d frames)",
            record.index,
            record.duration_s,
            record.frame_count,
        )
        if self.on_incident is not None:
            try:
                self.on_incident(record)
            except Exception:  # pragma: no cover - callback must not break capture
                LOGGER.exception("Incident callback failed")
