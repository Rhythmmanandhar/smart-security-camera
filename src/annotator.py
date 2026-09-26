"""Frame annotation: the evidence overlay drawn on every recorded frame."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .geometry import MotionBox, Rect
from .tracker import Track

FONT = cv2.FONT_HERSHEY_SIMPLEX

COLOR_ALERT = (60, 60, 235)
COLOR_WATCH = (80, 200, 255)
COLOR_IDLE = (120, 200, 120)
COLOR_ROI = (255, 190, 60)
COLOR_IGNORE = (120, 120, 120)
COLOR_TEXT = (255, 255, 255)
COLOR_PANEL = (32, 32, 32)


@dataclass
class OverlayState:
    """What the HUD should say about the current frame."""

    frame_idx: int = 0
    fps: float = 25.0
    timestamp: str = ""
    alert: bool = False
    warming_up: bool = False
    scene_change: bool = False
    recording: bool = False
    incident_index: Optional[int] = None
    incident_elapsed_s: float = 0.0
    incident_count: int = 0
    measured_fps: Optional[float] = None
    source_label: str = ""


def media_timestamp(frame_idx: int, fps: float) -> str:
    """HH:MM:SS position inside a recorded file."""
    seconds = frame_idx / fps if fps > 0 else 0.0
    return str(dt.timedelta(seconds=int(seconds))).rjust(8, "0")


def wall_clock_timestamp(when: Optional[dt.datetime] = None) -> str:
    return (when or dt.datetime.now()).strftime("%Y-%m-%d %H:%M:%S")


def _text_with_background(
    frame: np.ndarray,
    text: str,
    origin: Tuple[int, int],
    scale: float = 0.55,
    color: Tuple[int, int, int] = COLOR_TEXT,
    thickness: int = 1,
    pad: int = 5,
) -> Tuple[int, int, int, int]:
    """Draw text on a filled plate, kept inside the frame.  Returns the plate."""
    (tw, th), baseline = cv2.getTextSize(text, FONT, scale, thickness)
    height, width = frame.shape[:2]

    x = int(min(max(pad, origin[0]), max(pad, width - tw - pad)))
    y = int(min(max(th + pad, origin[1]), height - baseline - pad))

    plate = (x - pad, y - th - pad, x + tw + pad, y + baseline + pad - 2)
    cv2.rectangle(frame, (plate[0], plate[1]), (plate[2], plate[3]), COLOR_PANEL, -1)
    cv2.putText(frame, text, (x, y), FONT, scale, color, thickness, cv2.LINE_AA)
    return plate


def _plate_for(
    frame: np.ndarray, text: str, origin: Tuple[int, int], scale: float, pad: int = 5
) -> Rect:
    """Where :func:`_text_with_background` would put a plate, without drawing."""
    (tw, th), baseline = cv2.getTextSize(text, FONT, scale, 1)
    height, width = frame.shape[:2]
    x = int(min(max(pad, origin[0]), max(pad, width - tw - pad)))
    y = int(min(max(th + pad, origin[1]), height - baseline - pad))
    return (x - pad, y - th - pad, x + tw + pad, y + baseline + pad - 2)


def _plates_overlap(a: Rect, b: Rect) -> bool:
    return not (a[2] <= b[0] or a[0] >= b[2] or a[3] <= b[1] or a[1] >= b[3])


def draw_overlay(
    frame: np.ndarray,
    boxes: Sequence[MotionBox],
    tracks: Sequence[Track],
    alerting_ids: Iterable[int],
    state: OverlayState,
    roi_rect: Optional[Rect] = None,
    ignore_rects: Sequence[Rect] = (),
    draw_tracks: bool = True,
) -> np.ndarray:
    """Draw the full evidence overlay and return a new frame.

    Boxes are drawn per *track* rather than per raw contour, so a label always
    belongs to the object under it - the old code matched labels to boxes with
    a separate nearest-neighbour pass that could hand a box someone else's ID.
    """
    annotated = frame.copy()
    height, width = annotated.shape[:2]
    alerting = set(alerting_ids)

    if roi_rect is not None:
        x1, y1, x2, y2 = roi_rect
        cv2.rectangle(annotated, (x1, y1), (x2, y2), COLOR_ROI, 2)
        _text_with_background(
            annotated, "Monitored zone", (x1 + 6, max(22, y1 + 22)), 0.5, COLOR_ROI
        )

    for rect in ignore_rects:
        x1, y1, x2, y2 = rect
        overlay = annotated.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), COLOR_IGNORE, -1)
        cv2.addWeighted(overlay, 0.25, annotated, 0.75, 0, annotated)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), COLOR_IGNORE, 1)

    if draw_tracks:
        # Draw calm tracks first so an alerting box is never hidden behind one.
        ordered = sorted(tracks, key=lambda t: t.track_id in alerting)
        used_plates: List[Rect] = []
        for track in ordered:
            is_alert = track.track_id in alerting
            color = COLOR_ALERT if is_alert else COLOR_WATCH
            box = track.display_box or track.box
            cv2.rectangle(annotated, (box.x, box.y), (box.x2, box.y2), color, 2)

            # Red already means "moving", so the label stays short; long labels
            # collided with each other in busy scenes.  The class, when a
            # classifier is running, is the one detail worth the extra width.
            # "person 3" reads better than "ID 3 - person": the class is the
            # useful part, the number only distinguishes two of the same kind.
            if track.label and track.label != "motion":
                label = f"{track.label} {track.track_id}"
            else:
                label = f"ID {track.track_id}"
            label_y = max(18, box.y - 8)
            for _attempt in range(4):
                plate = _plate_for(annotated, label, (box.x + 4, label_y), 0.45)
                if not any(_plates_overlap(plate, other) for other in used_plates):
                    break
                label_y += plate[3] - plate[1] + 2
            plate = _text_with_background(
                annotated, label, (box.x + 4, label_y), 0.45, color
            )
            used_plates.append(plate)

            points = list(track.history)
            for previous, current in zip(points[:-1], points[1:]):
                cv2.line(
                    annotated,
                    (int(previous[0]), int(previous[1])),
                    (int(current[0]), int(current[1])),
                    color,
                    1,
                    cv2.LINE_AA,
                )
            cx, cy = track.centroid
            cv2.circle(annotated, (int(cx), int(cy)), 3, color, -1)
    else:
        for box in boxes:
            color = COLOR_ALERT if state.alert else COLOR_WATCH
            cv2.rectangle(annotated, (box.x, box.y), (box.x2, box.y2), color, 2)

    _draw_hud(annotated, state, width, height)
    return annotated


def _draw_hud(frame: np.ndarray, state: OverlayState, width: int, height: int) -> None:
    banner_height = 34
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (width, banner_height), COLOR_PANEL, -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    timestamp = state.timestamp or media_timestamp(state.frame_idx, state.fps)
    cv2.putText(frame, timestamp, (12, 23), FONT, 0.6, COLOR_TEXT, 1, cv2.LINE_AA)

    if state.source_label:
        label = state.source_label[:42]
        (tw, _th), _baseline = cv2.getTextSize(label, FONT, 0.5, 1)
        cv2.putText(
            frame, label, (max(12, width - 12 - tw), 23), FONT, 0.5,
            (200, 200, 200), 1, cv2.LINE_AA,
        )

    if state.warming_up:
        status, color = "CALIBRATING", (200, 200, 60)
    elif state.scene_change:
        status, color = "SCENE CHANGE - RELEARNING", (200, 160, 60)
    elif state.alert:
        status, color = "ALERT: Unauthorised motion", COLOR_ALERT
    else:
        status, color = "Monitoring - no incident", COLOR_IDLE

    _text_with_background(frame, status, (12, banner_height + 26), 0.6, color, 2)

    info: List[str] = []
    if state.recording and state.incident_index is not None:
        info.append(f"REC incident {state.incident_index:03d} ({state.incident_elapsed_s:.1f}s)")
    if state.incident_count:
        info.append(f"incidents: {state.incident_count}")
    if state.measured_fps:
        info.append(f"{state.measured_fps:.1f} fps")

    if info:
        _text_with_background(
            frame, "  |  ".join(info), (12, height - 14), 0.5, COLOR_TEXT
        )

    if state.recording:
        cv2.circle(frame, (width - 24, banner_height + 22), 8, COLOR_ALERT, -1)
