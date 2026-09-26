"""ARCHIVED: the original (v1) detector, kept for reference and benchmarking.

This file is the implementation that shipped before the rebuild.  It is not
imported by the application; `tests/test_regression_v1.py` and the A/B bench
script use it to show, in numbers, which behaviours were fixed.

Known defects (all reproduced by the benchmark):
  * no background warm-up, so frame 1 is 100% foreground and every video
    opens with a false incident;
  * a whole-scene light change is reported as motion;
  * pixel thresholds are absolute, so behaviour changes with resolution;
  * the mask is dilated with an 11x11 kernel and never eroded back, inflating
    every box;
  * unmatched detections count as "moving", which defeats the movement filter;
  * the greedy tracker loop can drop valid matches;
  * the trigger frame is written into the clip twice.
"""

import datetime as dt
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


class CentroidTracker:
    """Track moving objects across frames using centroid matching."""

    def __init__(self, max_disappeared: int = 30) -> None:
        self.next_object_id = 0
        self.objects: Dict[int, Tuple[float, float]] = {}
        self.disappeared: Dict[int, int] = {}
        self.max_disappeared = max_disappeared
        self.history: Dict[int, deque] = {}
        self.match_distance = 75

    def register(self, centroid: Tuple[float, float]) -> None:
        self.objects[self.next_object_id] = centroid
        self.disappeared[self.next_object_id] = 0
        self.history[self.next_object_id] = deque(maxlen=8)
        self.history[self.next_object_id].append(centroid)
        self.next_object_id += 1

    def deregister(self, object_id: int) -> None:
        del self.objects[object_id]
        del self.disappeared[object_id]
        if object_id in self.history:
            del self.history[object_id]

    def update(self, boxes: List["MotionBox"]) -> Dict[int, Tuple[float, float]]:
        if len(boxes) == 0:
            for obj_id in list(self.disappeared.keys()):
                self.disappeared[obj_id] += 1
                if self.disappeared[obj_id] > self.max_disappeared:
                    self.deregister(obj_id)
            return self.objects

        input_centroids = np.zeros((len(boxes), 2))
        for i, box in enumerate(boxes):
            input_centroids[i] = (box.x + box.w // 2, box.y + box.h // 2)

        if len(self.objects) == 0:
            for i in range(len(input_centroids)):
                self.register(tuple(input_centroids[i]))
            return self.objects

        object_ids = list(self.objects.keys())
        object_centroids = np.array([self.objects[oid] for oid in object_ids])

        distances = np.zeros((len(object_centroids), len(input_centroids)))
        for i, obj_centroid in enumerate(object_centroids):
            for j, input_centroid in enumerate(input_centroids):
                distances[i, j] = np.linalg.norm(obj_centroid - input_centroid)

        matched_objects = set()
        matched_inputs = set()

        for _ in range(min(len(object_ids), len(input_centroids))):
            i, j = np.unravel_index(np.argmin(distances), distances.shape)
            if i in matched_objects or j in matched_inputs:
                distances[i, j] = np.inf
                continue
            if distances[i, j] < self.match_distance:
                oid = object_ids[i]
                self.objects[oid] = tuple(input_centroids[j])
                if oid not in self.history:
                    self.history[oid] = deque(maxlen=5)
                self.history[oid].append(tuple(input_centroids[j]))
                self.disappeared[oid] = 0
                matched_objects.add(i)
                matched_inputs.add(j)
                distances[i, j] = np.inf
            else:
                break

        for j in range(len(input_centroids)):
            if j not in matched_inputs:
                self.register(tuple(input_centroids[j]))

        for i in range(len(object_ids)):
            if i not in matched_objects:
                oid = object_ids[i]
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.max_disappeared:
                    self.deregister(oid)

        return self.objects

    def is_moving(
        self, object_id: int, movement_thresh: float = 5.0, required_frames: int = 2
    ) -> bool:
        if object_id not in self.history:
            return False
        hist = list(self.history[object_id])
        if len(hist) < 2:
            return False
        count = 0
        for a, b in zip(hist[:-1], hist[1:]):
            dx, dy = a[0] - b[0], a[1] - b[1]
            if (dx * dx + dy * dy) ** 0.5 >= movement_thresh:
                count += 1
        return count >= required_frames


@dataclass
class MotionBox:
    x: int
    y: int
    w: int
    h: int


@dataclass
class Incident:
    start_time_s: float
    end_time_s: float
    output_path: Path


class SmartSecurityCamera:
    def __init__(
        self,
        video_path: Optional[Path],
        output_dir: Path,
        min_area: int = 900,
        min_motion_frames: int = 8,
        pre_event_frames: int = 30,
        post_event_frames: int = 30,
        roi: Optional[Tuple[float, float, float, float]] = None,
        enable_tracking: bool = False,
        enable_live: bool = False,
        persistence_frames: int = 3,
        persistence_threshold: int = 2,
        movement_thresh: float = 5.0,
        movement_required_frames: int = 2,
    ) -> None:
        self.video_path = video_path
        self.output_dir = output_dir
        self.min_area = min_area
        self.min_motion_frames = min_motion_frames
        self.pre_event_frames = pre_event_frames
        self.post_event_frames = post_event_frames
        self.roi = roi
        self.enable_tracking = enable_tracking
        self.enable_live = enable_live
        self.persistence_frames = persistence_frames
        self.persistence_threshold = persistence_threshold
        self._mask_history = deque(maxlen=self.persistence_frames)
        self.movement_thresh = movement_thresh
        self.movement_required_frames = movement_required_frames
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _timestamp_string(frame_idx: int, fps: float) -> str:
        seconds = frame_idx / fps if fps > 0 else 0
        base_time = dt.datetime(2000, 1, 1) + dt.timedelta(seconds=seconds)
        return base_time.strftime("%H:%M:%S")

    @staticmethod
    def _frame_time_seconds(frame_idx: int, fps: float) -> float:
        return frame_idx / fps if fps > 0 else 0.0

    def _roi_rect(self, width: int, height: int):
        if self.roi is None:
            return None
        x1r, y1r, x2r, y2r = self.roi
        x1 = int(max(0.0, min(1.0, x1r)) * width)
        y1 = int(max(0.0, min(1.0, y1r)) * height)
        x2 = int(max(0.0, min(1.0, x2r)) * width)
        y2 = int(max(0.0, min(1.0, y2r)) * height)
        if x2 <= x1 or y2 <= y1:
            return None
        return x1, y1, x2, y2

    @staticmethod
    def _intersects(box: MotionBox, rect) -> bool:
        if rect is None:
            return True
        rx1, ry1, rx2, ry2 = rect
        bx1, by1, bx2, by2 = box.x, box.y, box.x + box.w, box.y + box.h
        return not (bx2 < rx1 or bx1 > rx2 or by2 < ry1 or by1 > ry2)

    @staticmethod
    def _merge_boxes(
        boxes: List[MotionBox], iou_thresh: float = 0.15, dist_thresh: float = 0.5
    ) -> List[MotionBox]:
        if not boxes:
            return []
        rects = [(b.x, b.y, b.x + b.w, b.y + b.h) for b in boxes]
        used = [False] * len(rects)
        merged: List[MotionBox] = []

        def iou(a, b):
            ax1, ay1, ax2, ay2 = a
            bx1, by1, bx2, by2 = b
            ix1, iy1 = max(ax1, bx1), max(ay1, by1)
            ix2, iy2 = min(ax2, bx2), min(ay2, by2)
            if ix2 <= ix1 or iy2 <= iy1:
                return 0.0
            inter = (ix2 - ix1) * (iy2 - iy1)
            area_a = (ax2 - ax1) * (ay2 - ay1)
            area_b = (bx2 - bx1) * (by2 - by1)
            return inter / float(area_a + area_b - inter)

        for i, a in enumerate(rects):
            if used[i]:
                continue
            ax1, ay1, ax2, ay2 = a
            used[i] = True
            changed = True
            while changed:
                changed = False
                for j, b in enumerate(rects):
                    if used[j]:
                        continue
                    if iou((ax1, ay1, ax2, ay2), b) >= iou_thresh:
                        bx1, by1, bx2, by2 = b
                        ax1, ay1 = min(ax1, bx1), min(ay1, by1)
                        ax2, ay2 = max(ax2, bx2), max(ay2, by2)
                        used[j] = True
                        changed = True
                    else:
                        acx, acy = (ax1 + ax2) / 2.0, (ay1 + ay2) / 2.0
                        bcx, bcy = (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0
                        max_dim = max(ax2 - ax1, ay2 - ay1, b[2] - b[0], b[3] - b[1])
                        dist = ((acx - bcx) ** 2 + (acy - bcy) ** 2) ** 0.5
                        if dist <= dist_thresh * max_dim:
                            bx1, by1, bx2, by2 = b
                            ax1, ay1 = min(ax1, bx1), min(ay1, by1)
                            ax2, ay2 = max(ax2, bx2), max(ay2, by2)
                            used[j] = True
                            changed = True
            merged.append(MotionBox(int(ax1), int(ay1), int(ax2 - ax1), int(ay2 - ay1)))
        return merged

    def _detect_motion(self, frame, subtractor, roi_rect) -> List[MotionBox]:
        height, width = frame.shape[:2]
        fg_mask = subtractor.apply(frame)
        fg_mask = cv2.GaussianBlur(fg_mask, (5, 5), 0)
        _, thresh = cv2.threshold(fg_mask, 160, 255, cv2.THRESH_BINARY)

        mask_bin = (thresh > 0).astype(np.uint8)
        self._mask_history.append(mask_bin)
        eff_th = min(self.persistence_threshold, max(1, len(self._mask_history)))
        accumulated = np.sum(np.stack(list(self._mask_history), axis=0), axis=0)
        persistent_bin = (accumulated >= eff_th).astype(np.uint8) * 255

        kernel_person = np.ones((11, 11), np.uint8)
        persistent = cv2.morphologyEx(persistent_bin, cv2.MORPH_OPEN, kernel_person, iterations=1)
        persistent = cv2.morphologyEx(persistent, cv2.MORPH_CLOSE, kernel_person, iterations=1)
        persistent = cv2.dilate(persistent, kernel_person, iterations=1)

        contours, _ = cv2.findContours(persistent, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        boxes: List[MotionBox] = []
        for contour in contours:
            area = cv2.contourArea(contour)
            x, y, w, h = cv2.boundingRect(contour)
            box = MotionBox(x, y, w, h)
            center_y = y + (h / 2.0)
            if center_y < height * 0.35:
                area_threshold = max(40, int(self.min_area * 0.45))
            elif center_y < height * 0.7:
                area_threshold = max(60, int(self.min_area * 0.7))
            else:
                area_threshold = self.min_area
            if area < area_threshold:
                continue
            if self._intersects(box, roi_rect):
                boxes.append(box)

        return SmartSecurityCamera._merge_boxes(boxes, iou_thresh=0.12, dist_thresh=0.6)

    @staticmethod
    def _match_boxes_to_tracked_objects(boxes, tracked_objects):
        if not boxes or not tracked_objects:
            return [None] * len(boxes)
        remaining_ids = set(tracked_objects.keys())
        assignments = []
        for box in boxes:
            box_cx = box.x + box.w / 2.0
            box_cy = box.y + box.h / 2.0
            best_id, best_distance = None, float("inf")
            for obj_id in remaining_ids:
                cx, cy = tracked_objects[obj_id]
                distance = ((box_cx - cx) ** 2 + (box_cy - cy) ** 2) ** 0.5
                if distance < best_distance:
                    best_distance, best_id = distance, obj_id
            if best_id is not None:
                remaining_ids.remove(best_id)
                assignments.append(best_id)
            else:
                assignments.append(None)
        return assignments

    def detect_only(self) -> List[Incident]:
        """Run the v1 state machine without writing clips (benchmark helper)."""
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise RuntimeError("Could not open video")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        roi_rect = self._roi_rect(width, height)
        subtractor = cv2.createBackgroundSubtractorMOG2(
            history=150, varThreshold=12, detectShadows=False
        )
        tracker = CentroidTracker() if self.enable_tracking else None

        frame_idx = 0
        incident_active = False
        motion_streak = 0
        silent_streak = 0
        start_s = 0.0
        incidents: List[Incident] = []

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1
            boxes = self._detect_motion(frame, subtractor, roi_rect)

            if self.enable_tracking and tracker is not None:
                tracked = tracker.update(boxes)
                matched_ids = self._match_boxes_to_tracked_objects(boxes, tracked)
                moving_ids = {
                    mid
                    for mid in {m for m in matched_ids if m is not None}
                    if tracker.is_moving(mid, self.movement_thresh, self.movement_required_frames)
                }
                moving_boxes = sum(
                    1
                    for i, _box in enumerate(boxes)
                    if (matched_ids[i] if i < len(matched_ids) else None) is None
                    or matched_ids[i] in moving_ids
                )
                unauthorized = moving_boxes > 0
            else:
                unauthorized = len(boxes) > 0

            if unauthorized:
                motion_streak += 1
                silent_streak = 0
            else:
                silent_streak += 1
                motion_streak = 0

            if not incident_active and motion_streak >= self.min_motion_frames:
                incident_active = True
                start_s = self._frame_time_seconds(frame_idx, fps)
            elif incident_active and silent_streak >= self.post_event_frames:
                incident_active = False
                incidents.append(
                    Incident(start_s, self._frame_time_seconds(frame_idx, fps), Path("-"))
                )
                motion_streak = silent_streak = 0

        if incident_active:
            incidents.append(
                Incident(start_s, self._frame_time_seconds(frame_idx, fps), Path("-"))
            )
        cap.release()
        return incidents
