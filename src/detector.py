"""Motion detection based on MOG2 background subtraction.

The detector is deliberately conservative: it would rather miss a marginal
blob than hand the incident recorder a stream of false positives.  Everything
it rejects is reported through :class:`DetectionResult` so the UI can explain
*why* nothing was flagged.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .config import DetectionConfig
from .geometry import (
    MotionBox,
    denormalise,
    merge_boxes,
    overlap_fraction,
    suppress_contained,
)

LOGGER = logging.getLogger(__name__)


@dataclass
class DetectionResult:
    """Everything the pipeline needs to know about one analysed frame."""

    boxes: List[MotionBox] = field(default_factory=list)
    foreground_ratio: float = 0.0
    warming_up: bool = False
    scene_change: bool = False
    mask: Optional[np.ndarray] = None
    """Binary foreground mask at *analysis* resolution (debug view only)."""

    @property
    def has_motion(self) -> bool:
        return bool(self.boxes)


class MotionDetector:
    """Foreground segmentation plus blob filtering.

    Fixes over the original single-function implementation:

    * a warm-up window, so the first frames (100% foreground) never alarm;
    * shadow pixels are dropped using MOG2's own shadow label instead of a
      magic grey threshold;
    * thresholds scale with frame area, so they behave the same at any
      resolution, and analysis happens at a fixed width for constant cost;
    * a whole-scene change (lights on, camera knocked, exposure jump) triggers
      a re-learn instead of an incident;
    * morphology no longer dilates the final mask, so boxes hug the object
      instead of growing by ~10 px on every side;
    * blobs are filtered on aspect ratio and fill ratio, not only on area;
    * ROI membership is decided by overlap fraction rather than by "touches
      the ROI anywhere", and ignore-zones can mask burnt-in timestamps.
    """

    def __init__(
        self,
        config: DetectionConfig,
        frame_size: Tuple[int, int],
        fps: float = 25.0,
    ) -> None:
        self.config = config
        self.source_size = frame_size
        width, height = frame_size

        # Warm-up is capped in *seconds* as well as frames: on a 6 fps CCTV
        # export a flat 25-frame warm-up would blind the camera for 4 seconds.
        fps = fps if fps and fps > 0 else 25.0
        self.warmup_frames = int(
            min(config.warmup_frames, max(8, round(config.warmup_seconds * fps)))
        )

        self.scale = 1.0
        if config.process_width and 0 < config.process_width < width:
            self.scale = config.process_width / float(width)
        self.analysis_size = (
            max(16, int(round(width * self.scale))),
            max(16, int(round(height * self.scale))),
        )
        self.upscale = 1.0 / self.scale if self.scale else 1.0

        self._subtractor = cv2.createBackgroundSubtractorMOG2(
            history=config.bg_history,
            varThreshold=config.var_threshold,
            detectShadows=True,
        )
        self._open_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (config.open_kernel, config.open_kernel)
        )
        self._close_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (config.close_kernel, config.close_kernel)
        )

        aw, ah = self.analysis_size
        self._analysis_area = float(aw * ah)
        self._roi_rect_analysis = (
            denormalise(config.roi, aw, ah) if config.roi else None
        )
        self._roi_rect_source = denormalise(config.roi, width, height) if config.roi else None
        self._ignore_rects_analysis = [
            rect
            for rect in (denormalise(zone, aw, ah) for zone in config.ignore_zones)
            if rect is not None
        ]
        self._ignore_rects_source = [
            rect
            for rect in (denormalise(zone, width, height) for zone in config.ignore_zones)
            if rect is not None
        ]

        self.frames_seen = 0
        self._relearn_left = 0

    # -- public properties -------------------------------------------------
    @property
    def roi_rect(self):
        """ROI in source-frame pixels (for drawing)."""
        return self._roi_rect_source

    @property
    def ignore_rects(self):
        """Ignore zones in source-frame pixels (for drawing)."""
        return self._ignore_rects_source

    @property
    def is_warming_up(self) -> bool:
        return self.frames_seen < self.warmup_frames

    # -- detection ---------------------------------------------------------
    def detect(self, frame: np.ndarray, motion_expected: bool = False) -> DetectionResult:
        """Analyse one frame and return the accepted motion boxes."""
        self.frames_seen += 1
        small = self._prepare(frame)

        learning_rate = self._learning_rate(motion_expected)
        raw_mask = self._subtractor.apply(small, learningRate=learning_rate)

        # MOG2 labels shadows 127 and true foreground 255; keeping only 255
        # removes the shadow "second person" that used to trail everyone.
        mask = np.where(raw_mask >= 255, np.uint8(255), np.uint8(0))
        foreground_ratio = float(np.count_nonzero(mask)) / self._analysis_area

        if self.is_warming_up:
            return DetectionResult(
                foreground_ratio=foreground_ratio, warming_up=True, mask=mask
            )

        if foreground_ratio >= self.config.max_foreground_ratio:
            # The whole scene changed: re-learn rather than alarm.
            self._relearn_left = self.config.scene_change_relearn_frames
            LOGGER.debug(
                "scene change at frame %d (foreground %.0f%%)",
                self.frames_seen,
                foreground_ratio * 100,
            )
            return DetectionResult(
                foreground_ratio=foreground_ratio, scene_change=True, mask=mask
            )

        if self._relearn_left > 0:
            self._relearn_left -= 1
            return DetectionResult(
                foreground_ratio=foreground_ratio, scene_change=True, mask=mask
            )

        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._open_kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._close_kernel)
        for rect in self._ignore_rects_analysis:
            x1, y1, x2, y2 = rect
            mask[y1:y2, x1:x2] = 0

        boxes = self._boxes_from_mask(mask)
        boxes = merge_boxes(
            boxes,
            self.config.merge_iou,
            self.config.merge_gap_x_ratio,
            self.config.merge_gap_y_ratio,
            self.config.merge_max_union_aspect,
        )
        # Merging can leave a small box nested inside the one that absorbed its
        # neighbours; two rectangles on one body is what reads as a duplicate.
        boxes = suppress_contained(boxes)
        boxes = [box for box in boxes if self._inside_roi(box)]

        width, height = self.source_size
        scaled = [box.scaled(self.upscale).clipped(width, height) for box in boxes]
        return DetectionResult(
            boxes=scaled, foreground_ratio=foreground_ratio, mask=mask
        )

    # -- internals ---------------------------------------------------------
    def _prepare(self, frame: np.ndarray) -> np.ndarray:
        if self.scale != 1.0:
            frame = cv2.resize(
                frame, self.analysis_size, interpolation=cv2.INTER_AREA
            )
        # A light blur suppresses sensor noise without smearing real edges.
        return cv2.GaussianBlur(frame, (5, 5), 0)

    def _learning_rate(self, motion_expected: bool) -> float:
        if self.is_warming_up or self._relearn_left > 0:
            return self.config.warmup_learning_rate
        if motion_expected:
            return self.config.motion_learning_rate
        return self.config.idle_learning_rate

    def _min_area_for(self, centre_y: float, height: int) -> float:
        base = self.config.min_area_ratio * self._analysis_area
        if not self.config.perspective_compensation:
            return base
        # Objects near the top of an overhead view are further away, so allow
        # them to be smaller: perspective_top_scale at the top, 1.0 at the
        # bottom, interpolated by how far down the frame the blob sits.
        top = self.config.perspective_top_scale
        position = 0.0 if height <= 0 else max(0.0, min(1.0, centre_y / height))
        return base * (top + (1.0 - top) * position)

    def _min_side_for(self, centre_y: float, height: int) -> float:
        """Distant objects are allowed to be thinner as well as smaller."""
        base = float(self.config.min_box_side)
        if not self.config.perspective_compensation:
            return base
        top = max(0.4, self.config.perspective_top_scale)
        position = 0.0 if height <= 0 else max(0.0, min(1.0, centre_y / height))
        return base * (top + (1.0 - top) * position)

    def _boxes_from_mask(self, mask: np.ndarray) -> List[MotionBox]:
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        height = mask.shape[0]
        max_area = self.config.max_area_ratio * self._analysis_area
        boxes: List[MotionBox] = []

        for contour in contours:
            area = float(cv2.contourArea(contour))
            x, y, w, h = cv2.boundingRect(contour)
            min_side = self._min_side_for(y + h / 2.0, height)
            if w < min_side or h < min_side:
                continue

            box_area = float(w * h)
            if area < self._min_area_for(y + h / 2.0, height) or area > max_area:
                continue
            if box_area > 0 and area / box_area < self.config.min_fill_ratio:
                # Scattered noise fills its bounding box poorly; a person or a
                # vehicle fills roughly 40-70% of it.
                continue

            aspect = w / float(h) if h else 0.0
            if not (self.config.min_aspect_ratio <= aspect <= self.config.max_aspect_ratio):
                continue

            boxes.append(MotionBox(x, y, w, h, area))

        return boxes

    def _inside_roi(self, box: MotionBox) -> bool:
        if self._roi_rect_analysis is None:
            return True
        return overlap_fraction(box, self._roi_rect_analysis) >= self.config.roi_overlap
