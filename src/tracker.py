"""Centroid tracking with confirmation and displacement-based motion tests."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Sequence, Tuple

from .config import TrackingConfig
from .geometry import MotionBox, iou


@dataclass
class Track:
    """One tracked object."""

    track_id: int
    box: MotionBox
    centroid: Tuple[float, float]
    first_frame: int
    last_frame: int
    hits: int = 1
    missing: int = 0
    history: Deque[Tuple[float, float]] = field(default_factory=lambda: deque(maxlen=12))
    label: Optional[str] = None
    """Class assigned by the optional classifier (None = not checked yet)."""
    display_box: Optional[MotionBox] = None
    """Where to draw the object on frames between detector runs."""
    label_confidence: float = 0.0
    label_frame: int = -1
    """Frame the classification was last refreshed on."""
    classified: bool = False
    """True once the track has been through the classifier at least once."""

    @property
    def displacement(self) -> float:
        """How far the object actually travelled across the history window.

        Measured between the mean position of the first half of the window and
        the mean of the second half.  A raw first-to-last distance looks large
        for something that merely oscillates, because the window can end on the
        opposite phase of the wobble; averaging each half cancels that out
        while still growing linearly for genuine travel.
        """
        points = list(self.history)
        if len(points) < 2:
            return 0.0
        if len(points) < 4:
            (x0, y0), (x1, y1) = points[0], points[-1]
            return math.hypot(x1 - x0, y1 - y0)

        middle = len(points) // 2
        first, second = points[:middle], points[middle:]
        fx = sum(p[0] for p in first) / len(first)
        fy = sum(p[1] for p in first) / len(first)
        sx = sum(p[0] for p in second) / len(second)
        sy = sum(p[1] for p in second) / len(second)
        return math.hypot(sx - fx, sy - fy)

    @property
    def velocity(self) -> Tuple[float, float]:
        """Pixels per frame, averaged over the last few observations."""
        points = list(self.history)
        if len(points) < 2:
            return (0.0, 0.0)
        recent = points[-4:]
        steps = len(recent) - 1
        return (
            (recent[-1][0] - recent[0][0]) / steps,
            (recent[-1][1] - recent[0][1]) / steps,
        )

    @property
    def net_displacement(self) -> float:
        """Straight-line distance between the oldest and newest centroid."""
        if len(self.history) < 2:
            return 0.0
        (x0, y0), (x1, y1) = self.history[0], self.history[-1]
        return math.hypot(x1 - x0, y1 - y0)

    @property
    def path_length(self) -> float:
        """Total distance travelled across the history window."""
        points = list(self.history)
        return sum(
            math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(points[:-1], points[1:])
        )


class CentroidTracker:
    """Greedy nearest-centroid tracker.

    Differences from a naive implementation, all of which were real bugs in the
    first version of this project:

    * matching iterates over *every* candidate pair instead of a fixed number
      of rounds, so a pair that is skipped no longer costs a valid match;
    * the gating distance is a fraction of the frame diagonal, so tracking
      behaves the same at any resolution;
    * bounding-box overlap breaks ties, which keeps identities stable when two
      people cross;
    * a track must be seen ``min_hits`` times before anything trusts it.
    """

    def __init__(self, config: TrackingConfig, frame_size: Tuple[int, int]) -> None:
        self.config = config
        self.frame_size = frame_size
        self.diagonal = math.hypot(*frame_size) or 1.0
        self.max_distance = self.diagonal * config.max_match_distance_ratio
        self.min_displacement = self.diagonal * config.min_displacement_ratio
        self.tracks: Dict[int, Track] = {}
        self._next_id = 1

    # -- internals ---------------------------------------------------------
    def _register(self, box: MotionBox, frame_idx: int) -> Track:
        track = Track(
            track_id=self._next_id,
            box=box,
            centroid=box.centroid,
            first_frame=frame_idx,
            last_frame=frame_idx,
            history=deque(maxlen=self.config.history_frames),
        )
        track.history.append(box.centroid)
        self.tracks[track.track_id] = track
        self._next_id += 1
        return track

    # -- public API --------------------------------------------------------
    def update(self, boxes: Sequence[MotionBox], frame_idx: int) -> List[Track]:
        """Advance the tracker by one frame and return the live tracks."""
        unmatched_boxes = set(range(len(boxes)))
        unmatched_tracks = set(self.tracks)

        # Score every (track, detection) pair once, then consume them greedily
        # in ascending cost order.  Overlapping boxes get a cost bonus.
        candidates: List[Tuple[float, int, int]] = []
        for track_id in unmatched_tracks:
            track = self.tracks[track_id]
            for box_idx, box in enumerate(boxes):
                distance = math.dist(track.centroid, box.centroid)
                if distance > self.max_distance:
                    continue
                overlap = iou(track.box, box)
                candidates.append((distance * (1.0 - 0.5 * overlap), track_id, box_idx))

        for _cost, track_id, box_idx in sorted(candidates):
            if track_id not in unmatched_tracks or box_idx not in unmatched_boxes:
                continue
            box = boxes[box_idx]
            track = self.tracks[track_id]
            track.box = box
            track.centroid = box.centroid
            track.history.append(box.centroid)
            track.hits += 1
            track.missing = 0
            track.last_frame = frame_idx
            unmatched_tracks.discard(track_id)
            unmatched_boxes.discard(box_idx)

        for box_idx in sorted(unmatched_boxes):
            self._register(boxes[box_idx], frame_idx)

        for track_id in unmatched_tracks:
            track = self.tracks[track_id]
            track.missing += 1
            if track.missing > self.config.max_missing_frames:
                del self.tracks[track_id]

        return list(self.tracks.values())

    def is_confirmed(self, track: Track) -> bool:
        """A track seen often enough to be trusted."""
        return track.hits >= self.config.min_hits and track.missing == 0

    def is_moving(self, track: Track) -> bool:
        """True when the track actually travelled, rather than jittered.

        Displacement is used on purpose: a noisy blob wobbling around one spot
        covers a long *path* but displaces almost nothing, which is exactly the
        false alarm that per-step movement tests (v1) could not reject.
        """
        if len(track.history) < max(2, self.config.min_hits):
            return False
        return track.displacement >= self.min_displacement

    def visible_tracks(self) -> List[Track]:
        return [t for t in self.tracks.values() if t.missing == 0]
