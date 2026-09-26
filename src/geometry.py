"""Small geometry helpers shared by the detector, tracker and annotator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

Rect = Tuple[int, int, int, int]  # x1, y1, x2, y2 in pixels
NormRect = Tuple[float, float, float, float]  # x1, y1, x2, y2 in 0..1


@dataclass(frozen=True)
class MotionBox:
    """An axis-aligned motion region in *source frame* pixel coordinates."""

    x: int
    y: int
    w: int
    h: int
    area: float = 0.0
    """Area of the underlying contour (not of the bounding box)."""

    @property
    def x2(self) -> int:
        return self.x + self.w

    @property
    def y2(self) -> int:
        return self.y + self.h

    @property
    def centroid(self) -> Tuple[float, float]:
        return (self.x + self.w / 2.0, self.y + self.h / 2.0)

    @property
    def rect(self) -> Rect:
        return (self.x, self.y, self.x2, self.y2)

    def scaled(self, factor: float) -> "MotionBox":
        return MotionBox(
            int(round(self.x * factor)),
            int(round(self.y * factor)),
            int(round(self.w * factor)),
            int(round(self.h * factor)),
            self.area * factor * factor,
        )

    def clipped(self, width: int, height: int) -> "MotionBox":
        x1 = max(0, min(self.x, width - 1))
        y1 = max(0, min(self.y, height - 1))
        x2 = max(x1 + 1, min(self.x2, width))
        y2 = max(y1 + 1, min(self.y2, height))
        return MotionBox(x1, y1, x2 - x1, y2 - y1, self.area)


def rect_area(rect: Rect) -> float:
    return max(0, rect[2] - rect[0]) * max(0, rect[3] - rect[1])


def intersection_area(a: Rect, b: Rect) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    return float((x2 - x1) * (y2 - y1))


def iou(a: MotionBox, b: MotionBox) -> float:
    inter = intersection_area(a.rect, b.rect)
    if inter <= 0:
        return 0.0
    union = rect_area(a.rect) + rect_area(b.rect) - inter
    return inter / union if union > 0 else 0.0


def overlap_fraction(box: MotionBox, rect: Optional[Rect]) -> float:
    """How much of ``box`` falls inside ``rect`` (1.0 when rect is None)."""
    if rect is None:
        return 1.0
    box_area = rect_area(box.rect)
    if box_area <= 0:
        return 0.0
    return intersection_area(box.rect, rect) / box_area


def denormalise(rect: NormRect, width: int, height: int) -> Optional[Rect]:
    """Convert a normalised rectangle to pixels, or None when degenerate."""
    x1r, y1r, x2r, y2r = rect
    x1r, x2r = sorted((x1r, x2r))
    y1r, y2r = sorted((y1r, y2r))
    x1 = int(max(0.0, min(1.0, x1r)) * width)
    y1 = int(max(0.0, min(1.0, y1r)) * height)
    x2 = int(max(0.0, min(1.0, x2r)) * width)
    y2 = int(max(0.0, min(1.0, y2r)) * height)
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def merge_boxes(
    boxes: Sequence[MotionBox],
    iou_threshold: float = 0.05,
    gap_x_ratio: float = 0.18,
    gap_y_ratio: float = 0.85,
    max_union_aspect: float = 1.35,
) -> List[MotionBox]:
    """Merge overlapping or adjacent boxes so one person is one box.

    Background subtraction rarely returns a whole body: a person walking past
    boxes comes back as a head, a torso and a pair of legs, which the tracker
    then reports as three objects with three ids.  The pieces of one body are
    stacked *vertically* and overlap *horizontally*, so the gap between two
    boxes is measured edge to edge and a generous vertical gap is allowed while
    the horizontal gap must be small.  Centre-to-centre distance (the previous
    rule) cannot express that: it treats a tall thin body as a blob and merges
    two people standing side by side as readily as one person's own fragments.

    Runs to a fixed point, with each pass bounded so pathological inputs cannot
    spin.
    """
    current: List[MotionBox] = list(boxes)
    if len(current) < 2:
        return current

    for _pass in range(len(boxes)):
        merged: List[MotionBox] = []
        consumed = [False] * len(current)
        changed = False

        for i, box in enumerate(current):
            if consumed[i]:
                continue
            x1, y1, x2, y2 = box.rect
            area = box.area
            consumed[i] = True

            for j in range(i + 1, len(current)):
                if consumed[j]:
                    continue
                other = current[j]
                group = MotionBox(x1, y1, x2 - x1, y2 - y1, area)
                if _should_merge(
                    group, other, iou_threshold, gap_x_ratio, gap_y_ratio, max_union_aspect
                ):
                    ox1, oy1, ox2, oy2 = other.rect
                    x1, y1 = min(x1, ox1), min(y1, oy1)
                    x2, y2 = max(x2, ox2), max(y2, oy2)
                    area += other.area
                    consumed[j] = True
                    changed = True

            merged.append(MotionBox(x1, y1, x2 - x1, y2 - y1, area))

        current = merged
        if not changed:
            break

    return current


def _should_merge(
    a: MotionBox,
    b: MotionBox,
    iou_threshold: float,
    gap_x_ratio: float,
    gap_y_ratio: float,
    max_union_aspect: float = 1.35,
) -> bool:
    union_w = max(a.x2, b.x2) - min(a.x, b.x)
    union_h = max(a.y2, b.y2) - min(a.y, b.y)

    # A body's own fragments stack up, so the box they form stays upright.
    # Without this guard merging chains sideways and swallows the person
    # standing next to the one it started with.
    if union_h > 0 and union_w / union_h > max_union_aspect and iou(a, b) < 0.3:
        return False

    if iou(a, b) >= iou_threshold:
        return True

    # Edge-to-edge gaps; zero when the boxes overlap on that axis.
    gap_x = max(0, max(a.x, b.x) - min(a.x2, b.x2))
    gap_y = max(0, max(a.y, b.y) - min(a.y2, b.y2))

    reach_x = gap_x_ratio * max(a.w, b.w)
    reach_y = gap_y_ratio * max(a.h, b.h)
    return gap_x <= reach_x and gap_y <= reach_y


def suppress_contained(boxes: Sequence[MotionBox], overlap: float = 0.85) -> List[MotionBox]:
    """Drop boxes that sit almost entirely inside a larger one.

    Merging can leave a small box nested in the box that swallowed its
    neighbours; two rectangles around the same person is exactly the duplicate
    the operator sees as "two ids on one person".
    """
    ordered = sorted(boxes, key=lambda box: rect_area(box.rect), reverse=True)
    kept: List[MotionBox] = []
    for box in ordered:
        area = rect_area(box.rect) or 1.0
        if any(intersection_area(box.rect, other.rect) / area >= overlap for other in kept):
            continue
        kept.append(box)
    return kept


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))
