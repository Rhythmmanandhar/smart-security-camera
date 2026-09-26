"""Object classification: decide *what* moved, not just *that* something did.

Motion detection alone cannot tell a person from a swinging door, a cat or a
forklift.  A classifier turns "something moved" into "a person moved", which is
what an alarm should really be based on.

Two backends are available:

``hog``
    OpenCV's built-in HOG people detector.  No extra dependency, but it only
    finds upright, roughly side-on people and is slow per region.  It is kept
    for machines where the YOLO dependency is unwelcome.

``yolo``
    YOLOv8n via `ultralytics`.  One pass over the whole frame finds every
    object, so the cost does not grow with the number of tracks, and it
    recognises vehicles and animals as well as people.  The weights file is
    ~6 MB and is downloaded once into ``models/``.

Both are optional: with no classifier, every moving track alarms.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .geometry import MotionBox, intersection_area, rect_area

LOGGER = logging.getLogger(__name__)

# Classes worth alarming on at a typical site.  Everything else that YOLO
# recognises (chairs, boxes, plants, TVs...) is treated as "not a threat".
DEFAULT_CLASSES: Tuple[str, ...] = (
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "bus",
    "truck",
    "cat",
    "dog",
)

DEFAULT_MODEL = Path(os.environ.get("SC_YOLO_MODEL", "models/yolov8n.pt"))


@dataclass(frozen=True)
class ObjectDetection:
    """One object the model found: where it is and what it is."""

    box: MotionBox
    label: str
    confidence: float


@dataclass(frozen=True)
class Classification:
    """What a classifier decided about one region."""

    label: str
    confidence: float

    def __str__(self) -> str:  # pragma: no cover - display only
        return f"{self.label} {self.confidence:.2f}"


class BaseClassifier:
    """Interface: turn motion boxes into optional classifications."""

    name = "none"
    available = True

    def classify(
        self, frame: np.ndarray, boxes: Sequence[MotionBox]
    ) -> List[Optional[Classification]]:
        raise NotImplementedError

    def describe(self) -> str:
        return self.name


class NullClassifier(BaseClassifier):
    """Accepts everything - the behaviour when no classifier is configured."""

    name = "none"

    def classify(
        self, frame: np.ndarray, boxes: Sequence[MotionBox]
    ) -> List[Optional[Classification]]:
        return [Classification("motion", 1.0) for _ in boxes]


class HogPersonClassifier(BaseClassifier):
    """OpenCV HOG people detector, run on a padded crop of each region."""

    name = "hog"

    def __init__(self, confidence: float = 0.0) -> None:
        self._hog = cv2.HOGDescriptor()
        self._hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        self.confidence = confidence

    def classify(
        self, frame: np.ndarray, boxes: Sequence[MotionBox]
    ) -> List[Optional[Classification]]:
        height, width = frame.shape[:2]
        results: List[Optional[Classification]] = []

        for box in boxes:
            pad_x, pad_y = int(box.w * 0.25), int(box.h * 0.25)
            x1 = max(0, box.x - pad_x)
            y1 = max(0, box.y - pad_y)
            x2 = min(width, box.x2 + pad_x)
            y2 = min(height, box.y2 + pad_y)
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0 or crop.shape[0] < 16:
                results.append(None)
                continue

            scale = 128.0 / crop.shape[0]
            target = (max(64, int(round(crop.shape[1] * scale))), 128)
            try:
                resized = cv2.resize(crop, target)
                found, weights = self._hog.detectMultiScale(
                    resized, winStride=(8, 8), padding=(8, 8), scale=1.05
                )
            except cv2.error:  # pragma: no cover - defensive
                results.append(None)
                continue

            if len(found) == 0:
                results.append(None)
                continue
            score = float(np.max(weights)) if len(weights) else 1.0
            results.append(Classification("person", min(1.0, max(0.0, score))))

        return results


class YoloClassifier(BaseClassifier):
    """YOLOv8 detector restricted to a set of interesting classes.

    The model runs **once per frame** over the whole image and the detections
    are matched back to the motion boxes by overlap.  Running it per crop would
    make a busy frame many times more expensive and would lose the context YOLO
    needs to recognise an object.
    """

    name = "yolo"

    def __init__(
        self,
        model_path: Path = DEFAULT_MODEL,
        allowed_classes: Sequence[str] = DEFAULT_CLASSES,
        confidence: float = 0.35,
        image_size: int = 960,
        match_overlap: float = 0.15,
        device: str = "cpu",
    ) -> None:
        self.model_path = Path(model_path)
        self.allowed_classes = tuple(c.strip().lower() for c in allowed_classes if c.strip())
        self.confidence = confidence
        self.image_size = image_size
        self.match_overlap = match_overlap
        self.device = device
        self._model = None
        self._class_ids: Optional[List[int]] = None
        self.available = is_yolo_available()

    # -- model -------------------------------------------------------------
    def _load(self):
        """Load the weights on first use, not at import time."""
        if self._model is not None:
            return self._model

        os.environ.setdefault("YOLO_VERBOSE", "False")
        from ultralytics import YOLO  # imported lazily: heavy and optional

        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        LOGGER.info("Loading YOLO weights from %s", self.model_path)
        model = YOLO(str(self.model_path))

        names: Dict[int, str] = model.names
        wanted = set(self.allowed_classes)
        self._class_ids = [i for i, name in names.items() if name.lower() in wanted]
        unknown = wanted - {name.lower() for name in names.values()}
        if unknown:
            LOGGER.warning("Ignoring unknown class name(s): %s", ", ".join(sorted(unknown)))
        if not self._class_ids:
            LOGGER.warning("No known classes selected; the classifier will reject everything")

        self._model = model
        return model

    def warm_up(self) -> None:
        """Load the model and run one tiny inference, so the first real frame
        is not delayed by several seconds."""
        model = self._load()
        blank = np.zeros((64, 64, 3), dtype=np.uint8)
        model.predict(blank, verbose=False, device=self.device, imgsz=64)

    # -- classification ----------------------------------------------------
    def classify(
        self, frame: np.ndarray, boxes: Sequence[MotionBox]
    ) -> List[Optional[Classification]]:
        if not boxes:
            return []

        self._load()
        if not self._class_ids:
            return [None] * len(boxes)

        detections = self._run(frame)
        return [self._best_match(box, detections) for box in boxes]

    def detect_objects(self, frame: np.ndarray) -> List[ObjectDetection]:
        """Return one box per object the model recognises in the frame.

        This is the answer to "one box per person": background subtraction
        reports the parts of a body that happen to differ from the background,
        while a detector reports bodies.  The caller decides which of them are
        interesting - typically the ones that are also moving.
        """
        self._load()
        if not self._class_ids:
            return []

        objects: List[ObjectDetection] = []
        height, width = frame.shape[:2]
        for (x1, y1, x2, y2), label, score in self._run(frame):
            box = MotionBox(x1, y1, max(1, x2 - x1), max(1, y2 - y1),
                            float((x2 - x1) * (y2 - y1))).clipped(width, height)
            objects.append(ObjectDetection(box=box, label=label, confidence=score))
        return objects

    def _run(self, frame: np.ndarray) -> List[Tuple[Tuple[int, int, int, int], str, float]]:
        """One inference pass over the whole frame."""
        model = self._load()
        predictions = model.predict(
            frame,
            verbose=False,
            device=self.device,
            imgsz=self.image_size,
            conf=self.confidence,
            classes=self._class_ids,
        )

        detections: List[Tuple[Tuple[int, int, int, int], str, float]] = []
        names = model.names
        for prediction in predictions:
            for detection in prediction.boxes:
                x1, y1, x2, y2 = (int(v) for v in detection.xyxy[0].tolist())
                label = str(names[int(detection.cls[0])])
                score = float(detection.conf[0])
                detections.append(((x1, y1, x2, y2), label, score))
        return detections

    def _best_match(
        self,
        box: MotionBox,
        detections: Sequence[Tuple[Tuple[int, int, int, int], str, float]],
    ) -> Optional[Classification]:
        """Pick the detection that best covers this motion box.

        Coverage is measured against the *smaller* of the two rectangles: a
        motion box often captures only part of a person (a moving arm), and a
        YOLO box often covers a person who is only partly moving.
        """
        best: Optional[Classification] = None
        best_score = 0.0

        for rect, label, score in detections:
            overlap = intersection_area(box.rect, rect)
            if overlap <= 0:
                continue
            smaller = min(rect_area(box.rect), rect_area(rect)) or 1.0
            coverage = overlap / smaller
            if coverage < self.match_overlap:
                continue
            if coverage * score > best_score:
                best_score = coverage * score
                best = Classification(label, score)

        return best

    def describe(self) -> str:
        return f"yolo ({', '.join(self.allowed_classes)})"


def is_yolo_available() -> bool:
    """True when the optional ultralytics dependency can be imported."""
    try:
        import importlib.util

        return importlib.util.find_spec("ultralytics") is not None
    except Exception:  # pragma: no cover - defensive
        return False


def build_classifier(
    kind: str = "none",
    allowed_classes: Sequence[str] = DEFAULT_CLASSES,
    confidence: float = 0.35,
    model_path: Optional[Path] = None,
    image_size: int = 960,
) -> Optional[BaseClassifier]:
    """Create a classifier by name, or None when classification is disabled.

    Falls back to HOG (with a warning) when YOLO is requested but the optional
    dependency is missing, so a configuration mistake degrades instead of
    breaking the run.
    """
    kind = (kind or "none").strip().lower()

    if kind in {"", "none", "off", "false"}:
        return None
    if kind == "hog":
        return HogPersonClassifier()
    if kind in {"yolo", "yolov8", "yolov8n"}:
        if not is_yolo_available():
            LOGGER.warning(
                "YOLO was requested but 'ultralytics' is not installed; "
                "falling back to the HOG people detector. "
                "Install it with: pip install ultralytics"
            )
            return HogPersonClassifier()
        return YoloClassifier(
            model_path=model_path or DEFAULT_MODEL,
            allowed_classes=allowed_classes,
            confidence=confidence,
            image_size=image_size,
        )

    LOGGER.warning("Unknown classifier %r; classification is disabled", kind)
    return None
