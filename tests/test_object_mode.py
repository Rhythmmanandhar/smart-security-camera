"""One box per object, and movement in the distance.

These cover the two complaints that motivated object-detection mode: a single
person coming back as several boxes with several ids, and only the nearest
movement being reported.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from src.classifier import ObjectDetection, is_yolo_available
from src.config import build_config
from src.detector import MotionDetector
from src.geometry import MotionBox, merge_boxes, suppress_contained
from src.pipeline import SecurityPipeline

needs_yolo = pytest.mark.skipif(
    not is_yolo_available(), reason="ultralytics is not installed"
)


def box(x, y, w, h):
    return MotionBox(x, y, w, h, float(w * h) * 0.5)


# --- merging: one body, one box --------------------------------------------


def test_stacked_fragments_of_one_body_merge():
    """Head, torso and legs arrive as three blobs at the same x."""
    head = box(100, 40, 18, 16)
    torso = box(96, 60, 26, 34)
    legs = box(100, 98, 20, 30)
    merged = merge_boxes([head, torso, legs])
    assert len(merged) == 1
    assert merged[0].y <= 40 and merged[0].y2 >= 128


def test_two_people_side_by_side_stay_apart():
    left = box(100, 60, 24, 70)
    right = box(150, 60, 24, 70)
    assert len(merge_boxes([left, right])) == 2


def test_merging_does_not_chain_sideways_into_a_crowd():
    """Three people in a row must not collapse into one wide box."""
    people = [box(60, 60, 26, 70), box(100, 62, 26, 70), box(140, 58, 26, 70)]
    merged = merge_boxes(people)
    assert len(merged) >= 2
    widest = max(merged, key=lambda b: b.w)
    assert widest.w / widest.h < 1.6


def test_nested_duplicate_is_dropped():
    outer = box(100, 50, 60, 90)
    inner = box(110, 60, 30, 40)
    assert suppress_contained([outer, inner]) == [outer]


# --- reach: distant objects -------------------------------------------------


def _scene(width=640, height=360, level=90):
    return np.full((height, width, 3), level, dtype=np.uint8)


def _warm(detector, frames=None):
    background = _scene()
    for _ in range(frames or detector.warmup_frames + 3):
        detector.detect(background)


def test_a_small_distant_object_is_detected():
    """A person by the far wall is a handful of pixels; it must still count."""
    detector = MotionDetector(build_config().detection, (640, 360), fps=15.0)
    _warm(detector)

    frame = _scene()
    # ~14x30 px near the top of the frame: the far end of a warehouse.
    cv2.rectangle(frame, (300, 70), (314, 100), (20, 20, 200), -1)
    boxes = detector.detect(frame).boxes
    assert boxes, "distant movement was filtered out"


def test_the_same_object_close_up_is_also_detected():
    detector = MotionDetector(build_config().detection, (640, 360), fps=15.0)
    _warm(detector)

    frame = _scene()
    cv2.rectangle(frame, (300, 250), (340, 330), (20, 20, 200), -1)
    assert detector.detect(frame).boxes


def test_perspective_scaling_is_gentler_near_the_top():
    config = build_config().detection
    detector = MotionDetector(config, (640, 360), fps=15.0)
    height = detector.analysis_size[1]
    near = detector._min_area_for(height * 0.9, height)
    far = detector._min_area_for(height * 0.1, height)
    assert far < near
    assert far / near < 0.4, "distant objects still have to be almost as large"


# --- object mode ------------------------------------------------------------


class StubDetector:
    """Stands in for YOLO: reports fixed objects, no model needed."""

    name = "stub"

    def __init__(self, objects):
        self.objects = objects
        self.calls = 0

    def detect_objects(self, frame):
        self.calls += 1
        return self.objects

    def classify(self, frame, boxes):
        return [None] * len(boxes)

    def describe(self):
        return "stub"


def _object_pipeline(objects, fps=15.0, size=(320, 240)):
    config = build_config(
        "balanced",
        classifier={"kind": "yolo", "mode": "detect", "interval_seconds": 0.0},
        record_incidents=False,
    )
    pipeline = SecurityPipeline(config, fps=fps, frame_size=size)
    pipeline.classifier = StubDetector(objects)
    pipeline.object_mode = True
    return pipeline


def test_detect_mode_is_off_by_default():
    assert build_config().classifier.detects is False


def test_detect_mode_needs_yolo():
    assert build_config(classifier={"kind": "hog", "mode": "detect"}).classifier.detects is False


def test_one_track_per_detected_object():
    """Two people standing close together stay two objects, not one blob."""
    objects = [
        ObjectDetection(box(60, 80, 30, 80), "person", 0.9),
        ObjectDetection(box(110, 82, 30, 80), "person", 0.8),
    ]
    pipeline = _object_pipeline(objects)
    frame = np.full((240, 320, 3), 80, dtype=np.uint8)

    for _ in range(6):
        result = pipeline.process_frame(frame)

    assert len(result.tracks) == 2
    assert {t.label for t in result.tracks} == {"person"}


def test_objects_are_labelled_with_their_class():
    objects = [
        ObjectDetection(box(40, 60, 40, 90), "person", 0.9),
        ObjectDetection(box(200, 120, 90, 60), "car", 0.7),
    ]
    pipeline = _object_pipeline(objects)
    frame = np.full((240, 320, 3), 80, dtype=np.uint8)
    for _ in range(6):
        result = pipeline.process_frame(frame)

    assert sorted(t.label for t in result.tracks) == ["car", "person"]


def test_a_new_object_alarms_even_before_it_moves():
    """Someone who steps into view and stands still is still an intruder."""
    pipeline = _object_pipeline([ObjectDetection(box(60, 80, 30, 80), "person", 0.9)])
    frame = np.full((240, 320, 3), 80, dtype=np.uint8)

    alerted = False
    for _ in range(6):
        alerted = alerted or pipeline.process_frame(frame).alert
    assert alerted


def test_a_long_standing_object_stops_alarming():
    """A parked car must not hold the alarm open for ever."""
    pipeline = _object_pipeline([ObjectDetection(box(200, 120, 90, 60), "car", 0.8)])
    frame = np.full((240, 320, 3), 80, dtype=np.uint8)

    for _ in range(pipeline.new_object_frames + 20):
        result = pipeline.process_frame(frame)
    assert result.alert is False
    assert result.tracks, "the object should still be tracked, just not alarming"


def test_motion_the_detector_missed_is_still_reported():
    """An unnamed moving blob must not disappear just because YOLO ran."""
    pipeline = _object_pipeline([])  # the model recognises nothing
    background = np.full((240, 320, 3), 80, dtype=np.uint8)

    for _ in range(pipeline.detector.warmup_frames + 3):
        pipeline.process_frame(background)

    seen = 0
    for step in range(12):
        frame = background.copy()
        x = 40 + step * 8
        cv2.rectangle(frame, (x, 90), (x + 28, 170), (20, 20, 200), -1)
        result = pipeline.process_frame(frame)
        seen = max(seen, len(result.tracks))

    assert seen >= 1, "motion outside the detector's vocabulary was lost"


def test_the_detector_runs_on_an_interval_not_every_frame():
    config = build_config(
        "balanced",
        classifier={"kind": "yolo", "mode": "detect", "interval_seconds": 0.5},
        record_incidents=False,
    )
    pipeline = SecurityPipeline(config, fps=20.0, frame_size=(320, 240))
    stub = StubDetector([ObjectDetection(box(60, 80, 30, 80), "person", 0.9)])
    pipeline.classifier = stub
    pipeline.object_mode = True

    frame = np.full((240, 320, 3), 80, dtype=np.uint8)
    for _ in range(20):
        pipeline.process_frame(frame)

    assert stub.calls <= 4, f"detector ran {stub.calls} times in 20 frames"


def test_boxes_are_extrapolated_between_detector_runs():
    """The drawn box keeps moving while the model is not looking."""
    config = build_config(
        "balanced",
        classifier={"kind": "yolo", "mode": "detect", "interval_seconds": 0.5},
        record_incidents=False,
    )
    pipeline = SecurityPipeline(config, fps=20.0, frame_size=(320, 240))
    frame = np.full((240, 320, 3), 80, dtype=np.uint8)

    positions = [40, 60, 80]
    for index, x in enumerate(positions):
        pipeline.classifier = StubDetector(
            [ObjectDetection(box(x, 80, 30, 80), "person", 0.9)]
        )
        pipeline.object_mode = True
        pipeline._objects_frame = -1  # force a run
        pipeline.process_frame(frame)

    # A frame with no detector run should slide the drawn box along.
    result = pipeline.process_frame(frame)
    track = result.tracks[0]
    assert track.display_box is not None
    assert track.display_box.x >= track.box.x


# --- the real model ---------------------------------------------------------


@needs_yolo
def test_yolo_returns_labelled_object_boxes():
    from src.classifier import YoloClassifier

    capture = cv2.VideoCapture("cctv_video_1.mp4")
    capture.set(cv2.CAP_PROP_POS_FRAMES, 40)
    ok, frame = capture.read()
    capture.release()
    if not ok:
        pytest.skip("sample footage is unavailable")

    objects = YoloClassifier(allowed_classes=("person",), confidence=0.25).detect_objects(frame)
    assert objects, "no people found in a frame that clearly contains several"
    assert all(obj.label == "person" for obj in objects)
    assert all(obj.box.w > 0 and obj.box.h > 0 for obj in objects)
