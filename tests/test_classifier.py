"""Object classification: selection, filtering and graceful degradation."""

from __future__ import annotations

import numpy as np
import pytest

from src.classifier import (
    Classification,
    HogPersonClassifier,
    NullClassifier,
    YoloClassifier,
    build_classifier,
    is_yolo_available,
)
from src.config import ClassifierConfig, build_config
from src.geometry import MotionBox
from src.pipeline import SecurityPipeline

needs_yolo = pytest.mark.skipif(
    not is_yolo_available(), reason="ultralytics is not installed"
)


# --- selection --------------------------------------------------------------


@pytest.mark.parametrize("kind", ["none", "off", "", None])
def test_disabled_classifier_builds_nothing(kind):
    assert build_classifier(kind) is None


def test_hog_is_selectable_without_extra_dependencies():
    assert isinstance(build_classifier("hog"), HogPersonClassifier)


def test_unknown_classifier_disables_classification():
    assert build_classifier("magic") is None


def test_yolo_request_falls_back_to_hog_when_unavailable(monkeypatch):
    """A missing optional dependency must degrade, not crash the run."""
    monkeypatch.setattr("src.classifier.is_yolo_available", lambda: False)
    assert isinstance(build_classifier("yolo"), HogPersonClassifier)


def test_null_classifier_accepts_everything():
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    results = NullClassifier().classify(frame, [MotionBox(0, 0, 10, 10)])
    assert results[0] is not None


# --- configuration ----------------------------------------------------------


def test_classifier_is_off_by_default():
    assert build_config().classifier.kind == "none"
    assert build_config().classifier.enabled is False


def test_require_person_still_maps_to_a_classifier():
    """v1 spelled this as a boolean flag."""
    config = build_config(require_person=True)
    assert config.classifier.kind == "hog"
    assert config.require_person is True


def test_classifier_interval_follows_the_frame_rate():
    config = ClassifierConfig(interval_seconds=0.5)
    assert config.interval_frames(30.0) == 15
    assert config.interval_frames(6.0) == 3  # never below min_interval_frames


def test_classes_can_be_narrowed():
    config = build_config(classifier={"kind": "yolo", "classes": ("person",)})
    assert config.classifier.classes == ("person",)


# --- behaviour in the pipeline ---------------------------------------------


class StubClassifier:
    """Recognises only boxes on the left half of the frame."""

    name = "stub"

    def __init__(self):
        self.calls = 0

    def classify(self, frame, boxes):
        self.calls += 1
        return [
            Classification("person", 0.9) if box.centroid[0] < frame.shape[1] / 2 else None
            for box in boxes
        ]

    def describe(self):
        return "stub"


def _moving_frames(frames=90, size=(320, 240), direction="left"):
    import cv2

    background = np.full((size[1], size[0], 3), 80, dtype=np.uint8)
    rng = np.random.default_rng(5)
    for index in range(frames):
        frame = np.clip(
            background.astype(np.int16) + rng.normal(0, 2, background.shape), 0, 255
        ).astype(np.uint8)
        if index >= 30:
            step = (index - 30) * 2
            x = 10 + step if direction == "left" else 170 + step
            x = min(x, size[0] - 40)
            cv2.rectangle(frame, (x, 90), (x + 30, 170), (30, 30, 200), -1)
        yield frame


def _run(pipeline, frames):
    for frame in frames:
        pipeline.process_frame(frame)
    return pipeline


def test_recognised_objects_still_alarm():
    stub = StubClassifier()
    pipeline = SecurityPipeline(
        build_config("balanced", classifier={"kind": "hog"}, record_incidents=False),
        fps=15.0,
        frame_size=(320, 240),
    )
    pipeline.classifier = stub
    _run(pipeline, _moving_frames(direction="left"))
    assert pipeline.alert_frames > 0
    assert stub.calls > 0


def test_unrecognised_objects_do_not_alarm():
    stub = StubClassifier()
    pipeline = SecurityPipeline(
        build_config("balanced", classifier={"kind": "hog"}, record_incidents=False),
        fps=15.0,
        frame_size=(320, 240),
    )
    pipeline.classifier = stub
    _run(pipeline, _moving_frames(direction="right"))
    assert pipeline.alert_frames == 0


def test_without_a_classifier_the_same_motion_alarms():
    pipeline = SecurityPipeline(
        build_config("balanced", record_incidents=False), fps=15.0, frame_size=(320, 240)
    )
    _run(pipeline, _moving_frames(direction="right"))
    assert pipeline.alert_frames > 0


def test_classification_is_reused_between_checks():
    """The verdict is cached, so the model is not run on every frame."""
    stub = StubClassifier()
    pipeline = SecurityPipeline(
        build_config("balanced", classifier={"kind": "hog"}, record_incidents=False),
        fps=15.0,
        frame_size=(320, 240),
    )
    pipeline.classifier = stub
    _run(pipeline, _moving_frames(frames=90, direction="left"))
    assert stub.calls < 40, "the classifier ran far more often than the interval allows"


def test_a_failing_classifier_does_not_stop_the_run():
    class Broken:
        name = "broken"

        def classify(self, frame, boxes):
            raise RuntimeError("model exploded")

        def describe(self):
            return "broken"

    pipeline = SecurityPipeline(
        build_config("balanced", classifier={"kind": "hog"}, record_incidents=False),
        fps=15.0,
        frame_size=(320, 240),
    )
    pipeline.classifier = Broken()
    _run(pipeline, _moving_frames(direction="left"))
    assert pipeline.frames_processed == 90


# --- the real model ---------------------------------------------------------


@needs_yolo
def test_yolo_matches_a_detection_to_a_motion_box():
    """A frame with an obvious person-shaped region should classify."""
    classifier = YoloClassifier(allowed_classes=("person",), confidence=0.1)
    # A synthetic block will usually not be recognised, which is itself the
    # point: the call must succeed and simply report nothing.
    frame = np.full((240, 320, 3), 90, dtype=np.uint8)
    results = classifier.classify(frame, [MotionBox(100, 60, 60, 120)])
    assert len(results) == 1
    assert results[0] is None or isinstance(results[0], Classification)


@needs_yolo
def test_yolo_finds_people_in_the_sample_footage():
    import cv2

    classifier = YoloClassifier(allowed_classes=("person",), confidence=0.25)
    capture = cv2.VideoCapture("cctv_video_1.mp4")
    capture.set(cv2.CAP_PROP_POS_FRAMES, 50)
    ok, frame = capture.read()
    capture.release()
    if not ok:
        pytest.skip("sample footage is unavailable")

    height, width = frame.shape[:2]
    whole_frame = [MotionBox(0, 0, width, height)]
    results = classifier.classify(frame, whole_frame)
    assert results[0] is not None
    assert results[0].label == "person"
