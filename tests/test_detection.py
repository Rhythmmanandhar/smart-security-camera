"""Detection behaviour, expressed as the promises the system makes.

Each test names a defect that was found in v1, so a regression fails loudly
rather than quietly degrading into false alarms.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from src.config import build_config
from src.detector import MotionDetector
from src.pipeline import SecurityPipeline
from src.security_camera import SmartSecurityCamera


def run(path, output_dir=None, sensitivity="balanced", **overrides):
    config = build_config(sensitivity, record_incidents=output_dir is not None, **overrides)
    camera = SmartSecurityCamera(path, output_dir, config)
    return camera.process()


def frame_alerts(path, sensitivity="balanced"):
    """Per-frame alert flags for a clip."""
    capture = cv2.VideoCapture(str(path))
    fps = capture.get(cv2.CAP_PROP_FPS) or 15.0
    ok, frame = capture.read()
    assert ok
    height, width = frame.shape[:2]
    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)

    pipeline = SecurityPipeline(
        build_config(sensitivity, record_incidents=False), fps=fps, frame_size=(width, height)
    )
    alerts = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        alerts.append(pipeline.process_frame(frame).alert)
    capture.release()
    pipeline.close()
    return alerts


# --- the core promise -------------------------------------------------------


def test_intruder_is_detected(clips, output_dir):
    incidents = run(clips["intruder"], output_dir)
    assert len(incidents) == 1

    incident = incidents[0]
    # The intruder appears at frame 60 of a 15 fps clip, i.e. 4.0s.
    assert 3.0 <= incident.start_time_s <= 5.5
    assert incident.duration_s >= 1.0
    assert incident.max_objects >= 1


def test_empty_room_raises_no_incident(clips, output_dir):
    assert run(clips["static"], output_dir) == []


def test_nothing_is_flagged_before_the_intruder_arrives(clips):
    """v1's first frame was 100% foreground and alarmed immediately."""
    alerts = frame_alerts(clips["intruder"])
    assert not any(alerts[:40]), "motion was reported before anything moved"


def test_light_switch_is_not_an_incident(clips, output_dir):
    """A global illumination change must trigger a re-learn, not an alarm."""
    assert run(clips["light_switch"], output_dir) == []


def test_burnt_in_clock_is_not_an_incident(clips, output_dir):
    assert run(clips["osd_clock"], output_dir) == []


def test_object_moving_on_the_spot_is_not_an_incident(clips, output_dir):
    """Foliage oscillates but never travels, so net displacement stays ~0."""
    assert run(clips["swaying_plant"], output_dir) == []


# --- detector internals -----------------------------------------------------


def test_detector_reports_warming_up_then_stops():
    detector = MotionDetector(build_config().detection, (320, 240), fps=15.0)
    frame = np.full((240, 320, 3), 100, dtype=np.uint8)

    first = detector.detect(frame)
    assert first.warming_up
    assert first.boxes == []

    for _ in range(detector.warmup_frames):
        detector.detect(frame)
    assert not detector.detect(frame).warming_up


def test_detector_flags_a_whole_scene_change_instead_of_alarming():
    detector = MotionDetector(build_config().detection, (320, 240), fps=15.0)
    dark = np.full((240, 320, 3), 40, dtype=np.uint8)
    for _ in range(detector.warmup_frames + 5):
        detector.detect(dark)

    bright = np.full((240, 320, 3), 220, dtype=np.uint8)
    result = detector.detect(bright)
    assert result.scene_change
    assert result.boxes == []


def test_thresholds_are_resolution_independent():
    """The same object must be found at 480p and at 1080p."""
    found = []
    for width, height in ((640, 360), (1920, 1080)):
        detector = MotionDetector(build_config().detection, (width, height), fps=15.0)
        background = np.full((height, width, 3), 90, dtype=np.uint8)
        for _ in range(detector.warmup_frames + 3):
            detector.detect(background)

        moved = background.copy()
        # An object covering ~2% of the frame width, wherever the frame size.
        box_w, box_h = int(width * 0.06), int(height * 0.25)
        x, y = int(width * 0.4), int(height * 0.5)
        cv2.rectangle(moved, (x, y), (x + box_w, y + box_h), (20, 20, 200), -1)
        found.append(len(detector.detect(moved).boxes))

    assert found[0] >= 1 and found[1] >= 1, f"detections differ by resolution: {found}"


def test_boxes_hug_the_object():
    """v1 dilated the mask and never eroded it, inflating every box."""
    width, height = 640, 360
    detector = MotionDetector(build_config().detection, (width, height), fps=15.0)
    background = np.full((height, width, 3), 90, dtype=np.uint8)
    for _ in range(detector.warmup_frames + 3):
        detector.detect(background)

    moved = background.copy()
    truth = (300, 150, 350, 260)
    cv2.rectangle(moved, truth[:2], truth[2:], (20, 20, 200), -1)
    boxes = detector.detect(moved).boxes
    assert boxes

    truth_area = (truth[2] - truth[0]) * (truth[3] - truth[1])
    best = max(boxes, key=lambda b: b.w * b.h)
    assert best.w * best.h <= truth_area * 2.2, "detected box is far larger than the object"


@pytest.mark.parametrize("sensitivity", ["low", "balanced", "high"])
def test_every_sensitivity_level_runs(clips, output_dir, sensitivity):
    incidents = run(clips["intruder"], output_dir / sensitivity, sensitivity=sensitivity)
    assert isinstance(incidents, list)
