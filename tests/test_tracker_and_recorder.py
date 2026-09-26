"""Unit tests for the tracker, the recorder and the configuration."""

from __future__ import annotations

import json

import cv2
import numpy as np
import pytest

from src.config import IncidentConfig, TrackingConfig, build_config, normalise_sensitivity
from src.geometry import MotionBox, merge_boxes
from src.recorder import IncidentRecorder, probe_codec
from src.tracker import CentroidTracker

FRAME = (640, 360)


def box(x, y, w=40, h=80):
    return MotionBox(x, y, w, h, float(w * h) * 0.6)


# --- tracker ---------------------------------------------------------------


def test_track_keeps_its_id_while_it_moves():
    tracker = CentroidTracker(TrackingConfig(), FRAME)
    ids = []
    for step in range(10):
        tracks = tracker.update([box(100 + step * 8, 150)], step + 1)
        ids.append(tracks[0].track_id)
    assert len(set(ids)) == 1, "the object was given a new id mid-track"


def test_two_objects_keep_separate_ids():
    tracker = CentroidTracker(TrackingConfig(), FRAME)
    for step in range(8):
        tracks = tracker.update([box(60 + step * 6, 150), box(400 - step * 6, 150)], step + 1)
    assert len({t.track_id for t in tracks}) == 2


def test_every_detection_is_matched_when_several_are_close():
    """v1 ran a fixed number of matching rounds, so skipped pairs cost matches
    and produced a brand-new id for an object that was already tracked."""
    tracker = CentroidTracker(TrackingConfig(), FRAME)
    detections = [box(100, 100), box(150, 100), box(200, 100), box(250, 100)]
    tracker.update(detections, 1)
    first_ids = {t.track_id for t in tracker.update(detections, 2)}

    moved = [box(b.x + 4, b.y) for b in detections]
    second_ids = {t.track_id for t in tracker.update(moved, 3)}
    assert first_ids == second_ids, "tracker invented new ids for known objects"


def test_a_jittering_blob_is_not_moving():
    """Net displacement, not per-step distance: noise wobbles but goes nowhere."""
    tracker = CentroidTracker(TrackingConfig(), FRAME)
    track = None
    for step in range(14):
        offset = 6 if step % 2 else -6
        track = tracker.update([box(300 + offset, 200)], step + 1)[0]
    assert tracker.is_confirmed(track)
    assert not tracker.is_moving(track)


def test_a_travelling_blob_is_moving():
    tracker = CentroidTracker(TrackingConfig(), FRAME)
    track = None
    for step in range(14):
        track = tracker.update([box(100 + step * 10, 200)], step + 1)[0]
    assert tracker.is_moving(track)


def test_a_one_frame_blip_is_never_confirmed():
    tracker = CentroidTracker(TrackingConfig(), FRAME)
    track = tracker.update([box(10, 10)], 1)[0]
    assert not tracker.is_confirmed(track)


def test_tracks_expire_after_being_missing():
    config = TrackingConfig(max_missing_frames=3)
    tracker = CentroidTracker(config, FRAME)
    tracker.update([box(100, 100)], 1)
    for step in range(2, 8):
        tracker.update([], step)
    assert tracker.tracks == {}


def test_match_distance_scales_with_frame_size():
    small = CentroidTracker(TrackingConfig(), (320, 180))
    large = CentroidTracker(TrackingConfig(), (1920, 1080))
    assert large.max_distance > small.max_distance * 4


# --- geometry --------------------------------------------------------------


def test_merge_joins_touching_boxes_only():
    merged = merge_boxes([box(100, 100), box(120, 110), box(500, 300)])
    assert len(merged) == 2


def test_merge_is_stable_for_a_single_box():
    only = box(10, 10)
    assert merge_boxes([only]) == [only]


# --- configuration ---------------------------------------------------------


def test_timings_follow_the_frame_rate():
    config = IncidentConfig(min_motion_seconds=1.0, post_event_seconds=2.0)
    assert config.frames(30.0)["min_motion"] == 30
    assert config.frames(6.0)["min_motion"] == 6
    assert config.frames(30.0)["post_event"] == 60


def test_explicit_frame_counts_win_over_seconds():
    config = IncidentConfig(min_motion_seconds=1.0, min_motion_frames=4)
    assert config.frames(30.0)["min_motion"] == 4


def test_legacy_suppression_names_still_resolve():
    assert normalise_sensitivity("medium") == "balanced"
    assert normalise_sensitivity(None) == "balanced"
    assert normalise_sensitivity("nonsense") == "balanced"


def test_sensitivity_presets_are_ordered():
    low = build_config("low")
    high = build_config("high")
    assert low.detection.min_area_ratio > high.detection.min_area_ratio
    assert low.tracking.min_hits > high.tracking.min_hits


# --- recorder --------------------------------------------------------------


def test_probe_returns_a_codec_this_build_can_write():
    fourcc, extension = probe_codec()
    assert extension in {".mp4", ".webm", ".avi"}
    assert len(fourcc) == 4


def test_recorder_writes_a_playable_clip_with_metadata(tmp_path):
    recorder = IncidentRecorder(tmp_path, fps=15.0, frame_size=(320, 240), source="unit test")
    path = recorder.start(1, 0.0, 1)
    assert path is not None

    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    for index in range(30):
        cv2.rectangle(frame, (index, 50), (index + 30, 150), (200, 30, 30), -1)
        recorder.write(frame, object_count=1, track_ids=[7])

    record = recorder.finish(2.0, 30)
    assert record is not None
    assert record.frame_count == 30
    assert record.track_ids == [7]

    clip = tmp_path / record.filename
    capture = cv2.VideoCapture(str(clip))
    assert capture.isOpened() and capture.read()[0]
    capture.release()

    assert (tmp_path / record.thumbnail).exists()
    meta = json.loads(clip.with_suffix(".json").read_text(encoding="utf-8"))
    assert meta["duration_s"] == pytest.approx(2.0, abs=0.01)


def test_recorder_discards_clips_below_the_minimum(tmp_path):
    recorder = IncidentRecorder(
        tmp_path, fps=15.0, frame_size=(320, 240), min_duration_s=2.0
    )
    path = recorder.start(1, 0.0, 1)
    recorder.write(np.zeros((240, 320, 3), dtype=np.uint8))
    assert recorder.finish(0.5, 8) is None
    assert not path.exists()


def test_recorder_resizes_a_mismatched_frame(tmp_path):
    recorder = IncidentRecorder(tmp_path, fps=15.0, frame_size=(320, 240))
    recorder.start(1, 0.0, 1)
    recorder.write(np.zeros((480, 640, 3), dtype=np.uint8))
    assert recorder.frames_written == 1
    recorder.abort()
