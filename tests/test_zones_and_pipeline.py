"""Monitored zones, ignore zones and incident-state-machine behaviour."""

from __future__ import annotations

import cv2
import numpy as np

from src.config import build_config
from src.pipeline import SecurityPipeline
from src.security_camera import SmartSecurityCamera


def analyse(path, sensitivity="balanced", **detection):
    """Run detection only and return the pipeline (for its counters)."""
    config = build_config(sensitivity, detection=detection or None, record_incidents=False)
    camera = SmartSecurityCamera(path, None, config)
    camera.process()
    return camera.pipeline


def test_monitored_zone_excludes_motion_outside_it(clips):
    """The intruder walks along the lower half; a top-strip ROI must ignore it."""
    everywhere = analyse(clips["intruder"])
    top_strip = analyse(clips["intruder"], roi=(0.0, 0.0, 1.0, 0.25))

    assert everywhere.alert_frames > 0
    assert top_strip.alert_frames == 0


def test_monitored_zone_keeps_motion_inside_it(clips):
    covering = analyse(clips["intruder"], roi=(0.0, 0.3, 1.0, 1.0))
    assert covering.alert_frames > 0


def test_ignore_zone_suppresses_its_region(clips):
    """An ignore zone over the whole frame silences everything."""
    muted = analyse(clips["intruder"], ignore_zones=((0.0, 0.0, 1.0, 1.0),))
    assert muted.alert_frames == 0


def test_alert_frames_are_counted_even_without_recording(clips):
    pipeline = analyse(clips["intruder"])
    assert pipeline.alert_frames > 10
    assert pipeline.incidents == []


# --- incident state machine -------------------------------------------------


def _moving_clip(frames=200, fps=15.0, size=(320, 240)):
    """Frames of a rectangle crossing a static background, as a generator."""
    background = np.full((size[1], size[0], 3), 80, dtype=np.uint8)
    rng = np.random.default_rng(3)
    for index in range(frames):
        frame = np.clip(
            background.astype(np.int16) + rng.normal(0, 2, background.shape), 0, 255
        ).astype(np.uint8)
        if index >= 40:
            x = 20 + ((index - 40) * 3) % (size[0] - 60)
            cv2.rectangle(frame, (x, 90), (x + 34, 170), (30, 30, 200), -1)
        yield frame


def test_incident_is_split_when_it_runs_too_long(tmp_path):
    config = build_config(
        "balanced", incident={"max_incident_seconds": 2.0, "min_incident_seconds": 0.5}
    )
    pipeline = SecurityPipeline(
        config, fps=15.0, frame_size=(320, 240), output_dir=tmp_path
    )
    for frame in _moving_clip(frames=200):
        pipeline.process_frame(frame)
    incidents = pipeline.close()

    assert len(incidents) >= 2, "a long event should be split into several clips"
    assert all(incident.duration_s <= 2.6 for incident in incidents)


def test_incident_numbering_has_no_gaps(tmp_path):
    config = build_config(
        "balanced", incident={"max_incident_seconds": 2.0, "min_incident_seconds": 0.5}
    )
    pipeline = SecurityPipeline(
        config, fps=15.0, frame_size=(320, 240), output_dir=tmp_path
    )
    for frame in _moving_clip(frames=200):
        pipeline.process_frame(frame)
    incidents = pipeline.close()

    assert [i.index for i in incidents] == list(range(1, len(incidents) + 1))
    for incident in incidents:
        assert (tmp_path / incident.filename).exists()


def test_the_trigger_frame_is_written_once(tmp_path):
    """v1 wrote the triggering frame into the clip twice (pre-roll + current)."""
    config = build_config("balanced", incident={"pre_event_seconds": 1.0})
    pipeline = SecurityPipeline(
        config, fps=15.0, frame_size=(320, 240), output_dir=tmp_path
    )
    written = 0
    for frame in _moving_clip(frames=150):
        result = pipeline.process_frame(frame)
        if result.recording:
            written += 1
    incidents = pipeline.close()
    assert incidents

    clip_frames = incidents[0].frame_count
    pre_roll = pipeline.pre_event_frames
    # Recorded frames = the frames flagged while recording, plus the pre-roll.
    assert clip_frames <= written + pre_roll


def test_pipeline_reports_progress_of_a_cancelled_run(clips, tmp_path):
    import threading

    stop = threading.Event()
    camera = SmartSecurityCamera(clips["intruder"], tmp_path, build_config())

    def cancel_after_a_few(progress):
        if progress.frames_done >= 20:
            stop.set()

    incidents = camera.process(progress=cancel_after_a_few, stop_event=stop)
    assert camera.pipeline.frames_processed < 180
    assert isinstance(incidents, list)
