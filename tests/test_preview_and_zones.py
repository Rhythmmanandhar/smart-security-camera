"""Preview-frame endpoints and zone handling for the drawing UI."""

from __future__ import annotations

import io

import pytest

import app as web_app

def test_preview_returns_a_jpeg_frame(client, clips):
    with open(clips["intruder"], "rb") as handle:
        data = {"video_file": (io.BytesIO(handle.read()), "intruder.mp4")}
        response = client.post("/api/preview", data=data, content_type="multipart/form-data")

    assert response.status_code == 200
    assert response.mimetype == "image/jpeg"
    assert response.data[:2] == b"\xff\xd8", "not a JPEG"


def test_preview_cleans_up_its_temporary_upload(client, clips):
    with open(clips["static"], "rb") as handle:
        data = {"video_file": (io.BytesIO(handle.read()), "static.mp4")}
        client.post("/api/preview", data=data, content_type="multipart/form-data")

    leftovers = [path for path in web_app.UPLOAD_ROOT.rglob("*") if path.is_file()]
    assert leftovers == []


def test_preview_rejects_a_file_that_is_not_a_video(client):
    data = {"video_file": (io.BytesIO(b"nonsense"), "clip.mp4")}
    response = client.post("/api/preview", data=data, content_type="multipart/form-data")
    assert response.status_code == 400
    assert response.get_json()["error"]


def test_preview_without_a_source_is_an_error(client):
    response = client.post("/api/preview", data={})
    assert response.status_code == 400


def test_live_preview_reports_a_missing_camera(client, monkeypatch):
    from src import live as live_module

    def refuse(**_kwargs):
        raise live_module.CameraError("device 7 could not be opened")

    monkeypatch.setattr(web_app, "open_camera", refuse)
    response = client.get("/api/live/preview?device=7")
    assert response.status_code == 503


# --- zone payload validation ------------------------------------------------


def test_live_start_accepts_drawn_zones(client, monkeypatch):
    """Zones drawn in the browser arrive as JSON lists, not strings."""
    captured = {}

    def fake_start(job_id, config, record=True):
        captured["roi"] = config.detection.roi
        captured["ignore"] = config.detection.ignore_zones
        captured["classifier"] = config.classifier.kind
        return {"job_id": job_id, "device": 0, "backend": "test",
                "resolution": "640x480", "fps": 10.0, "recording": record,
                "sensitivity": config.sensitivity}

    monkeypatch.setattr(web_app.live, "start", fake_start)
    response = client.post(
        "/api/live/start",
        json={
            "device": 0,
            "roi": [0.1, 0.2, 0.8, 0.9],
            "ignore_zones": [[0.0, 0.0, 0.2, 0.1]],
            "classifier": "hog",
        },
    )

    assert response.status_code == 200
    assert captured["roi"] == (0.1, 0.2, 0.8, 0.9)
    assert captured["ignore"] == ((0.0, 0.0, 0.2, 0.1),)
    assert captured["classifier"] == "hog"


@pytest.mark.parametrize(
    "roi",
    [[0.1, 0.2], [0.1, 0.2, 0.3, 4.0], [0.9, 0.1, 0.2, 0.8]],
)
def test_live_start_rejects_malformed_zones(client, roi):
    response = client.post("/api/live/start", json={"device": 0, "roi": roi})
    assert response.status_code == 400


def test_upload_form_accepts_typed_zones(client, clips):
    """The manual-entry fields still work alongside the drawing UI."""
    with open(clips["intruder"], "rb") as handle:
        data = {
            "video_file": (io.BytesIO(handle.read()), "intruder.mp4"),
            "roi": "0.0 0.3 1.0 1.0",
            "ignore_zones": "0.0 0.0 0.2 0.1\n0.8 0.0 1.0 0.1",
        }
        response = client.post("/api/jobs", data=data, content_type="multipart/form-data")

    assert response.status_code == 202
    job = web_app.jobs.get(response.get_json()["job_id"])
    assert job.config.detection.roi == (0.0, 0.3, 1.0, 1.0)
    assert len(job.config.detection.ignore_zones) == 2


def test_health_reports_whether_yolo_is_available(client):
    payload = client.get("/api/health").get_json()
    assert "yolo_available" in payload
    assert isinstance(payload["yolo_available"], bool)
