"""Web-layer tests: job lifecycle, catalogue, validation and path safety."""

from __future__ import annotations

import base64
import io
import time
from pathlib import Path

import pytest

import app as web_app
from src.storage import IncidentStore


def wait_for_job(client, job_id, timeout=60.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        payload = client.get(f"/api/jobs/{job_id}").get_json()
        if payload["status"] in {"done", "failed", "cancelled"}:
            return payload
        time.sleep(0.15)
    raise AssertionError("job did not finish in time")


# --- basics ----------------------------------------------------------------


def test_index_renders(client):
    response = client.get("/")
    assert response.status_code == 200
    assert b"Smart Security Camera" in response.data


def test_health(client):
    payload = client.get("/api/health").get_json()
    assert payload["ok"] is True
    assert "incidents" in payload


# --- input validation ------------------------------------------------------


def test_submitting_nothing_is_a_clear_error(client):
    response = client.post("/api/jobs", data={})
    assert response.status_code == 400
    assert "video" in response.get_json()["error"].lower()


def test_unsupported_file_type_is_rejected(client):
    data = {"video_file": (io.BytesIO(b"not a video"), "notes.txt")}
    response = client.post("/api/jobs", data=data, content_type="multipart/form-data")
    assert response.status_code == 400
    assert "supported" in response.get_json()["error"].lower()


def test_empty_upload_is_rejected(client):
    data = {"video_file": (io.BytesIO(b""), "clip.mp4")}
    response = client.post("/api/jobs", data=data, content_type="multipart/form-data")
    assert response.status_code == 400


def test_a_corrupt_video_fails_the_job_with_a_readable_message(client):
    data = {"video_file": (io.BytesIO(b"\x00\x01\x02 not really an mp4"), "broken.mp4")}
    response = client.post("/api/jobs", data=data, content_type="multipart/form-data")
    assert response.status_code == 202

    job = wait_for_job(client, response.get_json()["job_id"])
    assert job["status"] == "failed"
    assert job["error"]


@pytest.mark.parametrize(
    "roi,expected",
    [
        ("0.1 0.1", "four numbers"),
        ("0.1 0.1 0.2 5", "between 0 and 1"),
        ("0.9 0.1 0.2 0.8", "x1 < x2"),
    ],
)
def test_bad_roi_values_are_explained(client, roi, expected):
    data = {"video_file": (io.BytesIO(b"x"), "clip.mp4"), "roi": roi}
    response = client.post("/api/jobs", data=data, content_type="multipart/form-data")
    assert response.status_code == 400
    assert expected in response.get_json()["error"]


def test_bad_url_scheme_is_rejected(client):
    response = client.post("/api/jobs", data={"video_url": "file:///etc/passwd"})
    assert response.status_code == 400
    assert "http" in response.get_json()["error"]


# --- job lifecycle ---------------------------------------------------------


def test_a_video_is_analysed_and_catalogued(client, clips):
    with open(clips["intruder"], "rb") as handle:
        data = {"video_file": (io.BytesIO(handle.read()), "intruder.mp4")}
        response = client.post("/api/jobs", data=data, content_type="multipart/form-data")
    assert response.status_code == 202

    job = wait_for_job(client, response.get_json()["job_id"])
    assert job["status"] == "done"
    assert job["incidents_detected"] == 1

    incident = job["incidents"][0]
    assert client.get(incident["media_url"]).status_code == 200
    assert client.get(incident["thumbnail_url"]).status_code == 200
    assert client.get(incident["view_url"]).status_code == 200

    listing = client.get("/api/incidents").get_json()
    assert listing["stats"]["total"] == 1


def test_unknown_job_is_404(client):
    assert client.get("/api/jobs/deadbeef1234").status_code == 404


def test_cancelling_a_finished_job_is_a_conflict(client, clips):
    with open(clips["static"], "rb") as handle:
        data = {"video_file": (io.BytesIO(handle.read()), "static.mp4")}
        response = client.post("/api/jobs", data=data, content_type="multipart/form-data")
    job_id = response.get_json()["job_id"]
    wait_for_job(client, job_id)
    assert client.post(f"/api/jobs/{job_id}/cancel").status_code == 409


def test_incident_can_be_deleted(client, clips):
    with open(clips["intruder"], "rb") as handle:
        data = {"video_file": (io.BytesIO(handle.read()), "intruder.mp4")}
        response = client.post("/api/jobs", data=data, content_type="multipart/form-data")
    wait_for_job(client, response.get_json()["job_id"])

    incidents = client.get("/api/incidents").get_json()["incidents"]
    assert incidents
    incident = incidents[0]
    clip_path = Path(web_app.JOB_ROOT) / incident["job_id"] / "incidents" / incident["filename"]
    assert clip_path.exists()

    assert client.delete(f"/api/incidents/{incident['id']}").status_code == 200
    assert not clip_path.exists()
    assert client.delete(f"/api/incidents/{incident['id']}").status_code == 404


# --- path safety -----------------------------------------------------------


@pytest.mark.parametrize(
    "job_id,filename",
    [
        ("../../../..", "incident_001.mp4"),
        ("abcdef123456", "../../../../../../Windows/win.ini"),
        ("abcdef123456", "..%2f..%2fsecrets.txt"),
        ("abcdef123456", "incidents.db"),
        ("not-a-job-id", "incident_001.mp4"),
    ],
)
def test_media_route_refuses_paths_outside_the_store(client, job_id, filename):
    response = client.get(f"/media/{job_id}/{filename}")
    assert response.status_code in {404, 308}


def test_deleting_by_a_crafted_id_is_impossible(client):
    """v1 base64-decoded the id straight into a path, so this deleted files."""
    crafted = base64.urlsafe_b64encode(b"../../../../boot.ini").decode().rstrip("=")
    assert client.delete(f"/api/incidents/{crafted}").status_code in {404, 405}


def test_catalogue_forgets_entries_whose_clip_is_gone(client, clips):
    """A clip removed outside the app must not leave a broken listing."""
    import io

    with open(clips["intruder"], "rb") as handle:
        data = {"video_file": (io.BytesIO(handle.read()), "intruder.mp4")}
        response = client.post("/api/jobs", data=data, content_type="multipart/form-data")
    wait_for_job(client, response.get_json()["job_id"])

    incident = client.get("/api/incidents").get_json()["incidents"][0]
    clip = Path(web_app.JOB_ROOT) / incident["job_id"] / "incidents" / incident["filename"]
    clip.unlink()  # deleted behind the application's back

    assert web_app.store.prune_missing() == 1
    assert client.get("/api/incidents").get_json()["incidents"] == []


def test_pruning_keeps_entries_whose_clip_exists(client, clips):
    import io

    with open(clips["intruder"], "rb") as handle:
        data = {"video_file": (io.BytesIO(handle.read()), "intruder.mp4")}
        response = client.post("/api/jobs", data=data, content_type="multipart/form-data")
    wait_for_job(client, response.get_json()["job_id"])

    assert web_app.store.prune_missing() == 0
    assert len(client.get("/api/incidents").get_json()["incidents"]) == 1


def test_store_rejects_traversal_directly(tmp_path):
    store = IncidentStore(tmp_path / "db.sqlite", tmp_path / "jobs")
    assert store.media_path("../..", "incident_001.mp4") is None
    assert store.media_path("abcdef123456", "../../secrets.txt") is None
    assert store.media_path("abcdef123456", "passwd") is None


# --- live camera (no hardware required) ------------------------------------


def test_live_status_when_idle(client):
    payload = client.get("/api/live/status").get_json()
    assert payload["stats"]["active"] is False


def test_live_feed_without_a_session_is_a_conflict(client):
    assert client.get("/api/live/feed").status_code == 409


def test_live_snapshot_without_a_session_is_a_conflict(client):
    assert client.get("/api/live/snapshot").status_code == 409


def test_starting_a_missing_camera_reports_a_helpful_error(client, monkeypatch):
    from src import live as live_module

    def refuse(**_kwargs):
        raise live_module.CameraError("Could not start camera 9: device 9 could not be opened")

    monkeypatch.setattr(live_module, "open_camera", refuse)
    response = client.post("/api/live/start", json={"device": 9})
    assert response.status_code == 503
    assert "camera" in response.get_json()["error"].lower()


def test_legacy_live_clips_endpoint_still_answers(client):
    payload = client.get("/live/clips").get_json()
    assert payload["ok"] is True
    assert payload["clips"] == []
