"""Shared fixtures: synthetic clips with known ground truth."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import app as web_app  # noqa: E402
from src.storage import IncidentStore  # noqa: E402
from tools.make_test_video import (  # noqa: E402
    make_intruder,
    make_light_switch,
    make_osd_clock,
    make_static,
    make_swaying_plant,
)

ENTER_FRAME, EXIT_FRAME = 60, 130


@pytest.fixture(scope="session")
def clips(tmp_path_factory) -> dict:
    """Build the synthetic clip set once for the whole session."""
    directory = tmp_path_factory.mktemp("clips")
    intruder, _window = make_intruder(
        directory / "intruder.mp4", enter_frame=ENTER_FRAME, exit_frame=EXIT_FRAME
    )
    return {
        "static": make_static(directory / "static_empty.mp4"),
        "intruder": intruder,
        "light_switch": make_light_switch(directory / "light_switch.mp4"),
        "osd_clock": make_osd_clock(directory / "osd_clock.mp4"),
        "swaying_plant": make_swaying_plant(directory / "swaying_plant.mp4"),
    }


@pytest.fixture
def client(tmp_path, monkeypatch):
    """A test client with isolated output folders."""
    job_root = tmp_path / "web_jobs"
    job_root.mkdir(parents=True)
    store = IncidentStore(tmp_path / "incidents.db", job_root)

    monkeypatch.setattr(web_app, "JOB_ROOT", job_root)
    monkeypatch.setattr(web_app, "UPLOAD_ROOT", tmp_path / "uploads")
    monkeypatch.setattr(web_app, "store", store)
    monkeypatch.setattr(web_app.jobs, "job_root", job_root)
    (tmp_path / "uploads").mkdir(parents=True, exist_ok=True)

    web_app.app.config.update(TESTING=True)
    with web_app.app.test_client() as test_client:
        yield test_client

    # The job manager is a module-level singleton shared by every test, so an
    # analysis still running here would catalogue its incidents into the *next*
    # test's store.  Stop anything in flight before the fixture is torn down.
    for job in web_app.jobs.list():
        web_app.jobs.cancel(job["job_id"])

    deadline = time.time() + 15
    while time.time() < deadline:
        running = [
            job for job in web_app.jobs.list() if job["status"] in {"queued", "running"}
        ]
        if not running:
            break
        time.sleep(0.05)


@pytest.fixture
def output_dir(tmp_path) -> Path:
    path = tmp_path / "incidents"
    path.mkdir(parents=True, exist_ok=True)
    return path
