"""Flask front end for the smart security camera.

Routes are grouped under ``/api`` and every handler is thin: analysis runs in
:mod:`src.jobs`, the camera lives in :mod:`src.live` and incidents are
catalogued by :mod:`src.storage`.

Configuration comes from the environment:

    SC_HOST              bind address                  (default 127.0.0.1)
    SC_PORT              port                          (default 5000)
    SC_DEBUG             "1" to enable the debugger    (default off)
    SC_MAX_UPLOAD_MB     largest accepted upload       (default 512)
    SC_MAX_DOWNLOAD_MB   largest accepted URL fetch    (default 512)
    SC_WORKERS           concurrent analysis jobs      (default 2)
    SC_OUTPUT_DIR        where clips are written       (default ./outputs)
"""

from __future__ import annotations

import logging
import os
import tempfile
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path
from typing import Optional, Tuple

from src.nativelog import configure_quiet_opencv

configure_quiet_opencv()  # must precede the first OpenCV import

from flask import (  # noqa: E402
    Flask,
    Response,
    jsonify,
    render_template,
    request,
    send_file,
    url_for,
)
from werkzeug.exceptions import HTTPException, RequestEntityTooLarge  # noqa: E402
from werkzeug.utils import secure_filename  # noqa: E402

import cv2  # noqa: E402

from src.camera_source import (  # noqa: E402
    CameraError,
    list_devices,
    open_camera,
    open_file,
)
from src.classifier import is_yolo_available  # noqa: E402
from src.config import SENSITIVITY_LEVELS, build_config, normalise_sensitivity  # noqa: E402
from src.jobs import JobManager  # noqa: E402
from src.live import LiveSession  # noqa: E402
from src.storage import IncidentStore  # noqa: E402

LOGGER = logging.getLogger("security_camera.web")

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = Path(os.environ.get("SC_OUTPUT_DIR", BASE_DIR / "outputs")).resolve()
UPLOAD_ROOT = OUTPUT_ROOT / "web_uploads"
JOB_ROOT = OUTPUT_ROOT / "web_jobs"
DB_PATH = OUTPUT_ROOT / "incidents.db"

ALLOWED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
MAX_UPLOAD_MB = int(os.environ.get("SC_MAX_UPLOAD_MB", "512"))
MAX_DOWNLOAD_BYTES = int(os.environ.get("SC_MAX_DOWNLOAD_MB", "512")) * 1024 * 1024
DOWNLOAD_TIMEOUT_S = 30

MEDIA_TYPES = {
    ".mp4": "video/mp4",
    ".webm": "video/webm",
    ".avi": "video/x-msvideo",
    ".jpg": "image/jpeg",
    ".json": "application/json",
}

for directory in (OUTPUT_ROOT, UPLOAD_ROOT, JOB_ROOT):
    directory.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024
# Jinja caches compiled templates unless told otherwise, and the debugger is
# off by default here; without this an edited page needs a server restart.
app.config["TEMPLATES_AUTO_RELOAD"] = True
# Static assets are versioned by mtime below, so they can be cached briefly.
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0

store = IncidentStore(DB_PATH, JOB_ROOT)


def _catalogue(job_id: str, record, source: str = "upload") -> None:
    """Add a freshly recorded incident to the catalogue."""
    try:
        store.add(job_id, record, source=source)
    except Exception:  # pragma: no cover - never break capture on a DB error
        LOGGER.exception("Could not catalogue incident %s", record.filename)


jobs = JobManager(
    JOB_ROOT,
    max_workers=int(os.environ.get("SC_WORKERS", "2")),
    on_incident=lambda job_id, record: _catalogue(job_id, record, "upload"),
)
live = LiveSession(
    JOB_ROOT, on_incident=lambda job_id, record: _catalogue(job_id, record, "live")
)

_device_cache: dict = {"devices": None}


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------
@app.route("/")
def index() -> str:
    return render_template("index.html", sensitivity_levels=SENSITIVITY_LEVELS)


@app.route("/view/<job_id>/<filename>")
def view_clip(job_id: str, filename: str):
    path = store.media_path(job_id, filename)
    if path is None:
        return render_template("clip_view.html", not_found=True, filename=filename), 404
    return render_template(
        "clip_view.html",
        filename=filename,
        media_url=url_for("media", job_id=job_id, filename=filename),
        content_type=MEDIA_TYPES.get(path.suffix.lower(), "video/mp4"),
        not_found=False,
    )


# ---------------------------------------------------------------------------
# Media
# ---------------------------------------------------------------------------
@app.route("/media/<job_id>/<filename>")
def media(job_id: str, filename: str):
    """Serve a clip, thumbnail or metadata file from the job store.

    ``store.media_path`` validates both path segments and confirms the
    resolved file really is inside the output folder, so a crafted id cannot
    reach anything else on disk.
    """
    path = store.media_path(job_id, filename)
    if path is None:
        return jsonify({"ok": False, "error": "File not found"}), 404
    return send_file(
        path,
        mimetype=MEDIA_TYPES.get(path.suffix.lower()),
        conditional=True,  # byte-range support, so clips can be scrubbed
    )


# ---------------------------------------------------------------------------
# Preview frames (for drawing monitored and ignore zones)
# ---------------------------------------------------------------------------
@app.route("/api/preview", methods=["POST"])
def preview_frame():
    """Return one JPEG frame from an uploaded file or a URL.

    The browser extracts its own frame when it can decode the file, so this is
    the fallback for containers the browser cannot open (.avi, .mkv) and for
    URL sources.
    """
    source_path, _name = _resolve_source()
    try:
        source = open_file(source_path)
    except CameraError as exc:
        _discard_temp_source(source_path)
        return jsonify({"ok": False, "error": str(exc)}), 400

    try:
        # A frame ~10% in is more representative than the very first one.
        if source.frame_count > 20:
            source.capture.set(cv2.CAP_PROP_POS_FRAMES, int(source.frame_count * 0.1))
        ok, frame = source.read()
        if not ok:
            source.capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = source.read()
        if not ok:
            return jsonify({"ok": False, "error": "Could not read a frame."}), 400
        encoded, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
        if not encoded:
            return jsonify({"ok": False, "error": "Could not encode the frame."}), 500
    finally:
        source.release()
        _discard_temp_source(source_path)

    return Response(buffer.tobytes(), mimetype="image/jpeg",
                    headers={"Cache-Control": "no-store"})


@app.route("/api/live/preview")
def live_preview():
    """Return one JPEG frame from a camera, for drawing zones before starting.

    While a session is running its current frame is reused, so the camera is
    never opened twice.
    """
    frame = live.snapshot()
    if frame is not None:
        return Response(frame, mimetype="image/jpeg",
                        headers={"Cache-Control": "no-store"})

    try:
        device = int(request.args.get("device", 0))
    except ValueError:
        return jsonify({"ok": False, "error": "Invalid device index"}), 400

    try:
        source = open_camera(index=device, measure_fps=False)
    except CameraError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 503

    try:
        ok, image = source.read()
        if not ok:
            return jsonify({"ok": False, "error": "The camera returned no frame."}), 503
        encoded, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 82])
        if not encoded:
            return jsonify({"ok": False, "error": "Could not encode the frame."}), 500
    finally:
        source.release()

    return Response(buffer.tobytes(), mimetype="image/jpeg",
                    headers={"Cache-Control": "no-store"})


def _discard_temp_source(path: Path) -> None:
    """Delete a preview upload and its temporary folder."""
    try:
        path.unlink(missing_ok=True)
        parent = path.parent
        if parent.name.startswith(("upload_", "download_")) and not any(parent.iterdir()):
            parent.rmdir()
    except OSError:  # pragma: no cover - best effort
        LOGGER.debug("Could not remove the preview source %s", path)


# ---------------------------------------------------------------------------
# Analysis jobs
# ---------------------------------------------------------------------------
@app.route("/api/jobs", methods=["POST"])
def create_job():
    source_path, source_name = _resolve_source()
    config = _config_from_form()
    job = jobs.submit(source_path, source_name, config)
    return jsonify({"ok": True, **job.to_dict()}), 202


@app.route("/api/jobs/<job_id>")
def job_status(job_id: str):
    job = jobs.get(job_id)
    if job is None:
        return jsonify({"ok": False, "error": "Unknown job"}), 404
    return jsonify({"ok": True, **job.to_dict()})


@app.route("/api/jobs/<job_id>/cancel", methods=["POST"])
def job_cancel(job_id: str):
    if not jobs.cancel(job_id):
        return jsonify({"ok": False, "error": "Job is not running"}), 409
    return jsonify({"ok": True, "job_id": job_id, "status": "cancelling"})


@app.route("/api/jobs")
def job_list():
    return jsonify({"ok": True, "jobs": jobs.list()})


# ---------------------------------------------------------------------------
# Incident catalogue
# ---------------------------------------------------------------------------
@app.route("/api/incidents")
def incident_list():
    source = request.args.get("source") or None
    limit = min(500, max(1, int(request.args.get("limit", 200))))
    incidents = [incident.to_dict() for incident in store.list(source=source, limit=limit)]
    return jsonify({"ok": True, "incidents": incidents, "stats": store.stats()})


@app.route("/api/incidents/<int:incident_id>", methods=["DELETE"])
def incident_delete(incident_id: int):
    if not store.delete(incident_id):
        return jsonify({"ok": False, "error": "Incident not found"}), 404
    return jsonify({"ok": True, "deleted": incident_id})


@app.route("/live/clips")
def legacy_live_clips():
    """Kept so the documented v1 endpoint keeps working."""
    clips = [incident.to_dict() for incident in store.list(source="live")]
    return jsonify({"ok": True, "clips": clips})


# ---------------------------------------------------------------------------
# Live camera
# ---------------------------------------------------------------------------
@app.route("/api/live/devices")
def live_devices():
    """List usable cameras.  Probing is slow, so the result is cached."""
    if request.args.get("refresh") == "1" or _device_cache["devices"] is None:
        _device_cache["devices"] = [device.to_dict() for device in list_devices()]
    return jsonify({"ok": True, "devices": _device_cache["devices"]})


@app.route("/api/live/start", methods=["POST"])
def live_start():
    data = request.get_json(silent=True) or {}
    config = build_config(
        normalise_sensitivity(data.get("sensitivity")),
        detection={
            "roi": _zone_from_json(data.get("roi")),
            "ignore_zones": _zones_from_json(data.get("ignore_zones")) or None,
        },
        camera={"device_index": int(data.get("device", 0))},
        classifier=_classifier_options(
            data.get("detection") or data.get("classifier"),
            data.get("classes"),
            # A camera cannot wait ~200 ms per frame for the detector, so the
            # live session runs it a few times a second and keeps tracking in
            # between; offline analysis has no such constraint.
            live=True,
        ),
        draw_tracks=bool(data.get("tracking", True)),
        record_incidents=bool(data.get("recording", True)),
    )
    job_id = uuid.uuid4().hex[:12]
    try:
        info = live.start(job_id, config, record=config.record_incidents)
    except CameraError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 503
    except RuntimeError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 409
    return jsonify({"ok": True, **info})


@app.route("/api/live/stop", methods=["POST"])
def live_stop():
    return jsonify({"ok": True, "stats": live.stop()})


@app.route("/api/live/status")
def live_status():
    return jsonify({"ok": True, "stats": live.stats()})


@app.route("/api/live/feed")
def live_feed():
    if not live.active:
        return jsonify({"ok": False, "error": "The live camera is not running"}), 409
    return Response(
        live.frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
    )


@app.route("/api/live/snapshot")
def live_snapshot():
    frame = live.snapshot()
    if frame is None:
        return jsonify({"ok": False, "error": "No frame available"}), 409
    return Response(
        frame,
        mimetype="image/jpeg",
        headers={"Content-Disposition": 'attachment; filename="snapshot.jpg"'},
    )


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------
@app.route("/api/health")
def health():
    return jsonify(
        {
            "ok": True,
            "yolo_available": is_yolo_available(),
            "live_active": live.active,
            "jobs_running": sum(1 for job in jobs.list() if job["status"] == "running"),
            "incidents": store.stats(),
            "output_root": str(OUTPUT_ROOT),
        }
    )


# ---------------------------------------------------------------------------
# Input handling
# ---------------------------------------------------------------------------
def _config_from_form():
    form = request.form
    roi = _parse_roi(form.get("roi", ""))
    ignore = _parse_zones(form.get("ignore_zones", ""))
    return build_config(
        normalise_sensitivity(form.get("sensitivity")),
        detection={
            "roi": roi,
            "ignore_zones": ignore or None,
            "min_area_ratio": _optional_float("min_area_ratio"),
        },
        incident={
            "pre_event_seconds": _optional_float("pre_event_seconds"),
            "post_event_seconds": _optional_float("post_event_seconds"),
            "min_motion_seconds": _optional_float("min_motion_seconds"),
        },
        classifier=_classifier_options(
            form.get("detection") or form.get("classifier"), form.get("classes")
        ),
        draw_tracks=form.get("tracking", "on") == "on",
    )


def _optional_float(name: str) -> Optional[float]:
    raw = (request.form.get(name) or "").strip()
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"{name.replace('_', ' ')} must be a number") from exc


def _parse_roi(raw: str):
    raw = (raw or "").strip()
    if not raw:
        return None
    parts = [item for item in raw.replace(",", " ").split() if item]
    if len(parts) != 4:
        raise ValueError("The ROI needs four numbers: x1 y1 x2 y2 (each 0-1).")
    try:
        values = tuple(float(item) for item in parts)
    except ValueError as exc:
        raise ValueError("The ROI values must be numbers between 0 and 1.") from exc
    if any(not 0.0 <= value <= 1.0 for value in values):
        raise ValueError("The ROI values must be between 0 and 1.")
    if values[0] >= values[2] or values[1] >= values[3]:
        raise ValueError("The ROI must have x1 < x2 and y1 < y2.")
    return values


def _classifier_options(detection, classes, live: bool = False):
    """Translate the dashboard's single "detection" choice into a config.

    * ``motion``       - background subtraction only (no dependency)
    * ``yolo``         - motion finds the boxes, YOLO vetoes them
    * ``yolo-objects`` - YOLO finds the objects; one labelled box each
    * ``hog``          - the built-in people detector, as a filter
    """
    choice = (detection or "none").strip().lower()
    aliases = {"": "none", "motion": "none", "off": "none"}
    choice = aliases.get(choice, choice)

    if choice in {"yolo-objects", "objects", "detect"}:
        return {
            "kind": "yolo",
            "mode": "detect",
            "classes": _classes_from_json(classes),
            # Offline analysis can afford a pass per frame; a live camera
            # gets an interval it can actually keep up with.
            "interval_seconds": 0.4 if live else 0.0,
        }
    if choice in {"yolo", "hog"}:
        return {"kind": choice, "mode": "filter", "classes": _classes_from_json(classes)}
    return {"kind": "none"}


def _zone_from_json(value):
    """Validate a rectangle sent as a JSON list of four numbers."""
    if not value:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError("A zone needs four numbers: x1 y1 x2 y2 (each 0-1).")
    return _parse_roi(" ".join(str(part) for part in value))


def _zones_from_json(value):
    if not value:
        return ()
    if not isinstance(value, (list, tuple)):
        raise ValueError("Ignore zones must be a list of rectangles.")
    return tuple(zone for zone in (_zone_from_json(item) for item in value) if zone)


def _classes_from_json(value):
    if not value:
        return None
    if isinstance(value, str):
        names = [part.strip().lower() for part in value.split(",")]
    else:
        names = [str(part).strip().lower() for part in value]
    return tuple(name for name in names if name) or None


def _parse_zones(raw: str):
    raw = (raw or "").strip()
    if not raw:
        return ()
    zones = []
    for line in raw.splitlines():
        if line.strip():
            zones.append(_parse_roi(line))
    return tuple(zone for zone in zones if zone)


def _resolve_source() -> Tuple[Path, str]:
    upload = request.files.get("video_file")
    if upload and upload.filename:
        return _save_upload(upload)

    video_url = (request.form.get("video_url") or "").strip()
    if video_url:
        return _download(video_url)

    raise ValueError("Choose a video file or paste a direct video URL.")


def _save_upload(upload) -> Tuple[Path, str]:
    original = secure_filename(upload.filename or "video")
    suffix = Path(original).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise ValueError(
            f"{suffix or 'That file type'} is not supported. Use "
            + ", ".join(sorted(ALLOWED_EXTENSIONS))
        )

    # Recreated on demand: the folder holds only temporary copies, so deleting
    # it while the server runs must not break the next upload.
    UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(tempfile.mkdtemp(prefix="upload_", dir=UPLOAD_ROOT))
    destination = temp_dir / f"source{suffix}"
    upload.save(destination)
    if destination.stat().st_size == 0:
        destination.unlink(missing_ok=True)
        raise ValueError("The uploaded file is empty.")
    return destination, original


def _download(video_url: str) -> Tuple[Path, str]:
    parsed = urllib.parse.urlparse(video_url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("The video URL must start with http:// or https://")

    suffix = Path(parsed.path).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        suffix = ".mp4"

    UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(tempfile.mkdtemp(prefix="download_", dir=UPLOAD_ROOT))
    destination = temp_dir / f"source{suffix}"

    request_obj = urllib.request.Request(
        video_url, headers={"User-Agent": "smart-security-camera/2.0"}
    )
    try:
        # nosec B310 - the scheme is checked above; this is a local-only tool.
        with urllib.request.urlopen(request_obj, timeout=DOWNLOAD_TIMEOUT_S) as response:
            declared = response.headers.get("Content-Length")
            if declared and int(declared) > MAX_DOWNLOAD_BYTES:
                raise ValueError(
                    f"That video is larger than the {MAX_DOWNLOAD_BYTES // (1024 * 1024)} MB limit."
                )
            written = 0
            with destination.open("wb") as out_file:
                while True:
                    chunk = response.read(1024 * 256)
                    if not chunk:
                        break
                    written += len(chunk)
                    if written > MAX_DOWNLOAD_BYTES:
                        raise ValueError(
                            "The download exceeded the "
                            f"{MAX_DOWNLOAD_BYTES // (1024 * 1024)} MB limit."
                        )
                    out_file.write(chunk)
    except urllib.error.HTTPError as exc:
        raise ValueError(f"The server returned HTTP {exc.code} for that URL.") from exc
    except urllib.error.URLError as exc:
        raise ValueError(f"Could not reach that URL: {exc.reason}") from exc
    except OSError as exc:
        raise ValueError(f"Could not download that URL: {exc}") from exc

    if destination.stat().st_size == 0:
        raise ValueError("The downloaded file is empty.")
    return destination, Path(parsed.path).name or "downloaded video"


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------
@app.errorhandler(ValueError)
def handle_value_error(error: ValueError):
    return jsonify({"ok": False, "error": str(error)}), 400


@app.errorhandler(CameraError)
def handle_camera_error(error: CameraError):
    return jsonify({"ok": False, "error": str(error)}), 503


@app.errorhandler(RequestEntityTooLarge)
def handle_too_large(_error):
    return (
        jsonify({"ok": False, "error": f"The upload is larger than {MAX_UPLOAD_MB} MB."}),
        413,
    )


@app.errorhandler(HTTPException)
def handle_http_exception(error: HTTPException):
    if request.path.startswith("/api/") or request.accept_mimetypes.best == "application/json":
        return jsonify({"ok": False, "error": error.description}), error.code
    return error


@app.errorhandler(Exception)
def handle_unexpected(error: Exception):
    LOGGER.exception("Unhandled error on %s", request.path)
    return jsonify({"ok": False, "error": "Internal server error"}), 500


def create_app() -> Flask:
    """Entry point for WSGI servers."""
    return app


def main() -> None:
    logging.basicConfig(
        level=logging.DEBUG if os.environ.get("SC_DEBUG") == "1" else logging.INFO,
        format="%(asctime)s  %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    imported = store.import_existing()
    if imported:
        LOGGER.info("Imported %d clip(s) recorded by earlier runs", imported)
    pruned = store.prune_missing()
    if pruned:
        LOGGER.info("Dropped %d catalogue entr(ies) whose clip is gone", pruned)

    host = os.environ.get("SC_HOST", "127.0.0.1")
    port = int(os.environ.get("SC_PORT", "5000"))
    debug = os.environ.get("SC_DEBUG") == "1"

    LOGGER.info("Smart Security Camera on http://%s:%d  (output: %s)", host, port, OUTPUT_ROOT)
    try:
        app.run(host=host, port=port, debug=debug, threaded=True)
    finally:
        live.stop()
        jobs.shutdown()


if __name__ == "__main__":
    main()
