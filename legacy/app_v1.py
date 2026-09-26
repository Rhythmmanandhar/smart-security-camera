from __future__ import annotations

import mimetypes
import os
import tempfile
import threading
import time
import urllib.parse
import urllib.request
import uuid
from pathlib import Path
from typing import Optional, Tuple

from flask import Flask, jsonify, render_template, request, send_from_directory, url_for, Response
from werkzeug.utils import secure_filename
import cv2
import sqlite3
from datetime import datetime
from collections import deque

from src.security_camera import SmartSecurityCamera


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = BASE_DIR / "outputs"
UPLOAD_ROOT = OUTPUT_ROOT / "web_uploads"
JOB_ROOT = OUTPUT_ROOT / "web_jobs"
ALLOWED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 1024 * 1024 * 1024  # 1 GB

# Global live camera state
live_camera_state = {
    "active": False,
    "frame": None,
    "frame_lock": threading.Lock(),
    "cap": None,
    "enable_tracking": False,
    "enable_recording": False,
    "output_dir": None,
    "job_id": None,
    "suppression_level": "medium",
}



@app.route("/")
def index() -> str:
    return render_template("index.html")


@app.route("/process", methods=["POST"])
def process_video():
    try:
        return _handle_process_request()
    except Exception as exc:  # noqa: BLE001 - surface friendly browser errors
        return jsonify({"ok": False, "error": str(exc)}), 400


@app.route("/jobs/<job_id>/<path:filename>")
def job_file(job_id: str, filename: str):
    job_dir = JOB_ROOT / job_id
    return send_from_directory(job_dir, filename, as_attachment=False, conditional=True)


@app.route("/jobs/<job_id>/view/<path:filename>")
def job_file_view(job_id: str, filename: str):
    file_url = url_for("job_file", job_id=job_id, filename=filename)
    stream_url = url_for("job_file_stream", job_id=job_id, filename=filename)
    return render_template(
        "clip_view.html",
        file_url=file_url,
        stream_url=stream_url,
        filename=Path(filename).name,
    )


@app.route("/jobs/<job_id>/stream/<path:filename>")
def job_file_stream(job_id: str, filename: str):
    video_path = JOB_ROOT / job_id / filename
    if not video_path.exists():
        return Response("Clip not found", status=404)

    def generate_stream():
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frame_delay = max(0.01, 1.0 / fps)

        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                ret, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if not ret:
                    continue

                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n"
                    + buffer.tobytes()
                    + b"\r\n"
                )
                time.sleep(frame_delay)
        finally:
            cap.release()

    return Response(generate_stream(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/live/start", methods=["POST"])
def live_start():
    try:
        if live_camera_state["active"]:
            return jsonify({"ok": False, "error": "Live camera already active"}), 400
        data = request.get_json() or {}
        enable_tracking = data.get("enable_tracking", False)
        enable_recording = data.get("enable_recording", False)
        suppression_level = data.get("suppression_level", "medium")

        live_camera_state["enable_tracking"] = enable_tracking
        live_camera_state["enable_recording"] = enable_recording
        live_camera_state["suppression_level"] = suppression_level

        # create a job id for this live session
        job_id = uuid.uuid4().hex[:12]
        live_camera_state["job_id"] = job_id

        if enable_recording:
            output_dir = JOB_ROOT / job_id / "incidents"
            output_dir.mkdir(parents=True, exist_ok=True)
            live_camera_state["output_dir"] = output_dir
        else:
            live_camera_state["output_dir"] = None

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            return jsonify({"ok": False, "error": "Could not open camera"}), 400

        live_camera_state["cap"] = cap
        live_camera_state["active"] = True

        # Start streaming thread
        thread = threading.Thread(target=_live_camera_thread, daemon=True)
        thread.start()

        return jsonify(
            {
                "ok": True,
                "tracking": enable_tracking,
                "recording": enable_recording,
                "job_id": job_id,
                "output_dir": str(live_camera_state.get("output_dir") or ""),
            }
        )
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400


@app.route("/live/stop", methods=["POST"])
def live_stop():
    live_camera_state["active"] = False
    if live_camera_state["cap"]:
        live_camera_state["cap"].release()
        live_camera_state["cap"] = None
    return jsonify({"ok": True})


@app.route("/live/feed")
def live_feed():
    def generate_mjpeg():
        while live_camera_state["active"]:
            with live_camera_state["frame_lock"]:
                frame = live_camera_state["frame"]
            if frame is None:
                continue
            ret, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if not ret:
                continue
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n"
                b"Content-Length: " + str(len(buffer)).encode() + b"\r\n\r\n"
                + buffer.tobytes()
                + b"\r\n"
            )

    return Response(
        generate_mjpeg(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )



def _handle_process_request():
    source_path = _resolve_source_video()

    job_id = uuid.uuid4().hex[:12]
    job_dir = JOB_ROOT / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    output_dir = job_dir / "incidents"
    output_dir.mkdir(parents=True, exist_ok=True)

    min_area = _get_int("min_area", 150)
    min_motion_frames = _get_int("min_motion_frames", 2)
    pre_event_frames = _get_int("pre_event_frames", 30)
    post_event_frames = _get_int("post_event_frames", 30)
    enable_tracking = request.form.get("enable_tracking") == "on"
    roi = _parse_roi(request.form)
    # Suppression level mapping: quick presets for persistence + movement filters
    suppression = request.form.get("suppression_level", "medium")
    if suppression == "low":
        persistence_frames = 2
        persistence_threshold = 1
        movement_thresh = 3.0
        movement_frames = 1
    elif suppression == "high":
        persistence_frames = 5
        persistence_threshold = 4
        movement_thresh = 8.0
        movement_frames = 3
    else:  # medium
        persistence_frames = 3
        persistence_threshold = 2
        movement_thresh = 5.0
        movement_frames = 2

    camera = SmartSecurityCamera(
        video_path=source_path,
        output_dir=output_dir,
        min_area=min_area,
        min_motion_frames=min_motion_frames,
        pre_event_frames=pre_event_frames,
        post_event_frames=post_event_frames,
        roi=roi,
        enable_tracking=enable_tracking,
        enable_live=False,
        persistence_frames=persistence_frames,
        persistence_threshold=persistence_threshold,
        movement_thresh=movement_thresh,
        movement_required_frames=movement_frames,
    )

    incidents = camera.process()

    results = []
    for incident in incidents:
        filename = incident.output_path.name
        results.append(
            {
                "filename": filename,
                "url": url_for("job_file", job_id=job_id, filename=f"incidents/{filename}"),
                "view_url": url_for("job_file_view", job_id=job_id, filename=f"incidents/{filename}"),
                "start_time_s": round(incident.start_time_s, 2),
                "end_time_s": round(incident.end_time_s, 2),
                "duration_s": round(max(0.0, incident.end_time_s - incident.start_time_s), 2),
            }
        )

    return jsonify(
        {
            "ok": True,
            "job_id": job_id,
            "source_name": source_path.name,
            "incidents_detected": len(results),
            "results": results,
            "tracking_enabled": enable_tracking,
            "output_folder": str(output_dir),
        }
    )


def _init_db():
    db_path = OUTPUT_ROOT / "live_clips.db"
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS clips (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            job_id TEXT,
            filename TEXT,
            start_s REAL,
            end_s REAL,
            duration REAL,
            path TEXT,
            created_at TEXT
        )
        """
    )
    conn.commit()
    conn.close()


def _insert_clip_record(job_id: str, filename: str, start_s: float, end_s: float, output_path: Path):
    db_path = OUTPUT_ROOT / "live_clips.db"
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO clips (job_id, filename, start_s, end_s, duration, path, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (job_id, filename, start_s, end_s, max(0.0, end_s - start_s), str(output_path), datetime.utcnow().isoformat()),
    )
    conn.commit()
    conn.close()


@app.route("/live/clips")
def live_clips():
    db_path = OUTPUT_ROOT / "live_clips.db"
    if not db_path.exists():
        return jsonify({"ok": True, "clips": []})
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    cur.execute("SELECT id, job_id, filename, start_s, end_s, duration, path, created_at FROM clips ORDER BY id DESC LIMIT 200")
    rows = cur.fetchall()
    conn.close()
    clips = [
        {
            "id": r[0],
            "job_id": r[1],
            "filename": r[2],
            "start_s": r[3],
            "end_s": r[4],
            "duration": r[5],
            "path": r[6],
            "created_at": r[7],
        }
        for r in rows
    ]
    return jsonify({"ok": True, "clips": clips})


@app.route("/live/clips/<int:clip_id>", methods=["DELETE"])
def delete_live_clip(clip_id: int):
    db_path = OUTPUT_ROOT / "live_clips.db"
    if not db_path.exists():
        return jsonify({"ok": False, "error": "Clip database not found"}), 404

    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    cur.execute("SELECT path FROM clips WHERE id = ?", (clip_id,))
    row = cur.fetchone()
    if not row:
        conn.close()
        return jsonify({"ok": False, "error": "Clip not found"}), 404

    clip_path = Path(row[0])
    cur.execute("DELETE FROM clips WHERE id = ?", (clip_id,))
    conn.commit()
    conn.close()

    if clip_path.exists():
        try:
            clip_path.unlink()
        except OSError:
            pass

    parent = clip_path.parent
    try:
        if parent.exists() and not any(parent.iterdir()):
            parent.rmdir()
    except OSError:
        pass

    return jsonify({"ok": True, "deleted": clip_id})


@app.route("/all-incidents")
def all_incidents():
    """List all incident files from all job folders (excluding live clips)."""
    # Get the exact live clip paths so we only exclude the live recordings themselves.
    live_clip_paths = set()
    db_path = OUTPUT_ROOT / "live_clips.db"
    if db_path.exists():
        conn = sqlite3.connect(str(db_path))
        cur = conn.cursor()
        cur.execute("SELECT path FROM clips")
        rows = cur.fetchall()
        conn.close()
        live_clip_paths = {str(Path(r[0]).resolve()).lower() for r in rows if r and r[0]}
    
    incidents = []
    if JOB_ROOT.exists():
        for job_dir in JOB_ROOT.iterdir():
            if not job_dir.is_dir():
                continue
            incidents_dir = job_dir / "incidents"
            if incidents_dir.exists():
                for mp4_file in incidents_dir.glob("*.mp4"):
                    # Skip only if this exact file path belongs to the live-clips table.
                    if str(mp4_file.resolve()).lower() in live_clip_paths:
                        continue
                    try:
                        stat = mp4_file.stat()
                        # Use URL-safe composite ID: base64(job_id/filename)
                        import base64
                        composite = f"{job_dir.name}/{mp4_file.name}"
                        safe_id = base64.urlsafe_b64encode(composite.encode()).decode().rstrip('=')
                        incidents.append(
                            {
                                "id": safe_id,
                                "job_id": job_dir.name,
                                "filename": mp4_file.name,
                                "path": str(mp4_file),
                                "size_bytes": stat.st_size,
                                "created_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                            }
                        )
                    except OSError:
                        pass
    # Sort by creation time, newest first
    incidents.sort(key=lambda x: x["created_at"], reverse=True)
    return jsonify({"ok": True, "incidents": incidents})


@app.route("/incidents/<path:incident_id>", methods=["DELETE"])
def delete_incident(incident_id: str):
    """Delete an incident file using base64-encoded job_id/filename ID."""
    if not incident_id:
        return jsonify({"ok": False, "error": "Invalid incident ID"}), 400

    try:
        import base64
        # Decode the base64 ID to get job_id/filename
        padding = 4 - (len(incident_id) % 4)
        if padding != 4:
            incident_id += "=" * padding
        composite = base64.urlsafe_b64decode(incident_id).decode()
        parts = composite.split("/", 1)
        if len(parts) != 2:
            raise ValueError("Invalid composite format")
        job_id, filename = parts
    except Exception:
        return jsonify({"ok": False, "error": "Invalid incident ID format"}), 400

    incident_path = JOB_ROOT / job_id / "incidents" / filename

    if not incident_path.exists():
        return jsonify({"ok": False, "error": "Incident file not found"}), 404

    try:
        incident_path.unlink()
    except OSError as e:
        return jsonify({"ok": False, "error": str(e)}), 500

    # Clean up empty parent folders
    try:
        parent = incident_path.parent
        if parent.exists() and not any(parent.iterdir()):
            parent.rmdir()
        job_folder = parent.parent
        if job_folder.exists() and not any(job_folder.iterdir()):
            job_folder.rmdir()
    except OSError:
        pass

    return jsonify({"ok": True, "deleted": incident_id})


def _resolve_source_video() -> Path:
    video_url = request.form.get("video_url", "").strip()
    upload = request.files.get("video_file")

    if upload and upload.filename:
        return _save_uploaded_file(upload)

    if video_url:
        return _download_video(video_url)

    raise ValueError("Please upload a video file or paste a direct video URL.")


def _save_uploaded_file(upload) -> Path:
    UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
    filename = secure_filename(upload.filename or "uploaded_video")
    suffix = Path(filename).suffix.lower() or ".mp4"
    if suffix not in ALLOWED_EXTENSIONS:
        raise ValueError("Unsupported file type. Please upload a video file.")

    temp_dir = Path(tempfile.mkdtemp(prefix="upload_", dir=UPLOAD_ROOT))
    destination = temp_dir / f"source{suffix}"
    upload.save(destination)
    return destination


def _download_video(video_url: str) -> Path:
    parsed = urllib.parse.urlparse(video_url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("Video URL must start with http:// or https://")

    suffix = Path(parsed.path).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        suffix = ".mp4"

    UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(tempfile.mkdtemp(prefix="download_", dir=UPLOAD_ROOT))
    destination = temp_dir / f"source{suffix}"

    with urllib.request.urlopen(video_url, timeout=30) as response, destination.open("wb") as out_file:  # nosec B310 - user-provided video fetch for local app
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            out_file.write(chunk)

    return destination


def _parse_roi(form) -> Optional[Tuple[float, float, float, float]]:
    raw = form.get("roi", "").strip()
    if not raw:
        return None

    parts = [item for item in raw.replace(",", " ").split() if item]
    if len(parts) != 4:
        raise ValueError("ROI must contain four values: x1 y1 x2 y2")

    values = tuple(float(item) for item in parts)
    return values


def _get_int(name: str, default: int) -> int:
    raw = request.form.get(name, "").strip()
    if not raw:
        return default
    return int(raw)


def _live_camera_thread():
    """Handle live camera frame processing in background."""
    cap = live_camera_state["cap"]
    if not cap:
        return
    # Initialize local detector/tracker using helper class methods
    from src.security_camera import SmartSecurityCamera, CentroidTracker

    helper_cam = SmartSecurityCamera(
        video_path=None,
        output_dir=JOB_ROOT / "live_temp",
        min_area=80,
        min_motion_frames=1,
        pre_event_frames=10,
        post_event_frames=10,
        enable_tracking=live_camera_state.get("enable_tracking", False),
        enable_live=True,
    )

    subtractor = cv2.createBackgroundSubtractorMOG2(history=150, varThreshold=12, detectShadows=False)
    tracker = CentroidTracker() if live_camera_state.get("enable_tracking") else None

    frame_idx = 0
    motion_streak = 0
    silent_streak = 0
    pre_buffer = []
    current_writer = None
    incident_counter = 0
    out_name = None

    # map suppression preset
    suppression = live_camera_state.get("suppression_level", "medium")
    if suppression == "low":
        persistence_frames = 2
        persistence_threshold = 1
        movement_thresh = 3.0
        movement_frames = 1
    elif suppression == "high":
        persistence_frames = 5
        persistence_threshold = 4
        movement_thresh = 8.0
        movement_frames = 3
    else:
        persistence_frames = 3
        persistence_threshold = 2
        movement_thresh = 5.0
        movement_frames = 2

    helper_cam._mask_history = deque(maxlen=persistence_frames)

    while live_camera_state["active"]:
        ok, frame = cap.read()
        if not ok:
            break

        frame_idx += 1

        boxes = helper_cam._detect_motion(frame, subtractor, helper_cam._roi_rect(frame.shape[1], frame.shape[0]))

        tracked_objects = None
        unauthorized = False
        if tracker is not None:
            tracked_objects = tracker.update(boxes)
            matched_ids = helper_cam._match_boxes_to_tracked_objects(boxes, tracked_objects)
            moving_ids = set()
            for mid in set([m for m in matched_ids if m is not None]):
                if tracker.is_moving(mid, movement_thresh, movement_frames):
                    moving_ids.add(mid)
            moving_boxes = 0
            for i, box in enumerate(boxes):
                mid = matched_ids[i] if i < len(matched_ids) else None
                if mid is None or mid in moving_ids:
                    moving_boxes += 1
            unauthorized = moving_boxes > 0
        else:
            unauthorized = len(boxes) > 0

        annotated = helper_cam._draw_annotations(frame, frame_idx, cap.get(cv2.CAP_PROP_FPS) or 25.0, boxes, unauthorized, helper_cam._roi_rect(frame.shape[1], frame.shape[0]), tracked_objects)

        # Update preview frame
        with live_camera_state["frame_lock"]:
            live_camera_state["frame"] = annotated

        # buffering
        pre_buffer.append(annotated.copy())
        if len(pre_buffer) > helper_cam.pre_event_frames:
            pre_buffer.pop(0)

        if unauthorized:
            motion_streak += 1
            silent_streak = 0
        else:
            silent_streak += 1
            motion_streak = 0

        # Recording/clipping
        if live_camera_state.get("enable_recording"):
            if current_writer is None and motion_streak >= helper_cam.min_motion_frames:
                incident_counter += 1
                current_incident_start = time.time()
                output_dir = live_camera_state.get("output_dir") or (JOB_ROOT / (live_camera_state.get("job_id") or "live") / "incidents")
                output_dir.mkdir(parents=True, exist_ok=True)
                out_name = f"incident_{incident_counter:03d}.mp4"
                out_path = output_dir / out_name
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(out_path), fourcc, cap.get(cv2.CAP_PROP_FPS) or 25.0, (frame.shape[1], frame.shape[0]))
                current_writer = writer
                for bf in pre_buffer:
                    current_writer.write(bf)

            if current_writer is not None:
                current_writer.write(annotated)
                if silent_streak >= helper_cam.post_event_frames:
                    current_writer.release()
                    current_writer = None
                    incident_end = time.time()
                    job_id = live_camera_state.get("job_id") or "live"
                    try:
                        _insert_clip_record(job_id, out_name, current_incident_start, incident_end, out_path)
                    except Exception:
                        pass

        time.sleep(0.01)

    # cleanup
    if current_writer is not None:
        current_writer.release()
    cap.release()
    live_camera_state["active"] = False




if __name__ == "__main__":
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    os.makedirs(UPLOAD_ROOT, exist_ok=True)
    os.makedirs(JOB_ROOT, exist_ok=True)
    _init_db()
    app.run(host="127.0.0.1", port=5000, debug=True)
