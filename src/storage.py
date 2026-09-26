"""SQLite-backed incident catalogue.

The first version kept live clips in one SQLite table and discovered uploaded
incidents by walking the job folders, deriving an id by base64-encoding
``job_id/filename``.  That id was decoded and pasted straight into a path,
which let a crafted id delete files outside the output folder, and the two
listings could disagree about the same file.  Everything now goes through this
one catalogue, and paths are always rebuilt from validated parts.
"""

from __future__ import annotations

import logging
import re
import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional

LOGGER = logging.getLogger(__name__)

JOB_ID_PATTERN = re.compile(r"^[0-9a-f]{8,32}$")
MEDIA_NAME_PATTERN = re.compile(r"^incident_\d{3}\.(mp4|webm|avi|jpg|json)$")

SCHEMA = """
CREATE TABLE IF NOT EXISTS incidents (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id        TEXT    NOT NULL,
    source        TEXT    NOT NULL DEFAULT 'upload',
    source_name   TEXT,
    filename      TEXT    NOT NULL,
    thumbnail     TEXT,
    start_time_s  REAL    NOT NULL DEFAULT 0,
    end_time_s    REAL    NOT NULL DEFAULT 0,
    duration_s    REAL    NOT NULL DEFAULT 0,
    frame_count   INTEGER NOT NULL DEFAULT 0,
    fps           REAL    NOT NULL DEFAULT 0,
    max_objects   INTEGER NOT NULL DEFAULT 0,
    sensitivity   TEXT,
    size_bytes    INTEGER NOT NULL DEFAULT 0,
    created_at    TEXT    NOT NULL,
    UNIQUE (job_id, filename)
);
CREATE INDEX IF NOT EXISTS idx_incidents_created ON incidents (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_incidents_source  ON incidents (source);
"""


def is_valid_job_id(job_id: str) -> bool:
    return bool(job_id) and bool(JOB_ID_PATTERN.match(job_id))


def is_valid_media_name(name: str) -> bool:
    return bool(name) and bool(MEDIA_NAME_PATTERN.match(name))


@dataclass
class StoredIncident:
    id: int
    job_id: str
    source: str
    source_name: Optional[str]
    filename: str
    thumbnail: Optional[str]
    start_time_s: float
    end_time_s: float
    duration_s: float
    frame_count: int
    fps: float
    max_objects: int
    sensitivity: Optional[str]
    size_bytes: int
    created_at: str

    def to_dict(self) -> Dict[str, object]:
        data = self.__dict__.copy()
        data["media_url"] = f"/media/{self.job_id}/{self.filename}"
        data["view_url"] = f"/view/{self.job_id}/{self.filename}"
        data["thumbnail_url"] = (
            f"/media/{self.job_id}/{self.thumbnail}" if self.thumbnail else None
        )
        return data


class IncidentStore:
    """Thread-safe catalogue of recorded incidents."""

    def __init__(self, db_path: Path, job_root: Path) -> None:
        self.db_path = Path(db_path)
        self.job_root = Path(job_root)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        with self._connect() as conn:
            conn.executescript(SCHEMA)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(str(self.db_path), timeout=10)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            yield conn
            conn.commit()
        finally:
            conn.close()

    # -- paths -------------------------------------------------------------
    def media_path(self, job_id: str, filename: str) -> Optional[Path]:
        """Resolve a media file, refusing anything outside the job root.

        Both parts are pattern-checked *and* the resolved path is confirmed to
        live under the job root, so neither ``..`` segments nor symlinks can
        escape.
        """
        if not is_valid_job_id(job_id) or not is_valid_media_name(filename):
            return None
        root = self.job_root.resolve()
        candidate = (root / job_id / "incidents" / filename).resolve()
        if not candidate.is_relative_to(root):
            return None
        return candidate if candidate.exists() else None

    def incident_dir(self, job_id: str) -> Path:
        return self.job_root / job_id / "incidents"

    # -- writes ------------------------------------------------------------
    def add(
        self,
        job_id: str,
        record,
        *,
        source: str = "upload",
        source_name: Optional[str] = None,
    ) -> Optional[int]:
        """Catalogue one incident produced by the pipeline."""
        path = Path(record.path)
        size = path.stat().st_size if path.exists() else 0
        created = record.started_at or datetime.now(timezone.utc).isoformat(timespec="seconds")

        with self._lock, self._connect() as conn:
            try:
                cursor = conn.execute(
                    """
                    INSERT INTO incidents (
                        job_id, source, source_name, filename, thumbnail,
                        start_time_s, end_time_s, duration_s, frame_count, fps,
                        max_objects, sensitivity, size_bytes, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        job_id,
                        source,
                        source_name or record.source,
                        record.filename,
                        record.thumbnail,
                        float(record.start_time_s),
                        float(record.end_time_s),
                        float(record.duration_s),
                        int(record.frame_count),
                        float(record.fps),
                        int(record.max_objects),
                        record.sensitivity,
                        int(size),
                        created,
                    ),
                )
                return int(cursor.lastrowid)
            except sqlite3.IntegrityError:
                LOGGER.debug("Incident %s/%s is already catalogued", job_id, record.filename)
                return None

    def delete(self, incident_id: int) -> bool:
        """Remove an incident and its files."""
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT job_id, filename, thumbnail FROM incidents WHERE id = ?",
                (incident_id,),
            ).fetchone()
            if row is None:
                return False
            conn.execute("DELETE FROM incidents WHERE id = ?", (incident_id,))

        for name in (row["filename"], row["thumbnail"]):
            if not name:
                continue
            path = self.media_path(row["job_id"], name)
            if path is not None:
                path.unlink(missing_ok=True)
        # The JSON sidecar shares the clip's stem.
        clip = self.media_path(row["job_id"], row["filename"])
        if clip is not None:
            clip.with_suffix(".json").unlink(missing_ok=True)

        self._prune_empty_dirs(row["job_id"])
        return True

    def _prune_empty_dirs(self, job_id: str) -> None:
        if not is_valid_job_id(job_id):
            return
        incidents = self.job_root / job_id / "incidents"
        for directory in (incidents, incidents.parent):
            try:
                if directory.exists() and not any(directory.iterdir()):
                    directory.rmdir()
            except OSError:
                return

    # -- reads -------------------------------------------------------------
    def list(self, source: Optional[str] = None, limit: int = 200) -> List[StoredIncident]:
        query = "SELECT * FROM incidents"
        params: List[object] = []
        if source:
            query += " WHERE source = ?"
            params.append(source)
        query += " ORDER BY datetime(created_at) DESC, id DESC LIMIT ?"
        params.append(int(limit))

        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [self._row_to_incident(row) for row in rows]

    def get(self, incident_id: int) -> Optional[StoredIncident]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM incidents WHERE id = ?", (incident_id,)
            ).fetchone()
        return self._row_to_incident(row) if row else None

    def stats(self) -> Dict[str, object]:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT COUNT(*) AS total,
                       COALESCE(SUM(size_bytes), 0) AS bytes,
                       COALESCE(SUM(duration_s), 0) AS seconds
                FROM incidents
                """
            ).fetchone()
            by_source = conn.execute(
                "SELECT source, COUNT(*) AS n FROM incidents GROUP BY source"
            ).fetchall()
        return {
            "total": int(row["total"]),
            "size_bytes": int(row["bytes"]),
            "duration_s": round(float(row["seconds"]), 2),
            "by_source": {r["source"]: int(r["n"]) for r in by_source},
        }

    @staticmethod
    def _row_to_incident(row: sqlite3.Row) -> StoredIncident:
        return StoredIncident(
            id=int(row["id"]),
            job_id=row["job_id"],
            source=row["source"],
            source_name=row["source_name"],
            filename=row["filename"],
            thumbnail=row["thumbnail"],
            start_time_s=float(row["start_time_s"]),
            end_time_s=float(row["end_time_s"]),
            duration_s=float(row["duration_s"]),
            frame_count=int(row["frame_count"]),
            fps=float(row["fps"]),
            max_objects=int(row["max_objects"]),
            sensitivity=row["sensitivity"],
            size_bytes=int(row["size_bytes"]),
            created_at=row["created_at"],
        )

    # -- maintenance -------------------------------------------------------
    def prune_missing(self) -> int:
        """Forget incidents whose clip is no longer on disk.

        Files can disappear underneath the catalogue - a folder tidied by hand,
        a sync tool, a drive that was not mounted - and a row pointing at
        nothing shows up as a broken entry in the dashboard.  Only the row is
        removed here; nothing is deleted from disk.
        """
        with self._connect() as conn:
            rows = conn.execute("SELECT id, job_id, filename FROM incidents").fetchall()

        orphans = [
            row["id"]
            for row in rows
            if self.media_path(row["job_id"], row["filename"]) is None
        ]
        if not orphans:
            return 0

        with self._lock, self._connect() as conn:
            conn.executemany(
                "DELETE FROM incidents WHERE id = ?", [(i,) for i in orphans]
            )
        LOGGER.info("Removed %d catalogue entr(ies) with no file on disk", len(orphans))
        return len(orphans)

    def import_existing(self, sources: Optional[Iterable[Path]] = None) -> int:
        """Catalogue clips already on disk from earlier runs.

        Without this, upgrading would hide every clip recorded by the previous
        version.  Metadata comes from the JSON sidecar when one exists and from
        the file itself otherwise.
        """
        import json

        roots = list(sources) if sources is not None else [self.job_root]
        imported = 0

        for root in roots:
            root = Path(root)
            if not root.exists():
                continue
            for clip in sorted(root.glob("*/incidents/*")):
                if clip.suffix.lower() not in {".mp4", ".webm", ".avi"}:
                    continue
                job_id = clip.parent.parent.name
                if not is_valid_job_id(job_id) or not is_valid_media_name(clip.name):
                    continue

                meta: Dict[str, object] = {}
                sidecar = clip.with_suffix(".json")
                if sidecar.exists():
                    try:
                        meta = json.loads(sidecar.read_text(encoding="utf-8"))
                    except (ValueError, OSError):
                        meta = {}

                thumbnail = clip.with_suffix(".jpg")
                stat = clip.stat()
                created = meta.get("started_at") or datetime.fromtimestamp(
                    stat.st_mtime, tz=timezone.utc
                ).isoformat(timespec="seconds")

                with self._lock, self._connect() as conn:
                    try:
                        conn.execute(
                            """
                            INSERT INTO incidents (
                                job_id, source, source_name, filename, thumbnail,
                                start_time_s, end_time_s, duration_s, frame_count,
                                fps, max_objects, sensitivity, size_bytes, created_at
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                            (
                                job_id,
                                str(meta.get("source_kind", "archive")),
                                meta.get("source"),
                                clip.name,
                                thumbnail.name if thumbnail.exists() else None,
                                float(meta.get("start_time_s", 0) or 0),
                                float(meta.get("end_time_s", 0) or 0),
                                float(meta.get("duration_s", 0) or 0),
                                int(meta.get("frame_count", 0) or 0),
                                float(meta.get("fps", 0) or 0),
                                int(meta.get("max_objects", 0) or 0),
                                meta.get("sensitivity"),
                                stat.st_size,
                                created,
                            ),
                        )
                        imported += 1
                    except sqlite3.IntegrityError:
                        continue

        if imported:
            LOGGER.info("Catalogued %d incident clip(s) from earlier runs", imported)
        return imported
