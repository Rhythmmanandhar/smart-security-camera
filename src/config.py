"""Configuration objects for the smart security camera.

All tunables live here so the CLI, the web app and the tests share one
definition of what a "sensitivity level" means.  Timing values are expressed
in *seconds* rather than frames: the same configuration then behaves the same
way on a 6 fps CCTV export and a 30 fps webcam feed.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Dict, Optional, Tuple

# Normalised rectangle (x1, y1, x2, y2) with values in 0..1.
NormRect = Tuple[float, float, float, float]


@dataclass(frozen=True)
class DetectionConfig:
    """Foreground-segmentation and blob-filtering parameters."""

    # --- frame handling -------------------------------------------------
    process_width: int = 480
    """Frames are analysed at this width (px) and results scaled back.

    Keeping the analysis resolution fixed makes every pixel-based threshold
    behave identically for 480p, 720p and 4K sources, and keeps the cost of a
    frame roughly constant.
    """

    warmup_frames: int = 40
    """Upper bound on the frames used to learn the background before any alarm
    is raised.  Without a warm-up the very first frame is 100% foreground,
    which used to make every single video open with a false incident."""

    warmup_seconds: float = 1.2
    """Preferred warm-up length in seconds; the detector uses whichever of the
    two limits is smaller, so low-frame-rate sources are not blinded."""

    # --- background subtraction ----------------------------------------
    bg_history: int = 400
    var_threshold: float = 24.0
    idle_learning_rate: float = 0.0025
    motion_learning_rate: float = 0.0005
    """Slower adaptation while something is moving, so a person who pauses for
    a moment is not absorbed into the background."""
    warmup_learning_rate: float = 0.08

    # --- blob filtering --------------------------------------------------
    min_area_ratio: float = 0.0015
    """Minimum blob area as a fraction of the analysed frame area."""
    max_area_ratio: float = 0.55
    min_box_side: int = 6
    """Smallest accepted box side, in analysis pixels.  A person at the far end
    of a warehouse is only a handful of pixels wide; at 10 they were discarded
    before any other rule saw them."""
    min_fill_ratio: float = 0.18
    """contour area / bounding-box area - rejects scattered noise skeletons."""
    min_aspect_ratio: float = 0.12
    max_aspect_ratio: float = 8.0
    perspective_compensation: bool = True
    """Overhead cameras render distant objects smaller, so the size a blob must
    reach is scaled down towards the top of the frame."""

    perspective_top_scale: float = 0.22
    """Area threshold at the top of the frame, as a fraction of the threshold
    at the bottom.  At the previous 0.45 a person by the far wall still fell
    short of the minimum and only nearby movement was ever reported."""

    # --- scene-level guards ---------------------------------------------
    max_foreground_ratio: float = 0.45
    """Above this fraction of moving pixels the scene itself changed (lights
    switched on, camera bumped, auto-exposure jump).  Those frames are used to
    re-learn the background instead of raising an alarm."""
    scene_change_relearn_frames: int = 15

    # --- morphology / merging -------------------------------------------
    open_kernel: int = 3
    close_kernel: int = 9
    merge_iou: float = 0.05
    merge_gap_x_ratio: float = 0.18
    """How far apart two fragments may be horizontally (as a fraction of the
    wider box) and still be treated as one object."""
    merge_gap_y_ratio: float = 0.85
    """The same vertically.  Larger, because background subtraction splits an
    upright body into stacked pieces far more often than side-by-side ones."""
    merge_max_union_aspect: float = 1.35
    """Refuse a merge that would produce a box wider than this relative to its
    height: one body stays upright, so a wide union means two people."""

    # --- masking ----------------------------------------------------------
    roi: Optional[NormRect] = None
    """Only motion inside this normalised rectangle is considered."""
    ignore_zones: Tuple[NormRect, ...] = ()
    """Normalised rectangles that never raise an alarm - useful for burnt-in
    timestamps, ticker overlays, a swaying tree or a busy public pavement."""
    roi_overlap: float = 0.35
    """Fraction of a box that must fall inside the ROI to count."""


@dataclass(frozen=True)
class TrackingConfig:
    """Centroid-tracker parameters (all distances relative to frame size)."""

    max_match_distance_ratio: float = 0.09
    """Maximum centroid travel between frames, as a fraction of the frame
    diagonal, for two detections to be considered the same object."""
    max_missing_frames: int = 15
    min_hits: int = 3
    """Detections needed before a track is trusted - kills one-frame noise."""
    history_frames: int = 12
    min_displacement_ratio: float = 0.008
    """Net (not per-step) displacement over the history window required to
    call a track "moving".  Net displacement ignores jitter, which oscillates
    around a fixed point and used to be counted as movement."""


@dataclass(frozen=True)
class IncidentConfig:
    """Incident state machine and recording parameters."""

    min_motion_seconds: float = 0.6
    """Continuous motion needed to open an incident."""
    post_event_seconds: float = 2.5
    """Quiet time before an incident is closed."""
    pre_event_seconds: float = 2.0
    """Footage kept from *before* the trigger."""
    min_incident_seconds: float = 1.0
    """Shorter recordings are discarded as noise."""
    max_incident_seconds: float = 300.0
    """Hard cap so a busy scene cannot produce one enormous file."""

    # Explicit frame counts override the seconds-based values when set.
    min_motion_frames: Optional[int] = None
    post_event_frames: Optional[int] = None
    pre_event_frames: Optional[int] = None

    def frames(self, fps: float) -> Dict[str, int]:
        """Resolve the timing values into frame counts for a given fps."""
        fps = fps if fps and fps > 0 else 25.0

        def to_frames(seconds: float, minimum: int) -> int:
            return max(minimum, int(round(seconds * fps)))

        return {
            "min_motion": self.min_motion_frames or to_frames(self.min_motion_seconds, 2),
            "post_event": self.post_event_frames or to_frames(self.post_event_seconds, 3),
            "pre_event": self.pre_event_frames
            if self.pre_event_frames is not None
            else to_frames(self.pre_event_seconds, 0),
            "max_incident": to_frames(self.max_incident_seconds, 30),
        }


@dataclass(frozen=True)
class CameraConfig:
    """Live-capture parameters."""

    device_index: int = 0
    width: int = 1280
    height: int = 720
    target_fps: float = 30.0
    warmup_reads: int = 5
    """Frames discarded after opening, while auto-exposure settles."""
    max_reconnect_attempts: int = 3


@dataclass(frozen=True)
class ClassifierConfig:
    """Optional "what is it?" stage that runs after motion detection."""

    kind: str = "none"
    """``none``, ``hog`` (OpenCV people detector) or ``yolo`` (YOLOv8n)."""

    mode: str = "filter"
    """``filter`` - motion finds the boxes and the classifier vetoes them.
    ``detect`` - the model finds the objects and each one becomes a single
    labelled box, which is the only way to get one box per person: background
    subtraction returns body parts, not bodies.  ``detect`` needs ``yolo``."""

    image_size: int = 960
    """Inference size.  Distant people are small; at 640 the model found four
    of the eight people in the sample frame, at 960 it found seven."""

    classes: Tuple[str, ...] = (
        "person",
        "bicycle",
        "car",
        "motorcycle",
        "bus",
        "truck",
        "cat",
        "dog",
    )
    """Only these classes raise an alarm (``yolo`` only)."""

    confidence: float = 0.25
    interval_seconds: float = 0.5
    """How often a track is re-checked.  Between checks the previous verdict
    stands, which keeps the cost of classification roughly constant.  Zero runs
    the model on every frame, which is what offline analysis wants."""

    min_interval_frames: int = 3

    new_object_seconds: float = 2.0
    """In detect mode, an object that has only just appeared raises the alarm
    even before it has travelled anywhere: someone who steps into view and
    stands still is exactly the event a security camera exists for."""

    @property
    def enabled(self) -> bool:
        return self.kind not in {"", "none", "off"}

    @property
    def detects(self) -> bool:
        """True when the model supplies the boxes instead of vetting them."""
        return self.enabled and self.mode == "detect" and self.kind.startswith("yolo")

    def interval_frames(self, fps: float) -> int:
        fps = fps if fps and fps > 0 else 25.0
        if self.interval_seconds <= 0:
            return 1
        return max(self.min_interval_frames, int(round(self.interval_seconds * fps)))

    def new_object_frames(self, fps: float) -> int:
        fps = fps if fps and fps > 0 else 25.0
        return max(1, int(round(self.new_object_seconds * fps)))


@dataclass(frozen=True)
class PipelineConfig:
    """Everything the pipeline needs, in one object."""

    detection: DetectionConfig = field(default_factory=DetectionConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    incident: IncidentConfig = field(default_factory=IncidentConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    classifier: ClassifierConfig = field(default_factory=ClassifierConfig)
    sensitivity: str = "balanced"
    record_incidents: bool = True
    draw_tracks: bool = True

    @property
    def require_person(self) -> bool:
        """Backwards-compatible view of the old boolean flag."""
        return self.classifier.enabled


# ---------------------------------------------------------------------------
# Sensitivity presets
# ---------------------------------------------------------------------------
# "low"      - only large, obvious, sustained movement (noisy outdoor scenes)
# "balanced" - the sane default for indoor/warehouse CCTV
# "high"     - small or brief movement still alarms (quiet, controlled rooms)

_SENSITIVITY_PRESETS: Dict[str, Dict[str, Dict[str, object]]] = {
    "low": {
        "detection": {
            "min_area_ratio": 0.0045,
            "var_threshold": 36.0,
            "min_fill_ratio": 0.25,
        },
        "tracking": {
            "min_hits": 5,
            "min_displacement_ratio": 0.018,
        },
        "incident": {
            "min_motion_seconds": 1.2,
            "min_incident_seconds": 1.5,
        },
    },
    "balanced": {"detection": {}, "tracking": {}, "incident": {}},
    "high": {
        "detection": {
            "min_area_ratio": 0.0006,
            "var_threshold": 16.0,
            "min_fill_ratio": 0.12,
        },
        "tracking": {
            "min_hits": 2,
            "min_displacement_ratio": 0.004,
        },
        "incident": {
            "min_motion_seconds": 0.3,
            "min_incident_seconds": 0.6,
        },
    },
}

# Accepted spellings, including the legacy "suppression level" wording where a
# *high* suppression level meant a *low* sensitivity.
_SENSITIVITY_ALIASES: Dict[str, str] = {
    "low": "low",
    "least": "low",
    "strict": "low",
    "medium": "balanced",
    "normal": "balanced",
    "balanced": "balanced",
    "default": "balanced",
    "high": "high",
    "max": "high",
    "sensitive": "high",
}

SENSITIVITY_LEVELS = ("low", "balanced", "high")


def normalise_sensitivity(name: Optional[str]) -> str:
    """Map any accepted spelling onto a canonical preset name."""
    if not name:
        return "balanced"
    return _SENSITIVITY_ALIASES.get(str(name).strip().lower(), "balanced")


def build_config(
    sensitivity: str = "balanced",
    *,
    detection: Optional[Dict[str, object]] = None,
    tracking: Optional[Dict[str, object]] = None,
    incident: Optional[Dict[str, object]] = None,
    camera: Optional[Dict[str, object]] = None,
    classifier: Optional[Dict[str, object]] = None,
    require_person: Optional[bool] = None,
    **pipeline_kwargs: object,
) -> PipelineConfig:
    """Build a :class:`PipelineConfig` from a preset plus explicit overrides.

    Overrides always win over the preset, and ``None`` values are ignored so
    callers can pass optional form fields straight through.
    """
    level = normalise_sensitivity(sensitivity)
    preset = _SENSITIVITY_PRESETS[level]

    # ``require_person=True`` was the v1 spelling of "classify before alerting".
    classifier_values = dict(classifier or {})
    if require_person and not classifier_values.get("kind"):
        classifier_values["kind"] = "hog"

    def merge(base, preset_values, overrides):
        values = dict(preset_values)
        for key, value in (overrides or {}).items():
            if value is not None:
                values[key] = value
        return replace(base, **values) if values else base

    return PipelineConfig(
        detection=merge(DetectionConfig(), preset["detection"], detection),
        tracking=merge(TrackingConfig(), preset["tracking"], tracking),
        incident=merge(IncidentConfig(), preset["incident"], incident),
        camera=merge(CameraConfig(), {}, camera),
        classifier=merge(ClassifierConfig(), {}, classifier_values),
        sensitivity=level,
        **{k: v for k, v in pipeline_kwargs.items() if v is not None},
    )
