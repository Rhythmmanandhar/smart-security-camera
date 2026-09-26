"""Smart Security Camera package."""

from .config import PipelineConfig, build_config, normalise_sensitivity
from .geometry import MotionBox
from .pipeline import FrameResult, SecurityPipeline
from .recorder import IncidentRecord
from .security_camera import Progress, SmartSecurityCamera
from .tracker import CentroidTracker, Track

__all__ = [
    "PipelineConfig",
    "build_config",
    "normalise_sensitivity",
    "MotionBox",
    "FrameResult",
    "SecurityPipeline",
    "IncidentRecord",
    "Progress",
    "SmartSecurityCamera",
    "CentroidTracker",
    "Track",
]
