"""A/B benchmark: the archived v1 detector against the current pipeline.

    python tools/benchmark.py                      # synthetic ground truth
    python tools/benchmark.py cctv_video_1.mp4     # plus real footage

Incident counts alone hide most of the difference between the two versions, so
this reports frame-level behaviour:

* ``fp frames``  - frames flagged as motion in a clip where nothing happens
                   (lower is better; the ideal is 0)
* ``recall``     - share of the frames containing the intruder that were
                   flagged
* ``box IoU``    - mean overlap between the detected box and the true box,
                   which is how box inflation from over-dilation shows up
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from legacy.security_camera_v1 import CentroidTracker as LegacyTracker  # noqa: E402
from legacy.security_camera_v1 import SmartSecurityCamera as LegacyCamera  # noqa: E402
from src.config import build_config  # noqa: E402
from src.pipeline import SecurityPipeline  # noqa: E402
from tools.make_test_video import (  # noqa: E402
    intruder_box,
    make_intruder,
    make_light_switch,
    make_osd_clock,
    make_static,
    make_swaying_plant,
)

ENTER_FRAME, EXIT_FRAME = 60, 130


def _box_iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / float(area_a + area_b - inter)


def _score(
    flagged: Dict[int, List[Tuple[int, int, int, int]]],
    total_frames: int,
    truth: bool,
) -> Dict[str, float]:
    """Turn per-frame boxes into false-positive / recall / IoU numbers."""
    if not truth:
        return {
            "fp_frames": float(len(flagged)),
            "fp_rate": len(flagged) / max(1, total_frames),
            "recall": float("nan"),
            "iou": float("nan"),
            "inflation": float("nan"),
        }

    hits, ious, inflation = 0, [], []
    truth_frames = 0
    false_frames = 0
    for index in range(total_frames):
        expected = intruder_box(index, ENTER_FRAME, EXIT_FRAME)
        boxes = flagged.get(index, [])
        if expected is None:
            false_frames += 1 if boxes else 0
            continue
        truth_frames += 1
        if boxes:
            hits += 1
            best = max(boxes, key=lambda box: _box_iou(expected, box))
            ious.append(_box_iou(expected, best))
            truth_area = (expected[2] - expected[0]) * (expected[3] - expected[1])
            best_area = (best[2] - best[0]) * (best[3] - best[1])
            inflation.append(best_area / float(truth_area))
    return {
        "fp_frames": float(false_frames),
        "fp_rate": false_frames / max(1, total_frames - truth_frames),
        "recall": hits / max(1, truth_frames),
        "iou": float(np.mean(ious)) if ious else 0.0,
        "inflation": float(np.mean(inflation)) if inflation else 0.0,
    }


def run_legacy_frames(path: Path) -> Tuple[Dict[int, List], int]:
    """Per-frame motion boxes from the v1 detector."""
    camera = LegacyCamera(
        video_path=path,
        output_dir=Path(tempfile.mkdtemp(prefix="legacy_")),
        min_area=150,
        min_motion_frames=2,
        enable_tracking=True,
    )
    subtractor = cv2.createBackgroundSubtractorMOG2(
        history=150, varThreshold=12, detectShadows=False
    )
    tracker = LegacyTracker()
    cap = cv2.VideoCapture(str(path))
    flagged: Dict[int, List] = {}
    index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        boxes = camera._detect_motion(frame, subtractor, None)
        tracked = tracker.update(boxes)
        matched = camera._match_boxes_to_tracked_objects(boxes, tracked)
        moving = {
            mid
            for mid in {m for m in matched if m is not None}
            if tracker.is_moving(mid, 5.0, 2)
        }
        alerting = [
            (b.x, b.y, b.x + b.w, b.y + b.h)
            for i, b in enumerate(boxes)
            if (matched[i] if i < len(matched) else None) is None or matched[i] in moving
        ]
        if alerting:
            flagged[index] = alerting
        index += 1
    cap.release()
    return flagged, index


def run_current_frames(path: Path, sensitivity: str = "balanced") -> Tuple[Dict[int, List], int]:
    """Per-frame alerting boxes from the current pipeline."""
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
    ok, frame = cap.read()
    if not ok:
        cap.release()
        return {}, 0
    height, width = frame.shape[:2]
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    config = build_config(sensitivity, record_incidents=False)
    pipeline = SecurityPipeline(config, fps=fps, frame_size=(width, height))

    flagged: Dict[int, List] = {}
    index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        result = pipeline.process_frame(frame)
        if result.alert:
            alerting = {tid for tid in result.alerting_ids}
            flagged[index] = [
                t.box.rect for t in result.tracks if t.track_id in alerting
            ]
        index += 1
    cap.release()
    pipeline.close()
    return flagged, index


CASES = [
    ("static_empty", "empty room, sensor noise", False),
    ("intruder", "person crossing the room", True),
    ("light_switch", "lights switched on", False),
    ("osd_clock", "burnt-in clock ticking", False),
    ("swaying_plant", "foliage swaying in place", False),
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("videos", nargs="*", type=Path)
    args = parser.parse_args()

    workspace = Path(tempfile.mkdtemp(prefix="bench_"))
    clips = workspace / "clips"
    make_static(clips / "static_empty.mp4")
    make_intruder(clips / "intruder.mp4", enter_frame=ENTER_FRAME, exit_frame=EXIT_FRAME)
    make_light_switch(clips / "light_switch.mp4")
    make_osd_clock(clips / "osd_clock.mp4")
    make_swaying_plant(clips / "swaying_plant.mp4")

    print(
        f"{'case':16s} {'scenario':26s} {'version':8s} {'fp frames':>9s} "
        f"{'recall':>7s} {'box IoU':>8s} {'box size':>9s}"
    )
    print("-" * 90)

    for name, scenario, truth in CASES:
        clip = clips / f"{name}.mp4"
        for label, runner in (("v1", run_legacy_frames), ("current", run_current_frames)):
            flagged, total = runner(clip)
            stats = _score(flagged, total, truth)
            recall = "-" if np.isnan(stats["recall"]) else f"{stats['recall']*100:5.1f}%"
            iou = "-" if np.isnan(stats["iou"]) else f"{stats['iou']:7.2f}"
            size = "-" if np.isnan(stats["inflation"]) else f"{stats['inflation']:8.2f}x"
            print(
                f"{name if label == 'v1' else '':16s} {scenario if label == 'v1' else '':26s} "
                f"{label:8s} {int(stats['fp_frames']):9d} {recall:>7s} {iou:>8s} {size:>9s}"
            )
        print()

    for video in args.videos:
        if not video.exists():
            print(f"{video}: not found")
            continue
        for label, runner in (("v1", run_legacy_frames), ("current", run_current_frames)):
            started = time.perf_counter()
            flagged, total = runner(video)
            elapsed = time.perf_counter() - started
            share = len(flagged) / max(1, total)
            print(
                f"{video.name if label == 'v1' else '':16s} "
                f"{'real footage' if label == 'v1' else '':26s} {label:8s} "
                f"{len(flagged):4d}/{total:<4d} frames flagged ({share*100:4.1f}%)  {elapsed:.1f}s"
            )
        print()

    shutil.rmtree(workspace, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
