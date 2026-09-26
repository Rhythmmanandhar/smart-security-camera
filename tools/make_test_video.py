"""Generate synthetic clips with known ground truth.

These are what the test-suite and the tuning runs use: an intruder clip where
the exact frames of the intrusion are known, plus three "nothing happened"
clips built from the noise sources that generate real-world false alarms.

    python tools/make_test_video.py --out outputs/testclips
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np

WIDTH, HEIGHT = 640, 360
FPS = 15.0


def _background(width: int = WIDTH, height: int = HEIGHT) -> np.ndarray:
    """A static "room": floor, wall, a shelf and a door."""
    frame = np.full((height, width, 3), 70, dtype=np.uint8)
    cv2.rectangle(frame, (0, 0), (width, int(height * 0.45)), (96, 92, 88), -1)
    cv2.rectangle(frame, (0, int(height * 0.45)), (width, height), (58, 58, 62), -1)
    cv2.rectangle(frame, (40, 120), (180, 240), (120, 110, 95), -1)
    cv2.rectangle(frame, (500, 90), (590, 250), (135, 125, 105), -1)
    for x in range(0, width, 80):
        cv2.line(frame, (x, int(height * 0.45)), (x, height), (52, 52, 56), 1)
    return frame


def _add_noise(frame: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    noise = rng.normal(0, sigma, frame.shape)
    return np.clip(frame.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def _writer(path: Path, fps: float = FPS, size: Tuple[int, int] = (WIDTH, HEIGHT)):
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    if not writer.isOpened():
        raise RuntimeError(f"Could not create {path}")
    return writer


def make_static(path: Path, frames: int = 150, noise: float = 3.0, seed: int = 7) -> Path:
    """An empty room: sensor noise only.  Ground truth: zero incidents."""
    rng = np.random.default_rng(seed)
    background = _background()
    writer = _writer(path)
    for _ in range(frames):
        writer.write(_add_noise(background, noise, rng))
    writer.release()
    return path


def intruder_box(index: int, enter_frame: int, exit_frame: int):
    """Ground-truth bounding box of the intruder on a given frame, or None.

    The box covers the body and the head, which is what a detector should
    return: the benchmark scores detections against exactly this rectangle.
    """
    if not enter_frame <= index < exit_frame:
        return None
    progress = (index - enter_frame) / float(exit_frame - enter_frame)
    x = int(60 + progress * (WIDTH - 180))
    y = int(190 + 30 * np.sin(progress * 6))
    # Matches the drawn pixels exactly: body rectangle plus the head circle
    # (centre x+23, y-14, radius 16).
    return (x, y - 30, x + 46, y + 104)


def make_intruder(
    path: Path,
    frames: int = 180,
    enter_frame: int = 60,
    exit_frame: int = 130,
    noise: float = 3.0,
    seed: int = 11,
) -> Tuple[Path, Tuple[int, int]]:
    """A person-sized object crossing the room between two known frames."""
    rng = np.random.default_rng(seed)
    background = _background()
    writer = _writer(path)

    for index in range(frames):
        frame = _add_noise(background, noise, rng)
        if enter_frame <= index < exit_frame:
            progress = (index - enter_frame) / float(exit_frame - enter_frame)
            x = int(60 + progress * (WIDTH - 180))
            y = int(190 + 30 * np.sin(progress * 6))
            cv2.rectangle(frame, (x, y), (x + 46, y + 104), (30, 30, 160), -1)
            cv2.circle(frame, (x + 23, y - 14), 16, (40, 40, 150), -1)
        writer.write(frame)
    writer.release()
    return path, (enter_frame, exit_frame)


def make_light_switch(path: Path, frames: int = 150, switch_frame: int = 70, seed: int = 13) -> Path:
    """The lights come on mid-clip.  Ground truth: zero incidents."""
    rng = np.random.default_rng(seed)
    background = _background()
    writer = _writer(path)
    for index in range(frames):
        frame = background.astype(np.float32)
        if index >= switch_frame:
            frame = np.clip(frame * 1.8 + 30, 0, 255)
        writer.write(_add_noise(frame.astype(np.uint8), 3.0, rng))
    writer.release()
    return path


def make_osd_clock(path: Path, frames: int = 150, seed: int = 17) -> Path:
    """A burnt-in timestamp ticking in the corner.  Ground truth: zero."""
    rng = np.random.default_rng(seed)
    background = _background()
    writer = _writer(path)
    for index in range(frames):
        frame = _add_noise(background, 3.0, rng)
        seconds = index // 3
        cv2.putText(
            frame,
            f"2026-01-08 10:{seconds // 60:02d}:{seconds % 60:02d}",
            (WIDTH - 250, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )
        writer.write(frame)
    writer.release()
    return path


def make_swaying_plant(path: Path, frames: int = 150, seed: int = 19) -> Path:
    """Foliage oscillating on the spot.  Ground truth: zero incidents."""
    rng = np.random.default_rng(seed)
    background = _background()
    writer = _writer(path)
    for index in range(frames):
        frame = _add_noise(background, 3.0, rng)
        offset = int(6 * np.sin(index * 0.6))
        cv2.ellipse(frame, (110 + offset, 150), (26, 44), 0, 0, 360, (40, 110, 50), -1)
        writer.write(frame)
    writer.release()
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("outputs/testclips"))
    args = parser.parse_args()

    out = args.out
    made = [
        ("static_empty", make_static(out / "static_empty.mp4")),
        ("intruder", make_intruder(out / "intruder.mp4")[0]),
        ("light_switch", make_light_switch(out / "light_switch.mp4")),
        ("osd_clock", make_osd_clock(out / "osd_clock.mp4")),
        ("swaying_plant", make_swaying_plant(out / "swaying_plant.mp4")),
    ]
    for name, path in made:
        print(f"{name:16s} -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
