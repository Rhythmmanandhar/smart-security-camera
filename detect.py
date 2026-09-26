#!/usr/bin/env python
"""Command-line entry point.

    python detect.py --input cctv_video_1.mp4
    python detect.py --live --preview
    python detect.py --list-cameras
"""

from src.nativelog import configure_quiet_opencv

configure_quiet_opencv()  # must precede the first OpenCV import

from src.security_camera import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
