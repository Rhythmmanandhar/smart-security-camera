"""Keep OpenCV/FFmpeg's native chatter out of the application's output.

OpenCV probes several encoders when a writer is created.  On Windows the
H.264 probe prints a multi-line "Failed to load OpenH264" complaint straight
to file descriptor 2 *even when the writer then succeeds through another
encoder*, which made every run look broken.  Those messages come from C code,
so Python-level logging cannot filter them; the file descriptor has to be
redirected for the duration of the call.

Only encoder/decoder construction is wrapped.  Everything the application
itself needs to report goes through :mod:`logging` as usual, and each wrapped
call is still checked for success by its caller.
"""

from __future__ import annotations

import contextlib
import os
import sys
from typing import Iterator

_ENV_DEFAULTS = {
    "OPENCV_FFMPEG_LOGLEVEL": "-8",
    "OPENCV_LOG_LEVEL": "SILENT",
    "OPENCV_VIDEOIO_DEBUG": "0",
}


def configure_quiet_opencv() -> None:
    """Set the OpenCV logging environment.  Call before importing cv2."""
    for key, value in _ENV_DEFAULTS.items():
        os.environ.setdefault(key, value)


@contextlib.contextmanager
def suppressed_native_stderr() -> Iterator[None]:
    """Silence writes to the real stderr file descriptor inside the block.

    Skipped under pytest: its capture owns file descriptor 2, and ``dup2``
    over that descriptor closes the handle pytest later expects to close
    itself, which surfaces as "Bad file descriptor" during teardown.  The
    suppression is cosmetic, so losing it in the test run costs nothing.
    """
    if "PYTEST_CURRENT_TEST" in os.environ:
        yield
        return

    try:
        stderr_fd = sys.stderr.fileno()
    except (AttributeError, OSError, ValueError):
        # No real file descriptor (captured output, GUI host): nothing to do.
        yield
        return

    saved_fd = None
    devnull_fd = None
    try:
        sys.stderr.flush()
        saved_fd = os.dup(stderr_fd)
        devnull_fd = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull_fd, stderr_fd)
        yield
    finally:
        if saved_fd is not None:
            try:
                os.dup2(saved_fd, stderr_fd)
            finally:
                os.close(saved_fd)
        if devnull_fd is not None:
            os.close(devnull_fd)
