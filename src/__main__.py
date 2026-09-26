"""Allow `python -m src` as an alternative to `python detect.py`."""

from .nativelog import configure_quiet_opencv

configure_quiet_opencv()

from .security_camera import main  # noqa: E402

raise SystemExit(main())
