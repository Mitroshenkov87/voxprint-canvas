"""Product name and the one responsibility line shown to users.

The version string lives in ``BUILD.json`` and is read by ``canvas.app.identity``.
"""
from __future__ import annotations

from canvas.app.i18n import text
from canvas.app.identity import version

APP_NAME = "VOXPrint AI Canvas"
APP_VERSION = version()
APP_ID = "canvas"
# Windows 11 24H2.
MIN_WINDOWS_BUILD = 26100
MIN_LINUX_YEAR = 2025
MIN_COMPUTE_CAPABILITY = (8, 9)

RESPONSIBILITY = text("notice.responsibility", "en")
THIRD_PARTY_NOTICE = text("notice.third_party", "en")

ENLARGE_NAME = "VOXPrint AI Enlarge"
ENLARGE_HINT = text("enlarge.hint", "en", name=ENLARGE_NAME)
ENLARGE_RELEASES = "https://github.com/Mitroshenkov87/voxprint-enlarge/releases"

RELEASES_API = "https://api.github.com/repos/Mitroshenkov87/voxprint-canvas/releases/latest"
