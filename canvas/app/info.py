"""Product name and the one responsibility line shown to users.

The version string lives in ``BUILD.json`` and is read by ``canvas.app.identity``.
"""
from __future__ import annotations

from canvas.app.identity import version

APP_NAME = "Voxprint AI Canvas"
APP_VERSION = version()
APP_ID = "canvas"
# Windows 11 24H2.
MIN_WINDOWS_BUILD = 26100
MIN_LINUX_YEAR = 2025
MIN_COMPUTE_CAPABILITY = (8, 9)

RESPONSIBILITY = (
    "Do not use the app to create content that is illegal or socially unacceptable; "
    "you are responsible for what you create."
)

THIRD_PARTY_NOTICE = (
    "Third-party weights are provided by their authors, who are responsible for the licence and training data. "
    "Voxprint does not host or redistribute them."
)

ENLARGE_HINT = "Install Voxprint AI Enlarge to use this."
ENLARGE_RELEASES = "https://github.com/Mitroshenkov87/voxprint-enlarge/releases"

RELEASES_API = "https://api.github.com/repos/Mitroshenkov87/voxprint-canvas/releases/latest"
