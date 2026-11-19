"""GitHub release check. This is the only network call that is not a model download."""
import json
import urllib.request
from typing import Callable

from canvas.app.info import RELEASES_API

Fetcher = Callable[[str], dict]


def fetch_release(url: str) -> dict:
    """GET the Canvas releases endpoint and return the JSON object."""
    if url != RELEASES_API:
        raise ValueError("unexpected update url")
    request = urllib.request.Request(
        url,
        headers={"Accept": "application/vnd.github+json", "User-Agent": "voxprint-canvas"},
    )
    with urllib.request.urlopen(request, timeout=8) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("unexpected update payload")
    return payload


def check_for_update(current: str, fetcher: Fetcher = fetch_release) -> str | None:
    """Return a different release tag, or None when the check fails or matches."""
    try:
        payload = fetcher(RELEASES_API)
    except (OSError, ValueError, TimeoutError, json.JSONDecodeError):
        return None
    tag = str(payload.get("tag_name") or "").strip()
    if not tag:
        return None
    if tag.lstrip("vV") == current.strip().lstrip("vV"):
        return None
    return tag
