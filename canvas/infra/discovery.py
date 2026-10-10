"""How Canvas finds another Voxprint app.

This is Canvas's copy of the suite discovery rules. A shared package replaces
it once that repository exists. Enlarge messages stay in ``canvas.infra.enlarge``.
"""
from __future__ import annotations

import ctypes
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from ctypes import wintypes
from dataclasses import dataclass
from pathlib import Path
from typing import Any

if sys.platform == "win32":
    import winreg
else:
    winreg = None

_CACHE_SECONDS = 24 * 60 * 60


@dataclass(frozen=True)
class AppInstall:
    command: list[str]
    version: str
    ipc_base: str


def pipe_name(ipc_base: str, sid: str) -> str:
    """Qt server name. Windows exposes it as ``\\\\.\\pipe\\<name>``."""
    digest = hashlib.sha256(sid.encode("utf-8")).hexdigest()[:16]
    return f"{ipc_base}-{digest}"


def socket_path(app: str, ipc_base: str) -> str:
    """Standard socket for ``app``."""
    if sys.platform == "win32":
        return pipe_name(ipc_base, _windows_user_sid())
    runtime = os.environ.get("XDG_RUNTIME_DIR", "").strip()
    if not runtime:
        raise OSError("XDG_RUNTIME_DIR is not set")
    return str(Path(runtime) / "voxprint" / f"{app}.sock")


def find_app(pascal: str, kebab: str) -> AppInstall | None:
    """Registry or desktop entry, then PATH. The executable must exist."""
    override = os.environ.get("CANVAS_ENLARGE_EXE", "").strip() if kebab == "enlarge" else ""
    if override:
        parts = _split_command(override)
        if not parts:
            return None
        head = parts[0]
        if not Path(head).is_file() and shutil.which(head) is None:
            return None
        version = os.environ.get("CANVAS_ENLARGE_VERSION", "").strip()
        return AppInstall(parts, version, f"voxprint-{kebab}")
    if sys.platform == "win32":
        found = _from_windows_registry(pascal, kebab)
    else:
        found = _from_desktop_dirs(pascal, kebab)
    if found is not None:
        return found
    program = f"voxprint-{kebab}.exe" if sys.platform == "win32" else f"voxprint-{kebab}"
    which = shutil.which(program)
    if which is None and sys.platform == "win32":
        which = shutil.which(f"voxprint-{kebab}")
    if which is None:
        return None
    return AppInstall([which], "", f"voxprint-{kebab}")


def probe_capabilities(install: AppInstall, app: str) -> dict[str, Any] | None:
    """``--capabilities --json``, cached for a day and refreshed when Version changes."""
    cached = _load_cached(app, install.version)
    if cached is not None:
        return cached
    try:
        proc = subprocess.run(
            [*install.command, "--capabilities", "--json"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0 or not proc.stdout.strip():
        return None
    try:
        data = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    version = install.version or data.get("version")
    _store_cached(app, version if isinstance(version, str) else "", data)
    return data


def _cache_file(app: str) -> Path:
    override = os.environ.get("CANVAS_CACHE", "").strip()
    if override:
        root = Path(override)
    else:
        xdg = os.environ.get("XDG_CACHE_HOME", "").strip()
        root = Path(xdg) if xdg else Path.home() / ".cache"
    return root / "voxprint" / f"{app}-capabilities.json"


def _load_cached(app: str, registered_version: str) -> dict[str, Any] | None:
    path = _cache_file(app)
    if not path.is_file():
        return None
    try:
        stored = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(stored, dict):
        return None
    fetched = stored.get("fetched_at")
    body = stored.get("body")
    if not isinstance(fetched, int | float) or not isinstance(body, dict):
        return None
    if time.time() - float(fetched) > _CACHE_SECONDS:
        return None
    if registered_version and stored.get("version") != registered_version:
        return None
    return body


def _store_cached(app: str, version: str, body: dict[str, Any]) -> None:
    path = _cache_file(app)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"fetched_at": time.time(), "version": version, "body": body}
    path.write_text(json.dumps(payload), encoding="utf-8")


def _from_desktop_dirs(pascal: str, kebab: str) -> AppInstall | None:
    roots: list[Path] = []
    data_home = os.environ.get("XDG_DATA_HOME", "").strip()
    roots.append(Path(data_home) if data_home else Path.home() / ".local" / "share")
    data_dirs = os.environ.get("XDG_DATA_DIRS", "/usr/local/share:/usr/share")
    roots.extend(Path(item) for item in data_dirs.split(":") if item)
    name = f"ai.voxprint.{pascal}.desktop"
    for root in roots:
        path = root / "applications" / name
        if not path.is_file():
            continue
        parsed = _desktop_fields(path.read_text(encoding="utf-8"))
        if parsed is None:
            continue
        command, version, ipc = parsed
        if shutil.which(command[0]) is None and not Path(command[0]).is_file():
            continue
        return AppInstall(command, version, ipc or f"voxprint-{kebab}")
    return None


def _split_command(text: str) -> list[str]:
    """Split an executable command. On Windows a backslash is a path separator."""
    if sys.platform == "win32":
        return _split_windows_command(text)
    return shlex.split(text)


def _split_windows_command(text: str) -> list[str]:
    args: list[str] = []
    buf: list[str] = []
    quoted = False
    for char in text:
        if char == '"':
            quoted = not quoted
            continue
        if char in " \t" and not quoted:
            if buf:
                args.append("".join(buf))
                buf = []
            continue
        buf.append(char)
    if buf:
        args.append("".join(buf))
    return args


def _desktop_fields(text: str) -> tuple[list[str], str, str] | None:
    protocol = ""
    version = ""
    ipc = ""
    exec_line = ""
    for line in text.splitlines():
        if line.startswith("X-Voxprint-Protocol="):
            protocol = line.split("=", 1)[1].strip()
        elif line.startswith("X-Voxprint-Version="):
            version = line.split("=", 1)[1].strip()
        elif line.startswith("X-Voxprint-Ipc="):
            ipc = line.split("=", 1)[1].strip()
        elif line.startswith("Exec="):
            exec_line = line.split("=", 1)[1].strip()
    if protocol != "1" or not exec_line:
        return None
    command = [part for part in shlex.split(exec_line) if not part.startswith("%")]
    if not command:
        return None
    return command, version, ipc


def _from_windows_registry(pascal: str, kebab: str) -> AppInstall | None:
    if sys.platform != "win32" or winreg is None:
        return None
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, rf"Software\Voxprint\{pascal}") as key:
            install_path, _ = winreg.QueryValueEx(key, "InstallPath")
            exe, _ = winreg.QueryValueEx(key, "Exe")
            version, _ = winreg.QueryValueEx(key, "Version")
            ipc_name, _ = winreg.QueryValueEx(key, "IpcName")
            protocol, _ = winreg.QueryValueEx(key, "ProtocolVersion")
    except OSError:
        install_path = exe = version = ipc_name = ""
        protocol = 0
    if protocol in (1, "1") and exe:
        path = Path(str(exe))
        if not path.is_absolute() and install_path:
            path = Path(str(install_path)) / path
        resolved = str(path) if path.is_file() else shutil.which(str(exe))
        if resolved:
            return AppInstall([resolved], str(version), str(ipc_name) or f"voxprint-{kebab}")
    try:
        app_paths = rf"Software\Microsoft\Windows\CurrentVersion\App Paths\voxprint-{kebab}.exe"
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, app_paths) as key:
            exe, _ = winreg.QueryValueEx(key, "")
    except OSError:
        return None
    resolved = shutil.which(str(exe)) or (str(exe) if Path(str(exe)).is_file() else "")
    if not resolved:
        return None
    return AppInstall([resolved], "", f"voxprint-{kebab}")


def _windows_user_sid() -> str:
    """Current user SID. The pipe name is the first 16 hex chars of its SHA-256."""
    if sys.platform != "win32":
        return ""
    advapi32 = ctypes.WinDLL("advapi32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    advapi32.ConvertSidToStringSidW.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_wchar_p)]
    advapi32.ConvertSidToStringSidW.restype = wintypes.BOOL
    kernel32.LocalFree.argtypes = [ctypes.c_void_p]
    kernel32.LocalFree.restype = ctypes.c_void_p

    class _SidAndAttributes(ctypes.Structure):
        _fields_ = [("Sid", ctypes.c_void_p), ("Attributes", wintypes.DWORD)]

    class _TokenUser(ctypes.Structure):
        _fields_ = [("User", _SidAndAttributes)]

    token = wintypes.HANDLE()
    fallback = os.environ.get("USERNAME", "user")
    if not advapi32.OpenProcessToken(kernel32.GetCurrentProcess(), 0x0008, ctypes.byref(token)):
        return fallback
    try:
        length = wintypes.DWORD()
        advapi32.GetTokenInformation(token, 1, None, 0, ctypes.byref(length))
        buf = ctypes.create_string_buffer(length.value)
        if not advapi32.GetTokenInformation(token, 1, buf, length, ctypes.byref(length)):
            return fallback
        user = ctypes.cast(buf, ctypes.POINTER(_TokenUser)).contents
        sid = ctypes.c_wchar_p()
        if not advapi32.ConvertSidToStringSidW(user.User.Sid, ctypes.byref(sid)):
            return fallback
        try:
            return sid.value or fallback
        finally:
            kernel32.LocalFree(sid)
    finally:
        kernel32.CloseHandle(token)
