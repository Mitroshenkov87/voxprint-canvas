"""Startup commands. The window itself is not opened here."""
from canvas.app.info import RELEASES_API
from canvas.app.main import main
from canvas.app.platform import OsCheck
from canvas.app.updates import check_for_update, fetch_release
from canvas.infra.gpu import GpuInfo
from canvas.infra.models_users import read_users
from canvas.models.modes import Tier
from canvas.ui.backdrop import apply_backdrop


def test_selftest_register_and_update(monkeypatch, tmp_path, capsys):
    assert main(["--selftest"]) == 0
    monkeypatch.setenv("VOXPRINT_MODELS_DIR", str(tmp_path))
    assert main(["--register-models"]) == 0
    assert read_users(tmp_path)["canvas"] is True
    assert main(["--unregister-models"]) == 0
    assert "canvas" not in read_users(tmp_path)
    monkeypatch.setattr("canvas.app.main.check_for_update", lambda current: "v9.9.9")
    assert main(["--check-update"]) == 0
    assert capsys.readouterr().out.strip() == "v9.9.9"


def test_update_check_uses_only_the_releases_endpoint():
    seen = []

    def fetcher(url: str) -> dict:
        seen.append(url)
        return {"tag_name": "v0.1.0"}

    assert check_for_update("0.1.0", fetcher) is None
    assert seen == [RELEASES_API]
    assert check_for_update("0.1.0", lambda url: {"tag_name": "v0.2.0"}) == "v0.2.0"

    def offline(url: str) -> dict:
        raise OSError("offline")

    assert check_for_update("0.1.0", offline) is None
    try:
        fetch_release("https://example.invalid/other")
    except ValueError as exc:
        assert "unexpected" in str(exc)
    else:
        raise AssertionError("expected a refused url")


def test_startup_refuses_or_launches(monkeypatch):
    monkeypatch.setattr("canvas.app.main.current_os_check", lambda: OsCheck(False, "Windows 11 24H2 or newer is required."))
    assert main([]) == 2
    monkeypatch.setattr("canvas.app.main.current_os_check", lambda: OsCheck(True, ""))
    monkeypatch.setattr("canvas.app.main.probe_gpu", lambda: None)
    assert main([]) == 2
    monkeypatch.setattr(
        "canvas.app.main.probe_gpu",
        lambda: GpuInfo("RTX 4090", 24, 20, 8, 9, "fake"),
    )
    launched = []
    monkeypatch.setattr("canvas.app.main._launch", lambda tier: launched.append(tier))
    assert main([]) == 0
    assert launched == [Tier.GB24]


def test_backdrop_is_plain_off_windows():
    assert apply_backdrop(0) == "plain"
