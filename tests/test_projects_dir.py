"""Suite rule 5: projects default to <Documents>/VOXPrint Projects/Canvas, Documents resolved per platform."""
from __future__ import annotations

from pathlib import Path

import pytest

from canvas.infra import paths
from canvas.infra.paths import documents_dir, projects_dir

RU_DOCS = "Документы"


def _user_dirs(config: Path, line: str) -> None:
    config.mkdir(parents=True, exist_ok=True)
    (config / "user-dirs.dirs").write_text(
        "# This file is written by xdg-user-dirs-update\n"
        'XDG_DESKTOP_DIR="$HOME/Рабочий стол"\n'
        f"{line}\n",
        encoding="utf-8",
    )


def test_linux_reads_the_russian_documents_folder_from_user_dirs(tmp_path: Path) -> None:
    _user_dirs(tmp_path / ".config", f'XDG_DOCUMENTS_DIR="$HOME/{RU_DOCS}"')
    assert documents_dir(platform="linux", environ={}, home=tmp_path) == tmp_path / RU_DOCS


def test_linux_keeps_spaces_and_honours_xdg_config_home(tmp_path: Path) -> None:
    config = tmp_path / "my config"
    _user_dirs(config, 'XDG_DOCUMENTS_DIR="${HOME}/Мои документы/Работа"')
    found = documents_dir(platform="linux", environ={"XDG_CONFIG_HOME": str(config)}, home=tmp_path)
    assert found == tmp_path / "Мои документы" / "Работа"


def test_the_environment_variable_wins_over_the_file(tmp_path: Path) -> None:
    _user_dirs(tmp_path / ".config", f'XDG_DOCUMENTS_DIR="$HOME/{RU_DOCS}"')
    direct = tmp_path / "Docs here"
    found = documents_dir(platform="linux", environ={"XDG_DOCUMENTS_DIR": str(direct)}, home=tmp_path)
    assert found == direct


@pytest.mark.parametrize("line", ['XDG_DOCUMENTS_DIR="$HOME/"', 'XDG_DOCUMENTS_DIR="$HOME"',
                                  'XDG_DOCUMENTS_DIR="relative/docs"', "# no documents line"])
def test_linux_falls_back_to_documents_when_unset_disabled_or_relative(tmp_path: Path, line: str) -> None:
    _user_dirs(tmp_path / ".config", line)
    assert documents_dir(platform="linux", environ={}, home=tmp_path) == tmp_path / "Documents"


def test_linux_without_user_dirs_falls_back_to_documents(tmp_path: Path) -> None:
    assert documents_dir(platform="linux", environ={}, home=tmp_path) == tmp_path / "Documents"


def test_macos_and_windows_use_the_platform_documents_folder(tmp_path: Path) -> None:
    assert documents_dir(platform="darwin", environ={}, home=tmp_path) == tmp_path / "Documents"
    profile = tmp_path / "Users" / "Алекс"
    assert documents_dir(platform="win32", environ={"USERPROFILE": str(profile)}, home=tmp_path) == (
        profile / "Documents"
    )


def test_projects_dir_is_documents_voxprint_projects_canvas(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("VOXPRINT_PROJECTS_DIR", raising=False)
    docs = tmp_path / RU_DOCS
    monkeypatch.setattr(paths, "documents_dir", lambda: docs)
    folder = projects_dir()
    assert folder == docs / "VOXPrint Projects" / "Canvas"
    assert folder.is_dir()
    picture = folder / "проба 1.webp"
    picture.write_bytes(b"RIFF")
    assert picture.read_bytes() == b"RIFF"


def test_projects_dir_override_replaces_the_suite_projects_root(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("VOXPRINT_PROJECTS_DIR", str(tmp_path / "elsewhere"))
    assert projects_dir(create=False) == tmp_path / "elsewhere" / "Canvas"
