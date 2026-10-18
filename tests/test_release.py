"""The tag workflow publishes a pre-release with checksums. It does not embed weights."""
from pathlib import Path


def test_release_workflow_publishes_a_prerelease_from_the_build_file():
    text = Path(".github/workflows/release.yml").read_text(encoding="utf-8")
    assert 'tags:' in text
    assert '- "v*"' in text
    assert "workflow_dispatch:" in text
    assert "repository_dispatch:" in text
    assert "build-installers" in text
    assert '[ "${GITHUB_EVENT_NAME}" = "workflow_dispatch" ] || [ "${GITHUB_EVENT_NAME}" = "repository_dispatch" ]' in text
    assert '$env:GITHUB_EVENT_NAME -eq "workflow_dispatch" -or $env:GITHUB_EVENT_NAME -eq "repository_dispatch"' in text
    assert "github.event_name == 'push' && startsWith(github.ref, 'refs/tags/v')" in text
    assert 'name: SHA256SUMS' in text
    assert "--prerelease" in text
    assert "spec_from_file_location" in text
    assert "release_title()" in text
    assert "RC release title must not include a codename" in text
    assert "unexpected RC release title" in text
    assert "SHA256SUMS" in text
    assert "gh release delete" in text
    assert "--cleanup-tag" not in text
    assert "voxprint-canvas.iss" in text
    assert "installer/linux/install.sh" in text
    assert "constraints.txt" in text
    assert "constraints-cuda.txt" in text
    assert "uv.exe" in text
    notes = Path("docs/release-notes/0.1.0-rc.1.md").read_text(encoding="utf-8")
    assert notes.startswith("New:")
    assert "Fixed:" in notes


def test_installers_fetch_gpu_libraries_and_the_default_pack():
    linux = Path("installer/linux/install.sh").read_text(encoding="utf-8")
    windows = Path("installer/windows/install.ps1").read_text(encoding="utf-8")
    iss = Path("installer/windows/voxprint-canvas.iss").read_text(encoding="utf-8")
    for script in (linux, windows):
        assert "https://download.pytorch.org/whl/cu130" in script
        assert "--register-models" in script
        assert "huggingface.co" not in script
    assert "install.ps1" in iss
    assert "voxprint-canvas.cmd" in iss
    assert "staging\\*" in iss


def test_windows_ci_builds_the_installer_and_runs_install_ps1():
    text = Path(".github/workflows/tests.yml").read_text(encoding="utf-8")
    assert "windows-install:" in text
    assert "runs-on: windows-latest" in text
    assert "voxprint-canvas.iss" in text
    assert "install.ps1" in text
    assert "ISCC.exe" in text
    assert "voxprint-setup-broke.txt" in text
    assert "a broken install.ps1 left Setup at exit 0" in text
    assert 'param([string]$Scope = "All")' in text
    assert 'Remove-Item -Recurse -Force "installer\\windows\\staging\\venv"' in text
    assert "venv base interpreter is outside the install dir" in text
    assert "venv base interpreter is outside {app}\\python" in text
    assert "venv base interpreter path contains a link" in text
    assert "venv base interpreter still uses the uv alias link" in text
    assert "torch.version.cuda is not 13.0" in text
    assert "pyvenv.cfg" in text
    assert "constraints-cuda.txt" in text
    assert "git is still on PATH" in text
    iss = Path("installer/windows/voxprint-canvas.iss").read_text(encoding="utf-8")
    windows = Path("installer/windows/install.ps1").read_text(encoding="utf-8")
    assert "if ResultCode <> 0 then" in iss
    assert "ExitProcess(Code)" in iss
    assert "if Code <> 0 then" in iss
    assert "exit 1" in windows


def test_install_sources_do_not_use_git_or_an_unsafe_index_strategy():
    root = Path(__file__).resolve().parents[1]
    paths = [root / "pyproject.toml"]
    paths.extend(sorted(root.glob("constraints*.txt")))
    paths.extend(sorted(path for path in (root / "installer").rglob("*") if path.is_file()))
    paths.extend(sorted(path for path in (root / ".github" / "workflows").rglob("*") if path.is_file()))
    for path in paths:
        text = path.read_text(encoding="utf-8")
        assert "unsafe-best-match" not in text, path
        assert "git+" not in text, path
    pyproject = (root / "pyproject.toml").read_text(encoding="utf-8")
    assert 'voxprint-common[ui]==0.3.1' in pyproject
    sha = "f14036fc70db07e40e5e5376427cea28c3173a94f1d031997a3259ed0afc91fd"
    for name in ("tests.yml", "audit.yml", "docs.yml"):
        workflow = (root / ".github" / "workflows" / name).read_text(encoding="utf-8")
        assert sha in workflow
        assert "voxprint_common-0.3.1-py3-none-any.whl" in workflow
    linux = (root / "installer" / "linux" / "install.sh").read_text(encoding="utf-8")
    windows = (root / "installer" / "windows" / "install.ps1").read_text(encoding="utf-8")
    for script in (linux, windows):
        assert sha in script
        assert '--index-url' in script or '"--index-url"' in script
        assert "torch==2.11.0+cu130" in script
        assert "torchvision==0.26.0+cu130" in script
    assert '--index-url "$TORCH_INDEX"' in linux
    assert '--index-url https://pypi.org/simple' in linux
    assert '--find-links "$WHEELS"' in linux
    assert '"--index-url", $IndexUrl' in windows or "--index-url $TorchIndex" in windows
    assert "--index-url https://pypi.org/simple" in windows
    assert "--find-links $Fetch" in windows


def test_release_ships_tools_and_vm_scripts():
    text = Path(".github/workflows/release.yml").read_text(encoding="utf-8")
    assert "tools" in text
    assert "scripts/vm_test.sh" in text
    assert "scripts/vm_test.ps1" in text
    root = Path(__file__).resolve().parents[1]
    assert (root / "tools" / "vm_test.py").is_file()
    assert (root / "tools" / "__init__.py").is_file()
    assert (root / "scripts" / "vm_test.sh").is_file()
    assert (root / "scripts" / "vm_test.ps1").is_file()
