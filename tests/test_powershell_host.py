"""The Windows install path prefers PowerShell 7 and keeps Windows PowerShell 5.1 as a fallback."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ISS = ROOT / "installer" / "windows" / "voxprint-canvas.iss"
CMD = ROOT / "installer" / "windows" / "voxprint-canvas.cmd"
LEGACY = r"{sys}\WindowsPowerShell\v1.0\powershell.exe"
LINK = "https://aka.ms/powershell-release?tag=stable"


def _code(iss: str) -> str:
    return iss.split("[Code]", 1)[1]


def test_setup_tries_pwsh_first_and_keeps_51_only_as_fallback():
    iss = ISS.read_text(encoding="utf-8")
    code = _code(iss)
    assert r"SOFTWARE\Microsoft\PowerShellCore\InstalledVersions" in code
    assert "'InstallLocation'" in code
    assert r"{commonpf64}\PowerShell\7\pwsh.exe" in code
    # The 5.1 path is named once, in the fallback helper, and only used when pwsh is missing.
    assert iss.count("powershell.exe") == 1
    assert LEGACY in code
    prepare = code.split("function PrepareToInstall", 1)[1].split("\nend;", 1)[0]
    assert prepare.index("ScriptHost := FindPwsh;") < prepare.index("ScriptHost := WindowsPowerShell;")
    assert "if ScriptHost = '' then" in prepare
    assert "if not WizardSilent then" in prepare
    run_setup = code.split("function RunSetup", 1)[1].split("procedure CurStepChanged", 1)[0]
    assert run_setup.index("FindPwsh") < run_setup.index("WindowsPowerShell")
    assert "Started := Exec(\n      ScriptHost," in run_setup
    assert "ExecAsOriginalUser(\n      ScriptHost," in run_setup
    assert "powershell.exe" not in run_setup


def test_setup_installs_only_the_pinned_verified_microsoft_package():
    code = _code(ISS.read_text(encoding="utf-8"))
    url = re.search(r"PwshMsiUrl = '([^']+)'", code).group(1)
    assert url.startswith("https://github.com/PowerShell/PowerShell/releases/download/v7.")
    assert url.endswith("-win-x64.msi")
    sha = re.search(r"PwshMsiSha256 = '([0-9a-f]+)'", code).group(1)
    assert len(sha) == 64
    assert "Page.Add(PwshMsiUrl, PwshMsiName, PwshMsiSha256);" in code
    assert "PwshPublisher = 'Microsoft Corporation';" in code
    install = code.split("function InstallPwsh", 1)[1].split("function PrepareToInstall", 1)[0]
    assert install.index("Page.Download;") < install.index("vxsigcheck.exe") < install.index("msiexec.exe")
    assert "/qn /norestart" in install
    assert (ROOT / "installer" / "windows" / "vxsigcheck.cs").is_file()
    assert "vxsigcheck.cs" in ISS.read_text(encoding="utf-8")


def test_setup_messages_exist_in_english():
    iss = ISS.read_text(encoding="utf-8")
    assert "russian" not in iss.casefold()
    assert "ShowLanguageDialog=no" in iss
    expected = {
        "english.PwshAsk": "PowerShell 7 is not installed. Install PowerShell 7 (Microsoft) now? Recommended.",
        "english.PwshInstalling": "Installing PowerShell 7 (Microsoft)",
    }
    for key, text in expected.items():
        assert f"{key}={text}" in iss
    for lang in ("english",):
        line = next(item for item in iss.splitlines() if item.startswith(f"{lang}.PwshFallback="))
        assert LINK in line
        assert f"{lang}.PwshInstallingDesc=" in iss


def test_cmd_tries_pwsh_first():
    cmd = CMD.read_text(encoding="utf-8")
    assert cmd.index("where pwsh.exe") < cmd.index("%VOXPRINT_PS% -NoProfile")
    assert 'set "VOXPRINT_PS=powershell"' in cmd
    assert 'set "VOXPRINT_PS=pwsh"' in cmd
    assert "if errorlevel 1 exit /b 1" in cmd


def test_ci_runs_pwsh_and_still_covers_the_51_fallback():
    workflow = (ROOT / ".github" / "workflows" / "tests.yml").read_text(encoding="utf-8")
    assert "shell: powershell" not in workflow.replace("shell: pwsh", "")
    assert "& pwsh -NoProfile -ExecutionPolicy Bypass -File $script" in workflow
    assert "& pwsh -NoProfile -ExecutionPolicy Bypass -File tools\\smoke_16gb.ps1" in workflow
    assert "& powershell.exe -NoProfile -ExecutionPolicy Bypass -File tools\\smoke_16gb.ps1" in workflow
    assert "Setup script host:" in workflow
    for name in ("install.ps1", "tools/smoke_16gb.ps1"):
        path = ROOT / "installer" / "windows" / name if name == "install.ps1" else ROOT / name
        assert "#Requires" not in path.read_text(encoding="utf-8")


def test_setup_scripts_run_without_a_console_window():
    iss = (ROOT / "installer" / "windows" / "voxprint-canvas.iss").read_text(encoding="utf-8")
    assert "SW_SHOW" not in iss
    assert "english.SetupScripts=" in iss


def test_setup_logs_each_stage_result_and_the_script_failure():
    iss = ISS.read_text(encoding="utf-8")
    run_setup = _code(iss).split("function RunSetup", 1)[1].split("procedure CurStepChanged", 1)[0]
    assert "Setup stage %s could not start" in run_setup
    assert "Setup stage %s finished with result code %d." in run_setup
    assert run_setup.index("finished with result code") < run_setup.index("if ResultCode <> 0 then")
    step = _code(iss).split("procedure CurStepChanged", 1)[1].split("procedure CurUninstallStepChanged", 1)[0]
    for stage in ("Machine", "User", "Cleanup"):
        assert f"RunSetup('{stage}', " in step
    assert step.index("Setup exits with %d.") < step.index("ExitProcess(Code)")
    script = (ROOT / "installer" / "windows" / "install.ps1").read_text(encoding="utf-8")
    assert "function Write-SetupFailure" in script
    assert "ScriptLineNumber" in script and "ScriptStackTrace" in script
    assert "Script host: " in script
    # The reason is written before the transcript closes, in both logged scopes.
    for name in ("function Invoke-Machine", "function Invoke-User"):
        body = script.split(name, 1)[1].split("\nfunction ", 1)[0]
        assert "Write-SetupHost" in body
        assert body.index("Write-SetupFailure $_") < body.index("Stop-SetupLog")
    assert "Invoke-WebRequest -UseBasicParsing" in script
