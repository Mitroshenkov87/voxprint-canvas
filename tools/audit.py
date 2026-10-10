"""One-shot code audit for Voxprint AI Canvas.

Adapted from Voxprint AI Audiobook Builder ``tools/audit.py``
(Apache-2.0, Copyright Aleksandr Mitroshenkov).

Runs ruff, mypy, bandit, pip-audit, vulture, radon and gitleaks. Prints one summary (and, on GitHub Actions, writes
that same summary to the job summary) and the full text to ``audit-report.txt``.

    python -m pip install -r requirements-audit.txt
    python tools/audit.py --python <app venv python> [--gitleaks-mode diff|full] [--gitleaks-range A..B]

``--python`` is the interpreter that has the application's dependencies installed (CPU PyTorch in CI). pip-audit
checks that environment. The other tools run with the interpreter that launches this script.

Exit code 1 when a blocking finding exists: a ruff error, a mypy error, a bandit issue of high severity, a gitleaks
finding, or a pip-audit advisory of high or critical severity that is not listed in ``tools/audit-allowlist.toml``.
An advisory whose severity cannot be read is treated as high, so a parser miss cannot hide it. Vulture and radon
are report-only and never change the exit code. A blocking tool that is not installed also fails the run.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import tomllib
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parent.parent
REPORT = ROOT / "audit-report.txt"
ALLOWLIST = ROOT / "tools" / "audit-allowlist.toml"
TARGETS = ["canvas", "tools", "main.py"]
BLOCKING_SEVERITY = {"high", "critical", "unknown"}
_CVSS_V3 = {
    "AV": {"N": 0.85, "A": 0.62, "L": 0.55, "P": 0.2},
    "AC": {"L": 0.77, "H": 0.44},
    "UI": {"N": 0.85, "R": 0.62},
    "C": {"N": 0.0, "L": 0.22, "H": 0.56},
    "I": {"N": 0.0, "L": 0.22, "H": 0.56},
    "A": {"N": 0.0, "L": 0.22, "H": 0.56},
}
_FINDING_LINE = re.compile(r":\d+:\d+: ")


@dataclass(frozen=True)
class Advisory:
    id: str
    reason: str
    package: str = ""


@dataclass
class Check:
    name: str
    summary: str
    detail: str
    blocks: bool


def _run(cmd: Sequence[str], env: Optional[Mapping[str, str]] = None, timeout: int = 1200) -> Tuple[int, str, str]:
    merged = dict(os.environ)
    if env:
        merged.update(env)
    merged["NO_COLOR"] = "1"
    merged.pop("FORCE_COLOR", None)
    try:
        proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, env=merged, timeout=timeout)
    except (OSError, subprocess.SubprocessError) as exc:
        return 127, "", str(exc)
    return proc.returncode, proc.stdout, proc.stderr


def _missing(err: str) -> bool:
    text = err.lower()
    return "no module named" in text or "no such file" in text or "not found" in text


def _rel(path: str) -> str:
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except ValueError:
        return path


def load_allowlist(path: Path = ALLOWLIST) -> List[Advisory]:
    """Read accepted advisories. A missing file, a bad file, or an entry without a reason is an error."""
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"cannot read {path}: {exc}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise ValueError(f"cannot parse {path}: {exc}") from exc
    rows = raw.get("advisory", [])
    if not isinstance(rows, list):
        raise ValueError(f"{path}: 'advisory' must be a list of tables")
    out: List[Advisory] = []
    for i, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            raise ValueError(f"{path}: advisory {i} is not a table")
        ident = str(row.get("id") or "").strip()
        reason = str(row.get("reason") or "").strip()
        package = str(row.get("package") or "").strip()
        if not ident or not reason:
            raise ValueError(f"{path}: advisory {i} needs an id and a reason")
        out.append(Advisory(ident, reason, package))
    return out


def _norm_pkg(name: str) -> str:
    return name.lower().replace("_", "-")


def allowlist_reason(adv: Advisory, package: str, ids: Sequence[str]) -> str:
    """Return the reason when ``adv`` covers this package and one of ``ids``, else ''."""
    if adv.package and _norm_pkg(adv.package) != _norm_pkg(package):
        return ""
    wanted = adv.id.lower()
    if any(item.lower() == wanted for item in ids):
        return adv.reason
    return ""


def _roundup(score: float) -> float:
    # CVSS v3 rounds up to one decimal place.
    return math.ceil(score * 10.0 - 1e-9) / 10.0


def cvss_v3_base(vector: str) -> Optional[float]:
    """Numeric base score of a CVSS v3.0 or v3.1 vector, or None when the vector is incomplete."""
    parts: Dict[str, str] = {}
    for piece in vector.strip().split("/"):
        if ":" not in piece:
            continue
        key, value = piece.split(":", 1)
        parts[key.upper()] = value.upper()
    needed = ("AV", "AC", "PR", "UI", "S", "C", "I", "A")
    if any(key not in parts for key in needed):
        return None
    try:
        av = _CVSS_V3["AV"][parts["AV"]]
        ac = _CVSS_V3["AC"][parts["AC"]]
        ui = _CVSS_V3["UI"][parts["UI"]]
        conf = _CVSS_V3["C"][parts["C"]]
        integ = _CVSS_V3["I"][parts["I"]]
        avail = _CVSS_V3["A"][parts["A"]]
    except KeyError:
        return None
    scope_changed = parts["S"] == "C"
    if parts["S"] not in ("C", "U"):
        return None
    pr_table = {"N": 0.85, "L": 0.68 if scope_changed else 0.62, "H": 0.50 if scope_changed else 0.27}
    if parts["PR"] not in pr_table:
        return None
    pr = pr_table[parts["PR"]]
    impact_sub = 1.0 - (1.0 - conf) * (1.0 - integ) * (1.0 - avail)
    if scope_changed:
        impact = 7.52 * (impact_sub - 0.029) - 3.25 * (impact_sub - 0.02) ** 15
    else:
        impact = 6.42 * impact_sub
    if impact <= 0:
        return 0.0
    exploitability = 8.22 * av * ac * pr * ui
    if scope_changed:
        return _roundup(min(1.08 * (impact + exploitability), 10.0))
    return _roundup(min(impact + exploitability, 10.0))


def score_band(score: float) -> str:
    if score >= 9.0:
        return "critical"
    if score >= 7.0:
        return "high"
    if score >= 4.0:
        return "medium"
    return "low"


def severity_from_osv(payload: Mapping[str, Any]) -> str:
    """Map one OSV document to critical/high/medium/low. A reviewed label wins over a raw CVSS vector."""
    return _reviewed_label(payload) or _cvss_label(payload)


def _reviewed_label(payload: Mapping[str, Any]) -> str:
    specific = payload.get("database_specific")
    if not isinstance(specific, dict):
        return ""
    label = str(specific.get("severity") or "").strip().lower()
    if label == "moderate":
        return "medium"
    if label in ("critical", "high", "medium", "low"):
        return label
    return ""


def _cvss_label(payload: Mapping[str, Any]) -> str:
    rows = payload.get("severity")
    if not isinstance(rows, list):
        return ""
    for row in rows:
        if not isinstance(row, dict):
            continue
        raw = str(row.get("score") or "").strip()
        if not raw:
            continue
        try:
            return score_band(float(raw))
        except ValueError:
            pass
        if raw.upper().startswith("CVSS:3"):
            score = cvss_v3_base(raw)
            if score is not None:
                return score_band(score)
    return ""


def lookup_severity(ids: Sequence[str], fetch: Callable[[str], Optional[Mapping[str, Any]]]) -> str:
    """Severity used by the release gate.

    A GitHub advisory label wins (``GHSA``). Another database's label is used only when there is no GHSA.
    Otherwise the CVSS v3 band is used. ``unknown`` when nothing can be read; callers treat that as blocking.
    """
    rank = {"low": 0, "medium": 1, "high": 2, "critical": 3}
    ghsa: List[str] = []
    other: List[str] = []
    scores: List[str] = []
    for ident in ids:
        if not ident:
            continue
        payload = fetch(ident)
        if not payload:
            continue
        reviewed = _reviewed_label(payload)
        if reviewed and ident.upper().startswith("GHSA-"):
            ghsa.append(reviewed)
        elif reviewed:
            other.append(reviewed)
        score = _cvss_label(payload)
        if score:
            scores.append(score)
    chosen = ghsa or other or scores
    if not chosen:
        return "unknown"
    return max(chosen, key=lambda label: rank[label])


def fetch_osv(ident: str, timeout: float = 20.0) -> Optional[Mapping[str, Any]]:
    url = "https://api.osv.dev/v1/vulns/" + urllib.parse.quote(ident)
    req = urllib.request.Request(url, headers={"User-Agent": "voxprint-audiobook-audit", "Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 - HTTPS to the OSV API only
            data = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _vuln_ids(vuln: Mapping[str, Any]) -> List[str]:
    ids = [str(vuln.get("id") or "")]
    aliases = vuln.get("aliases") or []
    if isinstance(aliases, list):
        ids.extend(str(item) for item in aliases)
    return [item for item in ids if item]


@dataclass
class PipFinding:
    package: str
    version: str
    ident: str
    severity: str
    fix: str
    accepted: str
    ids: Tuple[str, ...]

    @property
    def blocks(self) -> bool:
        return self.severity in BLOCKING_SEVERITY and not self.accepted


def classify_pip(payload: Mapping[str, Any], allow: Sequence[Advisory],
                 fetch: Callable[[str], Optional[Mapping[str, Any]]]) -> Tuple[List[PipFinding], List[str]]:
    """Split a pip-audit JSON document into findings and skip notes."""
    deps = payload.get("dependencies")
    if not isinstance(deps, list):
        raise ValueError("pip-audit JSON has no dependencies list")
    findings: List[PipFinding] = []
    skipped: List[str] = []
    cache: Dict[str, Optional[Mapping[str, Any]]] = {}

    def cached(ident: str) -> Optional[Mapping[str, Any]]:
        if ident not in cache:
            cache[ident] = fetch(ident)
        return cache[ident]

    for dep in deps:
        if not isinstance(dep, dict):
            continue
        name = str(dep.get("name") or "")
        if dep.get("skip_reason"):
            skipped.append(f"{name}: {dep.get('skip_reason')}")
            continue
        version = str(dep.get("version") or "")
        vulns = dep.get("vulns") or []
        if not isinstance(vulns, list):
            continue
        for vuln in vulns:
            if not isinstance(vuln, dict):
                continue
            ids = _vuln_ids(vuln)
            reason = ""
            for adv in allow:
                reason = allowlist_reason(adv, name, ids)
                if reason:
                    break
            fixes = vuln.get("fix_versions") or []
            fix = ",".join(str(item) for item in fixes) if isinstance(fixes, list) and fixes else "-"
            findings.append(PipFinding(
                package=name, version=version, ident=ids[0] if ids else "?",
                severity=lookup_severity(ids, cached), fix=fix, accepted=reason, ids=tuple(ids),
            ))
    unique: List[PipFinding] = []
    for item in findings:
        item_ids = {ident.lower() for ident in item.ids}
        duplicate = False
        for prev in unique:
            prev_ids = {ident.lower() for ident in prev.ids}
            if _norm_pkg(item.package) != _norm_pkg(prev.package) or not item_ids & prev_ids:
                continue
            # pip-audit sometimes emits the same advisory twice, and only one copy lists a fix.
            if prev.fix == "-" and item.fix != "-":
                prev.fix = item.fix
            duplicate = True
            break
        if not duplicate:
            unique.append(item)
    return unique, skipped


def ruff_check(py: str) -> Check:
    # Real errors only: pyflakes (F, including undefined names) and syntax (E9). Style (E501) and pyupgrade (UP) are out.
    code, out, err = _run([py, "-m", "ruff", "check", "--select", "F,E9", "--ignore", "E501,UP",
                           "--output-format", "concise", "--no-cache", *TARGETS])
    if code == 127 or _missing(err):
        return Check("ruff (F, E9)", "not installed", err.strip(), True)
    if code not in (0, 1):
        return Check("ruff (F, E9)", "failed to run", (out + err).strip(), True)
    lines = [ln for ln in out.splitlines() if _FINDING_LINE.search(ln)]
    return Check("ruff (F, E9)", f"{len(lines)} error(s)", out.strip(), bool(lines))


def mypy_check(py: str) -> Check:
    cache = os.path.join(tempfile.gettempdir(), "vx-mypy")
    code, out, err = _run([py, "-m", "mypy", "--config-file", str(ROOT / "mypy.ini"), "--no-error-summary",
                           "--hide-error-context", "--no-color-output", "--cache-dir", cache, *TARGETS], timeout=1200)
    if code == 127 or _missing(err):
        return Check("mypy (non-strict)", "not installed", err.strip(), True)
    if code not in (0, 1):
        return Check("mypy (non-strict)", "failed to run", (out + err).strip(), True)
    lines = [ln for ln in out.splitlines() if ": error:" in ln]
    return Check("mypy (non-strict)", f"{len(lines)} error(s)", out.strip(), bool(lines))


def bandit_check(py: str) -> Check:
    # -ll: medium and high (low is the usual subprocess-import noise). Only HIGH blocks.
    code, out, err = _run([py, "-m", "bandit", "-r", *TARGETS, "-q", "-f", "json", "-ll", "-ii", "-x", "tests"])
    if code == 127 or _missing(err):
        return Check("bandit (medium+)", "not installed", err.strip(), True)
    try:
        results = json.loads(out or "{}").get("results", [])
    except ValueError:
        return Check("bandit (medium+)", "failed to run", (out + err).strip(), True)
    if not isinstance(results, list):
        return Check("bandit (medium+)", "failed to run", out.strip(), True)
    high = [row for row in results if str(row.get("issue_severity")) == "HIGH"]
    text = "\n".join(
        f"{_rel(str(row.get('filename')))}:{row.get('line_number')}: {row.get('issue_severity')}/"
        f"{row.get('issue_confidence')} {row.get('test_id')} {row.get('issue_text')}"
        for row in results if isinstance(row, dict)
    )
    summary = f"{len(results)} issue(s), {len(high)} high"
    return Check("bandit (medium+)", summary, text, bool(high))


def _pypi_version(version: str) -> str:
    """Drop a local tag (``2.11.0+cpu``) so pip-audit can look the release up on PyPI."""
    return version.split("+", 1)[0]


def _freeze(app_py: str) -> Tuple[Optional[str], str]:
    code, out, err = _run([app_py, "-c",
                           "import importlib.metadata as meta\n"
                           "seen=set()\n"
                           "for dist in meta.distributions():\n"
                           "    name=(dist.metadata['Name'] or '').lower()\n"
                           "    if name and name not in seen:\n"
                           "        seen.add(name)\n"
                           "        print(f'{name}=={dist.version}')\n"])
    if code != 0:
        return None, err.strip() or out.strip()
    lines = []
    for line in out.splitlines():
        if "==" not in line:
            continue
        name, version = line.split("==", 1)
        lines.append(f"{name}=={_pypi_version(version)}")
    return "\n".join(lines) + "\n", ""


def pip_audit_check(py: str, app_py: str, allow: Sequence[Advisory],
                    fetch: Callable[[str], Optional[Mapping[str, Any]]] = fetch_osv) -> Check:
    frozen, err = _freeze(app_py)
    if frozen is None:
        return Check("pip-audit", "failed to list packages", err, True)
    handle = tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8")
    try:
        handle.write(frozen)
        handle.close()
        code, out, err = _run([py, "-m", "pip_audit", "-r", handle.name, "--no-deps", "--disable-pip",
                               "--format", "json", "--aliases", "on", "--desc", "off",
                               "--progress-spinner", "off"], timeout=1200)
    finally:
        os.unlink(handle.name)
    if code == 127 or _missing(err):
        return Check("pip-audit", "not installed", err.strip(), True)
    try:
        payload = json.loads(out or "{}")
    except ValueError:
        return Check("pip-audit", "failed to run", (out + err).strip(), True)
    if not isinstance(payload, dict):
        return Check("pip-audit", "failed to run", out.strip(), True)
    try:
        findings, skipped = classify_pip(payload, allow, fetch)
    except ValueError as exc:
        return Check("pip-audit", "failed to run", str(exc), True)
    blocking = [item for item in findings if item.blocks]
    accepted = [item for item in findings if item.accepted and item.severity in BLOCKING_SEVERITY]
    lines = []
    for item in findings:
        mark = "ALLOW" if item.accepted else ("BLOCK" if item.blocks else "note")
        extra = f" ({item.accepted})" if item.accepted else ""
        aliases = ", ".join(item.ids[1:]) if len(item.ids) > 1 else ""
        alias_txt = f" aliases: {aliases}" if aliases else ""
        lines.append(f"{mark} {item.package}=={item.version}: {item.ident} {item.severity} "
                     f"(fix: {item.fix}){alias_txt}{extra}")
    if skipped:
        lines.append("skipped:\n" + "\n".join(skipped))
    summary = (f"{len(findings)} advisory(ies), {len(blocking)} blocking high/critical, "
               f"{len(accepted)} allowlisted high/critical")
    return Check("pip-audit", summary, "\n".join(lines), bool(blocking))


def vulture_check(py: str) -> Check:
    code, out, err = _run([py, "-m", "vulture", *TARGETS, "--min-confidence", "80"])
    if code == 127 or _missing(err):
        return Check("vulture (report)", "skipped (not installed)", err.strip(), False)
    lines = [ln for ln in out.splitlines() if "% confidence" in ln]
    return Check("vulture (report)", f"{len(lines)} finding(s)", out.strip(), False)


def radon_check(py: str) -> Check:
    code, out, err = _run([py, "-m", "radon", "cc", "-n", "D", "-s", "-j", *TARGETS])
    if code == 127 or _missing(err):
        return Check("radon (report)", "skipped (not installed)", err.strip(), False)
    try:
        data = json.loads(out or "{}")
    except ValueError:
        return Check("radon (report)", "skipped (could not parse)", (out + err).strip(), False)
    items: List[str] = []
    if isinstance(data, dict):
        for filename, blocks in data.items():
            if not isinstance(blocks, list):
                continue
            for block in blocks:
                if isinstance(block, dict):
                    items.append(f"{_rel(str(filename))}:{block.get('lineno')} {block.get('name')} "
                                 f"{block.get('rank')} ({block.get('complexity')})")
    return Check("radon (report)", f"{len(items)} function(s) rank D or worse", "\n".join(items), False)


def _parse_gitleaks(text: str) -> List[dict]:
    if not text.strip():
        return []
    data = json.loads(text)
    if isinstance(data, list):
        return [row for row in data if isinstance(row, dict)]
    if isinstance(data, dict) and isinstance(data.get("findings"), list):
        return [row for row in data["findings"] if isinstance(row, dict)]
    return []


def gitleaks_check(mode: str, rev_range: str) -> Check:
    binary = shutil.which("gitleaks")
    if not binary:
        return Check("gitleaks", "not installed", "gitleaks is not on PATH", True)
    if mode not in ("full", "diff"):
        return Check("gitleaks", "failed to run", f"unknown gitleaks mode {mode!r}", True)
    if mode == "diff" and not rev_range:
        return Check("gitleaks", "failed to run", "diff mode needs a git range (A..B)", True)
    report = tempfile.NamedTemporaryFile(prefix="vx-gitleaks-", suffix=".json", delete=False)
    report.close()
    cmd = [binary, "detect", "--source", str(ROOT), "--redact", "--no-banner", "--exit-code", "1",
           "--report-format", "json", "--report-path", report.name]
    if mode == "diff":
        cmd.append("--log-opts=" + rev_range)
    try:
        code, out, err = _run(cmd, timeout=1200)
        try:
            text = Path(report.name).read_text(encoding="utf-8")
        except OSError:
            text = ""
    finally:
        os.unlink(report.name)
    label = "gitleaks (full history)" if mode == "full" else f"gitleaks (diff {rev_range})"
    if code not in (0, 1):
        return Check(label, "failed to run", (out + "\n" + err).strip(), True)
    try:
        rows = _parse_gitleaks(text)
    except ValueError:
        return Check(label, "failed to run", text.strip() or (out + err).strip(), True)
    lines = []
    for row in rows:
        lines.append(f"{row.get('File')}:{row.get('StartLine')}: {row.get('RuleID')} {row.get('Description')}")
    return Check(label, f"{len(rows)} finding(s)", "\n".join(lines), bool(rows))


def _summary_text(checks: Sequence[Check]) -> str:
    width = max(len(item.name) for item in checks)
    lines = ["Voxprint audit", ""]
    for item in checks:
        gate = "blocks release" if item.blocks else "does not block"
        lines.append(f"  {item.name.ljust(width)}  {item.summary}  ({gate})")
    blocking = [item.name for item in checks if item.blocks]
    lines.append("")
    if blocking:
        lines.append("Release gate: FAIL (" + ", ".join(blocking) + ")")
    else:
        lines.append("Release gate: pass")
    lines.append(f"Details: {REPORT.name}")
    return "\n".join(lines)


def _write_summary(text: str) -> None:
    path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not path:
        return
    body = "## Voxprint audit\n\n```\n" + text + "\n```\n"
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(body)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--python", default=sys.executable, help="interpreter with the app's dependencies (for pip-audit)")
    parser.add_argument("--gitleaks-mode", choices=("full", "diff"), default="full",
                        help="full scans all history (schedule, manual run, installer); diff scans a git range (push and pull request)")
    parser.add_argument("--gitleaks-range", default="", help="git log range for --gitleaks-mode diff, for example base..head")
    args = parser.parse_args(list(argv) if argv is not None else None)
    py = sys.executable
    try:
        allow = load_allowlist()
    except ValueError as exc:
        allow_check = Check("pip-audit allowlist", "invalid", str(exc), True)
        checks = [allow_check]
    else:
        checks = [
            ruff_check(py),
            mypy_check(py),
            bandit_check(py),
            pip_audit_check(py, args.python, allow),
            vulture_check(py),
            radon_check(py),
            gitleaks_check(args.gitleaks_mode, args.gitleaks_range),
        ]
    parts = [f"===== {item.name}: {item.summary}\n{item.detail.strip()}\n" for item in checks]
    REPORT.write_text("\n".join(parts), encoding="utf-8")
    text = _summary_text(checks)
    print(text)
    _write_summary(text)
    return 1 if any(item.blocks for item in checks) else 0


if __name__ == "__main__":
    sys.exit(main())
