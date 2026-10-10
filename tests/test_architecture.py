"""The issued types and the generator call have one construction site each."""
import ast
from pathlib import Path

ROOT = Path("canvas")


def _calls(tree: ast.AST):
    return [node for node in ast.walk(tree) if isinstance(node, ast.Call)]


def _is_generator(node: ast.AST) -> bool:
    if isinstance(node, ast.Name):
        return node.id == "generator"
    if isinstance(node, ast.Attribute):
        return node.attr == "generator"
    return False


def test_safe_image_and_screened_input_are_constructed_only_in_the_gate():
    offenders = []
    for path in ROOT.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in _calls(tree):
            func = node.func
            name = func.id if isinstance(func, ast.Name) else ""
            if name in {"SafeImage", "ScreenedInput"} and path.as_posix() != "canvas/safety/gate.py":
                offenders.append(f"{path}:{node.lineno}")
    assert offenders == []


def test_the_generator_is_called_only_from_the_engine():
    named = []
    attributes = []
    for path in ROOT.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in _calls(tree):
            func = node.func
            if isinstance(func, ast.Name) and func.id == "generate":
                named.append(path.as_posix())
            if isinstance(func, ast.Attribute) and func.attr == "generate" and _is_generator(func.value):
                attributes.append(f"{path.as_posix()}:{node.lineno}")
    assert named == ["canvas/pipeline/engine.py"]
    assert attributes == []


def test_safety_package_does_not_read_the_environment_or_import_nudenet():
    for path in (ROOT / "safety").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr == "environ":
                raise AssertionError(path.as_posix())
            if isinstance(node, ast.Import):
                assert all(alias.name != "os" for alias in node.names)
            if isinstance(node, ast.ImportFrom):
                assert node.module != "os"
                assert all(alias.name != "nudenet" for alias in node.names)
    blob = "\n".join(path.read_text(encoding="utf-8").casefold() for path in (ROOT / "models").rglob("*") if path.suffix in {".py", ".json"})
    assert "nudenet" not in blob
    assert "nudenet" not in Path("pyproject.toml").read_text(encoding="utf-8").casefold()


def test_run_generator_is_the_pipeline_entry():
    engine = (ROOT / "pipeline" / "engine.py").read_text(encoding="utf-8")
    assert "def run_generator" in engine
    pipeline = (ROOT / "pipeline" / "pipeline.py").read_text(encoding="utf-8")
    worker = (ROOT / "pipeline" / "worker.py").read_text(encoding="utf-8")
    assert "run_generator(" in pipeline
    assert "run_generator(" in worker
