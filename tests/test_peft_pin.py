"""LoRA loading needs peft in the installed app."""
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PIN = "peft==0.21.2"


def test_peft_is_pinned_for_the_installer():
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    gen = pyproject.split("gen = [", 1)[1].split("]", 1)[0]
    assert PIN in gen
    assert PIN in (ROOT / "constraints.txt").read_text(encoding="utf-8").splitlines()


@pytest.mark.skipif(importlib.util.find_spec("diffusers") is None, reason="generation extra not installed")
def test_peft_imports_where_diffusers_is_installed():
    from peft import inject_adapter_in_model, set_peft_model_state_dict

    assert callable(inject_adapter_in_model)
    assert callable(set_peft_model_state_dict)
