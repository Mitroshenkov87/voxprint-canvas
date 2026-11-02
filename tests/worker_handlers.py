"""Worker handlers for tests that spawn a real worker process.

Each job installs a fake session described by a fixture, then runs the app's
own handlers (:func:`canvas.worker_jobs.run` and ``check``) unchanged. The
fixture comes from the request's ``_fixture`` field or, for a window test,
from the ``CANVAS_TEST_WORKER`` environment variable the spawned child
inherits. It describes the fake classifiers (``tests.fakes.World``) and the
runner, which uses the real :class:`canvas.pipeline.pipeline.Pipeline` with a
fake generator, so input and output checks run exactly as in the app.

Some fixture keys make the handler misbehave on purpose, for the fail-closed
tests: ``fault`` is ``raise``, ``exit``, ``malformed``, ``outside`` or
``sleep``.
"""
from __future__ import annotations

import json
import os
import time
from collections.abc import Mapping

from canvas import worker_jobs
from canvas.app.shell_gate import ContentCheck
from canvas.app.shell_run import ShellRunner
from canvas.ipc.schema import PROTO
from canvas.models.modes import Tier
from canvas.pipeline.pipeline import Pipeline
from canvas.safety.classifiers import Face
from canvas.safety.image import RawImage
from tests.fakes import World

FIXTURE_ENV = "CANVAS_TEST_WORKER"
_installed: dict[str, str] = {}


class FakeGenerator:
    """Returns a small solid picture for any screened input."""

    def __init__(self, color: tuple[int, int, int] = (90, 120, 150)) -> None:
        self.color = color

    def generate(self, screened, width: int | None = None, height: int | None = None) -> RawImage:
        del screened, width, height
        return RawImage(8, 8, bytes(self.color) * 64)


class PipelineRunner(ShellRunner):
    """The app's runner (refusal mapping, GPU lock release) around the real pipeline and a fake generator."""

    def __init__(self, *, gate, tier: Tier, delay: float = 0.0) -> None:
        super().__init__(gate=gate, tier=tier, enlarge=None)
        self.delay = delay

    def _generator(self, job):
        del job
        return FakeGenerator()

    def precompute(self, prompts):
        del prompts

    def _through_pipeline(self, job, inputs):
        if self.delay:
            time.sleep(self.delay)
        pipeline = Pipeline(gate=self.gate, generator=FakeGenerator(), tier=self.tier)
        return pipeline.generate(
            job.prompt, job.mode, inputs, aspect=job.aspect, output=job.output, explicit=job.adult_content,
        )


def world_from(fixture: Mapping[str, object]) -> World:
    """A ``World`` with the classifier values the fixture names."""
    world = World()
    spec = fixture.get("world") or {}
    assert isinstance(spec, Mapping)
    for key, value in spec.items():
        if key in {"text_scores", "image_scores"}:
            getattr(world, key).update(value)  # type: ignore[arg-type]
        elif key == "faces":
            world.faces = [Face(*box) for box in value]  # type: ignore[union-attr]
        elif key == "ages":
            world.ages = [tuple(pair) for pair in value]  # type: ignore[union-attr]
        elif key == "categories":
            world.categories = tuple(value)  # type: ignore[arg-type]
        else:
            setattr(world, key, value)
    return world


def session_from(fixture: Mapping[str, object]) -> worker_jobs.Session:
    """The fake worker session for ``fixture``. ``gate: broken`` fails like a gate that cannot load."""
    if fixture.get("gate") == "broken":
        raise OSError("the check models could not be loaded")
    tier = Tier(str(fixture.get("tier", Tier.GB24.value)))
    gate = world_from(fixture).gate()
    runner = PipelineRunner(gate=gate, tier=tier, delay=float(fixture.get("delay", 0.0)))  # type: ignore[arg-type]

    def improve(*, prompt: str, adult: bool, progress) -> str:
        del adult, progress
        from canvas.pipeline.enhance import resolve_prompt

        gate.review_prompt(prompt)
        rewritten = str(fixture.get("rewrite", "{prompt} in warm light")).format(prompt=prompt)
        return resolve_prompt(gate, prompt, rewritten).prompt

    writer_reply = fixture.get("writer")
    writer = (lambda _prompt: str(writer_reply)) if writer_reply else None
    return worker_jobs.Session(
        gate=gate, check=ContentCheck(gate=gate), tier=tier, runner=runner, improver=improve, series_writer=writer,
    )


def _fixture(payload: object) -> dict[str, object]:
    if isinstance(payload, Mapping) and isinstance(payload.get("_fixture"), Mapping):
        return dict(payload["_fixture"])  # type: ignore[arg-type]
    raw = os.environ.get(FIXTURE_ENV, "")
    return dict(json.loads(raw)) if raw else {}


def _install(fixture: Mapping[str, object]) -> None:
    if fixture.get("base_ready"):
        from canvas.models import lora_library

        lora_library._base_on_disk = lambda: True  # type: ignore[assignment]
    key = json.dumps(fixture, sort_keys=True)
    if _installed.get("key") == key and worker_jobs._SESSION is not None:
        return
    _installed["key"] = key
    worker_jobs.install_session(None)
    try:
        worker_jobs.install_session(session_from(fixture))
    except Exception:
        _installed.pop("key", None)
        worker_jobs.install_session(None)
        # The app then builds its own session and fails the same way a broken gate does.
        worker_jobs._default_session = _broken  # type: ignore[assignment]


def _broken(_tier: str):
    raise OSError("the check models could not be loaded")


def _faults(ctx, fixture: Mapping[str, object]) -> dict[str, object] | None:
    op = ctx.payload.get("op") if isinstance(ctx.payload, Mapping) else ""
    fault = fixture.get("fault")
    if fault is None or op not in fixture.get("fault_ops", [op]):  # type: ignore[operator]
        return None
    if fault == "raise":
        raise RuntimeError("handler failed")
    if fault == "exit":
        os._exit(9)
    if fault == "sleep":
        ctx.progress(message="", step=0, total_steps=1)
        time.sleep(float(fixture.get("sleep", 60.0)))
        return {"proto": PROTO, "op": op, "status": "ok"}
    if fault == "malformed":
        return {"proto": PROTO, "op": op, "status": "ok", "item_id": "not-an-id", "width": "wide"}
    if fault == "outside":
        # The right shape, but the picture is not in the session folder the window gave.
        outside = os.path.join(os.path.dirname(str(ctx.payload.get("out_dir"))), "a" * 32 + ".avif")
        with open(outside, "wb") as handle:
            handle.write(b"not a picture")
        return {"proto": PROTO, "op": op, "status": "ok", "item_id": "a" * 32, "path": outside, "width": 8, "height": 8}
    if fault == "proto":
        return {"proto": "other/9", "op": op, "status": "ok"}
    return None


def run(ctx):
    """Install the fixture's session, then run the app's handler."""
    fixture = _fixture(ctx.payload)
    _install(fixture)
    broken = _faults(ctx, fixture)
    if broken is not None:
        return broken
    if fixture.get("pause") and ctx.payload.get("op") != "hello":
        time.sleep(float(fixture["pause"]))
    return worker_jobs.run(ctx)


def check(ctx):
    """Install the fixture's session, then run the app's check handler."""
    fixture = _fixture(ctx.payload)
    _install(fixture)
    broken = _faults(ctx, fixture)
    if broken is not None:
        return broken
    return worker_jobs.check(ctx)
