"""Every content route gives the same outcome through the worker process as in-process.

One real worker is spawned (``voxprint_common.worker.WorkerHost``) with
``tests.worker_handlers``, which installs fake classifiers from
``tests.fakes.World`` and then runs the app's handlers unchanged. The direct
side makes the same calls the window used to make in its own process, with
the same fakes. The fixture image is the documented 40-year-old adult from
``tests/fixtures/adult_cc0``. No new expectations are written here: each case
only asserts that both sides agree.
"""
from __future__ import annotations

import os
import uuid
from pathlib import Path

import pytest

from PIL import Image
from voxprint_common.worker import Handlers, JobKind, JobSpec, JobState, WorkerHost

from canvas.app.shell_run import RunStopped, load_rgb
from canvas.ipc.schema import parse_reply, request
from canvas.models.modes import Mode, Tier
from canvas.safety.constants import IMAGE_MINOR_SEXUAL
from canvas.safety.image import RawImage
from canvas.safety.inputs import ImageRole, PendingInput
from canvas.ui.shell_layout import PictureJob
from canvas.core.formats import OutputFormat
from tests.test_photoshoot import IDEA, _payload
from tests.voxlora_files import write_voxlora
from tests.worker_handlers import session_from

ADULT = Path(__file__).parent / "fixtures" / "adult_cc0" / "torso_40.jpg"

# World settings taken from the existing gate tests (tests/test_gate.py, tests/test_shell_gate.py).
WORLDS: dict[str, dict[str, object]] = {
    "clear": {},
    "guard_sexual": {"safety": "unsafe", "categories": ["Sexual Content or Sexual Acts"]},
    "guard_error": {"error": "guard"},
    "flagged_young_face": {"nsfw": 0.9, "faces": [[0, 0, 1, 1]], "ages": [[1.0, 0.0]]},
    "flagged_adult_face": {"nsfw": 0.9, "faces": [[0, 0, 2, 2]], "ages": [[0.0, 0.0]], "fine": [30.0]},
    "minor_sexual_label": {"nsfw": 0.9, "image_scores": {IMAGE_MINOR_SEXUAL: 0.35}},
    "nsfw_error": {"error": "nsfw"},
    "faces_error": {"nsfw": 0.9, "error": "faces"},
}
PROMPTS = ("a red bicycle", "nude child", "nude", "loli")
MODES = (Mode.TEXT, Mode.TEXT_CREATIVE, Mode.RESTYLE, Mode.INPAINT, Mode.OUTPAINT, Mode.POSE, Mode.DEPTH, Mode.EDGE)
ROLES = {Mode.RESTYLE: "init", Mode.POSE: "pose", Mode.DEPTH: "depth", Mode.EDGE: "edge"}


@pytest.fixture(scope="module")
def session(tmp_path_factory):
    home = tmp_path_factory.mktemp("parity-home")
    previous = os.environ.get("VOXPRINT_HOME")
    os.environ["VOXPRINT_HOME"] = str(home)
    host = WorkerHost(
        Handlers(run="tests.worker_handlers:run", check="tests.worker_handlers:check"),
        state_dir=home / "worker",
    )
    folder = tmp_path_factory.mktemp("parity-session")
    yield host, folder, home
    host.shutdown()
    if previous is None:
        os.environ.pop("VOXPRINT_HOME", None)
    else:
        os.environ["VOXPRINT_HOME"] = previous


def _ask(session, op: str, fixture: dict[str, object], kind: JobKind = JobKind.INFERENCE, **fields):
    host, folder, _home = session
    payload = request(op, tier=Tier.GB24.value, out_dir=str(folder), _fixture=fixture, **fields)
    job_id = host.submit(JobSpec(kind, payload=payload, phase=op))
    outcome = host.wait(job_id, timeout=120)
    assert outcome.state is JobState.FINISHED, outcome.message
    return parse_reply(op, outcome.value, session_dir=folder)


def _verdict(reply) -> tuple[str, str, bool]:
    if reply.ok:
        return ("released", "", False)
    if reply.status == "needs_confirm":
        return ("needs_confirm", str(reply.data.get("name")), False)
    return ("refused", reply.message, reply.unavailable)


def _job(mode: Mode, prompt: str, adult: bool) -> PictureJob:
    return PictureJob(
        prompt=prompt, negative="", mode=mode, aspect="1:1", output="native", fmt=OutputFormat.WEBP, seed=7,
        steps=8, guidance=1.0, sampler="euler", copies=1, adult_content=adult,
    )


def _inputs(mode: Mode, folder: Path) -> list[dict[str, str]]:
    if mode in {Mode.TEXT, Mode.TEXT_CREATIVE}:
        return []
    if mode in {Mode.INPAINT, Mode.OUTPAINT}:
        mask = folder / "mask.png"
        if not mask.exists():
            with Image.open(ADULT) as opened:
                Image.new("RGB", opened.size, (255, 255, 255)).save(mask)
        return [{"role": "init", "path": str(ADULT)}, {"role": "mask", "path": str(mask)}]
    return [{"role": ROLES[mode], "path": str(ADULT)}]


def _direct_generate(fixture, job: PictureJob, layout_prompt: str, inputs: list[dict[str, str]]) -> tuple[str, str, bool]:
    """The calls the window made in its own process before the split, with the same fakes."""
    current = session_from(fixture)
    for prompt in (layout_prompt, job.prompt):
        reason = current.check.review_prompt(prompt)
        if reason:
            return ("refused", reason, False)
    pending = tuple(PendingInput(ImageRole(item["role"]), load_rgb(Path(item["path"]))) for item in inputs)
    try:
        raw = current.runner(job, pending)
    except RunStopped as exc:
        return ("refused", exc.message, exc.unavailable)
    item_id = uuid.uuid4().hex
    current.check.stage(item_id, RawImage(raw.width, raw.height, raw.to_bytes()))
    reason = current.check.review_result(layout_prompt, item_id)
    if reason is not None or current.check.kept.pop(item_id, None) is None:
        return ("refused", reason or "", reason is None)
    return ("released", "", False)


@pytest.mark.parametrize("mode", MODES, ids=lambda mode: mode.value)
def test_generate_matches_in_every_mode(session, mode: Mode) -> None:
    _host, folder, _home = session
    inputs = _inputs(mode, folder)
    outcomes: set[str] = set()
    for world_name, world in WORLDS.items():
        for prompt in PROMPTS:
            for adult in (False, True):
                fixture = {"world": world, "tier": Tier.GB24.value}
                job = _job(mode, prompt, adult)
                fields = {"job": {**job._asdict(), "mode": mode.value, "fmt": job.fmt.value},
                          "layout_prompt": prompt, "inputs": inputs, "vram": "auto", "warm": False}
                through_worker = _verdict(_ask(session, "generate", fixture, **fields))
                direct = _direct_generate(fixture, job, prompt, inputs)
                assert through_worker == direct, (mode, world_name, prompt, adult)
                outcomes.add(direct[0])
    # The matrix covers both a release and a refusal in every mode.
    assert outcomes == {"released", "refused"}


def test_a_released_picture_is_a_file_in_the_session_folder(session) -> None:
    _host, folder, _home = session
    job = _job(Mode.TEXT, "a red bicycle", False)
    reply = _ask(session, "generate", {"world": {}}, job={**job._asdict(), "mode": "text", "fmt": "webp"},
                 layout_prompt="a red bicycle", inputs=[], vram="auto", warm=False)
    assert reply.ok
    path = Path(str(reply.data["path"]))
    assert path.parent == folder and path.is_file()
    refused = _ask(session, "generate", {"world": WORLDS["flagged_young_face"]},
                   job={**job._asdict(), "mode": "text", "fmt": "webp"},
                   layout_prompt="a red bicycle", inputs=[], vram="auto", warm=False)
    assert not refused.ok and "path" not in refused.data


def test_prompt_check_jobs_match(session) -> None:
    for world in WORLDS.values():
        for prompt in PROMPTS:
            fixture = {"world": world}
            through_worker = _verdict(_ask(session, "prompts", fixture, JobKind.CHECK, prompts=[prompt, prompt]))
            reason = session_from(fixture).check.review_prompt(prompt)
            direct = ("refused", reason, False) if reason else ("released", "", False)
            assert through_worker == direct, (world, prompt)


def test_improve_matches(session) -> None:
    for world in WORLDS.values():
        for prompt in PROMPTS:
            fixture = {"world": world}
            reply = _ask(session, "improve", fixture, prompt=prompt, adult=False)
            current = session_from(fixture)
            try:
                rewritten = current.improver(prompt=prompt, adult=False, progress=lambda _line: None)
            except Exception as exc:  # noqa: BLE001 - the window showed the same sentence
                message = getattr(exc, "user_message", "")
                assert not reply.ok and reply.message == (message if isinstance(message, str) else ""), (world, prompt)
                continue
            assert reply.ok and reply.data["prompt"] == rewritten, (world, prompt)


def test_series_matches(session, tmp_path: Path) -> None:
    from canvas.core.formats import OutputFormat as Format
    from canvas.pipeline.series import run_series

    kept_counts: set[int] = set()
    for name in ("clear", "flagged_young_face", "minor_sexual_label", "guard_error"):
        for idea in (IDEA, "nude child"):
            fixture = {"world": WORLDS[name], "writer": _payload(IDEA, 3)}
            series_id = uuid.uuid4().hex
            job = _job(Mode.TEXT, idea, False)
            reply = _ask(
                session, "series", fixture,
                idea=idea, count=3, style="photo", place="same", seed_mode="series", strength="medium", seed=5,
                adult=False, container="png", job={**job._asdict(), "mode": "text", "fmt": "webp"},
                template="", suffix="", series_id=series_id,
            )
            current = session_from(fixture)
            reason = current.check.review_prompt(idea)
            if reason:
                assert not reply.ok and reply.message == reason, (name, idea)
                continue

            def frame(line, seed, hero, strength, current=current):
                current.runner.img2img_strength = strength
                fresh = () if hero is None else (
                    PendingInput(ImageRole.INIT, RawImage(hero.width, hero.height, hero.to_bytes())),
                )
                picture = job._replace(prompt=line, seed=seed, mode=Mode.TEXT if hero is None else Mode.RESTYLE,
                                       copies=1, aspect="1:1", output="native")
                return picture, fresh

            def draw(line, seed, hero, strength, current=current):
                return current.runner(*frame(line, seed, hero, strength))

            # The old window set the cook pass and gave the series these two hooks.
            draw.sample = lambda *args, current=current: current.runner.sample(*frame(*args)) is True
            draw.prepare = lambda lines, current=current: current.runner.precompute(tuple(lines))
            current.runner.cook = True
            direct = run_series(
                idea=idea, count=3, style="photo", place_mode="same", seed=5, strength=None,
                include_camera=lambda: True, rewrite=current.series_writer, age_gate=current.gate,
                draw_frame=draw, cancelled=lambda: False, folder=tmp_path / series_id, container=Format.PNG,
            )
            assert reply.ok, (name, idea, reply)
            assert (reply.data["stopped"], reply.data["code"]) == (direct.stopped, direct.code), (name, idea)
            assert reply.data["kept"] == [path.name for path in direct.kept], (name, idea)
            kept_counts.add(len(direct.kept))
    # Both a full series and a series with refused frames are covered.
    assert 3 in kept_counts and min(kept_counts) < 3


def test_lora_import_matches(session, tmp_path: Path, monkeypatch) -> None:
    from canvas.models import lora_library
    from canvas.models.user_lora import LoraRejected

    monkeypatch.setattr(lora_library, "_base_on_disk", lambda: True)
    for name in ("clear", "flagged_young_face", "minor_sexual_label", "nsfw_error"):
        fixture = {"world": WORLDS[name], "base_ready": True}
        package = write_voxlora(tmp_path / f"in-{name}")
        reply = _ask(session, "lora_import", fixture, path=str(package), root=str(tmp_path / f"worker-{name}"),
                     confirmed=False)
        try:
            entry = lora_library.import_lora(package, session_from(fixture).gate, root=tmp_path / f"direct-{name}")
        except LoraRejected as exc:
            assert not reply.ok and reply.code == exc.code, name
            continue
        assert reply.ok and (reply.data["entry_id"] is None) == (entry is None), name


def test_a_third_party_lora_asks_through_the_worker_like_in_process(session, tmp_path: Path) -> None:
    from tests.lora_files import write_safetensors

    path = tmp_path / "x.safetensors"
    write_safetensors(path, {"layers.0.attention.to_q.lora_A.weight": b"\x01",
                             "layers.0.attention.to_q.lora_B.weight": b"\x02"}, {})
    fixture = {"world": {}, "base_ready": True}
    first = _ask(session, "lora_import", fixture, path=str(path), root=str(tmp_path / "lib"), confirmed=False)
    assert first.status == "needs_confirm"
    second = _ask(session, "lora_import", fixture, path=str(path), root=str(tmp_path / "lib"), confirmed=True)
    assert second.ok and isinstance(second.data["entry_id"], str)
