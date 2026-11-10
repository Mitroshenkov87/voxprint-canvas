"""Model work for the Canvas window. This module runs in the worker process.

The window imports nothing from here. It names these handlers by string
(``canvas.worker_jobs:run`` and ``canvas.worker_jobs:check``) and the
``voxprint_common.worker`` host imports them in a spawned child. The child
owns the gate, the generator, the warm pipeline, the rewriter and the GPU
lock. Checks run in the worker, on every route, before anything is handed
back.

Requests and replies follow :mod:`canvas.ipc.schema`. A picture leaves this
process only as a file written into the window's private session folder,
after the checks for that picture passed.
"""
from __future__ import annotations

import json
import os
import random
import uuid
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from voxprint_common.worker import JobCancelled, JobContext

from canvas.ipc.schema import PROTO, encode_event, refused, reply

if TYPE_CHECKING:
    from canvas.ui.shell_layout import PictureJob

__all__ = ["Session", "check", "install_session", "run", "session"]


@dataclass
class Session:
    """What the worker keeps between jobs.

    ``check`` is the screen's review adapter (:class:`canvas.app.shell_gate.ContentCheck`)
    around ``gate``. ``runner`` turns one picture job into pixels (a
    :class:`canvas.app.shell_run.ShellRunner` in the app). ``improver`` rewrites
    a prompt. ``series_writer`` replaces the scenario writer when it is set.
    """

    gate: object
    check: object
    tier: object
    runner: object
    improver: Callable[..., str]
    series_writer: Callable[[str], str] | None = None
    warm: bool = False
    extras: dict[str, object] = field(default_factory=dict)


_SESSION: Session | None = None


def install_session(value: Session | None) -> None:
    """Use ``value`` for the following jobs. None builds the default again on the next job."""
    global _SESSION
    _SESSION = value


def session(tier_name: str) -> Session:
    """The worker session, built on first use for ``tier_name``.

    Raises:
        Exception: The gate or the runner could not be built. The job then
            ends as unavailable.
    """
    global _SESSION
    if _SESSION is None:
        _SESSION = _default_session(tier_name)
    return _SESSION


def _default_session(tier_name: str) -> Session:
    from canvas.app.shell_gate import ContentCheck
    from canvas.app.shell_run import ShellRunner, improve_prompt
    from canvas.infra.enlarge import EnlargeClient
    from canvas.infra.paths import models_dir
    from canvas.models.loader import load_catalog_stack
    from canvas.models.modes import Tier
    from canvas.models.registry import load_catalog
    from canvas.safety.gate import AgeGate

    tier = Tier(tier_name)
    gate = AgeGate(load_catalog_stack(models_dir(), load_catalog()))
    runner = ShellRunner(gate=gate, tier=tier, enlarge=EnlargeClient())

    def improver(*, prompt: str, adult: bool, progress: Callable[[str], None]) -> str:
        return improve_prompt(gate=gate, prompt=prompt, tier=tier, adult=adult, progress=progress)

    return Session(gate=gate, check=ContentCheck(gate=gate), tier=tier, runner=runner, improver=improver)


# --- entry points -----------------------------------------------------------------------------------------------


def run(ctx: JobContext) -> dict[str, object]:
    """Run one window request. See :mod:`canvas.ipc.schema` for the ops."""
    _leave_if_cuda_lost()
    payload = ctx.payload
    if not isinstance(payload, Mapping) or payload.get("proto") != PROTO:
        return refused(str(getattr(payload, "get", lambda _k: "")("op") or "hello"), "proto_mismatch", "", unavailable=True)
    op = str(payload.get("op") or "")
    route = _ROUTES.get(op)
    if route is None:
        return refused(op or "hello", "proto_mismatch", "", unavailable=True)
    try:
        current = session(str(payload.get("tier") or ""))
    except Exception as exc:  # noqa: BLE001 - any failure to build the checks ends the job as unavailable
        return refused(op, "check.load", _unavailable_text(exc), unavailable=True)
    if op not in _GPU_ROUTES:
        return route(ctx, current, payload)
    return _gpu_route(ctx, op, lambda: route(ctx, current, payload))


def check(ctx: JobContext) -> dict[str, object]:
    """A ``check`` job: review prompts before the window starts a picture."""
    _leave_if_cuda_lost()
    payload = ctx.payload
    if not isinstance(payload, Mapping) or payload.get("proto") != PROTO or payload.get("op") != "prompts":
        return refused("prompts", "proto_mismatch", "", unavailable=True)
    try:
        current = session(str(payload.get("tier") or ""))
    except Exception as exc:  # noqa: BLE001
        return refused("prompts", "check.load", _unavailable_text(exc), unavailable=True)
    return _prompts(ctx, current, payload)


# --- GPU failures -------------------------------------------------------------------------------------------------


def _gpu_route(ctx: JobContext, op: str, body: Callable[[], dict[str, object]]) -> dict[str, object]:
    """Run a route that uses the GPU. A GPU memory failure ends this job with code ``vram``.

    While the job waits for RAM held by another Canvas job, a ``note`` event
    with stage ``waiting_gpu`` goes to the window. When the CUDA context is
    lost the reply is sent and this process leaves (see :func:`_reply_then_exit`).
    """
    from canvas.infra import cuda_health
    from canvas.models import hot_tier
    from canvas.pipeline.bf16_schedule import VramRefused, is_out_of_memory

    def waiting(stage: str, seconds_left: float) -> None:
        _stop_if_cancelled(ctx)
        ctx.progress(message=encode_event("note", text="", stage=stage, seconds_left=round(seconds_left, 1)))

    previous = hot_tier.set_wait_listener(waiting)
    try:
        result = body()
    except JobCancelled:
        raise
    except Exception as exc:
        if cuda_health.is_cuda_fault(exc):
            cuda_health.mark_lost(exc)
        if not (cuda_health.lost() or isinstance(exc, (VramRefused, MemoryError)) or is_out_of_memory(exc)):
            raise
        result = refused(op, "vram", str(VramRefused()), unavailable=True)
    finally:
        hot_tier.set_wait_listener(previous)
    if cuda_health.lost():
        _reply_then_exit(ctx, _fail_closed(op, result, str(VramRefused())))
    return result


def _fail_closed(op: str, result: dict[str, object], message: str) -> dict[str, object]:
    """A ``vram`` refusal in place of ``result``. Any picture file it named is removed."""
    names: list[Path] = []
    if isinstance(result.get("path"), str):
        names.append(Path(str(result["path"])))
    folder = result.get("folder")
    kept = result.get("kept")
    if isinstance(folder, str) and isinstance(kept, list):
        names.extend(Path(folder) / str(name) for name in kept)
    for path in names:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass
    return refused(op, "vram", message, unavailable=True)


def _reply_then_exit(ctx: JobContext, result: dict[str, object]) -> None:
    """Send ``result`` to the window, then leave with ``EXIT_GPU_UNRECOVERABLE`` (75).

    voxprint-common 0.6.1 treats a clean final reply followed by exit 75 as a
    planned restart: the job is delivered as finished and no crash bar shows.
    """
    from voxprint_common.worker import EXIT_GPU_UNRECOVERABLE

    ctx.reply_and_exit(result, code=EXIT_GPU_UNRECOVERABLE)


def _leave_if_cuda_lost() -> None:
    """A worker whose CUDA context broke takes no further jobs."""
    from canvas.infra import cuda_health

    if cuda_health.lost():
        cuda_health.exit_now(cuda_health.EXIT_CUDA_LOST)


# --- routes -------------------------------------------------------------------------------------------------------


def _hello(ctx: JobContext, current: Session, payload: Mapping[str, object]) -> dict[str, object]:
    del ctx, payload
    from canvas.infra.gpu_lock import prepare_startup_lock

    # The lock file is keyed by the CUDA device this process sees, so it is read here, not in the window.
    lock = prepare_startup_lock() or ""
    tier = getattr(current.tier, "value", str(current.tier))
    return reply("hello", tier=str(tier), pid=os.getpid(), lock=lock)


def _prompts(ctx: JobContext, current: Session, payload: Mapping[str, object]) -> dict[str, object]:
    del ctx
    prompts = payload.get("prompts")
    if not isinstance(prompts, list) or not all(isinstance(item, str) for item in prompts):
        return refused("prompts", "malformed", "", unavailable=True)
    reason = _review_prompts(current, prompts)
    if reason is not None:
        return refused("prompts", "prompt", reason, unavailable=False)
    return reply("prompts")


def _selected_lora(lora_path: str, runner: object):
    """The LoRA a job uses: the runner's bound one, else the one its path names. None without a LoRA."""
    from canvas.models.user_lora import UserLora, resolve_selected

    lora = getattr(runner, "bound_lora", None)
    if isinstance(lora, UserLora):
        return lora
    if not lora_path:
        return None
    try:
        return resolve_selected(lora_path)
    except Exception:  # noqa: BLE001 - a lookup failure means no trigger, never a stopped job
        return None


def _retrain_refusal(op: str, lora_path: str, runner: object) -> dict[str, object] | None:
    """Refuse a job whose LoRA was trained on a base Canvas cannot run.

    The diffusion model is not opened.
    """
    from canvas.app.i18n import text
    from canvas.models.user_lora import FORGE_BASE_ID, UserLora, retrain_rejection, resolve_selected

    lora = getattr(runner, "bound_lora", None)
    if not isinstance(lora, UserLora) and lora_path:
        try:
            lora = resolve_selected(lora_path)
        except Exception:  # noqa: BLE001 - a lookup failure is not a retrain
            lora = None
    if not isinstance(lora, UserLora):
        return None
    rejected = retrain_rejection(lora)
    if rejected is None:
        return None
    body = refused(op, rejected.code, text(rejected.code, **rejected.details), unavailable=True)
    body["retrain"] = {"from": str(rejected.details.get("base") or ""), "to": FORGE_BASE_ID}
    return body


def _generate(ctx: JobContext, current: Session, payload: Mapping[str, object]) -> dict[str, object]:
    from canvas.app.shell_run import RunStopped
    from canvas.core.formats import extension, save_image
    from canvas.infra import cuda_health
    from canvas.models import hot_tier
    from canvas.models.runtime import hold_warm, pipeline_scope, trim_warm

    job = _job(payload.get("job"))
    stopped = _retrain_refusal("generate", job.lora_path, current.runner)
    if stopped is not None:
        return stopped
    from canvas.models.user_lora import apply_trigger, rewrite_alias

    # The alias becomes the trigger and the trigger leads. The gate and the encoder get this prompt.
    lora = _selected_lora(job.lora_path, current.runner)
    job = job._replace(prompt=apply_trigger(job.prompt, lora))
    layout_prompt = rewrite_alias(str(payload.get("layout_prompt") or ""), lora)
    out_dir = Path(str(payload.get("out_dir")))
    _apply_vram(payload.get("vram"))
    reason = _review_prompts(current, [layout_prompt, job.prompt])
    if reason is not None:
        return refused("generate", "prompt", reason, unavailable=False)
    _stop_if_cancelled(ctx)
    try:
        inputs = _inputs(payload.get("inputs"))
    except OSError:
        from canvas.app.i18n import text

        return refused("generate", "job", text("error.job"), unavailable=False)
    ctx.progress(message="", step=0, total_steps=1)
    # One RAM plan per picture. It counts every other Canvas job on this machine, and may wait for one.
    hot_tier.begin_job()
    if payload.get("warm") and not current.warm:
        hold_warm()
        current.warm = True
    try:
        with pipeline_scope():
            raw = current.runner(job, inputs)  # type: ignore[operator]
    except RunStopped as exc:
        return refused("generate", exc.code or "job", exc.message, unavailable=exc.unavailable)
    finally:
        if current.warm:
            trim_warm()
    if ctx.cancelled():
        _wipe(raw)
        raise JobCancelled()
    if cuda_health.lost():
        _wipe(raw)
        from canvas.pipeline.bf16_schedule import VramRefused

        return refused("generate", "vram", str(VramRefused()), unavailable=True)
    item_id = uuid.uuid4().hex
    staged = _copy(raw)
    _wipe(raw)
    current.check.stage(item_id, staged)  # type: ignore[attr-defined]
    reason = current.check.review_result(layout_prompt, item_id)  # type: ignore[attr-defined]
    safe = current.check.kept.pop(item_id, None)  # type: ignore[attr-defined]
    if reason is not None or safe is None:
        return refused("generate", "output", reason or _unavailable_text(None), unavailable=reason is None)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{item_id}{extension(job.fmt)}"
    save_image(safe, path, job.fmt, metadata=_picture_metadata(job))
    return reply("generate", item_id=item_id, path=str(path), width=int(safe.width), height=int(safe.height))


def _improve(ctx: JobContext, current: Session, payload: Mapping[str, object]) -> dict[str, object]:
    prompt = str(payload.get("prompt") or "")

    def progress(message: str) -> None:
        ctx.progress(message=encode_event("note", text=str(message)))

    try:
        rewritten = current.improver(prompt=prompt, adult=bool(payload.get("adult")), progress=progress)
    except Exception as exc:  # noqa: BLE001 - the window keeps the old words and shows the reason
        message = getattr(exc, "user_message", None)
        code = getattr(exc, "code", "") or "internal"
        return refused("improve", str(code), message if isinstance(message, str) else "", unavailable=False)
    if not isinstance(rewritten, str):
        return refused("improve", "internal", "", unavailable=True)
    return reply("improve", prompt=rewritten)


def _lora_import(ctx: JobContext, current: Session, payload: Mapping[str, object]) -> dict[str, object]:
    from canvas.models.lora_library import import_lora
    from canvas.models.user_lora import LoraRejected

    asked: list[str] = []
    confirmed = payload.get("confirmed") is True

    def confirm(manifest: object) -> bool:
        if confirmed:
            return True
        asked.append(str(getattr(manifest, "name", "")))
        return False

    def progress(line: str) -> None:
        ctx.progress(message=encode_event("note", text=str(line)))

    root = payload.get("root")
    try:
        entry = import_lora(
            Path(str(payload.get("path"))),
            current.gate,
            root=Path(str(root)) if root else None,
            confirm=confirm,
            progress=progress,
        )
    except LoraRejected as exc:
        return refused("lora_import", exc.code, "", unavailable=False)
    if entry is None and asked:
        return {"proto": PROTO, "op": "lora_import", "status": "needs_confirm", "name": asked[0]}
    return reply("lora_import", entry_id=None if entry is None else entry.id)


def _selftest(ctx: JobContext, current: Session, payload: Mapping[str, object]) -> dict[str, object]:
    del ctx, current
    from canvas.app.selftest import selftest_report

    report = selftest_report(deep=bool(payload.get("deep")))
    return reply("selftest", report=json.loads(json.dumps(report, default=str)))


def _probe_gpu(ctx: JobContext, current: Session, payload: Mapping[str, object]) -> dict[str, object]:
    del ctx, current, payload
    from canvas.infra.gpu import probe_gpu

    return reply("probe_gpu", gpu=probe_gpu() is not None)


def _warm(ctx: JobContext, current: Session, payload: Mapping[str, object]) -> dict[str, object]:
    del ctx
    from canvas.models.runtime import drop_warm

    if payload.get("action") == "drop" and current.warm:
        current.warm = False
        drop_warm(release=True)
    return reply("warm", warm=current.warm)


# --- Photoshoot series --------------------------------------------------------------------------------------------


class _Halt(Exception):
    """The series cannot continue because the machine stopped the picture."""

    def __init__(self, message: str, *, unavailable: bool = False) -> None:
        super().__init__(message)
        self.message = message
        self.unavailable = unavailable


def _series(ctx: JobContext, current: Session, payload: Mapping[str, object]) -> dict[str, object]:
    from canvas.app.shell_run import RunStopped, load_rgb
    from canvas.core.formats import OutputFormat
    from canvas.infra.gpu_lock import lock_path, release
    from canvas.models.loader import ModelNotReady
    from canvas.models.modes import Mode
    from canvas.models.runtime import InferenceUnavailable
    from canvas.pipeline.photoshoot import assemble_line, frame_negative
    from canvas.pipeline.series import REFERENCE_STRENGTH, SeriesFrame, run_series
    from canvas.safety.image import RawImage
    from canvas.safety.inputs import ImageRole, PendingInput

    base = _job(payload.get("job"))
    stopped = _retrain_refusal("series", base.lora_path, current.runner)
    if stopped is not None:
        return stopped
    from canvas.models.user_lora import apply_trigger, rewrite_alias

    lora = _selected_lora(base.lora_path, current.runner)
    idea = rewrite_alias(str(payload.get("idea") or ""), lora)
    style = str(payload.get("style") or "photo")
    template = str(payload.get("template") or "")
    suffix = str(payload.get("suffix") or "")
    seed_mode = str(payload.get("seed_mode") or "series")
    base_seed = _whole(payload.get("seed"))
    adult = bool(payload.get("adult"))
    series_id = str(payload.get("series_id"))
    session_dir = Path(str(payload.get("out_dir")))
    folder = session_dir / "series" / series_id
    manifest = session_dir / "series" / f"{series_id}.json"
    strength = REFERENCE_STRENGTH if payload.get("strength") == "high" else None
    container = OutputFormat(str(payload.get("container") or "png"))
    runner = current.runner
    reason = _review_prompts(current, [apply_trigger(idea, lora)])
    if reason is not None:
        return refused("series", "prompt", reason, unavailable=False)
    # Adult content off: the same boundary on the idea. The gate review above ran first and is unchanged.
    from canvas.pipeline import adult_boundary

    off = adult_boundary.refusal(idea, allowed=adult)
    if off is not None:
        return refused("series", "adult_off", off, unavailable=False)

    def compose(line: str) -> str:
        return apply_trigger(" ".join(part for part in (template, line.strip(), suffix) if part), lora)

    def note(message: str) -> None:
        ctx.progress(message=encode_event("note", text=str(message)))

    writer = _writer_session(current, adult, note)

    def rewrite(prompt: str) -> str:
        if current.series_writer is not None:
            return current.series_writer(prompt)
        return writer(prompt)

    rewrite.release = writer.release  # type: ignore[attr-defined]
    seeds: dict[int, int] = {}
    scenario_box: dict[str, object] = {}

    def job_for(line: str, seed: int, hero: SeriesFrame | None, frame_strength: float | None):
        # Adult content off: every per-image line follows the same boundary. The gate calls stay as they are.
        from canvas.pipeline import adult_boundary

        off = adult_boundary.refusal(line, allowed=adult)
        if off is not None:
            raise RunStopped(off, code="adult_off")
        used = seeds.setdefault(seed, _series_seed(seed_mode, base_seed, seed))
        picture = base._replace(
            prompt=compose(line),
            negative=frame_negative(base.negative),
            seed=used,
            mode=Mode.TEXT if hero is None else Mode.RESTYLE,
            copies=1,
            aspect="1:1",
            output="native",
        )
        fresh: tuple[PendingInput, ...] = ()
        if hero is None:
            if hasattr(runner, "img2img_strength"):
                runner.img2img_strength = None
        else:
            if hasattr(runner, "img2img_strength"):
                runner.img2img_strength = frame_strength
            # Always a copy: the pipeline wipes its inputs, and the hero is reused for every shot.
            fresh = (PendingInput(ImageRole.INIT, RawImage(hero.width, hero.height, hero.to_bytes())),)
        return picture, fresh

    def generate(line: str, seed: int, hero: SeriesFrame | None, frame_strength: float | None):
        picture, fresh = job_for(line, seed, hero, frame_strength)
        try:
            return runner(picture, fresh)  # type: ignore[operator]
        except RunStopped as exc:
            if exc.unavailable:
                raise _Halt(exc.message, unavailable=True) from exc
            raise

    def sample(line: str, seed: int, hero: SeriesFrame | None, frame_strength: float | None) -> bool:
        sampler = getattr(runner, "sample", None)
        if not callable(sampler):
            return False
        picture, fresh = job_for(line, seed, hero, frame_strength)
        try:
            return sampler(picture, fresh) is True
        except RunStopped as exc:
            if exc.unavailable:
                raise _Halt(exc.message, unavailable=True) from exc
            raise

    def prepare(lines: list[str]) -> None:
        precompute = getattr(runner, "precompute", None)
        if not callable(precompute):
            return
        try:
            precompute(tuple(compose(line) for line in lines))
        except (ImportError, ModelNotReady, InferenceUnavailable) as exc:
            raise _Halt(str(exc), unavailable=True) from exc

    generate.prepare = prepare  # type: ignore[attr-defined]
    generate.sample = sample  # type: ignore[attr-defined]
    # The series puts the trigger first on every shot line before the gate reviews it.
    generate.lora = lora  # type: ignore[attr-defined]

    def on_scenario(scenario: object) -> None:
        scenario_box["scenario"] = scenario
        ctx.progress(message=encode_event(
            "scenario",
            subject=str(getattr(scenario, "subject_block", "")),
            shots=[asdict(shot) for shot in getattr(scenario, "shots", ())],
            seed=base_seed,
        ))

    def on_status(index: int, status: str) -> None:
        ctx.progress(message=encode_event("status", index=int(index), status=str(status)))

    def line_for(index: int) -> str:
        scenario = scenario_box.get("scenario")
        shots = getattr(scenario, "shots", ())
        return assemble_line(
            str(getattr(scenario, "subject_block", "")), shots[index], include_camera=True, index=index, seed=base_seed,
            lora=lora,
        )

    if hasattr(runner, "cook"):
        runner.cook = True
    try:
        result = run_series(
            idea=idea,
            count=max(1, _whole(payload.get("count"))),
            style=style,
            place_mode=str(payload.get("place") or "same"),
            seed=base_seed,
            strength=strength,
            include_camera=lambda: True,
            rewrite=rewrite,
            age_gate=current.gate,  # type: ignore[arg-type]
            draw_frame=generate,
            cancelled=ctx.cancelled,
            folder=folder,
            on_scenario=on_scenario,
            line_for=line_for,
            on_status=on_status,
            adult=adult,
            container=container,
        )
    except _Halt as exc:
        return refused("series", "halt", exc.message, unavailable=exc.unavailable)
    finally:
        writer.release()
        if hasattr(runner, "cook"):
            runner.cook = False
        release(lock_path())
    hero_released = False
    hero_reason = ""
    if result.kept and result.folder is not None:
        item_id = f"series-{series_id}"
        try:
            raw = load_rgb(result.kept[0])
        except OSError:
            raw = None
        if raw is not None:
            current.check.stage(item_id, raw)  # type: ignore[attr-defined]
            reason = current.check.review_result(idea, item_id)  # type: ignore[attr-defined]
            safe = current.check.kept.pop(item_id, None)  # type: ignore[attr-defined]
            hero_released = reason is None and safe is not None
            hero_reason = "" if hero_released else (reason or _unavailable_text(None))
    body = {
        "stopped": result.stopped,
        "reason": result.reason,
        "code": result.code,
        "source": result.source,
        "folder": None if result.folder is None else str(result.folder),
        "kept": [path.name for path in result.kept],
        "hero_released": hero_released,
        "hero_reason": hero_reason,
    }
    if ctx.cancelled():
        # The host drops a cancelled job's return value. The same fields go to a noted file instead.
        manifest.parent.mkdir(parents=True, exist_ok=True)
        temporary = manifest.with_suffix(".tmp")
        temporary.write_text(json.dumps({"proto": PROTO, "op": "series", "status": "ok", **body}), encoding="utf-8")
        os.replace(temporary, manifest)
        ctx.note_checkpoint(str(manifest))
    return reply("series", **body)


def _writer_session(current: Session, adult: bool, note: Callable[[str], None]):
    from canvas.infra.paths import models_dir
    from canvas.models.registry import load_catalog
    from canvas.models.snapshots import SnapshotError
    from canvas.pipeline.adult_direction import scenario_system
    from canvas.pipeline.enhance import WriterSession, open_rewriter
    from canvas.pipeline.photoshoot import SYSTEM_PROMPT, tokens_for_request

    def open_writer(budget: int):
        try:
            return open_rewriter(load_catalog(), current.tier, models_dir(), note, max_new_tokens=budget)  # type: ignore[arg-type]
        except (SnapshotError, OSError) as exc:
            raise _Halt(str(exc), unavailable=True) from exc

    return WriterSession(open_writer, scenario_system(SYSTEM_PROMPT, adult), tokens_for_request)


def _series_seed(mode: str, base: int, series_seed: int) -> int:
    if mode == "series":
        return series_seed
    if mode == "locked":
        return base
    if mode == "random":
        return random.SystemRandom().randrange(1, 2**31)
    raise ValueError(mode)


def _picture_metadata(job: PictureJob) -> dict[str, Any]:
    """Metadata recorded on one saved picture, in the shared schema. Empty when the job turned it off.

    ``seed`` and ``steps`` are integers, the LoRA strength is a float, and a
    resolved LoRA is listed under ``loras`` with only the fields that are
    known. Every field is best-effort: a LoRA that cannot be re-read never
    stops the picture; its raw strength field is recorded instead.
    """
    if not job.save_metadata:
        return {}
    from canvas.models.user_lora import clamp_strength, resolve_selected, running_base_id

    fields: dict[str, Any] = {
        "prompt": job.prompt,
        "seed": int(job.seed),
        "steps": int(job.steps),
    }
    base = running_base_id()
    if base:
        fields["base"] = base
    if not job.lora_path:
        return fields
    strength: float = job.lora_weight
    name = ""
    digest = ""
    lora_base = ""
    try:
        selected = resolve_selected(job.lora_path)
    except Exception:  # noqa: BLE001 - metadata never blocks a job
        selected = None
    if selected is not None:
        strength = clamp_strength(selected, job.lora_weight)
        name = selected.name or selected.id
        digest = selected.sha256
        lora_base = selected.base
    fields["lora_strength"] = strength
    if not name:
        return fields
    lora: dict[str, Any] = {"name": name, "strength": strength}
    if digest:
        lora["sha256"] = digest
    if lora_base:
        lora["base"] = lora_base
    fields["loras"] = [lora]
    return fields


# --- helpers ------------------------------------------------------------------------------------------------------


def _review_prompts(current: Session, prompts: list[str]) -> str | None:
    """The screen's prompt review for each prompt. The first reason stops."""
    for prompt in prompts:
        if not prompt.strip():
            continue
        reason = current.check.review_prompt(prompt)  # type: ignore[attr-defined]
        if reason:
            return str(reason)
    return None


def _job(value: object):
    from canvas.core.formats import OutputFormat
    from canvas.models.modes import Mode
    from canvas.ui.shell_layout import PictureJob

    if not isinstance(value, Mapping):
        raise ValueError("the job is missing")
    fields = dict(value)
    fields["mode"] = Mode(str(fields["mode"]))
    fields["fmt"] = OutputFormat(str(fields["fmt"]))
    return PictureJob(**fields)


def _inputs(value: object) -> tuple[object, ...]:
    from canvas.app.shell_run import load_rgb
    from canvas.safety.inputs import ImageRole, PendingInput

    if not value:
        return ()
    if not isinstance(value, list):
        raise OSError("inputs must be a list")
    found = []
    for item in value:
        if not isinstance(item, Mapping):
            raise OSError("bad input")
        found.append(PendingInput(ImageRole(str(item.get("role"))), load_rgb(Path(str(item.get("path"))))))
    return tuple(found)


def _apply_vram(value: object) -> None:
    from canvas.infra.vram import set_vram_cap_gb

    if value == "auto":
        set_vram_cap_gb(None)
    elif value in {"8", "12", "16", "24"}:
        set_vram_cap_gb(float(str(value)))


def _whole(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        return 0
    return value


def _copy(raw: object):
    from canvas.safety.image import RawImage

    return RawImage(raw.width, raw.height, raw.to_bytes())  # type: ignore[attr-defined]


def _wipe(raw: object) -> None:
    wipe = getattr(raw, "wipe", None)
    if callable(wipe):
        wipe()


def _stop_if_cancelled(ctx: JobContext) -> None:
    if ctx.cancelled():
        raise JobCancelled()


def _unavailable_text(exc: BaseException | None) -> str:
    from canvas.app.i18n import text

    message = getattr(exc, "user_message", None)
    if isinstance(message, str) and message:
        return message
    return text("gate.unavailable")


# Routes that put models on the GPU.
_GPU_ROUTES = frozenset({"generate", "series", "improve"})

_ROUTES: dict[str, Callable[[JobContext, Session, Mapping[str, object]], dict[str, object]]] = {
    "hello": _hello,
    "prompts": _prompts,
    "generate": _generate,
    "series": _series,
    "improve": _improve,
    "lora_import": _lora_import,
    "selftest": _selftest,
    "probe_gpu": _probe_gpu,
    "warm": _warm,
}
