"""Command line and window startup."""
import argparse
import json
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path

from PIL import Image
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication, QMessageBox
from voxprint_common.i18n import tr

from canvas.app.branding import icon_path, set_app_user_model_id
from canvas.app.identity import version_label
from canvas.app.i18n import text
from canvas.app.info import APP_NAME, APP_VERSION, ENLARGE_NAME
from canvas.app.platform import GpuCheck, check_gpu, current_os_check
from canvas.app.selftest import selftest_report
from canvas.app.updates import check_for_update
from canvas.core.formats import OutputFormat, parse_format
from canvas.safety.image import RawImage
from canvas.safety.inputs import ImageRole, PendingInput
from canvas.infra.enlarge import EnlargeClient, EnlargeError
from canvas.infra.gpu import probe_gpu
from canvas.infra.gpu_lock import prepare_startup_lock
from canvas.infra.manifest import ManifestError
from canvas.infra.models_users import register, unregister
from canvas.infra.paths import models_dir, shared_dir
from canvas.infra.vram import peak_allocated_bytes, set_vram_cap_gb
from canvas.models.loader import ModelNotReady, ZImageGenerator, load_catalog_stack
from canvas.models.views import remove_views
from canvas.models.provision import DownloadDeclined, ensure_plan, install_defaults, install_supporting, repair
from canvas.models.runtime import InferenceUnavailable
from canvas.models.modes import Mode, Tier
from canvas.models.plan import BaseNotSelected, ModeUnavailable, build_plan
from canvas.models.registry import load_catalog
from canvas.models.snapshots import SnapshotError
from canvas.app.photoshoot_bench import live_generate, live_rewrite, run_bench
from canvas.app.photoshoot_cli import model_generate, model_rewrite, reject_frames, run_command
from canvas.pipeline.photoshoot import DEFAULT_FRAMES, MAX_FRAMES
from canvas.ui.styles import LOOKS
from canvas.pipeline.bf16_schedule import VramRefused
from canvas.pipeline.enhance import EnhanceResult, fallback_message, open_rewriter, resolve_prompt, rewrite_text
from canvas.pipeline.schedule import last_image_seconds, reset_image_seconds
from canvas.pipeline.pipeline import CoolingDown, EnlargeMissing, JobError, Pipeline
from canvas.safety.gate import AgeGate, SafetyError
from canvas.ui.main_window import EnhanceReply, MainWindow
from canvas.ui.shell_layout import canvas_state_dir, open_layout
from canvas.ui.shell_screen import ShellWindow
from canvas.ui.splash import Splash, show_splash


_EXIT_CODES = (
    "exit codes:\n"
    "  0  the picture was written\n"
    "  1  the request failed\n"
    "  2  the command or this machine cannot run"
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="voxprint-canvas",
        description=APP_NAME,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_EXIT_CODES,
    )
    parser.add_argument("--register-models", action="store_true", help="register Canvas in the shared resource manifest")
    parser.add_argument("--unregister-models", action="store_true", help="remove Canvas from the shared resource manifest")
    parser.add_argument("--check-update", action="store_true", help="print a newer GitHub release tag, if one exists")
    parser.add_argument("--selftest", action="store_true", help="check the GPU, model files and encoders")
    parser.add_argument("--deep", action="store_true", help="re-hash every model file during --selftest")
    parser.add_argument(
        "--vram-cap-gb",
        type=float,
        default=None,
        help="plan as if the GPU had this many GB (CANVAS_VRAM_CAP_GB)",
    )
    parser.add_argument("--json", action="store_true", help="print a JSON document")
    parser.add_argument("--version", action="store_true", help="print the release identity and exit")
    commands = parser.add_subparsers(dest="command", required=False)
    generate = commands.add_parser(
        "generate",
        help="generate one image and exit",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_EXIT_CODES,
    )
    generate.add_argument("--prompt", default="", help="text prompt for a new picture")
    generate.add_argument("--edit-instruction", default="", help="instruction for editing --input")
    generate.add_argument("--input", default="", help="image for inpaint, used with --mask")
    generate.add_argument("--aspect", default="1:1")
    generate.add_argument("--out", required=True)
    generate.add_argument("--mask", default="", help="mask image for inpaint, used with --input")
    generate.add_argument("--output-size", default="native", choices=("native", "2k", "4k"))
    generate.add_argument("--seed", type=int, default=0)
    generate.add_argument(
        "--enhance",
        action="store_true",
        help="improve the prompt before generating",
    )
    generate.add_argument(
        "--vram-cap-gb",
        type=float,
        default=argparse.SUPPRESS,
        help="plan as if the GPU had this many GB (CANVAS_VRAM_CAP_GB)",
    )
    generate.add_argument("--json", action="store_true")
    bench = commands.add_parser(
        "photoshoot-bench",
        help="time a 6-frame Photoshoot series and exit",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_EXIT_CODES,
    )
    bench.add_argument("--prompt", required=True, help="idea for the 6-frame series")
    bench.add_argument("--out", required=True, help="folder for the series")
    bench.add_argument("--seed", type=int, default=1)
    bench.add_argument("--json", action="store_true")
    photoshoot = commands.add_parser(
        "photoshoot",
        help="write a Photoshoot series and exit",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_EXIT_CODES,
    )
    photoshoot.add_argument("--prompt", required=True, help="theme or idea for the series")
    photoshoot.add_argument(
        "--frames",
        type=int,
        default=DEFAULT_FRAMES,
        help=f"how many frames to write, from 1 to {MAX_FRAMES}",
    )
    photoshoot.add_argument("--out", required=True, help="folder for the frames and scenario.txt")
    photoshoot.add_argument("--seed", type=int, default=1)
    photoshoot.add_argument(
        "--style",
        default="photo",
        choices=tuple(look.style_id for look in LOOKS),
        help="style id, the same ids as the app",
    )
    photoshoot.add_argument(
        "--places",
        default="same",
        choices=("same", "different"),
        help="one place, or a different place for each frame",
    )
    photoshoot.add_argument(
        "--adult",
        action="store_true",
        help="use the Expert adult-content prompt policy",
    )
    photoshoot.add_argument(
        "--vram-cap-gb",
        type=float,
        default=argparse.SUPPRESS,
        help="plan as if the GPU had this many GB (CANVAS_VRAM_CAP_GB)",
    )
    photoshoot.add_argument("--json", action="store_true")
    return parser


def _callback(tier: Tier, window_box: list[MainWindow | None], enlarge: EnlargeClient):
    catalog = load_catalog()

    def on_generate(prompt: str, mode: Mode, fmt: OutputFormat, original: str | None = None) -> str:
        del fmt
        window = window_box[0]
        aspect = "1:1" if window is None else window.current_aspect()
        output = "4k" if window is not None and window.fourk.isChecked() else "native"
        progress = lambda message: _progress(message, window)
        try:
            install_supporting(catalog, progress)
            stack = load_catalog_stack(models_dir(), catalog)
        except (SafetyError, SnapshotError) as exc:
            return str(exc)
        gate = AgeGate(stack)
        active = prompt
        note = ""
        if original is not None:
            try:
                outcome = resolve_prompt(gate, original, prompt)
            except SafetyError as exc:
                return str(exc)
            active = outcome.prompt
            note = fallback_message(outcome)
        pipeline = Pipeline(
            gate, ZImageGenerator(catalog, tier), tier, use_gpu_lock=True, catalog=catalog, enlarge=enlarge,
        )
        try:
            screened = gate.release_request(active, mode.value, ())
            try:
                plan = build_plan(catalog, screened, tier)
            finally:
                screened.wipe()
            ensure_plan(catalog, plan, progress, lambda notice: _confirm(window, notice))
            pipeline.generate(active, mode, aspect=aspect, output=output)
        except (
            SafetyError, SnapshotError, DownloadDeclined, InferenceUnavailable, ModelNotReady,
            ModeUnavailable, BaseNotSelected, JobError, CoolingDown, EnlargeMissing, EnlargeError, VramRefused,
        ) as exc:
            return str(exc)
        if note:
            return note
        return text("status.done")

    def on_enhance(text: str) -> EnhanceReply:
        progress = lambda message: _progress(message, window_box[0])
        install_supporting(catalog, progress)
        stack = load_catalog_stack(models_dir(), catalog)
        gate = AgeGate(stack)
        gate.review_prompt(text)
        rewritten = rewrite_text(open_rewriter(catalog, tier, models_dir(), progress), text)
        outcome = resolve_prompt(gate, text, rewritten)
        return EnhanceReply(outcome.prompt, fallback_message(outcome))

    return on_generate, on_enhance


def _application() -> QApplication:
    """The running application, creating one when startup has not yet done so."""
    current = QApplication.instance()
    if isinstance(current, QApplication):
        return current
    set_app_user_model_id()
    app = QApplication(sys.argv)
    icon = icon_path()
    if icon.is_file():
        app.setWindowIcon(QIcon(str(icon)))
    return app


def _prepare_gate(gate: AgeGate) -> None:
    """Load stack components."""
    try:
        gate.warm()
    except SafetyError:
        return


def _stop_startup(splash: Splash, message: str) -> int:
    """Show ``message`` and leave. The splash alone closes before it can be read."""
    print(message, file=sys.stderr)
    splash.set_stage(message)
    box = QMessageBox()
    box.setIcon(QMessageBox.Icon.Warning)
    box.setWindowTitle(text("app.name"))
    box.setText(message)
    box.exec()
    splash.close()
    return 1


def _launch(tier: Tier, splash: Splash | None = None, app: QApplication | None = None) -> int:
    """Open the shared screen. A gate that cannot be built does not open a window."""
    if app is None:
        app = _application()
    if splash is None:
        splash = show_splash()
    lock_note = prepare_startup_lock()

    def stage(message: str) -> None:
        print(message, file=sys.stderr)
        splash.set_stage(message)

    try:
        install_defaults(load_catalog(), stage)
    except ManifestError:
        lock_note = "busy"
    except (SnapshotError, OSError) as exc:
        stage(str(exc))
    enlarge = EnlargeClient()
    try:
        stack = load_catalog_stack(models_dir(), load_catalog())
        gate = AgeGate(stack)
    except ManifestError:
        return _stop_startup(splash, text("startup.lock"))
    except Exception as exc:
        message = str(exc).strip() or text("gate.unavailable")
        return _stop_startup(splash, message)
    stage(text("canvas.stage.prepare"))
    app.processEvents()
    threading.Thread(target=_prepare_gate, args=(gate,), name="canvas-prepare", daemon=True).start()
    window = ShellWindow(
        open_layout(canvas_state_dir(), gate=gate, enlarge_installed=enlarge.ready()),
        tier=tier,
        enlarge=enlarge,
    )
    window.show()
    if lock_note == "busy":
        window.shell.layout.unavailable(text("startup.lock"))
        window.view.refresh()
    splash.finish(window)
    app.exec()
    return 0


def _emit(payload: dict[str, object], as_json: bool, code: int) -> int:
    if as_json:
        print(json.dumps(payload))
    elif payload.get("error"):
        print(payload["error"], file=sys.stderr)
    elif payload.get("path"):
        print(payload["path"])
    return code


def _load_rgb(path: Path) -> RawImage:
    """Read one RGB image from disk. PNG, JPEG, WebP, and AVIF are accepted as inputs."""
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        return RawImage(rgb.width, rgb.height, rgb.tobytes())


def _pending_init(path: Path) -> PendingInput:
    """An unscreened init image for inpaint."""
    return PendingInput(ImageRole.INIT, _load_rgb(path))


def _inputs(args: argparse.Namespace, mode: Mode) -> tuple[PendingInput, ...]:
    """Images the command supplied. Text has none. Inpaint adds the mask."""
    if mode is Mode.INPAINT:
        return (
            _pending_init(Path(args.input)),
            PendingInput(ImageRole.MASK, _load_rgb(Path(args.mask))),
        )
    return ()


def _generate(args: argparse.Namespace) -> int:
    """Download what is missing, generate one image, and write it."""
    as_json = bool(args.json)
    if str(args.edit_instruction or ""):
        return _instruction_removed(as_json)
    os_result = current_os_check()
    if not os_result.ok:
        return _emit({"ok": False, "error": os_result.message}, as_json, 2)
    gpu = check_gpu(probe_gpu())
    if not gpu.ok or gpu.tier is None:
        return _emit({"ok": False, "error": gpu.message}, as_json, 2)
    path = Path(args.out)
    try:
        fmt = parse_format(path.suffix)
    except ValueError as exc:
        return _emit({"ok": False, "error": str(exc)}, as_json, 2)
    catalog = load_catalog()
    root = models_dir()
    output = args.output_size if args.output_size in {"4k", "2k"} else "native"
    prompt = str(args.prompt or "")
    mask = str(args.mask or "")
    if mask:
        if not prompt or not args.input:
            return _emit({"ok": False, "error": "Inpaint needs a prompt, an input image, and a mask."}, as_json, 2)
        mode = Mode.INPAINT
        active_prompt = prompt
    elif prompt:
        mode = Mode.TEXT
        active_prompt = prompt
    else:
        return _emit({"ok": False, "error": "A prompt is required."}, as_json, 2)
    if args.enhance and mode is not Mode.TEXT:
        return _emit({"ok": False, "error": "Enhance applies to a prompt."}, as_json, 2)
    outcome: EnhanceResult | None = None
    active = active_prompt
    reset_image_seconds()
    progress = lambda message: print(message, file=sys.stderr)
    started = time.perf_counter()
    try:
        source = _inputs(args, mode)
        install_supporting(catalog, progress)
        stack = load_catalog_stack(root, catalog)
        gate = AgeGate(stack)
        gate.warm()
        if args.enhance:
            gate.review_prompt(prompt)
            rewritten = rewrite_text(open_rewriter(catalog, gpu.tier, root, progress), prompt)
            outcome = resolve_prompt(gate, prompt, rewritten)
            active = outcome.prompt
        screened = gate.release_request(active, mode.value, source)
        try:
            plan = build_plan(catalog, screened, gpu.tier)
        finally:
            screened.wipe()
        ensure_plan(catalog, plan, progress, confirm=None)
        generator = ZImageGenerator(catalog, gpu.tier)
        generator.seed = args.seed
        generator.models = root
        pipeline = Pipeline(gate, generator, gpu.tier, use_gpu_lock=True, catalog=catalog, enlarge=EnlargeClient())
        fresh = _inputs(args, mode)
        pipeline.export(active, mode, fresh, path, fmt, aspect=args.aspect, output=output)
    except (
        SafetyError, SnapshotError, DownloadDeclined, InferenceUnavailable, ModelNotReady,
        ModeUnavailable, BaseNotSelected, JobError, CoolingDown, EnlargeMissing, EnlargeError, VramRefused,
        OSError,
    ) as exc:
        return _emit({"ok": False, "error": str(exc)}, as_json, 1)
    if outcome is not None and outcome.fallback and not as_json:
        print(fallback_message(outcome), file=sys.stderr)
    return _emit(
        _generate_payload(path, active_prompt, outcome, time.perf_counter() - started),
        as_json,
        0,
    )


def _photoshoot_bench(args: argparse.Namespace) -> int:
    """Time a 6-frame series on this machine. A missing GPU writes no duration."""
    as_json = bool(args.json)
    os_result = current_os_check()
    if not os_result.ok:
        return _emit({"ok": False, "error": os_result.message}, as_json, 2)
    gpu = check_gpu(probe_gpu())
    if not gpu.ok or gpu.tier is None:
        return _emit({"ok": False, "error": gpu.message}, as_json, 2)
    folder = Path(args.out)
    try:
        install_supporting(load_catalog(), lambda message: print(message, file=sys.stderr))
        stack = load_catalog_stack(models_dir(), load_catalog())
        gate = AgeGate(stack)
        gate.warm()
        report = run_bench(
            prompt=str(args.prompt),
            folder=folder,
            seed=int(args.seed),
            tier=gpu.tier,
            age_gate=gate,
            rewrite=live_rewrite(gpu.tier),
            generate=live_generate(gate, gpu.tier),
        )
    except (
        SafetyError, SnapshotError, DownloadDeclined, InferenceUnavailable, ModelNotReady,
        ModeUnavailable, BaseNotSelected, JobError, CoolingDown, EnlargeMissing, EnlargeError, VramRefused,
        OSError,
    ) as exc:
        return _emit({"ok": False, "error": str(exc)}, as_json, 1)
    return _emit(report, as_json, 0 if report.get("ok") else 1)


def _photoshoot(args: argparse.Namespace) -> int:
    """Write a series into --out. A stopped frame is reported, not a crash."""
    as_json = bool(args.json)
    frames = int(args.frames)
    if frames < 1 or frames > MAX_FRAMES:
        return _emit_photoshoot(reject_frames(frames), as_json, 2)
    os_result = current_os_check()
    if not os_result.ok:
        return _emit({"ok": False, "error": os_result.message}, as_json, 2)
    gpu = check_gpu(probe_gpu())
    if not gpu.ok or gpu.tier is None:
        return _emit({"ok": False, "error": gpu.message}, as_json, 2)
    folder = Path(args.out)
    adult = bool(args.adult)
    style = str(args.style)
    try:
        install_supporting(load_catalog(), lambda message: print(message, file=sys.stderr))
        stack = load_catalog_stack(models_dir(), load_catalog())
        gate = AgeGate(stack)
        gate.warm()
        payload, code = run_command(
            idea=str(args.prompt),
            folder=folder,
            frames=frames,
            seed=int(args.seed),
            style=style,
            places=str(args.places),
            adult=adult,
            age_gate=gate,
            rewrite=model_rewrite(gpu.tier, adult),
            generate=model_generate(gate, gpu.tier, style, adult),
        )
    except (
        SafetyError, SnapshotError, DownloadDeclined, InferenceUnavailable, ModelNotReady,
        ModeUnavailable, BaseNotSelected, JobError, CoolingDown, EnlargeMissing, EnlargeError, VramRefused,
        OSError,
    ) as exc:
        return _emit({"ok": False, "error": str(exc)}, as_json, 1)
    return _emit_photoshoot(payload, as_json, code)


def _emit_photoshoot(payload: dict[str, object], as_json: bool, code: int) -> int:
    if as_json:
        print(json.dumps(payload))
        return code
    if not payload.get("ok"):
        print(_photoshoot_console(payload), file=sys.stderr)
        return code
    frames = payload.get("frames")
    if isinstance(frames, list):
        for path in frames:
            print(path)
    scenario = payload.get("scenario")
    if isinstance(scenario, str) and scenario:
        print(scenario)
    shots = payload.get("shots")
    if isinstance(shots, list):
        for shot in shots:
            if isinstance(shot, dict) and shot.get("code") == "not_produced":
                print(tr("photoshoot.frame.blocked"), file=sys.stderr)
    return code


def _photoshoot_console(payload: dict[str, object]) -> str:
    code = str(payload.get("code") or "")
    if code == "internal":
        return tr("photoshoot.code.internal")
    if code.startswith("check."):
        return text("gate.unavailable")
    if code == "gpu_busy":
        return text("startup.lock")
    if code == "enlarge_unavailable":
        return text("enlarge.hint", name=ENLARGE_NAME)
    if code:
        return tr(f"photoshoot.code.{code}")
    error = str(payload.get("error") or "")
    if error == "bad_frames":
        return tr("photoshoot.frames.invalid", limit=MAX_FRAMES)
    if error == "not_produced":
        return tr("photoshoot.frame.blocked")
    if error == "scenario":
        return tr("photoshoot.scenario.stopped")
    if error == "idea":
        reason = payload.get("message")
        if isinstance(reason, str) and reason.strip():
            return reason
        return tr("photoshoot.scenario.stopped")
    message = payload.get("message")
    if isinstance(message, str) and message.strip():
        return message
    return error


def main(argv: list[str] | None = None) -> int:
    """Register the shared model folder, check for a release, or open the window."""
    args = _parser().parse_args(argv)
    refused = _apply_vram_cap(args)
    if refused is not None:
        return refused
    if args.version:
        print(version_label())
        return 0
    if getattr(args, "command", None) == "generate":
        return _generate(args)
    if getattr(args, "command", None) == "photoshoot-bench":
        return _photoshoot_bench(args)
    if getattr(args, "command", None) == "photoshoot":
        return _photoshoot(args)
    if args.selftest:
        report = selftest_report(deep=bool(args.deep))
        if args.json:
            print(json.dumps(report, indent=2))
        else:
            print("ok" if report["ok"] else "failed")
        return 0 if report["ok"] else 2
    if args.register_models:
        register(shared_dir())
        return 0
    if args.unregister_models:
        unregister(shared_dir())
        # Views are Canvas links over the MDL blobs; the blobs stay with MDL.
        remove_views(models_dir())
        return 0
    if args.check_update:
        tag = check_for_update(APP_VERSION)
        if tag:
            print(tag)
        return 0
    app = _application()
    splash = show_splash()
    os_result = current_os_check()
    if not os_result.ok:
        splash.close()
        print(os_result.message, file=sys.stderr)
        return 2
    splash.set_stage(text("splash.gpu"))
    gpu: GpuCheck = check_gpu(probe_gpu())
    if not gpu.ok or gpu.tier is None:
        splash.close()
        print(gpu.message, file=sys.stderr)
        return 2
    _repair_shared(splash.set_stage)
    code = _launch(gpu.tier, splash, app)
    if code is None:
        return 0
    return int(code)


def _generate_payload(
    path: Path, original: str, outcome: EnhanceResult | None, seconds: float,
) -> dict[str, object]:
    """JSON body for one generate command. Both prompts are present when enhancing."""
    if outcome is None:
        body: dict[str, object] = {
            "ok": True,
            "path": str(path),
            "prompt": original,
            "enhanced_prompt": None,
            "used_prompt": original,
            "fallback": "",
        }
    else:
        body = {
            "ok": True,
            "path": str(path),
            "prompt": outcome.original,
            "enhanced_prompt": outcome.rewritten,
            "used_prompt": outcome.prompt,
            "fallback": outcome.fallback,
        }
    body["seconds_per_image"] = round(float(seconds), 3)
    diffusion = last_image_seconds()
    if diffusion is not None:
        body["diffusion_seconds"] = round(diffusion, 3)
    peak = peak_allocated_bytes()
    if peak is not None:
        body["peak_allocated_bytes"] = peak
    return body


def _progress(message: str, window: MainWindow | None) -> None:
    print(message, file=sys.stderr)
    if window is not None:
        window.status.setText(message)


def _confirm(window: MainWindow | None, notice: str) -> bool:
    """Missing weights are installed in VOXPrint MDL, not from this window."""
    del window, notice
    return False


def _instruction_removed(as_json: bool) -> int:
    """Exit 2. Instruction editing is not part of Canvas."""
    message = text("edit.removed", "en")
    if as_json:
        print(json.dumps({"ok": False, "error": "removed", "message": message}))
    else:
        print(message, file=sys.stderr)
    return 2


def _apply_vram_cap(args: argparse.Namespace) -> int | None:
    """Store ``--vram-cap-gb`` where tier selection and the other cap reader look."""
    cap = getattr(args, "vram_cap_gb", None)
    if cap is None:
        return None
    try:
        set_vram_cap_gb(float(cap))
    except ValueError:
        print("The VRAM cap must be greater than 0.", file=sys.stderr)
        return 2
    return None


def _repair_shared(stage: Callable[[str], None]) -> None:
    def progress(message: str) -> None:
        print(message, file=sys.stderr)
        stage(message)

    try:
        repair(load_catalog(), progress)
    except (SnapshotError, OSError) as exc:
        progress(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
