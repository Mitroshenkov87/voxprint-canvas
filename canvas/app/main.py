"""Command line and window startup."""
import argparse
import json
import sys
import time
from collections.abc import Callable
from pathlib import Path

from PIL import Image
from PySide6.QtWidgets import QApplication

from canvas.app.identity import version_label
from canvas.app.info import APP_VERSION
from canvas.app.platform import GpuCheck, check_gpu, current_os_check
from canvas.app.selftest import selftest_report
from canvas.app.updates import check_for_update
from canvas.core.formats import OutputFormat, parse_format
from canvas.safety.image import RawImage
from canvas.safety.inputs import ImageRole, PendingInput
from canvas.infra.enlarge import EnlargeClient, EnlargeError
from canvas.infra.gpu import probe_gpu
from canvas.infra.models_users import register, unregister
from canvas.infra.paths import models_dir, shared_dir
from canvas.infra.vram import peak_allocated_bytes, set_vram_cap_gb
from canvas.models.loader import ModelNotReady, ZImageGenerator, load_catalog_stack
from canvas.models.nunchaku import NunchakuUnavailable
from canvas.models.provision import DownloadDeclined, ensure_plan, install_defaults, install_supporting, repair
from canvas.models.runtime import InferenceUnavailable
from canvas.models.modes import Mode, Tier
from canvas.models.plan import BaseNotSelected, ModeUnavailable, build_plan
from canvas.models.registry import load_catalog
from canvas.models.snapshots import SnapshotError
from canvas.pipeline.bf16_schedule import VramRefused
from canvas.pipeline.enhance import EnhanceResult, fallback_message, open_rewriter, resolve_prompt, rewrite_text
from canvas.pipeline.schedule import SystemMemoryRefused, last_image_seconds, reset_image_seconds
from canvas.pipeline.pipeline import CoolingDown, EnlargeMissing, JobError, Pipeline
from canvas.safety.gate import AgeGate, SafetyError
from canvas.ui.main_window import EnhanceReply, MainWindow
from canvas.ui.splash import Splash, show_splash


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="voxprint-canvas", description="Voxprint AI Canvas")
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
    parser.add_argument(
        "--vram-cap-gb",
        type=float,
        default=None,
        help="plan as if the GPU had this many GB (CANVAS_VRAM_CAP_GB)",
    )
    generate = parser.add_subparsers(dest="command", required=False).add_parser(
        "generate", help="generate one image and exit",
    )
    generate.add_argument("--prompt", default="", help="text prompt for a new picture")
    generate.add_argument("--edit-instruction", default="", help="instruction for editing --input")
    generate.add_argument("--input", default="", help="image to edit, used with --edit-instruction")
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
            SystemMemoryRefused, NunchakuUnavailable,
        ) as exc:
            return str(exc)
        if note:
            return note
        return "Done"

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
    return QApplication(sys.argv)


def _launch(tier: Tier, splash: Splash | None = None, app: QApplication | None = None) -> None:
    if app is None:
        app = _application()
    if splash is None:
        splash = show_splash()

    def stage(message: str) -> None:
        print(message, file=sys.stderr)
        splash.set_stage(message)

    try:
        install_defaults(load_catalog(), stage)
    except (SnapshotError, OSError) as exc:
        stage(str(exc))
    enlarge = EnlargeClient()
    window_box: list[MainWindow | None] = [None]
    on_generate, on_enhance = _callback(tier, window_box, enlarge)
    window = MainWindow(
        on_generate, tier, glass=False, enlarge_ready=enlarge.ready(), on_enhance=on_enhance,
    )
    window_box[0] = window
    window.resize(960, 720)
    window.show()
    splash.finish(window)
    app.exec()


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
    """An unscreened init image for instruction edit."""
    return PendingInput(ImageRole.INIT, _load_rgb(path))


def _inputs(args: argparse.Namespace, mode: Mode) -> tuple[PendingInput, ...]:
    """Images the command supplied. Text has none. Inpaint adds the mask."""
    if mode is Mode.INSTRUCTION:
        return (_pending_init(Path(args.input)),)
    if mode is Mode.INPAINT:
        return (
            _pending_init(Path(args.input)),
            PendingInput(ImageRole.MASK, _load_rgb(Path(args.mask))),
        )
    return ()


def _generate(args: argparse.Namespace) -> int:
    """Download what is missing, generate one image, and write it."""
    as_json = bool(args.json)
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
    instruction = str(args.edit_instruction or "")
    prompt = str(args.prompt or "")
    mask = str(args.mask or "")
    if instruction and (prompt or mask):
        return _emit({"ok": False, "error": "Use a prompt or an edit instruction."}, as_json, 2)
    if instruction:
        if not args.input:
            return _emit({"ok": False, "error": "Instruction edit needs an input image."}, as_json, 2)
        mode = Mode.INSTRUCTION
        text = instruction
    elif mask:
        if not prompt or not args.input:
            return _emit({"ok": False, "error": "Inpaint needs a prompt, an input image, and a mask."}, as_json, 2)
        mode = Mode.INPAINT
        text = prompt
    elif prompt:
        mode = Mode.TEXT
        text = prompt
    else:
        return _emit({"ok": False, "error": "A prompt is required."}, as_json, 2)
    if args.enhance and mode is not Mode.TEXT:
        return _emit({"ok": False, "error": "Enhance applies to a prompt."}, as_json, 2)
    outcome: EnhanceResult | None = None
    active = text
    reset_image_seconds()
    progress = lambda message: print(message, file=sys.stderr)
    started = time.perf_counter()
    try:
        source = _inputs(args, mode)
        install_supporting(catalog, progress)
        stack = load_catalog_stack(root, catalog)
        gate = AgeGate(stack)
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
        SystemMemoryRefused, NunchakuUnavailable, OSError,
    ) as exc:
        return _emit({"ok": False, "error": str(exc)}, as_json, 1)
    if outcome is not None and outcome.fallback and not as_json:
        print(fallback_message(outcome), file=sys.stderr)
    return _emit(_generate_payload(path, text, outcome, time.perf_counter() - started), as_json, 0)


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
    splash.set_stage("Checking the GPU")
    gpu: GpuCheck = check_gpu(probe_gpu())
    if not gpu.ok or gpu.tier is None:
        splash.close()
        print(gpu.message, file=sys.stderr)
        return 2
    _repair_shared(splash.set_stage)
    _launch(gpu.tier, splash, app)
    return 0


def _generate_payload(path: Path, original: str, outcome: EnhanceResult | None, seconds: float) -> dict[str, object]:
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
    if window is None:
        return True
    window.status.setText(notice)
    app = QApplication.instance()
    if app is not None:
        app.processEvents()
    return window.confirm_download(notice)


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
