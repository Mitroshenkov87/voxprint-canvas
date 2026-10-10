"""Command line and window startup."""
import argparse
import sys

from PySide6.QtWidgets import QApplication

from canvas.app.identity import version_label
from canvas.app.info import APP_VERSION
from canvas.app.platform import GpuCheck, check_gpu, current_os_check
from canvas.app.updates import check_for_update
from canvas.core.formats import OutputFormat
from canvas.infra.enlarge import EnlargeClient
from canvas.infra.gpu import probe_gpu
from canvas.infra.models_users import register, unregister
from canvas.infra.paths import models_dir
from canvas.models.loader import ZImageGenerator, load_catalog_stack
from canvas.models.modes import Mode, Tier
from canvas.models.plan import BaseNotSelected, ModeUnavailable
from canvas.models.registry import load_catalog
from canvas.pipeline.pipeline import CoolingDown, EnlargeMissing, JobError, Pipeline
from canvas.safety.gate import AgeGate, SafetyError
from canvas.ui.main_window import MainWindow
from canvas.ui.splash import show_splash


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="voxprint-canvas", description="Voxprint AI Canvas")
    parser.add_argument("--register-models", action="store_true", help="mark Canvas as a user of the shared models folder")
    parser.add_argument("--unregister-models", action="store_true", help="remove Canvas from the shared models folder")
    parser.add_argument("--check-update", action="store_true", help="print a newer GitHub release tag, if one exists")
    parser.add_argument("--selftest", action="store_true", help="import the app and exit")
    parser.add_argument("--version", action="store_true", help="print the release identity and exit")
    return parser


def _callback(tier: Tier, window_box: list[MainWindow | None], enlarge: EnlargeClient):
    catalog = load_catalog()

    def on_generate(prompt: str, mode: Mode, fmt: OutputFormat) -> str:
        del fmt
        window = window_box[0]
        aspect = "1:1" if window is None else window.current_aspect()
        output = "4k" if window is not None and window.fourk.isChecked() else "native"
        try:
            stack = load_catalog_stack(models_dir(), catalog)
        except SafetyError as exc:
            return str(exc)
        pipeline = Pipeline(
            AgeGate(stack), ZImageGenerator(catalog, tier), tier, use_gpu_lock=True, catalog=catalog, enlarge=enlarge,
        )
        try:
            pipeline.generate(prompt, mode, aspect=aspect, output=output)
        except (SafetyError, ModeUnavailable, BaseNotSelected, JobError, CoolingDown, EnlargeMissing) as exc:
            return str(exc)
        return "Done"

    return on_generate


def _launch(tier: Tier) -> None:
    app = QApplication.instance() or QApplication(sys.argv)
    splash = show_splash()
    enlarge = EnlargeClient()
    window_box: list[MainWindow | None] = [None]
    window = MainWindow(_callback(tier, window_box, enlarge), tier, glass=False, enlarge_ready=enlarge.installed())
    window_box[0] = window
    window.resize(960, 720)
    window.show()
    splash.finish(window)
    app.exec()


def main(argv: list[str] | None = None) -> int:
    """Register the shared model folder, check for a release, or open the window."""
    args = _parser().parse_args(argv)
    if args.version:
        print(version_label())
        return 0
    if args.selftest:
        return 0
    if args.register_models:
        register(models_dir())
        return 0
    if args.unregister_models:
        unregister(models_dir())
        return 0
    if args.check_update:
        tag = check_for_update(APP_VERSION)
        if tag:
            print(tag)
        return 0
    os_result = current_os_check()
    if not os_result.ok:
        print(os_result.message, file=sys.stderr)
        return 2
    gpu: GpuCheck = check_gpu(probe_gpu())
    if not gpu.ok or gpu.tier is None:
        print(gpu.message, file=sys.stderr)
        return 2
    _launch(gpu.tier)
    return 0
