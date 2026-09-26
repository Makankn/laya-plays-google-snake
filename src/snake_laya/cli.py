from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

from .capture import CaptureRegion, ScreenCapture
from .controller import LiveController
from .parser import ParseError, SnakeFrameParser
from .policy import LayaPolicy, PlannerPolicy
from .region_selector import load_region, select_region
from .tracker import StateTracker


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="snake-state")
    subparsers = parser.add_subparsers(dest="command", required=True)

    parse = subparsers.add_parser("parse", help="parse one screenshot")
    parse.add_argument("image")
    parse.add_argument("--pretty", action="store_true")

    track = subparsers.add_parser("track", help="track consecutive screenshot files")
    track.add_argument("images", nargs="+")
    track.add_argument("--pretty", action="store_true")
    track.add_argument("--initial-length", type=int, default=3)

    select = subparsers.add_parser("select-region", help="drag and resize a capture box over the board")
    select.add_argument("--output", default="capture-region.json")
    select.add_argument("--width", type=int, default=650)
    select.add_argument("--height", type=int, default=580)

    live = subparsers.add_parser("live", help="watch the screen and decide once per new head cell")
    live.add_argument("--region", type=_region, help="capture rectangle: LEFT,TOP,WIDTH,HEIGHT")
    live.add_argument(
        "--region-file",
        default="capture-region.json",
        help="saved selector output; used automatically when it exists",
    )
    live.add_argument("--monitor", type=int, default=1)
    live.add_argument("--fps", type=float, default=60.0)
    live.add_argument("--duration", type=float, help="stop automatically after this many seconds")
    live.add_argument("--log", help="JSONL log path (default: a timestamped file in captures)")
    live.add_argument("--initial-length", type=int, default=3)
    live.add_argument("--planner-only", action="store_true", help="skip Laya while calibrating capture")
    live.add_argument("--execute", action="store_true", help="actually send arrow keys (default is dry-run)")
    live.add_argument("--model", default="convaiinnovations/laya")
    live.add_argument("--subfolder", default="multilingual")
    live.add_argument("--device", choices=("cpu", "cuda"))
    live.add_argument("--overlay", action="store_true", help="show a draggable live-performance overlay")

    train = subparsers.add_parser("train-laya", help="adapt Laya in a simulated Snake environment")
    train.add_argument("--episodes", type=int, default=0)
    train.add_argument("--max-steps", type=int, default=160)
    train.add_argument("--output", default="models/snake-laya")
    train.add_argument("--base-model", default="convaiinnovations/laya")
    train.add_argument("--subfolder", default="multilingual")
    train.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    train.add_argument("--learning-rate", type=float, default=3e-4)
    train.add_argument("--seed", type=int, default=0)
    train.add_argument("--imitation-samples", type=int, default=1200)
    train.add_argument("--imitation-epochs", type=int, default=8)

    evaluate = subparsers.add_parser("eval-laya", help="evaluate a Laya model in simulated Snake")
    evaluate.add_argument("--model", default="models/snake-laya")
    evaluate.add_argument("--subfolder", default="multilingual")
    evaluate.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    evaluate.add_argument("--episodes", type=int, default=10)
    evaluate.add_argument("--max-steps", type=int, default=160)
    evaluate.add_argument("--seed", type=int, default=10000)
    evaluate.add_argument("--validation-samples", type=int, default=200)
    evaluate.add_argument("--trace", action="store_true", help="include the last twelve moves of each simulated game")
    evaluate.add_argument("--input-delay-cells", type=int, choices=(0, 1), default=0, help="simulate a keypress landing one cell late")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    parser = SnakeFrameParser()
    indent = 2 if getattr(args, "pretty", False) else None
    try:
        if args.command == "parse":
            payload = parser.parse_file(args.image).as_dict()
        elif args.command == "track":
            tracker = StateTracker(initial_length=args.initial_length)
            states = []
            observations = []
            for path in args.images:
                observation = parser.parse_file(path)
                observations.append(observation.as_dict())
                state = tracker.update(observation)
                if state is not None:
                    states.append(state.as_dict())
            payload = {
                "observations": observations,
                "states": states,
                "latest_state": None if tracker.state is None else tracker.state.as_dict(),
            }
        elif args.command == "select-region":
            previous = load_region(args.output) if Path(args.output).is_file() else None
            region = select_region(
                args.output,
                initial_width=args.width,
                initial_height=args.height,
                initial_region=previous,
            )
            if region is None:
                print(json.dumps({"status": "cancelled"}))
                return 1
            print(json.dumps({"status": "saved", "path": str(Path(args.output).resolve()), "region": region.as_mss()}))
            return 0
        elif args.command == "train-laya":
            from .training import train_laya

            summary = train_laya(
                episodes=args.episodes,
                max_steps=args.max_steps,
                output=args.output,
                base_model=args.base_model,
                subfolder=args.subfolder,
                device=args.device,
                learning_rate=args.learning_rate,
                seed=args.seed,
                imitation_samples=args.imitation_samples,
                imitation_epochs=args.imitation_epochs,
            )
            print(
                json.dumps(
                    {
                        "status": "training_complete",
                        "episodes": summary.episodes,
                        "steps": summary.total_steps,
                        "apples": summary.total_apples,
                        "best_apples": summary.best_apples,
                        "output": str(summary.output.resolve()),
                    }
                )
            )
            return 0
        elif args.command == "eval-laya":
            from .training import evaluate_laya

            print(
                json.dumps(
                    evaluate_laya(
                        model=args.model,
                        episodes=args.episodes,
                        max_steps=args.max_steps,
                        subfolder=args.subfolder,
                        device=args.device,
                        seed=args.seed,
                        validation_samples=args.validation_samples,
                        trace=args.trace,
                        input_delay_cells=args.input_delay_cells,
                    )
                )
            )
            return 0
        else:
            policy = (
                PlannerPolicy()
                if args.planner_only
                else LayaPolicy(
                    model=args.model,
                    subfolder=args.subfolder,
                    device=args.device,
                    live_timing=True,
                )
            )
            region = args.region
            if region is None:
                previous = load_region(args.region_file) if Path(args.region_file).is_file() else None
                region = select_region(args.region_file, initial_region=previous)
                if region is None:
                    print(json.dumps({"status": "cancelled"}))
                    return 1
            capture = ScreenCapture(region=region, monitor=args.monitor)
            log_path = (
                Path(args.log)
                if args.log
                else Path("captures") / f"live-{datetime.now().strftime('%Y%m%d-%H%M%S')}.jsonl"
            )
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_file = log_path.open("w", encoding="utf-8")

            def emit(line: str) -> None:
                print(line)
                log_file.write(line + "\n")
                log_file.flush()

            emit(json.dumps({"status": "log", "path": str(log_path.resolve())}))
            overlay_process = None
            try:
                if args.overlay:
                    try:
                        overlay_process = subprocess.Popen(
                            [
                                sys.executable,
                                "-m",
                                "snake_laya.overlay",
                                "--log",
                                str(log_path.resolve()),
                                "--avoid-region",
                                f"{region.left},{region.top},{region.width},{region.height}",
                            ],
                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                        )
                    except OSError as error:
                        emit(json.dumps({"status": "overlay_unavailable", "detail": str(error)}))
                LiveController(
                    capture,
                    parser,
                    StateTracker(
                        initial_length=args.initial_length,
                        # Eye whites are already strongly filtered by colour,
                        # component size, nearby blue support, and confidence.
                        # Requiring two identical captures can skip a whole
                        # cell at normal game speed.
                        confirmation_frames=1,
                        minimum_head_confidence=0.55,
                    ),
                    policy,
                    execute=args.execute,
                    fps=args.fps,
                    duration=args.duration,
                    emit=emit,
                ).run()
            except KeyboardInterrupt:
                emit(json.dumps({"status": "stopped"}))
            finally:
                log_file.close()
                if overlay_process is not None and overlay_process.poll() is None:
                    overlay_process.terminate()
                    try:
                        overlay_process.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        overlay_process.kill()
            return 0
    except (OSError, ParseError, ValueError) as error:
        print(json.dumps({"error": str(error)}))
        return 2
    print(json.dumps(payload, indent=indent))
    return 0


def _region(value: str) -> CaptureRegion:
    try:
        left, top, width, height = (int(part.strip()) for part in value.split(","))
    except (ValueError, TypeError) as error:
        raise argparse.ArgumentTypeError("region must be LEFT,TOP,WIDTH,HEIGHT") from error
    if width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError("region width and height must be positive")
    return CaptureRegion(left, top, width, height)


if __name__ == "__main__":
    raise SystemExit(main())
