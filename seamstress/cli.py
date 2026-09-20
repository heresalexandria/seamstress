"""Command-line entry point. Analysis and rendering are imported on demand."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from . import __version__


def _positive_float(value: str) -> float:
    number = float(value)
    if not 0 < number < float("inf"):
        raise argparse.ArgumentTypeError("must be a finite number greater than zero")
    return number


def _crf(value: str) -> int:
    number = int(value)
    if not 0 <= number <= 51:
        raise argparse.ArgumentTypeError("must be an integer from 0 to 51")
    return number


def _seams(value: str) -> list[int]:
    try:
        frames = [int(part.strip()) for part in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("use comma-separated positive frame indices") from exc
    if not frames or any(frame <= 0 for frame in frames):
        raise argparse.ArgumentTypeError("seam indices must be positive; frame 0 is the first frame")
    return sorted(set(frames))


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        prog="seamstress",
        description="Refine AI-generated oners: review and correct seams between sequential generations of one continuous shot.",
        epilog="FFmpeg and ffprobe must be installed and available on PATH. Original media is preserved.",
    )
    result.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    commands = result.add_subparsers(dest="command", required=True)

    for name in ('process', 'detect'):
        cmd = commands.add_parser(name, help='complete source-conform workflow' if name == 'process' else 'scan the entire video for likely seams')
        if name == 'detect':
            cmd.add_argument('input', type=Path, nargs='?')
            cmd.add_argument('--project', type=Path, help='rescan an existing project')
        else:
            cmd.add_argument('input', type=Path)
        cmd.add_argument('--work-dir', type=Path, help='new project directory; defaults to <input>.seamstress')
        cmd.add_argument('--intervals', default='10,15,30', help='soft spacing hints in seconds; empty string disables hints')
        cmd.add_argument('--sensitivity', type=float, default=.5)
        markers = cmd.add_mutually_exclusive_group()
        markers.add_argument('--seams', type=_seams, help='manual first-incoming-frame indices, bypassing detection')
        markers.add_argument('--timecodes', help='comma-separated seconds or HH:MM:SS.mmm / HH:MM:SS:FF')
        if name == 'process':
            cmd.add_argument('--output', type=Path, help='new corrected MP4; default is inside the project')
            cmd.add_argument('--crf', type=_crf, default=14)
            cmd.add_argument('--preview-width', type=int, default=640)
    for name in ('calibrate', 'preview', 'export', 'resume', 'mark', 'inspect'):
        cmd = commands.add_parser(name, help={'calibrate':'analyze marked seams and build a correction plan',
            'preview':'render whole-video and seam previews', 'export':'render a full-resolution corrected video',
            'resume':'run remaining workflow on a saved project', 'mark':'replace seam markers', 'inspect':'print a saved project'}[name])
        cmd.add_argument('--project', type=Path, required=True, help='project directory or project.json')
        if name in ('export', 'resume'):
            cmd.add_argument('--output', type=Path)
            cmd.add_argument('--crf', type=_crf, default=14)
        if name in ('preview', 'resume'):
            cmd.add_argument('--preview-width', type=int, default=640)
        if name == 'mark':
            markers = cmd.add_mutually_exclusive_group(required=True)
            markers.add_argument('--seams', type=_seams)
            markers.add_argument('--timecodes')
            markers.add_argument('--clear', action='store_true')

    def analysis_args(command: argparse.ArgumentParser) -> None:
        command.add_argument("input", type=Path, help="original video")
        command.add_argument("--work-dir", type=Path, default=Path("output"), help="artifact directory (default: output)")
        command.add_argument("--interval", type=_positive_float, default=15.0, help="expected clip length in seconds (default: 15)")
        command.add_argument("--seams", type=_seams, help="explicit seam frame indices, e.g. 360,720; each is the first frame after a cut")

    analyze = commands.add_parser("analyze", help="experimental legacy field-warp analysis (use detect / calibrate)")
    analysis_args(analyze)

    repair = commands.add_parser("repair", help="experimental legacy field-warp renderer")
    repair.add_argument("input", type=Path, help="original video")
    repair.add_argument("--plan", type=Path, required=True, help="saved repair plan JSON")
    repair.add_argument("--output", type=Path, required=True, help="new output video; existing files are refused")
    repair.add_argument("--crf", type=_crf, default=16, help="H.264 quality, 0–51; lower is higher quality (default: 16)")

    verify = commands.add_parser("verify", help="compare original and repaired seam neighborhoods")
    verify.add_argument("input", type=Path, help="original video")
    verify.add_argument("output", type=Path, help="repaired video")
    verify.add_argument("--plan", type=Path, required=True, help="saved repair plan JSON")
    verify.add_argument("--work-dir", type=Path, default=Path("output/verification"), help="verification artifact directory")

    run = commands.add_parser("run", help="experimental legacy analyze/field-warp/verify workflow")
    analysis_args(run)
    run.add_argument("--crf", type=_crf, default=16, help="H.264 quality, 0–51; lower is higher quality (default: 16)")
    bridge = commands.add_parser("bridge", help="experimental neural bridges; rejected for morphing on this source")
    bridge.add_argument("input", type=Path)
    bridge.add_argument("--plan", type=Path, required=True)
    bridge.add_argument("--output", type=Path, required=True)
    bridge.add_argument("--weights", type=Path, default=Path("models/rife425/flownet.pkl"))
    bridge.add_argument("--device", choices=["auto","cpu","mps","cuda"], default="auto")
    bridge.add_argument("--crf", type=_crf, default=16)
    conform = commands.add_parser("conform", help="preserve source drawings with segment grading and global framing correction")
    conform.add_argument("input", type=Path)
    conform.add_argument("--plan", type=Path, required=True)
    conform.add_argument("--output", type=Path, required=True)
    conform.add_argument("--crf", type=_crf, default=14)
    conform.add_argument("--start-frame", type=int, default=0, help="preview start on original timeline (default: 0)")
    conform.add_argument("--end-frame", type=int, help="exclusive preview end; omitted means source end")
    design = commands.add_parser("design-conform", help="rebuild an experimental source-conform plan from reviewed calibration")
    design.add_argument("input", type=Path)
    design.add_argument("--calibration", type=Path, required=True, help="reviewed source-specific geometry and color measurements")
    design.add_argument("--output", type=Path, required=True, help="new schema-3 plan JSON")
    design.add_argument("--geometry-support", type=int, help="geometry return support on each side, in original frames")
    design.add_argument("--rate-support", type=int, help="camera-rate reconciliation support on each side, in original frames")
    setup = commands.add_parser("setup-model", help="download and verify the official RIFE 4.25 checkpoint")
    setup.add_argument("--output", type=Path, default=Path("models/rife425/flownet.pkl"))
    return result


def _source(path: Path, label: str = "input") -> Path:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} file does not exist: {path}")
    return path


def _new_output(path: Path, source: Path) -> Path:
    path = path.expanduser().resolve()
    if path == source:
        raise ValueError("output must differ from the original input")
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {path}")
    return path


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"cannot encode {type(value).__name__} as JSON")


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "setup-model":
            from .model_setup import setup_model
            print(json.dumps(setup_model(args.output), indent=2))
            return 0
        if args.command in ('process', 'detect', 'calibrate', 'preview', 'export', 'resume', 'mark', 'inspect'):
            from .pipeline import prepare_project, run_stage
            from .projects import load_project, set_seams, timecode_to_frame
            def progress(event):
                fraction = event.get('fraction', 0)
                print(f"{event.get('stage', 'working')}: {fraction:.0%} {event.get('message', '')}", file=sys.stderr, flush=True)
            options = {}
            if args.command in ('process', 'detect'):
                if args.command == 'detect' and args.project:
                    if args.input or args.work_dir:raise ValueError('Use either an input video or --project, not both')
                    project = load_project(args.project)
                else:
                    if not args.input:raise ValueError('Provide an input video or --project')
                    source = _source(args.input)
                    folder = args.work_dir or source.with_suffix('.seamstress')
                    project = prepare_project(source, folder, progress=progress)
                options = {'interval_hints':[float(v) for v in args.intervals.split(',') if v.strip()], 'sensitivity':args.sensitivity}
            else:
                project = load_project(args.project)
            if getattr(args, 'seams', None) is not None or getattr(args, 'timecodes', None) is not None or getattr(args, 'clear', False):
                frames = getattr(args, 'seams', None)
                if frames is None:
                    frames = [timecode_to_frame(v, project['metadata']['fps_fraction']) for v in (getattr(args, 'timecodes', None) or '').split(',') if v.strip()]
                project = set_seams(Path(project['projectPath']), frames)
            if args.command not in ('mark', 'inspect'):
                if getattr(args, 'output', None):options['exportPath'] = str(args.output.expanduser().resolve())
                options['crf'] = getattr(args, 'crf', 14)
                options['previewWidth'] = getattr(args, 'preview_width', 640)
                if args.command in ('process', 'resume'):options['export'] = True
                stage = {'calibrate':'analyze', 'resume':'process'}.get(args.command, args.command)
                # Explicit points bypass auto-detection even in the detect command.
                if not (stage == 'detect' and (getattr(args, 'seams', None) or getattr(args, 'timecodes', None))):
                    project = run_stage(project['projectPath'], stage, options=options, progress=progress)
            print(json.dumps(project, indent=2, default=_json_default, allow_nan=False))
            return 0
        source = _source(args.input)
        if args.command == "design-conform":
            from .design import design_conform
            result = design_conform(source, _source(args.calibration, "calibration"), _new_output(args.output, source),
                                    geometry_support=args.geometry_support, rate_support=args.rate_support)
        elif args.command == "conform":
            from .conform import render_conform
            result = render_conform(source, _source(args.plan, "plan"), _new_output(args.output, source), crf=args.crf, start_frame=args.start_frame, end_frame=args.end_frame)
        elif args.command == "bridge":
            from .bridge import render_bridges
            result = render_bridges(source, _source(args.plan, "plan"), _new_output(args.output, source), args.weights, device=args.device, crf=args.crf)
        elif args.command == "analyze":
            from .analysis import analyze

            result = analyze(source, args.work_dir.expanduser().resolve(), args.interval, args.seams)
        elif args.command == "repair":
            from .repair import repair

            plan = _source(args.plan, "plan")
            output = _new_output(args.output, source)
            result = repair(source, plan, output, args.crf)
        elif args.command == "verify":
            from .analysis import verify

            plan = _source(args.plan, "plan")
            output = _source(args.output, "repaired video")
            if output == source:
                raise ValueError("verification requires distinct original and repaired videos")
            result = verify(source, output, plan, args.work_dir.expanduser().resolve())
        else:
            from .analysis import analyze, verify
            from .repair import repair

            work_dir = args.work_dir.expanduser().resolve()
            output = _new_output(work_dir / "repaired.mp4", source)
            analysis = analyze(source, work_dir, args.interval, args.seams)
            plan = Path(analysis.get("plan_path", work_dir / "plan.json"))
            rendered = repair(source, plan, output, args.crf)
            verification = verify(source, output, plan, work_dir / "verification")
            result = {"analysis": analysis, "repair": rendered, "verification": verification}
        print(json.dumps(result, indent=2, default=_json_default, allow_nan=False))
        return 0
    except KeyboardInterrupt:
        print("seamstress: interrupted", file=sys.stderr)
        return 130
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"seamstress: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
