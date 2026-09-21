"""Command-line entry point. Analysis and rendering are imported on demand."""

from __future__ import annotations

import argparse
import json
import os
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


def _positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError('must be a positive whole number')
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
            cmd.add_argument('--reconstruct', action='store_true', help='opt in to automatic layer-reconstruction candidates after normal correction')
            cmd.add_argument('--allow-ai', action='store_true', help='permit bounded background uploads; requires --reconstruct and OPENAI_API_KEY')
            cmd.add_argument('--max-ai-requests', type=_positive_int, default=1)
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
        if name == 'resume':
            cmd.add_argument('--reconstruct', action='store_true')
            cmd.add_argument('--allow-ai', action='store_true')
            cmd.add_argument('--max-ai-requests', type=_positive_int, default=1)
        if name == 'preview':
            cmd.add_argument('--frame', type=_positive_int, help='render only this enabled seam preview from the current plan')
        if name == 'mark':
            markers = cmd.add_mutually_exclusive_group(required=True)
            markers.add_argument('--seams', type=_seams)
            markers.add_argument('--timecodes')
            markers.add_argument('--clear', action='store_true')

    refine = commands.add_parser('refine', help='refine one seam while preserving a saved plan elsewhere')
    refine.add_argument('input', type=Path, nargs='?', help='original source video; omit with --project')
    refine.add_argument('--project', type=Path, help='saved project with an accepted baseline plan')
    refine.add_argument('--base-plan', type=Path, help='frozen source-conform plan; required with an input video')
    target = refine.add_mutually_exclusive_group(required=True)
    target.add_argument('--frame', type=_positive_int, help='exact zero-based first incoming source frame')
    target.add_argument('--timecode', help='seconds, HH:MM:SS.mmm or HH:MM:SS:FF, rounded to a source frame')
    refine.add_argument('--correction', type=Path, help='JSON correction overrides for the selected seam')
    refine.add_argument('--support-frames', type=_positive_int, help='return support on each side; defaults to the existing seam support')
    refine.add_argument('--work-dir', type=Path, help='new standalone artifact folder; not used with --project')
    refine.add_argument('--output', type=Path, help='optional new full-resolution corrected MP4')
    refine.add_argument('--crf', type=_crf, default=14)

    reconstruct = commands.add_parser('reconstruct', help='prepare, review and accept an isolated layer reconstruction')
    reconstruct.add_argument('input', type=Path, nargs='?', help='new input video, or use --project')
    reconstruct.add_argument('--project', type=Path)
    reconstruct.add_argument('--work-dir', type=Path, help='new project directory when using an input video')
    reconstruct.add_argument('--base-plan', type=Path, help='optional accepted conform plan for a new input project')
    target = reconstruct.add_mutually_exclusive_group()
    target.add_argument('--frame', type=_positive_int)
    target.add_argument('--timecode')
    target.add_argument('--all-seams', action='store_true')
    reconstruct.add_argument('--stage', choices=['propose','edit','background','render','accept','reject','revert','auto','import','segment'], default='auto')
    reconstruct.add_argument('--bundle', type=Path, help='portable reconstruction manifest for --stage import')
    reconstruct.add_argument('--edits', type=Path, help='mask strokes, keyframes or layer adjustments as JSON')
    reconstruct.add_argument('--reach-frames', type=_positive_int)
    reconstruct.add_argument('--motion-strength', type=float)
    reconstruct.add_argument('--segmentation', choices=['auto','classic','neural'], default='auto')
    reconstruct.add_argument('--allow-ai', action='store_true', help='allow source-frame upload for missing background only')
    reconstruct.add_argument('--max-ai-requests', type=_positive_int, default=1)
    reconstruct.add_argument('--quality', choices=['low','medium','high'], default='medium')
    reconstruct.add_argument('--reviewed', action='store_true', help='confirm review when accepting a rendered candidate')
    reconstruct.add_argument('--output', type=Path, help='export the accepted result to a new full movie')
    reconstruct.add_argument('--crf', type=_crf, default=14)
    segment_setup=commands.add_parser('setup-segmentation-model',help='install verified optional local subject-mask weights')
    segment_setup.add_argument('--output',type=Path,help='model cache directory (default: OS cache)')
    segment_setup.add_argument('--source-dir',type=Path,help='import already downloaded weights after checksum verification')
    segment_setup.add_argument('--download',action='store_true',help='authorize the pinned model download')

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


def _refine_command(args):
    """CLI orchestration only; the shared backend owns preservation checks."""
    import copy
    from .corrections import normalize_correction
    from .projects import inspect_source, load_project, set_seams, timecode_to_frame
    from .pipeline import run_stage
    from .refinement import load_baseline, refine_video
    from .repair import fingerprint

    def progress(event):
        print(f"{event.get('stage', 'working')}: {event.get('fraction', 0):.0%} {event.get('message', '')}",
              file=sys.stderr, flush=True)
    if args.project:
        if args.input or args.base_plan or args.work_dir:
            raise ValueError('Use --project alone; input, --base-plan and --work-dir are standalone options')
        project = load_project(args.project)
        source = Path(project['source'])
    else:
        if not args.input or not args.base_plan:
            raise ValueError('Provide either --project or an input video with --base-plan')
        source = _source(args.input)
        project = None
    destination = None
    if args.output:
        destination = _new_output(args.output, source)
        for sidecar in (destination.with_suffix('.repair.json'), destination.with_suffix('.verification.json')):
            if sidecar.exists():raise FileExistsError(f'Refusing to overwrite existing sidecar: {sidecar}')
    overrides = {}
    if args.correction:
        path = _source(args.correction, 'correction')
        with path.open('rb') as handle:data = handle.read(1024*1024+1)
        if len(data) > 1024*1024:raise ValueError('Correction JSON must be smaller than 1 MiB')
        overrides = json.loads(data)
        if not isinstance(overrides, dict):raise ValueError('Correction JSON must contain a correction object')
    if project:
        metadata = project['metadata'];digest = project['sourceSha256']
    else:
        # Preflight the baseline and artifact folder before scanning source timing.
        baseline_path = _source(args.base_plan, 'baseline plan')
        if args.work_dir and args.work_dir.expanduser().resolve().exists():
            raise FileExistsError('Choose a new --work-dir; existing artifact folders are preserved')
        metadata = inspect_source(source, progress=progress)
        digest = fingerprint(source)
    frame = args.frame if args.frame is not None else timecode_to_frame(args.timecode, metadata['fps_fraction'])
    if not 0 < frame < metadata['frame_count']:
        raise ValueError('The selected seam must be a first incoming frame inside the video')
    if project:
        selected = next((row for row in project['seams'] if row['frame'] == frame), None)
        if selected is None or not selected['enabled']:
            raise ValueError('Select an existing enabled project seam; add or enable its marker first')
        if not project['artifacts'].get('plan') and not project.get('refinementBaseline'):
            raise ValueError('Analyze the shot once before refining a single seam; an accepted baseline plan is required')
        choice = normalize_correction({**selected['correction'], **overrides}, metadata=metadata, source_sha256=digest, frame=frame)
        if choice != selected['correction']:
            rows = copy.deepcopy(project['seams'])
            next(row for row in rows if row['frame'] == frame)['correction'] = choice
            project = set_seams(Path(project['projectPath']), rows)
        options = {'frame': frame}
        if args.support_frames is not None:options['supportFrames'] = args.support_frames
        result = run_stage(project['projectPath'], 'refine', options=options, progress=progress)
        if destination:
            result = run_stage(project['projectPath'], 'export',
                               options={'exportPath': str(destination), 'crf': args.crf}, progress=progress)
        return result
    baseline, _ = load_baseline(baseline_path, metadata, digest)
    inherited = next((row['correction'] for row in baseline.get('correction_settings', []) if row.get('frame') == frame), {})
    choice = normalize_correction({**inherited, **overrides}, metadata=metadata, source_sha256=digest, frame=frame)
    folder = (args.work_dir or source.with_name(f'{source.stem}-seam-{frame}.refinement')).expanduser().resolve()
    if folder.exists():raise FileExistsError('Choose a new --work-dir; existing artifact folders are preserved')
    if destination in {folder/'calibration.json', folder/'calibration.plan.json', folder/'calibration.report.json'}:
        raise ValueError('The video output must differ from the refinement JSON artifacts')
    folder.mkdir(parents=True, exist_ok=False)
    result = refine_video(source, frame, baseline_path, folder/'calibration.json', correction=choice,
                          support_frames=args.support_frames, progress=progress)
    if destination:
        from .conform import render_conform
        from .media import probe
        from .pipeline import _audio_hash
        from .projects import atomic_json
        rendered = render_conform(source, result['plan_path'], destination, crf=args.crf, progress=progress)
        actual = probe(destination)
        starts, new_starts = metadata.get('audio_start_times', []), actual.get('audio_start_times', [])
        checks = {key: actual[key] == metadata[key] for key in ('frame_count', 'width', 'height', 'fps_fraction')}
        checks['video_start_time_preserved'] = abs(metadata.get('video_start_time', 0)-actual.get('video_start_time', 0)) < .0002
        checks['audio_start_times_preserved'] = len(starts) == len(new_starts) and all(abs(a-b) < .002 for a, b in zip(starts, new_starts))
        checks['audio_streams_unchanged'] = _audio_hash(source) == _audio_hash(destination) if metadata['has_audio'] else not actual['has_audio']
        verification = {'checks': checks, 'passed': all(checks.values()), 'source': metadata, 'output': actual,
                        'visual_perfection_verified': False, 'render': rendered}
        path = destination.with_suffix('.verification.json');atomic_json(path, verification)
        if not verification['passed']:raise RuntimeError(f'Export verification failed. Inspect {path}')
        result['export_path'] = str(destination);result['verification_path'] = str(path)
    return result


def _reconstruct_command(args):
    from .pipeline import prepare_project,run_stage
    from .projects import load_project,set_seams,timecode_to_frame,atomic_json,save_project
    from .conform import validate_conform_plan
    def progress(event):
        print(f"reconstruct: {event.get('fraction',0):.0%} {event.get('message','')}",file=sys.stderr,flush=True)
    if args.project:
        if args.input or args.work_dir or args.base_plan:
            raise ValueError('Use --project or a new input with --work-dir/--base-plan, not both')
        project=load_project(args.project)
    else:
        if not args.input:raise ValueError('Provide an input video or --project')
        source=_source(args.input)
        if args.output:
            destination=_new_output(args.output,source)
            for suffix in ('.repair.json','.verification.json'):
                if destination.with_suffix(suffix).exists():raise FileExistsError('Output sidecar already exists; choose a new output path')
        project=prepare_project(source,args.work_dir or source.with_suffix('.seamstress'),progress=progress)
    if args.output:
        destination=_new_output(args.output,Path(project['source']))
        for suffix in ('.repair.json','.verification.json'):
            if destination.with_suffix(suffix).exists():raise FileExistsError('Output sidecar already exists; choose a new output path')
    frame=args.frame
    if args.timecode is not None:frame=timecode_to_frame(args.timecode,project['metadata']['fps_fraction'])
    if args.stage=='import':
        if not args.bundle:raise ValueError('--stage import requires --bundle')
        if frame is None:
            data=json.loads(_source(args.bundle,'reconstruction manifest').read_text())
            frame=data.get('frame')
    if args.base_plan:
        from .pipeline import _run_dir
        plan_path=_source(args.base_plan,'accepted plan');recipe=json.loads(plan_path.read_text())
        validate_conform_plan(recipe,project['metadata'])
        if recipe['source_sha256']!=project['sourceSha256']:raise ValueError('Accepted plan belongs to another source')
        seams=[row['frame'] if isinstance(row,dict) else row for row in recipe.get('seams',[])]
        if frame is not None and frame not in seams:seams.append(frame)
        seams.extend(row['frame'] for row in recipe.get('reconstructions',[]))
        corrections={row['frame']:row['correction'] for row in recipe.get('correction_settings',[])}
        project=set_seams(Path(project['projectPath']),[{'frame':n,'correction':corrections.get(n)} for n in sorted(set(seams))])
        target=_run_dir(project,'baseline')/'plan.json';atomic_json(target,recipe)
        project['artifacts']['plan']=str(target);project['status']='analyzed'
        if recipe.get('reconstructions'):
            from .reconstruction_render import FrameReconstruction
            from .reconstruction_workflow import _bind_summary
            FrameReconstruction(project['source'],recipe,project['metadata']).close()
            project['reconstructions']={str(row['frame']):{'accepted':{
                **_bind_summary(row['manifest'],project),'acceptance':'baseline'}} for row in recipe['reconstructions']}
        project=save_project(project)
    if frame is not None and not any(row['frame']==frame for row in project['seams']):
        if args.project:raise ValueError('Mark the requested seam in this project before reconstructing it')
        project=set_seams(Path(project['projectPath']),[*project['seams'],{'frame':frame,'origin':'manual'}])
    if frame is None and args.stage!='auto':raise ValueError('This reconstruction stage requires --frame or --timecode')
    if args.all_seams and args.stage!='auto':raise ValueError('--all-seams is available with --stage auto')
    if not project['seams'] and args.stage=='auto':
        project=run_stage(project['projectPath'],'detect',progress=progress)
    settings={'segmentation':args.segmentation} if args.stage in ('propose','auto') else {}
    if args.edits:
        path=_source(args.edits,'reconstruction edits')
        if path.stat().st_size>8*1024*1024:raise ValueError('Reconstruction edits must be smaller than 8 MiB')
        edits=json.loads(path.read_text())
        if not isinstance(edits,dict):raise ValueError('Reconstruction edits must be an object')
        settings.update(edits)
    if args.reach_frames is not None:settings['reachFrames']=args.reach_frames
    if args.motion_strength is not None:settings['motionStrength']=args.motion_strength
    if args.bundle:settings['manifestPath']=str(_source(args.bundle,'reconstruction manifest').resolve())
    if args.allow_ai:
        if args.stage not in ('background','auto'):raise ValueError('--allow-ai is only used by background or auto stages')
        settings.update(allowAI=True,maxAIRequests=args.max_ai_requests,quality=args.quality)
    if args.reviewed:settings['review_approved']=True
    project=run_stage(project['projectPath'],'reconstruct',
        options={'frame':frame,'action':args.stage,'reconstruction':settings},
        provider_key=os.environ.get('OPENAI_API_KEY') if args.allow_ai else None,progress=progress)
    if args.output:
        selected=[frame] if frame is not None else [row['frame'] for row in project['seams'] if row['enabled']]
        pending=[n for n in selected if project.get('reconstructions',{}).get(str(n),{}).get('candidate')]
        if pending:raise ValueError(f'Candidate needs review at frames {pending}; render, then accept with --reviewed before exporting. Project: {project["projectPath"]}')
        project=run_stage(project['projectPath'],'export',
            options={'exportPath':str(args.output.expanduser().resolve()),'crf':args.crf},progress=progress)
    return project


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command=='setup-segmentation-model':
            from .segmentation_model import setup_model
            print(json.dumps(setup_model(args.output,allow_download=args.download,source_dir=args.source_dir),indent=2))
            return 0
        if args.command=='reconstruct':
            print(json.dumps(_reconstruct_command(args),indent=2,default=_json_default,allow_nan=False))
            return 0
        if args.command == "setup-model":
            from .model_setup import setup_model
            print(json.dumps(setup_model(args.output), indent=2))
            return 0
        if args.command == 'refine':
            print(json.dumps(_refine_command(args), indent=2, default=_json_default, allow_nan=False))
            return 0
        if args.command in ('process', 'detect', 'calibrate', 'preview', 'export', 'resume', 'mark', 'inspect'):
            from .pipeline import prepare_project, run_stage
            from .projects import load_project, set_seams, timecode_to_frame
            def progress(event):
                fraction = event.get('fraction', 0)
                print(f"{event.get('stage', 'working')}: {fraction:.0%} {event.get('message', '')}", file=sys.stderr, flush=True)
            options = {}
            provider_key=None
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
                if args.command == 'preview' and args.frame is not None:options['frame'] = args.frame
                if args.command in ('process', 'resume'):options['export'] = True
                if getattr(args,'allow_ai',False) and not getattr(args,'reconstruct',False):
                    raise ValueError('--allow-ai requires --reconstruct')
                if getattr(args,'reconstruct',False):
                    options['reconstructionEnabled']=True
                    options['reconstruction']={'allowAI':args.allow_ai,'maxAIRequests':args.max_ai_requests}
                    provider_key=os.environ.get('OPENAI_API_KEY') if args.allow_ai else None
                stage = {'calibrate':'analyze', 'resume':'process'}.get(args.command, args.command)
                # Explicit points bypass auto-detection even in the detect command.
                if not (stage == 'detect' and (getattr(args, 'seams', None) or getattr(args, 'timecodes', None))):
                    project = run_stage(project['projectPath'], stage, options=options, progress=progress,provider_key=provider_key)
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
