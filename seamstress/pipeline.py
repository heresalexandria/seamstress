"""Resumable source-conform workflow shared by the CLI and desktop application."""
from __future__ import annotations
import json, os, subprocess, tempfile, uuid
from pathlib import Path
import cv2
from .projects import create_project, load_project, save_project, set_seams, atomic_json
from .media import probe, iter_frames, _tool, _run


def _check(cancelled):
    if cancelled and cancelled(): raise InterruptedError('Operation cancelled')


def _emit(progress, stage, fraction, message):
    if progress: progress({'stage':stage,'fraction':max(0.,min(1.,fraction)),'message':message})


def _command(args, *, duration=1., progress=None, stage='import', cancelled=None):
    """Run FFmpeg with incremental progress; never expose a partial file as an artifact."""
    with tempfile.TemporaryFile() as errors:
        child=subprocess.Popen([_tool('ffmpeg'),'-v','error','-nostdin','-n','-progress','pipe:1',*args],
                               stdout=subprocess.PIPE,stderr=errors,text=True)
        try:
            for line in child.stdout:
                _check(cancelled)
                if line.startswith('out_time_us='):
                    try: fraction=int(line.split('=')[1])/1e6/max(duration,.001)
                    except ValueError: continue
                    _emit(progress,stage,fraction,'Preparing playback media')
            status=child.wait()
            if status:
                errors.seek(0)
                raise RuntimeError(errors.read().decode(errors='replace').strip() or 'FFmpeg failed')
        finally:
            if child.poll() is None:child.terminate();child.wait()
            child.stdout.close()
    _check(cancelled)


def _commit(project, artifacts, status, warnings=None, *, invalidate=(), require_plan=False):
    current=load_project(Path(project['projectPath']))
    if current['revision']!=project['revision']:
        raise RuntimeError('Seams changed during this job. Run this stage again with the current markers')
    if require_plan and current['artifacts'].get('plan')!=project['artifacts'].get('plan'):
        raise RuntimeError('The correction plan changed during rendering. Run this stage again')
    for key in invalidate:current['artifacts'].pop(key,None)
    current['artifacts'].update(artifacts);current['status']=status
    if warnings is not None:current['warnings']=warnings
    return save_project(current)


def _run_dir(project, stage):
    directory=Path(project['projectPath']).parent/'artifacts'/f"r{project['revision']}-{stage}-{uuid.uuid4().hex[:8]}"
    directory.mkdir(parents=True)
    return directory


def prepare_project(source, folder, *, progress=None,cancelled=None):
    project=create_project(Path(source),Path(folder),progress=progress,cancelled=cancelled)
    return prepare_media(project,progress=progress,cancelled=cancelled)


def prepare_media(project, *, progress=None,cancelled=None):
    if project['artifacts'].get('proxy') and Path(project['artifacts']['proxy']).is_file():return project
    folder=_run_dir(project,'source');proxy=folder/'source.mp4';meta=project['metadata']
    width=min(meta['width'],960)//2*2;height=max(2,round(meta['height']*width/meta['width']/2)*2)
    _command(['-i',project['source'],'-map',f"0:{meta.get('video_stream_index',0)}",'-map','0:a:0?',
              '-vf',f'scale={width}:{height}', '-c:v','libx264','-preset','veryfast','-crf','20',
              '-pix_fmt','yuv420p','-fps_mode','passthrough','-c:a','aac','-b:a','160k',
              '-movflags','+faststart',str(proxy)],duration=meta['frame_count']/meta['fps'],
              progress=progress,cancelled=cancelled)
    import numpy as np
    indices=set(np.linspace(0,meta['frame_count']-1,min(18,meta['frame_count']),dtype=int).tolist());thumbs=[]
    thumb_height=max(2,round(meta['height']*160/meta['width']))
    for number,frame in enumerate(iter_frames(proxy,size=(160,thumb_height))):
        _check(cancelled)
        if number in indices:
            path=folder/f'thumb-{number}.jpg'
            if not cv2.imwrite(str(path),cv2.cvtColor(frame,cv2.COLOR_RGB2BGR),[cv2.IMWRITE_JPEG_QUALITY,84]):
                raise RuntimeError('Could not write timeline thumbnail')
            thumbs.append({'frame':number,'time':number/meta['fps'],'path':str(path)})
    return _commit(project,{'proxy':str(proxy),'thumbnails':thumbs},project['status'])


def detect_project(project, *, options=None,progress=None,cancelled=None):
    from .detection import detect_seams
    detection_options={k:v for k,v in (options or {}).items() if k in ('interval_hints','sensitivity','min_spacing','scan_width')}
    callback=(lambda event: progress({**event,'stage':'detect'})) if progress else None
    result=detect_seams(Path(project['source']),options=detection_options,progress=callback,cancelled=cancelled)
    _check(cancelled)
    current=load_project(Path(project['projectPath']))
    if current['revision']!=project['revision']:raise RuntimeError('Seams changed during detection; results were not applied')
    project=set_seams(Path(project['projectPath']),result['seams'])
    report=_run_dir(project,'detect')/'detection.json';atomic_json(report,result)
    warnings=['Detection suggests boundaries; review markers before your final export.']
    if not result['seams']:warnings=['No likely seams found. Add markers manually if a transition is still visible.']
    return _commit(project,{'detection':str(report)},'detected',warnings)


def analyze_project(project, *, options=None,progress=None,cancelled=None):
    from .calibration import calibrate_video, DEFAULT_OPTIONS
    frames=[s['frame'] for s in project['seams'] if s['enabled']]
    folder=_run_dir(project,'analysis')
    calibration_options={k:v for k,v in (options or {}).items() if k in DEFAULT_OPTIONS}
    latest_fraction=0.
    def callback(event):
        nonlocal latest_fraction
        stage=event.get('stage');fraction=event.get('fraction',0)
        base,span={'fingerprint':(0,.02),'decode':(.02,.23),'geometry':(.25,.3),
                   'color':(.55,.42),'publish':(.98,.01),'complete':(1,0)}.get(stage,(0,1))
        latest_fraction=max(latest_fraction,min(1,base+span*fraction))
        if progress:progress({**event,'stage':'analyze','fraction':latest_fraction})
    result=calibrate_video(Path(project['source']),frames,folder/'calibration.json',
                           options=calibration_options,progress=callback,cancelled=cancelled)
    _check(cancelled)
    plan=json.loads(Path(result['plan_path']).read_text())
    warnings=[]
    exclusions={item['frame']:item['reason'] for item in result.get('calibration',{}).get('excluded_geometry',[])}
    for issue in plan.get('unresolved_seams',[]):
        frame=issue.get('frame') if isinstance(issue,dict) else issue
        reason=issue.get('reason','Review this seam') if isinstance(issue,dict) else exclusions.get(frame,'Review this seam')
        warnings.append(f"Frame {frame}: {reason}")
    for item in result.get('report',{}).get('seams',[]):
        partial=item.get('partial_geometry',{})
        if partial.get('accepted') and not item.get('geometry_excluded_reason'):
            warnings.append(f"Frame {item.get('frame','?')}, partial framing: {partial['limitation']}")
        color=item.get('color',{})
        if color.get('status')=='excluded':warnings.append(f"Frame {item.get('frame','?')}, color: {color.get('reason','Unreliable color match')}")
    if not frames:warnings.append('No enabled seams: the correction plan preserves the original framing and colors.')
    return _commit(project,{'calibration':str(result['calibration_path']),'plan':str(result['plan_path']),
                            'report':str(result['report_path'])},'analyzed',warnings,
                   invalidate=('fullPreview','seamPreviews','export','verification'))


def preview_project(project, *, options=None,progress=None,cancelled=None):
    from .conform import render_conform
    options=options or {}
    project=prepare_media(project,progress=progress,cancelled=cancelled)
    if not project['artifacts'].get('plan'):project=analyze_project(project,progress=progress,cancelled=cancelled)
    folder=_run_dir(project,'preview');video=folder/'corrected.mp4'
    width=int(options.get('previewWidth',640))
    render_conform(project['source'],project['artifacts']['plan'],video,crf=20,preview_width=width,
                   progress=progress,cancelled=cancelled)
    previews=[];meta=project['metadata'];radius=max(1,round(float(options.get('previewSeconds',4))*meta['fps']/2))
    frames=[s['frame'] for s in project['seams'] if s['enabled']]
    for i,frame in enumerate(frames):
        _check(cancelled);start=max(0,frame-radius);end=min(meta['frame_count'],frame+radius)
        clip=folder/f'seam-{frame}.mp4'
        args=['-i',str(video),'-i',project['source'],'-map','0:v:0','-map','1:a:0?',
              '-vf',f'trim=start_frame={start}:end_frame={end},setpts=PTS-STARTPTS',
              '-c:v','libx264','-preset','veryfast','-crf','18','-r',meta['fps_fraction'],'-fps_mode','cfr']
        if meta['has_audio']:args+=['-af',f'atrim=start={start/meta["fps"]}:end={end/meta["fps"]},asetpts=PTS-{start/meta["fps"]}/TB','-c:a','aac']
        args+=['-movflags','+faststart',str(clip)]
        _command(args,duration=(end-start)/meta['fps'],cancelled=cancelled,stage='preview')
        clip_meta=probe(clip)
        if clip_meta['frame_count']!=end-start or clip_meta['fps_fraction']!=meta['fps_fraction']:
            raise RuntimeError('Seam preview timing verification failed; the clip was not published')
        previews.append({'frame':frame,'path':str(clip),'startFrame':start,'endFrame':end})
        _emit(progress,'preview',1,f'Prepared seam preview {i+1} / {len(frames)}')
    return _commit(project,{'fullPreview':str(video),'seamPreviews':previews},'previewed',require_plan=True)


def _audio_hash(path):
    return _run([_tool('ffmpeg'),'-v','error','-i',str(path),'-map','0:a?','-c','copy','-f','streamhash','-hash','sha256','-']).stdout.decode()


def export_project(project, *, options=None,progress=None,cancelled=None):
    from .conform import render_conform
    options=options or {}
    if not project['artifacts'].get('plan'):project=analyze_project(project,progress=progress,cancelled=cancelled)
    folder=_run_dir(project,'export')
    output=Path(options.get('exportPath') or folder/(project['name']+'-corrected.mp4')).expanduser().resolve()
    report=render_conform(project['source'],project['artifacts']['plan'],output,crf=int(options.get('crf',14)),
                          progress=progress,cancelled=cancelled)
    _emit(progress,'export',1,'Checking frame count, dimensions, frame rate and original audio')
    meta=probe(output);original=project['metadata']
    checks={key:meta[key]==original[key] for key in ('frame_count','width','height','fps_fraction')}
    starts=original.get('audio_start_times',[]);new_starts=meta.get('audio_start_times',[])
    checks['audio_start_times_preserved']=len(starts)==len(new_starts) and all(abs(a-b)<.002 for a,b in zip(starts,new_starts))
    checks['video_start_time_preserved']=abs(original.get('video_start_time',0)-meta.get('video_start_time',0))<.0002
    checks['audio_streams_unchanged']=_audio_hash(project['source'])==_audio_hash(output) if original['has_audio'] else not meta['has_audio']
    verification={'checks':checks,'passed':all(checks.values()),'source':original,'output':meta,
                  'visual_perfection_verified':False,'render':report}
    path=output.with_suffix('.verification.json');atomic_json(path,verification)
    if not verification['passed']:raise RuntimeError(f'Export verification failed. Inspect {path}')
    _check(cancelled)
    return _commit(project,{'export':str(output),'verification':str(path)},'exported',require_plan=True)


def run_stage(path, stage, *, options=None,progress=None,cancelled=None):
    project=load_project(Path(path));options=options or {};_check(cancelled)
    functions={'detect':detect_project,'analyze':analyze_project,'preview':preview_project,'export':export_project}
    if stage in functions:return functions[stage](project,options=options,progress=progress,cancelled=cancelled)
    if stage!='process':raise ValueError(f'Unknown workflow stage: {stage}')
    project=prepare_media(project,progress=progress,cancelled=cancelled)
    # Preserve reviewed/manual markers when a saved project is run again.
    if project['status']=='imported' and not project['seams'] and not project['artifacts'].get('detection'):
        project=detect_project(project,options=options,progress=progress,cancelled=cancelled)
    if not project['artifacts'].get('plan'):
        project=analyze_project(project,options=options,progress=progress,cancelled=cancelled)
    if not project['artifacts'].get('fullPreview'):
        project=preview_project(project,options=options,progress=progress,cancelled=cancelled)
    if options.get('exportPath') or options.get('export',False):
        project=export_project(project,options=options,progress=progress,cancelled=cancelled)
    return project
