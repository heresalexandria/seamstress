"""Resumable source-conform workflow shared by the CLI and desktop application."""
from __future__ import annotations
import copy, json, os, subprocess, tempfile, uuid
from pathlib import Path
import cv2
from .projects import create_project, load_project, save_project, set_seams, atomic_json, capture_refinement_baseline
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


def _commit(project, artifacts, status, warnings=None, *, invalidate=(), require_plan=False,seam_results=None, require_refinement_baseline=None):
    current=load_project(Path(project['projectPath']))
    if current['revision']!=project['revision']:
        raise RuntimeError('Seams changed during this job. Run this stage again with the current markers')
    if require_plan and current['artifacts'].get('plan')!=project['artifacts'].get('plan'):
        raise RuntimeError('The correction plan changed during rendering. Run this stage again')
    if require_refinement_baseline is not None:
        import hashlib
        baseline=capture_refinement_baseline(current) or current.get('refinementBaseline')
        baseline_path=Path(require_refinement_baseline['plan'])
        if (baseline!=require_refinement_baseline or not baseline_path.is_file() or
                hashlib.sha256(baseline_path.read_bytes()).hexdigest()!=require_refinement_baseline['planSha256']):
            raise RuntimeError('The accepted baseline changed during refinement. Run this stage again')
    for key in invalidate:current['artifacts'].pop(key,None)
    current['artifacts'].update(artifacts);current['status']=status
    if warnings is not None:current['warnings']=warnings
    if seam_results is not None:current['seamResults']=seam_results
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
    from .corrections import DEFAULT_CORRECTION
    detection_options={k:v for k,v in (options or {}).items() if k in ('interval_hints','sensitivity','min_spacing','scan_width')}
    callback=(lambda event: progress({**event,'stage':'detect'})) if progress else None
    result=detect_seams(Path(project['source']),options=detection_options,progress=callback,cancelled=cancelled)
    _check(cancelled)
    current=load_project(Path(project['projectPath']))
    if current['revision']!=project['revision']:raise RuntimeError('Seams changed during detection; results were not applied')
    existing={row['frame']:row for row in current['seams']}
    rows=[]
    for candidate in result['seams']:
        old=existing.get(candidate['frame'])
        if old:
            candidate={**candidate,**{key:old[key] for key in ('id','enabled','origin','correction')}}
        rows.append(candidate)
    detected={row['frame'] for row in rows}
    # Re-detection may add evidence, but must not discard reviewed choices or
    # move source-bound measurements to a nearby suggested frame.
    rows.extend(row for row in current['seams'] if row['frame'] not in detected and
                (row['origin']=='manual' or not row['enabled'] or row['correction']!=DEFAULT_CORRECTION))
    project=set_seams(Path(project['projectPath']),rows)
    report=_run_dir(project,'detect')/'detection.json';atomic_json(report,result)
    warnings=['Detection suggests boundaries; review markers before your final export.']
    if not result['seams']:warnings=['No new likely seams found. Existing reviewed markers and settings are retained. Add markers manually if a transition is still visible.']
    return _commit(project,{'detection':str(report)},'detected',warnings)


def _seam_results(project,result):
    """Small applied-treatment readout; raw diagnostic fits stay in the report."""
    calibration=result.get('calibration',{})
    cuts={row['frame']:row for row in calibration.get('cuts',[])}
    settings={row['frame']:row['correction'] for row in project['seams']}
    tone={row['frame'] for row in calibration.get('grade_curves',[])}
    local={row['frame'] for row in calibration.get('local_color_curves',[])}
    rows=[]
    for item in result.get('report',{}).get('seams',[]):
        frame=item['frame'];policy=settings[frame];reason=item.get('geometry_excluded_reason');notes=[]
        if policy['geometry']=='off':geometry='off'
        elif reason:geometry='excluded'
        elif policy['geometry']=='manual':geometry='manual'
        elif item.get('partial_geometry',{}).get('accepted'):geometry='partial'
        elif item.get('framing_recovery',{}).get('accepted'):geometry='endpoint'
        else:geometry='auto'
        applied_sides=('pre_rate','post_rate') if item.get('rate_easing') else ('pre_rate',)
        cadence=geometry=='auto' and any(item.get(side,{}).get('cadence_recovery',{}).get('accepted') for side in applied_sides)
        color=item.get('color',{})
        if policy['color']=='off':color_label='off'
        elif frame in tone and frame in local:color_label='tone + local'
        elif frame in tone:color_label='tone'
        elif frame in local:color_label='local'
        else:color_label=color.get('status','unchanged')
        if geometry in ('partial','endpoint'):
            detail=item['partial_geometry' if geometry=='partial' else 'framing_recovery'].get('limitation')
            if detail:notes.append(detail)
        if color.get('status')=='excluded' and policy['color']!='off':notes.append(color.get('reason','Color match was unreliable'))
        row={'frame':frame,'geometry':geometry,'cadence':bool(cadence),'rateEasing':bool(item.get('rate_easing')),
             'color':color_label,'notes':notes}
        if reason:row['geometryReason']=reason
        if geometry not in ('off','excluded') and frame in cuts:
            record=cuts[frame]
            manual={key:copy.deepcopy(record[key]) for key in ('right_to_left_matrix','pre_rate','post_rate','ease_rate')}
            manual['provenance']=(copy.deepcopy(policy['manual'].get('provenance')) if geometry=='manual' and policy['manual'].get('provenance') else
                {'kind':'snapshot','label':'Locked from analyzed framing','frame':frame,'source_sha256':project['sourceSha256']})
            row['manual']=manual
        rows.append(row)
    return rows


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
                           options=calibration_options,progress=callback,cancelled=cancelled,
                           seam_settings={s['frame']:s['correction'] for s in project['seams'] if s['enabled']})
    _check(cancelled)
    plan=json.loads(Path(result['plan_path']).read_text())
    warnings=[]
    exclusions={item['frame']:item['reason'] for item in result.get('calibration',{}).get('excluded_geometry',[])}
    for issue in plan.get('unresolved_seams',[]):
        frame=issue.get('frame') if isinstance(issue,dict) else issue
        reason=issue.get('reason','Review this seam') if isinstance(issue,dict) else exclusions.get(frame,'Review this seam')
        warnings.append(f"Frame {frame}: {reason}")
    for item in result.get('report',{}).get('seams',[]):
        policy=next(s['correction'] for s in project['seams'] if s['frame']==item['frame'])
        partial=item.get('partial_geometry',{})
        if partial.get('accepted') and not item.get('geometry_excluded_reason'):
            warnings.append(f"Frame {item.get('frame','?')}, partial framing: {partial['limitation']}")
        endpoint=item.get('framing_recovery',{})
        if endpoint.get('accepted') and not item.get('geometry_excluded_reason'):
            warnings.append(f"Frame {item.get('frame','?')}, framing restored: {endpoint['limitation']}")
        color=item.get('color',{})
        if color.get('status')=='excluded' and policy['color']!='off':warnings.append(f"Frame {item.get('frame','?')}, color: {color.get('reason','Unreliable color match')}")
    if not frames:warnings.append('No enabled seams: the correction plan preserves the original framing and colors.')
    return _commit(project,{'calibration':str(result['calibration_path']),'plan':str(result['plan_path']),
                            'report':str(result['report_path'])},'analyzed',warnings,
                   invalidate=('fullPreview','seamPreviews','export','verification'),seam_results=_seam_results(project,result))



def refine_project(project, *, options=None,progress=None,cancelled=None):
    """Analyze one seam against a frozen full-shot plan, never recropping it."""
    import hashlib
    from .calibration import DEFAULT_OPTIONS
    from .refinement import refine_video, RefinementError
    options=options or {};frame=options.get('frame')
    if type(frame) is not int:raise ValueError('Refine needs the selected incoming source frame')
    selected=next((row for row in project['seams'] if row['frame']==frame),None)
    if selected is None or not selected['enabled']:
        raise ValueError('Select an enabled seam before refining it')
    baseline=capture_refinement_baseline(project) or project.get('refinementBaseline')
    if not baseline:
        raise RefinementError('Analyze the shot once before refining a single seam; an accepted baseline plan is required')
    if baseline.get('sourceSha256')!=project['sourceSha256']:
        raise RefinementError('The accepted baseline belongs to a different source video')
    path=Path(baseline['plan'])
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=baseline['planSha256']:
        raise RefinementError('The accepted baseline plan is missing or changed; restore it or analyze the whole shot again')
    def others(rows):
        return [(row['frame'],row['enabled'],row['correction']) for row in rows if row['frame']!=frame]
    if others(project['seams'])!=others(baseline['seams']):
        raise RefinementError('Other seam markers or settings changed since the baseline. Restore those changes or analyze the whole shot before refining one seam')
    folder=_run_dir(project,'refine')
    def callback(event):
        if progress:progress({**event,'stage':'refine'})
    result=refine_video(project['source'],frame,path,folder/'calibration.json',
        correction=selected['correction'],analysis_boundaries=[row['frame'] for row in project['seams']],
        support_frames=options.get('supportFrames'),options={key:value for key,value in options.items() if key in DEFAULT_OPTIONS},
        progress=callback,cancelled=cancelled)
    _check(cancelled)
    readout=_seam_results(project,result)
    combined=sorted([copy.deepcopy(row) for row in baseline.get('seamResults',[]) if row['frame']!=frame]+readout,key=lambda row:row['frame'])
    warnings=[warning for warning in baseline.get('warnings',[]) if not warning.startswith(f'Frame {frame}:') and not warning.startswith(f'Frame {frame},')]
    for row in readout:
        if row.get('geometry')=='excluded':warnings.append(f"Frame {frame}: {row.get('geometryReason','Review this seam')}")
        warnings.extend(f'Frame {frame}: {note}' for note in row.get('notes',[]))
    window=result['refinement']
    warnings.append(f"Single-seam refinement at frame {frame}: only frames {window['start_frame']}–{window['end_frame']-1} may differ; the viewing crop and all other corrections are preserved.")
    return _commit(project,{'calibration':str(result['calibration_path']),'plan':str(result['plan_path']),
                            'report':str(result['report_path'])},'analyzed',warnings,
        invalidate=('fullPreview','seamPreviews','export','verification'),seam_results=combined,
        require_refinement_baseline=baseline)


def preview_project(project, *, options=None,progress=None,cancelled=None):
    from .conform import render_conform
    options=options or {}
    if 'frame' in options:
        frame=options['frame']
        if type(frame) is not int:
            raise ValueError('Seam preview needs the selected incoming source frame')
        selected=next((row for row in project['seams'] if row['frame']==frame),None)
        if selected is None or not selected['enabled']:
            raise ValueError('Select an enabled seam before previewing it')
        plan=project['artifacts'].get('plan')
        if not plan or not Path(plan).is_file():
            raise ValueError('Analyze or refine the selected seam before previewing it; a current correction plan is required')
        # A local review renders only its source interval. It must
        # not trigger analysis, rebuild a full proxy, or replace other previews.
        _check(cancelled)
        meta=project['metadata'];radius=max(1,round(float(options.get('previewSeconds',4))*meta['fps']/2))
        start=max(0,frame-radius);end=min(meta['frame_count'],frame+radius)
        folder=_run_dir(project,'preview');clip=folder/f'seam-{frame}.mp4'
        render_conform(project['source'],plan,clip,crf=20,preview_width=int(options.get('previewWidth',640)),
                       start_frame=start,end_frame=end,progress=progress,cancelled=cancelled)
        clip_meta=probe(clip)
        if clip_meta['frame_count']!=end-start or clip_meta['fps_fraction']!=meta['fps_fraction']:
            raise RuntimeError('Seam preview timing verification failed; the clip was not published')
        _check(cancelled)
        previews=[copy.deepcopy(row) for row in project['artifacts'].get('seamPreviews',[]) if row['frame']!=frame]
        previews.append({'frame':frame,'path':str(clip),'startFrame':start,'endFrame':end})
        previews.sort(key=lambda row:row['frame'])
        _emit(progress,'preview',1,'Prepared selected seam preview')
        return _commit(project,{'seamPreviews':previews},'previewed',require_plan=True)
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
    functions={'detect':detect_project,'analyze':analyze_project,'refine':refine_project,'preview':preview_project,'export':export_project}
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
