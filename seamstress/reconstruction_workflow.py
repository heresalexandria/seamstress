"""Candidate/acceptance workflow shared by the desktop app and CLI.

Reconstruction never replaces the accepted plan while it is being proposed,
edited, generated, or previewed. Only explicit acceptance (or a supported
automatic eligibility decision) adds a bounded native-frame provider.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

from .projects import atomic_json, load_project, save_project
from .reconstruction_render import context_signature, entry_for_manifest, validate_entries


ACTIONS = {'propose', 'edit', 'background', 'render', 'accept', 'reject', 'revert', 'auto', 'import', 'segment'}


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _check(cancelled):
    if cancelled and cancelled():
        raise InterruptedError('Reconstruction cancelled')


def _plan(project, *, allow_frozen=False):
    value = project['artifacts'].get('plan')
    if not value and allow_frozen:
        baseline=project.get('refinementBaseline',{})
        value=baseline.get('plan')
        if value and _sha(value)!=baseline.get('planSha256'):
            raise ValueError('The frozen reconstruction baseline changed')
    if not value or not Path(value).is_file():
        raise ValueError('Analyze the shot before reconstructing a seam; a saved baseline is required')
    path = Path(value)
    data = json.loads(path.read_text())
    from .conform import validate_conform_plan
    validate_conform_plan(data, project['metadata'])
    if data['source_sha256'] != project['sourceSha256']:
        raise ValueError('The reconstruction baseline belongs to a different source')
    return path, data


def _selected(project, frame):
    if type(frame) is not int:
        raise ValueError('Reconstruction needs an incoming source frame')
    seam = next((row for row in project['seams'] if row['frame'] == frame), None)
    if seam is None or not seam['enabled']:
        raise ValueError('Select an enabled seam before reconstructing it')
    return seam


def _current_candidate(project, frame):
    result = project.get('reconstructions', {}).get(str(frame), {}).get('candidate')
    if not isinstance(result, dict) or not result.get('manifestPath'):
        raise ValueError('Prepare or import a reconstruction candidate first')
    path, recipe = _plan(project)
    if result.get('baselinePlanSha256') != _sha(path):
        prior=Path(result.get('baselinePlanPath',''))
        if not prior.is_file() or _sha(prior)!=result.get('baselinePlanSha256'):
            raise ValueError('The candidate baseline is missing or changed; propose a new candidate')
        from .reconstruction import load_bundle
        bundle=load_bundle(result['manifestPath'])
        start,end=bundle['support']['start'],bundle['support']['end']
        if context_signature(json.loads(prior.read_text()),start,end)!=context_signature(recipe,start,end):
            raise ValueError('The accepted framing or grade changed after this candidate was prepared; propose a new candidate')
        result={**result,'baselinePlanPath':str(path.resolve()),'baselinePlanSha256':_sha(path)}
    if _sha(result['manifestPath']) != result.get('manifestSha256'):
        raise ValueError('The reconstruction candidate changed outside the project')
    return result


def _editable_candidate(project, frame):
    record=project.get('reconstructions',{}).get(str(frame),{})
    if record.get('candidate'):
        return _current_candidate(project,frame)
    if record.get('accepted'):
        return _bind_summary(record['accepted']['manifestPath'],project)
    raise ValueError('Prepare or import a reconstruction candidate first')


def _bind_summary(manifest, project, **extra):
    from .reconstruction import summary, load_bundle
    path, recipe = _plan(project)
    manifest = Path(manifest).resolve()
    bundle = load_bundle(manifest)
    source = bundle['source']
    if source['sha256'] != project['sourceSha256']:
        raise ValueError('Reconstruction belongs to a different source')
    if any(source[k] != project['metadata'][k] for k in ('width', 'height', 'fps_fraction', 'frame_count')):
        raise ValueError('Reconstruction dimensions or timing differ from this project')
    frame = bundle['frame']
    _selected(project, frame)
    start, end = bundle['support']['start'], bundle['support']['end']
    for seam in project['seams']:
        if seam['frame'] != frame and start <= seam['frame'] < end:
            raise ValueError('Reconstruction support reaches another marked seam; shorten its reach')
    for row in recipe.get('reconstructions', []):
        if row['frame'] != frame and start < row['end'] and row['start'] < end:
            raise ValueError('Reconstruction support overlaps another accepted layer repair')
    baseline = bundle.get('baseline')
    if baseline:
        from .reconstruction.bundle import asset_path
        prior_path = asset_path(bundle,baseline['asset']) if baseline.get('asset') else Path(baseline['path'])
        if not prior_path.is_file() or _sha(prior_path) != baseline['sha256']:
            raise ValueError('The reconstruction bundle baseline is missing or changed')
        original = json.loads(prior_path.read_text())
        if context_signature(original, start, end) != context_signature(recipe, start, end):
            raise ValueError('Imported reconstruction has different framing or grading in its support window')
    return {**summary(manifest), 'manifestPath': str(manifest), 'manifestSha256': _sha(manifest),
            'baselinePlanPath': str(path.resolve()), 'baselinePlanSha256': _sha(path), **extra}


def _publish(snapshot, updated, *, cancelled=None):
    _check(cancelled)
    current = load_project(Path(snapshot['projectPath']))
    if current['revision'] != snapshot['revision']:
        raise RuntimeError('Seam settings changed during reconstruction; the candidate was not applied')
    if current.get('reconstructionRevision', 0) != snapshot.get('reconstructionRevision', 0):
        raise RuntimeError('A newer reconstruction job completed; this result was not applied')
    old_plan = snapshot['artifacts'].get('plan')
    if current['artifacts'].get('plan') != old_plan:
        raise RuntimeError('The accepted plan changed during reconstruction')
    frozen=snapshot.get('_reconstructionFrozenPlan')
    if frozen and current.get('refinementBaseline')!=snapshot.get('refinementBaseline'):
        raise RuntimeError('The frozen baseline changed during reconstruction')
    if _sha(old_plan or frozen) != snapshot['_reconstructionPlanSha256']:
        raise RuntimeError('The accepted plan was modified during reconstruction')
    current['reconstructions'] = updated.get('reconstructions', {})
    current['reconstructionRevision'] = current.get('reconstructionRevision', 0) + 1
    # A concurrent preview may complete without changing the plan. Retain its
    # unrelated artifacts instead of restoring our older snapshot wholesale.
    for key in snapshot['artifacts'].keys()-updated['artifacts'].keys():
        current['artifacts'].pop(key,None)
    current['artifacts'].update({key:value for key,value in updated['artifacts'].items()
                                if value!=snapshot['artifacts'].get(key)})
    if '_reconstructionInvalidateFrame' in updated:
        _invalidate_selected(current,updated['_reconstructionInvalidateFrame'])
        current.pop('_reconstructionInvalidateFrame',None)
    if updated['status']!=snapshot['status'] or '_reconstructionInvalidateFrame' in updated:
        current['status'] = updated['status']
    if updated.get('warnings')!=snapshot.get('warnings'):
        current['warnings'] = updated.get('warnings', [])
    if frozen:
        current['refinementBaseline']=updated['refinementBaseline']
    return save_project(current)


def _history(record, value, disposition):
    if value:
        record['history'] = [{**copy.deepcopy(value), 'disposition': disposition}, *record.get('history', [])][:12]


def _replace_plan(recipe, manifest, frame):
    result = copy.deepcopy(recipe)
    rows = [row for row in result.get('reconstructions', []) if row['frame'] != frame]
    if manifest is not None:
        rows.append(entry_for_manifest(manifest, result))
    if rows:
        result['reconstructions'] = sorted(rows, key=lambda row: row['start'])
    else:
        result.pop('reconstructions', None)
    validate_entries(result, result['source'])
    return result


def _invalidate_selected(project, frame):
    project['_reconstructionInvalidateFrame']=frame
    for key in ('fullPreview', 'export', 'verification'):
        project['artifacts'].pop(key, None)
    if 'seamPreviews' in project['artifacts']:
        project['artifacts']['seamPreviews'] = [row for row in project['artifacts']['seamPreviews'] if row['frame'] != frame]


def preserve_accepted_reconstructions(previous, next_recipe):
    """A new global analysis may keep only repairs with identical local context."""
    if not previous.get('reconstructions'):
        return next_recipe
    result = copy.deepcopy(next_recipe)
    result['reconstructions'] = copy.deepcopy(previous['reconstructions'])
    try:
        validate_entries(result, result['source'])
    except ValueError as exc:
        raise ValueError('This analysis changes an accepted layer repair. Revert that reconstruction before changing its framing or grade') from exc
    return result


def reconstruct_project(project, *, options=None, provider_key=None, progress=None, cancelled=None, budget=None):
    from . import reconstruction as engine
    from .pipeline import _run_dir
    options = options or {}
    action = options.get('action', 'propose')
    if action not in ACTIONS:
        raise ValueError('Unknown reconstruction action')
    settings = options.get('reconstruction', {})
    if not isinstance(settings, dict):
        raise ValueError('Reconstruction settings must be an object')
    frame = options.get('frame')
    if frame is None and action == 'auto':
        if settings.get('allowAI'):
            from .reconstruction_cloud import RequestBudget
            budget = budget or RequestBudget(settings.get('maxAIRequests', 1),
                ledger_path=_run_dir(project, 'reconstruction-budget')/'requests.json')
        for seam in list(project['seams']):
            if not seam['enabled'] or project.get('reconstructions', {}).get(str(seam['frame']), {}).get('accepted'):
                continue
            try:
                project = reconstruct_project(project, options={**options, 'frame': seam['frame']},
                    provider_key=provider_key, progress=progress, cancelled=cancelled, budget=budget)
            except ValueError as exc:
                # Unsupported local geometry/window is an abstention, not a
                # reason to lose completed candidates at other seams. Transport,
                # cancellation and concurrent-state failures still stop the job.
                path,_=_plan(project)
                snapshot=copy.deepcopy(project)
                snapshot['_reconstructionPlanSha256']=_sha(path)
                updated=copy.deepcopy(project)
                updated['warnings']=[*updated.get('warnings',[]),
                    f"Frame {seam['frame']}: reconstruction was not applied: {exc}"]
                project=_publish(snapshot,updated,cancelled=cancelled)
        return project
    _selected(project, frame)
    plan_path, recipe = _plan(project,allow_frozen=action in ('revert','reject'))
    snapshot = copy.deepcopy(project)
    snapshot['_reconstructionPlanSha256'] = _sha(plan_path)
    if not project['artifacts'].get('plan'):
        snapshot['_reconstructionFrozenPlan']=str(plan_path)
    updated = copy.deepcopy(project)
    record = updated.setdefault('reconstructions', {}).setdefault(str(frame), {})
    _check(cancelled)
    folder = _run_dir(project, 'reconstruction-'+action)
    def emit(message, fraction=0.):
        if progress:
            progress({'stage': 'reconstruct', 'fraction': fraction, 'message': message})
    def callback(event):
        if progress:
            progress({**event, 'stage': 'reconstruct'})
    if action in ('propose', 'auto'):
        emit('Separating coherent motion and tracking source drawings')
        proposal_settings={**settings, 'boundaries': [s['frame'] for s in project['seams']]}
        if 'reachFrames' not in proposal_settings:
            default_reach=max(3,round(project['metadata']['fps']*.6))
            neighbors=[abs(s['frame']-frame)-1 for s in project['seams'] if s['frame']!=frame]
            proposal_settings['reachFrames']=min([default_reach,*neighbors])
        manifest = engine.propose(project['source'], frame, folder/'bundle',
            baseline_plan=plan_path, options=proposal_settings,
            progress=callback, cancelled=cancelled)
        _history(record, record.get('candidate'), 'superseded')
        record['candidate'] = _bind_summary(manifest, project)
    elif action == 'import':
        manifest_path = settings.get('manifestPath')
        if not isinstance(manifest_path, str):
            raise ValueError('Choose a reconstruction manifest to import')
        manifest = engine.import_bundle(manifest_path, folder/'bundle', source=project['source'], baseline_plan=plan_path)
        candidate = _bind_summary(manifest, project)
        if candidate.get('frame') != frame:
            raise ValueError('The imported reconstruction targets a different seam')
        _history(record, record.get('candidate'), 'superseded')
        record['candidate'] = candidate
    elif action == 'edit':
        candidate = _editable_candidate(project, frame)
        manifest = engine.edit_bundle(candidate['manifestPath'], folder/'bundle', settings,
            progress=callback, cancelled=cancelled)
        _history(record, record.get('candidate'), 'edited')
        record['candidate'] = _bind_summary(manifest, project)
    elif action == 'segment':
        from .segmentation_model import predict_mask
        from .reconstruction.bundle import asset_path
        from PIL import Image
        import numpy as np
        candidate=_editable_candidate(project,frame)
        bundle=engine.load_bundle(candidate['manifestPath'])
        prompt=settings.get('neuralPrompt')
        if not isinstance(prompt,dict):raise ValueError('Select a subject with neural mask points or a box')
        number=prompt.get('frame');layer_id=prompt.get('layerId')
        if type(number) is not int or str(number) not in bundle['frames']:
            raise ValueError('Neural mask frame must be inside this reconstruction window')
        if layer_id not in {row['id'] for row in bundle['layers']}:
            raise ValueError('Choose an existing reconstruction layer')
        source=np.asarray(Image.open(asset_path(bundle,bundle['frames'][str(number)]['source'])).convert('RGB'))
        emit('Segmenting the selected subject locally')
        prediction=predict_mask(source,points=prompt.get('points',[]),box=prompt.get('box'),cancelled=cancelled)
        mask=folder/'neural-mask.png'
        Image.fromarray(np.rint(prediction['mask']*255).clip(0,255).astype(np.uint8)).save(mask)
        manifest=engine.edit_bundle(candidate['manifestPath'],folder/'bundle',
            {'mask_keyframes':[{'frame':number,'layer_id':layer_id,'path':str(mask),'model':prediction['model']}]},
            progress=callback,cancelled=cancelled)
        _history(record,record.get('candidate'),'segmented')
        record['candidate']=_bind_summary(manifest,project)
    elif action == 'reject':
        _history(record, record.pop('candidate', None), 'rejected')
    elif action == 'revert':
        _history(record, record.pop('accepted', None), 'reverted')
        _history(record, record.pop('candidate', None), 'baseline-reverted')
        result = _replace_plan(recipe, None, frame)
        next_plan = folder/'plan.json'
        atomic_json(next_plan, result)
        if project['artifacts'].get('plan'):
            updated['artifacts']['plan'] = str(next_plan)
            updated['status'] = 'analyzed'
        else:
            # Marker changes may require recalibration. Keep that state while
            # removing the requested reconstruction from the frozen baseline,
            # so reversion never depends on a successful new analysis.
            updated['refinementBaseline']={**updated['refinementBaseline'],
                'plan':str(next_plan),'planSha256':_sha(next_plan)}
        _invalidate_selected(updated, frame)
    # Background fill is an explicit step; auto runs it only after opt-in and
    # only if the engine reports real uncovered pixels.
    if action == 'background' or (action == 'auto' and settings.get('allowAI')
                                  and record['candidate'].get('needsBackgroundFill', False)):
        candidate = record.get('candidate') if action == 'auto' else _current_candidate(project, frame)
        provider = None
        if settings.get('allowAI'):
            from .reconstruction_cloud import RequestBudget, generate_background_anchor
            budget = budget or RequestBudget(settings.get('maxAIRequests', 1), ledger_path=folder/'requests.json')
            cache_dir = Path(project['projectPath']).parent/'artifacts'/'background-cache'
            def provider(rgb, editable, number):
                return generate_background_anchor(rgb, editable, cache_dir, api_key=provider_key,
                    allow_upload=True, budget=budget, quality=settings.get('quality', 'medium'), cancelled=cancelled)
        manifest = engine.prepare_backgrounds(candidate['manifestPath'], folder/'filled', provider=provider,
            progress=callback, cancelled=cancelled)
        record['candidate'] = _bind_summary(manifest, project)
    if action in ('render', 'auto'):
        from .conform import render_conform
        candidate = record.get('candidate') if action == 'auto' else _current_candidate(project, frame)
        if candidate.get('canRender') is False:
            raise ValueError('This candidate cannot be rendered; review its diagnostics')
        preview_recipe = _replace_plan(recipe, candidate['manifestPath'], frame)
        preview_plan = folder/'preview-plan.json'
        atomic_json(preview_plan, preview_recipe)
        bundle = engine.load_bundle(candidate['manifestPath'])
        radius = max(1, round(project['metadata']['fps']))
        start = max(0, bundle['support']['start'] - radius)
        end = min(project['metadata']['frame_count'], bundle['support']['end'] + radius)
        preview = folder/'candidate.mp4'
        emit('Rendering this candidate with the accepted framing and grade')
        render_conform(project['source'], preview_plan, preview, crf=16,
            start_frame=start, end_frame=end, preview_width=options.get('previewWidth', 960),
            progress=callback, cancelled=cancelled)
        record['candidate'] = {**candidate, 'candidatePreviewPath': str(preview),
                               'previewStartFrame': start, 'previewEndFrame': end}
    automatic = action == 'auto' and bool(record.get('candidate', {}).get('autoEligible'))
    if action == 'accept' or automatic:
        candidate = record.get('candidate') if automatic else _current_candidate(project, frame)
        if not automatic and settings.get('review_approved') is not True:
            raise ValueError('Review the candidate and confirm acceptance first')
        if not candidate.get('candidatePreviewPath') or not Path(candidate['candidatePreviewPath']).is_file():
            raise ValueError('Render and review a candidate preview before accepting it')
        manifest = engine.edit_bundle(candidate['manifestPath'], folder/'approved',
            {'review_approved': True}, progress=callback, cancelled=cancelled)
        approved = _bind_summary(manifest, project,
            candidatePreviewPath=candidate['candidatePreviewPath'],
            previewStartFrame=candidate['previewStartFrame'], previewEndFrame=candidate['previewEndFrame'])
        if not approved.get('canAccept', False):
            raise ValueError('Candidate has blocking reconstruction issues and cannot be accepted')
        result = _replace_plan(recipe, manifest, frame)
        next_plan = folder/'plan.json'
        atomic_json(next_plan, result)
        _history(record, record.get('accepted'), 'superseded')
        record['accepted'] = {**approved, 'acceptance': 'automatic' if automatic else 'reviewed'}
        record.pop('candidate', None)
        updated['artifacts']['plan'] = str(next_plan)
        _invalidate_selected(updated, frame)
        updated['status'] = 'analyzed'
    if action not in ('accept', 'revert') and not automatic:
        updated['status'] = project['status']
    emit('Reconstruction saved; existing corrections are preserved', 1.)
    return _publish(snapshot, updated, cancelled=cancelled)
