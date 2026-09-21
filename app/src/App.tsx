import { useCallback, useEffect, useRef, useState } from 'react';
import { api, isDesktop } from './api';
import { clamp, filename, message, parseTimecode, timecode } from './format';
import { Icon, WeaveMark, WovenIllustration } from './Icons';
import { Timeline } from './Timeline';
import { UpdateControl } from './UpdateControl';
import { VideoViewer, type ViewerHandle } from './VideoViewer';
import { SeamCorrection } from './SeamCorrection';
import { ReconstructionPanel } from './ReconstructionPanel';
import { AISettings } from './AISettings';
import type { AISettings as AISettingsState, JobEvent, JobStage, Project, ReconstructionAction, ReconstructionOptions, Seam, SegmentationStatus, Stage } from './types';

const stageNames: Record<JobStage, string> = { import: 'Opening your video', detect: 'Finding the joins', analyze: 'Measuring continuity', refine: 'Refining this seam', reconstruct: 'Reconstructing this seam', preview: 'Making review previews', process: 'Processing your video', export: 'Exporting your film' };
type ActiveJob = { id: string | null; stage: JobStage; progress?: number; message: string };

export default function App() {
  const [project, setProject] = useState<Project | null>(null);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [frame, setFrame] = useState(0);
  const [job, setJob] = useState<ActiveJob | null>(null);
  const [operation, setOperation] = useState('');
  const [error, setError] = useState('');
  const [notice, setNotice] = useState('');
  const [dragOver, setDragOver] = useState(false);
  const [cancelling, setCancelling] = useState(false);
  const [exportDialog, setExportDialog] = useState(false);
  const [exportError, setExportError] = useState('');
  const [exportIntent, setExportIntent] = useState<'process' | 'export'>('export');
  const [crf, setCrf] = useState(14);
  const [workflowReconstruction, setWorkflowReconstruction] = useState(false);
  const [previewSeconds, setPreviewSeconds] = useState(6);
  const [supportDraft, setSupportDraft] = useState('');
  const [timeDraft, setTimeDraft] = useState('');
  const [frameDraft, setFrameDraft] = useState('');
  const [help, setHelp] = useState(false);
  const [correctionDirty, setCorrectionDirty] = useState(false);
  const [reconstructionDirty, setReconstructionDirty] = useState(false);
  const [aiSettingsOpen, setAISettingsOpen] = useState(false);
  const [aiSettings, setAISettings] = useState<AISettingsState>();
  const [segmentationModel, setSegmentationModel] = useState<SegmentationStatus>();
  const unsavedSeamChanges = correctionDirty || reconstructionDirty;
  const viewer = useRef<ViewerHandle>(null);
  const projectRef = useRef(project);
  const jobRef = useRef(job);
  const pendingJob = useRef(false);
  const operationRef = useRef(false);
  const completedJobs = useRef(new Set<string>());
  const reconstructionCompletion = useRef<((success: boolean) => void) | null>(null);
  const dragDepth = useRef(0);
  const selected = project?.seams.find(s => s.id === selectedId);
  const selectedIndex = project?.seams.findIndex(s => s.id === selectedId) ?? -1;
  const busy = Boolean(job || operation);
  const enabledCount = project?.seams.filter(s => s.enabled).length ?? 0;
  const lowConfidence = project?.seams.filter(s => s.enabled && s.confidence !== undefined && s.confidence < .6).length ?? 0;
  const hasPlan = Boolean(project?.artifacts.plan);
  projectRef.current = project; jobRef.current = job;
  const isWorking = () => Boolean(jobRef.current || pendingJob.current || operationRef.current);

  const applyProject = useCallback((next: Project, fresh = false) => {
    setProject(next);
    setSelectedId(current => !fresh && next.seams.some(s => s.id === current) ? current : next.seams[0]?.id ?? null);
    if (fresh) setFrame(0);
    try { localStorage.setItem('seamstress:last-project', next.projectPath); } catch { /* Storage is optional. */ }
  }, []);

  useEffect(() => api.onJobEvent((event: JobEvent) => {
    if (completedJobs.current.has(event.jobId)) return;
    if (jobRef.current?.id && event.jobId !== jobRef.current.id) return;
    if (!jobRef.current && !pendingJob.current) return;
    if (event.type === 'progress') {
      setJob(current => ({ id: event.jobId, stage: event.stage, progress: event.progress ?? current?.progress, message: event.message || stageNames[event.stage] }));
      return;
    }
    completedJobs.current.add(event.jobId); pendingJob.current = false;
    if (event.project && (!projectRef.current || event.project.projectPath === projectRef.current.projectPath)) applyProject(event.project);
    setJob(null); setCancelling(false);
    if (event.stage === 'reconstruct' && reconstructionCompletion.current) {
      reconstructionCompletion.current(event.type === 'complete'); reconstructionCompletion.current = null;
    }
    if (event.type === 'error') setError(event.error || event.message || 'The job could not complete. Your source video is unchanged.');
    else setNotice(event.type === 'cancelled' ? 'Processing stopped. Your source video is unchanged.' : event.stage === 'reconstruct' ? (event.message || 'Reconstruction updated. Review the candidate and its notes before accepting it.') : event.stage === 'refine' ? 'Selected seam refined. The viewing crop and other corrections are preserved. Preview this seam to review the change.' : event.stage === 'export' ? 'Export complete. Your film is ready to review.' : event.stage === 'process' && event.project?.artifacts.export ? 'Workflow complete. Your export is ready to review.' : 'Stage complete. Your project has been updated.');
  }), [applyProject]);

  useEffect(() => {
    let active = true;
    if (isDesktop) void api.getAISettings().then(next => { if (active) setAISettings(next); }).catch(() => {});
    if (isDesktop) void api.getSegmentationStatus().then(next => { if (active) setSegmentationModel(next); }).catch(() => {});
    return () => { active = false; };
  }, []);

  useEffect(() => {
    setTimeDraft(selected ? timecode(selected.frame / (project?.metadata.fps ?? 24)) : '');
    setFrameDraft(selected ? String(selected.frame) : '');
    setSupportDraft('');
  }, [selected?.id, selected?.frame, project?.metadata.fps]);

  useEffect(() => {
    const key = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement;
      if (target.closest('input, textarea, select, button, [contenteditable="true"], [role="dialog"]') || !project || exportDialog) return;
      if (event.code === 'Space') { event.preventDefault(); viewer.current?.toggle(); }
      if (event.code === 'ArrowLeft' || event.code === 'ArrowRight') { event.preventDefault(); viewer.current?.step(event.code === 'ArrowLeft' ? -1 : 1); }
    };
    window.addEventListener('keydown', key); return () => window.removeEventListener('keydown', key);
  }, [project, exportDialog]);

  async function importVideo(path?: string) {
    if (isWorking()) return;
    if (unsavedSeamChanges) { setError('Save or discard your seam edits before opening another video.'); return; }
    setError(''); setNotice('');
    operationRef.current = true;
    try {
      const source = path ?? await api.pickVideo();
      if (!source) return;
      pendingJob.current = true;
      setJob({ id: null, stage: 'import', message: 'Preparing the source, timeline and seam candidates…' });
      applyProject(await api.createProject({ source }), true);
    } catch (reason) {
      if (/cancelled|canceled/i.test(message(reason))) setNotice('Import stopped. Your source video is unchanged.');
      else setError(message(reason));
    } finally { operationRef.current = false; pendingJob.current = false; setJob(null); setCancelling(false); }
  }

  async function openProject() {
    if (isWorking()) return;
    if (unsavedSeamChanges) { setError('Save or discard your seam edits before opening another project.'); return; }
    setError('');
    operationRef.current = true; setOperation('Opening project…');
    try { const next = await api.openProject(); if (next) { applyProject(next, true); setNotice('Project reopened.'); } }
    catch (reason) { setError(message(reason)); }
    finally { operationRef.current = false; setOperation(''); }
  }

  async function reopenLast() {
    if (isWorking()) return;
    try {
      const path = localStorage.getItem('seamstress:last-project');
      if (!path) { await openProject(); return; }
      operationRef.current = true; setOperation('Reopening your project…'); setError('');
      applyProject(await api.getProject(path), true);
    } catch (reason) { setError(message(reason)); } finally { operationRef.current = false; setOperation(''); }
  }

  async function saveSeams(seams: Seam[], focusId?: string, successNotice = 'Seams saved. Analyze again to update corrected previews.') {
    if (!project || isWorking()) return false;
    const frames = seams.map(s => s.frame);
    if (new Set(frames).size !== frames.length) { setError('There is already a seam at that frame. Choose a different frame.'); return false; }
    operationRef.current = true; setOperation('Saving seam changes…'); setError('');
    try {
      const next = await api.setSeams({ projectPath: project.projectPath, seams: [...seams].sort((a, b) => a.frame - b.frame) });
      applyProject(next);
      if (focusId) setSelectedId(focusId);
      setNotice(successNotice);
      return true;
    } catch (reason) { setError(message(reason)); return false; }
    finally { operationRef.current = false; setOperation(''); }
  }

  function moveSeam(seam: Seam, nextFrame: number) {
    if (!project || nextFrame === seam.frame) return;
    if (unsavedSeamChanges) { setError('Save or discard your seam edits before moving a marker.'); return; }
    if (!Number.isInteger(nextFrame) || nextFrame <= 0 || nextFrame >= project.metadata.frame_count) { setError(`Use a frame between 1 and ${project.metadata.frame_count - 1}.`); return; }
    const correction = seam.correction ? { ...seam.correction } : undefined;
    const hadManual = Boolean(correction?.manual);
    if (correction?.manual) { if (correction.geometry === 'manual') correction.geometry = 'auto'; delete correction.manual; }
    void saveSeams(project.seams.map(s => s.id === seam.id ? { ...s, correction, frame: nextFrame, time: nextFrame / project.metadata.fps, origin: 'manual', confidence: undefined, reasons: undefined } : s), seam.id,
      hadManual ? 'Seam moved. Its custom framing was cleared because it belongs to the previous frame. Analyze again to update previews.' : undefined);
  }

  function selectSeam(seam: Seam) {
    if (unsavedSeamChanges && seam.id !== selectedId) { setError('Save or discard your seam edits before selecting another seam.'); return; }
    setSelectedId(seam.id); viewer.current?.seek(Math.max(0, seam.frame - 1));
  }
  function addSeam() {
    if (!project || isWorking()) return;
    if (unsavedSeamChanges) { setError('Save or discard your seam edits before adding a marker.'); return; }
    const at = clamp(frame, 1, project.metadata.frame_count - 1);
    const existing = project.seams.find(s => s.frame === at);
    if (existing) { selectSeam(existing); setNotice('A seam already exists at this frame.'); return; }
    const id = crypto.randomUUID();
    void saveSeams([...project.seams, { id, frame: at, time: at / project.metadata.fps, enabled: true, origin: 'manual' }], id);
  }

  async function runStage(stage: Stage, exportPath?: string, selectedOnly = false) {
    if (!project || isWorking()) return;
    if (unsavedSeamChanges) { setError('Save or discard your seam edits before running a stage.'); return; }
    const seamOnly = stage === 'refine' || selectedOnly;
    if (seamOnly && !selected?.enabled) { setError('Select an enabled seam first.'); return; }
    const supportFrames = supportDraft.trim() ? Number(supportDraft) : undefined;
    if (stage === 'refine' && supportFrames !== undefined && (!Number.isInteger(supportFrames) || supportFrames < 1 || supportFrames > Math.round(60 * project.metadata.fps))) { setError('Correction reach must be a positive whole frame count of at most 60 seconds, or blank for automatic.'); return; }
    if (selectedOnly) viewer.current?.reviewSeam();
    setError(''); setNotice(''); setCancelling(false); pendingJob.current = true;
    setJob({ id: null, stage, message: stageNames[stage] });
    try {
      const result = await api.run({ projectPath: project.projectPath, stage, options: { crf, previewSeconds, ...(seamOnly ? { frame: selected!.frame } : {}), ...(stage === 'refine' && supportFrames !== undefined ? { supportFrames } : {}), ...(exportPath ? { exportPath } : {}), ...(stage === 'process' && workflowReconstruction ? { reconstructionEnabled: true, reconstruction: { allowAI: false } } : {}) } });
      if (!completedJobs.current.has(result.jobId)) setJob(current => ({ ...(current ?? { stage, message: stageNames[stage] }), id: result.jobId }));
    } catch (reason) { pendingJob.current = false; setJob(null); setError(message(reason)); }
  }

  async function runReconstruction(action: ReconstructionAction, reconstruction: ReconstructionOptions = {}) {
    if (!project || !selected?.enabled || isWorking()) return false;
    if (correctionDirty) { setError('Apply or reset your framing and color settings before reconstructing this seam.'); return false; }
    setError(''); setNotice(''); setCancelling(false); pendingJob.current = true;
    const stage = 'reconstruct' as const;
    setJob({ id: null, stage, message: stageNames[stage] });
    const completion = new Promise<boolean>(resolve => { reconstructionCompletion.current = resolve; });
    try {
      const result = await api.run({ projectPath: project.projectPath, stage, options: { frame: selected.frame, action, reconstruction } });
      if (!completedJobs.current.has(result.jobId)) setJob(current => ({ ...(current ?? { stage, message: stageNames[stage] }), id: result.jobId }));
      const success = await completion;
      if (action === 'setup-model') {
        try { setSegmentationModel(await api.getSegmentationStatus()); } catch { /* The settings screen keeps the last known status. */ }
      }
      return success;
    } catch (reason) {
      reconstructionCompletion.current?.(false); reconstructionCompletion.current = null;
      pendingJob.current = false; setJob(null); setError(message(reason)); return false;
    }
  }

  async function importReconstruction() {
    if (!project || !selected?.enabled || isWorking() || unsavedSeamChanges) return;
    operationRef.current = true; setOperation('Importing reconstruction bundle…'); setError('');
    try {
      const next = await api.importReconstruction({ projectPath: project.projectPath, frame: selected.frame });
      if (next) { applyProject(next); setNotice('Reconstruction imported as a candidate. Build its preview and review it before accepting.'); }
    } catch (reason) { setError(message(reason)); }
    finally { operationRef.current = false; setOperation(''); }
  }

  async function exportVideo() {
    if (!project) return;
    setExportError('');
    try {
      const destination = await api.chooseExportPath({ suggestedName: `${filename(project.source).replace(/\.[^.]+$/, '')}-seamstress.mp4` });
      if (!destination) return;
      setExportDialog(false); await runStage(exportIntent, destination);
    } catch (reason) { setExportError(message(reason)); }
  }

  function showExport(intent: 'process' | 'export') {
    if (unsavedSeamChanges) { setError('Save or discard your seam edits before exporting.'); return; }
    setExportError(''); setExportIntent(intent); setExportDialog(true);
  }

  async function importSeamCorrection() {
    if (!project || !selected || isWorking() || unsavedSeamChanges) return;
    operationRef.current = true; setOperation('Importing reviewed framing…'); setError('');
    try {
      const next = await api.importSeamCorrection({ projectPath: project.projectPath, frame: selected.frame });
      if (next) { applyProject(next); setNotice('Reviewed framing imported. Refine this seam only to preserve the rest of your shot, or analyze all joins.'); }
    } catch (reason) { setError(message(reason)); }
    finally { operationRef.current = false; setOperation(''); }
  }

  function dialogKeys(event: React.KeyboardEvent<HTMLElement>) {
    if (event.key === 'Escape') { setExportDialog(false); return; }
    if (event.key !== 'Tab') return;
    const controls = [...event.currentTarget.querySelectorAll<HTMLElement>('button:not(:disabled), select:not(:disabled), input:not(:disabled)')];
    const first = controls[0], last = controls[controls.length - 1];
    if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last?.focus(); }
    else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first?.focus(); }
  }

  async function cancelJob() {
    if (!job?.id) return;
    setCancelling(true);
    try { await api.cancelJob(job.id); }
    catch (reason) { setCancelling(false); setError(message(reason)); }
  }

  function dropVideo(event: React.DragEvent) {
    event.preventDefault(); setDragOver(false); dragDepth.current = 0;
    if (busy) return;
    const files = event.dataTransfer.files;
    if (files.length !== 1) { setError('Drop one video at a time. Each video becomes its own project.'); return; }
    try { void importVideo(api.pathForFile(files[0])); } catch (reason) { setError(message(reason)); }
  }

  const progress = job?.progress === undefined ? undefined : clamp(job.progress > 1 ? job.progress / 100 : job.progress, 0, 1);
  const stageItems: { stage: Stage; label: string; icon: string; detail: string; ready: boolean; done: boolean }[] = [
    { stage: 'detect', label: 'Find seams', icon: 'scan', detail: 'Locate generation joins', ready: Boolean(project), done: Boolean(project?.seams.length) },
    { stage: 'analyze', label: 'Analyze & match', icon: 'sliders', detail: 'Framing, motion & color', ready: enabledCount > 0, done: hasPlan },
    { stage: 'preview', label: 'Review previews', icon: 'play', detail: 'Check continuity', ready: hasPlan, done: Boolean(project?.artifacts.seamPreviews?.length || project?.artifacts.fullPreview) },
    { stage: 'export', label: 'Export film', icon: 'export', detail: 'Keep the shot intact', ready: hasPlan, done: Boolean(project?.artifacts.export) },
  ];

  return <div className={`app-shell ${dragOver ? 'is-dragging' : ''}`} onDragEnter={event => { if (event.dataTransfer.types.includes('Files')) { event.preventDefault(); dragDepth.current++; setDragOver(true); } }} onDragOver={event => { if (event.dataTransfer.types.includes('Files')) event.preventDefault(); }} onDragLeave={event => { event.preventDefault(); dragDepth.current = Math.max(0, dragDepth.current - 1); if (!dragDepth.current) setDragOver(false); }} onDrop={dropVideo}>
    <div className="window-bar"><span>SEAMSTRESS STUDIO</span><UpdateControl busy={busy}/><span className="window-local"><i/> {isDesktop ? 'LOCAL WORKSPACE' : 'BROWSER PREVIEW'}</span></div>
    <aside className="sidebar"><div className="brand"><WeaveMark/><div><span className="wordmark">Seamstress</span><span className="brand-tagline">Continuity for AI oners.</span></div></div><div className="sidebar-project"><span className="eyebrow">YOUR WORKSPACE</span><button className="project-card" onClick={openProject} disabled={busy}><span className="project-icon"><Icon name="film"/></span><span><strong>{project?.name || 'Untitled project'}</strong><small>{project ? `${timecode(project.metadata.duration, false)} · ${project.metadata.fps.toFixed(3)} fps` : 'A new thread starts here'}</small></span><Icon name="down" size={13}/></button><div className="project-actions"><button onClick={() => void importVideo()} disabled={busy}><Icon name="plus" size={13}/> New video</button><button onClick={openProject} disabled={busy}><Icon name="folder" size={13}/> Open</button></div></div><div className="workflow">{project && <button className="project-path-button" onClick={() => void api.revealFile(project.projectPath).catch(e => setError(message(e)))}><Icon name="folder" size={12}/> Show project in Finder <Icon name="external" size={11}/></button>}<span className="eyebrow">THE WORKFLOW</span>{stageItems.map((item, i) => <button key={item.stage} className={`workflow-step ${item.done ? 'is-complete' : ''} ${job?.stage === item.stage ? 'is-current' : ''}`} onClick={() => item.stage === 'export' ? showExport('export') : void runStage(item.stage)} disabled={!item.ready || busy}><span className="step-number">{item.done ? <Icon name="check" size={13}/> : `0${i + 1}`}</span><span><strong>{item.label}</strong><small>{item.detail}</small></span><Icon name={item.icon} size={17}/></button>)}</div><div className="sidebar-bottom"><div className="preserve-note"><span className="stitch-line"/><p>Every frame has a story.<br/><em>Keep it together.</em></p></div><button className="sidebar-help" onClick={() => setAISettingsOpen(true)} disabled={busy}><Icon name="sliders" size={16}/> AI settings <span>OPTIONAL</span></button><button className="sidebar-help" onClick={() => setHelp(v => !v)}><Icon name="help" size={16}/> A little guidance <span>?</span></button><div className="local-status"><i/> ORIGINAL VIDEO PRESERVED</div></div></aside>
    <main className="workspace"><header className="workspace-header"><div className="breadcrumb">Workspace <Icon name="chevron" size={12}/><span>{project ? project.name : 'New project'}</span></div><div className="header-actions">{project?.artifacts.export && <button className="text-button" onClick={() => void api.revealFile(project.artifacts.export!).catch(e => setError(message(e)))}><Icon name="external" size={14}/> Show export</button>}<button className="secondary-button" onClick={() => showExport('export')} disabled={!hasPlan || busy}><Icon name="export" size={15}/> Export</button><button className="primary-button compact" onClick={() => showExport('process')} disabled={!project || busy}>{job?.stage === 'process' ? <span className="spinner"/> : <Icon name="diamond" size={14}/>} Run workflow</button></div></header>
      {!isDesktop && <div className="browser-banner"><Icon name="help" size={14}/> Interface preview · importing and processing video require the desktop app.</div>}
      {(error || notice) && <div className={`notification ${error ? 'is-error' : ''}`} role={error ? 'alert' : 'status'}><Icon name={error ? 'warning' : 'check'} size={15}/><span>{error || notice}</span><button className="icon-button" onClick={() => { setError(''); setNotice(''); }} aria-label="Dismiss message"><Icon name="close" size={14}/></button></div>}
      {project ? <><div className="project-heading"><div><span className="eyebrow">ONER REVIEW</span><h1>Keep the shot <em>flowing.</em></h1></div><div className="project-meta"><span>{project.metadata.width >= project.metadata.height ? 'LANDSCAPE' : 'PORTRAIT'}</span><i/>{project.seams.length} {project.seams.length === 1 ? 'JOIN' : 'JOINS'}<i/>{timecode(project.metadata.duration, false)}</div></div><VideoViewer key={project.id} ref={viewer} project={project} selected={selected} frame={frame} onFrame={setFrame} previewSeconds={previewSeconds}/><Timeline project={project} selected={selected} frame={frame} disabled={busy} onSeek={at => viewer.current?.seek(at)} onSelect={selectSeam} onMove={moveSeam} onAdd={addSeam}/>{project.seams.length > 0 && <div className="seam-strip" aria-label="Select a seam">{project.seams.map((seam, index) => <button key={seam.id} className={`${selectedId === seam.id ? 'selected' : ''} ${!seam.enabled ? 'off' : ''}`} onClick={() => selectSeam(seam)}><span className="mono">{String(index + 1).padStart(2, '0')}</span><span>{timecode(seam.time, false)}</span>{seam.confidence !== undefined && seam.confidence < .6 ? <Icon name="warning" size={12}/> : <span className="seam-pill-dot"/>}</button>)}</div>}</> : <section className="empty-state"><div className="empty-eyebrow"><span/> FOR AI-GENERATED ONERS</div><h1>Many generations.<br/><em>One continuous shot.</em></h1><p className="empty-description">An oner is a single continuous shot. Build yours across AI generations,<br/>then smooth the small shifts in framing, motion and color at each join.</p><div className="import-dropzone" onClick={() => !busy && void importVideo()} onKeyDown={e => { if ((e.key === 'Enter' || e.key === ' ') && !busy) { e.preventDefault(); void importVideo(); } }} role="button" tabIndex={busy ? -1 : 0} aria-label="Import a stitched video"><WovenIllustration/><div className="import-dropzone-copy"><span className="import-plus"><Icon name={operation ? 'film' : 'plus'} size={22}/></span><strong>{operation || 'Drop your stitched video here'}</strong><span>one file containing your sequential generations</span></div><div className="import-formats">MP4 · MOV · MKV · WEBM <span>PROCESSED ON YOUR DEVICE</span></div></div><div className="empty-bottom"><button className="text-button" onClick={openProject} disabled={busy}><Icon name="folder" size={16}/> Open a project <Icon name="arrow" size={14}/></button><span className="subtle-dot"/><button className="text-button dim" onClick={reopenLast} disabled={busy}>Continue last session</button></div><div className="empty-principles"><span><Icon name="film" size={15}/> Original frames</span><span><Icon name="sliders" size={15}/> Measured corrections</span><span><Icon name="loop" size={15}/> Review every seam</span></div></section>}
      {(job || operation) && <div className="job-panel" role="status"><div className="job-symbol"><span className="spinner"/></div><div className="job-copy"><strong>{operation || (job && stageNames[job.stage])}</strong><span>{operation ? 'Your source remains untouched.' : job?.message}</span><div className={`job-track ${progress === undefined ? 'indeterminate' : ''}`}><span style={progress === undefined ? undefined : { width: `${progress * 100}%` }}/></div></div>{job && <><span className="job-percent mono">{progress === undefined ? '···' : `${Math.round(progress * 100)}%`}</span><button className="small-button" onClick={cancelJob} disabled={!job.id || cancelling}>{cancelling ? 'Stopping…' : 'Cancel'}</button></>}</div>}
      <footer className="workspace-footer"><span><WeaveMark small/> Made for continuity between generations.</span><span className="mono">{project ? `REV ${String(project.revision).padStart(3, '0')}` : 'READY WHEN YOU ARE'}</span></footer>
    </main>
    <aside className="inspector"><div className="inspector-header"><span className="eyebrow">SEAM INSPECTOR</span><Icon name="sliders" size={16}/></div>{selected && project ? <><div className="seam-heading"><span className="seam-index">{String(selectedIndex + 1).padStart(2, '0')}</span><div><h2>A closer look.</h2><span>{selected.origin === 'manual' ? 'Manually placed seam' : 'Detected join'}</span></div></div><div className="inspector-section"><div className="label-row"><label htmlFor="seam-enabled">Include in correction</label><button id="seam-enabled" role="switch" aria-checked={selected.enabled} className={`toggle ${selected.enabled ? 'on' : ''}`} disabled={busy || unsavedSeamChanges} onClick={() => void saveSeams(project.seams.map(s => s.id === selected.id ? { ...s, enabled: !s.enabled } : s))}><span/></button></div><label className="field-label" htmlFor="seam-time">Join time <span>MIN : SEC</span></label><div className="input-wrap"><input id="seam-time" value={timeDraft} className="mono" disabled={busy || unsavedSeamChanges} onChange={e => setTimeDraft(e.target.value)} onBlur={() => { const seconds = parseTimecode(timeDraft); if (seconds === null) { setError('Enter a time as mm:ss.mmm, hh:mm:ss.mmm, or seconds.'); setTimeDraft(timecode(selected.time)); } else moveSeam(selected, Math.round(seconds * project.metadata.fps)); }} onKeyDown={e => { if (e.key === 'Enter') e.currentTarget.blur(); }}/><Icon name="diamond" size={13}/></div><label className="field-label" htmlFor="seam-frame">Exact frame <span>ZERO BASED</span></label><div className="input-wrap"><input id="seam-frame" inputMode="numeric" value={frameDraft} className="mono" disabled={busy || unsavedSeamChanges} onChange={e => setFrameDraft(e.target.value)} onBlur={() => { const n = Number(frameDraft); if (!frameDraft.trim() || !Number.isInteger(n)) { setError('Enter a whole frame number.'); setFrameDraft(String(selected.frame)); } else moveSeam(selected, n); }} onKeyDown={e => { if (e.key === 'Enter') e.currentTarget.blur(); }}/><span className="input-suffix">F</span></div><p className="field-help">The first frame of the next generation. Drag its marker or enter an exact position.</p></div><div className="inspector-section"><div className="section-caption">DETECTION READOUT</div>{selected.confidence !== undefined ? <div className="confidence"><span>{selected.confidence < .6 ? <Icon name="warning" size={15}/> : <span className="status-dot"/>}{selected.confidence < .6 ? 'Needs a closer look' : 'Detection confidence'}</span><strong className="mono">{Math.round(clamp(selected.confidence, 0, 1) * 100)}%</strong></div> : <div className="muted-info"><Icon name="diamond" size={15}/>{selected.origin === 'manual' ? 'Placed by you' : 'Confidence not provided'}</div>}{selected.kind && <div className="kind-chip">{selected.kind.replace(/[_-]/g, ' ')}</div>}{selected.reasons?.length ? <ul className="reason-list">{selected.reasons.map((reason, i) => <li key={i}>{reason}</li>)}</ul> : <p className="field-help">Analyze this join to inspect continuity. Check that each marker matches a generation boundary.</p>}</div><SeamCorrection key={`${project.id}:${selected.id}`} seam={selected} project={project} result={project.seamResults?.find(item => item.frame === selected.frame)} disabled={busy || reconstructionDirty} onDirtyChange={setCorrectionDirty} onSave={async correction => { return await saveSeams(project.seams.map(seam => seam.id === selected.id ? { ...seam, correction } : seam), selected.id, 'Seam settings saved. Refine this seam only to preserve other fixes, or analyze all joins.'); }} onImport={importSeamCorrection}/><div className="inspector-section"><div className="section-caption">REFINE ONE JOIN</div><label className="field-label" htmlFor="refine-support">Correction reach <span>FRAMES EACH SIDE</span></label><div className="input-wrap"><input id="refine-support" inputMode="numeric" className="mono" placeholder="Automatic" value={supportDraft} disabled={busy || unsavedSeamChanges} onChange={e => setSupportDraft(e.target.value)}/></div><button className="secondary-button full" disabled={!selected.enabled || !(hasPlan || project.refinementBaseline) || busy || unsavedSeamChanges} onClick={() => void runStage('refine')}><Icon name="sliders" size={14}/> Refine this seam only</button><p className="field-help">{hasPlan || project.refinementBaseline ? 'Keeps the existing viewing crop and every other fix. Automatic reach covers this seam’s previous correction. Other pending seam edits must be analyzed together.' : 'Analyze the shot once to establish a baseline, then refine individual seams without rebuilding the other fixes.'}</p></div><ReconstructionPanel key={`reconstruction:${project.id}:${selected.id}:${selected.frame}`} project={project} seam={selected} disabled={busy || correctionDirty} aiSettings={aiSettings} onSettings={() => setAISettingsOpen(true)} onDirtyChange={setReconstructionDirty} onAction={runReconstruction} onImport={importReconstruction} onCancelJob={() => void cancelJob()} processingMessage={job?.message} model={segmentationModel}/><div className="inspector-section"><div className="section-caption">REVIEW WINDOW</div><label className="select-row" htmlFor="preview-seconds"><span>Time around join</span><select id="preview-seconds" value={previewSeconds} disabled={busy} onChange={e => setPreviewSeconds(Number(e.target.value))}>{[4, 6, 8, 12].map(n => <option key={n} value={n}>{n} seconds</option>)}</select></label><button className="secondary-button full" disabled={!hasPlan || !selected.enabled || busy || unsavedSeamChanges} onClick={() => void runStage('preview', undefined, true)}><Icon name="play" size={14}/> Preview this seam</button><p className="field-help">Review movement and color at normal speed before exporting.</p></div><button className="delete-seam" disabled={busy || unsavedSeamChanges} onClick={() => void saveSeams(project.seams.filter(s => s.id !== selected.id))}><Icon name="trash" size={14}/> Remove seam</button></> : <div className="inspector-empty"><div className="inspector-glyph"><Icon name="diamond" size={32}/><span/></div><h2>Mind the<br/><em>in-between.</em></h2><p>{project ? 'Find the joins, then select a marker to inspect and adjust its correction.' : 'Each generation continues the shot a little differently. Inspect the small shifts where they meet.'}</p><div className="inspector-empty-rule"/><div className="inspector-tip"><span>01</span><p>Find where one generation<br/>becomes the next.</p></div><div className="inspector-tip"><span>02</span><p>Match the framing,<br/>motion and color.</p></div><div className="inspector-tip"><span>03</span><p>Watch it through.<br/>Trust what you see.</p></div>{project && <button className="secondary-button full" disabled={busy} onClick={() => void runStage('detect')}><Icon name="scan" size={15}/> Find seams</button>}</div>}{Boolean(project?.warnings?.length || lowConfidence) && <div className="project-warnings"><div><Icon name="warning" size={15}/><strong>Review notes</strong></div>{lowConfidence > 0 && <p>{lowConfidence} {lowConfidence === 1 ? 'join needs' : 'joins need'} a closer look.</p>}{project?.warnings?.map((warning, i) => <p key={i}>{warning}</p>)}</div>}<div className="inspector-footnote"><span className="mini-stitch"/> A good seam is one you don’t notice.</div></aside>
    {aiSettingsOpen && <AISettings busy={busy} onChange={setAISettings} onClose={() => setAISettingsOpen(false)} model={segmentationModel} canInstall={Boolean(project && selected?.enabled && !unsavedSeamChanges)} onInstall={() => runReconstruction('setup-model')} onCancelJob={() => void cancelJob()} processingMessage={job?.message}/>}
    {dragOver && <div className="drop-overlay"><WeaveMark/><h2>Start a new thread.</h2><p>{busy ? 'Finish or cancel the current job before importing.' : 'Drop one stitched video to create a project.'}</p></div>}
    {help && <div className="help-popover" role="dialog" aria-label="Workflow guidance"><button className="icon-button" onClick={() => setHelp(false)} aria-label="Close guidance"><Icon name="close" size={15}/></button><span className="eyebrow">A LITTLE GUIDANCE</span><h2>Keep one shot going.</h2><p>An oner is a single continuous shot. Sequential AI generations can extend it, but slight changes in framing, scale, composition or color can reveal the joins.</p><p>Import the generations already stitched into one video. Check the detected joins and adjust their markers. Analyze, review the corrected previews, then export the full shot.</p><p>Select a seam to choose automatic, custom or disabled framing and set its color treatment. Apply settings, analyze again, and regenerate previews. Import reviewed framing to reuse measurements for the same source and seam.</p><p>Framing and color corrections use your original frames, with no morphs or generated in-betweens. Optional layer reconstruction can recover hidden background and adjust layers independently; cloud background generation requires your explicit choice. Review every candidate at normal speed.</p><div><kbd>Space</kbd> Play / pause <kbd>← →</kbd> Step one frame</div></div>}
    {exportDialog && project && <div className="modal-backdrop" onClick={() => setExportDialog(false)}><section className="export-modal" role="dialog" aria-modal="true" aria-labelledby="export-title" onClick={e => e.stopPropagation()} onKeyDown={dialogKeys}><button className="modal-close icon-button" onClick={() => setExportDialog(false)} aria-label="Close export"><Icon name="close"/></button><div className="export-emblem"><Icon name="export" size={27}/></div><span className="eyebrow">THE FINAL THREAD</span><h2 id="export-title">Bring it all <em>together.</em></h2><p>{exportIntent === 'process' ? 'Analyze the generation joins, build previews, and export your shot in one workflow.' : 'Export the full corrected shot at its original size and frame rate.'}</p>{exportIntent === 'process' && <label className="reconstruction-check workflow-reconstruction-choice"><input type="checkbox" aria-label="Attempt layer reconstruction at enabled seams" checked={workflowReconstruction} disabled={busy} onChange={event => setWorkflowReconstruction(event.target.checked)}/><span>Attempt layer reconstruction at enabled seams<small>Source-only recovery; no uploads. Repairs that pass conservative checks are included. Uncertain candidates are saved for review and excluded from this export until you accept them.</small></span></label>}<div className="export-facts"><span>{project.metadata.width} × {project.metadata.height}</span><span>{project.metadata.fps_fraction} fps</span><span>{timecode(project.metadata.duration, false)}</span></div><label className="field-label" htmlFor="export-quality">Picture quality</label><select autoFocus id="export-quality" className="quality-select" value={crf} onChange={e => setCrf(Number(e.target.value))}><option value={10}>Very high · CRF 10 · larger file</option><option value={14}>High · CRF 14</option><option value={18}>Balanced · CRF 18 · smaller file</option></select><p className="field-help">H.264 video · {project.metadata.has_audio ? 'original audio copied' : 'no source audio'} · original timing</p>{Boolean(project.warnings?.length || lowConfidence) && <div className="export-review-note"><Icon name="warning" size={15}/> Your project has review notes. Check the previews before calling the edit finished.</div>}{exportError && <div className="export-error" role="alert"><Icon name="warning" size={15}/><span>{exportError}</span></div>}<button className="primary-button full" onClick={() => void exportVideo()} disabled={busy || (exportIntent === 'export' && !hasPlan)}><Icon name="export" size={16}/>{exportIntent === 'process' ? 'Choose location & run workflow' : 'Choose location & export'} <Icon name="arrow" size={15}/></button><button className="text-button modal-cancel" onClick={() => setExportDialog(false)}>Back to the editing room</button></section></div>}
  </div>;
}
