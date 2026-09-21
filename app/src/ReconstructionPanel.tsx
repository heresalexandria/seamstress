import { useEffect, useState } from 'react';
import { Icon } from './Icons';
import { timecode } from './format';
import { LayerMaskEditor } from './LayerMaskEditor';
import { StudioDialog } from './StudioDialog';
import { VideoViewer } from './VideoViewer';
import type { AISettings, Project, ReconstructionAction, ReconstructionOptions, ReconstructionSummary, Seam, SegmentationStatus } from './types';
import './reconstruction.css';

type Props = {
  project: Project; seam: Seam; disabled: boolean; aiSettings?: AISettings;
  onSettings(): void; onDirtyChange(dirty: boolean): void;
  onAction(action: ReconstructionAction, options?: ReconstructionOptions): Promise<boolean>;
  onImport(): Promise<void>;
  onCancelJob(): void; processingMessage?: string;
  model?: SegmentationStatus;
};

export function ReconstructionPanel({ project, seam, disabled, aiSettings, onSettings, onDirtyChange, onAction, onImport, onCancelJob, processingMessage, model }: Props) {
  const entry = project.reconstructions?.[String(seam.frame)];
  const candidate = entry?.candidate;
  const accepted = entry?.accepted;
  const summary = candidate ?? accepted;
  const defaultReach = Math.max(2, Math.round(project.metadata.fps * .6));
  const [open, setOpen] = useState(Boolean(entry));
  const [reach, setReach] = useState(String(summary?.reachFrames ?? defaultReach));
  const [strength, setStrength] = useState(summary?.motionStrength ?? 1);
  const [allowAI, setAllowAI] = useState(false);
  const [maxRequests, setMaxRequests] = useState(1);
  const [segmentation, setSegmentation] = useState<'auto' | 'classic' | 'neural'>('auto');
  const [editing, setEditing] = useState(false);
  const [maskDirty, setMaskDirty] = useState(false);
  const [review, setReview] = useState<ReconstructionSummary | null>(null);
  const [reviewFrame, setReviewFrame] = useState(seam.frame);
  const [reviewed, setReviewed] = useState(false);
  const [error, setError] = useState('');
  const locked = disabled || !seam.enabled;
  const tuningSupported = !summary || (typeof summary.reachFrames === 'number' && typeof summary.motionStrength === 'number');
  const tuningDirty = tuningSupported && Boolean(summary) && (reach !== String(summary?.reachFrames ?? defaultReach) || strength !== (summary?.motionStrength ?? 1));
  const validReach = Number.isInteger(Number(reach)) && Number(reach) >= 2 && Number(reach) <= Math.round(project.metadata.fps * 10);
  const cloudReady = !allowAI || Boolean(aiSettings?.configured);

  useEffect(() => {
    setReach(String(summary?.reachFrames ?? defaultReach)); setStrength(summary?.motionStrength ?? 1);
    setReviewed(false); setError('');
  }, [summary?.id, summary?.manifestPath, summary?.candidatePreviewPath, summary?.reachFrames, summary?.motionStrength, defaultReach]);
  useEffect(() => { onDirtyChange(maskDirty || tuningDirty); }, [maskDirty, tuningDirty, onDirtyChange]);
  useEffect(() => () => onDirtyChange(false), [onDirtyChange]);

  async function action(kind: ReconstructionAction, overrides: ReconstructionOptions = {}) {
    if (locked || maskDirty) return false;
    if (['propose', 'auto', 'edit'].includes(kind) && !validReach) { setError('Use a whole frame count from 2 to 10 seconds of source footage on each side.'); return false; }
    setError(''); setReviewed(false);
    const tuning = ['propose', 'auto', 'edit'].includes(kind) ? { reachFrames: Number(reach), motionStrength: strength } : {};
    const cloud = ['auto', 'background'].includes(kind) ? { allowAI, maxAIRequests: maxRequests } : {};
    return onAction(kind, { ...tuning, ...cloud, ...(['propose', 'auto'].includes(kind) ? { segmentation } : {}), ...overrides });
  }
  function showReview(item: ReconstructionSummary) { setReviewFrame(item.previewStartFrame ?? item.startFrame); setReview(item); }
  const reviewStart = review?.previewStartFrame ?? review?.startFrame ?? 0;
  const reviewEnd = review?.previewEndFrame ?? review?.endFrame ?? 0;
  const reviewProject: Project | null = review?.candidatePreviewPath ? { ...project, artifacts: {
    ...project.artifacts, fullPreview: undefined, export: undefined,
    seamPreviews: [{ frame: seam.frame, path: review.candidatePreviewPath, startFrame: reviewStart, endFrame: reviewEnd }],
  } } : null;

  return <section className="inspector-section reconstruction-panel" aria-label="Layer reconstruction">
    <button className="reconstruction-heading" aria-expanded={open} onClick={() => setOpen(value => !value)}><span><Icon name="split" size={15}/> Layer reconstruction <small>OPTIONAL</small></span><Icon name={open ? 'down' : 'chevron'} size={13}/></button>
    {open && <div className="reconstruction-body">
      <p className="field-help">For a join where foreground and background shift differently. Separate their movement, recover hidden background, then review a bounded repair.</p>
      {!seam.enabled && <p className="reconstruction-warning">Enable this seam to build or change a reconstruction.</p>}
      {!project.artifacts.plan && <p className="field-help">Analyze the shot first to establish the framing and color that this repair will preserve.</p>}
      {accepted && <div className="reconstruction-status accepted" role="status"><Icon name="check" size={15}/><span>Reconstruction accepted<small>Included in new previews and exports.</small></span></div>}
      {candidate && <div className="reconstruction-status" role="status"><Icon name="diamond" size={15}/><span>{candidate.candidatePreviewPath ? 'Candidate ready to review' : 'Layer proposal ready'}<small>{accepted ? 'Your accepted repair stays active until replaced.' : 'No candidate changes are active yet.'}</small></span></div>}
      <div className="reconstruction-actions"><button className="secondary-button full" disabled={locked || !project.artifacts.plan || maskDirty} onClick={() => void action('propose')}><Icon name="scan" size={14}/>{summary ? 'Make a new proposal' : 'Propose layers'}</button><button className="small-button full" disabled={locked || !project.artifacts.plan || maskDirty || !cloudReady} onClick={() => void action('auto')}><Icon name="diamond" size={13}/> Attempt automatic repair</button></div>
      <p className="field-help">Automatic repair is accepted only when the engine’s quality checks pass. Uncertain results stay as candidates for review.</p>
      <details className="reconstruction-details"><summary>Layer selection method</summary><label className="field-label" htmlFor="reconstruction-segmentation">For new proposals</label><select id="reconstruction-segmentation" className="reconstruction-number" value={segmentation} disabled={locked || maskDirty} onChange={event => setSegmentation(event.target.value as typeof segmentation)}><option value="auto">Automatic · use installed model</option><option value="classic">Classical motion grouping</option><option value="neural" disabled={!model?.available}>Local model · MobileSAM</option></select><p className="field-help">Automatic uses MobileSAM if it is already installed and a subject can be located. Otherwise it uses local motion grouping. No model is downloaded automatically.</p><button className="text-button" disabled={disabled} onClick={onSettings}>Local model settings</button></details>
      <details className="reconstruction-details"><summary>Reach & motion</summary><label className="field-label" htmlFor="reconstruction-reach">Reach on each side <span>FRAMES</span></label><input id="reconstruction-reach" className="reconstruction-number mono" type="number" min={2} max={Math.round(project.metadata.fps * 10)} value={reach} disabled={locked || maskDirty || !tuningSupported} onChange={event => setReach(event.target.value)}/><label className="field-label" htmlFor="reconstruction-strength">Motion correction <span>{Math.round(strength * 100)}%</span></label><input id="reconstruction-strength" type="range" min={0} max={1} step={.05} value={strength} disabled={locked || maskDirty || !tuningSupported} onChange={event => setStrength(Number(event.target.value))}/><p className="field-help">The repair eases to the original motion at its boundaries. Smaller reach confines the change; lower strength reduces layer displacement. Neighboring seams constrain the usable window. Make a new proposal to extend an existing window.</p>{!tuningSupported && <p className="field-help">This imported bundle has authored layer motion. Make a new proposal to use automatic reach and strength controls.</p>}{tuningDirty && <div className="reconstruction-actions"><button className="small-button" disabled={locked || !validReach || maskDirty} onClick={() => void action('edit')}>Apply repair controls</button><button className="text-button" disabled={locked || maskDirty} onClick={() => { setReach(String(summary?.reachFrames ?? defaultReach)); setStrength(summary?.motionStrength ?? 1); }}>Reset controls</button></div>}</details>
      <details className="reconstruction-details"><summary>Background recovery</summary><p className="field-help">Visible pixels from nearby source frames are tried first. AI fill is optional and can invent background detail; foreground characters remain from the source.</p><label className="reconstruction-check"><input type="checkbox" aria-label="Allow AI background fill" checked={allowAI} disabled={locked} onChange={event => setAllowAI(event.target.checked)}/><span>Allow AI background fill<small>Upload selected source frames, masks and nearby context to OpenAI. Requests are billed to my API account.</small></span></label>{allowAI && <><label className="select-row">Maximum requests<select aria-label="Maximum AI requests" value={maxRequests} disabled={locked} onChange={event => setMaxRequests(Number(event.target.value))}>{[1, 2, 3, 4, 6, 8].map(count => <option key={count} value={count}>{count}</option>)}</select></label><p className="field-help">This limit includes retries for each run. It is a request cap, not a dollar budget. No cloud request starts when this checkbox changes.</p><button className="text-button" disabled={disabled} onClick={onSettings}><Icon name="sliders" size={13}/>{aiSettings?.configured ? 'Manage saved OpenAI key' : 'Add an OpenAI key'}</button>{!cloudReady && <p className="reconstruction-warning">Save an API key in Settings before using AI fill.</p>}</>}<button className="small-button full" disabled={!candidate || locked || tuningDirty || maskDirty || !cloudReady} onClick={() => void action('background')}>Recover background{allowAI ? ' · AI allowed' : ' · source only'}</button></details>
      {summary && <><div className="reconstruction-layer-list" aria-label="Proposed layers">{summary.layers.map(layer => <div key={layer.id}><span>{layer.name}<small>{layer.role}{layer.keyframes.length ? ` · ${layer.keyframes.length} keyframes` : ''}</small></span>{typeof layer.confidence === 'number' && <span className="mono" title="Layer proposal confidence">{Math.round(Math.max(0, Math.min(1, layer.confidence)) * 100)}%</span>}</div>)}</div><p className="field-help">Repair window {timecode(summary.startFrame / project.metadata.fps)}–{timecode(summary.endFrame / project.metadata.fps)} · {summary.endFrame - summary.startFrame} frames</p></>}
      {summary && <div className="reconstruction-actions"><button className="secondary-button full" disabled={locked || tuningDirty || !summary.layers.length} onClick={() => setEditing(true)}><Icon name="sliders" size={14}/> Edit layer masks</button>{!candidate && accepted && <p className="field-help">Editing makes a new candidate. Your accepted repair stays active until you approve its replacement.</p>}{candidate && <button className="secondary-button full" disabled={locked || tuningDirty || maskDirty || !candidate.canRender} onClick={() => void action('render')}><Icon name="film" size={14}/> Build candidate preview</button>}</div>}
      {summary?.issues?.length ? <div className="reconstruction-warning"><strong>Review notes</strong><ul>{summary.issues.map((issue, index) => <li key={index}>{issue}</li>)}</ul></div> : null}
      {candidate?.candidatePreviewPath && <><button className="primary-button full" disabled={disabled || tuningDirty || maskDirty} onClick={() => showReview(candidate)}><Icon name="loop" size={14}/> Review candidate loop</button><label className="reconstruction-check"><input type="checkbox" aria-label="I reviewed this reconstruction" checked={reviewed} disabled={locked || tuningDirty || maskDirty} onChange={event => setReviewed(event.target.checked)}/><span>I reviewed movement, layer edges and the window boundaries.</span></label>{candidate.canAccept === false && <p className="reconstruction-warning">Resolve the blocking review notes before accepting this candidate.</p>}<button className="secondary-button full" disabled={locked || !reviewed || tuningDirty || maskDirty || candidate.canAccept === false} onClick={() => void action('accept', { review_approved: true })}><Icon name="check" size={14}/>{accepted ? 'Replace accepted reconstruction' : 'Accept reconstruction'}</button></>}
      {candidate && <button className="text-button reconstruction-reject" disabled={locked || tuningDirty || maskDirty} onClick={() => void action('reject')}>Reject candidate</button>}
      {accepted && <div className="reconstruction-accepted-actions">{accepted.candidatePreviewPath && <button className="text-button" disabled={disabled} onClick={() => showReview(accepted)}>Review accepted reconstruction</button>}<button className="text-button" disabled={locked || tuningDirty || maskDirty} onClick={() => void action('revert')}>Revert reconstruction</button><p className="field-help">Revert restores this seam’s baseline framing and color. Other accepted seam repairs remain in place.</p></div>}
      <button className="text-button" disabled={locked || tuningDirty || maskDirty} onClick={() => void onImport()}><Icon name="folder" size={13}/> Import reconstruction bundle</button>
      {error && <p className="reconstruction-error" role="alert">{error}</p>}
    </div>}
    {editing && summary && <LayerMaskEditor summary={summary} disabled={disabled} onClose={() => setEditing(false)} onDirtyChange={setMaskDirty} onSave={strokes => onAction('edit', { strokes })} onCancelJob={onCancelJob} processingMessage={processingMessage} model={model} onSettings={onSettings} onSegment={neuralPrompt => onAction('segment', { neuralPrompt })}/>}
    {review && reviewProject && <StudioDialog title="Watch the seam breathe." label="Reconstruction review" onClose={() => setReview(null)} className="reconstruction-review-dialog"><p className="studio-intro">Loop the full repair at normal speed, then step through small details. The split compares the original source with this candidate, including the project’s existing framing and color corrections.</p><VideoViewer key={review.candidatePreviewPath} project={reviewProject} selected={seam} frame={reviewFrame} onFrame={setReviewFrame} previewSeconds={(reviewEnd - reviewStart) / project.metadata.fps} reviewRange={{ startFrame: reviewStart, endFrame: reviewEnd }}/><div className="studio-dialog-actions"><span className="field-help">Check both boundaries, hands, outlines, flames and exposed background.</span><button className="secondary-button" onClick={() => setReview(null)}>Back to reconstruction</button></div></StudioDialog>}
  </section>;
}
