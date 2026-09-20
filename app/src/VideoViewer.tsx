import { forwardRef, useEffect, useImperativeHandle, useRef, useState } from 'react';
import { api } from './api';
import { clamp, timecode } from './format';
import { Icon } from './Icons';
import type { Project, Seam } from './types';

export type ViewerHandle = { seek(frame: number): void; step(direction: number): void; toggle(): void };
type Props = { project: Project; selected?: Seam; frame: number; onFrame(frame: number): void; previewSeconds: number };

export const VideoViewer = forwardRef<ViewerHandle, Props>(function VideoViewer({ project, selected, frame, onFrame, previewSeconds }, ref) {
  const source = useRef<HTMLVideoElement>(null);
  const candidate = useRef<HTMLVideoElement>(null);
  const viewport = useRef<HTMLDivElement>(null);
  const [playing, setPlaying] = useState(false);
  const [mode, setMode] = useState<'split' | 'source' | 'candidate'>('split');
  const [scope, setScope] = useState<'seam' | 'whole'>('seam');
  const [loop, setLoop] = useState(true);
  const [muted, setMuted] = useState(false);
  const [split, setSplit] = useState(50);
  const [mediaError, setMediaError] = useState('');
  const fps = project.metadata.fps;
  const seamPreview = project.artifacts.seamPreviews?.find(p => p.frame === selected?.frame);
  const fullPath = project.artifacts.fullPreview ?? project.artifacts.export;
  const corrected = scope === 'seam' && seamPreview ? seamPreview.path : fullPath;
  const offset = scope === 'seam' && seamPreview ? seamPreview.startFrame / fps : 0;
  const candidateEnd = scope === 'seam' && seamPreview ? seamPreview.endFrame / fps : project.metadata.duration;
  const currentTime = frame / fps;
  const inCandidate = Boolean(corrected) && currentTime >= offset && currentTime < candidateEnd;
  const loopStart = selected ? Math.max(0, selected.time - previewSeconds / 2) : 0;
  const loopEnd = selected ? Math.min(project.metadata.duration, selected.time + previewSeconds / 2) : project.metadata.duration;

  function synchronize(force = false) {
    const a = source.current, b = candidate.current;
    if (!a || !b || !corrected) return;
    const desired = a.currentTime - offset;
    if (desired < 0 || a.currentTime >= candidateEnd) { b.pause(); return; }
    if (b.readyState >= 1 && (force || Math.abs(b.currentTime - desired) > .055)) b.currentTime = Math.max(0, desired);
    if (!a.paused && b.paused) void b.play().catch(() => {});
    if (a.paused && !b.paused) b.pause();
  }

  function seek(target: number) {
    const video = source.current;
    if (!video) return;
    video.pause(); candidate.current?.pause(); setPlaying(false);
    const next = clamp(Math.round(target), 0, project.metadata.frame_count - 1);
    video.currentTime = (next + .05) / fps;
    onFrame(next); synchronize(true);
  }

  function toggle() {
    const video = source.current;
    if (!video) return;
    if (video.paused) {
      if (scope === 'seam' && selected && loop && (video.currentTime < loopStart || video.currentTime >= loopEnd - 1 / fps)) video.currentTime = loopStart;
      void video.play().then(() => { setPlaying(true); synchronize(true); }).catch(error => setMediaError(String(error)));
    } else { video.pause(); candidate.current?.pause(); setPlaying(false); }
  }

  function step(direction: number) { seek(Math.floor((source.current?.currentTime ?? frame / fps) * fps + .001) + direction); }
  useImperativeHandle(ref, () => ({ seek, toggle, step }));
  useEffect(() => { setMediaError(''); synchronize(true); }, [corrected, offset]);
  useEffect(() => {
    const a = source.current;
    let request: number;
    const tick = () => { if (a && !a.paused) synchronize(); request = requestAnimationFrame(tick); };
    request = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(request);
  }, [corrected, offset, candidateEnd]);

  function scrubSplit(event: React.PointerEvent<HTMLButtonElement>) {
    event.preventDefault();
    const handle = event.currentTarget;
    handle.setPointerCapture(event.pointerId);
    const update = (e: PointerEvent) => {
      const bounds = viewport.current?.getBoundingClientRect();
      if (bounds) setSplit(clamp((e.clientX - bounds.left) / bounds.width * 100, 2, 98));
    };
    const stop = () => { handle.removeEventListener('pointermove', update); handle.removeEventListener('pointerup', stop); handle.removeEventListener('pointercancel', stop); };
    handle.addEventListener('pointermove', update); handle.addEventListener('pointerup', stop); handle.addEventListener('pointercancel', stop);
  }

  return <section className="viewer-section" aria-label="Video comparison">
    <div className="viewer-topline"><div className="viewer-label"><span className="live-dot"/> REVIEW CANVAS</div><div className="segmented scope-switch" aria-label="Playback range"><button className={scope === 'seam' ? 'active' : ''} onClick={() => setScope('seam')} disabled={!selected}>Selected seam</button><button className={scope === 'whole' ? 'active' : ''} onClick={() => { setScope('whole'); setLoop(false); }}>Whole video</button></div><span className="mono dim">{project.metadata.width} × {project.metadata.height}</span></div>
    <div className="video-viewport" ref={viewport} style={{ '--video-aspect': `${project.metadata.width} / ${project.metadata.height}` } as React.CSSProperties}>
      <video ref={source} src={api.mediaUrl(project.artifacts.proxy ?? project.source)} className="source-video" playsInline preload="metadata" muted={muted} onLoadedMetadata={() => { onFrame(0); setMediaError(''); }} onTimeUpdate={() => {
        const a = source.current; if (!a) return;
        if (scope === 'seam' && selected && loop && a.currentTime >= loopEnd && !a.paused) { a.currentTime = loopStart; synchronize(true); }
        onFrame(clamp(Math.floor(a.currentTime * fps + .001), 0, project.metadata.frame_count - 1));
      }} onSeeked={() => synchronize(true)} onPlay={() => setPlaying(true)} onPause={() => setPlaying(false)} onEnded={() => { candidate.current?.pause(); setPlaying(false); }} onError={() => setMediaError('This video could not be played. A browser-compatible proxy may still be processing; try reopening the project after import completes.')} />
      {corrected && <video ref={candidate} src={api.mediaUrl(corrected)} className="candidate-video" playsInline muted preload="metadata" style={{ clipPath: mode === 'split' ? `inset(0 0 0 ${split}%)` : undefined, opacity: mode === 'source' || !inCandidate ? 0 : 1 }} onLoadedMetadata={() => synchronize(true)} onError={() => setMediaError('The corrected preview could not be loaded. Generate a fresh preview and try again.')} />}
      <div className="video-badges">{mode !== 'candidate' && <span>ORIGINAL</span>}{mode !== 'source' && <span className={inCandidate ? 'candidate-badge' : 'pending-badge'}>{inCandidate ? 'CORRECTED' : 'NO CORRECTED PREVIEW'}</span>}</div>
      {inCandidate && mode === 'split' && <button className="split-handle" style={{ left: `${split}%` }} onPointerDown={scrubSplit} onKeyDown={e => { if (e.key === 'ArrowLeft' || e.key === 'ArrowRight') { e.stopPropagation(); e.preventDefault(); setSplit(v => clamp(v + (e.key === 'ArrowLeft' ? -2 : 2), 2, 98)); } }} aria-label="Drag comparison divider" aria-valuemin={2} aria-valuemax={98} aria-valuenow={split} role="slider"><span>‹ ›</span></button>}
      {!corrected && <div className="preview-note"><Icon name="split" size={15}/> Your corrected preview will appear here</div>}
      {corrected && !inCandidate && mode !== 'source' && <div className="preview-note">Preview range {timecode(offset, false)}–{timecode(candidateEnd, false)} · seek to the selected seam</div>}
      {mediaError && <div className="media-error" role="alert"><Icon name="warning"/>{mediaError}</div>}
    </div>
    <div className="viewer-controls"><div className="transport"><button className="icon-button" aria-label="Previous frame" title="Previous frame · ←" onClick={() => step(-1)}><Icon name="back" size={16}/></button><button className="play-button" onClick={toggle} aria-label={playing ? 'Pause video' : 'Play video'} title="Play / pause · Space"><Icon name={playing ? 'pause' : 'play'} size={18}/></button><button className="icon-button" aria-label="Next frame" title="Next frame · →" onClick={() => step(1)}><Icon name="forward" size={16}/></button><span className="transport-time mono">{timecode(currentTime)} <span>/ {timecode(project.metadata.duration)}</span></span></div><div className="segmented comparison-switch" aria-label="Comparison view"><button className={mode === 'source' ? 'active' : ''} onClick={() => setMode('source')}>Original</button><button className={mode === 'split' ? 'active' : ''} onClick={() => setMode('split')}><Icon name="split" size={13}/> Split</button><button className={mode === 'candidate' ? 'active' : ''} onClick={() => setMode('candidate')} disabled={!corrected}>Corrected</button></div><div className="transport-right"><button className={`icon-button ${loop && scope === 'seam' ? 'is-on' : ''}`} disabled={!selected} aria-label="Loop selected seam" aria-pressed={loop && scope === 'seam'} title="Loop selected seam" onClick={() => { setScope('seam'); setLoop(v => !v); }}><Icon name="loop"/></button><button className="icon-button" aria-label={muted ? 'Unmute video' : 'Mute video'} onClick={() => setMuted(v => !v)}><Icon name={muted ? 'muted' : 'sound'} size={16}/></button></div></div>
  </section>;
});
