import { useRef, useState } from 'react';
import { api } from './api';
import { clamp, timecode } from './format';
import { Icon } from './Icons';
import type { Project, Seam } from './types';

type Props = { project: Project; selected?: Seam; frame: number; disabled: boolean; onSeek(frame: number): void; onSelect(seam: Seam): void; onMove(seam: Seam, frame: number): void; onAdd(): void };
export function Timeline({ project, selected, frame, disabled, onSeek, onSelect, onMove, onAdd }: Props) {
  const track = useRef<HTMLDivElement>(null);
  const markerWasDragged = useRef(false);
  const [focused, setFocused] = useState(false);
  const [drag, setDrag] = useState<{ id: string; frame: number } | null>(null);
  const count = project.metadata.frame_count;
  const fps = project.metadata.fps;
  const center = selected?.frame ?? frame;
  const lo = focused ? Math.max(0, center - Math.round(fps * 5)) : 0;
  const hi = focused ? Math.min(count - 1, lo + Math.round(fps * 10)) : count - 1;
  const position = (f: number) => (f - lo) / Math.max(1, hi - lo) * 100;
  const eventFrame = (clientX: number) => {
    const rect = track.current?.getBoundingClientRect();
    return rect ? clamp(Math.round(lo + (clientX - rect.left) / rect.width * (hi - lo)), 0, count - 1) : frame;
  };
  const thumbs = (project.artifacts.thumbnails ?? []).filter(t => t.frame >= lo && t.frame <= hi);

  function dragMarker(event: React.PointerEvent<HTMLButtonElement>, seam: Seam) {
    event.stopPropagation(); onSelect(seam);
    markerWasDragged.current = false;
    if (disabled) return;
    const target = event.currentTarget, startX = event.clientX;
    let next = seam.frame;
    target.setPointerCapture(event.pointerId);
    const move = (e: PointerEvent) => { if (Math.abs(e.clientX - startX) < 3 && next === seam.frame) return; markerWasDragged.current = true; next = clamp(eventFrame(e.clientX), 1, count - 1); setDrag({ id: seam.id, frame: next }); };
    const stop = (e: PointerEvent) => {
      target.removeEventListener('pointermove', move); target.removeEventListener('pointerup', stop); target.removeEventListener('pointercancel', stop); setDrag(null);
      if (e.type !== 'pointercancel' && next !== seam.frame) { onMove(seam, next); onSeek(next - 1); }
    };
    target.addEventListener('pointermove', move); target.addEventListener('pointerup', stop); target.addEventListener('pointercancel', stop);
  }

  function scrub(event: React.PointerEvent<HTMLDivElement>) {
    if ((event.target as HTMLElement).closest('button')) return;
    const target = event.currentTarget;
    onSeek(eventFrame(event.clientX)); target.setPointerCapture(event.pointerId);
    const move = (e: PointerEvent) => onSeek(eventFrame(e.clientX));
    const stop = () => { target.removeEventListener('pointermove', move); target.removeEventListener('pointerup', stop); target.removeEventListener('pointercancel', stop); };
    target.addEventListener('pointermove', move); target.addEventListener('pointerup', stop); target.addEventListener('pointercancel', stop);
  }

  return <section className="timeline-panel" aria-label="Source timeline"><div className="timeline-title"><div><span className="eyebrow">THE THREAD</span><span className="timeline-count">{project.seams.length} {project.seams.length === 1 ? 'seam' : 'seams'}</span></div><div className="timeline-actions"><button className="text-button" onClick={() => setFocused(v => !v)} aria-pressed={focused}>{focused ? 'Fit whole video' : 'Focus 10 seconds'}</button><button className="small-button" onClick={onAdd} disabled={disabled || count < 2}><Icon name="plus" size={13}/> Add seam</button></div></div><div className="timeline-track" ref={track} onPointerDown={scrub} role="slider" tabIndex={0} aria-label="Seek source video" aria-valuemin={lo} aria-valuemax={hi} aria-valuenow={frame} onKeyDown={e => { if (e.key === 'ArrowLeft' || e.key === 'ArrowRight') { e.preventDefault(); e.stopPropagation(); onSeek(clamp(frame + (e.key === 'ArrowLeft' ? -1 : 1), 0, count - 1)); } }}><div className="timeline-ruler">{Array.from({ length: 7 }, (_, index) => <span key={index} style={{ left: `${index / 6 * 100}%` }}>{timecode((lo + (hi - lo) * index / 6) / fps, false)}</span>)}</div><div className="thumbnail-strip">{thumbs.length ? thumbs.map(t => <img key={t.frame} src={api.mediaUrl(t.path)} alt="" draggable={false}/>) : <div className="thumbnail-pending"><Icon name="film" size={17}/><span>Source timeline</span></div>}</div><div className="timeline-seam-band"/>{project.seams.map((seam, index) => { const f = drag?.id === seam.id ? drag.frame : seam.frame; return f < lo || f > hi ? null : <button key={seam.id} className={`seam-marker ${seam.id === selected?.id ? 'selected' : ''} ${!seam.enabled ? 'disabled-seam' : ''} ${seam.confidence !== undefined && seam.confidence < .6 ? 'uncertain' : ''}`} style={{ left: `${position(f)}%` }} aria-label={`Seam ${index + 1}, ${timecode(f / fps)}${seam.enabled ? '' : ', disabled'}`} onPointerDown={e => dragMarker(e, seam)} onClick={() => { if (markerWasDragged.current) { markerWasDragged.current = false; return; } onSelect(seam); }} title={`Seam ${index + 1} · ${timecode(f / fps)} · drag to move`}><span className="marker-label">{String(index + 1).padStart(2, '0')}</span><span className="marker-line"/><span className="marker-diamond"/></button>; })}{frame >= lo && frame <= hi && <div className="playhead" style={{ left: `${position(frame)}%` }}><span/></div>}</div><div className="timeline-footer"><span><i className="legend-seam"/> Seam <i className="legend-review"/> Needs review</span><span className="mono">FRAME {frame.toLocaleString()} <span className="dim">/ {(count - 1).toLocaleString()}</span></span><span>Drag a marker to adjust · ← → step</span></div></section>;
}
