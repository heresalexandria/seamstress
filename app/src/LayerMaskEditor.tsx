import { useEffect, useRef, useState } from 'react';
import { api } from './api';
import { clamp } from './format';
import { Icon } from './Icons';
import { StudioDialog } from './StudioDialog';
import type { MaskStroke, NeuralPrompt, ReconstructionSummary, SegmentationStatus } from './types';

const colors: Record<MaskStroke['mode'], string> = { include: '#67dec8', exclude: '#eb6eba', protect: '#f6d57d', emission: '#ff9958' };
export function LayerMaskEditor({ summary, disabled, onSave, onClose, onDirtyChange, onCancelJob, processingMessage, model, onSettings, onSegment }: {
  summary: ReconstructionSummary; disabled: boolean;
  onSave(strokes: MaskStroke[]): Promise<boolean>; onClose(): void; onDirtyChange(dirty: boolean): void;
  onCancelJob(): void; processingMessage?: string;
  model?: SegmentationStatus; onSettings(): void; onSegment(prompt: NeuralPrompt): Promise<boolean>;
}) {
  const frames = summary.frames?.length ? [...summary.frames].sort((left, right) => left.frame - right.frame) : [{ frame: summary.sourceFrame, sourceFramePath: summary.sourceFramePath, masks: Object.fromEntries(summary.layers.map(layer => [layer.id, layer.maskPreviewPath])) }];
  const [frameIndex, setFrameIndex] = useState(Math.max(0, frames.findIndex(item => item.frame === summary.sourceFrame)));
  const [layerId, setLayerId] = useState(summary.layers.find(layer => layer.role === 'foreground')?.id ?? summary.layers[0]?.id ?? '');
  const [mode, setMode] = useState<MaskStroke['mode']>('include');
  const [radius, setRadius] = useState(Math.max(2, Math.round(summary.width / 128)));
  const [opacity, setOpacity] = useState(.4);
  const [zoom, setZoom] = useState(1);
  const [strokes, setStrokes] = useState<MaskStroke[]>([]);
  const [cursor, setCursor] = useState<[number, number] | null>(null);
  const [error, setError] = useState('');
  const [discarding, setDiscarding] = useState(false);
  const [saving, setSaving] = useState(false);
  const [segmenting, setSegmenting] = useState(false);
  const [tool, setTool] = useState<'brush' | 'prompt'>('brush');
  const [points, setPoints] = useState<NeuralPrompt['points']>([]);
  const canvas = useRef<HTMLCanvasElement>(null);
  const pointer = useRef<number | null>(null);
  const active = frames[frameIndex];
  const layer = summary.layers.find(item => item.id === layerId);
  const dirty = strokes.length > 0 || points.length > 0;
  const locked = disabled || saving || segmenting;
  useEffect(() => { onDirtyChange(dirty); }, [dirty, onDirtyChange]);
  useEffect(() => () => onDirtyChange(false), [onDirtyChange]);
  useEffect(() => { setError(''); }, [active?.frame, layerId]);
  useEffect(() => { if (layer?.role !== 'foreground' && mode === 'emission') setMode('include'); }, [layer?.role, mode]);
  useEffect(() => {
    const context = canvas.current?.getContext('2d');
    if (!context) return;
    context.clearRect(0, 0, summary.width, summary.height);
    context.lineCap = 'round'; context.lineJoin = 'round';
    for (const stroke of strokes) {
      if (stroke.frame !== active?.frame || stroke.layer_id !== layerId || !stroke.points.length) continue;
      context.globalAlpha = .6; context.strokeStyle = colors[stroke.mode]; context.fillStyle = colors[stroke.mode]; context.lineWidth = stroke.radius * 2;
      context.beginPath(); context.moveTo(...stroke.points[0]);
      for (const point of stroke.points.slice(1)) context.lineTo(...point);
      context.stroke();
      context.beginPath(); context.arc(...stroke.points[0], stroke.radius, 0, Math.PI * 2); context.fill();
    }
    if (cursor) {
      context.globalAlpha = 1; context.lineWidth = Math.max(1, summary.width / 1000); context.strokeStyle = colors[mode];
      context.beginPath(); context.arc(...cursor, radius, 0, Math.PI * 2); context.stroke();
    }
    for (const point of points) {
      const dotRadius = Math.max(5, summary.width / 160);
      context.globalAlpha = 1; context.fillStyle = point.label === 1 ? colors.include : colors.exclude; context.strokeStyle = '#0f2923'; context.lineWidth = 2;
      context.beginPath(); context.arc(point.x, point.y, dotRadius, 0, Math.PI * 2); context.fill(); context.stroke();
      context.beginPath(); context.moveTo(point.x - dotRadius / 2, point.y); context.lineTo(point.x + dotRadius / 2, point.y);
      if (point.label === 1) { context.moveTo(point.x, point.y - dotRadius / 2); context.lineTo(point.x, point.y + dotRadius / 2); }
      context.stroke();
    }
  }, [strokes, points, active?.frame, layerId, cursor, radius, mode, summary.width, summary.height]);
  function coordinates(event: React.PointerEvent<HTMLCanvasElement>): [number, number] {
    const rect = event.currentTarget.getBoundingClientRect();
    return [Math.round(clamp((event.clientX - rect.left) / rect.width * summary.width, 0, summary.width - 1)), Math.round(clamp((event.clientY - rect.top) / rect.height * summary.height, 0, summary.height - 1))];
  }
  function begin(point: [number, number]) {
    if (locked || !active || !layerId) return;
    setDiscarding(false);
    if (tool === 'prompt') { setPoints(previous => [...previous, { x: point[0], y: point[1], label: mode === 'exclude' ? 0 : 1 }]); return; }
    setStrokes(previous => [...previous, { frame: active.frame, layer_id: layerId, mode, radius, points: [point] }]);
  }
  function finish(event: React.PointerEvent<HTMLCanvasElement>) {
    if (pointer.current === event.pointerId) {
      pointer.current = null;
      if (event.currentTarget.hasPointerCapture(event.pointerId)) event.currentTarget.releasePointerCapture(event.pointerId);
    }
  }
  async function save() {
    if (!dirty || locked) return;
    setSaving(true); setError('');
    try { if (await onSave(strokes)) { onDirtyChange(false); onClose(); } else setError('Mask propagation did not complete. Your strokes are still here; retry or discard them.'); }
    finally { setSaving(false); }
  }
  async function segment() {
    if (locked || !points.some(point => point.label === 1)) return;
    setSegmenting(true); setError('');
    try {
      if (await onSegment({ frame: active.frame, layerId, points })) setPoints([]);
      else setError('Local selection did not complete. Your prompt points remain available to retry.');
    } finally { setSegmenting(false); }
  }
  function close() { if (locked) return; if (dirty) setDiscarding(true); else onClose(); }
  return <StudioDialog title="Keep the right details." label="Layer mask editor" onClose={close} className="mask-editor-dialog">
    <p className="studio-intro">Paint on the original frame. Saved strokes become keyframes and the local tracker propagates them through the repair window. Inspect hands, thin outlines and bright effects after rendering.</p>
    <div className="mask-editor-toolbar">
      <label>Layer<select aria-label="Mask layer" value={layerId} disabled={locked || points.length > 0} onChange={event => setLayerId(event.target.value)}>{summary.layers.map(item => <option key={item.id} value={item.id}>{item.name} · {item.role}</option>)}</select></label>
      <div className="segmented" aria-label="Mask brush mode">{(['include', 'exclude'] as const).map(value => <button key={value} className={mode === value ? 'active' : ''} aria-pressed={mode === value} disabled={locked} onClick={() => setMode(value)}>{value === 'include' ? 'Include' : 'Exclude'}</button>)}</div>
      <label>Brush radius<input aria-label="Brush radius" type="number" min={1} max={Math.max(1, Math.round(Math.min(summary.width, summary.height) / 4))} value={radius} disabled={locked} onChange={event => setRadius(clamp(Number(event.target.value) || 1, 1, Math.round(Math.min(summary.width, summary.height) / 4)))}/><span className="mono">px</span></label>
      <label>Zoom<select aria-label="Mask editor zoom" value={zoom} onChange={event => setZoom(Number(event.target.value))}>{[1, 2, 4].map(value => <option key={value} value={value}>{value === 1 ? 'Fit' : `${value}×`}</option>)}</select></label>
      <button className="small-button" disabled={!dirty || locked} onClick={() => tool === 'prompt' ? setPoints(previous => previous.slice(0, -1)) : setStrokes(previous => previous.slice(0, -1))}>{tool === 'prompt' ? 'Undo point' : 'Undo stroke'}</button>
    </div>
    <div className="mask-editor-canvas-scroll"><div className="mask-editor-image" style={{ width: `${zoom * 100}%`, aspectRatio: `${summary.width}/${summary.height}` }}>
      <img src={api.mediaUrl(active.sourceFramePath)} alt={`Original source frame ${active.frame}`} draggable={false} onError={() => setError('The original frame could not be loaded. Close and reopen the candidate, or make a new proposal.')}/>
      {active.masks[layerId] && <div className="mask-existing-overlay" style={{ opacity, maskImage: `url("${api.mediaUrl(active.masks[layerId])}")`, maskMode: 'luminance' }}/>}
      <canvas ref={canvas} width={summary.width} height={summary.height} tabIndex={locked ? -1 : 0} aria-label={`Paint ${layer?.name ?? 'layer'} mask on source frame ${active.frame}`} aria-describedby="mask-keyboard-help" onPointerDown={event => {
        if (locked || event.button !== 0) return;
        event.preventDefault(); event.currentTarget.focus(); event.currentTarget.setPointerCapture(event.pointerId); pointer.current = event.pointerId;
        const point = coordinates(event); setCursor(point); begin(point);
      }} onPointerMove={event => {
        const point = coordinates(event); setCursor(point);
        if (pointer.current !== event.pointerId || locked || tool === 'prompt') return;
        setStrokes(previous => {
          const last = previous[previous.length - 1]; if (!last) return previous;
          const tail = last.points[last.points.length - 1];
          if (Math.hypot(point[0] - tail[0], point[1] - tail[1]) < Math.max(1, last.radius / 4) || last.points.length >= 10000) return previous;
          return [...previous.slice(0, -1), { ...last, points: [...last.points, point] }];
        });
      }} onPointerUp={finish} onPointerCancel={finish} onLostPointerCapture={() => { pointer.current = null; }} onPointerLeave={() => { if (pointer.current === null) setCursor(null); }} onKeyDown={event => {
        if (locked) return;
        const point: [number, number] = cursor ?? [Math.round(summary.width / 2), Math.round(summary.height / 2)];
        const step = event.shiftKey ? 1 : Math.max(1, Math.round(radius / 2));
        if (event.key.startsWith('Arrow')) { event.preventDefault(); setCursor([clamp(point[0] + (event.key === 'ArrowRight' ? step : event.key === 'ArrowLeft' ? -step : 0), 0, summary.width - 1), clamp(point[1] + (event.key === 'ArrowDown' ? step : event.key === 'ArrowUp' ? -step : 0), 0, summary.height - 1)]); }
        if (event.code === 'Space') { event.preventDefault(); begin(point); }
      }}/>
    </div></div>
    <div className="mask-frame-controls"><button className="icon-button" aria-label="Previous mask frame" disabled={locked || points.length > 0 || frameIndex === 0} onClick={() => setFrameIndex(index => index - 1)}><Icon name="back"/></button><input type="range" aria-label="Mask source frame" min={0} max={frames.length - 1} value={frameIndex} disabled={locked || points.length > 0} onChange={event => setFrameIndex(Number(event.target.value))}/><button className="icon-button" aria-label="Next mask frame" disabled={locked || points.length > 0 || frameIndex === frames.length - 1} onClick={() => setFrameIndex(index => index + 1)}><Icon name="forward"/></button><label>Frame<select aria-label="Exact mask source frame" value={frameIndex} disabled={locked || points.length > 0} onChange={event => setFrameIndex(Number(event.target.value))}>{frames.map((item, index) => <option key={item.frame} value={index}>{item.frame}{layer?.keyframes.includes(item.frame) ? ' · keyframe' : ''}{strokes.some(stroke => stroke.frame === item.frame && stroke.layer_id === layerId) ? ' · edited' : ''}</option>)}</select></label></div>
    <div className="mask-editor-legend"><span><i style={{ background: colors.include }}/> Include layer</span><span><i style={{ background: colors.exclude }}/> Exclude layer</span><label>Existing mask<input aria-label="Mask overlay opacity" type="range" min={0} max={.8} step={.05} value={opacity} onChange={event => setOpacity(Number(event.target.value))}/></label></div>
    <details className="reconstruction-details"><summary>Local model selection</summary><p className="field-help">Click Include points inside the subject and Exclude points outside it. MobileSAM runs on this device, then the mask is propagated locally. Review the new selection before rendering.</p>{model?.available ? <><div className="mask-detail-tools"><button className="small-button" aria-pressed={tool === 'brush'} disabled={locked || points.length > 0} onClick={() => setTool('brush')}>Paint strokes</button><button className="small-button" aria-pressed={tool === 'prompt'} disabled={locked || strokes.length > 0} onClick={() => { setTool('prompt'); setMode('include'); }}>Place selection points</button></div>{strokes.length > 0 && <p className="field-help">Save or undo brush strokes before using model selection.</p>}{tool === 'prompt' && <div className="mask-detail-tools"><button className="primary-button" disabled={locked || !points.some(point => point.label === 1)} onClick={() => void segment()}>Run local selection</button><button className="small-button" disabled={locked || !points.length} onClick={() => setPoints([])}>Clear selection points</button></div>}</> : <button className="small-button" disabled={locked} onClick={onSettings}>Set up local selection model</button>}</details>
    <details className="reconstruction-details"><summary>Fine detail tools</summary><div className="mask-detail-tools"><button className="small-button" aria-pressed={mode === 'protect'} disabled={locked || tool === 'prompt'} onClick={() => setMode('protect')}>Keep original detail</button><button className="small-button" aria-pressed={mode === 'emission'} disabled={locked || tool === 'prompt' || layer?.role !== 'foreground'} onClick={() => setMode('emission')}>Warm glow</button></div><p className="field-help">Keep original detail preserves opaque source pixels such as claws or ink lines. Warm glow marks a bright additive effect on the foreground layer for review; use it only on an actual light or flame.</p></details>
    <p className="field-help" id="mask-keyboard-help">Native {summary.width} × {summary.height} coordinates · arrow keys move the brush, Shift moves one pixel, Space paints a dot. Cyan and magenta strokes show pending edits; saved masks appear after propagation.</p>
    {error && <p className="reconstruction-error" role="alert">{error}</p>}
    {discarding && <div className="reconstruction-warning" role="alert">You have unsaved mask strokes.<div className="mask-discard-actions"><button className="small-button" onClick={() => setDiscarding(false)}>Keep editing</button><button className="small-button" onClick={() => { onDirtyChange(false); onClose(); }}>Discard strokes & close</button></div></div>}
    <div className="studio-dialog-actions"><span className="field-help" role={saving || segmenting ? 'status' : undefined}>{saving || segmenting ? (processingMessage || 'Processing layer selection…') : tool === 'prompt' ? `${points.length} pending selection points` : `${strokes.length} pending ${strokes.length === 1 ? 'stroke' : 'strokes'}`}</span>{saving || segmenting ? <button className="small-button" onClick={onCancelJob}>Cancel propagation</button> : <button className="primary-button" disabled={!strokes.length || locked} onClick={() => void save()}>Save & propagate masks</button>}</div>
  </StudioDialog>;
}
