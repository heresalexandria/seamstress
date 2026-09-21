import { useEffect, useState } from 'react';
import { Icon } from './Icons';
import type { CameraRate, ManualFraming, Project, Seam, SeamCorrection as Correction, SeamResult } from './types';
import './seam-correction.css';

export const defaultCorrection = (): Correction => ({
  geometry: 'auto', partial_recovery: true, endpoint_recovery: true,
  cadence: true, rate_easing: true, color: 'auto',
});

type ManualDraft = { matrix: string[]; pre: string[]; post: string[]; edited: boolean };
function fields(manual?: ManualFraming): ManualDraft {
  return {
    matrix: (manual?.right_to_left_matrix.slice(0, 2).flat() ?? [1, 0, 0, 0, 1, 0]).map(String),
    pre: (manual?.pre_rate ?? [0, 0, 0, 0]).map(String),
    post: (manual?.post_rate ?? [0, 0, 0, 0]).map(String), edited: false,
  };
}
const geometryLabels: Record<string, string> = {
  auto: 'Automatic framing', partial: 'Partial framing recovered', endpoint: 'Repeated endpoint recovered',
  manual: 'Custom framing', off: 'Framing off', excluded: 'Framing skipped',
};
const colorLabels: Record<string, string> = {
  auto: 'Automatic', tone: 'Tone only', local: 'Local color', 'tone + local': 'Tone and local color',
  off: 'Off', unchanged: 'Unchanged', excluded: 'Skipped', accepted: 'Matched',
};
const matrixLabels = ['Horizontal scale (a)', 'Horizontal shear (b)', 'Horizontal shift (tx)', 'Vertical shear (c)', 'Vertical scale (d)', 'Vertical shift (ty)'];
const rateLabels = ['log scale/frame', 'rotation rad/frame', 'x pixels/frame', 'y pixels/frame'];

export function SeamCorrection({ seam, project, result, disabled, onSave, onImport, onDirtyChange }: {
  seam: Seam; project: Project; result?: SeamResult; disabled: boolean;
  onSave: (correction: Correction) => Promise<boolean>; onImport: () => Promise<void>;
  onDirtyChange: (dirty: boolean) => void;
}) {
  const savedKey = JSON.stringify({ ...defaultCorrection(), ...seam.correction });
  const [draft, setDraft] = useState<Correction>(() => JSON.parse(savedKey));
  const [manual, setManual] = useState<ManualDraft>(() => fields(seam.correction?.manual));
  const [localError, setLocalError] = useState('');
  const dirty = JSON.stringify(draft) !== savedKey || manual.edited;
  useEffect(() => {
    const next: Correction = JSON.parse(savedKey);
    setDraft(next); setManual(fields(next.manual)); setLocalError('');
  }, [savedKey, seam.frame]);
  useEffect(() => { onDirtyChange(dirty); }, [dirty, onDirtyChange]);
  useEffect(() => () => onDirtyChange(false), [onDirtyChange]);

  function reset() {
    const next: Correction = JSON.parse(savedKey);
    setDraft(next); setManual(fields(next.manual)); setLocalError('');
  }
  function identity(): ManualFraming {
    return { right_to_left_matrix: [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
      pre_rate: [0, 0, 0, 0], post_rate: [0, 0, 0, 0], ease_rate: false,
      provenance: { kind: 'manual', label: 'Edited in Seamstress', source_sha256: project.sourceSha256, frame: seam.frame } };
  }
  function useMeasurements(value: ManualFraming) {
    const copy = structuredClone(value);
    setDraft(current => ({ ...current, geometry: 'manual', rate_easing: copy.ease_rate, manual: copy }));
    setManual(fields(copy)); setLocalError('');
  }
  function geometry(value: Correction['geometry']) {
    if (value === 'manual' && draft.manual) {
      // Switching modes must not discard uncommitted matrix/rate edits.
      setDraft(current => ({ ...current, geometry: 'manual', rate_easing: current.manual!.ease_rate }));
    } else if (value === 'manual') useMeasurements(result?.manual ?? identity());
    else setDraft(current => ({ ...current, geometry: value }));
  }
  function edit(field: 'matrix' | 'pre' | 'post', index: number, value: string) {
    setManual(current => ({ ...current, [field]: current[field].map((item, at) => at === index ? value : item), edited: true }));
    setLocalError('');
  }
  function easing(enabled: boolean) {
    setDraft(current => ({ ...current, rate_easing: enabled,
      ...(current.manual && current.geometry === 'manual' ? { manual: { ...current.manual, ease_rate: enabled } } : {}) }));
    if (draft.geometry === 'manual') setManual(current => ({ ...current, edited: true }));
  }
  async function apply() {
    setLocalError('');
    let next = { ...draft };
    if (draft.geometry === 'manual' || (manual.edited && draft.manual)) {
      const numeric = (values: string[]) => values.map(value => {
        if (!value.trim() || !Number.isFinite(Number(value))) throw new Error('Every custom value must be a finite number.');
        return Number(value);
      });
      try {
        const matrix = numeric(manual.matrix);
        const provenance = manual.edited ? { kind: 'manual', label: 'Edited in Seamstress', source_sha256: project.sourceSha256, frame: seam.frame } : draft.manual?.provenance;
        next = { ...next, manual: { right_to_left_matrix: [matrix.slice(0, 3), matrix.slice(3, 6), [0, 0, 1]],
          pre_rate: numeric(manual.pre) as CameraRate, post_rate: numeric(manual.post) as CameraRate,
          ease_rate: draft.geometry === 'manual' ? draft.rate_easing : draft.manual?.ease_rate ?? false,
          ...(provenance ? { provenance } : {}) } };
      } catch (error) { setLocalError((error as Error).message); return; }
    }
    if (await onSave(next)) { setDraft(next); setManual(fields(next.manual)); }
  }

  return <>
    <section className="inspector-section seam-correction" aria-label="Seam correction settings">
      <div className="section-caption">THIS SEAM’S CORRECTION</div>
      <label className="select-row" htmlFor="correction-geometry"><span>Framing</span><select aria-label="Framing" id="correction-geometry" value={draft.geometry} disabled={disabled} onChange={event => geometry(event.target.value as Correction['geometry'])}><option value="auto">Automatic</option><option value="off">Off</option><option value="manual">Custom</option></select></label>
      <label className="select-row" htmlFor="correction-color"><span>Color</span><select aria-label="Color" id="correction-color" value={draft.color} disabled={disabled} onChange={event => setDraft(current => ({ ...current, color: event.target.value as Correction['color'] }))}><option value="auto">Automatic</option><option value="tone">Tone only</option><option value="off">Off</option></select></label>
      <p className="field-help">Automatic uses supported measurements. Framing and color can be handled independently for each join.</p>
      <details className="correction-advanced">
        <summary>Automatic checks</summary>
        <p className="field-help">These checks keep their evidence requirements. Turning one off never forces a rejected correction.</p>
        {([
          ['partial_recovery', 'Partial framing recovery', 'Recover a common framing change when scene motion differs.'],
          ['endpoint_recovery', 'Repeated endpoint recovery', 'Recognize a reframed repeated endpoint with broad stationary evidence.'],
          ['cadence', 'Animation cadence check', 'Measure camera speed over complete drawing cycles.'],
        ] as const).map(([key, label, detail]) => <label className="correction-check" key={key}><input type="checkbox" aria-label={label} checked={draft[key]} disabled={disabled || draft.geometry !== 'auto'} onChange={event => setDraft(current => ({ ...current, [key]: event.target.checked }))}/><span>{label}<small>{detail}</small></span></label>)}
      </details>
      <label className="correction-check"><input type="checkbox" aria-label="Camera-rate easing" checked={draft.rate_easing} disabled={disabled || draft.geometry === 'off'} onChange={event => easing(event.target.checked)}/><span>Camera-rate easing<small>Smooth a change in camera speed when supported. Custom framing uses your chosen rates.</small></span></label>
      <div className="correction-actions">
        <button className="secondary-button full" disabled={disabled || !result?.manual} onClick={() => result?.manual && useMeasurements(result.manual)}><Icon name="sliders" size={13}/> Use analyzed framing</button>
        <button className="secondary-button full" disabled={disabled || dirty} onClick={() => void onImport()}><Icon name="folder" size={13}/> Import reviewed framing</button>
      </div>
      <p className="field-help">Use analyzed framing copies accepted measurements into a custom draft. Import reads this exact seam from a calibration for the same source video.</p>
      {draft.geometry === 'manual' && <div className="custom-framing">
        <div className="custom-provenance"><Icon name="diamond" size={12}/><span>{manual.edited ? 'Edited in Seamstress' : draft.manual?.provenance?.label || 'Custom measurements'}<small>Bound to source frame {seam.frame}</small></span></div>
        <details className="correction-advanced custom-values">
          <summary>Edit custom measurements</summary>
          <p className="field-help">Maps the incoming picture onto the outgoing picture: x′ = a·x + b·y + tx; y′ = c·x + d·y + ty. Shifts use source pixels. A value of 1 on both scales preserves size. These measurements are distributed across both sides of the join.</p>
          <div className="correction-numbers">{matrixLabels.map((label, index) => <label key={label}>{label}<input type="text" inputMode="decimal" value={manual.matrix[index]} disabled={disabled} onChange={event => edit('matrix', index, event.target.value)}/></label>)}</div>
          <p className="field-help">Camera rates describe ordinary motion before and after the join. The before rate still separates motion from a framing jump when easing is off. Leave zero only for a stationary camera or a measured edit that already excludes motion.</p>
          {(['pre', 'post'] as const).map((side, sideIndex) => <fieldset className="correction-rates" key={side}><legend>{sideIndex === 0 ? 'Before' : 'After'} the seam</legend><div className="correction-numbers">{rateLabels.map((label, index) => <label key={label}>{label}<input aria-label={`${sideIndex === 0 ? 'Before' : 'After'} ${label}`} type="text" inputMode="decimal" value={manual[side][index]} disabled={disabled} onChange={event => edit(side, index, event.target.value)}/></label>)}</div></fieldset>)}
        </details>
        <p className="field-help">Custom framing is your review decision. Analysis checks its bounds and source coverage, then refits color in the chosen geometry.</p>
      </div>}
      {localError && <p className="correction-error" role="alert">{localError}</p>}
      <div className="correction-save"><button className="primary-button" disabled={disabled || !dirty} onClick={() => void apply()}>Apply seam settings</button><button className="text-button" disabled={disabled || !dirty} onClick={reset}>Reset changes</button></div>
      <p className={`field-help ${dirty ? 'correction-unsaved' : ''}`} role="status">{dirty ? 'Unapplied changes. Apply or reset before continuing.' : 'Settings are saved with this project. After a change, analyze again and regenerate previews.'}</p>
    </section>
    <section className="inspector-section correction-result" aria-label="Applied correction">
      <div className="section-caption">APPLIED CORRECTION</div>
      {!seam.enabled ? <p className="field-help">This seam is disabled.</p> : result ? <>
        <strong className={result.geometry === 'excluded' ? 'correction-review' : ''}>{geometryLabels[result.geometry] ?? result.geometry}</strong>
        {result.geometryReason && <p className="field-help">{result.geometryReason}</p>}
        <dl><div><dt>Cadence adjustment</dt><dd>{result.cadence ? 'Applied' : 'Not applied'}</dd></div><div><dt>Camera-rate easing</dt><dd>{result.rateEasing ? 'Applied' : 'Not applied'}</dd></div><div><dt>Color</dt><dd>{colorLabels[result.color] ?? result.color}</dd></div></dl>
        {Boolean(result.notes?.length) && <ul className="reason-list">{result.notes.map((note, index) => <li key={index}>{note}</li>)}</ul>}
      </> : <p className="field-help">Analyze to see which corrections were applied and which need review. Detection confidence only locates the seam.</p>}
    </section>
  </>;
}
