import { useEffect, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import { api, isDesktop } from './api';
import { message } from './format';
import { Icon, WeaveMark } from './Icons';
import type { UpdateState } from './types';
import './updater.css';

export function UpdateControl({ busy }: { busy: boolean }) {
  const [state, setState] = useState<UpdateState | null>(null);
  const [open, setOpen] = useState(false);
  const [error, setError] = useState('');
  const [action, setAction] = useState(false);
  const button = useRef<HTMLButtonElement>(null);
  const dialog = useRef<HTMLElement>(null);
  const stateEvents = useRef(0);

  useEffect(() => {
    let alive = true;
    const initial = stateEvents.current;
    const unsubscribe = api.onUpdateState(next => { stateEvents.current++; if (alive) setState(next); });
    void api.getUpdateState().then(next => { if (alive && stateEvents.current === initial) setState(next); }).catch(() => {});
    return () => { alive = false; unsubscribe(); };
  }, []);

  useEffect(() => api.onOpenUpdates?.(() => { setError(''); setOpen(true); }), []);

  useEffect(() => {
    if (!open) return;
    const focused = document.activeElement as HTMLElement | null;
    dialog.current?.querySelector<HTMLButtonElement>('button')?.focus();
    const key = (event: KeyboardEvent) => {
      // Stop the editor's space/arrow shortcuts when the update dialog is open.
      event.stopPropagation();
      if (event.key === 'Escape') { event.preventDefault(); setOpen(false); return; }
      if (event.key !== 'Tab') return;
      const controls = Array.from(dialog.current?.querySelectorAll<HTMLElement>('button:not(:disabled),a[href], [tabindex="0"]') ?? []);
      const first = controls[0], last = controls[controls.length - 1];
      if (event.shiftKey && (document.activeElement === first || !dialog.current?.contains(document.activeElement))) { event.preventDefault(); last?.focus(); }
      else if (!event.shiftKey && (document.activeElement === last || !dialog.current?.contains(document.activeElement))) { event.preventDefault(); first?.focus(); }
    };
    window.addEventListener('keydown', key, true);
    return () => { window.removeEventListener('keydown', key, true); (focused?.isConnected ? focused : button.current)?.focus(); };
  }, [open]);

  if (!isDesktop) return null;
  const working = state?.status === 'checking' || state?.status === 'downloading' || state?.status === 'installing';
  const processing = busy || state?.busy;
  const available = state?.status === 'available' || state?.status === 'downloading' || state?.status === 'downloaded';
  const progress = Math.round(Math.max(0, Math.min(1, state?.progress ?? 0)) * 100);
  const heading = state?.status === 'downloaded' ? 'Ready for the next thread.'
    : available ? 'A little more seamless.' : 'Made to keep improving.';
  const offerLabel = state?.status === 'downloading' ? `Downloading ${progress}%`
    : state?.status === 'downloaded' ? 'Restart to update' : 'Update available';

  async function perform(operation: () => Promise<UpdateState>) {
    setAction(true); setError('');
    try { setState(await operation()); }
    catch (reason) { setError(message(reason)); }
    finally { setAction(false); }
  }
  function show() {
    setOpen(true); setError('');
    if (state?.status !== 'disabled' && !working && state?.status !== 'downloaded') void perform(() => api.checkForUpdates());
  }

  return <>
    <div className="update-controls">
      <button ref={button} className="version-button" aria-label="About Seamstress and updates" onClick={show}>{state?.currentVersion ? `v${state.currentVersion}` : 'About'}</button>
      {available && <button className="update-offer" onClick={() => { setError(''); setOpen(true); }}><i/>{offerLabel}</button>}
    </div>
    {open && createPortal(<div className="modal-backdrop update-backdrop" onMouseDown={event => { if (event.target === event.currentTarget) setOpen(false); }}>
      <section ref={dialog} className="update-modal" role="dialog" aria-modal="true" aria-labelledby="update-title" aria-describedby="update-description">
        <button className="modal-close icon-button" onClick={() => setOpen(false)} aria-label="Close updates"><Icon name="close"/></button>
        <div className="update-mark"><WeaveMark/></div>
        <span className="eyebrow">SEAMSTRESS {state?.currentVersion && <span className="mono">{state.currentVersion}</span>}</span>
        <h2 id="update-title">{heading}</h2>
        <p id="update-description">{state?.message || 'Checking the update status…'}</p>
        {state?.version && <div className="update-version-row"><span>INSTALLED <strong>v{state.currentVersion}</strong></span><Icon name="arrow" size={15}/><span>AVAILABLE <strong>v{state.version}</strong></span></div>}
        {state?.releaseNotes && <div className="update-notes" tabIndex={0} aria-label="Release notes"><h3>What’s new</h3><div>{state.releaseNotes}</div></div>}
        {state?.status === 'downloading' && <div className="update-download" role="status"><div><span>Downloading & verifying</span><strong className="mono">{progress}%</strong></div><progress max={1} value={state.progress ?? 0} aria-label="Update download progress"/></div>}
        {(error || state?.error) && <p className="update-error" role="alert"><Icon name="warning" size={15}/><span>{error || state?.error}</span></p>}
        {state?.status === 'downloaded' && processing && <p className="update-busy" role="status"><Icon name="film" size={16}/> Finish or cancel processing before restarting. Your update can wait.</p>}
        <div className="update-actions">
          {state?.status === 'downloaded' ? <button className="primary-button full" disabled={Boolean(processing || action)} onClick={() => void perform(() => api.installUpdate())}><Icon name="loop" size={16}/>{processing ? 'Finish processing first' : 'Restart & install update'}</button>
            : state?.status === 'available' ? <button className="primary-button full" disabled={action} onClick={() => void perform(() => api.downloadUpdate())}><Icon name="down" size={16}/> Download update</button>
            : <button className="primary-button full" disabled={Boolean(working || action || !state || state.status === 'disabled')} onClick={() => void perform(() => api.checkForUpdates())}>{working ? <span className="spinner"/> : <Icon name="scan" size={16}/>} {state?.status === 'downloading' ? 'Downloading update…' : state?.status === 'installing' ? 'Restarting…' : state?.status === 'checking' ? 'Checking for updates…' : 'Check for updates'}</button>}
          <button className="text-button" onClick={() => { void api.openReleases().catch(reason => setError(message(reason))); }}><Icon name="external" size={13}/> View releases</button>
        </div>
        <p className="update-footnote">Downloads start only when you choose. Your videos and saved projects stay in place.</p>
        {state?.checkedAt && <p className="update-checked">Last checked {new Date(state.checkedAt).toLocaleString(undefined, { dateStyle: 'medium', timeStyle: 'short' })}</p>}
      </section>
    </div>, document.body)}
  </>;
}
