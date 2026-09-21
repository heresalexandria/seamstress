import { useEffect, useState } from 'react';
import { api } from './api';
import { message } from './format';
import { Icon } from './Icons';
import { StudioDialog } from './StudioDialog';
import type { AISettings as Settings, SegmentationStatus } from './types';
import './reconstruction.css';

export function AISettings({ onClose, onChange, busy, model, canInstall, onInstall, onCancelJob, processingMessage }: {
  onClose(): void; onChange(state: Settings): void; busy: boolean; model?: SegmentationStatus;
  canInstall: boolean; onInstall(): Promise<boolean>; onCancelJob(): void; processingMessage?: string;
}) {
  const [settings, setSettings] = useState<Settings | null>(null);
  const [key, setKey] = useState('');
  const [working, setWorking] = useState(false);
  const [error, setError] = useState('');
  const [notice, setNotice] = useState('');
  const [installing, setInstalling] = useState(false);
  useEffect(() => {
    let active = true;
    void api.getAISettings().then(next => { if (active) setSettings(next); }).catch(reason => { if (active) setError(message(reason)); });
    return () => { active = false; };
  }, []);
  async function update(remove = false) {
    if (working || busy) return;
    setWorking(true); setError(''); setNotice('');
    const value = key.trim(); setKey('');
    try {
      const next = remove ? await api.clearAIKey() : await api.setAIKey(value);
      setSettings(next); onChange(next); setNotice(remove ? 'Saved key removed.' : 'Key saved securely. It will be used only when you opt into an AI background request.');
    } catch (reason) { setError(message(reason)); }
    finally { setWorking(false); }
  }
  async function install() {
    if (working || busy || !canInstall) return;
    setInstalling(true); setError(''); setNotice('');
    try { if (await onInstall()) setNotice('Local model download finished. Its availability is shown below.'); else setError('Model setup did not complete. You can retry; local framing and color correction remain available.'); }
    finally { setInstalling(false); }
  }
  return <StudioDialog title="Optional AI assistance." label="AI settings" onClose={() => { if (!working && !installing) { setKey(''); onClose(); } }} className="ai-settings-dialog">
    <p className="studio-intro">Local analysis and correction work without an API key. OpenAI can help reconstruct missing background behind a layer when you explicitly enable it for a repair.</p>
    <div className="reconstruction-status"><Icon name={settings?.configured ? 'check' : 'sliders'} size={16}/><span>{settings ? settings.configured ? 'OpenAI key saved' : 'No OpenAI key saved' : 'Reading secure storage…'}</span></div>
    {settings && !settings.secureStorageAvailable && <p className="reconstruction-warning">Secure credential storage is unavailable on this device. A key cannot be saved here.</p>}
    <form onSubmit={event => { event.preventDefault(); void update(); }}>
      <label className="field-label" htmlFor="openai-api-key">{settings?.configured ? 'Replace OpenAI API key' : 'OpenAI API key'}</label>
      <input id="openai-api-key" type="password" value={key} onChange={event => setKey(event.target.value)} autoComplete="off" spellCheck={false} autoCapitalize="none" disabled={working || busy || !settings?.secureStorageAvailable} placeholder="Paste your API key"/>
      <p className="field-help">Stored using your operating system’s secure storage. The saved key is never shown again or included in project files. API usage is billed to your OpenAI account.</p>
      {error && <p className="reconstruction-error" role="alert">{error}</p>}
      {notice && <p className="reconstruction-note" role="status">{notice}</p>}
      <div className="studio-dialog-actions"><button type="button" className="text-button" disabled={!settings?.configured || working || busy} onClick={() => void update(true)}>Remove saved key</button><button className="primary-button" type="submit" disabled={!key.trim() || working || busy || !settings?.secureStorageAvailable}>{working ? 'Saving…' : 'Save key securely'}</button></div>
    </form>
    <section className="local-model-settings" aria-label="Local segmentation model"><span className="eyebrow">LOCAL LAYER SELECTION</span><h3>MobileSAM</h3><p className="field-help">An optional model helps select a subject using foreground and background clicks. It runs on this computer; video frames are not uploaded. Brush editing works without it.</p><div className="reconstruction-status"><Icon name={model?.available ? 'check' : 'down'} size={15}/><span>{model ? model.available ? 'Local model ready' : model.downloaded ? 'Model downloaded · runtime unavailable' : 'Model not downloaded' : 'Model status unavailable'}</span></div>{model?.reason && <p className="field-help">{model.reason}</p>}<p className="field-help">{((model?.downloadBytes ?? 44658416) / 1e6).toFixed(1)} MB download · {model?.license || 'model license provided with download'}{model?.publisher ? ` · ${model.publisher}` : ''}</p>{installing ? <><p className="reconstruction-note" role="status">{processingMessage || 'Downloading and verifying model files…'}</p><button className="small-button full" onClick={onCancelJob}>Cancel model download</button></> : <button className="secondary-button full" disabled={!canInstall || busy || working || !model || model.available || !model.runtimeAvailable} onClick={() => void install()}>{model?.available ? 'Model ready on this device' : 'Download local model'}</button>}{!canInstall && !installing && <p className="field-help">Open a project, select an enabled seam and save pending edits before downloading.</p>}</section>
  </StudioDialog>;
}
