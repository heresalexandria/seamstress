import { useEffect, useRef, type ReactNode } from 'react';
import { createPortal } from 'react-dom';
import { Icon } from './Icons';

/** Focus stays inside an explicitly opened dialog, never activates the app window. */
export function StudioDialog({ title, label, children, onClose, className = '' }: {
  title: string; label: string; children: ReactNode; onClose(): void; className?: string;
}) {
  const ref = useRef<HTMLElement>(null);
  const close = useRef(onClose); close.current = onClose;
  useEffect(() => {
    const previous = document.activeElement as HTMLElement | null;
    ref.current?.focus();
    return () => { if (previous?.isConnected) previous.focus(); };
  }, []);
  return createPortal(<div className="modal-backdrop reconstruction-backdrop" onMouseDown={event => {
    if (event.target === event.currentTarget) close.current();
  }}><section ref={ref} className={`studio-dialog ${className}`} role="dialog" aria-modal="true" aria-label={label} tabIndex={-1} onKeyDown={event => {
    event.stopPropagation();
    if (event.key === 'Escape') { event.preventDefault(); close.current(); }
    if (event.key !== 'Tab') return;
    const controls = [...event.currentTarget.querySelectorAll<HTMLElement>('button:not(:disabled),input:not(:disabled),select:not(:disabled),a[href],[tabindex="0"]')].filter(node => node.getClientRects().length);
    const first = controls[0], last = controls[controls.length - 1];
    if (event.shiftKey && (document.activeElement === first || document.activeElement === ref.current)) { event.preventDefault(); last?.focus(); }
    else if (!event.shiftKey && (document.activeElement === last || document.activeElement === ref.current)) { event.preventDefault(); first?.focus(); }
  }}><header className="studio-dialog-heading"><div><span className="eyebrow">SEAMSTRESS STUDIO</span><h2>{title}</h2></div><button className="icon-button" onClick={onClose} aria-label={`Close ${label.toLowerCase()}`}><Icon name="close"/></button></header>{children}</section></div>, document.body);
}
