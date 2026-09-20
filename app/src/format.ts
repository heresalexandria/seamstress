export function timecode(seconds: number, milliseconds = true) {
  const total = Math.max(0, Math.round(seconds * 1000));
  const hours = Math.floor(total / 3_600_000);
  const minutes = Math.floor(total / 60_000) % 60;
  const secs = Math.floor(total / 1000) % 60;
  const prefix = hours ? `${String(hours).padStart(2, '0')}:` : '';
  return `${prefix}${String(minutes).padStart(2, '0')}:${String(secs).padStart(2, '0')}${milliseconds ? `.${String(total % 1000).padStart(3, '0')}` : ''}`;
}

/** Seconds, mm:ss.mmm or hh:mm:ss.mmm, with no silent invalid coercion. */
export function parseTimecode(value: string): number | null {
  const parts = value.trim().split(':');
  if (!parts.length || parts.length > 3 || parts.some(p => !/^\d+(?:\.\d+)?$/.test(p))) return null;
  const values = parts.map(Number);
  if (values.some(v => !Number.isFinite(v))) return null;
  if (values.slice(1).some(v => v >= 60)) return null;
  if (values.slice(0, -1).some(v => !Number.isInteger(v))) return null;
  return values.reduce((total, part) => total * 60 + part, 0);
}

export function filename(path: string) { return path.split(/[\\/]/).pop() || path; }
export function clamp(value: number, minimum: number, maximum: number) { return Math.min(maximum, Math.max(minimum, value)); }
export function message(error: unknown) { return error instanceof Error ? error.message : String(error); }
