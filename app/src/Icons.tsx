import type { CSSProperties } from 'react';

const paths: Record<string, React.ReactNode> = {
  plus: <path d="M12 5v14M5 12h14" />,
  folder: <path d="M3 7V5h6l2 2h10v13H3zM3 10h18" />,
  upload: <><path d="M12 16V3m-5 5 5-5 5 5M4 16v5h16v-5" /></>,
  play: <path d="m8 4 12 8-12 8z" />,
  pause: <><path d="M8 5v14M16 5v14" /></>,
  back: <><path d="M5 5v14m14-14L8 12l11 7z" /></>,
  forward: <><path d="M19 5v14M5 5l11 7-11 7z" /></>,
  scan: <><path d="M8 3H3v5m13-5h5v5M3 16v5h5m13-5v5h-5M3 12h18" /></>,
  sliders: <><path d="M3 7h6m4 0h8M3 17h10m4 0h4M9 3v8m4 2v8" /></>,
  check: <path d="m5 12 4 4L20 5" />,
  chevron: <path d="m9 5 7 7-7 7" />,
  down: <path d="m5 9 7 7 7-7" />,
  arrow: <path d="M4 12h16m-6-6 6 6-6 6" />,
  export: <path d="M12 15V3m-5 5 5-5 5 5M4 14v7h16v-7" />,
  loop: <path d="M4 8h13a4 4 0 0 1 4 4v2M4 8l4-4M4 8l4 4m12 4H7a4 4 0 0 1-4-4v-2m17 6-4 4m4-4-4-4" />,
  split: <><path d="M3 5h18v14H3zM12 2v20" /></>,
  film: <><path d="M3 3h18v18H3zM7 3v18M17 3v18M3 8h4m-4 8h4m10-8h4m-4 8h4" /></>,
  trash: <path d="M4 6h16M9 6V3h6v3M6 6l1 15h10l1-15M10 10v7m4-7v7" />,
  close: <path d="m6 6 12 12M6 18 18 6" />,
  warning: <><path d="m12 3 10 18H2zM12 9v5m0 3v.1" /></>,
  sound: <><path d="M3 9h4l5-4v14l-5-4H3zM16 8a6 6 0 0 1 0 8m3-11a10 10 0 0 1 0 14" /></>,
  muted: <><path d="M3 9h4l5-4v14l-5-4H3zM16 9l5 6m0-6-5 6" /></>,
  help: <><circle cx="12" cy="12" r="9" /><path d="M9 9a3 3 0 1 1 5 2c-2 1-2 2-2 3m0 3v.1" /></>,
  external: <path d="M14 3h7v7m0-7L10 14M10 3H3v18h18v-7" />,
  diamond: <path d="m12 3 9 9-9 9-9-9z" />,
};
export function Icon({ name, size = 18, style }: { name: string; size?: number; style?: CSSProperties }) {
  return <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.55" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true" style={style}>{paths[name] ?? paths.diamond}</svg>;
}

export function WeaveMark({ small = false }: { small?: boolean }) {
  return <svg viewBox="0 0 48 48" className={small ? 'weave-mark small' : 'weave-mark'} aria-hidden="true"><path d="M7 10h12v28H7zM29 10h12v28H29z" fill="currentColor" opacity=".45" /><path d="M3 17h42v7H3zM3 29h42v7H3z" fill="currentColor" /><path d="M7 17h12v7H7zM29 29h12v7H29z" fill="var(--surface)" opacity=".82" /></svg>;
}

export function WovenIllustration() {
  return <svg viewBox="0 0 560 270" className="woven-illustration" role="img" aria-label="Two woven film ribbons joining into one continuous strip"><defs><pattern id="fabric" width="5" height="5" patternUnits="userSpaceOnUse"><path d="M0 1h5M1 0v5" stroke="white" strokeOpacity=".14" strokeWidth=".55" /></pattern><linearGradient id="tealRibbon"><stop stopColor="#336563"/><stop offset=".5" stopColor="#80c9b9"/><stop offset="1" stopColor="#316b63"/></linearGradient><linearGradient id="pinkRibbon"><stop stopColor="#914f78"/><stop offset=".6" stopColor="#d995b5"/><stop offset="1" stopColor="#7c4868"/></linearGradient><mask id="ribbonMask"><path d="M42 167C128 56 220 197 305 121S422 41 518 97" fill="none" stroke="white" strokeWidth="58" /></mask></defs><ellipse cx="280" cy="229" rx="170" ry="10" fill="#000" opacity=".19"/><path d="M48 84C160 174 210 29 296 139s151 60 213 20" fill="none" stroke="url(#pinkRibbon)" strokeWidth="48"/><path d="M48 84C160 174 210 29 296 139s151 60 213 20" fill="none" stroke="#ebc3d5" strokeWidth="1" strokeDasharray="2 4"/><path d="M42 167C128 56 220 197 305 121S422 41 518 97" fill="none" stroke="url(#tealRibbon)" strokeWidth="58"/><rect width="560" height="270" fill="url(#fabric)" mask="url(#ribbonMask)"/><path d="M42 149C128 38 220 179 305 103S422 23 518 79M42 185C128 74 220 215 305 139S422 59 518 115" fill="none" stroke="#d3e7df" strokeOpacity=".42" strokeWidth="1" strokeDasharray="2 5"/><path d="M262 167c19-19 34-33 52-48" stroke="#ece7db" strokeWidth="2" strokeDasharray="2 6"/><circle cx="273" cy="155" r="3" fill="#f0eada"/><path d="m270 148 7 6-4 8" fill="none" stroke="#173b37" strokeWidth="1"/></svg>;
}
