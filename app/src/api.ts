import type { SeamstressAPI } from './types';

const desktopRequired = () => {
  throw new Error('Open the Seamstress desktop app to import videos and run local processing. This browser preview only displays the interface.');
};

/** Explicit browser adapter: never creates pretend projects or successful jobs. */
export const browserAdapter: SeamstressAPI = {
  pickVideo: async () => desktopRequired(),
  pathForFile: () => desktopRequired(),
  createProject: async () => desktopRequired(),
  openProject: async () => desktopRequired(),
  getProject: async () => desktopRequired(),
  setSeams: async () => desktopRequired(),
  importSeamCorrection: async () => desktopRequired(),
  run: async () => desktopRequired(),
  cancelJob: async () => desktopRequired(),
  chooseExportPath: async () => desktopRequired(),
  onJobEvent: () => () => {},
  mediaUrl: () => '',
  revealFile: async () => desktopRequired(),
  getUpdateState: async () => ({ currentVersion: '', status: 'disabled', releaseNotes: '', busy: false, message: 'Open the installed app to check for updates.' }),
  checkForUpdates: async () => desktopRequired(),
  downloadUpdate: async () => desktopRequired(),
  installUpdate: async () => desktopRequired(),
  openReleases: async () => desktopRequired(),
  onUpdateState: () => () => {},
  onOpenUpdates: () => () => {},
};

export const isDesktop = Boolean(window.seamstress);
export const api = window.seamstress ?? browserAdapter;
