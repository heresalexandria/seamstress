'use strict';

// Download hashes and the macOS signing identity are verified by electron-updater
// and Squirrel.Mac. The renderer never chooses a feed, asset, command or path.
const fs = require('node:fs');
const path = require('node:path');

const RELEASES_URL = 'https://github.com/heresalexandria/seamstress/releases/latest';
const CHECK_INTERVAL = 24 * 60 * 60 * 1000;
const RETRY_INTERVAL = 60 * 60 * 1000;
const MAX_NOTES = 24000;

function releaseNotesText(value) {
  const notes = Array.isArray(value)
    ? value.map(note => typeof note?.note === 'string' ? note.note : '').join('\n\n')
    : typeof value === 'string' ? value : '';
  // React renders this as text. Keeping markup inert also avoids fetching remote
  // release-note images inside a project window.
  return notes.replace(/\u0000/g, '').slice(0, MAX_NOTES);
}

function stableVersion(value) {
  return typeof value === 'string' && /^\d{1,9}\.\d{1,9}\.\d{1,9}$/.test(value) ? value : null;
}

function newer(candidate, current) {
  if (!stableVersion(candidate) || !stableVersion(current)) return false;
  const a = candidate.split('.').map(Number), b = current.split('.').map(Number);
  for (let i = 0; i < 3; i++) if (a[i] !== b[i]) return a[i] > b[i];
  return false;
}

function displayError(error) {
  const text = String(error?.message || error || 'The update could not complete.');
  if (/signature|code.?sign|certificate|checksum|sha512|digest/i.test(text)) {
    return 'The update could not be verified. Your installed app is unchanged. Try again later or download the installer from Releases.';
  }
  if (/404|ERR_UPDATER_LATEST_VERSION_NOT_FOUND|ERR_UPDATER_CHANNEL_FILE_NOT_FOUND/i.test(text)) {
    return 'The latest release is not available for updates yet. Try again later or check Releases.';
  }
  if (/ENOTFOUND|ECONN|timed? ?out|network|internet|ERR_INTERNET_DISCONNECTED/i.test(text)) {
    return 'Could not reach GitHub. Check your connection and try again.';
  }
  // Error details can contain signed download URLs or local paths. Keep those in
  // the library's local log rather than exposing them as UI or cached state.
  return 'The update could not complete. Your projects are safe. Try again or download the installer from Releases.';
}

function createUpdater({
  app, autoUpdater, nativeUpdater, hasActiveJobs = () => false, onState = () => {},
  openExternal = () => {}, platform = process.platform, disabled = false,
  now = Date.now, timers = { setInterval, clearInterval },
}) {
  if (!app || !autoUpdater) throw new Error('An app and autoUpdater are required');
  const currentVersion = app.getVersion();
  const enabled = app.isPackaged && platform === 'darwin' && !disabled;
  if (enabled && !nativeUpdater) throw new Error('The native macOS updater is required for signature verification');
  const stateFile = path.join(app.getPath('userData'), 'update-state.json');
  let state = {
    currentVersion, status: enabled ? 'idle' : 'disabled', releaseNotes: '', busy: Boolean(hasActiveJobs()),
    message: enabled ? 'Check for a new version of Seamstress.'
      : !app.isPackaged ? 'Updates are available in the installed app. This is a development checkout.'
      : disabled ? 'Update checks are off for this test run.' : 'In-app updates are currently available on macOS.',
  };
  let checked = false, downloaded = false, nativeVerified = false, quitRequested = false, disposed = false, started = false;
  let checking = null, downloading = null, manualCheck = false, activityTimer = null;
  let lastAttempt = 0;
  const listeners = [], nativeListeners = [];

  autoUpdater.autoDownload = false;
  autoUpdater.autoInstallOnAppQuit = false;
  if ('autoInstallEvent' in autoUpdater) autoUpdater.autoInstallEvent = 'manual';
  autoUpdater.autoRunAppAfterInstall = true;
  autoUpdater.allowPrerelease = false;
  autoUpdater.allowDowngrade = false;
  autoUpdater.disableWebInstaller = true;

  if (enabled) {
    try {
      const cache = JSON.parse(fs.readFileSync(stateFile, 'utf8'));
      if (Number.isFinite(cache.checkedAt) && cache.checkedAt <= now()) state.checkedAt = cache.checkedAt;
      if (Number.isFinite(cache.lastAttempt) && cache.lastAttempt <= now()) lastAttempt = cache.lastAttempt;
      if (newer(cache.version, currentVersion)) {
        state = { ...state, status: 'available', version: cache.version,
          releaseNotes: releaseNotesText(cache.releaseNotes), message: `Seamstress ${cache.version} is available.` };
      }
    } catch { /* The first launch and a corrupt optional cache both use a fresh check. */ }
  }

  function snapshot() { return { ...state, busy: Boolean(hasActiveJobs()) }; }
  function publish(patch = {}) {
    state = { ...state, ...patch, busy: Boolean(hasActiveJobs()) };
    if (!disposed) onState(snapshot());
    return snapshot();
  }
  function persist() {
    try {
      fs.mkdirSync(path.dirname(stateFile), { recursive: true });
      const temporary = stateFile + '.tmp';
      fs.writeFileSync(temporary, JSON.stringify({
        checkedAt: state.checkedAt, lastAttempt, version: state.version, releaseNotes: state.releaseNotes,
      }), { mode: 0o600 });
      fs.chmodSync(temporary, 0o600);
      fs.renameSync(temporary, stateFile);
    } catch { /* Losing an optional cache must not prevent an update. */ }
  }
  function fail(error) {
    downloaded = false;
    nativeVerified = false;
    quitRequested = false;
    if (manualCheck || downloading || state.status === 'downloading' || state.status === 'installing') {
      publish({ status: 'error', error: displayError(error), progress: undefined, message: 'The update could not complete.' });
    } else {
      // An offline background check does not interrupt editing or hide a known offer.
      publish({ status: state.version ? 'available' : 'idle', error: undefined, progress: undefined,
        message: state.version ? `Seamstress ${state.version} is available.` : 'Check for a new version of Seamstress.' });
    }
  }
  function listen(event, fn) { autoUpdater.on(event, fn); listeners.push([event, fn]); }
  listen('checking-for-update', () => {
    if (enabled && !disposed) publish({ status: 'checking', error: undefined, message: 'Checking GitHub Releases…' });
  });
  listen('update-available', info => {
    if (!enabled || disposed) return;
    const version = stableVersion(info?.version);
    if (!version || !newer(version, currentVersion)) {
      checked = false;
      fail(new Error('Invalid stable update version'));
      return;
    }
    checked = true;
    publish({ status: 'available', version, releaseNotes: releaseNotesText(info.releaseNotes),
      checkedAt: now(), error: undefined, progress: undefined, message: `Seamstress ${version} is available.` });
    persist();
  });
  listen('update-not-available', () => {
    if (!enabled || disposed) return;
    checked = true;
    publish({ status: 'idle', version: undefined, releaseNotes: '', checkedAt: now(),
      error: undefined, progress: undefined, message: 'You have the latest version.' });
    persist();
  });
  listen('download-progress', progress => {
    if (!enabled || disposed || state.status !== 'downloading') return;
    const percent = Number(progress?.percent);
    publish({ progress: Number.isFinite(percent) ? Math.max(0, Math.min(1, percent / 100)) : undefined });
  });
  listen('update-downloaded', info => {
    if (!enabled || disposed) return;
    // Only a user-requested download may create an installable state. The
    // library emits this after checksum verification. With automatic installation
    // disabled, Squirrel verifies the signing identity on the explicit install.
    if (!downloading && state.status !== 'downloading') return;
    if (info?.version !== state.version) { fail(new Error('Update version changed during download')); return; }
    downloaded = true;
    publish({ status: 'downloaded', progress: 1, error: undefined,
      message: 'Your update is ready. Restart when your work is finished.' });
  });
  listen('error', error => { if (enabled && !disposed) fail(error); });
  if (enabled) {
    const verified = () => {
      if (disposed || state.status !== 'installing' || !downloaded) return;
      nativeVerified = true;
      if (hasActiveJobs()) {
        publish({ status: 'downloaded', error: 'Finish or cancel all processing before restarting to update.' });
        return;
      }
      // MacUpdater registered its native listener during construction, so its
      // squirrelDownloadedUpdate flag is now true. No deferred quit callback is
      // left behind if signature verification fails before reaching this point.
      finishInstall();
    };
    const failed = error => { if (!disposed && state.status === 'installing') fail(error); };
    nativeUpdater.on('update-downloaded', verified);
    nativeUpdater.on('error', failed);
    nativeListeners.push(['update-downloaded', verified], ['error', failed]);
  }

  async function check({ manual = true } = {}) {
    if (!enabled || disposed || downloaded || downloading || state.status === 'installing') return snapshot();
    if (checking) return checking;
    if (!manual && ((state.checkedAt && now() - state.checkedAt < CHECK_INTERVAL)
      || (lastAttempt && now() - lastAttempt < RETRY_INTERVAL))) return snapshot();
    manualCheck = manual;
    lastAttempt = now();
    checked = false;
    publish({ status: 'checking', error: undefined, message: 'Checking GitHub Releases…' });
    persist();
    checking = (async () => {
      try { await Promise.resolve().then(() => autoUpdater.checkForUpdates()); }
      catch (error) { fail(error); }
      finally { checking = null; manualCheck = false; }
      return snapshot();
    })();
    return checking;
  }

  async function download() {
    if (!enabled || disposed) throw new Error(state.message);
    if (downloaded || state.status === 'installing') return snapshot();
    if (downloading) return downloading;
    // Even a restored offer gets fresh server metadata. The library may then
    // reuse an already downloaded archive after checking its recorded digest.
    if (!checked || state.status === 'error') await check({ manual: true });
    if (!checked || state.status !== 'available') {
      if (state.error) throw new Error(state.error);
      return snapshot();
    }
    publish({ status: 'downloading', progress: 0, error: undefined, message: `Downloading Seamstress ${state.version}…` });
    downloading = (async () => {
      try {
        await autoUpdater.downloadUpdate();
        if (!downloaded) throw new Error('Update download finished without verification');
      } catch (error) { fail(error); }
      finally { downloading = null; }
      return snapshot();
    })();
    return downloading;
  }

  function install() {
    if (!enabled || disposed || !downloaded || state.status !== 'downloaded') throw new Error('Download and verify the update before restarting.');
    if (hasActiveJobs()) { publish(); throw new Error('Finish or cancel all processing before restarting to update.'); }
    publish({ status: 'installing', error: undefined, message: 'Verifying the signed update before restarting Seamstress…' });
    // electron-updater has already configured this native updater with its
    // authenticated localhost feed for the hash-checked ZIP. Calling the native
    // check first lets us wait for signature verification and recheck active jobs
    // before permitting the restart.
    try { nativeVerified ? finishInstall() : nativeUpdater.checkForUpdates(); }
    catch (error) { fail(error); throw new Error(displayError(error)); }
    return snapshot();
  }

  function finishInstall() {
    if (quitRequested) return;
    quitRequested = true;
    try { autoUpdater.quitAndInstall(); }
    catch (error) { fail(error); }
  }

  function onFocus() { void check({ manual: false }); }
  function start() {
    if (!enabled || disposed || started) return;
    started = true;
    app.on('browser-window-focus', onFocus);
    activityTimer = timers.setInterval(onFocus, RETRY_INTERVAL);
    activityTimer?.unref?.();
    void check({ manual: false });
  }
  function dispose() {
    disposed = true;
    if (activityTimer) timers.clearInterval(activityTimer);
    app.removeListener('browser-window-focus', onFocus);
    for (const [event, listener] of listeners) autoUpdater.removeListener(event, listener);
    for (const [event, listener] of nativeListeners) nativeUpdater.removeListener(event, listener);
  }
  return {
    getState: snapshot, check, download, install, start, dispose,
    refreshActivity: () => publish(),
    isInstalling: () => state.status === 'installing',
    openReleases: () => openExternal(RELEASES_URL),
  };
}

module.exports = { createUpdater, releaseNotesText, stableVersion, newer, displayError, RELEASES_URL, CHECK_INTERVAL };
