const { test } = require('node:test');
const assert = require('node:assert/strict');
const { EventEmitter } = require('node:events');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { createUpdater, newer, releaseNotesText, displayError, RELEASES_URL, CHECK_INTERVAL } = require('./updater.cjs');

function setup(t, options = {}) {
  const folder = fs.mkdtempSync(path.join(os.tmpdir(), 'seamstress-updater-'));
  t.after(() => fs.rmSync(folder, { recursive: true, force: true }));
  const app = new EventEmitter();
  app.isPackaged = options.packaged ?? true;
  app.getVersion = () => '0.2.0';
  app.getPath = () => folder;
  const autoUpdater = new EventEmitter();
  const nativeUpdater = new EventEmitter();
  nativeUpdater.checkForUpdates = () => {};
  let checks = 0, downloads = 0, installs = 0, busy = false, clock = 2000000000000, interval;
  autoUpdater.checkForUpdates = async () => {
    checks++;
    autoUpdater.emit('checking-for-update');
    if (options.check) return options.check(autoUpdater);
    autoUpdater.emit('update-available', { version: '0.3.0', releaseNotes: 'Better previews.' });
  };
  autoUpdater.downloadUpdate = async () => {
    downloads++;
    if (options.download) return options.download(autoUpdater);
    autoUpdater.emit('download-progress', { percent: 42.5 });
    autoUpdater.emit('update-downloaded', { version: '0.3.0' });
  };
  autoUpdater.quitAndInstall = () => { installs++; };
  if (options.cache) fs.writeFileSync(path.join(folder, 'update-state.json'), JSON.stringify(options.cache));
  const events = [], urls = [];
  const updater = createUpdater({ app, autoUpdater, nativeUpdater, onState: state => events.push(state), hasActiveJobs: () => busy,
    platform: options.platform ?? 'darwin', disabled: options.disabled ?? false,
    openExternal: url => urls.push(url), now: () => clock,
    timers: { setInterval: fn => { interval = fn; return { unref() {} }; }, clearInterval: () => { interval = null; } },
  });
  t.after(() => updater.dispose());
  return { app, autoUpdater, nativeUpdater, updater, events, urls, folder, setBusy: value => { busy = value; },
    advance: ms => { clock += ms; }, tick: () => interval?.(),
    counts: () => ({ checks, downloads, installs }) };
}

test('updates never download or install during a check, including focus and timer checks', async t => {
  const f = setup(t);
  f.updater.start();
  await f.updater.check();
  f.app.emit('browser-window-focus');
  f.tick();
  assert.deepEqual(f.counts(), { checks: 1, downloads: 0, installs: 0 });
  assert.equal(f.updater.getState().status, 'available');
  assert.equal(f.autoUpdater.autoDownload, false);
  assert.equal(f.autoUpdater.autoInstallOnAppQuit, false);
  assert.equal(f.autoUpdater.allowPrerelease, false);
  assert.equal(f.autoUpdater.allowDowngrade, false);
  f.advance(CHECK_INTERVAL + 1);
  await f.updater.check({ manual: false });
  assert.equal(f.counts().checks, 2);
});

test('only an explicit verified download enables install; active work blocks restart', async t => {
  const f = setup(t);
  assert.throws(() => f.updater.install(), /Download and verify/);
  await f.updater.check();
  await f.updater.download();
  assert.equal(f.updater.getState().status, 'downloaded');
  assert(f.events.some(event => event.progress === .425));
  f.setBusy(true);
  f.updater.refreshActivity();
  assert.equal(f.updater.getState().busy, true);
  assert.throws(() => f.updater.install(), /Finish or cancel/);
  assert.equal(f.counts().installs, 0);
  f.setBusy(false);
  f.updater.install();
  assert.equal(f.updater.isInstalling(), true);
  assert.equal(f.counts().installs, 0);
  f.nativeUpdater.emit('update-downloaded');
  assert.equal(f.counts().installs, 1);
  assert.throws(() => f.updater.install(), /Download and verify/);
});

test('unexpected downloaded events do not enable installation', async t => {
  const f = setup(t);
  f.autoUpdater.emit('update-downloaded', { version: '0.3.0' });
  assert.notEqual(f.updater.getState().status, 'downloaded');
  assert.throws(() => f.updater.install());
});

test('digest and signature failures never enable install or expose sensitive URLs', async t => {
  const f = setup(t, { download: async updater => {
    const error = new Error('SHA512 checksum mismatch https://secret.example/?token=private');
    updater.emit('error', error);
    throw error;
  } });
  await f.updater.check();
  await f.updater.download();
  assert.equal(f.updater.getState().status, 'error');
  assert.match(f.updater.getState().error, /could not be verified/);
  assert.doesNotMatch(f.updater.getState().error, /secret|private/);
  assert.throws(() => f.updater.install());
  assert.equal(f.counts().installs, 0);
});

test('a resolved download without verification is rejected', async t => {
  const f = setup(t, { download: async () => [] });
  await f.updater.download();
  assert.equal(f.updater.getState().status, 'error');
  assert.throws(() => f.updater.install());
});

test('an update whose version changes while downloading is rejected', async t => {
  const f = setup(t, { download: async updater => updater.emit('update-downloaded', { version: '9.9.9' }) });
  await f.updater.download();
  assert.equal(f.updater.getState().status, 'error');
  assert.throws(() => f.updater.install());
});

test('cached offers cannot stage or install a file without fresh release metadata', async t => {
  const f = setup(t, { cache: { version: '0.4.0', releaseNotes: '<img src=x onerror=alert(1)>', checkedAt: 1999999999990 } });
  assert.equal(f.updater.getState().version, '0.4.0');
  assert.equal(f.updater.getState().releaseNotes, '<img src=x onerror=alert(1)>');
  assert.throws(() => f.updater.install());
  await f.updater.download();
  assert.equal(f.counts().checks, 1);
  assert.equal(f.updater.getState().version, '0.3.0');
  assert.equal(f.updater.getState().status, 'downloaded');
  const saved = JSON.parse(fs.readFileSync(path.join(f.folder, 'update-state.json'), 'utf8'));
  assert.equal(saved.version, '0.3.0');
  assert.equal(saved.status, undefined);
  assert.equal(saved.path, undefined);
  assert.equal(fs.statSync(path.join(f.folder, 'update-state.json')).mode & 0o777, 0o600);
});

test('a failed background check remains quiet and retries are throttled', async t => {
  const f = setup(t, { check: async updater => {
    const error = new Error('ENOTFOUND github.com');
    updater.emit('error', error); throw error;
  } });
  await f.updater.check({ manual: false });
  assert.equal(f.updater.getState().status, 'idle');
  assert.equal(f.updater.getState().error, undefined);
  await f.updater.check({ manual: false });
  assert.equal(f.counts().checks, 1);
  await f.updater.check({ manual: true });
  assert.equal(f.updater.getState().status, 'error');
  assert.match(f.updater.getState().error, /connection/);
  assert.equal(f.counts().checks, 2);
});

test('cached offer survives an offline background check', async t => {
  const f = setup(t, { cache: { version: '0.3.0' }, check: async () => { throw new Error('ENOTFOUND'); } });
  await f.updater.check({ manual: false });
  assert.equal(f.updater.getState().status, 'available');
  assert.equal(f.updater.getState().version, '0.3.0');
});

test('development, smoke and unsupported-platform sessions never contact release servers', async t => {
  for (const options of [{ packaged: false }, { disabled: true }, { platform: 'linux' }]) {
    const f = setup(t, options);
    f.updater.start();
    await f.updater.check();
    assert.equal(f.updater.getState().status, 'disabled');
    await assert.rejects(f.updater.download());
    assert.deepEqual(f.counts(), { checks: 0, downloads: 0, installs: 0 });
  }
});

test('simultaneous checks and downloads share one operation', async t => {
  let resolveCheck, resolveDownload;
  const f = setup(t, {
    check: updater => new Promise(resolve => { resolveCheck = () => { updater.emit('update-available', { version: '0.3.0' }); resolve(); }; }),
    download: updater => new Promise(resolve => { resolveDownload = () => { updater.emit('update-downloaded', { version: '0.3.0' }); resolve(); }; }),
  });
  const first = f.updater.check(), second = f.updater.check();
  await Promise.resolve();
  assert.equal(f.counts().checks, 1); resolveCheck();
  await Promise.all([first, second]);
  const a = f.updater.download(), b = f.updater.download();
  assert.equal(f.counts().downloads, 1); resolveDownload();
  await Promise.all([a, b]);
  assert.equal(f.updater.getState().status, 'downloaded');
});

test('release navigation is fixed and release notes remain bounded inert text', async t => {
  const f = setup(t);
  await f.updater.openReleases('https://evil.example');
  assert.deepEqual(f.urls, [RELEASES_URL]);
  assert.equal(releaseNotesText([{ note: 'One' }, { note: '<script>two</script>' }]), 'One\n\n<script>two</script>');
  assert.equal(releaseNotesText('x'.repeat(100000)).length, 24000);
  assert.equal(newer('0.10.0', '0.2.0'), true);
  assert.equal(newer('0.2.0', '0.2.0'), false);
  assert.equal(newer('1.0.0-beta', '0.2.0'), false);
  assert.equal(newer('99999999999999999.0.0', '0.2.0'), false);
  assert.doesNotMatch(displayError(new Error('oops token=secret')), /secret/);
});

test('dispose removes timers and event subscriptions', async t => {
  const f = setup(t);
  f.updater.start(); await f.updater.check();
  f.updater.dispose();
  assert.equal(f.app.listenerCount('browser-window-focus'), 0);
  assert.equal(f.autoUpdater.listenerCount('update-available'), 0);
  assert.equal(f.nativeUpdater.listenerCount('update-downloaded'), 0);
  f.advance(CHECK_INTERVAL + 1);
  f.tick();
  await f.updater.check();
  assert.equal(f.counts().checks, 1);
});

test('native signature failure cannot leave a deferred quit behind', async t => {
  const f = setup(t);
  await f.updater.download();
  f.updater.install();
  f.nativeUpdater.emit('error', new Error('Code signature does not match'));
  assert.equal(f.updater.getState().status, 'error');
  assert.equal(f.updater.isInstalling(), false);
  f.setBusy(true);
  f.nativeUpdater.emit('update-downloaded');
  assert.equal(f.counts().installs, 0);
});

test('native verification rechecks activity before the final restart', async t => {
  const f = setup(t);
  await f.updater.download();
  f.updater.install();
  f.setBusy(true);
  f.nativeUpdater.emit('update-downloaded');
  assert.equal(f.counts().installs, 0);
  assert.equal(f.updater.getState().status, 'downloaded');
  assert.match(f.updater.getState().error, /Finish or cancel/);
});

test('synchronous provider failures release the in-flight check lock', async t => {
  const f = setup(t);
  f.autoUpdater.checkForUpdates = () => { throw new Error('offline'); };
  await f.updater.check();
  assert.equal(f.updater.getState().status, 'error');
  f.autoUpdater.checkForUpdates = async () => f.autoUpdater.emit('update-not-available');
  await f.updater.check();
  assert.equal(f.updater.getState().status, 'idle');
});
