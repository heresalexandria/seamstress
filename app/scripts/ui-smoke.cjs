/* Real Electron/preload/backend/media smoke check; native file dialogs use a
 * deterministic choice. Update UI fixtures exercise rendering without a live
 * release request. No projects, jobs, or media are faked.
 * Run after npm run build: node scripts/ui-smoke.cjs /absolute/video.mp4 [--workflow]
 * --workflow additionally runs real analysis, preview, and export on the video.
 * Set SEAMSTRESS_TEST_APP to the packaged executable to test its bundled runtime.
 * --smoke launches hidden with activation prohibited on macOS. Screenshots and
 * DOM input remain available without showing or focusing any application window.
 */
const path = require('node:path');
const fs = require('node:fs');
const assert = require('node:assert/strict');
const { spawnSync } = require('node:child_process');
const { _electron: electron } = require('@playwright/test');

async function main() {
  const sourceArgument = process.argv.slice(2).find(value => !value.startsWith('--'));
  const workflow = process.argv.includes('--workflow');
  const output = path.resolve('../output/app-ui-smoke');
  fs.mkdirSync(output, { recursive: true });
  const source = sourceArgument ? path.resolve(sourceArgument) : path.join(output, `fixture-${Date.now()}.mp4`);
  if (!sourceArgument) {
    const generated = spawnSync('ffmpeg', ['-v', 'error', '-nostdin', '-n', '-f', 'lavfi', '-i', "testsrc2=size=320x180:rate=12:duration=6,eq=brightness='if(gte(t,3),0.08,0)':eval=frame", '-f', 'lavfi', '-i', 'sine=frequency=440:sample_rate=48000:duration=6', '-c:v', 'libx264', '-crf', '18', '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-shortest', source], { encoding: 'utf8' });
    if (generated.error || generated.status !== 0) throw new Error(`FFmpeg could not create the real smoke fixture. Install FFmpeg or provide an existing video. ${generated.error?.message || generated.stderr}`);
  }
  if (!fs.existsSync(source)) throw new Error(`Provide a short real video: ${source}`);
  const packagedExecutable = process.env.SEAMSTRESS_TEST_APP;
  if (packagedExecutable && (!path.isAbsolute(packagedExecutable) || !fs.existsSync(packagedExecutable))) throw new Error('SEAMSTRESS_TEST_APP must be an existing absolute packaged executable path.');
  const application = await electron.launch({
    ...(packagedExecutable ? { executablePath: packagedExecutable } : {}),
    args: packagedExecutable ? ['--smoke', `--user-data-dir=${path.resolve(__dirname, '../../.app-data/packaged-smoke')}`] : ['.', '--smoke'],
    cwd: path.resolve(__dirname, '..'), env: { ...process.env, SEAMSTRESS_DEV_URL: '' },
  });
  const errors = [];
  try {
    const page = await application.firstWindow();
    const hidden = await application.evaluate(({ BrowserWindow }) => BrowserWindow.getAllWindows().every(window => !window.isVisible() && !window.isFocused() && !window.isFocusable()));
    assert.equal(hidden, true, 'Smoke windows are hidden and cannot receive native focus');
    page.on('pageerror', error => errors.push(error.message));
    await page.waitForSelector('.empty-state');
    await page.screenshot({ path: path.join(output, 'empty.png') });
    // Exercise the production IPC/preload surface. Smoke mode deliberately
    // disables release network calls and cannot download or install anything.
    const updateState = await page.evaluate(() => window.seamstress.getUpdateState());
    assert.equal(updateState.status, 'disabled', 'Smoke runs never check live releases');
    assert.match(updateState.currentVersion, /^\d+\.\d+\.\d+/);
    assert.equal((await page.evaluate(() => window.seamstress.checkForUpdates())).status, 'disabled');
    for (const method of ['downloadUpdate', 'installUpdate']) {
      const refusal = await page.evaluate(async name => { try { await window.seamstress[name](); return ''; } catch (error) { return error.message; } }, method);
      assert.ok(refusal, `${method} is refused in smoke mode`);
    }
    await page.getByRole('button', { name: 'About Seamstress and updates', exact: true }).click();
    await page.getByRole('dialog', { name: 'Made to keep improving.' }).waitFor();
    assert.equal(await page.getByRole('button', { name: 'Check for updates', exact: true }).isDisabled(), true);
    await page.screenshot({ path: path.join(output, 'updates.png') });
    await page.keyboard.press('Escape');
    assert.equal(await page.locator('.update-modal').count(), 0);
    assert.equal(await page.getByRole('button', { name: 'About Seamstress and updates', exact: true }).evaluate(button => document.activeElement === button), true, 'Closing restores keyboard focus');
    await application.evaluate(({ Menu, BrowserWindow }) => {
      const item = Menu.getApplicationMenu().items[0].submenu.items.find(item => item.label === 'Check for Updates…');
      if (!item) throw new Error('Native update menu is missing');
      item.click(item, BrowserWindow.getAllWindows()[0]);
    });
    await page.locator('.update-modal').waitFor();
    // Release data is untrusted text. These event fixtures test the renderer,
    // while the real main-process updater remains disabled throughout the run.
    const updateFixture = { ...updateState, status: 'available', version: '99.0.0', releaseNotes: '<img src="https://example.invalid/unsafe" onerror="window.__unsafeUpdate = true">\nBetter previews.', message: 'Update rendering fixture.', busy: false };
    await application.evaluate(({ BrowserWindow }, state) => BrowserWindow.getAllWindows()[0].webContents.send('update:event', state), updateFixture);
    await page.getByRole('button', { name: 'Download update', exact: true }).waitFor();
    assert.match(await page.locator('.update-notes').innerText(), /<img src=/, 'Release markup stays inert text');
    assert.equal(await page.locator('.update-notes img, .update-notes script').count(), 0);
    assert.equal(await page.evaluate(() => window.__unsafeUpdate), undefined);
    await page.screenshot({ path: path.join(output, 'updates-offer-fixture.png') });
    await application.evaluate(({ BrowserWindow }, state) => BrowserWindow.getAllWindows()[0].webContents.send('update:event', state), { ...updateFixture, status: 'downloading', progress: .375 });
    await page.getByRole('progressbar', { name: 'Update download progress' }).waitFor();
    assert.equal(await page.getByRole('progressbar', { name: 'Update download progress' }).getAttribute('value'), '0.375');
    await application.evaluate(({ BrowserWindow }, state) => BrowserWindow.getAllWindows()[0].webContents.send('update:event', state), { ...updateFixture, status: 'downloaded', progress: 1, busy: true });
    assert.equal(await page.getByRole('button', { name: 'Finish processing first', exact: true }).isDisabled(), true);
    await application.evaluate(({ BrowserWindow }, state) => BrowserWindow.getAllWindows()[0].webContents.send('update:event', state), { ...updateFixture, status: 'downloaded', progress: 1 });
    assert.equal(await page.getByRole('button', { name: 'Restart & install update', exact: true }).isEnabled(), true);
    await application.evaluate(({ BrowserWindow }, state) => BrowserWindow.getAllWindows()[0].webContents.send('update:event', state), updateState);
    await page.getByRole('button', { name: 'Close updates', exact: true }).click();
    await application.evaluate(({ shell }) => {
      global.__smokeOpenedUrls = [];
      global.__smokeOpenExternal = shell.openExternal;
      shell.openExternal = async url => { global.__smokeOpenedUrls.push(url); };
    });
    await page.evaluate(() => window.seamstress.openReleases('https://example.invalid'));
    const openedUrls = await application.evaluate(({ shell }) => { shell.openExternal = global.__smokeOpenExternal; return global.__smokeOpenedUrls; });
    assert.deepEqual(openedUrls, ['https://github.com/heresalexandria/seamstress/releases/latest'], 'Renderer cannot override the release URL');
    const rejectedSender = await application.evaluate(async ({ BrowserWindow }) => {
      const other = new BrowserWindow({ show: false, focusable: false, skipTaskbar: true, webPreferences: { nodeIntegration: true, contextIsolation: false, sandbox: false } });
      try {
        await other.loadURL('data:text/html,<title>Untrusted IPC test</title>');
        return await other.webContents.executeJavaScript("require('electron').ipcRenderer.invoke('update:state').then(()=>'accepted',error=>error.message)");
      } finally { other.destroy(); }
    });
    assert.match(rejectedSender, /Untrusted request/, 'Updater IPC refuses another webContents');
    await page.evaluate(() => {
      window.__updateEvents = 0;
      window.__stopUpdateEvents = window.seamstress.onUpdateState(() => { window.__updateEvents++; });
    });
    await application.evaluate(({ BrowserWindow }, state) => BrowserWindow.getAllWindows()[0].webContents.send('update:event', state), updateState);
    await page.waitForFunction(() => window.__updateEvents === 1);
    await page.evaluate(() => window.__stopUpdateEvents());
    await application.evaluate(({ BrowserWindow }, state) => BrowserWindow.getAllWindows()[0].webContents.send('update:event', state), updateState);
    // A following IPC round-trip ensures the preceding event has been delivered.
    await page.evaluate(() => window.seamstress.getUpdateState());
    assert.equal(await page.evaluate(() => window.__updateEvents), 1, 'Unsubscribe removes only the listener it registered');
    await page.evaluate(() => {
      window.seamstress.onJobEvent(event => { if (event.project) window.__smokeProject = event.project; });
      window.__updateBusyStates = [];
      window.__stopUpdateBusy = window.seamstress.onUpdateState(state => window.__updateBusyStates.push(state.busy));
    });
    await application.evaluate(({ dialog }, file) => { dialog.showOpenDialog = async () => ({ canceled: false, filePaths: [file] }); }, source);
    await page.getByRole('button', { name: 'New video', exact: true }).click();
    await page.waitForSelector('.source-video', { timeout: 120000 });
    await page.waitForFunction(() => document.querySelector('.source-video')?.readyState >= 2, undefined, { timeout: 60000 });
    await page.waitForFunction(() => window.__updateBusyStates.includes(true) && window.__updateBusyStates.at(-1) === false);
    await page.evaluate(() => window.__stopUpdateBusy());
    const sourceInfo = await page.locator('.source-video').evaluate(video => ({ url: video.currentSrc, width: video.videoWidth, height: video.videoHeight, duration: video.duration, error: video.error?.message }));
    assert.ok(sourceInfo.url.startsWith('seamstress-media://'), 'Real custom media protocol is used');
    assert.ok(sourceInfo.width > 0 && sourceInfo.height > 0 && sourceInfo.duration > 0, 'Real video decodes');
    assert.equal(sourceInfo.error, undefined);
    await page.locator('.source-video').evaluate(video => { video.currentTime = Math.min(1, video.duration / 3); });
    await page.waitForTimeout(250);
    const initialSeams = await page.locator('.seam-marker').count();
    await page.getByRole('button', { name: 'Add seam', exact: true }).click();
    await page.waitForFunction(count => document.querySelectorAll('.seam-marker').length === count + 1 && !document.querySelector('.job-panel'), initialSeams);
    await page.waitForSelector('#seam-frame');
    const before = Number(await page.locator('#seam-frame').inputValue());
    await page.locator('#seam-frame').fill(String(before + 2));
    await page.locator('#seam-frame').press('Enter');
    await page.waitForFunction(at => document.querySelector('#seam-frame')?.value === String(at) && !document.querySelector('#seam-frame')?.disabled, before + 2);
    await page.getByRole('switch', { name: 'Include in correction' }).click();
    await page.waitForFunction(() => document.querySelector('[role="switch"]')?.getAttribute('aria-checked') === 'false');
    await page.waitForFunction(() => !document.querySelector('.job-panel'));
    const currentTime = await page.locator('.source-video').evaluate(video => video.currentTime);
    await page.getByRole('button', { name: 'Next frame', exact: true }).click();
    await page.waitForFunction(prior => document.querySelector('.source-video').currentTime > prior, currentTime);
    const dismiss = page.getByRole('button', { name: 'Dismiss message' });
    if (await dismiss.count()) await dismiss.click();
    await page.screenshot({ path: path.join(output, 'project.png') });
    const project = await page.evaluate(() => window.__smokeProject);
    assert.ok(project?.projectPath, 'Import completion returns a real project');
    await application.evaluate(({ dialog }, file) => { dialog.showOpenDialog = async () => ({ canceled: false, filePaths: [file] }); }, project.projectPath);
    await page.getByRole('button', { name: 'Open', exact: true }).click();
    await page.waitForSelector('.source-video');
    await page.waitForFunction(() => !document.querySelector('.job-panel'));
    await page.getByRole('button', { name: 'Remove seam', exact: true }).click();
    await page.waitForFunction(count => document.querySelectorAll('.seam-marker').length === count && !document.querySelector('.job-panel'), initialSeams);
    const checks = ['updater IPC with no live network', 'update menu + keyboard focus', 'inert release notes + update progress fixtures', 'update install controls respect busy state', 'fixed release URL', 'untrusted updater sender rejected', 'update listener cleanup', 'real import updates busy guard', 'real import + detection', 'custom-protocol media decoding', 'manual seam creation', 'frame edit', 'toggle', 'frame step', 'project reopen', 'seam deletion', 'renderer errors'];
    if (workflow) {
      const current = await page.evaluate(projectPath => window.seamstress.getProject(projectPath), project.projectPath);
      if (!current.seams.some(seam => seam.enabled)) {
        await page.locator('.source-video').evaluate(video => { video.currentTime = video.duration / 2; });
        await page.waitForTimeout(250);
        await page.getByRole('button', { name: 'Add seam', exact: true }).click();
        await page.waitForFunction(() => !document.querySelector('.job-panel') && document.querySelector('[role="switch"]')?.getAttribute('aria-checked') === 'true');
      }
      const exportPath = path.join(output, `workflow-${Date.now()}.mp4`);
      await application.evaluate(({ dialog }, file) => { dialog.showSaveDialog = async () => ({ canceled: false, filePath: file }); }, exportPath);
      await page.getByRole('button', { name: 'Run workflow', exact: true }).click();
      await page.getByRole('dialog').waitFor();
      await page.getByRole('button', { name: 'Choose location & run workflow', exact: true }).click();
      await page.waitForFunction(() => (!document.querySelector('[role="dialog"]') && !!document.querySelector('.job-panel')) || document.querySelector('.export-error, .notification.is-error'));
      assert.equal(await page.locator('.export-error, .notification.is-error').count(), 0, (await page.locator('.export-error, .notification.is-error').allTextContents()).join('\n'));
      await page.waitForFunction(() => !document.querySelector('.job-panel'), undefined, { timeout: 600000 });
      const finished = await page.evaluate(() => window.__smokeProject);
      assert.equal(finished?.artifacts.export, exportPath, 'Whole workflow exports to the chosen destination');
      assert.ok(fs.statSync(exportPath).size > 0, 'Real encoded export exists');
      await page.getByRole('button', { name: 'Whole shot', exact: true }).click();
      await page.waitForFunction(() => document.querySelector('.candidate-video')?.readyState >= 2, undefined, { timeout: 60000 });
      await page.locator('.source-video').evaluate(video => { video.currentTime = Math.min(2, video.duration / 2); });
      await page.waitForFunction(() => Math.abs(document.querySelector('.source-video').currentTime - document.querySelector('.candidate-video').currentTime) < .08);
      const correctedInfo = await page.locator('.candidate-video').evaluate(video => ({ width: video.videoWidth, height: video.videoHeight, time: video.currentTime, error: video.error?.message }));
      assert.ok(correctedInfo.width > 0 && correctedInfo.height > 0 && correctedInfo.time > 0, 'Corrected media decodes and seeks');
      assert.equal(correctedInfo.error, undefined);
      const divider = page.getByRole('slider', { name: 'Drag comparison divider' });
      await divider.focus();
      await page.keyboard.press('ArrowRight');
      assert.equal(await divider.getAttribute('aria-valuenow'), '52', 'Comparison divider responds to keyboard');
      await page.getByRole('button', { name: 'Corrected', exact: true }).click();
      await page.getByRole('button', { name: 'Split', exact: true }).click();
      await page.screenshot({ path: path.join(output, 'workflow.png') });
      checks.push('whole workflow + export', 'corrected custom-protocol playback and seek', 'synchronized comparison', 'comparison divider');
    }
    assert.equal(await page.locator('.media-error').count(), 0, 'No playback error');
    assert.deepEqual(errors, [], 'No renderer exceptions');
    assert.equal(await application.evaluate(({ BrowserWindow }) => BrowserWindow.getAllWindows().every(window => !window.isVisible() && !window.isFocused() && !window.isFocusable())), true, 'Smoke windows remain hidden and unfocusable after UI interactions');
    checks.push('hidden nonactivating smoke windows');
    console.log(JSON.stringify({ ok: true, source: sourceInfo, projectPath: project.projectPath, screenshots: output, checks }, null, 2));
  } catch (error) {
    try { await (await application.firstWindow()).screenshot({ path: path.join(output, 'failure.png') }); } catch { /* Preserve the original failure. */ }
    throw error;
  } finally { await application.close(); }
}
main().catch(error => { console.error(error); process.exitCode = 1; });
