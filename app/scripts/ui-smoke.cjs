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

async function chooseNativeFile(application, file) {
  await application.evaluate(({ dialog }, value) => {
    dialog.showOpenDialog = async () => value ? { canceled: false, filePaths: [value] } : { canceled: true, filePaths: [] };
  }, file);
}

async function readProject(page, projectPath) {
  return page.evaluate(value => window.seamstress.getProject(value), projectPath);
}

async function selectSeam(page, project, frame) {
  const index = project.seams.findIndex(seam => seam.frame === frame);
  assert.ok(index >= 0, `Frame ${frame} exists in the project`);
  await page.getByRole('button', { name: new RegExp(`^Seam ${index + 1},`) }).click();
  await page.waitForFunction(value => document.querySelector('#seam-frame')?.value === String(value), frame);
}

async function applySeamSettings(page) {
  await page.getByRole('button', { name: 'Apply seam settings', exact: true }).click();
  await page.waitForFunction(() => !document.querySelector('.job-panel') && document.querySelector('#seam-frame')?.disabled === false);
  assert.equal(await page.locator('.notification.is-error').count(), 0, (await page.locator('.notification.is-error').allTextContents()).join('\n'));
}

async function checkReviewedImport(application, page, project, output, checks) {
  // A real analysis supplies the source fingerprint and calibration schema. The
  // explicitly reviewed identity below is a test input, never a fabricated
  // analysis result: it exercises importing a user's chosen correction.
  const seam = project.seams.find(row => row.enabled);
  assert.ok(seam, 'The workflow has an enabled seam to review');
  await selectSeam(page, project, seam.frame);
  const readout = page.getByRole('region', { name: 'Applied correction', exact: true });
  await readout.waitFor();
  assert.ok((await readout.innerText()).trim().length > 0, 'Analyzed seam reports the applied correction');
  assert.ok(project.seamResults?.some(row => row.frame === seam.frame), 'The readout comes from real backend analysis');
  await readout.screenshot({ path: path.join(output, 'analyzed-seam.png') });
  const calibration = JSON.parse(fs.readFileSync(project.artifacts.calibration, 'utf8'));
  const reviewed = {
    ...calibration,
    cuts: [{ frame: seam.frame, right_to_left_matrix: [[1, 0, 0], [0, 1, 0], [0, 0, 1]], pre_rate: [0, 0, 0, 0], post_rate: [0, 0, 0, 0], ease_rate: false }],
    excluded_geometry: [],
  };
  const before = fs.readFileSync(project.projectPath, 'utf8');
  await chooseNativeFile(application, null);
  assert.equal(await page.evaluate(options => window.seamstress.importSeamCorrection(options), { projectPath: project.projectPath, frame: seam.frame }), null, 'Canceling the native picker does not import');
  assert.equal(fs.readFileSync(project.projectPath, 'utf8'), before, 'Canceling preserves saved corrections and previews');
  const rejected = [
    ['wrong-source', { ...reviewed, source_sha256: '0'.repeat(64) }, /source|fingerprint/i],
    ['wrong-frame', { ...reviewed, cuts: [{ ...reviewed.cuts[0], frame: seam.frame === 1 ? 2 : 1 }] }, /frame|cut|seam/i],
    ['excluded-frame', { ...reviewed, excluded_geometry: [{ frame: seam.frame, reason: 'Unreliable geometry in this test recipe' }] }, /excluded|review|geometry/i],
  ];
  for (const [name, value, expected] of rejected) {
    const file = path.join(output, `reviewed-${name}.json`);
    fs.writeFileSync(file, JSON.stringify(value));
    await chooseNativeFile(application, file);
    const error = await page.evaluate(async options => { try { await window.seamstress.importSeamCorrection(options); return ''; } catch (reason) { return reason.message; } }, { projectPath: project.projectPath, frame: seam.frame });
    assert.match(error, expected, `${name} calibration is rejected through production IPC`);
    assert.equal(fs.readFileSync(project.projectPath, 'utf8'), before, `${name} cannot mutate the project or discard existing previews`);
  }
  const reviewedPath = path.join(output, 'reviewed-identity.json');
  fs.writeFileSync(reviewedPath, JSON.stringify(reviewed));
  await chooseNativeFile(application, reviewedPath);
  await page.getByRole('button', { name: 'Import reviewed framing', exact: true }).click();
  await page.waitForFunction(() => document.querySelector('#seam-frame')?.disabled === false && [...document.querySelectorAll('select')].some(select => select.value === 'manual'));
  let imported = await readProject(page, project.projectPath);
  let correction = imported.seams.find(row => row.frame === seam.frame).correction;
  assert.equal(correction.geometry, 'manual');
  assert.deepEqual(correction.manual.right_to_left_matrix, reviewed.cuts[0].right_to_left_matrix);
  assert.equal(correction.manual.provenance.source_sha256, project.sourceSha256);
  assert.equal(correction.manual.provenance.frame, seam.frame);
  assert.match(correction.manual.provenance.calibration_sha256, /^[a-f0-9]{64}$/);
  assert.ok(imported.revision > project.revision, 'Importing reviewed framing advances the saved revision');
  for (const key of ['plan', 'calibration', 'report', 'fullPreview', 'seamPreviews', 'export']) {
    assert.equal(imported.artifacts[key], undefined, `Importing framing invalidates stale ${key}`);
    if (typeof project.artifacts[key] === 'string') assert.ok(fs.existsSync(project.artifacts[key]), `Existing ${key} remains on disk`);
  }
  assert.equal(imported.seamResults?.length || 0, 0, 'Stale applied-correction readouts are removed');
  assert.equal(await page.locator('.candidate-video').count(), 0, 'Stale corrected media is removed from the viewer');
  assert.match(await readout.innerText(), /Analyze to see/i, 'The inspector no longer presents a stale applied result');
  await page.getByText('Edit custom measurements', { exact: true }).click();
  await page.getByLabel('Horizontal shift (tx)', { exact: true }).fill('0.5');
  await page.getByLabel('Framing', { exact: true }).selectOption('off');
  await page.getByLabel('Framing', { exact: true }).selectOption('manual');
  assert.equal(Number(await page.getByLabel('Horizontal shift (tx)', { exact: true }).inputValue()), .5, 'An unapplied custom edit survives switching Off and back to Custom');
  await applySeamSettings(page);
  imported = await readProject(page, project.projectPath);
  correction = imported.seams.find(row => row.frame === seam.frame).correction;
  assert.equal(correction.manual.right_to_left_matrix[0][2], .5, 'Custom affine editor saves native-pixel shifts');
  assert.equal(correction.manual.provenance.kind, 'manual', 'An edited import records a new review decision');
  assert.equal(correction.manual.provenance.calibration_sha256, undefined, 'An edited import does not claim exact calibration provenance');
  await chooseNativeFile(application, project.projectPath);
  await page.getByRole('button', { name: 'Open', exact: true }).click();
  await page.waitForFunction(() => !document.querySelector('.job-panel') && document.querySelector('#seam-frame')?.disabled === false);
  await selectSeam(page, imported, seam.frame);
  assert.equal(await page.getByLabel('Framing', { exact: true }).inputValue(), 'manual', 'Custom mode survives reopening');
  if (!(await page.getByLabel('Horizontal shift (tx)', { exact: true }).isVisible())) await page.getByText('Edit custom measurements', { exact: true }).click();
  assert.equal(Number(await page.getByLabel('Horizontal shift (tx)', { exact: true }).inputValue()), .5, 'Custom transform survives reopening');
  await page.getByLabel('Framing', { exact: true }).selectOption('off');
  await applySeamSettings(page);
  const temporarilyOff = await readProject(page, project.projectPath);
  assert.equal(temporarilyOff.seams.find(row => row.frame === seam.frame).correction.manual.right_to_left_matrix[0][2], .5, 'Turning framing off preserves reviewed measurements');
  await page.getByLabel('Framing', { exact: true }).selectOption('manual');
  assert.equal(Number(await page.getByLabel('Horizontal shift (tx)', { exact: true }).inputValue()), .5, 'Turning custom framing back on restores the measurements');
  await applySeamSettings(page);
  if (!(await page.getByLabel('Horizontal shift (tx)', { exact: true }).isVisible())) await page.getByText('Edit custom measurements', { exact: true }).click();
  await page.getByLabel('Horizontal shift (tx)', { exact: true }).scrollIntoViewIfNeeded();
  await page.screenshot({ path: path.join(output, 'seam-settings.png') });
  const movedFrame = Array.from({ length: project.metadata.frame_count - 1 }, (_, index) => index + 1).find(frame => !project.seams.some(row => row.frame === frame));
  assert.ok(movedFrame, 'The short source has an unmarked frame for the movement check');
  await page.locator('#seam-frame').fill(String(movedFrame));
  await page.locator('#seam-frame').press('Enter');
  await page.waitForFunction(value => document.querySelector('#seam-frame')?.value === String(value) && document.querySelector('#seam-frame')?.disabled === false, movedFrame);
  const moved = (await readProject(page, project.projectPath)).seams.find(row => row.frame === movedFrame);
  assert.equal(moved.correction.geometry, 'auto', 'Moving a reviewed boundary restores automatic inference');
  assert.equal(moved.correction.manual, undefined, 'Exact-frame measurements cannot move silently to another boundary');
  checks.push('real per-seam analysis readout', 'native reviewed-calibration import', 'source/frame/exclusion import validation', 'canceled and rejected imports preserve project', 'reviewed correction invalidates stale artifacts without deleting files', 'custom affine editor + reopen', 'unapplied custom edits survive mode toggles', 'reversible custom framing toggle retains reviewed measurements', 'moving a seam clears exact-frame custom geometry');
}

async function checkSingleSeamRefinement(page, project, output, checks) {
  const seam = project.seams.find(row => row.enabled);
  await selectSeam(page, project, seam.frame);
  const baselineBytes = fs.readFileSync(project.artifacts.plan, 'utf8');
  const baseline = JSON.parse(baselineBytes);
  await page.getByLabel('Framing', { exact: true }).selectOption('off');
  await page.getByLabel('Color', { exact: true }).selectOption('off');
  await applySeamSettings(page);
  const edited = await readProject(page, project.projectPath);
  assert.equal(edited.refinementBaseline.plan, project.artifacts.plan, 'Editing only this seam retains the accepted plan');
  await page.getByRole('button', { name: 'Refine this seam only', exact: true }).click();
  await page.waitForFunction(() => !!document.querySelector('.job-panel') || !!document.querySelector('.notification.is-error'));
  await page.waitForFunction(() => !document.querySelector('.job-panel'), undefined, { timeout: 180000 });
  assert.equal(await page.locator('.notification.is-error').count(), 0, (await page.locator('.notification.is-error').allTextContents()).join('\n'));
  let refined = await readProject(page, project.projectPath);
  const plan = JSON.parse(fs.readFileSync(refined.artifacts.plan, 'utf8'));
  assert.equal(plan.refinement.frame, seam.frame, 'UI sends the selected incoming frame to the worker');
  assert.deepEqual(plan.view_matrix, baseline.view_matrix, 'Refinement preserves the full-shot viewing crop');
  const { start_frame: start, end_frame: end } = plan.refinement;
  assert.deepEqual(plan.frame_matrices.slice(0, start), baseline.frame_matrices.slice(0, start));
  assert.deepEqual(plan.frame_matrices.slice(end), baseline.frame_matrices.slice(end));
  assert.deepEqual(refined.seamResults.filter(row => row.frame !== seam.frame), project.seamResults.filter(row => row.frame !== seam.frame));
  assert.equal(fs.readFileSync(project.artifacts.plan, 'utf8'), baselineBytes, 'Refining creates a new plan without altering the accepted artifact');
  await page.getByRole('button', { name: 'Preview this seam', exact: true }).click();
  await page.waitForFunction(() => !!document.querySelector('.job-panel') || !!document.querySelector('.notification.is-error'));
  await page.waitForFunction(() => !document.querySelector('.job-panel'), undefined, { timeout: 180000 });
  assert.equal(await page.locator('.notification.is-error').count(), 0, (await page.locator('.notification.is-error').allTextContents()).join('\n'));
  refined = await readProject(page, project.projectPath);
  assert.equal(refined.artifacts.fullPreview, undefined, 'A selected preview does not render the entire shot');
  assert.equal(refined.artifacts.seamPreviews.length, 1);
  assert.equal(refined.artifacts.seamPreviews[0].frame, seam.frame);
  assert.match(await page.getByRole('button', { name: 'Selected seam', exact: true }).getAttribute('class'), /active/, 'Preview this seam switches playback to its review interval');
  await page.waitForFunction(() => document.querySelector('.candidate-video')?.readyState >= 2, undefined, { timeout: 60000 });
  await page.screenshot({ path: path.join(output, 'single-seam-refinement.png') });
  checks.push('selected-seam refinement through real IPC', 'baseline artifact and neighboring corrections preserved', 'selected preview without full render', 'selected corrected clip decodes');
  return refined;
}

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
        return await other.webContents.executeJavaScript("Promise.all(['update:state', 'project:import-correction'].map(channel => require('electron').ipcRenderer.invoke(channel).then(()=>'accepted',error=>error.message)))");
      } finally { other.destroy(); }
    });
    assert.equal(rejectedSender.length, 2);
    for (const rejection of rejectedSender) assert.match(rejection, /Untrusted request/, 'Updater and reviewed-import IPC refuse another webContents');
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
    await page.getByLabel('Color', { exact: true }).selectOption('tone');
    await page.getByText('Automatic checks', { exact: true }).click();
    await page.getByLabel('Partial framing recovery', { exact: true }).uncheck();
    await page.getByLabel('Repeated endpoint recovery', { exact: true }).uncheck();
    await page.getByLabel('Animation cadence check', { exact: true }).uncheck();
    await page.getByLabel('Camera-rate easing', { exact: true }).uncheck();
    await page.getByLabel('Framing', { exact: true }).selectOption('off');
    await applySeamSettings(page);
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
    const reopened = await readProject(page, project.projectPath);
    await selectSeam(page, reopened, before + 2);
    const savedSettings = reopened.seams.find(seam => seam.frame === before + 2).correction;
    assert.equal(savedSettings.geometry, 'off');
    assert.equal(savedSettings.color, 'tone');
    for (const key of ['partial_recovery', 'endpoint_recovery', 'cadence', 'rate_easing']) assert.equal(savedSettings[key], false, `${key} persists after reopen`);
    assert.equal(await page.getByLabel('Framing', { exact: true }).inputValue(), 'off');
    assert.equal(await page.getByLabel('Color', { exact: true }).inputValue(), 'tone');
    for (const label of ['Partial framing recovery', 'Repeated endpoint recovery', 'Animation cadence check', 'Camera-rate easing']) assert.equal(await page.getByLabel(label, { exact: true }).isChecked(), false, `${label} displays its saved value`);
    await page.getByRole('button', { name: 'Remove seam', exact: true }).click();
    await page.waitForFunction(count => document.querySelectorAll('.seam-marker').length === count && !document.querySelector('.job-panel'), initialSeams);
    const checks = ['updater IPC with no live network', 'update menu + keyboard focus', 'inert release notes + update progress fixtures', 'update install controls respect busy state', 'fixed release URL', 'untrusted updater and reviewed-import senders rejected', 'update listener cleanup', 'real import updates busy guard', 'real import + detection', 'custom-protocol media decoding', 'manual seam creation', 'frame edit', 'toggle', 'frame step', 'per-seam settings persist after project reopen', 'seam deletion', 'renderer errors'];
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
      const refined = await checkSingleSeamRefinement(page, finished, output, checks);
      await checkReviewedImport(application, page, refined, output, checks);
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
