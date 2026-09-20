'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs/promises');
const os = require('node:os');
const path = require('node:path');
const {
  requireTeam, notarizationOptions, assertReleaseEnvironment,
  codeInventory, assertSignatureDetails, assertEntitlements,
} = require('./common.cjs');

const valid = {
  SEAMSTRESS_RELEASE: '1', APPLE_TEAM_ID: 'ABC123DE45',
  CSC_LINK: '/private/certificate.p12', CSC_KEY_PASSWORD: 'test-only-password',
  APPLE_API_KEY: '/private/AuthKey.p8', APPLE_API_KEY_ID: 'ABC123DE45',
  APPLE_API_ISSUER: '12345678-1234-1234-1234-123456789abc',
};

test('release preflight refuses unsigned fallback and incomplete credentials', () => {
  assert.doesNotThrow(() => assertReleaseEnvironment(valid));
  for (const key of Object.keys(valid)) {
    const missing = { ...valid }; delete missing[key];
    assert.throws(() => assertReleaseEnvironment(missing), undefined, `Must reject missing ${key}`);
  }
  assert.throws(() => requireTeam({ APPLE_TEAM_ID: 'another team' }));
  assert.doesNotThrow(() => assertReleaseEnvironment({ ...valid, CSC_LINK: '', CSC_KEY_PASSWORD: '', CSC_NAME: 'Local Developer ID' }));
});

test('notary credential precedence matches electron-builder without fallback', () => {
  assert.deepEqual(notarizationOptions(valid), {
    appleApiKey: valid.APPLE_API_KEY,
    appleApiKeyId: valid.APPLE_API_KEY_ID,
    appleApiIssuer: valid.APPLE_API_ISSUER,
  });
  assert.throws(() => notarizationOptions({ ...valid, APPLE_ID: 'test@example.invalid' }));
  assert.deepEqual(notarizationOptions({ APPLE_KEYCHAIN_PROFILE: 'local-profile' }), { keychainProfile: 'local-profile' });
  assert.throws(() => notarizationOptions({ APPLE_KEYCHAIN: '/tmp/keychain' }));
});

test('every distributed executable must have Developer ID, expected team, timestamp and runtime', () => {
  const detail = [
    'CodeDirectory v=20500 size=423 flags=0x10000(runtime) hashes=8+7 location=embedded',
    'Authority=Developer ID Application: Example (ABC123DE45)',
    'Authority=Developer ID Certification Authority',
    'Timestamp=Sep 20, 2026 at 12:00:00 PM',
    'TeamIdentifier=ABC123DE45',
  ].join('\n');
  assert.doesNotThrow(() => assertSignatureDetails(detail, 'worker', valid.APPLE_TEAM_ID));
  for (const required of ['Authority=Developer ID Application:', 'TeamIdentifier=ABC123DE45', 'Timestamp=', 'runtime']) {
    assert.throws(() => assertSignatureDetails(detail.replace(required, 'removed'), 'worker', valid.APPLE_TEAM_ID));
  }
  assert.doesNotThrow(() => assertSignatureDetails(detail.replace('runtime', 'none'), 'installer.dmg', valid.APPLE_TEAM_ID, { runtime: false }));
});

test('code inventory signs real Mach-O targets once and refuses escaping links', async () => {
  const root = await fs.realpath(await fs.mkdtemp(path.join(os.tmpdir(), 'seamstress-sign-test-')));
  try {
    await fs.mkdir(path.join(root, 'Python.framework/Versions/A'), { recursive: true });
    const binary = path.join(root, 'Python.framework/Versions/A/Python');
    await fs.writeFile(binary, Buffer.from('cffaedfe00000000', 'hex'));
    await fs.writeFile(path.join(root, 'movie.bin'), Buffer.from('0000000061626364', 'hex'));
    await fs.symlink('A', path.join(root, 'Python.framework/Versions/Current'));
    await fs.symlink('Versions/Current/Python', path.join(root, 'Python.framework/Python'));
    const inventory = await codeInventory(root);
    assert.deepEqual(inventory.binaries, [binary]);
    assert.deepEqual(inventory.bundles, [path.join(root, 'Python.framework')]);
    await fs.symlink('/usr/bin/true', path.join(root, 'external'));
    await assert.rejects(codeInventory(root), /escapes/);
  } finally { await fs.rm(root, { recursive: true, force: true }); }
});

test('release verification rejects weakened runtime entitlements', () => {
  assert.doesNotThrow(() => assertEntitlements({ 'com.apple.security.cs.allow-jit': true }, 'Electron'));
  assert.throws(() => assertEntitlements({ 'com.apple.security.cs.allow-jit': true }, 'worker', { backend: true }));
  for (const key of ['com.apple.security.get-task-allow', 'com.apple.security.cs.disable-library-validation', 'com.apple.security.cs.allow-unsigned-executable-memory']) {
    assert.throws(() => assertEntitlements({ [key]: true }, 'Electron'));
    assert.doesNotThrow(() => assertEntitlements({ [key]: false }, 'Electron'));
  }
});

test('release configuration cannot inherit unsigned local settings', () => {
  const release = require('../../app/electron-builder.release.cjs');
  assert.equal(release.forceCodeSigning, true);
  assert.equal(Object.hasOwn(release.mac, 'identity'), false);
  assert.equal(release.mac.hardenedRuntime, true);
  assert.equal(release.mac.notarize, true);
  assert.equal(release.mac.strictVerify, true);
  assert.equal(release.dmg.sign, true);
  assert.equal(release.dmg.writeUpdateInfo, false);
  assert.equal(release.publish[0].owner, 'heresalexandria');
  assert.equal(release.publish[0].repo, 'seamstress');
});

test('runtime entitlements do not disable library validation or enable debugging', async () => {
  for (const file of ['entitlements.mac.plist', 'entitlements.mac.inherit.plist']) {
    const xml = await fs.readFile(path.resolve(__dirname, '../../app/build', file), 'utf8');
    assert.match(xml, /com\.apple\.security\.cs\.allow-jit/);
    assert.doesNotMatch(xml, /disable-library-validation|allow-unsigned-executable-memory|get-task-allow/);
  }
});
