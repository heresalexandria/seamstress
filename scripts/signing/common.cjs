'use strict';

const fs = require('node:fs/promises');
const path = require('node:path');
const { execFile } = require('node:child_process');
const { promisify } = require('node:util');
const { createRequire } = require('node:module');

const execute = promisify(execFile);
const appRequire = createRequire(path.resolve(__dirname, '../../app/package.json'));
const MACHO_MAGIC = new Set([
  'feedface', 'cefaedfe', 'feedfacf', 'cffaedfe',
  'cafebabe', 'bebafeca', 'cafebabf', 'bfbafeca',
]);

async function run(program, args, options = {}) {
  // Never use a shell, print process environments, or echo secret arguments.
  try {
    const { stdout, stderr } = await execute(program, args, {
      maxBuffer: 16 * 1024 * 1024, timeout: 120000, ...options,
    });
    return stdout + stderr;
  } catch (error) {
    const detail = [error.stdout, error.stderr].filter(Boolean).join('\n').trim();
    throw new Error(`${path.basename(program)} failed (${error.code ?? error.signal ?? 'unknown'})${detail ? `: ${detail}` : ''}`);
  }
}

function requireTeam(env = process.env) {
  if (!/^[A-Z0-9]{10}$/.test(env.APPLE_TEAM_ID || '')) {
    throw new Error('APPLE_TEAM_ID must contain the 10-character Developer ID team identifier.');
  }
  return env.APPLE_TEAM_ID;
}

function notarizationOptions(env = process.env) {
  // Match electron-builder 26's credential precedence exactly. Do not silently
  // fall back when the caller supplied an incomplete credential family.
  if (env.APPLE_ID || env.APPLE_APP_SPECIFIC_PASSWORD) {
    if (!env.APPLE_ID || !env.APPLE_APP_SPECIFIC_PASSWORD) {
      throw new Error('Set both APPLE_ID and APPLE_APP_SPECIFIC_PASSWORD.');
    }
    return { appleId: env.APPLE_ID, appleIdPassword: env.APPLE_APP_SPECIFIC_PASSWORD, teamId: requireTeam(env) };
  }
  if (env.APPLE_API_KEY || env.APPLE_API_KEY_ID || env.APPLE_API_ISSUER) {
    if (!env.APPLE_API_KEY || !env.APPLE_API_KEY_ID || !env.APPLE_API_ISSUER) {
      throw new Error('Set APPLE_API_KEY, APPLE_API_KEY_ID and APPLE_API_ISSUER together.');
    }
    return { appleApiKey: env.APPLE_API_KEY, appleApiKeyId: env.APPLE_API_KEY_ID, appleApiIssuer: env.APPLE_API_ISSUER };
  }
  if (env.APPLE_KEYCHAIN_PROFILE) {
    return { keychainProfile: env.APPLE_KEYCHAIN_PROFILE, ...(env.APPLE_KEYCHAIN ? { keychain: env.APPLE_KEYCHAIN } : {}) };
  }
  throw new Error('Release notarization credentials are missing. Set the Apple API key, app-specific password, or keychain profile variables.');
}

function assertReleaseEnvironment(env = process.env) {
  if (env.SEAMSTRESS_RELEASE !== '1') {
    throw new Error('Signed distribution requires SEAMSTRESS_RELEASE=1. Use the explicit unsigned build command for local development.');
  }
  requireTeam(env);
  if (!env.CSC_LINK && !env.CSC_NAME) {
    throw new Error('A Developer ID Application identity is required: set CSC_LINK and CSC_KEY_PASSWORD, or CSC_NAME for an existing keychain identity.');
  }
  if (env.CSC_LINK && !env.CSC_KEY_PASSWORD) {
    throw new Error('CSC_KEY_PASSWORD is required for the encrypted Developer ID certificate.');
  }
  notarizationOptions(env);
}

async function isMachO(file) {
  const handle = await fs.open(file, 'r');
  try {
    const buffer = Buffer.alloc(4);
    const { bytesRead } = await handle.read(buffer, 0, 4, 0);
    return bytesRead === 4 && MACHO_MAGIC.has(buffer.toString('hex'));
  } finally {
    await handle.close();
  }
}

async function codeInventory(root) {
  const absoluteRoot = await fs.realpath(root);
  const binaries = [];
  const bundles = [];
  async function walk(directory) {
    for (const entry of await fs.readdir(directory, { withFileTypes: true })) {
      const location = path.join(directory, entry.name);
      if (entry.isSymbolicLink()) {
        const resolved = await fs.realpath(location);
        const relative = path.relative(absoluteRoot, resolved);
        if (relative === '..' || relative.startsWith(`..${path.sep}`) || path.isAbsolute(relative)) {
          throw new Error(`A bundled symlink escapes the signed tree: ${location}`);
        }
        // The real target is visited once, avoiding framework/symlink loops.
      } else if (entry.isDirectory()) {
        await walk(location);
        if (/\.(app|framework|xpc|bundle)$/.test(entry.name)) bundles.push(location);
      } else if (entry.isFile() && await isMachO(location)) {
        binaries.push(location);
      }
    }
  }
  await walk(absoluteRoot);
  return { binaries: binaries.sort(), bundles };
}

function assertSignatureDetails(details, file, team, { runtime = true } = {}) {
  if (!details.includes('Authority=Developer ID Application:')) {
    throw new Error(`${file} is not signed with a Developer ID Application certificate.`);
  }
  if (!details.split(/\r?\n/).includes(`TeamIdentifier=${team}`)) {
    throw new Error(`${file} is signed by a different Apple team.`);
  }
  if (!/^Timestamp=.+/m.test(details)) {
    throw new Error(`${file} has no secure timestamp.`);
  }
  if (runtime && !/^CodeDirectory .*flags=.*\bruntime\b/m.test(details)) {
    throw new Error(`${file} does not enable the hardened runtime.`);
  }
}

function assertEntitlements(entitlements, file, { backend = false } = {}) {
  const forbidden = [
    'com.apple.security.get-task-allow',
    'com.apple.security.cs.disable-library-validation',
    'com.apple.security.cs.disable-executable-page-protection',
    'com.apple.security.cs.allow-unsigned-executable-memory',
    'com.apple.security.cs.allow-dyld-environment-variables',
  ];
  if (backend) forbidden.push('com.apple.security.cs.allow-jit');
  for (const key of forbidden) {
    if (entitlements[key]) throw new Error(`${file} has an unexpected runtime exception: ${key}`);
  }
}

async function verifyCode(file, team, options = {}) {
  await run('/usr/bin/codesign', ['--verify', '--strict', '--all-architectures', file]);
  const details = await run('/usr/bin/codesign', ['--display', '--verbose=4', file]);
  assertSignatureDetails(details, file, team, options);
  if (options.runtime !== false) {
    const output = await run('/usr/bin/codesign', ['--display', '--entitlements', '-', '--xml', file]);
    const xml = output.match(/<plist[\s\S]*?<\/plist>/)?.[0];
    const entitlements = xml ? appRequire('plist').parse(xml) : {};
    assertEntitlements(entitlements, file, { backend: file.includes('/Contents/Resources/backend/') });
  }
}

async function verifyApp(app, { team = requireTeam(), notarized = true, smoke = true } = {}) {
  const inventory = await codeInventory(app);
  if (!inventory.binaries.length) throw new Error(`No executable code found in ${app}`);
  await verifyCode(app, team);
  await run('/usr/bin/codesign', ['--verify', '--deep', '--strict', '--all-architectures', app]);
  for (const binary of inventory.binaries) await verifyCode(binary, team);
  if (notarized) {
    await run('/usr/bin/xcrun', ['stapler', 'validate', app]);
    await run('/usr/sbin/spctl', ['--assess', '--type', 'execute', '--verbose=4', app]);
  }
  if (smoke) {
    const worker = path.join(app, 'Contents/Resources/backend/seamstress-worker/seamstress-worker');
    const env = { ...process.env, PATH: '/usr/bin:/bin:/usr/sbin:/sbin', OPENBLAS_NUM_THREADS: '2', OMP_NUM_THREADS: '2' };
    for (const key of ['PYTHONHOME', 'PYTHONPATH', 'DYLD_LIBRARY_PATH', 'DYLD_FALLBACK_LIBRARY_PATH']) delete env[key];
    const output = await run(worker, ['--self-test'], { env, cwd: '/private/tmp' });
    if (!output.split(/\r?\n/).some(line => {
      try { return JSON.parse(line).self_test === 'ok'; } catch { return false; }
    })) throw new Error('The signed bundled worker self-test did not pass.');
  }
  return { app: path.basename(app), team, binaries: inventory.binaries.length, notarized, workerSmokeTest: smoke };
}

module.exports = {
  appRequire, run, requireTeam, notarizationOptions, assertReleaseEnvironment,
  isMachO, codeInventory, assertSignatureDetails, assertEntitlements, verifyCode, verifyApp,
};
