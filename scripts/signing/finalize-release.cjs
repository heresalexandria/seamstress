'use strict';

const fs = require('node:fs/promises');
const os = require('node:os');
const path = require('node:path');
const crypto = require('node:crypto');
const { createReadStream } = require('node:fs');
const {
  appRequire, run, requireTeam, notarizationOptions, assertReleaseEnvironment,
  verifyCode, verifyApp,
} = require('./common.cjs');

async function digest(file, algorithm, encoding) {
  const hash = crypto.createHash(algorithm);
  for await (const block of createReadStream(file)) hash.update(block);
  return hash.digest(encoding);
}

async function appIdentity(app, expectedVersion) {
  const info = JSON.parse(await run('/usr/bin/plutil', ['-convert', 'json', '-o', '-', path.join(app, 'Contents/Info.plist')]));
  const expectedId = appRequire('./package.json').build.appId;
  if (info.CFBundleIdentifier !== expectedId || info.CFBundleShortVersionString !== expectedVersion) {
    throw new Error('Packaged application has an unexpected bundle identifier or version.');
  }
  const detail = await run('/usr/bin/codesign', ['--display', '--verbose=4', app]);
  const cdhash = detail.match(/^CDHash=([0-9a-f]+)$/m)?.[1];
  if (!cdhash) throw new Error('Packaged application has no code-directory hash.');
  return { version: info.CFBundleShortVersionString, bundleId: info.CFBundleIdentifier, cdhash };
}

async function finalize(directory) {
  assertReleaseEnvironment();
  const team = requireTeam();
  const root = path.resolve(directory);
  const files = (await fs.readdir(root)).sort();
  const dmgs = files.filter(name => name.endsWith('.dmg'));
  const zips = files.filter(name => name.endsWith('.zip'));
  if (!dmgs.length || !zips.length) throw new Error('A signed release must contain both the installer DMG and the updater ZIP.');
  const { notarize } = appRequire('@electron/notarize');
  const { load: loadYaml } = appRequire('js-yaml');
  const version = appRequire('./package.json').version;
  const arch = process.arch;
  if (!['arm64', 'x64'].includes(arch)) throw new Error(`Unsupported native release architecture: ${arch}`);
  if (zips.length !== 1 || dmgs.length !== 1 || zips[0] !== `Seamstress-${version}-${arch}.zip` || dmgs[0] !== `Seamstress-${version}-${arch}.dmg`) {
    throw new Error('Release output must contain only the current version and native architecture. Clean the release directory before building.');
  }
  const report = { version, arch, team_id: team, archives: [], installers: [], updateMetadata: [] };

  for (const name of zips) {
    const temporary = await fs.mkdtemp(path.join(os.tmpdir(), 'seamstress-release-verification-'));
    try {
      // The update archive must contain the same signed, stapled runnable app.
      await run('/usr/bin/ditto', ['-x', '-k', path.join(root, name), temporary], { timeout: 600000 });
      const apps = (await fs.readdir(temporary)).filter(entry => entry.endsWith('.app'));
      if (apps.length !== 1) throw new Error(`${name} must contain exactly one application.`);
      const app = path.join(temporary, apps[0]);
      const executable = path.join(app, 'Contents/MacOS', path.basename(app, '.app'));
      const worker = path.join(app, 'Contents/Resources/backend/seamstress-worker/seamstress-worker');
      const nativeArch = arch === 'x64' ? 'x86_64' : 'arm64';
      for (const binary of [executable, worker]) {
        const arches = (await run('/usr/bin/lipo', ['-archs', binary])).trim().split(/\s+/);
        if (!arches.includes(nativeArch)) throw new Error(`${name} does not contain the expected ${arch} executable.`);
      }
      report.archives.push({ file: name, ...await verifyApp(app), ...await appIdentity(app, version) });
    } finally {
      await fs.rm(temporary, { recursive: true, force: true });
    }
  }
  for (const name of dmgs) {
    const dmg = path.join(root, name);
    await verifyCode(dmg, team, { runtime: false });
    await notarize({ appPath: dmg, ...notarizationOptions() });
    await run('/usr/bin/xcrun', ['stapler', 'validate', dmg]);
    await run('/usr/sbin/spctl', ['--assess', '--type', 'open', '--context', 'context:primary-signature', '--verbose=4', dmg]);
    const mounted = await fs.mkdtemp(path.join(os.tmpdir(), 'seamstress-installer-verification-'));
    let attached = false;
    try {
      await run('/usr/bin/hdiutil', ['attach', '-readonly', '-nobrowse', '-mountpoint', mounted, dmg], { timeout: 300000 });
      attached = true;
      const apps = (await fs.readdir(mounted)).filter(entry => entry.endsWith('.app'));
      if (apps.length !== 1) throw new Error(`${name} must contain exactly one application.`);
      const app = path.join(mounted, apps[0]);
      const identity = await appIdentity(app, version);
      if (identity.cdhash !== report.archives[0].cdhash) throw new Error('The DMG and updater ZIP contain different application signatures.');
      await run('/usr/bin/codesign', ['--verify', '--deep', '--strict', '--all-architectures', app]);
      await run('/usr/bin/xcrun', ['stapler', 'validate', app]);
      await run('/usr/sbin/spctl', ['--assess', '--type', 'execute', '--verbose=4', app]);
      report.installers.push({ file: name, notarized: true, ...identity });
    } finally {
      // Never recursively remove a live mount if verification or detaching fails.
      if (attached) await run('/usr/bin/hdiutil', ['detach', mounted]);
      await fs.rmdir(mounted);
    }
  }

  for (const name of files.filter(name => /^(latest|beta|alpha).*\.yml$/.test(name))) {
    const metadata = loadYaml(await fs.readFile(path.join(root, name), 'utf8'));
    if (!Array.isArray(metadata?.files) || !metadata.files.length) throw new Error(`${name} has no update archives.`);
    for (const item of metadata.files) {
      // Stapling a DMG changes its bytes. DMGs deliberately stay out of the
      // updater metadata; Squirrel.Mac installs the signed ZIP instead.
      if (typeof item.url !== 'string' || path.basename(item.url) !== item.url || !item.url.endsWith('.zip')) {
        throw new Error(`${name} must reference only local ZIP archives. Set dmg.writeUpdateInfo=false.`);
      }
      const archive = path.join(root, item.url);
      if (item.sha512 !== await digest(archive, 'sha512', 'base64')) throw new Error(`${name} contains an incorrect ZIP checksum.`);
      if (item.size !== (await fs.stat(archive)).size) throw new Error(`${name} contains an incorrect ZIP size.`);
    }
    if (metadata.path && (typeof metadata.path !== 'string' || path.basename(metadata.path) !== metadata.path || !metadata.files.some(item => item.url === metadata.path && item.sha512 === metadata.sha512))) {
      throw new Error(`${name} contains inconsistent legacy update metadata.`);
    }
    report.updateMetadata.push(name);
  }
  if (!report.updateMetadata.length) throw new Error('No release update metadata was generated. Configure the GitHub publish provider.');

  const assets = [...dmgs, ...zips, ...files.filter(name => name.endsWith('.blockmap')), ...report.updateMetadata].sort();
  const sums = [];
  for (const file of assets) sums.push(`${await digest(path.join(root, file), 'sha256', 'hex')}  ${file}`);
  await fs.writeFile(path.join(root, 'SHA256SUMS.txt'), sums.join('\n') + '\n');
  report.signed = true;
  report.notarized = true;
  await fs.writeFile(path.join(root, `verification-${arch}.json`), JSON.stringify(report, null, 2) + '\n');
  console.log(JSON.stringify(report));
  return report;
}

if (require.main === module) {
  if (process.argv.length !== 3) {
    console.error('Usage: node scripts/signing/finalize-release.cjs <electron-builder output directory>');
    process.exitCode = 1;
  } else {
    finalize(process.argv[2]).catch(error => { console.error(error.message); process.exitCode = 1; });
  }
}

module.exports = { finalize, digest, appIdentity };
