'use strict';

const fs = require('node:fs/promises');
const path = require('node:path');
const {
  appRequire, run, requireTeam, assertReleaseEnvironment,
  codeInventory, verifyCode,
} = require('./common.cjs');

/** electron-builder 26 mac.sign hook, after its certificate import. */
module.exports = async function signApp(options) {
  assertReleaseEnvironment();
  if (!options.identity || options.identity === '-') {
    throw new Error('Refusing to release an unsigned or ad-hoc signed application.');
  }
  const team = requireTeam();
  const backend = path.join(options.app, 'Contents/Resources/backend');
  const inventory = await codeInventory(backend);
  if (inventory.binaries.length < 3) {
    throw new Error('The application is missing its bundled worker and FFmpeg executables.');
  }
  const sign = async file => {
    const args = ['--force', '--sign', options.identity, '--options', 'runtime', '--timestamp'];
    if (options.keychain) args.push('--keychain', options.keychain);
    // CPython and FFmpeg do not need V8's JIT or any library-validation
    // exception. Re-sign every bundled library with the same Developer ID.
    await run('/usr/bin/codesign', [...args, file]);
    await verifyCode(file, team);
  };
  for (const binary of inventory.binaries) await sign(binary);
  for (const bundle of inventory.bundles) await sign(bundle);

  const manifestPath = path.join(backend, 'build-manifest.json');
  const manifest = JSON.parse(await fs.readFile(manifestPath, 'utf8'));
  manifest.signing = 'Developer ID Application; hardened runtime; secure timestamp';
  manifest.distribution_signing = { team_id: team, macho_binary_count: inventory.binaries.length };
  await fs.writeFile(manifestPath, JSON.stringify(manifest, null, 2) + '\n');

  const { signAsync } = appRequire('@electron/osx-sign');
  const existingIgnore = options.ignore;
  const originalOptionsForFile = options.optionsForFile;
  await signAsync({
    ...options,
    gatekeeperAssess: false, // The next builder stage obtains the notarization ticket.
    preAutoEntitlements: false,
    ignore(file) {
      // Avoid granting Electron's JIT entitlements to the already signed worker.
      return file === backend || file.startsWith(backend + path.sep) || Boolean(existingIgnore?.(file));
    },
    optionsForFile(file) {
      const original = originalOptionsForFile ? originalOptionsForFile(file) : {};
      return {
        ...original,
        hardenedRuntime: true,
        // These explicit files replace older builder templates that disabled
        // library validation or allowed unrestricted executable memory.
        entitlements: path.resolve(__dirname, '../../app/build',
          file === options.app ? 'entitlements.mac.plist' : 'entitlements.mac.inherit.plist'),
      };
    },
  });
  await verifyCode(options.app, team);
  console.log(`Signed Electron and ${inventory.binaries.length} backend binaries with the configured Developer ID.`);
};
