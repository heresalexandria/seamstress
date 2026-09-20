'use strict';

const path = require('node:path');
const base = require('./package.json').build;

// Keep local packaging explicitly unsigned. Release CI selects this config
// explicitly and cannot turn missing certificates into a successful build.
const mac = {
  ...base.mac,
  hardenedRuntime: true,
  entitlements: 'build/entitlements.mac.plist',
  entitlementsInherit: 'build/entitlements.mac.inherit.plist',
  gatekeeperAssess: false,
  strictVerify: true,
  preAutoEntitlements: false,
  notarize: true,
  sign: path.resolve(__dirname, '../scripts/signing/sign-app.cjs'),
};
delete mac.identity;

module.exports = {
  ...base,
  extends: null,
  forceCodeSigning: true,
  mac,
  // A later step notarizes/staples the installer. Stapling changes DMG bytes,
  // so only the ZIP is advertised in Squirrel.Mac's update metadata.
  dmg: { ...base.dmg, sign: true, writeUpdateInfo: false },
  afterSign: path.resolve(__dirname, '../scripts/signing/after-sign.cjs'),
  publish: [{ provider: 'github', owner: 'heresalexandria', repo: 'seamstress', releaseType: 'release' }],
};
