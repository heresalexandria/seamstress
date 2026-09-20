'use strict';

const path = require('node:path');
const { assertReleaseEnvironment, verifyApp } = require('./common.cjs');

/** Runs after electron-builder's built-in notarization and stapling. */
module.exports = async function afterSign(context) {
  assertReleaseEnvironment();
  const app = path.join(context.appOutDir, `${context.packager.appInfo.productFilename}.app`);
  console.log(JSON.stringify(await verifyApp(app)));
};
