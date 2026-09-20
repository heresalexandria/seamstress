'use strict';

const { assertReleaseEnvironment } = require('./common.cjs');

try {
  assertReleaseEnvironment();
  console.log('Developer ID signing and notarization inputs are present.');
} catch (error) {
  console.error(error.message);
  process.exitCode = 1;
}
