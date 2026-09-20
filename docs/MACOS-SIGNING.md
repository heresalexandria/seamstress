# Signing and notarizing Seamstress

Release builds use an Apple **Developer ID Application** certificate. The Python
worker, its extension modules and libraries, FFmpeg, Electron and its helpers all
receive the same team's signature, a secure timestamp, and hardened runtime.
The release is stopped if signing, notarization, Gatekeeper verification, or the
signed worker's encode/decode self-test fails.

`app/electron-builder.release.cjs` is the release configuration for the pinned
electron-builder 26 version. The base configuration in `app/package.json` remains
an explicitly unsigned local development build; it is never used to publish a
release.

## GitHub Actions secrets

Configure these secrets on `heresalexandria/seamstress`:

| Secret | Value |
| --- | --- |
| `CSC_LINK` | Base64 of the encrypted `.p12` containing the Developer ID Application certificate **and its private key**. |
| `CSC_KEY_PASSWORD` | The password used to encrypt that `.p12`. |
| `APPLE_API_KEY` | The complete private `.p8` contents of the App Store Connect team API key used for notarization. The workflow writes this to a private temporary file and passes its path to the tools. |
| `APPLE_API_KEY_ID` | That API key's key ID. |
| `APPLE_API_ISSUER` | The App Store Connect team API issuer ID. |
| `APPLE_TEAM_ID` | The ten-character team ID on the Developer ID Application certificate. |

The certificate must be an exportable Developer ID Application identity, not an
Apple Development, Mac App Distribution, or Developer ID Installer certificate.
The API key authenticates notarization; it does not replace the signing
certificate. No App Store provisioning profile is needed for this distribution.
Credentials belong in the keychain or Actions secrets, never source files.

The workflow sets `SEAMSTRESS_RELEASE=1` and imports the certificate through
electron-builder's temporary-keychain support. All six credentials are required
before a CI release build starts. There is no unsigned-release fallback.

## Local signed builds

Use a native Mac for the intended architecture, install the documented build
dependencies, and supply the same environment variables. Locally, `APPLE_API_KEY`
is the path to the `.p8` file, not its text. An existing keychain identity can be
selected with `CSC_NAME` instead of `CSC_LINK` and `CSC_KEY_PASSWORD`.

An existing `APPLE_KEYCHAIN_PROFILE` notarization profile is also supported
instead of API-key credentials. `APPLE_ID` with `APPLE_APP_SPECIFIC_PASSWORD` is
supported as another explicit alternative; do not set multiple credential
families. `APPLE_TEAM_ID` is required whichever method you use.

Run from `app/`, with a clean `release/` directory:

```sh
export SEAMSTRESS_RELEASE=1
node ../scripts/signing/preflight.cjs
npm run build
npm run package:backend
npx electron-builder --config electron-builder.release.cjs --mac dmg zip --publish never
node ../scripts/signing/finalize-release.cjs release
```

Choose `--arm64` or `--x64` only on a matching native runner: the bundled Python
runtime and FFmpeg are compiled for the build machine. Do not cross-package an
Intel Electron shell with an Apple Silicon worker, or vice versa.

## What is checked before publication

1. `sign-app.cjs` signs every real Mach-O file in the backend and any enclosing
   frameworks, then delegates signing of Electron to `@electron/osx-sign`.
   Symlinks must stay inside the bundled tree.
2. Electron's main process and helpers receive only the `allow-jit` exception
   needed by V8. Python and FFmpeg receive no runtime exceptions. Library
   validation and debugger restrictions stay enabled.
3. electron-builder notarizes and staples the app. The `after-sign.cjs` hook
   verifies every executable's team, certificate type, timestamp and runtime,
   validates the app ticket, runs Gatekeeper assessment, and runs the relocated
   worker without Homebrew or the developer's Python on `PATH`.
4. The finalizer extracts and verifies the actual updater ZIP, including its
   architecture and worker. It independently notarizes and staples the DMG,
   validates its ticket, and runs the disk-image Gatekeeper assessment. It mounts
   the installer read-only and checks that its app has the same signature as the
   updater ZIP, the expected version, and a valid stapled ticket.
5. It checks the ZIP hashes and sizes in the update metadata, then writes
   `verification-arm64.json` or `verification-x64.json` and `SHA256SUMS.txt`.
   CI publishes only after both native builds pass.

Only the ZIP appears in `latest-mac.yml`, because Squirrel.Mac uses that archive
for in-app updates. The separately notarized DMG is the download/install asset.
`dmg.writeUpdateInfo=false` deliberately prevents a checksum calculated before
DMG stapling from appearing in update metadata. All assets are built with
`--publish never`; CI publishes them only after final verification succeeds.

If Apple rejects a submission, inspect the notary log, fix the reported binary
or entitlement, and rebuild. Do not disable Gatekeeper or remove quarantine as
a release workaround. A normal first-launch confirmation naming the identified
developer can still appear on macOS; signing and notarization establish trust
rather than suppressing the operating system's normal confirmation.

## References

- [Apple: notarizing macOS software before distribution](https://developer.apple.com/documentation/security/notarizing-macos-software-before-distribution)
- [Apple: resolving common notarization issues](https://developer.apple.com/documentation/security/resolving-common-notarization-issues)
- [Electron: code signing](https://www.electronjs.org/docs/latest/tutorial/code-signing)
- [Electron notarization tooling](https://github.com/electron/notarize)

The implementation follows the installed electron-builder 26 schema. Newer
electron-builder documentation may describe the incompatible version 27
`mac.sign` configuration object; upgrading requires reviewing this custom signing
hook and release tests together.
