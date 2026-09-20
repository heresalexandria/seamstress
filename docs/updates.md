# Updating Seamstress

The running version appears in the title bar. Click it to open the update
dialog and check GitHub Releases. When a newer release is available, an
**Update available** button appears beside the version.

1. Choose **Download update**. You can keep editing while it downloads.
2. Finish or cancel any import, analysis, preview or export.
3. Choose **Restart & install update**. Seamstress verifies the new app's
   macOS signing identity before restarting into it.

Videos and saved projects live outside the application bundle and are preserved.
Downloads and installation both require an explicit action. Closing the app
after downloading does not request installation.

## Automatic checks

The installed macOS app checks on launch, then checks at most once every
24 hours when a window regains focus or its hourly timer runs. Failed background
checks stay quiet and retry no more than once an hour. Clicking the version or
**Check for updates** requests a fresh answer.

The last successful result and check times are cached in `update-state.json`
in the app's user-data directory. The cache contains version information and
text notes, never credentials, executable paths or install authorization.
After a restart, choosing to download first refreshes the release metadata.
`electron-updater` can reuse its cached archive after validating the digest.

Development checkouts and smoke-test runs do not check for or install releases.
No GitHub account or token is needed by app users: the update feed is public.

## Release contract and trust

The app uses `electron-updater` 6, with its GitHub provider pinned by the packaged
`app-update.yml` to `heresalexandria/seamstress`. Stable releases publish:

- A Developer ID signed, notarized and stapled `.app` inside the installer DMG.
- A ZIP containing that same app for Squirrel.Mac updates.
- `latest-mac.yml`, containing architecture-specific ZIP names and SHA-512 hashes.

CI must publish the updater metadata together with the matching ZIPs. The
[release guide](RELEASES.md) describes the release workflow and its signing
requirements. macOS updates require both a signed app and a ZIP update target,
as described in the [official electron-builder documentation](https://www.electron.build/v26/docs/features/auto-update/).

The library verifies the downloaded ZIP's digest. With automatic installation
disabled, native signature verification happens only after **Restart & install
update** is chosen. Seamstress waits for the native verification result before
allowing the final restart, and checks for active jobs again at that moment.
New processing jobs are blocked while installation is being prepared. A failed
verification leaves the running app open with an actionable error; it cannot
leave an unconditional deferred restart behind.

Release notes are bounded plain text rendered through React, never HTML. The
renderer cannot choose a feed, download URL, filesystem destination, signing
identity or shell command. **View releases** opens only the fixed repository
releases page. Error messages shown in the UI do not expose signed URLs or
local paths from lower-level failures.

## Validation

Run the unit tests with:

```sh
cd app
npm test
```

Updater tests use injected event emitters. They cover concurrent requests,
background throttling, stale caches, failed digest/signature checks, inert
release notes, native staging, explicit installation and the active-job guard.
They never create a release or alter an installed application.

A full macOS update test also requires two published signed releases:

1. Install the older release in Applications and open an existing project.
2. Check for updates, download the newer release and start a preview or export.
3. Verify that restarting to update is unavailable until the job finishes or is
   cancelled.
4. Restart to install; confirm the title-bar version changed and the same project
   can be reopened with its markers and exported files intact.
5. Verify the installed bundle with `codesign --verify --deep --strict` and
   `spctl --assess --type execute`.

Unit tests and a successful signed build do not by themselves prove that
installed-to-installed update path. Record that check separately when it has
been exercised with actual releases.
