# Releases and updates

Seamstress uses the same label-driven release convention as Aesthetician. Open a
pull request, apply one release label, and merge it into `main`. The release
workflow updates the app version, builds both native macOS architectures, signs
and notarizes them, tests the packaged applications, and publishes a GitHub
release. The app discovers new versions from that release.

| Exactly one PR label | Starting at 0.2.0 | Purpose |
|---|---|---|
| `major` | 1.0.0 | Breaking change |
| `minor` | 0.3.0 | New feature |
| `patch` | 0.2.1 | Fix or small change |
| `no-release` | 0.2.0 | Documentation or work that should not ship |

The `release label` check enforces this rule. Run **Create release labels** once
from Actions after adding the workflows; it is safe to run again.

## Download links that stay current

Use these links in the README or a website:

- [Apple silicon installer](https://github.com/heresalexandria/seamstress/releases/latest/download/Seamstress-mac-arm64.dmg)
- [Intel installer](https://github.com/heresalexandria/seamstress/releases/latest/download/Seamstress-mac-x64.dmg)
- [Latest release and notes](https://github.com/heresalexandria/seamstress/releases/latest)

Every release includes versioned DMGs and updater ZIPs, the two stable installer
aliases, `latest-mac.yml`, signing verification reports, and `SHA256SUMS.txt`.
Only the ZIPs appear in updater metadata. DMGs are notarized and stapled after
packaging, so their final checksums are computed afterward. The metadata merger
validates every ZIP's size and SHA-512 before it combines both architectures.

Both targets are required: Apple silicon uses `macos-15`, Intel uses
`macos-15-intel`. Windows and Linux packages are not currently supported.

## Signing setup

Create these **repository Actions secrets** in `heresalexandria/seamstress`:

| Secret | Value |
|---|---|
| `CSC_LINK` | Base64-encoded `.p12` containing a Developer ID Application certificate and its private key |
| `CSC_KEY_PASSWORD` | The nonempty password used to encrypt that `.p12` |
| `APPLE_API_KEY` | The complete contents of the App Store Connect API `.p8` private key |
| `APPLE_API_KEY_ID` | That API key's identifier |
| `APPLE_API_ISSUER` | The API key's issuer UUID |
| `APPLE_TEAM_ID` | The Apple developer team that owns the signing certificate |

The certificate must be a **Developer ID Application** identity, not an Apple
Development or Mac App Store identity. The API key needs access to Apple's
notarization service. The workflow writes the API key only to a private temporary
file, removes it after signing, and never includes it in artifacts. Certificate
private keys, passwords, `.p8` and `.p12` files do not belong in Git.

Release configuration is in `app/electron-builder.release.cjs`; signing and
verification hooks are in `scripts/signing/`. Release builds require hardened
runtime signing and notarization, including the embedded Python worker and
FFmpeg libraries. Verification checks the final application, the app in its
updater ZIP, and the stapled DMG. Missing credentials, signature failures, Apple
rejection, or an architecture's failed smoke test block publication. There is no
unsigned release fallback. Ordinary local and **Build check** packages remain
development builds and are labeled accordingly.

See [macOS signing](MACOS-SIGNING.md) for certificate setup, entitlement details,
and local signed-build commands, and [in-app updates](updates.md) for the app's
download and installation behavior.

## Repository permissions and identity

Local commits and GitHub API operations use the repo-local
`use-heresalexandria.sh` account guard. That helper and the machine's `AGENTS.md`
are intentionally ignored; copy them from the configured checkout when setting
up another machine. Do not switch or reuse the primary GitHub account.

Actions uses its repository-scoped `GITHUB_TOKEN`. The only automatic Git author
is `github-actions[bot]` for version commits. The preparation and publication jobs
request `contents: write`; tests and native builds receive read-only repository
access. Configure branch rules so Actions can push the four version files to
`main`, while requiring the CI checks for ordinary PRs. No private user token is
embedded in the workflows or shipped app.

Third-party Actions are pinned to reviewed commit SHAs rather than moving tags.

The release trigger uses `pull_request_target` only for **closed, merged** PRs
targeting `main`. It checks out the trusted `main` branch and then the exact
prepared commit; it never checks out an unmerged PR head with signing secrets.
Manual release dispatch must also select `main`.

## Release notes

Begin a PR description with a flat list of user-visible changes, one per line:

```markdown
- **New:** Adjust seam markers with exact timecodes.
- **Fixed:** Corrected previews now retain the source aspect ratio.

---

Reviewer context and validation details.
```

The text above the rule becomes the release notes. The release script adds
download links, installer sizes, and a compare link. Template comments and the
reviewer-only details do not appear in the update notes.

## Checks and build debugging

CI runs the Python engine tests, release invariant tests, Electron/updater tests,
renderer typecheck/build, and a real macOS Electron import-to-export smoke test.
Release builds repeat checks at the precise version commit and test the actual
packaged executable with its embedded runtime.

To exercise native packaging without publishing or exposing signing secrets,
run **Build check** from Actions and choose an architecture or `all`. Pushing a
`ci/**` branch also runs both architectures. Unsigned installers are attached to
the workflow run for three days; they are never attached to a public release.

Local release checks:

```sh
.venv/bin/python -m pip install PyYAML==6.0.3
.venv/bin/python scripts/release/bump_version.py --check
.venv/bin/python -m unittest discover -s scripts/release -p 'test_*.py' -v
```

## First release and recovery

Run **Release** from Actions on `main`. Choose `current` to publish the version
already in the four version files, or choose `patch`, `minor`, or `major` to
increment it. The workflow refuses to replace any version that is already
public. Normally only the workflow changes versions.

A failed build can leave a version commit on `main`, but no release tag is
created until both native builds and their verification checks succeed. Rerun
the failed jobs to retain the same commit, or dispatch `current` to build the
current unpublished version after a build fix. A later labeled PR can simply
increment past the failed version.

Publication creates a draft, uploads all assets, checks the complete asset list
and sizes, then publishes and marks the release Latest. If uploading fails, the
draft can be resumed for the same commit; public assets are never overwritten.
A draft for another commit must be explicitly resolved before reusing its
version. This prevents a partly uploaded release from becoming visible to app
update checks.

The in-app updater reads GitHub release metadata, chooses the matching native
ZIP, verifies its digest, and uses macOS's signed-app update mechanism. The
installer links keep pointing to the last completely published release while a
new build is still running or has failed.
