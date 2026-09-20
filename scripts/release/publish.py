"""Publish a verified build atomically from a draft, without replacing public assets."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess

from bump_version import parse

REPOSITORY = 'heresalexandria/seamstress'


def gh(*args: str, missing_ok: bool = False):
    result = subprocess.run(['gh', *args], capture_output=True, text=True)
    if result.returncode:
        if missing_ok and ('HTTP 404' in result.stderr or 'release not found' in result.stderr.lower()):
            return None
        raise RuntimeError(f'GitHub command failed: {result.stderr.strip()}')
    return result.stdout


def publish(version: str, sha: str, files: Path, notes: Path) -> str:
    parse(version)
    if not re.fullmatch(r'[0-9a-f]{40}', sha):
        raise ValueError('Release target must be an exact 40-character commit SHA')
    if os.environ.get('GH_REPO', REPOSITORY) != REPOSITORY:
        raise ValueError('This publisher is restricted to heresalexandria/seamstress')
    paths = sorted(path for path in files.iterdir() if path.is_file())
    required = {'latest-mac.yml', 'SHA256SUMS.txt'} | {
        f'Seamstress-{version}-{arch}.{suffix}' for arch in ('arm64', 'x64') for suffix in ('zip', 'dmg')
    } | {f'Seamstress-mac-{arch}.dmg' for arch in ('arm64', 'x64')}
    if not required.issubset({path.name for path in paths}) or any(path.is_symlink() for path in paths):
        raise ValueError('Only a complete validated release directory can be published')
    tag = f'v{version}'
    prior = gh('api', f'repos/{REPOSITORY}/releases/tags/{tag}', missing_ok=True)
    if prior:
        release = json.loads(prior)
        if not release.get('draft'):
            raise ValueError(f'{tag} is already public. Published versions are immutable; increment the version.')
        if release.get('target_commitish') != sha:
            raise ValueError('The existing draft targets a different commit; resolve that draft before retrying.')
    reference = gh('api', f'repos/{REPOSITORY}/git/ref/tags/{tag}', missing_ok=True)
    if reference:
        target = json.loads(reference)['object']
        for _ in range(8):
            if target.get('type') != 'tag':
                break
            target = json.loads(gh('api', f'repos/{REPOSITORY}/git/tags/{target["sha"]}'))['object']
        if target.get('type') != 'commit' or target.get('sha') != sha:
            raise ValueError('The existing version tag targets a different commit; refusing to reuse or move it.')
    if not prior:
        gh('release', 'create', tag, '--repo', REPOSITORY, '--target', sha,
           '--title', tag, '--notes-file', str(notes), '--draft')
    # Idempotent recovery may replace assets only while this release is a draft.
    gh('release', 'upload', tag, '--repo', REPOSITORY, '--clobber', *map(str, paths))
    release = json.loads(gh('api', f'repos/{REPOSITORY}/releases/tags/{tag}'))
    if not release.get('draft') or release.get('target_commitish') != sha:
        raise ValueError('Release state changed during upload; refusing publication')
    assets = {item['name']: item['size'] for item in release.get('assets', [])}
    expected = {path.name: path.stat().st_size for path in paths}
    if assets != expected:
        raise ValueError('Uploaded release assets do not match the complete local set')
    gh('release', 'edit', tag, '--repo', REPOSITORY, '--notes-file', str(notes), '--draft=false', '--latest')
    url = f'https://github.com/{REPOSITORY}/releases/tag/{tag}'
    if os.environ.get('GITHUB_STEP_SUMMARY'):
        with open(os.environ['GITHUB_STEP_SUMMARY'], 'a') as summary:
            summary.write(f'Published [{tag}]({url}) with signed, notarized Apple silicon and Intel installers.\n')
    return url


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--version', required=True)
    parser.add_argument('--sha', required=True)
    parser.add_argument('--files', type=Path, required=True)
    parser.add_argument('--notes', type=Path, required=True)
    args = parser.parse_args()
    print(publish(args.version, args.sha, args.files, args.notes))
