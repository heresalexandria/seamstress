"""Validate both native builds and assemble one complete, immutable release."""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path
import shutil

import yaml

TARGETS = ('arm64', 'x64')


def digest(path: Path, algorithm: str) -> bytes:
    value = hashlib.new(algorithm)
    with path.open('rb') as file:
        for block in iter(lambda: file.read(4 * 1024 * 1024), b''):
            value.update(block)
    return value.digest()


def assemble(incoming: Path, output: Path, version: str) -> dict:
    from bump_version import parse
    parse(version)
    if output.exists() and any(output.iterdir()):
        raise ValueError('Release output must be empty; stale artifacts must never be published.')
    files, reports, sources = [], [], {}
    for arch in TARGETS:
        folder = incoming/f'build-mac-{arch}'
        info = yaml.safe_load((folder/'latest-mac.yml').read_text())
        if not isinstance(info, dict) or str(info.get('version')) != version:
            raise ValueError(f'{arch} update metadata has the wrong version')
        report = json.loads((folder/f'verification-{arch}.json').read_text())
        if (report.get('version') != version or report.get('arch') != arch
                or report.get('signed') is not True or report.get('notarized') is not True
                or not report.get('team_id')):
            raise ValueError(f'{arch} has no successful signing and notarization verification')
        reports.append(report)
        expected = {f'Seamstress-{version}-{arch}.zip'}
        installer = folder/f'Seamstress-{version}-{arch}.dmg'
        if installer.is_symlink() or not installer.is_file() or installer.stat().st_size == 0:
            raise ValueError(f'Missing finalized installer: {installer.name}')
        sources[installer.name] = installer
        seen = set()
        for entry in info.get('files', []):
            name = entry.get('url')
            if name not in expected or name in seen:
                raise ValueError(f'Unexpected or duplicated update asset: {name!r}')
            path = folder/name
            if path.is_symlink() or not path.is_file() or path.stat().st_size == 0:
                raise ValueError(f'Missing regular update asset: {name}')
            sha512 = base64.b64encode(digest(path, 'sha512')).decode()
            if entry.get('sha512') != sha512 or entry.get('size') != path.stat().st_size:
                raise ValueError(f'Update asset checksum or size does not match metadata: {name}')
            files.append(dict(entry))
            sources[name] = path
            seen.add(name)
        if seen != expected:
            raise ValueError(f'{arch} is missing its ZIP from update metadata')
        for path in folder.glob('*.blockmap'):
            if path.name.endswith('.dmg.blockmap'):
                # DMGs are stapled after packaging. Only ZIP updater blockmaps
                # remain valid; installers receive final SHA-256 checksums.
                continue
            if path.is_symlink() or path.name.removesuffix('.blockmap') not in expected:
                raise ValueError(f'Unexpected blockmap: {path.name}')
            sources[path.name] = path
        sources[f'verification-{arch}.json'] = folder/f'verification-{arch}.json'
    if len({report['team_id'] for report in reports}) != 1:
        raise ValueError('Both architectures must be signed by the same Apple developer team')
    # x64 first preserves the legacy top-level path convention. Modern clients
    # choose their architecture from the complete files list.
    files.sort(key=lambda item: ('-arm64.' in item['url'], not item['url'].endswith('.zip'), item['url']))
    primary = files[0]
    merged = {'version': version, 'files': files,
              'path': primary['url'], 'sha512': primary['sha512']}
    output.mkdir(parents=True, exist_ok=True)
    for name, path in sources.items():
        shutil.copy2(path, output/name)
    for arch in TARGETS:
        shutil.copy2(output/f'Seamstress-{version}-{arch}.dmg', output/f'Seamstress-mac-{arch}.dmg')
    (output/'latest-mac.yml').write_text(yaml.safe_dump(merged, sort_keys=False))
    checksums = [f'{digest(path, "sha256").hex()}  {path.name}'
                 for path in sorted(output.iterdir()) if path.is_file()]
    (output/'SHA256SUMS.txt').write_text('\n'.join(checksums) + '\n')
    return merged


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--incoming', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--version', required=True)
    args = parser.parse_args()
    assemble(args.incoming, args.output, args.version)
    print(f'Validated both signed architectures and assembled {args.output}')
