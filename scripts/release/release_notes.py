"""Use user-visible PR changes as the release notes, followed by actual downloads."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re


def build(version: str, files: Path, body: str = '', previous: str = '') -> str:
    repo = 'heresalexandria/seamstress'
    body = re.sub(r'<!--.*?-->', '', body, flags=re.S).strip()
    # PRs open with the release bullets; reviewer details follow the rule.
    summary = re.split(r'^\s*---\s*$', body, maxsplit=1, flags=re.M)[0].strip()
    parts = [summary] if summary else [f'Seamstress {version}']
    rows = []
    for arch, label in [('arm64', 'Apple silicon'), ('x64', 'Intel')]:
        name = f'Seamstress-{version}-{arch}.dmg'
        path = files/name
        if not path.is_file():
            raise ValueError(f'Release notes cannot advertise a missing installer: {name}')
        url = f'https://github.com/{repo}/releases/download/v{version}/{name}'
        rows.append(f'| macOS, {label} | [Download installer]({url}) | {path.stat().st_size / 1024**2:.0f} MB |')
    parts.append('## Download\n\n| Platform | Installer | Size |\n|---|---|---|\n' + '\n'.join(rows))
    parts.append('The app is Developer ID signed and notarized by Apple. Open the disk image and drag Seamstress to Applications.\n\n'
                 'Existing installations can use **Check for updates** inside Seamstress. ZIP files and '
                 '`latest-mac.yml` are used by the updater; `SHA256SUMS.txt` records all published files.')
    if previous:
        parts.append(f'[Full changelog](https://github.com/{repo}/compare/{previous}...v{version})')
    return '\n\n'.join(parts) + '\n'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--version', required=True)
    parser.add_argument('--files', type=Path, required=True)
    parser.add_argument('--pr', type=Path)
    parser.add_argument('--previous', default='')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    body = json.loads(args.pr.read_text()).get('body') or '' if args.pr else ''
    args.output.write_text(build(args.version, args.files, body, args.previous))
