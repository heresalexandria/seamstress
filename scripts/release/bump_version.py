"""Keep the Python package, Electron app, and npm lockfile on one version."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
VERSION = re.compile(r'(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)\Z')


def parse(value: str) -> tuple[int, int, int]:
    match = VERSION.fullmatch(value)
    if not match:
        raise ValueError(f'Expected a stable MAJOR.MINOR.PATCH version, received {value!r}')
    return tuple(map(int, match.groups()))


def bump(value: str, kind: str) -> str:
    parts = list(parse(value))
    if kind not in ('major', 'minor', 'patch', 'current'):
        raise ValueError(f'Unknown version change: {kind}')
    if kind != 'current':
        position = ('major', 'minor', 'patch').index(kind)
        parts[position] += 1
        parts[position + 1:] = [0] * (2 - position)
    return '.'.join(map(str, parts))


def check(root: Path = ROOT) -> str:
    pyproject = re.search(r'^version\s*=\s*"([^"]+)"', (root/'pyproject.toml').read_text(), re.M)
    engine = re.search(r'^__version__\s*=\s*"([^"]+)"', (root/'seamstress/__init__.py').read_text(), re.M)
    package = json.loads((root/'app/package.json').read_text())
    lock = json.loads((root/'app/package-lock.json').read_text())
    values = [pyproject[1] if pyproject else None, engine[1] if engine else None,
              package.get('version'), lock.get('version'), lock.get('packages', {}).get('', {}).get('version')]
    if None in values or len(set(values)) != 1:
        raise ValueError(f'Version files disagree: {values}')
    parse(values[0])
    return values[0]


def write(version: str, root: Path = ROOT) -> None:
    parse(version)
    check(root)
    updates = {}
    for name, pattern, replacement in [
        ('pyproject.toml', r'^version\s*=\s*"[^"]+"', f'version = "{version}"'),
        ('seamstress/__init__.py', r'^__version__\s*=\s*"[^"]+"', f'__version__ = "{version}"'),
        ('app/package.json', r'^(\s*)"version":\s*"[^"]+"', rf'\g<1>"version": "{version}"'),
    ]:
        path = root/name
        replacement_text, count = re.subn(pattern, replacement, path.read_text(), count=1, flags=re.M)
        if count != 1:
            raise ValueError(f'Expected one version in {name}')
        updates[path] = replacement_text
    lock_path = root/'app/package-lock.json'
    lock = json.loads(lock_path.read_text())
    lock['version'] = lock['packages']['']['version'] = version
    updates[lock_path] = json.dumps(lock, indent=2) + '\n'
    for path, content in updates.items():
        path.write_text(content)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('kind', nargs='?', choices=['major', 'minor', 'patch', 'current'])
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    version = check()
    if not args.check:
        if not args.kind:
            parser.error('Choose major, minor, patch, current, or --check.')
        version = bump(version, args.kind)
        if not args.dry_run:
            write(version)
    print(version)
    if os.environ.get('GITHUB_OUTPUT'):
        with open(os.environ['GITHUB_OUTPUT'], 'a') as output:
            output.write(f'version={version}\ntag=v{version}\n')


if __name__ == '__main__':
    main()
