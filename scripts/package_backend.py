"""Build a relocatable macOS worker and FFmpeg dependency closure.

Uses the active Python environment's PyInstaller and installed ffmpeg/ffprobe.
The output is local/ad-hoc signed; Developer ID signing/notarization, when
needed for distribution, belongs to the enclosing Electron release step.
"""
from __future__ import annotations

import argparse
import fcntl
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT/'app/resources/backend'
SYSTEM_PREFIXES = ('/usr/lib/', '/System/Library/', '/Library/Apple/System/Library/')
MAGIC = {b'\xfe\xed\xfa\xce', b'\xce\xfa\xed\xfe', b'\xfe\xed\xfa\xcf', b'\xcf\xfa\xed\xfe',
         b'\xca\xfe\xba\xbe', b'\xbe\xba\xfe\xca', b'\xca\xfe\xba\xbf', b'\xbf\xba\xfe\xca'}
EXCLUDES = ['torch', 'torchvision', 'torchaudio', 'skimage', 'matplotlib', 'IPython',
            'pytest', 'pandas', 'tkinter', 'seamstress.cli', 'seamstress.bridge',
            'seamstress.camera_bridge', 'seamstress.model_setup', 'seamstress._vendor',
            'seamstress.retime', 'seamstress.texture']


def run(args, *, env=None, input=None, quiet=True, cwd=None):
    result = subprocess.run(list(map(str, args)), input=input, text=True,
                            stdout=subprocess.PIPE if quiet else None,
                            stderr=subprocess.PIPE if quiet else None, env=env, cwd=cwd)
    if result.returncode:
        raise RuntimeError(f'Command failed ({result.returncode}): {args[0]}\n{result.stderr or ""}\n{result.stdout or ""}')
    return result.stdout or ''


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(4*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def macho(path):
    if not path.is_file() or path.is_symlink():
        return False
    with path.open('rb') as handle:
        return handle.read(4) in MAGIC


def dependencies(path):
    return [line.strip().split(' (compatibility version', 1)[0]
            for line in run(['/usr/bin/otool', '-L', path]).splitlines()[1:]
            if ' (compatibility version' in line]


def rpaths(path):
    text = run(['/usr/bin/otool', '-l', path])
    return re.findall(r'cmd LC_RPATH\s+cmdsize \d+\s+path (.*?) \(offset', text)


def system_dependency(reference):
    return reference.startswith(SYSTEM_PREFIXES)


def expand_reference(reference, loader, executable):
    return reference.replace('@loader_path', str(loader.parent)).replace('@executable_path', str(executable.parent))


def resolve_reference(reference, loader, executable):
    if system_dependency(reference):
        return None
    if reference.startswith('@rpath/'):
        suffix = reference[len('@rpath/'):]
        for candidate in rpaths(loader)+rpaths(executable):
            location = Path(expand_reference(candidate, loader, executable))/suffix
            if location.is_file():
                return location.resolve()
        raise RuntimeError(f'Unresolved Mach-O dependency {reference} of {loader}')
    candidate = Path(expand_reference(reference, loader, executable))
    if not candidate.is_absolute():
        candidate = loader.parent/candidate
    if not candidate.is_file():
        raise RuntimeError(f'Missing Mach-O dependency {reference} of {loader}')
    return candidate.resolve()


def ffmpeg_closure(programs):
    libraries, links, visited = {}, {}, set()
    pending = [(path, path) for path in programs.values()]
    while pending:
        source, executable = pending.pop()
        if source in visited:
            continue
        visited.add(source); links[source] = []
        for reference in dependencies(source):
            resolved = resolve_reference(reference, source, executable)
            if resolved is None or resolved == source:
                continue
            links[source].append((reference, resolved))
            libraries[resolved] = hashlib.sha256(str(resolved).encode()).hexdigest()[:10]+'-'+resolved.name
            pending.append((resolved, executable))
    return libraries, links


def version_record():
    result = {'python': sys.version, 'machine': platform.machine(), 'platform': platform.platform(), 'packages': {}}
    for name in ('pyinstaller', 'pyinstaller-hooks-contrib', 'numpy', 'scipy', 'opencv-python-headless', 'pillow'):
        try:
            result['packages'][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return result


def source_files():
    return sorted([*ROOT.joinpath('seamstress').rglob('*.py'), ROOT/'pyproject.toml',
                   ROOT/'scripts/package_backend.py', ROOT/'scripts/worker_entry.py'])


def input_record(programs, libraries):
    return {'sources': {str(p.relative_to(ROOT)): sha(p) for p in source_files()},
            'environment': version_record(),
            'ffmpeg_inputs': {str(p): sha(p) for p in sorted(set(programs.values())|set(libraries))},
            'excludes': EXCLUDES}


def input_fingerprint(record):
    return hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()


def collect_licenses(destination, paths):
    folder = destination/'licenses'; folder.mkdir()
    records, seen = [], set()
    def copy(path, package):
        path = Path(path)
        if not path.is_file() or path.resolve() in seen:
            return
        seen.add(path.resolve()); target = folder/package/(hashlib.sha256(str(path).encode()).hexdigest()[:8]+'-'+path.name)
        target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(path, target)
        records.append({'component': package, 'source': str(path), 'file': str(target.relative_to(destination))})
    formula_roots = set()
    for path in paths:
        parts = path.parts
        if 'Cellar' in parts:
            i = parts.index('Cellar')
            formula_roots.add(Path(*parts[:i+3]))
    for root in sorted(formula_roots):
        package = root.parent.name+'-'+root.name
        for base in (root, root/'share/doc', root/'share/licenses'):
            if base.is_dir():
                candidates = base.iterdir() if base == root else base.rglob('*')
                for file in candidates:
                    if file.is_file() and any(s in file.name.upper() for s in ('COPYING', 'LICENSE', 'LICENCE', 'NOTICE', 'COPYRIGHT')):
                        copy(file, package)
        receipt = root/'INSTALL_RECEIPT.json'
        if receipt.is_file():
            copy(receipt, package)
    for name in ('numpy', 'scipy', 'opencv-python-headless', 'pillow', 'pyinstaller', 'pyinstaller-hooks-contrib'):
        try:
            distribution = importlib.metadata.distribution(name)
        except importlib.metadata.PackageNotFoundError:
            continue
        for relative in distribution.files or []:
            if any(s in Path(relative).name.upper() for s in ('COPYING', 'LICENSE', 'LICENCE', 'NOTICE')):
                copy(distribution.locate_file(relative), name)
    for candidate in (Path(sys.base_prefix)/'LICENSE.txt', Path(sys.base_prefix)/'lib'/f'python{sys.version_info.major}.{sys.version_info.minor}'/'LICENSE.txt'):
        copy(candidate, 'Python')
    text = '# Bundled third-party components\n\nThis local bundle includes the worker runtime and installed FFmpeg binaries with their non-system dynamic dependencies. Available license/notice texts and package receipts are copied below. The manifest records exact binary hashes and build configuration; these records are not a substitute for the applicable license terms.\n\n'
    for record in records:
        text += f"- {record['component']}: [{Path(record['file']).name}]({record['file']})\n"
    (destination/'THIRD_PARTY_LICENSES.md').write_text(text)
    return records


def bundle_ffmpeg(destination, programs, libraries, links):
    (destination/'bin').mkdir(); (destination/'lib').mkdir()
    targets = {source: destination/'bin'/name for name, source in programs.items()}
    targets.update({source: destination/'lib'/name for source, name in libraries.items()})
    for source, target in targets.items():
        shutil.copy2(source, target); target.chmod(0o755)
    for source, target in targets.items():
        args = ['/usr/bin/install_name_tool']
        if source in libraries:
            args += ['-id', '@loader_path/'+target.name]
        for reference, dependency in links[source]:
            relative = os.path.relpath(targets[dependency], target.parent)
            args += ['-change', reference, '@loader_path/'+relative]
        for path in rpaths(source):
            args += ['-delete_rpath', path]
        if len(args) > 1:
            run([*args, target])
    def sign(target):
        run(['/usr/bin/codesign', '--force', '--sign', '-', target])
        run(['/usr/bin/codesign', '--verify', '--strict', target])
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(sign, targets.values()))
    return targets


def audit_standalone(destination):
    binaries = [p for p in destination.rglob('*') if macho(p)]
    external = []
    for binary in binaries:
        for reference in dependencies(binary):
            if reference.startswith('/') and not system_dependency(reference):
                location = Path(reference).resolve()
                if not location.is_relative_to(destination.resolve()):
                    external.append({'binary': str(binary.relative_to(destination)), 'dependency': reference})
        for reference in rpaths(binary):
            if reference.startswith('/') and not system_dependency(reference):
                external.append({'binary': str(binary.relative_to(destination)), 'rpath': reference})
    if external:
        raise RuntimeError('External non-system Mach-O paths remain: '+json.dumps(external[:12]))
    return len(binaries)


def smoke(destination, project=None):
    worker = destination/'seamstress-worker/seamstress-worker'
    # Exclude Homebrew/Python from PATH and launch from a foreign directory.
    env = {**os.environ, 'PATH': '/usr/bin:/bin:/usr/sbin:/sbin', 'OPENBLAS_NUM_THREADS': '2', 'OMP_NUM_THREADS': '2'}
    for key in ('PYTHONPATH', 'PYTHONHOME', 'DYLD_LIBRARY_PATH', 'DYLD_FALLBACK_LIBRARY_PATH'):
        env.pop(key, None)
    with tempfile.TemporaryDirectory(prefix='seamstress-standalone-smoke-') as foreign:
        output = run([worker, '--self-test'], env=env, cwd=foreign)
    check = json.loads(output.strip().splitlines()[-1])
    if check.get('self_test') != 'ok':
        raise RuntimeError('Packaged worker self-test failed')
    if project:
        request = json.dumps({'operation': 'get', 'args': {'projectPath': str(project)}})+'\n'
        with tempfile.TemporaryDirectory(prefix='seamstress-worker-project-') as foreign:
            events = run([worker], env=env, input=request, cwd=foreign)
        last = json.loads(events.strip().splitlines()[-1])
        if last.get('type') != 'complete':
            raise RuntimeError('Bundled worker failed to open the validation project')
        check['project_opened'] = str(project)
    return check


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--check-project', type=Path)
    args = parser.parse_args()
    if sys.platform != 'darwin':
        raise SystemExit('This packaging script currently targets native macOS builds only')
    build_root = ROOT/'build/backend'; build_root.mkdir(parents=True, exist_ok=True)
    lock = (build_root/'package.lock').open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit('Another backend build is already running')
    output = args.output.expanduser().resolve()
    if ROOT.is_relative_to(output) or output in (ROOT/'app', ROOT/'seamstress', ROOT/'scripts'):
        raise SystemExit('Refusing to replace a source directory with a build output')
    if output.exists() and any(output.iterdir()) and not (output/'build-manifest.json').is_file():
        raise SystemExit('Refusing to replace a nonempty directory that is not a generated backend')
    programs = {}
    for name in ('ffmpeg', 'ffprobe'):
        location = shutil.which(name)
        if not location:
            raise SystemExit(f'{name} must be installed to build the desktop bundle')
        programs[name] = Path(location).resolve()
    libraries, links = ffmpeg_closure(programs)
    record = input_record(programs, libraries); fingerprint = input_fingerprint(record)
    manifest_path = output/'build-manifest.json'
    project = args.check_project.expanduser().resolve() if args.check_project else None
    default_project = ROOT/'output/seamstress-app-validation.seamstress/project.json'
    if project is None and default_project.is_file():
        project = default_project
    if project is not None and not project.is_file():
        raise SystemExit(f'Validation project does not exist: {project}')
    if not args.force and manifest_path.is_file():
        prior = json.loads(manifest_path.read_text())
        if prior.get('input_fingerprint') == fingerprint and (output/'seamstress-worker/seamstress-worker').is_file():
            check = smoke(output, project)
            print(json.dumps({'backend': str(output), 'cached': True, 'self_test': check})); return
    output.parent.mkdir(parents=True, exist_ok=True)
    build_root = ROOT/'build/backend'; build_root.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, 'PYINSTALLER_CONFIG_DIR': str(build_root/'pyinstaller-cache'), 'PYTHONDONTWRITEBYTECODE': '1'}
    with tempfile.TemporaryDirectory(prefix='.backend-build-', dir=output.parent) as temporary:
        temp = Path(temporary); snapshot = temp/'source'; snapshot.mkdir()
        # Freeze a consistent per-file snapshot. Subsequent source edits change
        # the next invocation's fingerprint and cannot falsely reuse this build.
        snapshot_hashes = {}
        for relative in record['sources']:
            target = snapshot/relative; target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT/relative, target); snapshot_hashes[relative] = sha(target)
        record['sources'] = snapshot_hashes; fingerprint = input_fingerprint(record)
        stage = temp/'backend'; stage.mkdir()
        command = [sys.executable, '-m', 'PyInstaller', '--noconfirm', '--clean', '--onedir',
                   '--name', 'seamstress-worker', '--distpath', str(stage), '--workpath', str(build_root/'work'),
                   '--specpath', str(build_root), '--paths', str(snapshot), '--noupx',
                   '--target-arch', platform.machine()]
        for module in EXCLUDES:
            command += ['--exclude-module', module]
        command += [str(snapshot/'scripts/worker_entry.py')]
        print('Building standalone Python worker…', flush=True)
        log = build_root/'pyinstaller.log'
        with log.open('w') as handle:
            result = subprocess.run(command, cwd=snapshot, env=env, stdout=handle, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f'PyInstaller failed; inspect {log}\n'+log.read_text()[-9000:])
        print(f'Bundling FFmpeg and {len(libraries)} dynamic libraries…', flush=True)
        bundle_ffmpeg(stage, programs, libraries, links)
        license_records = collect_licenses(stage, list(programs.values())+list(libraries))
        binary_count = audit_standalone(stage)
        check = smoke(stage, project)
        manifest = {'schema_version': 1, 'generated_at': datetime.now(timezone.utc).isoformat(),
                    'input_fingerprint': fingerprint, 'inputs': record, 'architecture': platform.machine(),
                    'signing': 'ad-hoc local signatures; not Developer ID notarized',
                    'macho_binary_count': binary_count, 'ffmpeg_dependency_count': len(libraries),
                    'ffmpeg_configuration': run([programs['ffmpeg'], '-buildconf']),
                    'licenses': license_records, 'self_test': check}
        (stage/'build-manifest.json').write_text(json.dumps(manifest, indent=2))
        backup = output.with_name(output.name+'.previous')
        if backup.exists():
            if not (backup/'build-manifest.json').is_file():
                raise RuntimeError('Refusing to remove an unrecognized previous-build directory')
            shutil.rmtree(backup)
        if output.exists():
            output.rename(backup)
        try:
            stage.rename(output)
            relocated_check = smoke(output, project)
            manifest['self_test'] = relocated_check
            (output/'build-manifest.json').write_text(json.dumps(manifest, indent=2))
            check = relocated_check
        except BaseException:
            if output.exists():
                shutil.rmtree(output)
            if backup.exists():
                backup.rename(output)
            raise
        if backup.exists():
            shutil.rmtree(backup)
    current = input_record(programs, libraries)
    changed = input_fingerprint(current) != fingerprint
    print(json.dumps({'backend': str(output), 'cached': False, 'source_changed_during_build': changed,
                      'macho_binary_count': binary_count, 'ffmpeg_dependency_count': len(libraries),
                      'self_test': check}))
    if changed:
        print('Source files changed during packaging; rerun before packaging the final app.', file=sys.stderr)
        raise SystemExit(2)


if __name__ == '__main__':
    main()
