"""Recover publication from two proven signed Release builds, without rebuilding."""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import tempfile
import zipfile

import yaml

import bump_version

REPOSITORY = 'heresalexandria/seamstress'
WORKFLOW = '.github/workflows/release.yml'
CHECKOUT = 'actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1'
UPLOAD = 'actions/upload-artifact@043fb46d1a93c77aae656e7c1c64a875d1fc6a0a'
SHA = re.compile(r'[0-9a-f]{40}\Z')
STAMP = re.compile(r'^(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d(?:\.\d+)?Z) (.*)$')
REQUIRED_STEPS = (
    f'Run {CHECKOUT}',
    'Validate this exact release commit',
    'Build renderer and native backend',
    'Sign, notarize, staple, and verify the app and installers',
    'Test the actual signed package, including correction and export',
    'Stage only finalized release files',
    f'Run {UPLOAD}',
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def inputs(run_id: str, sha: str, version: str) -> None:
    require(re.fullmatch(r'[1-9][0-9]*', run_id) is not None, 'Run ID must be a positive integer')
    require(SHA.fullmatch(sha) is not None, 'Expected SHA must be 40 lowercase hexadecimal characters')
    bump_version.parse(version)


def api(endpoint: str, *, pages: bool = False):
    args = ['gh', 'api', f'repos/{REPOSITORY}/{endpoint}']
    if pages:
        args += ['--paginate', '--slurp']
    return json.loads(subprocess.check_output(args))


def git(*args: str) -> str:
    return subprocess.check_output(['git', *args], text=True).strip()


def ancestor(older: str, newer: str) -> None:
    require(SHA.fullmatch(older) is not None, 'Invalid ancestry commit')
    subprocess.run(['git', 'merge-base', '--is-ancestor', older, newer], check=True)


def validate_run(run: dict, workflow: dict, run_id: str) -> None:
    require(run.get('id') == int(run_id), 'Source run ID changed')
    require(run.get('repository', {}).get('full_name') == REPOSITORY, 'Source run belongs to another repository')
    require(workflow.get('path') == WORKFLOW and workflow.get('name') == 'Release', 'Wrong canonical workflow')
    require(run.get('workflow_id') == workflow.get('id') and isinstance(workflow.get('id'), int), 'Source run is not the canonical Release workflow')
    require(run.get('path') == WORKFLOW and run.get('name') == 'Release', 'Unexpected source workflow path or name')
    require(run.get('status') == 'completed', 'Source run must be completed')
    require(type(run.get('run_attempt')) is int and run['run_attempt'] > 0, 'Missing source attempt')
    require(SHA.fullmatch(run.get('head_sha', '')) is not None, 'Invalid source trigger SHA')
    require(run.get('event') in ('workflow_dispatch', 'pull_request_target'), 'Untrusted release event')
    if run['event'] == 'workflow_dispatch':
        require(run.get('head_branch') == 'main', 'Manual source release must run on main')


def merged_pr(run: dict, prs: list[dict]) -> dict:
    matches = [pr for pr in prs
               if pr.get('head', {}).get('sha') == run['head_sha']
               and pr.get('head', {}).get('ref') == run['head_branch']
               and pr.get('base', {}).get('ref') == 'main'
               and pr.get('base', {}).get('repo', {}).get('full_name') == REPOSITORY
               and pr.get('state') == 'closed' and pr.get('merged_at')]
    require(len(matches) == 1, 'Source event must identify exactly one merged PR targeting this repository/main')
    pr = matches[0]
    # A run started before the PR merged does not establish a closed/merged event.
    require(moment(pr['merged_at']) <= moment(run['created_at']), 'Source run predates the PR merge')
    require(type(pr.get('number')) is int and pr['number'] > 0, 'Invalid source PR number')
    return pr


def validate_source(sha: str, version: str, event: str) -> None:
    ancestor(sha, 'origin/main')
    # Read historical data without executing scripts from the recovered commit.
    with tempfile.TemporaryDirectory(prefix='release-version-') as temporary:
        root = Path(temporary)
        for name in ('pyproject.toml', 'seamstress/__init__.py', 'app/package.json', 'app/package-lock.json'):
            path = root/name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(git('show', f'{sha}:{name}'))
        require(bump_version.check(root) == version, 'Prepared source version does not match the expected version')
    source = yaml.safe_load(git('show', f'{sha}:{WORKFLOW}'))
    triggers = source.get('on', source.get(True, {}))
    jobs = source['jobs']
    require(source.get('name') == 'Release', 'Prepared source has an unexpected release workflow')
    require(jobs['prepare'].get('if') == "github.event_name == 'workflow_dispatch' || github.event.pull_request.merged == true", 'Source preparation does not guard merged PRs')
    prepare_checkout = [step for step in jobs['prepare']['steps'] if step.get('uses') == CHECKOUT]
    build_checkout = [step for step in jobs['build']['steps'] if step.get('uses') == CHECKOUT]
    require(len(prepare_checkout) == 1 and prepare_checkout[0].get('with', {}).get('ref') == 'main', 'Source preparation must check out main')
    require(len(build_checkout) == 1 and build_checkout[0].get('with', {}).get('ref') == '${{ needs.prepare.outputs.sha }}', 'Source native builds must check out the prepared commit')
    if event == 'pull_request_target':
        trigger = triggers.get(event, {})
        require(trigger.get('types') == ['closed'] and trigger.get('branches') == ['main'], 'Source workflow must only accept closed PRs targeting main')
    else:
        require('workflow_dispatch' in triggers, 'Source workflow does not support manual releases')


def successful_step(job: dict, name: str) -> dict:
    matches = [step for step in job.get('steps', []) if step.get('name') == name]
    require(len(matches) == 1 and matches[0].get('conclusion') == 'success', f'Missing successful step: {name}')
    return matches[0]


def validate_jobs(jobs: list[dict], run: dict) -> dict[str, dict]:
    result = {}
    for label in ('prepare release version', 'signed mac-arm64', 'signed mac-x64'):
        matches = [job for job in jobs if job.get('name') == label]
        require(len(matches) == 1, f'Expected exactly one source job: {label}')
        job = matches[0]
        require(job.get('run_id') == run['id'] and job.get('run_attempt') == run['run_attempt'], 'Source jobs span runs or attempts')
        require(job.get('status') == 'completed' and job.get('conclusion') == 'success', f'Source job did not pass: {label}')
        require(type(job.get('id')) is int and job['id'] > 0, 'Invalid source job ID')
        if label.startswith('signed mac-'):
            for name in REQUIRED_STEPS:
                successful_step(job, name)
            result[label.removeprefix('signed mac-')] = job
    return result


def moment(value: str) -> datetime:
    return datetime.fromisoformat(value.replace('Z', '+00:00'))


def during(value: str, step: dict) -> bool:
    # Job API times have only second precision; log/artifact times may be finer.
    return moment(step['started_at']) <= moment(value) < moment(step['completed_at']) + timedelta(seconds=1)


def step_lines(log: str, job: dict, name: str) -> list[str]:
    step = successful_step(job, name)
    lines = []
    for line in log.splitlines():
        match = STAMP.fullmatch(line)
        if match and during(match[1], step):
            lines.append(match[2])
    return lines


def checkout_sha(log: str, job: dict, expected: str) -> None:
    lines = step_lines(log, job, f'Run {CHECKOUT}')
    commits = [lines[index + 1] for index, line in enumerate(lines[:-1])
               if re.fullmatch(r'\[command\]/[^\s]+/git log -1 --format=%H', line)]
    require(commits == [expected], 'Native checkout log does not prove the exact expected source SHA')
    require(f'  ref: {expected}' in lines, 'Native checkout requested a different ref')
    require(any(re.fullmatch(r'\[command\]/[^\s]+/git checkout --progress --force ' + expected, line)
                for line in lines), 'Native checkout did not check out the exact expected source SHA')


def validate_artifacts(artifacts: list[dict], run: dict, jobs: dict[str, dict], logs: dict[str, str]) -> dict[str, dict]:
    selected = {}
    for arch, job in jobs.items():
        matches = [item for item in artifacts if item.get('name') == f'build-mac-{arch}']
        require(len(matches) == 1, f'Expected one immutable build-mac-{arch} artifact')
        item = matches[0]
        require(item.get('expired') is False, 'Source artifact expired')
        require(type(item.get('id')) is int and item['id'] > 0 and item.get('size_in_bytes', 0) > 0, 'Invalid source artifact')
        require(re.fullmatch(r'sha256:[0-9a-f]{64}', item.get('digest', '')) is not None, 'Artifact has no verifiable SHA-256 digest')
        require(item.get('workflow_run', {}).get('id') == run['id'] and item['workflow_run'].get('head_sha') == run['head_sha'], 'Artifact belongs to a different source run')
        # API timestamps and runner clocks can differ by over a minute. Bind the
        # immutable ID, bytes and digest to the upload action's own output instead.
        lines = step_lines(logs[arch], job, f'Run {UPLOAD}')
        uploaded = [match.groups() for line in lines if (match := re.fullmatch(
            rf'Artifact build-mac-{arch} has been successfully uploaded! Final size is ([0-9]+) bytes\. Artifact ID is ([0-9]+)', line))]
        digests = [match[1] for line in lines if (match := re.fullmatch(r'SHA256 digest of uploaded artifact is ([0-9a-f]{64})', line))]
        require(uploaded == [(str(item['size_in_bytes']), str(item['id']))] and digests == [item['digest'][7:]], 'Artifact ID, size or digest does not match the successful native upload log')
        selected[arch] = item
    return selected


def extract_verified(archive: Path, item: dict, destination: Path, version: str, arch: str) -> None:
    with archive.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    require(f'sha256:{digest}' == item['digest'], 'Downloaded artifact SHA-256 does not match GitHub provenance')
    require(archive.stat().st_size == item['size_in_bytes'], 'Downloaded artifact size changed')
    required = {f'Seamstress-{version}-{arch}.dmg', f'Seamstress-{version}-{arch}.zip', 'latest-mac.yml', f'verification-{arch}.json'}
    allowed = required | {f'Seamstress-{version}-{arch}.zip.blockmap'}
    with zipfile.ZipFile(archive) as source:
        entries = source.infolist()
        names = {entry.filename for entry in entries}
        require(len(names) == len(entries) and required <= names <= allowed, 'Artifact contains missing, duplicate, or unexpected paths')
        for entry in entries:
            mode = entry.external_attr >> 16
            require(not entry.is_dir() and (stat.S_IFMT(mode) in (0, stat.S_IFREG)), 'Artifact entries must be regular files')
        destination.mkdir()
        for entry in entries:
            with source.open(entry) as incoming, (destination/entry.filename).open('wb') as outgoing:
                shutil.copyfileobj(incoming, outgoing)


def recover(run_id: str, sha: str, version: str, incoming: Path, pr_output: Path) -> dict:
    inputs(run_id, sha, version)
    require(not incoming.exists(), 'Recovery download directory must not exist')
    run = api(f'actions/runs/{run_id}')
    validate_run(run, api('actions/workflows/release.yml'), run_id)
    validate_source(sha, version, run['event'])
    pr = None
    if run['event'] == 'pull_request_target':
        prs = [pr for page in api(f"commits/{run['head_sha']}/pulls?per_page=100", pages=True) for pr in page]
        pr = merged_pr(run, prs)
        ancestor(pr['merge_commit_sha'], sha)
    else:
        ancestor(run['head_sha'], sha)
    pages = api(f"actions/runs/{run_id}/attempts/{run['run_attempt']}/jobs?per_page=100", pages=True)
    jobs = validate_jobs([job for page in pages for job in page['jobs']], run)
    logs = {}
    for arch, job in jobs.items():
        # Raw job logs may contain ANSI escapes. Capture them for validation;
        # never render them to the terminal or relax the JSON API boundary.
        log = subprocess.check_output(['gh', 'api', '--allow-escape-sequences', f"repos/{REPOSITORY}/actions/jobs/{job['id']}/logs"], text=True)
        checkout_sha(log, job, sha)
        logs[arch] = log
    pages = api(f'actions/runs/{run_id}/artifacts?per_page=100', pages=True)
    artifacts = validate_artifacts([item for page in pages for item in page['artifacts']], run, jobs, logs)
    incoming.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='release-recovery-', dir=incoming.parent) as temporary:
        staging = Path(temporary)/'incoming'
        staging.mkdir()
        for arch, item in artifacts.items():
            archive = Path(temporary)/f'{arch}.zip'
            with archive.open('wb') as output:
                # Binary archives may contain any byte, including ESC. Keep the
                # response in a file and verify its digest before extraction.
                subprocess.run(['gh', 'api', '--allow-escape-sequences', f"repos/{REPOSITORY}/actions/artifacts/{item['id']}/zip"], stdout=output, check=True)
            extract_verified(archive, item, staging/f'build-mac-{arch}', version, arch)
        current = api(f'actions/runs/{run_id}')
        require(current.get('status') == 'completed' and current.get('run_attempt') == run['run_attempt'], 'Source run changed while downloading')
        staging.rename(incoming)
    pr_output.write_text(json.dumps(pr or {}, indent=2) + '\n')
    return {'run_id': run_id, 'sha': sha, 'version': version,
            'pr': str(pr['number']) if pr else '',
            'arm64_artifact': str(artifacts['arm64']['id']), 'x64_artifact': str(artifacts['x64']['id'])}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--sha', required=True)
    parser.add_argument('--version', required=True)
    parser.add_argument('--incoming', type=Path, required=True)
    parser.add_argument('--pr-output', type=Path, required=True)
    args = parser.parse_args()
    require(os.environ.get('GITHUB_REPOSITORY') == REPOSITORY and os.environ.get('GITHUB_REF') == 'refs/heads/main', 'Recovery must execute from this repository/main')
    result = recover(args.run_id, args.sha, args.version, args.incoming, args.pr_output)
    if os.environ.get('GITHUB_OUTPUT'):
        with open(os.environ['GITHUB_OUTPUT'], 'a') as output:
            for name, value in result.items():
                output.write(f'{name}={value}\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
