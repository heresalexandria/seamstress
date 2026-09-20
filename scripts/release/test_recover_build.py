"""Reject unproven or altered native build artifacts during publish-only recovery."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import io
from pathlib import Path
import stat
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import yaml

import recover_build as recovery

SHA = 'a' * 40
TRIGGER_SHA = 'b' * 40
VERSION = '1.2.3'
SIGNED_BYTES = b'original signed bytes\x00\x1b[31m\xff'


def stamped(message: str, second: int = 1) -> str:
    return f'2026-09-20T12:00:{second:02d}.1234567Z {message}\n'


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.run = {'id': 123, 'repository': {'full_name': recovery.REPOSITORY},
                    'workflow_id': 42, 'path': recovery.WORKFLOW, 'name': 'Release',
                    'status': 'completed', 'conclusion': 'failure', 'run_attempt': 1,
                    'head_sha': TRIGGER_SHA, 'head_branch': 'feature',
                    'event': 'pull_request_target', 'created_at': '2026-09-20T12:00:00Z'}
        self.workflow = {'id': 42, 'path': recovery.WORKFLOW, 'name': 'Release'}
        self.pr = {'number': 7, 'head': {'sha': TRIGGER_SHA, 'ref': 'feature'},
                   'base': {'ref': 'main', 'repo': {'full_name': recovery.REPOSITORY}},
                   'state': 'closed', 'merged_at': '2026-09-20T11:59:59Z', 'merge_commit_sha': 'c' * 40}
        self.jobs = []
        for index, name in enumerate(('prepare release version', 'signed mac-arm64', 'signed mac-x64')):
            self.jobs.append({'id': 100 + index, 'run_id': 123, 'run_attempt': 1,
                              'name': name, 'status': 'completed', 'conclusion': 'success',
                              'steps': [{'name': step, 'conclusion': 'success',
                                         'started_at': '2026-09-20T12:00:00Z',
                                         'completed_at': '2026-09-20T12:00:02Z'}
                                        for step in recovery.REQUIRED_STEPS]})
        self.native = recovery.validate_jobs(self.jobs, self.run)
        self.artifacts = []
        self.logs = {}
        for index, arch in enumerate(('arm64', 'x64')):
            item = {'id': 500 + index, 'name': f'build-mac-{arch}', 'expired': False,
                    'size_in_bytes': 1234, 'digest': 'sha256:' + 'd' * 64,
                    'workflow_run': {'id': 123, 'head_sha': TRIGGER_SHA},
                    # Deliberately different API/runner clocks: never use this as
                    # a substitute for ID/digest provenance from the upload log.
                    'created_at': '2026-09-20T12:01:16Z'}
            self.artifacts.append(item)
            self.logs[arch] = ''.join(stamped(line) for line in (
                f'  ref: {SHA}',
                f'[command]/opt/homebrew/bin/git checkout --progress --force {SHA}',
                '[command]/opt/homebrew/bin/git log -1 --format=%H', SHA,
                'SHA256 digest of uploaded artifact is ' + 'd' * 64,
                f'Artifact build-mac-{arch} has been successfully uploaded! Final size is 1234 bytes. Artifact ID is {item["id"]}',
            ))

    def test_strict_dispatch_inputs(self):
        recovery.inputs('123', SHA, VERSION)
        for run_id, sha, version in [('0', SHA, VERSION), ('1\n', SHA, VERSION),
                                     ('1;exit', SHA, VERSION), ('123', SHA.upper(), VERSION),
                                     ('123', SHA[:-1], VERSION), ('123', SHA, '01.2.3'),
                                     ('123', SHA, '1.2.3-rc.1')]:
            with self.subTest(values=(run_id, sha, version)), self.assertRaises(ValueError):
                recovery.inputs(run_id, sha, version)

    def test_json_api_retains_escape_sequence_protection(self):
        with patch.object(recovery.subprocess, 'check_output', return_value=b'{"id":123}') as call:
            self.assertEqual(recovery.api('actions/runs/123'), {'id': 123})
        self.assertNotIn('--allow-escape-sequences', call.call_args.args[0])

    def test_failed_publisher_run_can_still_have_valid_native_builds(self):
        recovery.validate_run(self.run, self.workflow, '123')
        self.assertEqual(set(self.native), {'arm64', 'x64'})

    def test_wrong_run_repository_workflow_event_or_status_rejected(self):
        for key, value in [('id', 456), ('repository', {'full_name': 'other/repo'}),
                           ('workflow_id', 99), ('path', '.github/workflows/build-check.yml'),
                           ('name', 'Build check'), ('event', 'pull_request'),
                           ('status', 'in_progress'), ('run_attempt', 0)]:
            run = {**self.run, key: value}
            with self.subTest(key=key), self.assertRaises(ValueError):
                recovery.validate_run(run, self.workflow, '123')

    def test_manual_source_must_be_main(self):
        run = {**self.run, 'event': 'workflow_dispatch'}
        with self.assertRaises(ValueError):
            recovery.validate_run(run, self.workflow, '123')
        recovery.validate_run({**run, 'head_branch': 'main'}, self.workflow, '123')

    def test_commit_associated_pr_must_match_merged_main_trigger(self):
        self.assertEqual(recovery.merged_pr(self.run, [self.pr]), self.pr)
        for key, value in [('state', 'open'), ('merged_at', None),
                           ('merged_at', '2026-09-20T12:00:01Z'),
                           ('head', {'sha': SHA, 'ref': 'feature'}),
                           ('base', {'ref': 'main', 'repo': {'full_name': 'other/repo'}}),
                           ('base', {'ref': 'staging', 'repo': {'full_name': recovery.REPOSITORY}})]:
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                recovery.merged_pr(self.run, [{**self.pr, key: value}])
        with self.assertRaises(ValueError):
            recovery.merged_pr(self.run, [self.pr, self.pr])

    def test_native_jobs_must_pass_in_the_same_attempt(self):
        for index, key, value in [(0, 'conclusion', 'skipped'), (1, 'conclusion', 'failure'),
                                  (2, 'conclusion', 'cancelled'), (1, 'run_attempt', 2),
                                  (2, 'run_id', 456)]:
            jobs = deepcopy(self.jobs)
            jobs[index][key] = value
            with self.subTest(index=index, key=key), self.assertRaises(ValueError):
                recovery.validate_jobs(jobs, self.run)
        with self.assertRaises(ValueError):
            recovery.validate_jobs(self.jobs + [self.jobs[-1]], self.run)
        for name in recovery.REQUIRED_STEPS:
            jobs = deepcopy(self.jobs)
            next(step for step in jobs[1]['steps'] if step['name'] == name)['conclusion'] = 'skipped'
            with self.subTest(step=name), self.assertRaises(ValueError):
                recovery.validate_jobs(jobs, self.run)

    def test_checkout_proves_prepared_sha_not_trigger_sha(self):
        recovery.checkout_sha(self.logs['arm64'], self.native['arm64'], SHA)
        with self.assertRaises(ValueError):
            recovery.checkout_sha(self.logs['arm64'], self.native['arm64'], TRIGGER_SHA)
        for log in [self.logs['arm64'].replace('git log -1', 'echo git log -1'),
                    self.logs['arm64'].replace('12:00:01.', '12:00:05.'),
                    self.logs['arm64'] + stamped('[command]/usr/bin/git log -1 --format=%H') + stamped(SHA),
                    self.logs['arm64'].replace(f'  ref: {SHA}', '  ref: main')]:
            with self.assertRaises(ValueError):
                recovery.checkout_sha(log, self.native['arm64'], SHA)

    def test_artifact_provenance_tolerates_server_runner_clock_skew(self):
        selected = recovery.validate_artifacts(self.artifacts, self.run, self.native, self.logs)
        self.assertEqual(selected['arm64']['id'], 500)
        self.assertNotEqual(self.run['head_sha'], SHA)

    def test_artifact_id_size_digest_and_run_must_match_native_upload(self):
        for key, value in [('id', 999), ('size_in_bytes', 999), ('digest', 'sha256:' + 'e' * 64),
                           ('expired', True), ('digest', None),
                           ('workflow_run', {'id': 456, 'head_sha': TRIGGER_SHA}),
                           ('workflow_run', {'id': 123, 'head_sha': SHA})]:
            artifacts = deepcopy(self.artifacts)
            artifacts[0][key] = value
            with self.subTest(key=key), self.assertRaises((ValueError, TypeError)):
                recovery.validate_artifacts(artifacts, self.run, self.native, self.logs)
        with self.assertRaises(ValueError):
            recovery.validate_artifacts(self.artifacts + [self.artifacts[0]], self.run, self.native, self.logs)
        logs = {**self.logs, 'arm64': self.logs['arm64'].replace('12:00:01.', '12:00:05.')}
        with self.assertRaises(ValueError):
            recovery.validate_artifacts(self.artifacts, self.run, self.native, logs)

    def archive(self, extra=None, arch='arm64'):
        stream = io.BytesIO()
        with zipfile.ZipFile(stream, 'w') as target:
            for name in (f'Seamstress-{VERSION}-{arch}.dmg', f'Seamstress-{VERSION}-{arch}.zip',
                         'latest-mac.yml', f'verification-{arch}.json'):
                target.writestr(name, SIGNED_BYTES)
            if extra is not None:
                target.writestr(extra, b'unexpected')
        return stream.getvalue()

    def extract(self, data, *, digest=None, size=None):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root/'artifact.zip'
            archive.write_bytes(data)
            item = {'digest': digest or 'sha256:' + hashlib.sha256(data).hexdigest(),
                    'size_in_bytes': len(data) if size is None else size}
            recovery.extract_verified(archive, item, root/'native', VERSION, 'arm64')
            self.assertEqual((root/'native'/f'Seamstress-{VERSION}-arm64.zip').read_bytes(), SIGNED_BYTES)

    def test_download_digest_and_size_are_hard_failures(self):
        data = self.archive()
        self.extract(data)
        with self.assertRaises(ValueError):
            self.extract(data, digest='sha256:' + '0' * 64)
        with self.assertRaises(ValueError):
            self.extract(data, size=len(data) + 1)

    def test_archive_rejects_traversal_extra_paths_and_symlinks(self):
        for name in ['../escape', '/absolute', 'nested/latest-mac.yml', 'secret.pem']:
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.extract(self.archive(name))
        link = zipfile.ZipInfo(f'Seamstress-{VERSION}-arm64.zip.blockmap')
        link.create_system = 3
        link.external_attr = (stat.S_IFLNK | 0o777) << 16
        with self.assertRaises(ValueError):
            self.extract(self.archive(link))

    def test_recovery_stages_exact_downloads_and_rechecks_source_attempt(self):
        payloads = {}
        for arch, item in zip(('arm64', 'x64'), self.artifacts):
            data = self.archive(arch=arch)
            payloads[item['id']] = data
            item['size_in_bytes'] = len(data)
            item['digest'] = 'sha256:' + hashlib.sha256(data).hexdigest()
            self.logs[arch] = self.logs[arch].replace('1234 bytes', f'{len(data)} bytes').replace('d' * 64, item['digest'][7:])
            self.logs[arch] += stamped('\x1b[36mCaptured log decoration\x1b[0m')
        for changed in (False, True):
            reads = 0
            def api(endpoint, **kwargs):
                nonlocal reads
                if endpoint == 'actions/runs/123':
                    reads += 1
                    return {**self.run, 'run_attempt': 2 if changed and reads == 2 else 1}
                return {'actions/workflows/release.yml': self.workflow,
                        f'commits/{TRIGGER_SHA}/pulls?per_page=100': [[self.pr]],
                        'actions/runs/123/attempts/1/jobs?per_page=100': [{'jobs': self.jobs}],
                        'actions/runs/123/artifacts?per_page=100': [{'artifacts': self.artifacts}]}[endpoint]
            def download(command, *, stdout, check):
                self.assertTrue(check)
                self.assertEqual(command[:3], ['gh', 'api', '--allow-escape-sequences'])
                artifact_id = int(command[-1].split('/')[-2])
                stdout.write(payloads[artifact_id])
            def log(command, **kwargs):
                self.assertEqual(command[:3], ['gh', 'api', '--allow-escape-sequences'])
                self.assertTrue(kwargs['text'])
                job_id = int(command[-1].split('/')[-2])
                return self.logs['arm64' if job_id == 101 else 'x64']
            with self.subTest(changed=changed), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                incoming, pr = root/'incoming', root/'pr.json'
                with patch.object(recovery, 'api', side_effect=api), \
                     patch.object(recovery, 'validate_source'), patch.object(recovery, 'ancestor'), \
                     patch.object(recovery.subprocess, 'run', side_effect=download), \
                     patch.object(recovery.subprocess, 'check_output', side_effect=log):
                    if changed:
                        with self.assertRaisesRegex(ValueError, 'Source run changed'):
                            recovery.recover('123', SHA, VERSION, incoming, pr)
                        self.assertFalse(incoming.exists())
                        self.assertFalse(pr.exists())
                        self.assertEqual(list(root.iterdir()), [])
                    else:
                        result = recovery.recover('123', SHA, VERSION, incoming, pr)
                        self.assertEqual(result['pr'], '7')
                        self.assertEqual(result['sha'], SHA)
                        self.assertEqual({path.name for path in incoming.iterdir()}, {'build-mac-arm64', 'build-mac-x64'})
                        for arch in ('arm64', 'x64'):
                            self.assertEqual((incoming/f'build-mac-{arch}'/f'Seamstress-{VERSION}-{arch}.zip').read_bytes(), SIGNED_BYTES)
                        self.assertTrue(pr.is_file())

    def test_source_validation_reads_the_expected_commit_and_checks_ancestry(self):
        workflow = (Path(__file__).resolve().parents[2]/recovery.WORKFLOW).read_text()
        files = {'pyproject.toml': f'version = "{VERSION}"',
                 'seamstress/__init__.py': f'__version__ = "{VERSION}"',
                 'app/package.json': '{"version":"1.2.3"}',
                 'app/package-lock.json': '{"version":"1.2.3","packages":{"":{"version":"1.2.3"}}}',
                 recovery.WORKFLOW: workflow}
        def show(command, ref):
            self.assertEqual(command, 'show')
            commit, name = ref.split(':', 1)
            self.assertEqual(commit, SHA)
            return files[name]
        with patch.object(recovery, 'git', side_effect=show), patch.object(recovery, 'ancestor') as ancestor:
            recovery.validate_source(SHA, VERSION, 'pull_request_target')
            ancestor.assert_called_once_with(SHA, 'origin/main')
            with self.assertRaises(ValueError):
                recovery.validate_source(SHA, '1.2.4', 'pull_request_target')
            files[recovery.WORKFLOW] = workflow.replace('types: [closed]', 'types: [opened]')
            with self.assertRaises(ValueError):
                recovery.validate_source(SHA, VERSION, 'pull_request_target')

    def test_workflow_is_publish_only_from_main_with_shared_release_lock(self):
        path = Path(__file__).resolve().parents[2]/'.github/workflows/recover-release.yml'
        text = path.read_text()
        workflow = yaml.safe_load(text)
        self.assertEqual(workflow['concurrency'], {'group': 'release', 'cancel-in-progress': False})
        self.assertEqual(set(workflow.get('on', workflow.get(True))), {'workflow_dispatch'})
        jobs = list(workflow['jobs'].values())
        self.assertEqual(len(jobs), 1)
        job = jobs[0]
        self.assertEqual(job['permissions']['actions'], 'read')
        self.assertEqual(job['permissions']['contents'], 'write')
        checkout = next(step for step in job['steps'] if step.get('uses', '').startswith('actions/checkout@'))
        self.assertEqual(checkout['with']['ref'], 'main')
        self.assertFalse(checkout['with']['persist-credentials'])
        self.assertEqual(checkout['with']['fetch-depth'], 0)
        self.assertIn('refs/heads/main', text)
        self.assertNotIn('secrets.CSC_', text)
        self.assertNotIn('secrets.APPLE_', text)
        self.assertNotIn('electron-builder', text)


if __name__ == '__main__':
    unittest.main()
