"""Release invariants: version agreement, two verified builds, immutable publication."""
from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import yaml

import bump_version
import collect_artifacts
import publish
import release_notes
import select_bump


class ReleaseRulesTests(unittest.TestCase):
    def test_merged_pr_signing_override_stays_on_trusted_main_build(self):
        workflow = yaml.safe_load((Path(__file__).resolve().parents[2]/'.github/workflows/release.yml').read_text())
        # PyYAML's YAML 1.1 loader treats the GitHub "on" key as true.
        triggers = workflow.get('on', workflow.get(True))
        self.assertEqual(triggers['pull_request_target']['types'], ['closed'])
        self.assertEqual(triggers['pull_request_target']['branches'], ['main'])
        prepare = workflow['jobs']['prepare']
        self.assertEqual(prepare['if'], "github.event_name == 'workflow_dispatch' || github.event.pull_request.merged == true")
        prepare_checkout = next(step for step in prepare['steps'] if step.get('uses', '').startswith('actions/checkout@'))
        self.assertEqual(prepare_checkout['with']['ref'], 'main')
        build = workflow['jobs']['build']
        build_checkout = next(step for step in build['steps'] if step.get('uses', '').startswith('actions/checkout@'))
        self.assertEqual(build_checkout['with']['ref'], '${{ needs.prepare.outputs.sha }}')
        signing = next(step for step in build['steps'] if step.get('env', {}).get('SEAMSTRESS_RELEASE') == '1')
        self.assertEqual(signing['env']['CSC_FOR_PULL_REQUEST'], 'true')
        self.assertNotIn('CSC_FOR_PULL_REQUEST', workflow.get('env', {}))
        overrides = []
        for job_id, job in workflow['jobs'].items():
            self.assertNotIn('CSC_FOR_PULL_REQUEST', job.get('env', {}))
            for step in job['steps']:
                if 'CSC_FOR_PULL_REQUEST' in step.get('env', {}):
                    overrides.append((job_id, step['name']))
        self.assertEqual(overrides, [('build', signing['name'])])

    def test_exactly_one_release_label(self):
        self.assertEqual(select_bump.select(['minor', 'ui']), 'minor')
        self.assertEqual(select_bump.select(['no-release']), 'none')
        for labels in [[], ['patch', 'minor'], ['no-release', 'patch']]:
            with self.assertRaises(ValueError):
                select_bump.select(labels)

    def test_bumps_and_recovery(self):
        self.assertEqual(bump_version.bump('1.2.3', 'major'), '2.0.0')
        self.assertEqual(bump_version.bump('1.2.3', 'minor'), '1.3.0')
        self.assertEqual(bump_version.bump('1.2.3', 'patch'), '1.2.4')
        self.assertEqual(bump_version.bump('1.2.3', 'current'), '1.2.3')
        for invalid in ['01.2.3', 'v1.2.3', '1.2.3-rc.1', '1.2.3\nextra']:
            with self.assertRaises(ValueError):
                bump_version.parse(invalid)

    def test_lockfile_mismatch_blocks_bump(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root/'app').mkdir(); (root/'seamstress').mkdir()
            (root/'pyproject.toml').write_text('version = "1.2.3"\n')
            (root/'seamstress/__init__.py').write_text('__version__ = "1.2.3"\n')
            (root/'app/package.json').write_text('{"version": "1.2.3"}\n')
            lock = {'version': '1.2.3', 'packages': {'': {'version': '1.2.2'}}}
            (root/'app/package-lock.json').write_text(json.dumps(lock))
            with self.assertRaises(ValueError):
                bump_version.check(root)


class ArtifactTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.incoming = self.root/'incoming'
        self.output = self.root/'release'
        for arch in ('arm64', 'x64'):
            folder = self.incoming/f'build-mac-{arch}'
            folder.mkdir(parents=True)
            name = f'Seamstress-1.2.3-{arch}.zip'
            payload = f'zip payload for {arch}'.encode()
            (folder/name).write_bytes(payload)
            (folder/f'Seamstress-1.2.3-{arch}.dmg').write_bytes(b'final stapled dmg')
            (folder/f'verification-{arch}.json').write_text(json.dumps({
                'version': '1.2.3', 'arch': arch, 'signed': True, 'notarized': True, 'team_id': 'TEAM123456'}))
            (folder/'latest-mac.yml').write_text(yaml.safe_dump({'version': '1.2.3', 'files': [{
                'url': name, 'size': len(payload),
                'sha512': base64.b64encode(hashlib.sha512(payload).digest()).decode()}]}))

    def test_combines_both_architectures_and_current_download_aliases(self):
        metadata = collect_artifacts.assemble(self.incoming, self.output, '1.2.3')
        self.assertEqual(len(metadata['files']), 2)
        self.assertEqual({item['url'] for item in metadata['files']}, {
            'Seamstress-1.2.3-arm64.zip', 'Seamstress-1.2.3-x64.zip'})
        for arch in ('arm64', 'x64'):
            self.assertEqual((self.output/f'Seamstress-mac-{arch}.dmg').read_bytes(), b'final stapled dmg')
        for line in (self.output/'SHA256SUMS.txt').read_text().splitlines():
            checksum, name = line.split('  ')
            self.assertEqual(hashlib.sha256((self.output/name).read_bytes()).hexdigest(), checksum)

    def test_missing_architecture_or_tampered_zip_prevents_any_release(self):
        zip_path = self.incoming/'build-mac-x64/Seamstress-1.2.3-x64.zip'
        zip_path.write_bytes(b'tampered')
        with self.assertRaisesRegex(ValueError, 'checksum'):
            collect_artifacts.assemble(self.incoming, self.output, '1.2.3')
        self.assertFalse(self.output.exists())

    def test_unsigned_reports_and_mismatched_teams_fail_closed(self):
        report = self.incoming/'build-mac-x64/verification-x64.json'
        data = json.loads(report.read_text())
        data['signed'] = False
        report.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, 'signing'):
            collect_artifacts.assemble(self.incoming, self.output, '1.2.3')
        data['signed'] = True; data['team_id'] = 'OTHERTEAM1'
        report.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, 'same Apple'):
            collect_artifacts.assemble(self.incoming, self.output, '1.2.3')

    def test_stale_or_remote_update_urls_are_rejected(self):
        path = self.incoming/'build-mac-x64/latest-mac.yml'
        data = yaml.safe_load(path.read_text())
        data['files'][0]['url'] = 'https://example.com/payload.zip'
        path.write_text(yaml.safe_dump(data))
        with self.assertRaisesRegex(ValueError, 'Unexpected'):
            collect_artifacts.assemble(self.incoming, self.output, '1.2.3')

    def test_notes_keep_user_changes_and_real_downloads(self):
        collect_artifacts.assemble(self.incoming, self.output, '1.2.3')
        notes = release_notes.build('1.2.3', self.output, '<!-- private template -->\n- **Fixed:** A visible seam.\n\n---\n\nReviewer detail', 'v1.2.2')
        self.assertIn('**Fixed:** A visible seam.', notes)
        self.assertNotIn('Reviewer detail', notes)
        self.assertNotIn('private template', notes)
        self.assertIn('/releases/download/v1.2.3/Seamstress-1.2.3-arm64.dmg', notes)

    def test_published_release_is_never_overwritten(self):
        collect_artifacts.assemble(self.incoming, self.output, '1.2.3')
        notes = self.root/'notes.md'; notes.write_text('Changes')
        with patch.object(publish, 'gh', return_value=json.dumps({'draft': False})) as call:
            with self.assertRaisesRegex(ValueError, 'immutable'):
                publish.publish('1.2.3', 'a'*40, self.output, notes)
        self.assertEqual(call.call_count, 1)

    def test_only_complete_upload_is_published(self):
        collect_artifacts.assemble(self.incoming, self.output, '1.2.3')
        notes = self.root/'notes.md'; notes.write_text('Changes')
        sha = 'a'*40
        state = {'draft': True, 'target_commitish': sha, 'assets': []}
        with patch.object(publish, 'gh', side_effect=[None, None, '', '', json.dumps(state)]) as call:
            with self.assertRaisesRegex(ValueError, 'Uploaded release assets'):
                publish.publish('1.2.3', sha, self.output, notes)
        self.assertFalse(any('edit' in args.args for args in call.call_args_list))

    def test_an_existing_tag_cannot_point_elsewhere(self):
        collect_artifacts.assemble(self.incoming, self.output, '1.2.3')
        notes = self.root/'notes.md'; notes.write_text('Changes')
        reference = {'object': {'type': 'commit', 'sha': 'b'*40}}
        with patch.object(publish, 'gh', side_effect=[None, json.dumps(reference)]) as call:
            with self.assertRaisesRegex(ValueError, 'version tag'):
                publish.publish('1.2.3', 'a'*40, self.output, notes)
        self.assertFalse(any('create' in args.args for args in call.call_args_list))

    def test_complete_draft_is_published_only_after_verified_upload(self):
        collect_artifacts.assemble(self.incoming, self.output, '1.2.3')
        notes = self.root/'notes.md'; notes.write_text('Changes')
        sha = 'a'*40
        assets = [{'name': path.name, 'size': path.stat().st_size} for path in self.output.iterdir()]
        state = {'draft': True, 'target_commitish': sha, 'assets': assets}
        with patch.object(publish, 'gh', side_effect=[None, None, '', '', json.dumps(state), '']) as call:
            url = publish.publish('1.2.3', sha, self.output, notes)
        self.assertTrue(url.endswith('/v1.2.3'))
        self.assertIn('edit', call.call_args.args)
        self.assertIn('--draft=false', call.call_args.args)


if __name__ == '__main__':
    unittest.main()
