"""Exercise preview scripts with real root checks and inert process boundaries."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tomllib
import unittest

from checkout_fixture import ROOT, checkout


# Only replace package installation and the long-running HTTP server. Python
# preflights execute normally; no provider clients or network tools run here.
PYTHON_SHIM = '''#!{python}
import json, os, sys
from pathlib import Path
if '-m' in sys.argv and 'streamlit' in sys.argv:
    keys = ('NORTHSTAR_PROJECT_ROOT', 'NORTHSTAR_UI_ENABLE_LIVE',
            'NORTHSTAR_UI_ENABLE_PINECONE', 'HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE')
    Path('.local/launch.json').write_text(json.dumps({{
        'args': sys.argv[1:], 'env': {{k:os.environ.get(k) for k in keys}},
        'credentials_present': any(k in os.environ for k in ('NEBIUS_API_KEY','PINECONE_API_KEY','FIREWORKS_API_KEY'))
    }}))
else:
    os.execv({python!r}, [{python!r}, *sys.argv[1:]])
'''
UV_SHIM = '''#!{python}
import json, os, shutil, sys
from pathlib import Path
args = sys.argv[1:]
with Path('.local/uv-calls.jsonl').open('a') as f:
    f.write(json.dumps(args) + '\\n')
if args == ['--version']:
    print(os.environ.get('FAKE_UV_VERSION', 'uv 0.12.23'))
    raise SystemExit(int(os.environ.get('FAKE_UV_VERSION_EXIT', '0')))
elif args[0] == 'export':
    target = Path(args[args.index('--output-file') + 1])
    # Export fixture is created from the actual reviewed lock offline by tests.
    shutil.copyfile(os.environ['REQUIREMENTS_FIXTURE'], target)
    if os.environ.get('ALTER_EXPORT') == '1':
        with target.open('a') as f: f.write('torch==2.2.2\\n')
elif args[0] == 'venv':
    Path('.venv/bin').mkdir(parents=True)
    shutil.copyfile('.local/bin/python3.12', '.venv/bin/python')
    Path('.venv/bin/python').chmod(0o700)
elif args[:2] != ['pip', 'install']:
    raise SystemExit('unexpected uv operation')
'''


class ReplitPreviewTests(unittest.TestCase):
    def setUp(self):
        self.root = checkout(self)
        (self.root / 'scripts').mkdir()
        for name in ('setup_replit_offline.sh', 'run_replit_offline.sh'):
            source = ROOT / 'scripts' / name
            self.assertTrue(source.is_file(), f'missing preview script: {name}')
            shutil.copyfile(source, self.root / 'scripts' / name)
        shutil.copyfile(ROOT / 'uv.lock', self.root / 'uv.lock')
        (self.root / '.local/bin').mkdir(parents=True)
        self.python = self.root / '.local/bin/python3.12'
        self.python.write_text(PYTHON_SHIM.format(python=sys.executable))
        self.python.chmod(0o700)
        uv = self.root / '.local/bin/uv'
        uv.write_text(UV_SHIM.format(python=sys.executable))
        uv.chmod(0o700)
        self.env = {k: v for k, v in os.environ.items() if not k.startswith(('NEBIUS_', 'PINECONE_', 'FIREWORKS_', 'NORTHSTAR_', 'UV_'))}
        self.env.update(PATH=str(self.root / '.local/bin') + os.pathsep + os.environ['PATH'],
                        REPLIT_DEV_DOMAIN='preview-fixture.replit.dev',
                        PYTHONDONTWRITEBYTECODE='1', TMPDIR=str(ROOT / '.local/tmp'))

    def run_script(self, name, **env):
        return subprocess.run(['bash', str(self.root / 'scripts' / name)], cwd=self.root,
                              env=self.env | env, capture_output=True, text=True, timeout=30)

    def venv(self):
        (self.root / '.venv/bin').mkdir(parents=True)
        shutil.copyfile(self.python, self.root / '.venv/bin/python')
        (self.root / '.venv/bin/python').chmod(0o700)

    def test_launch_overrides_live_flags_and_credentials_preserving_protections(self):
        self.venv()
        result = self.run_script('run_replit_offline.sh', NORTHSTAR_UI_ENABLE_LIVE='1',
                                 NORTHSTAR_UI_ENABLE_PINECONE='1', NEBIUS_API_KEY='synthetic-test-value',
                                 PINECONE_API_KEY='synthetic-test-value', FIREWORKS_API_KEY='synthetic-test-value')
        self.assertEqual(result.returncode, 0, result.stderr)
        captured = json.loads((self.root / '.local/launch.json').read_text())
        self.assertEqual(captured['env']['NORTHSTAR_PROJECT_ROOT'], str(self.root))
        for key in ('NORTHSTAR_UI_ENABLE_LIVE', 'NORTHSTAR_UI_ENABLE_PINECONE'):
            self.assertEqual(captured['env'][key], '0')
        self.assertFalse(captured['credentials_present'])
        for flag in ('--server.address=0.0.0.0', '--server.port=8501',
                     '--browser.serverAddress=preview-fixture.replit.dev', '--browser.serverPort=443',
                     '--server.enableCORS=true', '--server.enableXsrfProtection=true',
                     '--browser.gatherUsageStats=false', '--client.showErrorDetails=none'):
            self.assertIn(flag, captured['args'])
        self.assertNotIn('synthetic-test-value', result.stdout + result.stderr)

    def test_standard_venv_interpreter_symlink_is_supported(self):
        (self.root / '.venv/bin').mkdir(parents=True)
        (self.root / '.venv/bin/python').symlink_to(self.python)
        result = self.run_script('run_replit_offline.sh')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue((self.root / '.local/launch.json').exists())

    def test_reviewed_export_installs_only_hashed_subset_and_no_deps_project(self):
        uv = shutil.which('uv')
        if uv is None:
            self.skipTest('uv required for offline lock export; no package installation')
        fixture = self.root / '.local/requirements.txt'
        env = self.env | {
            'UV_CACHE_DIR': str(self.root / '.local/uv-cache'),
            'UV_PYTHON_INSTALL_DIR': str(self.root / '.local/uv-python'),
            'UV_PYTHON_BIN_DIR': str(self.root / '.local/bin'),
            'UV_PROJECT_ENVIRONMENT': str(self.root / '.venv'),
            'UV_PYTHON_DOWNLOADS': 'never',
            'UV_PYTHON': sys.executable,  # Actual installed interpreter, not the process-boundary shim.
        }
        exported = subprocess.run([uv, 'export', '--locked', '--offline', '--extra', 'ui',
            '--no-emit-project', '--prune', 'chromadb', '--prune', 'sentence-transformers',
            '--prune', 'torch', '--prune', 'transformers', '--no-header', '--no-annotate',
            '--output-file', str(fixture)], cwd=self.root, env=env, capture_output=True,
            text=True, timeout=30)
        self.assertEqual(exported.returncode, 0, exported.stderr)
        result = self.run_script('setup_replit_offline.sh', REQUIREMENTS_FIXTURE=str(fixture))
        self.assertEqual(result.returncode, 0, result.stderr)
        calls = self.uv_calls()
        self.assertEqual([call[0] for call in calls], ['--version', 'export', 'venv', 'pip', 'pip'])
        self.assertEqual(calls[-2], ['pip', 'install', '--python', '.venv/bin/python',
            '--require-hashes', '--only-binary', ':all:', '-r', '.local/replit-preview/ui-requirements.txt'])
        self.assertEqual(calls[-1], ['pip', 'install', '--python', '.venv/bin/python',
            '--no-deps', '--editable', '.'])

    def test_invalid_host_never_launches(self):
        self.venv()
        for host in ('', 'example.com', 'https://preview.replit.dev', '.replit.dev', 'x.replit.dev;echo unsafe'):
            with self.subTest(host=host):
                result = self.run_script('run_replit_offline.sh', REPLIT_DEV_DOMAIN=host)
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse((self.root / '.local/launch.json').exists())

    def test_changed_policy_never_launches(self):
        self.venv()
        with (self.root / 'data/policies/HR-001.md').open('a') as f:
            f.write('\nchanged fixture\n')
        result = self.run_script('run_replit_offline.sh')
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse((self.root / '.local/launch.json').exists())

    def test_symlink_cache_never_launches_or_writes_through_it(self):
        self.venv()
        outside = checkout(self)
        (self.root / '.local/cache').symlink_to(outside, target_is_directory=True)
        result = self.run_script('run_replit_offline.sh')
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse((self.root / '.local/launch.json').exists())

    def test_setup_refuses_existing_environment_before_uv(self):
        self.venv()
        before = (self.root / '.venv/bin/python').read_bytes()
        result = self.run_script('setup_replit_offline.sh')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('existing .venv', result.stderr)
        self.assertEqual((self.root / '.venv/bin/python').read_bytes(), before)
        self.assertFalse((self.root / '.local/uv-calls.jsonl').exists())

    def test_setup_refuses_symlink_environment_before_uv(self):
        (self.root / '.venv').symlink_to(ROOT / '.venv', target_is_directory=True)
        result = self.run_script('setup_replit_offline.sh')
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse((self.root / '.local/uv-calls.jsonl').exists())

    def test_setup_refuses_wrong_uv_version_without_installing(self):
        result = self.run_script('setup_replit_offline.sh', FAKE_UV_VERSION='uv 0.0.0')
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse((self.root / '.venv').exists())
        self.assertEqual(self.uv_calls(), [['--version']])

    def test_setup_accepts_exact_uv_version_with_platform_target_suffix(self):
        fixture = self.root / '.local/requirements.txt'
        fixture.write_text('intentionally unapproved export\n')
        expected_calls = []
        for version in (
            'uv 0.12.23',
            'uv 0.12.23 (x86_64-unknown-linux-gnu)',
            'uv 0.12.23 (aarch64-unknown-linux-musl)',
            'uv 0.12.23 (aarch64-apple-darwin)',
            'uv 0.12.23 (x86_64-pc-windows-msvc)',
        ):
            with self.subTest(version=version):
                result = self.run_script('setup_replit_offline.sh', FAKE_UV_VERSION=version,
                                         REQUIREMENTS_FIXTURE=str(fixture))
                # Reaching the unchanged export guard proves the version passed;
                # the deliberately invalid fixture prevents any installation.
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('reviewed UI subset', result.stderr)
                calls = self.uv_calls()
                self.assertEqual([call[0] for call in calls[len(expected_calls):]], ['--version', 'export'])
                expected_calls = calls
                self.assertFalse((self.root / '.venv').exists())

    def test_setup_rejects_different_malformed_or_ambiguous_uv_output(self):
        expected_calls = []
        for version in (
            '', 'uv 0.12.22', 'uv 0.12.24 (x86_64-unknown-linux-gnu)',
            'uv 0.12.230', 'uv 0.12.23rc1', 'uv 0.12.23+local',
            'other 0.12.23', ' uv 0.12.23', 'uv  0.12.23', 'uv 0.12.23 ',
            'uv 0.12.23 ()', 'uv 0.12.23 (linux)',
            'uv 0.12.23 (x86_64--linux-gnu)', 'uv 0.12.23 (x86_64-unknown-linux-gnu',
            'uv 0.12.23 (x86_64-unknown-linux-gnu) extra',
            'uv 0.12.23 (aarch64-apple-darwin) (x86_64-apple-darwin)',
            'uv 0.12.23 (aarch64-apple-darwin extra)',
            'uv 0.12.23\nuv 0.12.24', 'uv 0.12.23\n',
        ):
            with self.subTest(version=version):
                result = self.run_script('setup_replit_offline.sh', FAKE_UV_VERSION=version)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('requires reviewed uv 0.12.23', result.stderr)
                expected_calls.append(['--version'])
                self.assertEqual(self.uv_calls(), expected_calls)
                self.assertFalse((self.root / '.venv').exists())

    def test_failed_uv_version_command_is_rejected_even_with_valid_output(self):
        result = self.run_script('setup_replit_offline.sh', FAKE_UV_VERSION='uv 0.12.23',
                                FAKE_UV_VERSION_EXIT='1')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('requires reviewed uv 0.12.23', result.stderr)
        self.assertEqual(self.uv_calls(), [['--version']])
        self.assertFalse((self.root / '.venv').exists())

    def uv_calls(self):
        return [json.loads(line) for line in (self.root / '.local/uv-calls.jsonl').read_text().splitlines()]

    def test_changed_requirements_stop_before_environment_or_install(self):
        fixture = self.root / '.local/requirements.txt'
        fixture.write_text('torch==2.2.2\n')
        result = self.run_script('setup_replit_offline.sh', REQUIREMENTS_FIXTURE=str(fixture))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('reviewed UI subset', result.stderr)
        self.assertFalse((self.root / '.venv').exists())
        self.assertFalse(any(call[0] in ('venv', 'pip') for call in self.uv_calls()))

    def test_replit_runs_only_offline_launcher_without_automatic_install(self):
        config = tomllib.loads((ROOT / '.replit').read_text())
        self.venv()
        result = subprocess.run(config['run']['args'], cwd=self.root, env=self.env,
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue((self.root / '.local/launch.json').exists())
        self.assertFalse((self.root / '.local/uv-calls.jsonl').exists())
        self.assertEqual(config['ports'], [{'localPort':8501, 'externalPort':80}])
        self.assertFalse(any(config['packager']['features'].values()))
        self.assertNotIn('deployment', config)


if __name__ == '__main__':
    unittest.main()
