"""Credential and control-plane tests never contact Pinecone."""

from copy import deepcopy
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
from checkout_fixture import checkout
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from northstar_atlas.vector_store.pinecone_cloud import (
    CloudError, CloudSettings, load_settings, record_confirmation,
    check_confirmation, preflight, create_index, validate_description,
)

ROOT = Path(__file__).resolve().parents[1]
KEY = "FAKE_TEST_ONLY_credential_1234567890"
TARGET = {"name": "northstar-atlas-dev", "cloud": "azure", "region": "eastus2",
          "organization": "Synthetic Test Org", "project_name": "synthetic-project",
          "project_id": "00000000-0000-4000-8000-000000000001", "plan": "Builder"}

PROJECT_ID = TARGET["project_id"]
ENVIRONMENT = {"PINECONE_API_KEY": KEY, "PINECONE_INDEX_NAME": "northstar-atlas-dev",
               "PINECONE_CLOUD": "azure", "PINECONE_REGION": "eastus2"}


class UnreadableEnvironment(dict):
    """Prove invalid roots/files/incomplete inputs fail before reading secrets."""
    def __getitem__(self, key):
        raise AssertionError("environment values must not be accessed")


def cloud_settings(key=KEY):
    return CloudSettings(api_key=key, **TARGET)


def description(ready=True):
    return {"name": "northstar-atlas-dev", "host": "https://example.svc.eastus2.pinecone.io",
            "status": {"ready": ready}, "deletion_protection": "enabled",
            "deployment": {"deployment_type": "managed", "cloud": "azure", "region": "eastus2"},
            "schema": {"fields": {"embedding": {"type": "dense_vector", "dimension": 384, "metric": "cosine"}}}}


class Indexes:
    def __init__(self, value=None):
        self.value, self.created = value, []
        self.stuck = False

    def list(self):
        return [] if self.value is None else [self.value]

    def describe(self, name):
        return deepcopy(self.value)

    def create(self, **kwargs):
        self.created.append(kwargs)
        self.value = description(not self.stuck)
        return deepcopy(self.value)


class CloudTests(unittest.TestCase):
    def setUp(self):
        self.root = checkout(self)
        environment = patch.dict(os.environ, NORTHSTAR_PROJECT_ROOT=str(self.root))
        environment.start()
        self.addCleanup(environment.stop)
        (self.root / '.local/pinecone').mkdir(parents=True)
        self.target_path = self.root / '.local/pinecone/target.json'
        self.target_path.write_text(json.dumps(TARGET))
        (self.root / ".env").write_text(f"PINECONE_API_KEY={KEY}\nPINECONE_INDEX_NAME=northstar-atlas-dev\nPINECONE_CLOUD=azure\nPINECONE_REGION=eastus2\n")
        self.settings = cloud_settings()

    def load(self, ignored=True, tracked=False, *, environ=None):
        def git(args, **kwargs):
            from subprocess import CompletedProcess
            return CompletedProcess(args, 0 if ignored else 1, b".env\0" if tracked and "ls-files" in args else b"", b"")
        with patch("northstar_atlas.vector_store.pinecone_cloud.subprocess.run", side_effect=git):
            return load_settings(self.root, environ={} if environ is None else environ)

    def test_secure_loading_no_key_repr(self):
        self.assertEqual(self.load().api_key, KEY)
        self.assertNotIn(KEY, repr(self.load()))

    def test_complete_environment_without_file_or_git(self):
        (self.root / '.env').unlink()
        with patch('northstar_atlas.vector_store.pinecone_cloud.subprocess.run',
                   side_effect=AssertionError('environment-only loading must not require Git')):
            settings = load_settings(self.root, environ=ENVIRONMENT)
        self.assertEqual(settings, self.settings)
        self.assertFalse((self.root / '.env').exists())

    def test_environment_only_uses_process_variables_by_default(self):
        (self.root / '.env').unlink()
        with patch.dict(os.environ, ENVIRONMENT), \
             patch('northstar_atlas.vector_store.pinecone_cloud.subprocess.run',
                   side_effect=AssertionError('environment-only loading must not require Git')):
            self.assertEqual(load_settings(self.root), self.settings)

    def test_environment_only_rejects_every_incomplete_subset_before_values(self):
        (self.root / '.env').unlink()
        keys = tuple(ENVIRONMENT)
        for mask in range(2 ** len(keys) - 1):
            partial = UnreadableEnvironment({key: ENVIRONMENT[key] for i, key in enumerate(keys)
                                             if mask & (1 << i)})
            with self.subTest(present=tuple(partial)), self.assertRaises(CloudError):
                self.load(environ=partial)

    def test_environment_only_rejects_empty_or_nonstring_values(self):
        (self.root / '.env').unlink()
        for name in ENVIRONMENT:
            for value in ('', ' ', '\t', None, 42):
                with self.subTest(name=name, value=value), self.assertRaises(CloudError):
                    self.load(environ={**ENVIRONMENT, name: value})

    def test_environment_only_requires_matching_explicit_target(self):
        (self.root / '.env').unlink()
        for name, value in (('PINECONE_INDEX_NAME', 'another-index'),
                            ('PINECONE_CLOUD', 'aws'), ('PINECONE_REGION', 'us-east-1')):
            with self.subTest(name=name), self.assertRaises(CloudError):
                self.load(environ={**ENVIRONMENT, name: value})
        for raw in ('{}', json.dumps({**TARGET, 'project_id': 'invalid'})):
            self.target_path.write_text(raw)
            with self.assertRaises(CloudError): self.load(environ=ENVIRONMENT)
        self.target_path.unlink()
        with self.assertRaises(CloudError): self.load(environ=ENVIRONMENT)

    def test_existing_file_fallback_and_matching_environment(self):
        self.assertEqual(self.load(), self.settings)
        self.assertEqual(self.load(environ=ENVIRONMENT), self.settings)
        self.assertEqual(self.load(environ={'PINECONE_REGION': 'eastus2'}), self.settings)

    def test_existing_file_rejects_conflicting_and_empty_process_values(self):
        for name, original in ENVIRONMENT.items():
            for value in ('', original + '_different'):
                with self.subTest(name=name), self.assertRaises(CloudError) as raised:
                    self.load(environ={name: value})
                self.assertNotIn(KEY, str(raised.exception))

    def test_environment_cannot_fill_or_override_invalid_existing_file(self):
        path = self.root / '.env'
        original = path.read_text()
        for raw in ('', '\n'.join(original.splitlines()[1:]),
                    original + 'PINECONE_REGION=eastus2\n',
                    original.replace(KEY, '"unterminated')):
            path.write_text(raw)
            with self.assertRaises(CloudError): self.load(environ=ENVIRONMENT)

    def test_existing_file_git_guards_precede_credential_access(self):
        read_text = Path.read_text
        def guarded_read(path, *args, **kwargs):
            if path == self.root / '.env':
                raise AssertionError('unsafe credential file must not be read')
            return read_text(path, *args, **kwargs)
        with patch.object(Path, 'read_text', guarded_read):
            for flags in ({'ignored': False}, {'tracked': True}):
                with self.subTest(flags=flags), self.assertRaises(CloudError):
                    self.load(**flags, environ=UnreadableEnvironment(ENVIRONMENT))

    def test_invalid_root_precedes_environment_credential_access(self):
        with self.assertRaises(CloudError):
            load_settings(self.root / 'missing', environ=UnreadableEnvironment(ENVIRONMENT))
        (self.root / 'data/corpus_manifest.json').write_text('{}')
        with self.assertRaises(CloudError):
            load_settings(self.root, environ=UnreadableEnvironment(ENVIRONMENT))

    def test_dangling_credential_symlink_is_not_environment_only_absence(self):
        path = self.root / '.env'
        path.unlink()
        path.symlink_to(self.root / 'missing-secret-file')
        with self.assertRaises(CloudError):
            load_settings(self.root, environ=UnreadableEnvironment(ENVIRONMENT))

    def test_environment_only_preserves_owner_and_credential_binding(self):
        (self.root / '.env').unlink()
        settings = self.load(environ=ENVIRONMENT)
        with self.assertRaises(CloudError): check_confirmation(self.root, settings)
        record_confirmation(self.root, settings, project_id=PROJECT_ID)
        check_confirmation(self.root, self.load(environ=ENVIRONMENT))
        changed_key = self.load(environ={**ENVIRONMENT, 'PINECONE_API_KEY': KEY + '_rotated'})
        with self.assertRaises(CloudError): check_confirmation(self.root, changed_key)
        self.target_path.write_text(json.dumps({**TARGET, 'project_id': '00000000-0000-4000-8000-000000000002'}))
        changed_project = self.load(environ=ENVIRONMENT)
        with self.assertRaises(CloudError): check_confirmation(self.root, changed_project)

    def test_environment_only_redacts_repr_errors_and_output(self):
        (self.root / '.env').unlink()
        output = io.StringIO()
        with redirect_stdout(output), redirect_stderr(output), self.assertNoLogs(level='DEBUG'):
            settings = self.load(environ=ENVIRONMENT)
            with self.assertRaises(CloudError) as raised:
                self.load(environ={**ENVIRONMENT, 'PINECONE_API_KEY': KEY + '!'})
        self.assertNotIn(KEY, repr(settings))
        self.assertNotIn(KEY, str(raised.exception))
        self.assertEqual(output.getvalue(), '')

    def test_environment_only_invalid_configuration_stops_before_sdk(self):
        from northstar_atlas import pinecone
        (self.root / '.env').unlink()
        for values in ({}, {'PINECONE_API_KEY': KEY},
                       {**ENVIRONMENT, 'PINECONE_REGION': 'us-east-1'}, ENVIRONMENT):
            # The final case has valid settings but no owner confirmation.
            with self.subTest(names=tuple(values)), \
                 patch.dict(os.environ, {'NORTHSTAR_PROJECT_ROOT': str(self.root), **values}, clear=True), \
                 patch.object(Path, 'cwd', return_value=self.root), \
                 patch.object(pinecone, 'sdk_client', side_effect=AssertionError('cloud forbidden')) as client, \
                 redirect_stderr(io.StringIO()):
                self.assertEqual(pinecone.main(['status']), 1)
                client.assert_not_called()

    def test_identity_must_be_explicit_not_inferred_from_credentials(self):
        self.target_path.unlink()
        with self.assertRaises(CloudError):
            self.load()

    def test_target_identity_validation_rejects_unknown_duplicate_and_malformed_fields(self):
        for raw in ('{}', json.dumps({**TARGET, 'project_id': ''}),
                    json.dumps({**TARGET, 'project_id': 'not-a-project-id'}),
                    json.dumps({**TARGET, 'organization': ' ' }),
                    json.dumps({**TARGET, 'unknown': True}),
                    json.dumps(TARGET)[:-1] + ', "project_id":"duplicate"}'):
            with self.subTest(raw=raw):
                self.target_path.write_text(raw)
                with self.assertRaises(CloudError):
                    self.load()

    def test_target_file_symlink_and_env_conflict_rejected(self):
        self.target_path.write_text(json.dumps({**TARGET, 'name': 'another-index'}))
        with self.assertRaises(CloudError):
            self.load()
        self.target_path.rename(self.target_path.with_name('saved-target.json'))
        self.target_path.symlink_to(self.target_path.with_name('saved-target.json'))
        with self.assertRaises(CloudError):
            self.load()

    def test_ignored_and_untracked_required(self):
        for args in ({"ignored": False}, {"tracked": True}):
            with self.assertRaises(CloudError): self.load(**args)

    def test_bad_missing_duplicate_and_conflicting_configuration(self):
        original = (self.root / ".env").read_text()
        for bad in ("", original + "PINECONE_API_KEY=another\n", original.replace("eastus2", "us-east-1"),
                    original.replace(KEY, "$(secret_command)")):
            (self.root / ".env").write_text(bad)
            with self.assertRaises(CloudError) as raised: self.load()
            self.assertNotIn(KEY, str(raised.exception))

    def test_symlink_credentials_rejected(self):
        (self.root / ".env").rename(self.root / "saved")
        (self.root / ".env").symlink_to(self.root / "saved")
        with self.assertRaises(CloudError): self.load()

    def test_confirmation_binds_key_and_exact_project(self):
        with self.assertRaises(CloudError): check_confirmation(self.root, self.settings)
        record_confirmation(self.root, self.settings, project_id=PROJECT_ID)
        check_confirmation(self.root, self.settings)
        with self.assertRaises(CloudError): check_confirmation(self.root, cloud_settings(KEY + "other"))
        with self.assertRaises(CloudError): record_confirmation(self.root, self.settings, project_id="wrong")
        raw = (self.root / ".local/pinecone/target-confirmation.json").read_text()
        self.assertNotIn(KEY, raw)

    def test_confirmation_binds_index_organization_and_project_without_auto_switch(self):
        record_confirmation(self.root, self.settings, project_id=PROJECT_ID)
        for field, value in (('name', 'another-index'), ('organization', 'Other Org'),
                             ('project_name', 'other-project'),
                             ('project_id', '00000000-0000-4000-8000-000000000002')):
            changed = replace(self.settings, **{field: value})
            api = Indexes()
            with patch.object(api, 'list', side_effect=AssertionError('cloud forbidden')):
                with self.assertRaises(CloudError):
                    preflight(api, self.root, changed)
            self.assertFalse(api.created)

    def test_alternative_explicit_index_is_used_only_after_its_own_confirmation(self):
        settings = replace(self.settings, name='synthetic-other-index')
        record_confirmation(self.root, settings, project_id=PROJECT_ID)
        value = description(); value['name'] = settings.name
        api = Indexes(value)
        self.assertEqual(preflight(api, self.root, settings)['index_state'], 'compatible')
        with self.assertRaises(CloudError):
            validate_description(description(), settings)
        self.assertFalse(api.created)

    def test_missing_credentials_and_confirmation_stop_before_sdk_construction(self):
        from northstar_atlas import pinecone
        for content in ('', (self.root / '.env').read_text()):
            (self.root / '.env').write_text(content)
            with patch.object(Path, 'cwd', return_value=self.root), \
                 patch.object(pinecone, 'sdk_client', side_effect=AssertionError('cloud forbidden')) as client:
                from contextlib import redirect_stderr
                import io
                with redirect_stderr(io.StringIO()):
                    self.assertEqual(pinecone.main(['status']), 1)
                client.assert_not_called()

    def test_invalid_explicit_namespace_rejected_before_status_cloud_reads(self):
        from types import SimpleNamespace
        from northstar_atlas.vector_store.pinecone_workflow import status_report, RequestMeter
        record_confirmation(self.root, self.settings, project_id=PROJECT_ID)
        for namespace in ('', ' ', '../outside', 'not-a-namespace'):
            with self.subTest(namespace=namespace):
                api = Indexes()
                with patch.object(api, 'list', wraps=api.list) as listed:
                    with self.assertRaises(ValueError):
                        status_report(self.root, self.settings, SimpleNamespace(indexes=api),
                                      RequestMeter(), namespace=namespace)
                    listed.assert_not_called()

    def test_cli_empty_namespace_stops_before_credentials_or_provider_setup(self):
        from contextlib import redirect_stderr
        import io
        from northstar_atlas import pinecone
        for command in ('index', 'status'):
            with patch.object(Path, 'cwd', return_value=self.root), \
                 patch.object(pinecone, 'load_settings', side_effect=AssertionError('credentials forbidden')) as settings:
                with redirect_stderr(io.StringIO()):
                    self.assertEqual(pinecone.main([command, '--namespace', '']), 1)
                settings.assert_not_called()

    def test_preflight_is_read_only_and_requires_confirmation(self):
        api = Indexes()
        with self.assertRaises(CloudError): preflight(api, self.root, self.settings)
        record_confirmation(self.root, self.settings, project_id=PROJECT_ID)
        result = preflight(api, self.root, self.settings)
        self.assertEqual(result["index_state"], "absent")
        self.assertEqual(api.created, [])

    def test_create_only_exact_configuration_and_reuse(self):
        api = Indexes()
        record_confirmation(self.root, self.settings, project_id=PROJECT_ID)
        result = create_index(api, self.root, self.settings, attempts=2, interval=0)
        self.assertTrue(result["status"]["ready"])
        self.assertEqual(len(api.created), 1)
        self.assertEqual(api.created[0]["name"], "northstar-atlas-dev")
        self.assertEqual(api.created[0]["timeout"], -1)
        self.assertEqual(api.created[0]["schema"], description()["schema"])
        create_index(api, self.root, self.settings, attempts=2, interval=0)
        self.assertEqual(len(api.created), 1)

    def test_incompatible_settings_and_untrusted_hosts_rejected(self):
        for key, value in (("deployment", {"deployment_type":"managed", "cloud":"aws", "region":"us-east-1"}),
                           ("host", "https://pinecone.io.attacker.invalid"),
                           ("schema", {"fields":{"embedding":{"type":"dense_vector","dimension":768,"metric":"cosine"}}})):
            bad = description(); bad[key] = value
            api = Indexes(bad)
            record_confirmation(self.root, self.settings, project_id=PROJECT_ID)
            with self.assertRaises(CloudError): create_index(api, self.root, self.settings)
            self.assertEqual(api.created, [])

    def test_readiness_timeout_and_provider_error_are_safe(self):
        record_confirmation(self.root, self.settings, project_id=PROJECT_ID)
        api = Indexes(); api.stuck = True
        with self.assertRaisesRegex(CloudError, "readiness"):
            create_index(api, self.root, self.settings, attempts=2, interval=0)
        with patch.object(api, "list", side_effect=RuntimeError(KEY)):
            with self.assertRaises(CloudError) as raised: preflight(api, self.root, self.settings)
        self.assertNotIn(KEY, str(raised.exception))

    def test_initializing_index_without_host_is_polled(self):
        record_confirmation(self.root, self.settings, project_id=PROJECT_ID)
        api = Indexes()
        pending = description(False); pending["host"] = None
        with patch.object(api, "create", return_value=pending), patch.object(api, "describe", return_value=description()) as describe:
            result = create_index(api, self.root, self.settings, attempts=2, interval=0)
        self.assertTrue(result["status"]["ready"])
        describe.assert_called_once()
        with self.assertRaises(CloudError): validate_description(pending, self.settings, require_ready=True)


if __name__ == "__main__": unittest.main()
