"""Root authority is server configuration, not a question or caller path."""
import os
from pathlib import Path
import unittest
from unittest.mock import patch

from checkout_fixture import ROOT, checkout


class RootTests(unittest.TestCase):
    def setUp(self):
        self.root = checkout(self)
        env = patch.dict(os.environ, NORTHSTAR_PROJECT_ROOT=str(self.root))
        env.start()
        self.addCleanup(env.stop)

    def test_authorized_alternative_checkout_resource_owner(self):
        from northstar_atlas.ui.runtime import ResourceOwner
        owner = ResourceOwner(self.root)
        self.addCleanup(owner.close)
        self.assertEqual(owner.root, self.root)
        self.assertEqual(owner.attempts, 0)

    def test_alternative_checkout_qa_service_keeps_real_configuration_boundary(self):
        from contextlib import contextmanager
        from northstar_atlas.qa import open_qa_service
        from test_qa import FakeLLM, FixtureStore
        @contextmanager
        def retrieval(root, backend):
            self.assertEqual(root, self.root)
            self.assertEqual(backend, 'chroma')
            yield FixtureStore(()), object(), 'fixture-corpus'
        with patch('northstar_atlas.qa.open_retriever', retrieval):
            with open_qa_service(self.root, backend='chroma', provider=FakeLLM()) as service:
                result = service.ask('Unknown policy?')
        self.assertTrue(result.accepted)
        self.assertEqual(result.public.kind, 'abstain')

    def validate(self, path=None):
        from northstar_atlas.project import validate_project_root
        return validate_project_root(path)

    def test_current_checkout_with_source_anchored_default(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(self.validate(ROOT), ROOT)

    def test_caller_cannot_override_trusted_root_even_with_valid_corpus(self):
        with self.assertRaises(ValueError):
            self.validate(ROOT)

    def test_relative_or_blank_config_and_missing_root_are_rejected_without_creation(self):
        absent = self.root / 'absent'
        for value in ('', '.', str(absent)):
            with patch.dict(os.environ, NORTHSTAR_PROJECT_ROOT=value), self.assertRaises(ValueError):
                self.validate()
        self.assertFalse(absent.exists())

    def test_invalid_repository_identity(self):
        (self.root / 'pyproject.toml').write_text('[project]\nname="another-project"\n')
        with self.assertRaises(ValueError):
            self.validate()

    def test_missing_manifest_is_rejected_before_service_setup(self):
        (self.root / 'data/corpus_manifest.json').unlink()
        from northstar_atlas.qa import QAConfigurationError, open_qa_service
        with patch('northstar_atlas.qa.open_retriever') as opened:
            with self.assertRaises(QAConfigurationError), open_qa_service(self.root, provider=object()):
                pass
            opened.assert_not_called()
        with self.assertRaises(ValueError):
            self.validate()

    def test_incorrect_hash_of_even_draft_source_is_rejected(self):
        source = self.root / 'data/policies/PLAT-001.md'
        source.write_bytes(source.read_bytes() + b'\nChanged\n')
        with self.assertRaises(ValueError):
            self.validate()

    def test_manifest_and_runtime_symlink_escape(self):
        other = checkout(self)
        manifest = self.root / 'data/corpus_manifest.json'
        original = manifest.read_bytes()
        manifest.unlink()
        manifest.symlink_to(other / 'data/corpus_manifest.json')
        with self.assertRaises(ValueError):
            self.validate()
        manifest.unlink()
        manifest.write_bytes(original)
        (self.root / '.local').symlink_to(other, target_is_directory=True)
        with self.assertRaises(ValueError):
            self.validate()

    def test_symlink_root_cannot_alias_authority(self):
        alias = self.root / 'alias'
        alias.symlink_to(ROOT, target_is_directory=True)
        with patch.dict(os.environ, NORTHSTAR_PROJECT_ROOT=str(alias)), self.assertRaises(ValueError):
            self.validate()

    def test_contained_paths_reject_traversal_absolute_and_interior_symlinks(self):
        from northstar_atlas.project import contained_path
        (self.root / '.local').mkdir()
        (self.root / '.local/link').symlink_to(ROOT, target_is_directory=True)
        for path in ('../outside', str(ROOT / '.env'), '.local/link/.env'):
            with self.assertRaises(ValueError):
                contained_path(self.root, path)
        self.assertEqual(contained_path(self.root, '.local/new/file'), self.root / '.local/new/file')
        self.assertFalse((self.root / '.local/new').exists())
