"""Controlled startup boundaries; HTTP/provider processes are always mocked."""
from contextlib import contextmanager, redirect_stdout
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import unittest
from unittest.mock import Mock, patch

from checkout_fixture import ROOT, checkout
from northstar_atlas import live_preflight
from northstar_atlas.live_budget import BudgetError
from northstar_atlas.llm import MODEL_ID


class PreflightTests(unittest.TestCase):
    def setUp(self):
        self.root = checkout(self)
        p = patch.dict(os.environ, NORTHSTAR_PROJECT_ROOT=str(self.root), VECTOR_STORE='chroma')
        p.start(); self.addCleanup(p.stop)

    def test_missing_budget_prevents_all_network_and_secret_access(self):
        with patch.object(Path, 'cwd', return_value=self.root), \
             patch('northstar_atlas.live_budget.configured_budget') as factory, \
             patch.object(live_preflight, 'require_private_preview') as access, \
             patch.object(live_preflight, 'require_secret_environment') as secrets, \
             patch('northstar_atlas.qa.open_retriever') as retrieval, \
             patch('northstar_atlas.nebius.load_settings') as credentials, redirect_stdout(io.StringIO()):
            factory.return_value.check_ready.side_effect = BudgetError('synthetic')
            self.assertEqual(live_preflight.main(), 1)
            for call in (access, secrets, retrieval, credentials): call.assert_not_called()

    def test_existing_chroma_status_contract_and_no_pinecone_requirement(self):
        store = Mock(spec=['status'])
        store.status.return_value = {'state':'ready', 'corpus_fingerprint':'verified'}
        @contextmanager
        def retriever(root, backend):
            self.assertEqual(backend, 'chroma')
            yield store, object(), 'verified'
        with patch.object(Path, 'cwd', return_value=self.root), \
             patch('northstar_atlas.live_budget.configured_budget'), \
             patch.object(live_preflight, 'require_private_preview'), \
             patch.object(live_preflight, 'require_secret_environment'), \
             patch('northstar_atlas.qa.open_retriever', side_effect=retriever), \
             patch('northstar_atlas.nebius.load_settings', return_value=Mock(model=MODEL_ID)), \
             patch('northstar_atlas.nebius.NebiusProvider') as provider, redirect_stdout(io.StringIO()):
            provider.return_value.__enter__.return_value.preflight.return_value = {'model_listed':True}
            self.assertEqual(live_preflight.main(), 0)
            provider.return_value.__enter__.return_value.generate.assert_not_called()
            store.status.return_value['corpus_fingerprint'] = 'stale'
            self.assertEqual(live_preflight.main(), 1)

    def test_secrets_depend_on_explicit_backend(self):
        with patch.dict(os.environ, {'NEBIUS_API_KEY':'synthetic'}, clear=True):
            live_preflight.require_secret_environment('chroma')
            with self.assertRaises(live_preflight.LivePreflightError):
                live_preflight.require_secret_environment('pinecone')
            with self.assertRaises(live_preflight.LivePreflightError):
                live_preflight.require_secret_environment('unknown')

    def test_pinecone_graph_preflight_checks_local_assets_without_external_setup(self):
        with patch.dict(os.environ, VECTOR_STORE='pinecone', NORTHSTAR_UI_ENABLE_GRAPH='1',
                        NORTHSTAR_CONTROLLED_LIVE='1', NORTHSTAR_PRIVATE_PREVIEW_CONFIRMED='1',
                        NORTHSTAR_UI_ENABLE_LIVE='0', NORTHSTAR_UI_ENABLE_PINECONE='0'), \
             patch.object(Path, 'cwd', return_value=self.root), \
             patch('northstar_atlas.live_budget.configured_budget') as factory, \
             patch('northstar_atlas.live_graph.verify_graph_assets') as assets, \
             patch.object(live_preflight, 'require_secret_environment') as secrets, \
             patch.object(live_preflight, 'require_private_preview') as access, \
             patch('northstar_atlas.qa.open_retriever') as retrieval, \
             patch('northstar_atlas.nebius.load_settings') as credentials, \
             patch('northstar_atlas.nebius.NebiusProvider') as provider, redirect_stdout(io.StringIO()):
            self.assertEqual(live_preflight.main(), 0)
            factory.return_value.check_ready.assert_called_once_with(profile='llm_routing_grading_semantic_v1')
            assets.assert_called_once_with(self.root, backend='pinecone')
            secrets.assert_called_once_with('pinecone')
            for call in (access, retrieval, credentials, provider): call.assert_not_called()

    def test_graph_preflight_keeps_operator_gates_and_backend_validation(self):
        for changes in ({'NORTHSTAR_CONTROLLED_LIVE':'0'},
                        {'NORTHSTAR_PRIVATE_PREVIEW_CONFIRMED':'0'}, {'VECTOR_STORE':'unknown'}):
            with self.subTest(changes=changes), \
                 patch.dict(os.environ, dict(VECTOR_STORE='pinecone', NORTHSTAR_UI_ENABLE_GRAPH='1',
                    NORTHSTAR_CONTROLLED_LIVE='1', NORTHSTAR_PRIVATE_PREVIEW_CONFIRMED='1') | changes), \
                 patch.object(Path, 'cwd', return_value=self.root), \
                 patch('northstar_atlas.live_budget.configured_budget'), \
                 patch('northstar_atlas.live_graph.verify_graph_assets') as assets, \
                 patch.object(live_preflight, 'require_secret_environment') as secrets, \
                 patch.object(live_preflight, 'require_private_preview') as access, \
                 redirect_stdout(io.StringIO()):
                self.assertEqual(live_preflight.main(), 1)
                for call in (assets, secrets, access): call.assert_not_called()


class LauncherTests(unittest.TestCase):
    def test_separate_launcher_keeps_gates_closed_until_preflight_and_preserves_backend(self):
        root = checkout(self)
        (root/'.local/tmp').mkdir(parents=True)
        (root/'scripts').mkdir()
        source = ROOT/'scripts/run_replit_live.sh'
        self.assertTrue(source.is_file())
        shutil.copyfile(source, root/'scripts/run_replit_live.sh')
        (root/'.venv/bin').mkdir(parents=True)
        python = root/'.venv/bin/python'
        python.write_text('#!'+sys.executable+'\n'+'''
import json, os, sys
from pathlib import Path
if 'northstar_atlas.live_preflight' in sys.argv:
    assert os.environ['NORTHSTAR_UI_ENABLE_LIVE']=='0'
    assert os.environ['NORTHSTAR_UI_ENABLE_PINECONE']=='0'
    raise SystemExit(int(os.environ.get('PREFLIGHT_EXIT','0')))
if 'streamlit' in sys.argv:
    Path('.local/launch.json').write_text(json.dumps({'args':sys.argv,
        'controlled':os.environ.get('NORTHSTAR_CONTROLLED_LIVE'),
        'live':os.environ.get('NORTHSTAR_UI_ENABLE_LIVE'),
        'pinecone':os.environ.get('NORTHSTAR_UI_ENABLE_PINECONE'),
        'backend':os.environ.get('VECTOR_STORE','chroma')}))
else:
    os.execv(sys.executable,[sys.executable,*sys.argv[1:]])
''')
        python.chmod(0o700)
        env = {k:v for k,v in os.environ.items() if not k.startswith(('NORTHSTAR_', 'NEBIUS_', 'PINECONE_'))}
        env.update(REPLIT_DEV_DOMAIN='fixture.replit.dev', PYTHONDONTWRITEBYTECODE='1')
        for backend in ('chroma','pinecone'):
            result = subprocess.run(['bash','scripts/run_replit_live.sh'],cwd=root,
                                    env=env|{'VECTOR_STORE':backend},capture_output=True,text=True,timeout=15)
            self.assertEqual(result.returncode,0,result.stderr)
            capture=json.loads((root/'.local/launch.json').read_text())
            self.assertEqual(capture['backend'],backend)
            self.assertEqual(capture['controlled'],'1');self.assertEqual(capture['live'],'1')
            self.assertEqual(capture['pinecone'],'1' if backend=='pinecone' else '0')
            for flag in ('--server.enableCORS=true','--server.enableXsrfProtection=true','--client.showErrorDetails=none'):
                self.assertIn(flag,capture['args'])
        before=(root/'.local/launch.json').read_bytes()
        failed=subprocess.run(['bash','scripts/run_replit_live.sh'],cwd=root,
                              env=env|{'PREFLIGHT_EXIT':'1'},capture_output=True,timeout=15)
        self.assertNotEqual(failed.returncode,0)
        self.assertEqual((root/'.local/launch.json').read_bytes(),before)
