"""Interactive batch UI with synthetic authority and mocked HTTP only."""
import os
import sys
from pathlib import Path
import unittest
from unittest.mock import patch

import streamlit as st
from streamlit.testing.v1 import AppTest
import test_demo_batch as fixtures


class BatchUITests(unittest.TestCase):
    def setUp(self):
        # AppTest installs a temporary __main__; restore it before later spawn tests.
        original_main = sys.modules['__main__']
        self.addCleanup(sys.modules.__setitem__, '__main__', original_main)
        self.fixture = fixtures.BatchTests('runTest')
        self.fixture.setUp(); self.addCleanup(self.fixture.doCleanups)
        self.preview, self.owner = self.fixture.enable_runtime()
        self.root = self.fixture.root
        self.enterContext(patch.object(Path, 'cwd', return_value=self.root))
        self.enterContext(patch.dict(os.environ, NORTHSTAR_UI_ENABLE_LIVE='1', NORTHSTAR_UI_ENABLE_PINECONE='1'))
        self.configured = self.enterContext(patch('northstar_atlas.live_budget.configured_budget', return_value=self.fixture.budget))
        st.cache_resource.clear(); self.addCleanup(st.cache_resource.clear)

    def app(self):
        app = AppTest.from_file(str(self.root/'src/northstar_atlas/ui/app.py')).run(timeout=20)
        self.assertFalse(app.exception)
        return app

    def test_offline_default_then_five_arbitrary_questions_and_exhaustion(self):
        app = self.app()
        self.assertEqual(app.radio(key='experience').value, 'Offline demo')
        self.assertEqual(app.selectbox(key='backend').value, 'chroma')
        self.configured.assert_not_called()
        app.radio(key='experience').set_value('Live Q&A').run()
        self.assertNotIn('workflow', [x.key for x in app.selectbox])
        self.assertEqual(app.selectbox(key='backend').value, 'pinecone')
        self.assertTrue(app.chat_input[0].disabled)
        app.checkbox(key='paid_opt_in').check().run()
        for i in range(5):
            self.assertTrue(any(f'{5-i} of 5' in x.value for x in app.caption))
            self.assertFalse(app.chat_input[0].disabled)
            app.selectbox(key='mode').set_value('draft_review' if i % 2 else 'approved_only').run()
            app.chat_input[0].set_value(f'Arbitrary enterprise policy question {i}?').run(timeout=20)
            self.assertFalse(app.exception)
            self.assertEqual(len(self.fixture.sent), i+1)
            turn = app.session_state['conversation'].history[-1]
            self.assertEqual(turn.view.kind, 'answer')
            self.assertTrue(turn.view.citations)
            self.assertTrue(any(x.label == 'Evidence details' for x in app.expander))
            app.run(); self.assertEqual(len(self.fixture.sent), i+1)
        self.assertTrue(app.chat_input[0].disabled)
        self.assertTrue(any('exhausted' in x.value for x in app.warning))
        app.button(key='clear_history').click().run()
        self.assertTrue(app.chat_input[0].disabled)
        self.assertEqual(len(self.fixture.sent), 5)

    def test_bad_activation_disables_input_without_provider_or_demo_regression(self):
        with patch.dict(os.environ, NORTHSTAR_DEMO_BATCH_RUN_APPROVAL='invalid'):
            app = self.app()
            app.radio(key='experience').set_value('Live Q&A').run()
            app.checkbox(key='paid_opt_in').check().run()
            self.assertTrue(app.chat_input[0].disabled)
            self.assertTrue(any('unavailable' in x.value for x in app.warning))
            self.assertEqual(self.fixture.sent, [])
            app.radio(key='experience').set_value('Offline demo').run()
            app.button(key='run_demo').click().run()
            self.assertTrue(any('authored examples' in x.value for x in app.info))
            self.assertEqual(self.fixture.sent, [])
