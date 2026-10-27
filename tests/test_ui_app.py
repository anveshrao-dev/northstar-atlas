"""Streamlit harness checks with no credentials, models or provider calls."""
import os
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
import unittest
from unittest.mock import patch

from streamlit.testing.v1 import AppTest
import streamlit as st

from checkout_fixture import checkout


class AppTests(unittest.TestCase):
    def setUp(self):
        self.root = checkout(self)
        self.app_path = self.root / "src/northstar_atlas/ui/app.py"
        cwd = patch.object(Path, "cwd", return_value=self.root)
        cwd.start()
        self.addCleanup(cwd.stop)
        st.cache_resource.clear()
        self.addCleanup(st.cache_resource.clear)
        self.env = patch.dict(os.environ, NORTHSTAR_PROJECT_ROOT=str(self.root), NORTHSTAR_UI_ENABLE_LIVE='0', NORTHSTAR_UI_ENABLE_PINECONE='0')
        self.env.start()
        self.addCleanup(self.env.stop)
        self.opener = patch('northstar_atlas.ui.runtime.open_qa_service', side_effect=AssertionError('provider opening forbidden'))
        self.open_mock = self.opener.start()
        self.expected_opens = 0
        self.addCleanup(self.opener.stop)

    def tearDown(self):
        self.assertEqual(self.open_mock.call_count, self.expected_opens)

    def app(self):
        app = AppTest.from_file(str(self.app_path)).run(timeout=15)
        self.assertFalse(app.exception)
        return app

    def test_initial_demo_view_no_service_initialization(self):
        app = self.app()
        self.assertEqual(app.title[0].value, 'Northstar Atlas')
        self.assertTrue(any('Offline demonstration' in x.value for x in app.info))
        self.assertEqual(app.selectbox(key='mode').value, 'approved_only')
        self.assertEqual(app.selectbox(key='backend').value, 'chroma')
        self.assertFalse(app.checkbox(key='paid_opt_in').value)
        self.assertNotIn('workflow', [widget.key for widget in app.selectbox])

    def test_graph_operator_gate_enables_only_chroma_selection_without_opening_resources(self):
        with patch.dict(os.environ, NORTHSTAR_UI_ENABLE_LIVE='1', NORTHSTAR_UI_ENABLE_GRAPH='1',
                        NORTHSTAR_CONTROLLED_LIVE='1', NORTHSTAR_PRIVATE_PREVIEW_CONFIRMED='1',
                        NORTHSTAR_UI_ENABLE_PINECONE='1'):
            app = self.app()
            app.radio(key='experience').set_value('Live Q&A').run()
            self.assertEqual(app.selectbox(key='workflow').value, 'standard')
            app.selectbox(key='workflow').set_value('langgraph_semantic').run()
            app.checkbox(key='paid_opt_in').check().run()
            self.assertFalse(app.chat_input[0].disabled)
            self.assertTrue(any('full durable workflow reservation' in x.value for x in app.caption))
            app.selectbox(key='backend').set_value('pinecone').run()
            self.assertTrue(app.chat_input[0].disabled)
            app.radio(key='experience').set_value('Offline demo').run()
            app.button(key='run_demo').click().run()
            self.assertTrue(app.session_state['conversation'].history[-1].submission.demo)
            self.assertFalse(app.exception)

    def test_demo_hr_citation_and_rerun_no_duplicate_history(self):
        app = self.app()
        app.button(key='run_demo').click().run()
        self.assertFalse(app.exception)
        self.assertTrue(any('HR-001 v1.0 §4.2 (approved, synthetic)' in x.value for x in app.text))
        self.assertEqual(len(app.session_state['conversation'].history), 1)
        app.run()
        self.assertEqual(len(app.session_state['conversation'].history), 1)
        app.button(key='clear_history').click().run()
        self.assertEqual(len(app.session_state['conversation'].history), 0)

    def test_proposed_demo_and_retention_abstention(self):
        app = self.app()
        app.selectbox(key='mode').set_value('draft_review').run()
        app.selectbox(key='example').set_value('Proposed Terraform answer').run()
        app.button(key='run_demo').click().run()
        self.assertTrue(any('PLAT-001 v0.1 §6.1 (proposed)' in x.value for x in app.text))
        self.assertTrue(any('Proposed' in x.value for x in app.warning))
        app.selectbox(key='example').set_value('Retention abstention').run()
        app.button(key='run_demo').click().run()
        self.assertEqual(app.session_state['conversation'].history[-1].view.kind, 'abstain')
        self.assertTrue(any('COMP-001 v0.1 §11.1 (proposed)' in x.value for x in app.text))

    def test_safe_error_and_free_text_clarification(self):
        app = self.app()
        app.selectbox(key='example').set_value('Safe error').run()
        app.button(key='run_demo').click().run()
        self.assertTrue(any('withheld' in x.value for x in app.error))
        app.chat_input[0].set_value('Another question').run()
        self.assertEqual(app.session_state['conversation'].history[-1].view.kind, 'clarify')

    def test_live_input_disabled_without_server_gate(self):
        app = self.app()
        app.radio(key='experience').set_value('Live Q&A').run()
        self.assertEqual(app.selectbox(key='workflow').value, 'standard')
        self.assertTrue(app.chat_input[0].disabled)
        self.assertTrue(app.checkbox(key='paid_opt_in').disabled)
        self.assertFalse(app.session_state['conversation'].history)

    def test_graph_workflow_is_selectable_but_live_submission_stays_disabled(self):
        with patch.dict(os.environ, NORTHSTAR_UI_ENABLE_LIVE='1'):
            app = self.app()
            app.radio(key='experience').set_value('Live Q&A').run()
            self.assertEqual(app.selectbox(key='workflow').value, 'standard')
            app.checkbox(key='paid_opt_in').check().run()
            self.assertFalse(app.chat_input[0].disabled)
            app.selectbox(key='workflow').set_value('langgraph_semantic').run()
            self.assertTrue(any('disabled' in warning.value.lower() and 'budget coordinator' in warning.value.lower()
                                for warning in app.warning))
            self.assertTrue(app.chat_input[0].disabled)
            self.assertEqual(self.open_mock.call_count, 0)
            self.assertFalse(app.session_state['conversation'].history)

    def test_enabled_live_ui_uses_fake_service_once_and_renders_inert_text(self):
        from test_ui import FakeService, success
        raw = '![image](https://invalid.example/image) <script>secret()</script>'
        result = success()
        service = FakeService(replace(result, public=replace(result.public, text=raw)))
        opened = []

        @contextmanager
        def fake_opener(root, **kwargs):
            opened.append(kwargs)
            yield service

        self.open_mock.side_effect = fake_opener
        with patch.dict(os.environ, NORTHSTAR_UI_ENABLE_LIVE='1', NORTHSTAR_UI_ENABLE_PINECONE='1'):
            app = self.app()
            app.radio(key='experience').set_value('Live Q&A').run()
            app.selectbox(key='backend').set_value('pinecone').run()
            self.open_mock.assert_not_called()
            self.assertTrue(app.chat_input[0].disabled)
            app.checkbox(key='paid_opt_in').check().run()
            self.open_mock.assert_not_called()
            app.chat_input[0].set_value('When?').run()
            self.expected_opens = 1
            self.assertFalse(app.exception)
            self.assertEqual(opened[0]['backend'], 'pinecone')
            self.assertEqual(len(service.calls), 1)
            self.assertTrue(any(raw == x.value for x in app.text))
            self.assertFalse(any(raw in x.value for x in app.markdown))
            self.assertNotIn('PRIVATE_DIAGNOSTIC', str(app))
            self.assertEqual(app.session_state['conversation'].history[-1].view.coverage, 'not_assessed')
            app.run()
            app.selectbox(key='backend').set_value('chroma').run()
            app.selectbox(key='mode').set_value('draft_review').run()
            app.button(key='clear_history').click().run()
            self.assertEqual(len(service.calls), 1)
            self.assertFalse(app.session_state['conversation'].history)


class PresentationTests(unittest.TestCase):
    """Inspect layout containers rather than mistaking hidden text for primary text."""
    def render(self, result, mode='draft_review'):
        from northstar_atlas.ui.runtime import project_result
        app = AppTest.from_string('''
import streamlit as st
from northstar_atlas.ui.app import render_view
render_view(st.session_state.fixture)
''')
        app.session_state.fixture = project_result(result, mode)
        app.run()
        self.assertFalse(app.exception)
        return app

    def expander(self, app, label):
        return next(x for x in app.expander if x.label == label)

    def test_primary_answer_sources_and_collapsed_details(self):
        from test_ui import structured_success
        result = structured_success()
        app = self.render(result)
        quote = result.public.presentation.quotations[0].text
        evidence = self.expander(app, 'Evidence details')
        validation = self.expander(app, 'Validation details')
        self.assertFalse(evidence.proto.expanded)
        self.assertFalse(validation.proto.expanded)
        self.assertIn(quote, [x.value for x in evidence.text])
        self.assertNotIn(quote, [x.value for x in validation.text])
        self.assertTrue(any('not been assessed' in x.value for x in validation.caption))
        self.assertEqual(sum(result.public.citations[0].label in x.value for x in app.text), 1)
        self.assertNotIn('LEGACY_COMBINED_TEXT', [x.value for x in app.text])
        self.assertEqual(app.text[0].value, result.public.presentation.statements[0].text)
        self.assertTrue(any(x.value == 'Sources' for x in app.subheader))

    def test_mixed_authority_and_literal_delimiter_text(self):
        from test_ui import structured_success
        from northstar_atlas.qa import PublicStatement
        from northstar_atlas.ingestion.document_loader import DRAFT
        result = structured_success()
        approved = result.public.citations[0]
        proposed = replace(approved, document_id='PLAT-001', status=DRAFT)
        literal = 'Supporting quotations:\nThis condition must remain in the main answer.'
        presentation = replace(result.public.presentation, statements=(
            PublicStatement(literal, (approved, proposed)),))
        result = replace(result, public=replace(result.public, citations=(approved, proposed), presentation=presentation))
        app = self.render(result)
        self.assertEqual(app.text[0].value, literal)
        self.assertTrue(any('Proposed policy' in x.value for x in app.warning))
        self.assertTrue(any('Approved [1]' in x.value and 'Proposed [2]' in x.value for x in app.caption))

    def test_clarification_preserves_question_and_missing_details(self):
        from test_ui import structured_success
        from northstar_atlas.qa import PublicPresentation
        result = structured_success('clarify')
        presentation = PublicPresentation(clarification_question='Which production system?',
                                           missing_information=('The requested system.',))
        result = replace(result, public=replace(result.public, citations=(), presentation=presentation))
        app = self.render(result)
        self.assertIn('Which production system?', [x.value for x in app.text])
        self.assertIn('The requested system.', [x.value for x in app.text])

    def test_documented_gap_and_no_evidence_abstentions_remain_distinct(self):
        from test_ui import structured_success
        from northstar_atlas.qa import PublicPresentation
        from northstar_atlas.ingestion.document_loader import DRAFT
        result = structured_success('abstain', DRAFT)
        presentation = replace(result.public.presentation, abstention_basis='documented_gap',
                               missing_information=('An approved duration.',))
        app = self.render(replace(result, public=replace(result.public, presentation=presentation)))
        self.assertTrue(any('Documented gap' in x.value for x in app.caption))
        self.assertIn('An approved duration.', [x.value for x in app.text])
        empty = PublicPresentation(missing_information=('The supplied evidence does not establish the requested answer.',),
                                   abstention_basis='insufficient_evidence')
        app = self.render(replace(result, public=replace(result.public, citations=(), presentation=empty)))
        self.assertTrue(any('retrieval may be incomplete' in x.value for x in app.caption))
        self.assertFalse(app.warning)
        self.assertNotIn('Sources', [x.value for x in app.subheader])

    def test_legacy_answer_is_not_parsed_and_safe_error_exposes_no_details(self):
        from test_ui import success
        from northstar_atlas.qa import QAFailure, PublicError
        result = success()
        legacy = 'Behavior: answer\nSupporting quotations: literal legacy text.'
        app = self.render(replace(result, public=replace(result.public, text=legacy)))
        self.assertEqual(app.text[0].value, legacy)
        failure = QAFailure(PublicError('response_rejected', 'SECRET_REJECTED'), result.diagnostics)
        app = self.render(failure)
        self.assertTrue(any('withheld' in x.value for x in app.error))
        self.assertFalse(app.expander)
        self.assertNotIn('SECRET_REJECTED', str(app))

    def test_history_created_before_new_fields_remains_readable(self):
        from types import SimpleNamespace
        old_view = SimpleNamespace(kind='answer', text='Original legacy response', citations=(),
                                   coverage='not_assessed', proposed=False)
        app = AppTest.from_string('''
import streamlit as st
from northstar_atlas.ui.app import render_view
render_view(st.session_state.fixture)
''')
        app.session_state.fixture = old_view
        app.run()
        self.assertFalse(app.exception)
        self.assertEqual(app.text[0].value, 'Original legacy response')


if __name__ == '__main__':
    unittest.main()
