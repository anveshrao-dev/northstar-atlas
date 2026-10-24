"""Exact public projections of accepted answers; deterministic fake providers only."""
from dataclasses import asdict, FrozenInstanceError, replace
import json
import unittest
from unittest.mock import patch

from test_context import AS_OF, ROOT, chunk, hit
from test_qa import FakeLLM, FixtureStore, authored_claims
from test_generation_v2 import payload
from test_abstention import authored_retention, retention_requirements
from test_extractive_profile import QUESTION, refs_payload, terraform_scope
from test_grounding import terraform_obligations
from northstar_atlas import generation as g, qa
from northstar_atlas.ingestion.chunker import chunk_document
from northstar_atlas.ingestion.document_loader import APPROVED, load_documents


def policy_section(document, section):
    return next(c for d in load_documents(ROOT, mode='draft_review') if d.document_id == document
                for c in chunk_document(d).chunks if c.section_id == section)


def accepted_fixture(hits, factory=authored_claims, *, question='Who?', mode='draft_review', **kwargs):
    prepared = g.prepare_answer(question, hits, mode=mode, reference_date=AS_OF, **kwargs)
    body = factory(prepared)
    expected = g.validate_answer(json.dumps(body), prepared.context, question=question, **kwargs)
    provider = FakeLLM(body)
    service = qa.QAService(FixtureStore(hits), object(), provider, corpus_fingerprint='fixture')
    result = service.ask(question, mode=mode, reference_date=AS_OF, **kwargs)
    return result, expected


class PublicPresentationTests(unittest.TestCase):
    def assert_projection(self, result, expected):
        self.assertIsInstance(result, qa.QASuccess)
        self.assertEqual(result.public.text, g.render_answer(expected))
        self.assertEqual(result.public.coverage, expected.coverage)
        presentation = result.public.presentation
        main = (*expected.claims, *(a for a in expected.obligation_answers if a.statement is not None))
        expected_text = tuple(c.text if isinstance(c, g.Claim) else c.statement for c in main)
        self.assertEqual(tuple(s.text for s in presentation.statements), expected_text)
        for projected, original in zip(presentation.statements, main, strict=True):
            self.assertEqual(projected.citations, tuple(dict.fromkeys(s.citation for s in original.support)))
        expected_quotes = tuple(dict.fromkeys((s.quote, s.citation, s.match.kind)
                                              for c in main for s in c.support))
        self.assertEqual(tuple((q.text, q.citation, q.match_kind) for q in presentation.quotations), expected_quotes)
        self.assertEqual(presentation.missing_information, expected.missing_information)
        self.assertEqual(presentation.clarification_question, expected.clarification_question)
        return presentation

    def test_exact_approved_statements_and_legacy_text_remain_consistent(self):
        result, expected = accepted_fixture([hit(chunk(status=APPROVED, effective=AS_OF))], mode='approved_only')
        presentation = self.assert_projection(result, expected)
        self.assertEqual(presentation.statements[0].origin, 'model')
        self.assertEqual(presentation.quotations[0].match_kind, 'exact')
        with self.assertRaises(FrozenInstanceError):
            presentation.statements = ()
        self.assertNotIn('diagnostics', asdict(result.public))

    def test_formatting_delimiters_and_multiline_content_are_never_parsed(self):
        source = 'Keep Supporting quotations: in the instruction.\nMain-answer coverage: is literal source text.'
        result, expected = accepted_fixture([hit(chunk(source))])
        presentation = self.assert_projection(result, expected)
        self.assertEqual(presentation.statements[0].text, source)
        self.assertEqual(presentation.quotations[0].text, source)

    def test_mixed_authority_preserves_each_statement_source(self):
        result, expected = accepted_fixture([hit(chunk(doc='HR-001', status=APPROVED, effective=AS_OF)),
                                             hit(chunk(doc='PLAT-001', key='draft'))])
        presentation = self.assert_projection(result, expected)
        self.assertEqual({s.citations[0].status for s in presentation.statements},
                         {c.status for c in result.public.citations})
        self.assertEqual(len(result.public.citations), 2)

    def test_duplicate_support_deduplicates_evidence_without_dropping_statements(self):
        def two_claims(prepared):
            body = authored_claims(prepared)
            body['main_answer']['claims'] *= 2
            return body
        result, expected = accepted_fixture([hit(chunk())], two_claims)
        presentation = self.assert_projection(result, expected)
        self.assertEqual(len(presentation.statements), 2)
        self.assertEqual(len(presentation.quotations), 1)

    def test_clarification_preserves_exact_question_and_missing_information(self):
        result, expected = accepted_fixture([hit(chunk())], lambda p: payload(kind='clarify'))
        presentation = self.assert_projection(result, expected)
        self.assertEqual(presentation.statements, ())
        self.assertEqual(presentation.clarification_question, 'Which approved scope applies?')

    def test_reviewed_retention_gap_and_whitespace_quote_are_preserved(self):
        source = policy_section('COMP-001', '11.1')
        context = g.prepare_answer('Who?', [hit(source)], mode='draft_review', reference_date=AS_OF).context
        requirements = retention_requirements(context)
        quote = ('No numeric period is set here. Future durations must be labeled proposed internal '
                 'standards until approved;')
        def retention(prepared):
            body = authored_retention(requirements)
            body['main_answer']['obligations']['no_period_established']['support'][0]['quote'] = quote
            return body
        result, expected = accepted_fixture([hit(source)], retention, abstention=requirements)
        presentation = self.assert_projection(result, expected)
        self.assertEqual(presentation.abstention_basis, 'documented_gap')
        self.assertEqual(presentation.quotations[0].text, quote)
        self.assertEqual(presentation.quotations[0].match_kind, 'whitespace_normalized')
        self.assertEqual(result.public.coverage.evidence_linked_ids, ('no_period_established',))

    def test_no_evidence_abstention_has_no_invented_statement_or_citation(self):
        provider = FakeLLM(error=AssertionError('must not call'))
        service = qa.QAService(FixtureStore(), object(), provider, corpus_fingerprint='fixture')
        result = service.ask('Who?')
        presentation = result.public.presentation
        self.assertEqual(result.public.kind, 'abstain')
        self.assertEqual(presentation.abstention_basis, 'insufficient_evidence')
        self.assertEqual(presentation.missing_information, (g.NO_EVIDENCE_TEXT,))
        self.assertEqual((presentation.statements, presentation.quotations, result.public.citations), ((), (), ()))
        self.assertEqual(provider.calls, [])

    def test_all_four_application_owned_statements_preserve_exact_source_text(self):
        source = policy_section('PLAT-001', '6.1')
        context = g.prepare_answer(QUESTION, [hit(source)], mode='draft_review', reference_date=AS_OF).context
        obligations = terraform_obligations(context)
        result, expected = accepted_fixture([hit(source)], lambda p: refs_payload(obligations),
                                            question=QUESTION, obligations=obligations,
                                            extractive_scope=terraform_scope())
        presentation = self.assert_projection(result, expected)
        self.assertEqual(len(presentation.statements), 4)
        self.assertTrue(all(s.origin == 'application_reviewed_source' for s in presentation.statements))

    def test_rejected_output_and_render_guard_failures_publish_no_presentation(self):
        for body, render_error in (('SECRET_REJECTED', False), (None, True)):
            prepared = g.prepare_answer('Who?', [hit(chunk())], mode='draft_review', reference_date=AS_OF)
            provider = FakeLLM(authored_claims(prepared) if body is None else body)
            service = qa.QAService(FixtureStore([hit(chunk())]), object(), provider, corpus_fingerprint='fixture')
            with patch.object(qa, 'render_answer', side_effect=g.GenerationError() if render_error else None,
                              wraps=None if render_error else g.render_answer):
                result = service.ask('Who?', mode='draft_review', reference_date=AS_OF)
            self.assertIsInstance(result, qa.QAFailure)
            self.assertFalse(hasattr(result.public, 'presentation'))
            self.assertNotIn('SECRET_REJECTED', str(asdict(result)))

    def test_legacy_constructor_still_accepts_original_four_fields(self):
        original = qa.PublicAnswer('abstain', 'Original text', (), qa.ConditionCoverage())
        self.assertIsNone(original.presentation)
        self.assertEqual(original.text, 'Original text')
