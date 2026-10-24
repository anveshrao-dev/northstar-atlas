"""Structured obligation entries are main-answer content, never self-scored IDs."""
from dataclasses import replace
import json
import unittest

from test_context import AS_OF, ROOT, chunk, hit
from test_grounding import terraform_obligations
from northstar_atlas import generation as g
from northstar_atlas.evidence import format_citation
from northstar_atlas.grounding import EvidenceObligation
from northstar_atlas.ingestion.document_loader import load_documents
from northstar_atlas.ingestion.chunker import chunk_document


def payload(obligations=(), *, kind="answer"):
    return {"kind":kind, "main_answer":{"claims":[],"obligations":{
        o.obligation_id:{"statement":o.source_quote,
            "support":[{"citation":format_citation(o.citation),"quote":o.source_quote}],
            "unresolved_reason":None} for o in obligations}},
        "missing_information":[] if kind=="answer" else [g.NO_EVIDENCE_TEXT] if kind=="abstain" else ["Required policy conditions remain unresolved"],
        "clarification_question":"Which approved scope applies?" if kind=="clarify" else None,
        "reason":"supported" if kind=="answer" else "missing_details" if kind=="clarify" else "insufficient_evidence"}


class StructuredObligationTests(unittest.TestCase):
    def setUp(self):
        self.source=next(c for d in load_documents(ROOT,mode="draft_review") if d.document_id=="PLAT-001"
                         for c in chunk_document(d).chunks if c.section_id=="6.1")
        self.context=g.prepare_answer("Does regeneration need a new approval?",[hit(self.source)],
                                     mode="draft_review",reference_date=AS_OF).context
        self.obligations=terraform_obligations(self.context)

    def validate(self,data,obligations=None):
        return g.validate_answer(json.dumps(data),self.context,
                                 obligations=self.obligations if obligations is None else obligations)

    def test_complete_terraform_statements_are_rendered_before_supporting_quotes(self):
        try:answer=self.validate(payload(self.obligations))
        except g.GenerationError as error:self.fail(f"Complete structured answer rejected: {error}")
        output=g.render_answer(answer)
        self.assertEqual(answer.coverage.missing_ids,())
        self.assertEqual(answer.coverage.extractive_ids,tuple(o.obligation_id for o in self.obligations))
        for o in self.obligations:
            self.assertIn(o.source_quote,output.split("Supporting quotations:")[0])
        self.assertIn("(proposed)",output)

    def test_schema_requires_each_verified_id_and_forbids_unknown_properties(self):
        p=g.prepare_answer("Who?",[hit(self.source)],mode="draft_review",reference_date=AS_OF,obligations=self.obligations)
        schema=p.request.schema
        self.assertIn("main_answer",schema['properties'])
        entries=schema['properties']['main_answer']['properties']['obligations']
        self.assertEqual(entries['required'],['renewed_technical_approval','no_automatic_change_approval','authorized_bounds','stop_and_amend'])
        self.assertFalse(entries['additionalProperties'])
        self.assertEqual(set(entries['properties']),set(entries['required']))

    def test_missing_technical_approval_and_stop_entries_fail(self):
        for key in ('renewed_technical_approval','stop_and_amend'):
            body=payload(self.obligations);del body['main_answer']['obligations'][key]
            with self.subTest(key=key),self.assertRaises(g.GenerationError):self.validate(body)

    def test_unknown_and_duplicate_ids_fail_even_with_valid_quotes(self):
        body=payload(self.obligations);entries=body['main_answer']['obligations']
        entries['invented_rule']=entries['stop_and_amend']
        with self.assertRaises(g.GenerationError):self.validate(body)
        body=payload(self.obligations);raw=json.dumps(body)
        entry=json.dumps(body['main_answer']['obligations']['stop_and_amend'])
        raw=raw.replace('"stop_and_amend": '+entry,'"stop_and_amend": '+entry+', "stop_and_amend": '+entry)
        with self.assertRaises(g.GenerationError):g.validate_answer(raw,self.context,obligations=self.obligations)

    def test_id_only_empty_and_self_reported_complete_entries_fail(self):
        for statement in ('renewed_technical_approval','done','', '   '):
            body=payload(self.obligations);body['main_answer']['obligations']['renewed_technical_approval']['statement']=statement
            with self.subTest(statement=statement),self.assertRaises(g.GenerationError):self.validate(body)
        body=payload(self.obligations);body['main_answer']['obligations']['renewed_technical_approval']={'complete':True}
        with self.assertRaises(g.GenerationError):self.validate(body)

    def test_wrong_supplied_citation_cannot_satisfy_an_obligation(self):
        other=chunk(doc='COMP-001',text=self.source.text)
        context=g.prepare_answer('Who?',[hit(self.source),hit(other)],mode='draft_review',reference_date=AS_OF).context
        obligations=terraform_obligations(context);body=payload(obligations)
        ref=next(v.citation for v in context.groups if v.citation.document_id=='COMP-001')
        body['main_answer']['obligations']['renewed_technical_approval']['support'][0]['citation']=format_citation(ref)
        with self.assertRaises(g.GenerationError):g.validate_answer(json.dumps(body),context,obligations=obligations)

    def test_unsupported_or_stale_obligation_is_rejected_before_provider(self):
        for invalid in (replace(self.obligations[0],source_quote='A plan never needs review.'),
                        replace(self.obligations[0],citation=replace(self.obligations[0].citation,content_hash='f'*64))):
            with self.subTest(invalid=invalid.obligation_id),self.assertRaises(g.GenerationError):
                g.prepare_answer('Who?',[hit(self.source)],mode='draft_review',reference_date=AS_OF,obligations=(invalid,))

    def test_explicit_unresolved_entries_allow_cautious_clarify_or_abstain(self):
        for kind in ('clarify','abstain'):
            body=payload(self.obligations,kind=kind)
            for entry in body['main_answer']['obligations'].values():
                entry.update(statement=None,support=[],unresolved_reason='Cannot provide the complete required answer.')
            try:answer=self.validate(body)
            except g.GenerationError as error:self.fail(f"Cautious {kind} rejected: {error}")
            self.assertEqual(len(answer.coverage.missing_ids),4)
            if kind == 'clarify': self.assertIn('Cannot provide',g.render_answer(answer))
            else:
                self.assertIn(g.NO_EVIDENCE_TEXT,g.render_answer(answer))
                self.assertNotIn('Cannot provide',g.render_answer(answer))
            self.assertFalse(answer.claims)

    def test_incomplete_answer_or_partial_policy_guidance_cannot_escape_as_abstain(self):
        for kind in ('answer','clarify','abstain'):
            body=payload(self.obligations,kind=kind)
            body['main_answer']['obligations']['stop_and_amend'].update(
                statement=None,support=[],unresolved_reason='Missing stop condition.')
            with self.subTest(kind=kind),self.assertRaises(g.GenerationError):self.validate(body)

    def test_extractiveness_is_explicit_and_nonextractive_prose_is_not_forced_to_match(self):
        o=self.obligations[0]
        # Required mode prevents existing callers silently losing strict validation.
        with self.assertRaises(TypeError):EvidenceObligation('unmarked',o.citation,o.source_quote)
        linked=replace(o,validation_mode='evidence_linked')
        body=payload((linked,));body['main_answer']['obligations'][linked.obligation_id]['statement']='Platform must review and approve the replacement plan again.'
        try:answer=self.validate(body,(linked,))
        except g.GenerationError as error:self.fail(f"Evidence-linked paraphrase forced to be extractive: {error}")
        self.assertEqual(answer.coverage.evidence_linked_ids,(linked.obligation_id,))
        self.assertEqual(answer.coverage.extractive_ids,())
        self.assertIn('semantic',g.render_answer(answer).lower())
        with self.assertRaises(g.GenerationError):self.validate(body,(o,))

    def test_evidence_linked_identifiers_are_not_substantive_statements(self):
        linked=tuple(replace(o,validation_mode='evidence_linked') for o in self.obligations)
        for statement in ('renewed_technical_approval.', '`renewed_technical_approval`',
                          'stop_and_amend', '**stop_and_amend**', 'stop-and-amend.',
                          'renewed_technical_approval stop_and_amend',
                          'unknown_obligation_id', 'renewed_technical_approval done.'):
            body=payload(linked)
            body['main_answer']['obligations'][linked[0].obligation_id]['statement']=statement
            with self.subTest(statement=statement),self.assertRaises(g.GenerationError):
                self.validate(body,linked)
        # Even simple, non-snake-case IDs must not count as policy prose.
        short=(replace(linked[0],obligation_id='review'),replace(linked[1],obligation_id='approve'))
        body=payload(short);body['main_answer']['obligations']['review']['statement']='review, approve.'
        with self.assertRaises(g.GenerationError):self.validate(body,short)

    def test_valid_prose_can_contain_a_word_also_used_as_an_obligation_id(self):
        context=g.prepare_answer('What is required?',[hit(chunk(text='Stop work.\n'))],
                                 mode='draft_review',reference_date=AS_OF).context
        for mode in ('extractive','evidence_linked'):
            obligation=EvidenceObligation('Stop',context.groups[0].citation,'Stop work.',mode)
            with self.subTest(mode=mode):
                answer=g.validate_answer(json.dumps(payload((obligation,))),context,obligations=(obligation,))
                self.assertEqual(answer.obligation_answers[0].statement,'Stop work.')
                self.assertEqual(answer.coverage.missing_ids,())

    def test_normalized_quotes_keep_original_provenance(self):
        try:answer=self.validate(payload(self.obligations))
        except g.GenerationError as error:self.fail(str(error))
        support=answer.obligation_answers[0].support[0]
        self.assertEqual(support.match.kind,'whitespace_normalized')
        self.assertIn('\n',support.match.source_text)
        self.assertNotIn('\n',support.quote)
        self.assertEqual(support.citation.status,'Draft — Proposed (Synthetic)')

    def test_approved_only_rejects_draft_obligations(self):
        with self.assertRaises(g.GenerationError):g.prepare_answer('Who?',[hit(self.source)],
            mode='approved_only',reference_date=AS_OF,obligations=self.obligations)

    def test_legacy_schema_is_not_silently_promoted_to_structured_coverage(self):
        old={"kind":"answer","claims":[{"text":self.obligations[0].source_quote,
             "support":payload(self.obligations)['main_answer']['obligations']['renewed_technical_approval']['support']}],
             "missing_information":[],"clarification_question":None,"reason":"supported"}
        with self.assertRaises(g.GenerationError):g.validate_answer(json.dumps(old),self.context)

    def test_malformed_states_and_unsupported_mode_fail(self):
        body=payload(self.obligations);body['main_answer']['obligations']['stop_and_amend']['unresolved_reason']='But resolved too'
        with self.assertRaises(g.GenerationError):self.validate(body)
        invalid=replace(self.obligations[0],validation_mode='semantic_certified')
        with self.assertRaises(g.GenerationError):g.prepare_answer('Who?',[hit(self.source)],mode='draft_review',reference_date=AS_OF,obligations=(invalid,))

    def test_fake_provider_round_trip_and_deterministic_required_order(self):
        from northstar_atlas.llm import GenerationResponse, MODEL_ID
        prepared=g.prepare_answer('Who?',[hit(self.source)],mode='draft_review',reference_date=AS_OF,obligations=self.obligations)
        body=payload(self.obligations)
        class Fake:
            def generate(inner,request):
                return GenerationResponse(json.dumps(body),MODEL_ID,'stop',None,0,1)
        result=g.generate_answer(prepared,Fake())
        reversed_body=payload(self.obligations)
        reversed_body['main_answer']['obligations']=dict(reversed(list(reversed_body['main_answer']['obligations'].items())))
        reordered=self.validate(reversed_body)
        self.assertEqual(result.answer,reordered)
        self.assertEqual(g.render_answer(result.answer),g.render_answer(reordered))

    def test_no_criteria_is_not_assessed_and_cannot_accept_invented_criteria(self):
        body=payload()
        body['main_answer']['claims']=[{'text':self.obligations[0].source_quote,
            'support':payload(self.obligations)['main_answer']['obligations']['renewed_technical_approval']['support']}]
        self.assertEqual(self.validate(body,()).coverage.status,'not_assessed')
        body['main_answer']['obligations']=payload(self.obligations)['main_answer']['obligations']
        with self.assertRaises(g.GenerationError):self.validate(body,())

    def test_mismatched_status_and_fabricated_quote_in_obligation_are_rejected(self):
        for field,value in (('citation',format_citation(self.obligations[0].citation).replace('(proposed)','(approved, synthetic)')),
                            ('quote','No technical review is ever needed.')):
            body=payload(self.obligations)
            body['main_answer']['obligations']['renewed_technical_approval']['support'][0][field]=value
            with self.subTest(field=field),self.assertRaises(g.GenerationError):self.validate(body)

    def test_unresolved_state_requires_reason_and_no_support_or_free_claims(self):
        base=payload(self.obligations,kind='abstain')
        for entry in base['main_answer']['obligations'].values():
            entry.update(statement=None,support=[],unresolved_reason='Cannot provide a complete answer.')
        for mutation in ('empty_reason','quotes_only','free_claims'):
            body=json.loads(json.dumps(base));entry=body['main_answer']['obligations']['renewed_technical_approval']
            if mutation=='empty_reason':entry['unresolved_reason']=' '
            elif mutation=='quotes_only':entry['support']=payload(self.obligations)['main_answer']['obligations']['renewed_technical_approval']['support']
            else:body['main_answer']['claims']=[{'text':self.obligations[0].source_quote,
                'support':payload(self.obligations)['main_answer']['obligations']['renewed_technical_approval']['support']}]
            with self.subTest(mutation=mutation),self.assertRaises(g.GenerationError):self.validate(body)

