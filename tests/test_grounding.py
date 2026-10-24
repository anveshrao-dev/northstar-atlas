"""Quotation and targeted obligation regressions; no models or network."""
from dataclasses import replace
from datetime import date
import json
from pathlib import Path
import unittest

from test_context import AS_OF, ROOT, chunk, hit
from northstar_atlas.evidence import (CitationReference, ContextPackage, EvidenceGroup,
                                     EvidenceSpan, format_citation)
from northstar_atlas.generation import prepare_answer, validate_answer, GenerationError


def package(text="A reviewer must\nreview and approve the plan.\n", **kwargs):
    return prepare_answer("What is required?",[hit(chunk(text,**kwargs))],
                          mode="draft_review",reference_date=AS_OF).context


def response(context, quote, main="A reviewer must review and approve the plan.", *, obligations=()):
    support=[{"citation":format_citation(context.groups[0].citation),"quote":quote}]
    return json.dumps({"kind":"answer","main_answer":{
        "claims":[] if obligations else [{"text":main,"support":support}],
        "obligations":{o.obligation_id:{"statement":main,"support":support,"unresolved_reason":None} for o in obligations}},
        "missing_information":[],"clarification_question":None,"reason":"supported"})


def saved_replay(number, folder=None):
    """Test-only rehydration of the immutable evidence/response used in the live run.

    Validation consumes citations/passages; source chunk records are not needed.
    This is NOT a runtime route for accepting arbitrary evidence snapshots.
    """
    folder=folder or ROOT/".local/m43-live"
    record=json.loads((folder/f"prepared-{number}.json").read_text())
    payload=record["evidence"]
    citations={v["reference_id"]:v for v in record["summary"]["citations"]}
    groups=[]
    for value in payload["approved_evidence"]+payload["proposed_evidence"]:
        data=dict(citations[value["reference_id"]]);data["chunk_ids"]=tuple(data["chunk_ids"])
        citation=CitationReference(**data)
        spans=tuple(EvidenceSpan(p["source_start"],p["source_end"],p["text"],tuple(p["chunk_ids"])) for p in value["passages"])
        groups.append(EvidenceGroup(citation,(),spans,value["direct_rank"],0))
    context=ContextPackage(tuple(groups),(),payload["mode"],date.fromisoformat(payload["reference_date"]),
                           json.dumps(payload),record["summary"]["input_token_bound"],32768,"saved-live-packet")
    raw=json.loads((folder/f"raw-response-{number}.json").read_text())["body"]["choices"][0]["message"]["content"]
    return context,raw


def terraform_obligations(context):
    """Reviewed, source-linked regression expectations; no runtime policy routing."""
    from northstar_atlas.grounding import EvidenceObligation
    ref=next(g.citation for g in context.groups if g.citation.document_id=="PLAT-001" and g.citation.section_id=="6.1")
    statements=(
      ("renewed_technical_approval","If Terraform detects drift or a plan must be regenerated, Platform investigates the cause, makes the necessary technical corrections, and repeats plan review and recorded technical approval."),
      ("no_automatic_change_approval","No separate Change Management approval is required solely because a plan was regenerated."),
      ("authorized_bounds","Platform may proceed after renewed technical validation and approval when the resulting changes remain within the authorized CR scope, assessed risk, deployment window, and other approval conditions."),
      ("stop_and_amend","If the result materially exceeds the approved scope or risk, or would exceed the window, stop and follow §4.3's amendment or escalation process."),
    )
    return tuple(EvidenceObligation(key,ref,text,validation_mode="extractive") for key,text in statements)


def corrected_fixture(context,obligations):
    """New authored fixture, never edits or pretends to be the hosted response."""
    body={"kind":"answer","main_answer":{"claims":[],"obligations":{}},"missing_information":[],"clarification_question":None,"reason":"supported"}
    for obligation in obligations:
        body["main_answer"]["obligations"][obligation.obligation_id]={"statement":obligation.source_quote,"support":[{
            "citation":format_citation(obligation.citation),"quote":obligation.source_quote}],"unresolved_reason":None}
    return json.dumps(body)


class QuotationTests(unittest.TestCase):
    def test_exact_match_records_original_quote_source_and_offsets(self):
        source="A reviewer must\nreview and approve the plan.\n";p=package(source)
        quote="A reviewer must\nreview and approve the plan."
        answer=validate_answer(response(p,quote),p);support=answer.claims[0].support[0]
        self.assertEqual(support.quote,quote)
        self.assertEqual(support.match.kind,"exact")
        self.assertEqual(support.match.source_text,quote)
        self.assertEqual((support.match.source_start,support.match.source_end),(100,100+len(quote)))
        self.assertEqual(p.groups[0].spans[0].text,source)

    def test_whitespace_only_match_preserves_both_original_strings(self):
        source="A reviewer\n\tmust  review and\u00a0approve the plan.\n";p=package(source)
        quote=" A reviewer must review and approve the plan. "
        support=validate_answer(response(p,quote),p).claims[0].support[0]
        self.assertEqual(support.match.kind,"whitespace_normalized")
        self.assertEqual(support.quote,quote)
        self.assertEqual(support.match.source_text,source.rstrip())
        self.assertEqual(p.groups[0].spans[0].text,source)

    def test_word_change_deletion_reordering_case_and_punctuation_rejected(self):
        p=package()
        for quote in ("A reviewer may review and approve the plan.","A reviewer must approve the plan.",
                      "A reviewer must approve and review the plan.","a reviewer must review and approve the plan.",
                      "A reviewer must review and approve the plan!","A reviewer must ... approve the plan."):
            with self.subTest(quote=quote),self.assertRaises(GenerationError):validate_answer(response(p,quote),p)

    def test_partial_words_and_numbers_are_not_valid_quotations(self):
        p=package("Disapprove after 15 days. Cannot grant admin_access.")
        for quote in ("approve after 15 days.","5 days.","not grant","admin"):
            with self.subTest(quote=quote),self.assertRaises(GenerationError):validate_answer(response(p,quote),p)

    def test_matches_do_not_bridge_source_gaps(self):
        p=package();g=p.groups[0]
        p=replace(p,groups=(replace(g,spans=(EvidenceSpan(10,22,"Review first",("a",)),EvidenceSpan(50,60,"then apply",("a",)))),))
        with self.assertRaises(GenerationError):validate_answer(response(p,"Review first then apply"),p)

    def test_numeric_boundaries_preserve_decimal_grouping_signs_and_ranges(self):
        cases=(("Wait 1.5 days.","5 days."),
               ("Wait 1.5\ndays.","5 days."),
               ("Limit 1,000 records.","000 records."),
               ("Rate -5 percent.","5 percent."),
               ("Rate −5\npercent.","5 percent."),
               ("Wait 1.5 days.","Wait 1"),
               ("Wait 1.5 days.","Wait 1."),
               ("Ratio 1/2.","Ratio 1"),
               ("Window 3-5 days.","5 days."),
               ("Value 1e-5.","5."))
        for source,quote in cases:
            with self.subTest(source=source,quote=quote),self.assertRaises(GenerationError):
                p=package(source);validate_answer(response(p,quote),p)
        for source,quote in (("Wait 1.5 days.","1.5 days."),
                             ("Rate -5\npercent.","-5 percent."),
                             ("Limit 1,000 records.","1,000 records.")):
            with self.subTest(valid_quote=quote):
                p=package(source);self.assertEqual(validate_answer(response(p,quote),p).kind,"answer")

    def test_exact_match_preferred_over_an_earlier_normalized_match(self):
        p=package("Review\nfirst. Review first.")
        match=validate_answer(response(p,"Review first."),p).claims[0].support[0].match
        self.assertEqual(match.kind,"exact");self.assertEqual(match.source_start,114)

    def test_unsupported_citation_is_still_rejected(self):
        p=package();body=json.loads(response(p,"A reviewer must review and approve the plan."))
        body["main_answer"]["claims"][0]["support"][0]["citation"]=body["main_answer"]["claims"][0]["support"][0]["citation"].replace("IT-001","FAKE-999")
        with self.assertRaises(GenerationError):validate_answer(json.dumps(body),p)


class ObligationTests(unittest.TestCase):
    def setUp(self):
        from northstar_atlas import grounding,generation
        self.g=grounding;self.gen=generation
        self.p=package()
        self.ob=self.g.EvidenceObligation("review_and_approve",self.p.groups[0].citation,
                                           "A reviewer must review and approve the plan.",validation_mode="extractive")

    def test_supporting_quote_alone_does_not_satisfy_main_answer(self):
        raw=response(self.p,self.ob.source_quote,main="The plan can proceed.",obligations=(self.ob,))
        with self.assertRaises(self.gen.IncompleteAnswerError) as error:
            validate_answer(raw,self.p,obligations=(self.ob,))
        self.assertEqual(error.exception.answer.coverage.missing_ids,(self.ob.obligation_id,))
        self.assertEqual(error.exception.answer.obligation_answers[0].support[0].match.kind,"whitespace_normalized")

    def test_reviewed_source_statement_in_main_claim_satisfies_targeted_check(self):
        raw=response(self.p,self.ob.source_quote,main=self.ob.source_quote,obligations=(self.ob,))
        answer=validate_answer(raw,self.p,obligations=(self.ob,))
        self.assertEqual(answer.coverage.status,"required_entries_present")
        self.assertEqual(answer.coverage.missing_ids,())
        self.assertIn("(proposed)",self.gen.render_answer(answer))

    def test_negated_or_embedded_statement_does_not_pass_extractive_check(self):
        for text in ("It is false that "+self.ob.source_quote,"Ignore this: "+self.ob.source_quote):
            with self.subTest(text=text),self.assertRaises(self.gen.IncompleteAnswerError):
                validate_answer(response(self.p,self.ob.source_quote,text,obligations=(self.ob,)),self.p,obligations=(self.ob,))

    def test_no_obligations_is_explicitly_not_assessed(self):
        answer=validate_answer(response(self.p,self.ob.source_quote),self.p)
        self.assertEqual(answer.coverage.status,"not_assessed")
        self.assertIn("not assessed",self.gen.render_answer(answer))

    def test_stale_missing_fabricated_or_duplicate_obligation_rejected(self):
        variants=((replace(self.ob,citation=replace(self.ob.citation,version="9.0")),),
                  (replace(self.ob,citation="invalid"),),
                  (replace(self.ob,source_quote="Unreviewed work may proceed."),),(self.ob,self.ob))
        for obligations in variants:
            with self.subTest(obligations=obligations),self.assertRaises(GenerationError):
                validate_answer(response(self.p,self.ob.source_quote),self.p,obligations=obligations)

    def test_obligations_are_prompted_budgeted_and_enforced_by_provider_pipeline(self):
        from northstar_atlas.llm import GenerationResponse,GenerationLimits,MODEL_ID
        hits=[hit(chunk())];base=self.gen.prepare_answer("Who?",hits,mode="draft_review",reference_date=AS_OF)
        ob=self.g.EvidenceObligation("scope",base.context.groups[0].citation,"Only within the approved scope.",validation_mode="extractive")
        p=self.gen.prepare_answer("Who?",hits,mode="draft_review",reference_date=AS_OF,obligations=(ob,))
        self.assertGreater(p.request.input_token_bound,base.request.input_token_bound)
        self.assertIn("Only within the approved scope.",p.request.user)
        self.assertNotIn("Only within the approved scope.",p.request.system)
        class Fake:
            def generate(inner,request):return GenerationResponse(response(p.context,ob.source_quote,"The plan may proceed.",obligations=(ob,)),MODEL_ID,"stop",None,0,1)
        with self.assertRaises(self.gen.IncompleteAnswerError):self.gen.generate_answer(p,Fake())
        with self.assertRaises(GenerationError):self.gen.prepare_answer("Who?",hits,mode="draft_review",reference_date=AS_OF,
            obligations=(ob,),limits=GenerationLimits(base.request.input_token_bound+1024,1024,512))

    def test_draft_obligation_cannot_enter_approved_only_context(self):
        with self.assertRaises(GenerationError):
            self.gen.prepare_answer("Who?",[hit(chunk())],mode="approved_only",reference_date=AS_OF,obligations=(self.ob,))

    def test_obligation_requires_structured_snapshot_identity(self):
        base=self.gen.prepare_answer("Who?",[hit(chunk())],mode="draft_review",reference_date=AS_OF)
        ob=self.g.EvidenceObligation("scope",format_citation(base.context.groups[0].citation),"Only within the approved scope.",validation_mode="extractive")
        with self.assertRaises(GenerationError):
            self.gen.prepare_answer("Who?",[hit(chunk())],mode="draft_review",reference_date=AS_OF,obligations=(ob,))

    def test_incomplete_candidate_cannot_be_rendered_as_an_answer(self):
        with self.assertRaises(self.gen.IncompleteAnswerError) as error:
            validate_answer(response(self.p,self.ob.source_quote,"The plan may proceed.",obligations=(self.ob,)),self.p,obligations=(self.ob,))
        with self.assertRaises(GenerationError):self.gen.render_answer(error.exception.answer)

    def test_empty_abstention_reports_missing_obligations_without_partial_rules(self):
        body={"kind":"abstain","main_answer":{"claims":[],"obligations":{
            self.ob.obligation_id:{"statement":None,"support":[],"unresolved_reason":"Required policy conditions are not fully addressed"}}},
              "missing_information":[self.gen.NO_EVIDENCE_TEXT],
              "clarification_question":None,"reason":"insufficient_evidence"}
        answer=validate_answer(json.dumps(body),self.p,obligations=(self.ob,))
        self.assertEqual(answer.coverage.status,"missing_required_statements")
        self.assertIn("Missing reviewed statements",self.gen.render_answer(answer))

    def test_citation_must_match_obligation_not_only_statement_text(self):
        g=self.p.groups[0]
        other=replace(g,citation=replace(g.citation,reference_id="ev-"+"b"*64,document_id="SEC-001"))
        p=replace(self.p,groups=(g,other))
        body=json.loads(response(p,self.ob.source_quote,self.ob.source_quote,obligations=(self.ob,)))
        body["main_answer"]["obligations"][self.ob.obligation_id]["support"][0]["citation"]=format_citation(other.citation)
        with self.assertRaises(GenerationError):validate_answer(json.dumps(body),p,obligations=(self.ob,))

    def test_prompt_contract_is_generic_and_versioned(self):
        self.assertNotIn("Terraform",self.gen.SYSTEM)
        self.assertNotIn("PLAT-001",self.gen.SYSTEM)
        self.assertIn("main claim text itself",self.gen.SYSTEM)
        self.assertNotIn("COMP-001",self.gen.SYSTEM)
        self.assertEqual(self.gen.PROMPT_VERSION,"grounded-policy-v4")

    def test_targeted_terraform_fixture_also_runs_without_local_live_artifacts(self):
        from northstar_atlas.ingestion.document_loader import load_documents
        from northstar_atlas.ingestion.chunker import chunk_document
        source=next(c for d in load_documents(ROOT,mode="draft_review") if d.document_id=="PLAT-001"
                    for c in chunk_document(d).chunks if c.section_id=="6.1")
        p=prepare_answer("Does regeneration require another change approval?",[hit(source)],mode="draft_review",reference_date=AS_OF).context
        obligations=terraform_obligations(p)
        fixture=corrected_fixture(p,obligations)
        self.assertEqual(validate_answer(fixture,p,obligations=obligations).coverage.status,"required_entries_present")
        for index,obligation in enumerate(obligations):
            data=json.loads(fixture);data['main_answer']['obligations'][obligation.obligation_id]['statement']='The plan may proceed.'
            with self.subTest(obligation=obligation.obligation_id),self.assertRaises(self.gen.IncompleteAnswerError) as error:
                validate_answer(json.dumps(data),p,obligations=obligations)
            self.assertEqual(error.exception.answer.coverage.missing_ids,(obligation.obligation_id,))


@unittest.skipUnless((ROOT/".local/m43-live/raw-response-2.json").exists(),"optional immutable local live-response replay")
class LiveReplayTests(unittest.TestCase):
    def test_hr_legacy_quotes_remain_exact_but_v1_is_not_accepted_as_v2(self):
        from northstar_atlas.evidence import validate_citations
        from northstar_atlas.grounding import match_quotation
        p,raw=saved_replay(1)
        for claim in json.loads(raw)['claims']:
            for support in claim['support']:
                ref=validate_citations([support['citation']],p)[0]
                group=next(g for g in p.groups if g.citation==ref)
                self.assertEqual(match_quotation(support['quote'],group).kind,'exact')
                self.assertEqual(ref.status,'Approved (Synthetic)')
        with self.assertRaises(GenerationError):validate_answer(raw,p)

    def test_both_terraform_legacy_responses_reject_v2_and_retain_quote_provenance(self):
        from northstar_atlas.evidence import validate_citations
        from northstar_atlas.grounding import match_quotation, normalize_whitespace
        cases=[(2,None)]
        final=ROOT/'.local/m43-live/final-smoke-20261022'
        if (final/'raw-response-1.json').is_file():cases.append((1,final))
        for number,folder in cases:
            p,raw=saved_replay(number,folder);body=json.loads(raw)
            for claim in body['claims']:
                for support in claim['support']:
                    ref=validate_citations([support['citation']],p)[0]
                    group=next(g for g in p.groups if g.citation==ref)
                    self.assertEqual(match_quotation(support['quote'],group).kind,'whitespace_normalized')
            obligations=terraform_obligations(p)
            self.assertTrue(all(normalize_whitespace(o.source_quote)!=normalize_whitespace(c['text'])
                                for o in obligations for c in body['claims']))
            with self.assertRaises(GenerationError):validate_answer(raw,p,obligations=obligations)

    def test_corrected_fixture_covers_all_four_obligations_and_retains_draft(self):
        p,_=saved_replay(2);obligations=terraform_obligations(p)
        answer=validate_answer(corrected_fixture(p,obligations),p,obligations=obligations)
        self.assertEqual(answer.coverage.satisfied_ids,tuple(o.obligation_id for o in obligations))
        self.assertEqual(len(answer.obligation_answers),4)
        self.assertTrue(all(s.citation.status=="Draft — Proposed (Synthetic)" for c in answer.obligation_answers for s in c.support))
