"""Authored demonstration fixtures, NOT real retrieval or model validation.

No model, vector store, credential loader or external provider is opened here.
Only exact example questions select fixtures; arbitrary questions clarify.
"""
from dataclasses import dataclass

from northstar_atlas.evidence import CitationReference
from northstar_atlas.grounding import ConditionCoverage
from northstar_atlas.ingestion.document_loader import APPROVED, DRAFT, RetrievalMode
from northstar_atlas.qa import (PublicAnswer, PublicError, PublicPresentation, PublicStatement,
                                QADiagnostics, QAFailure, QASuccess)


@dataclass(frozen=True)
class DemoScenario:
    label: str
    question: str
    mode: RetrievalMode


SCENARIOS = (
    DemoScenario('Approved HR answer', 'When must HR notify IT about planned versus urgent transfers?', 'approved_only'),
    DemoScenario('Proposed Terraform answer', 'Does regenerating a Terraform plan require another Change Management approval?', 'draft_review'),
    DemoScenario('Retention abstention', 'What is the exact approved retention period for production change evidence?', 'draft_review'),
    DemoScenario('Safe error', 'Demonstrate a rejected answer.', 'approved_only'),
    DemoScenario('Clarification', 'Can I access production?', 'approved_only'),
)


class DemoQAService:
    """Fake ask interface used only while the view is explicitly in offline demo."""
    def ask(self, question: str, *, mode: RetrievalMode = 'approved_only') -> QASuccess | QAFailure:
        diagnostic = QADiagnostics('complete', 'demo', mode, None, 'authored-demo-fixture', 'no-model')
        selected = next((i for i, s in enumerate(SCENARIOS) if s.question == question), 4)
        if (selected in (1, 2) and mode != 'draft_review') or selected == 3:
            return QAFailure(PublicError('response_rejected', 'Demonstration error.'), diagnostic)
        kind, citations = 'answer', ()
        if selected == 0:
            text = ('HR must notify IT Operations at least two business days before a planned internal transfer. '
                    'For urgent transfers without advance notice, HR notifies IT as soon as the change is '
                    'approved and identifies the request as urgent.')
            doc, section, status = 'HR-001', '4.2', APPROVED
        elif selected == 1:
            text = ('Proposed policy: A regenerated Terraform plan requires renewed Platform technical review '
                    'and recorded approval; the old approval does not approve the new plan. Regeneration alone '
                    'does not require a separate Change Management approval. Platform may proceed after renewed '
                    'technical validation and approval only within the authorized CR scope, assessed risk, '
                    'deployment window and other approval conditions, recording that comparison. If the result '
                    'materially exceeds approved scope or risk, or would exceed the window, stop and follow '
                    'the amendment or escalation process.')
            doc, section, status = 'PLAT-001', '6.1', DRAFT
        elif selected == 2:
            kind = 'abstain'
            text = ('The proposed Compliance policy explicitly sets no numeric retention period. I cannot '
                    'establish an exact approved duration from this evidence. This is a documented gap in '
                    'the supplied policy, not proof that no approved schedule could exist elsewhere.')
            doc, section, status = 'COMP-001', '11.1', DRAFT
        else:
            kind = 'clarify'
            text = ('Which system, role and type of access do you mean? In this offline demonstration, '
                    'choose an example to see an authored response; free-text questions do not run retrieval.')
            doc = None
        if doc:
            version = '1.0' if doc == 'HR-001' else '0.1'
            citations = (CitationReference('demo-reference', doc, section, version, status,
                                            'authored-demo-fixture', ('demo-chunk',)),)
        # Keep authored fixture wording and legacy text intact; do not invent quote support.
        presentation = (PublicPresentation(clarification_question=text) if kind == 'clarify'
                        else PublicPresentation(statements=(PublicStatement(text, citations),),
                                                abstention_basis='documented_gap' if kind == 'abstain' else None))
        return QASuccess(PublicAnswer(kind, text, citations, ConditionCoverage(), presentation), diagnostic)
