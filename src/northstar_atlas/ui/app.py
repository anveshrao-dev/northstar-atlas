"""Run from the authorized root with `python -m streamlit run .../ui/app.py`."""
import atexit
from datetime import datetime, timezone
import os
from pathlib import Path
from uuid import uuid4

import streamlit as st

from northstar_atlas.ingestion.document_loader import DRAFT
from northstar_atlas.ui.demo import SCENARIOS
from northstar_atlas.project import ProjectRootError, contained_path, validate_project_root
from northstar_atlas.ui.runtime import (
    Conversation, LANGGRAPH_WORKFLOW, MAX_QUESTION_CHARS, PublicView, ResourceOwner,
    POLICY_SEARCH, STANDARD_WORKFLOW, Submission,
    graph_operator_authorized,
)


@st.cache_resource(show_spinner=False, on_release=lambda owner: owner.close())
def resource_owner(root: str, live_enabled: bool, pinecone_enabled: bool,
                   graph_enabled: bool = False) -> ResourceOwner:
    owner = ResourceOwner(Path(root), live_enabled=live_enabled, pinecone_enabled=pinecone_enabled,
                          graph_enabled=graph_enabled)
    atexit.register(owner.close)
    return owner


def render_view(view: PublicView) -> None:
    """Render dynamic text inertly: never execute model Markdown, HTML or URLs."""
    if view.kind == 'error':
        st.error(view.text)  # Fixed application strings only.
        return
    if view.kind == 'evidence':
        st.caption('Policy search — retrieved evidence only')
        st.text(view.text, width='stretch')
        st.caption('Cosine similarity measures retrieval relevance, not answer confidence. Retrieved passages may not cover every applicable condition.')
        if view.proposed:
            st.warning('Proposed policy evidence — this is not an effective approved rule.')
        for rank, hit in enumerate(view.passages, start=1):
            st.subheader(f'Result {rank}')
            st.text(hit.chunk.citation, width='stretch')
            st.caption(f'Cosine similarity: {hit.cosine_similarity:.4f} · higher means closer in embedding space')
            st.text(hit.chunk.text, width='stretch')
            if hit.context:
                with st.expander(f'Required policy context for result {rank}', expanded=True):
                    st.caption('Required sibling passages; these are not additional ranked hits.')
                    for sibling in hit.context:
                        st.text(sibling.citation, width='stretch')
                        st.text(sibling.text, width='stretch')
            with st.expander(f'Source identity for result {rank}', expanded=False):
                for chunk in (hit.chunk, *hit.context):
                    st.text(f'{chunk.citation}\nChunk: {chunk.chunk_id}\nSource: {chunk.source_path.as_posix()}\nSHA-256: {chunk.content_hash}', width='stretch')
        return
    st.caption({'answer': 'Answer', 'clarify': 'Clarification needed', 'abstain': 'Evidence-limited abstention'}[view.kind])
    if view.proposed:
        st.warning('Proposed policy evidence — this is not an effective approved rule.')
    presentation = getattr(view, 'presentation', None)
    source_numbers = {source: i for i, source in enumerate(getattr(view, 'sources', ()), start=1)}

    def source_caption(citations) -> str:
        return ' · '.join(f'{"Proposed" if c.status == DRAFT else "Approved"} [{source_numbers[c]}]'
                          for c in citations)

    if presentation is None:
        # Older accepted results are displayed intact, never parsed for formatting markers.
        st.text(view.text, width='stretch')
    else:
        if presentation.abstention_basis == 'documented_gap':
            st.caption('Documented gap in the supplied evidence; this does not establish whether rules exist elsewhere.')
        elif presentation.abstention_basis == 'insufficient_evidence':
            st.caption('Insufficient supplied evidence; retrieval may be incomplete.')
        for statement in presentation.statements:
            st.text(statement.text, width='stretch')
            origin = 'Reviewed policy excerpt · ' if statement.origin == 'application_reviewed_source' else ''
            if statement.citations:
                st.caption(origin + source_caption(statement.citations))
        if presentation.clarification_question:
            st.text(presentation.clarification_question, width='stretch')
        if presentation.missing_information:
            if view.kind == 'abstain' and presentation.statements:
                st.caption('Cannot establish from supplied evidence:')
            elif view.kind != 'abstain':
                st.caption('Missing information:')
            for item in presentation.missing_information:
                st.text(item, width='stretch')
    if view.citations:
        st.subheader('Sources')
        for index, label in enumerate(view.citations, start=1):
            st.text(f'[{index}] {label}', width='stretch')
    with st.expander('Evidence details', expanded=False):
        if presentation is None:
            st.caption('Separate quotation details are unavailable for this older response. Its complete text is preserved above.')
        elif not presentation.quotations:
            st.caption('No supporting quotations were supplied for this result.')
        else:
            for quotation in presentation.quotations:
                st.caption(source_caption((quotation.citation,)))
                st.text(quotation.text, width='stretch')
    with st.expander('Validation details', expanded=False):
        if view.coverage == 'not_assessed':
            st.caption('Policy-condition completeness has not been assessed. Citations alone do not prove completeness.')
        else:
            st.caption('Reviewed obligation coverage: ' + view.coverage.replace('_', ' ') + '.')
        if presentation is not None:
            for index, quotation in enumerate(presentation.quotations, start=1):
                st.caption(f'Quotation {index}: {quotation.match_kind.replace("_", " ")}.')
        st.caption('Citation and quotation matching do not prove answer correctness, semantic completeness or faithfulness.')


def main() -> None:
    st.set_page_config(page_title='Northstar Atlas', page_icon='🧭', layout='centered')
    try:
        root = validate_project_root(Path.cwd())
        if Path(__file__).absolute() != contained_path(root, 'src/northstar_atlas/ui/app.py'):
            raise ProjectRootError('application script differs from selected checkout')
    except ProjectRootError:
        st.error('Launch this application from the authorized Northstar Atlas project root.')
        return
    # Server gates are deliberately process environment settings, not .env or UI toggles.
    live_enabled = os.environ.get('NORTHSTAR_UI_ENABLE_LIVE') == '1'
    pinecone_enabled = os.environ.get('NORTHSTAR_UI_ENABLE_PINECONE') == '1'
    graph_enabled = graph_operator_authorized()
    owner = resource_owner(str(root), live_enabled, pinecone_enabled, graph_enabled)
    if 'conversation' not in st.session_state:
        st.session_state.conversation = Conversation()
    conversation = st.session_state.conversation
    pending = conversation.pending is not None

    st.title('Northstar Atlas')
    st.write('Enterprise knowledge & operations assistant')
    st.caption('Explore synthetic Northstar Technologies policies with source citations.')

    with st.sidebar:
        st.header('Workspace')
        experience = st.radio('Experience', ('Offline demo', POLICY_SEARCH, 'Live Q&A'), key='experience', disabled=pending)
        demo = experience == 'Offline demo'
        retrieval_only = experience == POLICY_SEARCH
        batch_live = (os.environ.get('NORTHSTAR_DEMO_BATCH_ENABLED', '0') != '0'
                      and not demo and not retrieval_only)
        batch_remaining = owner.batch_remaining() if batch_live else None
        if demo or retrieval_only or batch_live:
            workflow = STANDARD_WORKFLOW
        else:
            workflow = st.selectbox(
                'Answer workflow', (STANDARD_WORKFLOW, LANGGRAPH_WORKFLOW), key='workflow',
                format_func=lambda value: {
                    STANDARD_WORKFLOW: 'Standard RAG · default',
                    LANGGRAPH_WORKFLOW: 'LangGraph RAG · semantic checks',
                }[value], disabled=pending)
            if workflow == LANGGRAPH_WORKFLOW and not graph_enabled:
                st.warning('LangGraph live execution is disabled pending shared budget coordinator and security activation approval. No provider request will be sent.')
            elif workflow == LANGGRAPH_WORKFLOW:
                st.caption('LangGraph requires a full durable workflow reservation. Budget, pricing, evidence and private-access checks must pass before execution.')
        if batch_live:
            st.caption('Approved batch · Standard RAG · Pinecone')
            if batch_remaining is None:
                st.warning('Batch authorization or budget is unavailable. Live questions are disabled.')
            elif batch_remaining == 0:
                st.warning('The approved five-question batch is exhausted. A new reviewed approval is required.')
            else:
                st.caption(f'Remaining batch submissions: {batch_remaining} of 5. Shared across sessions and restarts.')
        mode = st.selectbox('Policy scope', ('approved_only', 'draft_review'), key='mode', disabled=pending,
                            format_func=lambda v: {'approved_only': 'Approved policies only',
                                                   'draft_review': 'Approved + proposed drafts'}[v])
        backends = ('chroma', 'pinecone') if pinecone_enabled and not retrieval_only else ('chroma',)
        if batch_live:
            backends = ('pinecone',)
        if st.session_state.get('backend') not in backends:
            st.session_state.backend = backends[0]
        backend = st.selectbox('Retrieval backend', backends, key='backend', disabled=pending,
                               format_func=lambda v: 'Chroma · local' if v == 'chroma' else 'Pinecone · hosted')
        if retrieval_only:
            st.caption('Policy search uses the existing local Chroma index and cached MiniLM model. No hosted providers or LLM inference.')
        elif not pinecone_enabled:
            st.caption('Pinecone is optional and disabled by the server.')
        elif backend == 'pinecone':
            st.warning('Connecting to Pinecone may incur read usage. Selecting it alone does not connect.')
        consent = st.checkbox('Enable paid answers for this session', key='paid_opt_in',
                              disabled=demo or retrieval_only or not live_enabled or pending)
        if os.environ.get('NORTHSTAR_CONTROLLED_LIVE') == '1':
            st.caption('Controlled live use requires a durable budget reservation. Limits persist across restarts. No automatic retries.')
        else:
            st.caption('Live use: at most five attempted questions per server process. No automatic retries.')
        if not live_enabled:
            st.caption('Live inference is disabled by the server. Offline demo needs no credentials.')
        st.button('Clear conversation', key='clear_history', on_click=conversation.clear, disabled=pending)
        if st.button('Disconnect resources', key='disconnect', disabled=pending):
            if owner.close():
                st.success('Disconnected. Request accounting is preserved.')
            else:
                st.warning('Resources are busy or cleanup failed. No request was cancelled.')

    if demo:
        st.info('Offline demonstration — authored examples, not live retrieval or model responses. No API calls.')
    elif retrieval_only:
        st.info('Policy search — actual local retrieval, no generated answer. Local retrieval dependencies, a cached model and a verified index are required.')
    else:
        st.info('Each question is independent. Conversation history is display-only; include all needed details.')
    if mode == 'draft_review':
        st.warning('Proposed drafts are included for review and must not be treated as effective policy.')

    def queue(question: str) -> None:
        conversation.queue(Submission(uuid4().hex, question, mode, backend,
                                      datetime.now(timezone.utc).date(), demo=demo,
                                      workflow=workflow, retrieval_only=retrieval_only))

    if demo:
        scenarios = tuple(s for s in SCENARIOS if mode == 'draft_review' or s.mode == 'approved_only')
        labels = tuple(s.label for s in scenarios)
        if st.session_state.get('example') not in labels:
            st.session_state.example = labels[0]
        selected = st.selectbox('Demonstration example', labels, key='example', disabled=pending)
        scenario = next(s for s in scenarios if s.label == selected)
        st.button('Show example', key='run_demo', disabled=pending,
                  on_click=queue, args=(scenario.question,))
        if mode == 'approved_only':
            st.caption('Select approved + proposed drafts to explore Terraform and retention examples.')

    def queue_chat() -> None:
        queue(st.session_state.question)

    st.chat_input('Ask about a policy…', key='question', max_chars=MAX_QUESTION_CHARS,
                  disabled=(pending or (batch_live and (batch_remaining is None or batch_remaining == 0
                                                       or not pinecone_enabled))
                            or (not demo and not retrieval_only and not (live_enabled and consent))
                            or (not demo and not retrieval_only and workflow == LANGGRAPH_WORKFLOW
                                and not graph_enabled)), on_submit=queue_chat)
    if conversation.pending is not None:
        with st.spinner('Preparing demonstration…' if conversation.pending.demo else
                        'Searching verified local policy evidence…' if conversation.pending.retrieval_only else
                        'Retrieving evidence and preparing a validated response…'):
            conversation.run_pending(owner, consent=consent)
        st.rerun()  # Display stored results; pending has been consumed, never re-submit.

    for turn in conversation.history:
        with st.chat_message('user'):
            st.text(turn.submission.question, width='stretch')
        with st.chat_message('assistant'):
            if turn.submission.demo:
                origin = 'Offline demonstration'
            elif turn.submission.retrieval_only:
                origin = f'Policy search · local Chroma · as of {turn.submission.reference_date.isoformat()}'
            else:
                workflow_label = ('LangGraph RAG' if turn.submission.workflow == LANGGRAPH_WORKFLOW
                                  else 'Standard RAG')
                origin = f'Live · {turn.submission.backend} · {workflow_label}'
            st.caption(f'{origin} · {turn.submission.mode}')
            render_view(turn.view)
    if not conversation.history:
        st.caption('Start with an example or ask a question. Proposed evidence is always labeled.')


if __name__ == '__main__':
    try:
        main()
    except Exception:
        # Streamlit control-flow exceptions derive from BaseException and must pass through.
        # Never print a caught exception: it may carry provider details or credentials.
        st.error('The interface could not complete this action. No answer was accepted.')
