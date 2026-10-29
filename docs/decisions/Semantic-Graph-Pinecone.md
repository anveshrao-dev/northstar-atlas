# Pinecone retrieval for the semantic graph

## Scope and implementation

The optional `llm_routing_grading_semantic_v1` profile now accepts either Chroma
or Pinecone through the existing `open_retriever()` contract. Standard RAG and
Chroma remain the defaults. This extension is offline validated; it does not
establish that the semantic workflow has completed a live Pinecone/Nebius run.

The five functional stages remain routing, retrieval, evidence grading,
generation and semantic checking. The existing deterministic validator still
guards publication. There are no new nodes, prompts, models, retrieval algorithms
or category namespaces. The historical standalone routing/grading profile keeps
its previous factory restrictions.

Pinecone already implements the graph's `department_scope` contract. The first
retrieval applies the validated route to document department metadata before
top-k selection. The existing alternate attempt is unscoped; the default strategy
plan remains dense followed by local BM25. A configured hybrid strategy uses the
same dense adapter. Policy status, effective date, source identity and required
siblings are checked by the existing adapter and verified catalog. No publication,
upload, index creation or document embedding regeneration is added.

The graph forwards its explicit immutable snapshot ID to the shared Pinecone
opener. The active publication must match that exact ID before store construction
and remote audit. Matching corpus/model metadata alone is insufficient: vectors
also contribute to snapshot identity. A mismatch fails closed without selecting
an older namespace or publishing another snapshot. Explicit snapshot pins also
survive the shared hybrid opener. Existing unpinned linear dense retrieval keeps
its active-publication behavior.

## Authorization and resource lifecycle

The controlled opener captures the selected backend once. The existing durable
request binding includes backend, profile, question, policy mode, reference date,
corpus and model. Opener, cached-resource reuse and graph invocation verify that
binding; a Pinecone permit cannot authorize Chroma or a changed question. Source
catalog and resource cleanup checks remain in place.

The budget changes remove only the two application checks forbidding the
semantic-profile/Pinecone pair. Ledger version, fields, history, reservation
formula, pricing, limits, stage permits and settlement rules are unchanged.
The complete reservation remains seven calls and 204,800 aggregate token bounds:
184,064 input-bound tokens plus 20,736 output tokens. One routing call, up to two
grading calls, two generation calls and two semantic checks remain the limits.
The graph can retrieve at most twice and regenerate at most once. Provider
failures retain conservative uncertain accounting; rejected answers stay withheld.

Live graph execution still requires the controlled-live, live-UI, graph and
private-preview operator flags, a ready shared coordinator and an explicit paid
session opt-in. Pinecone additionally requires `NORTHSTAR_UI_ENABLE_PINECONE=1`.
No flags were enabled by this implementation. The frozen 189-record local
snapshot, source manifest and pinned MiniLM assets remain required for both
backends. Only Chroma requires a local Chroma database. Pinecone retains its
credential validation, target confirmation, active publication and remote audit.
The local launcher preflight does not construct providers or contact Pinecone.
Remote setup occurs only after the complete workflow reservation.

Historical diagnostic-window authorizations are unchanged: the graph retest
contract remains Chroma-specific and the one-shot Pinecone demonstration remains
linear-only. This change cannot repurpose either authorization for a Pinecone
graph. No real ledger, publication record, limit or pricing revision was read or
modified as part of this task.

## Offline validation

Tests use authored responses, synthetic temporary ledgers and fake Pinecone HTTP
transport with the real adapter/SDK codecs. The in-memory ranking fixture proves
filtering and integrity, not retrieval quality. Coverage includes:

- Every functional graph stage and final deterministic publication validation.
- Department filtering, approved-only isolation and explicit proposed labels.
- Complete grading of direct chunks and required siblings; missing context and
  altered source metadata fail closed before generation.
- Alternate hybrid retrieval plus bounded regeneration within seven calls.
- Negative semantic-check exhaustion with no rejected-candidate publication.
- Exact active snapshot matching, owner-confirmation failure, cleanup and reuse.
- Durable backend/request binding, duplicate rejection, timeout accounting,
  resource cleanup and unchanged live/graph/Pinecone/consent gates.

Focused graph, orchestration, Pinecone and scope tests: **211 run, 210 passed,
one existing opt-in integration test skipped**. The M1.8 validator passes all 15
frozen questions, source hashes, evidence references, modes and behaviors.
The full network-blocked suite on 2026-10-28 ran **866 tests in 77.632 seconds:
853 passed, 13 skipped**. The unchanged skips are two opt-in MiniLM tests, nine
token-refinement tests, one offline full-corpus Pinecone round-trip and one Chroma
integration test. No graph, authorization, budget, provider-contract or UI test
was skipped. Streamlit emitted its existing bare-mode ScriptRunContext warnings.
The UI-only export remains 35 packages with SHA-256
`0d7a42c564358febb8bde9167fc108c7e022953946cd9fef5e8d1ec38f1e1faf`.

Focused reproduction commands in the existing environment:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m unittest discover -s tests -p 'test_pinecone_graph.py'
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m unittest discover -s tests -p 'test_graph_retriever.py'
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m unittest discover -s tests -p 'test_live_graph.py'
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B scripts/validate_evaluations.py
```

## Remaining live activation steps

1. Review and synchronize the source changes separately. Preserve private runtime
   files and the authoritative ledger; do not overwrite them with test fixtures.
2. Verify the full pinned runtime, optional orchestration/Pinecone packages, local
   MiniLM cache and original frozen publication snapshot on the target host.
   The 35-package UI-only offline environment cannot run this workflow. No local
   Chroma index is needed when Pinecone is explicitly selected.
3. Independently verify private-preview access, process credentials, target owner
   confirmation and the existing publication state. Remote readiness/audit checks
   require separate permission; none ran during this implementation.
4. Obtain explicit approval for one backend-bound semantic workflow and sufficient
   seven-call capacity at current verified pricing. Preserve all uncertain
   reservations. Existing diagnostic-only authorization must be resolved by a
   separately reviewed operational procedure; do not remove its guards or reuse
   a linear/Chroma authorization. This task adds no allowance or amendment tool.
5. Only after those checks, explicitly select Pinecone and the semantic workflow,
   enable its reviewed gates, and execute the separately approved bounded test.
   Stop on a safe failure; never retry the entire workflow automatically.

Semantic checking remains an additional model assessment, not proof of complete
or correct answers. Ordinary questions still report obligation coverage as
`not_assessed`. Offline success does not verify hosted credentials, region,
publication readiness, request compatibility or answer quality.

Related history: [controlled live wiring](Controlled-Live-Graph-Wiring.md),
[shared budget](Shared-Durable-Live-Budget.md), and
[controlled Pinecone linear demonstration](Controlled-Pinecone-Demonstration.md).

## Exact implementation file inventory

| File | Change |
| --- | --- |
| `src/northstar_atlas/orchestration.py` | Permit explicit semantic Pinecone selection; forward exact snapshot pin. |
| `src/northstar_atlas/qa.py` | Reject mismatched active snapshots and retain pins through hybrid opening. |
| `src/northstar_atlas/live_budget.py` | Remove the two graph/Pinecone combination bans; no ledger format or accounting changes. |
| `src/northstar_atlas/live_graph.py` | Backend-bound gates, local assets and controlled resource opening. |
| `src/northstar_atlas/live_preflight.py` | Local graph preflight uses the explicitly selected backend. |
| `src/northstar_atlas/ui/runtime.py` | Remove the obsolete graph/Pinecone exclusion, preserving other gates. |
| `src/northstar_atlas/ui/app.py` | Allow the enabled backend selection for the optional graph. |
| `tests/test_graph_retriever.py` | New exact snapshot, confirmation and shared-opener tests. |
| `tests/test_pinecone_graph.py` | New fake-Pinecone semantic graph and failure-path tests. |
| `tests/test_live_graph.py` | Pinecone reservation binding, seven stages, timeout and asset tests. |
| `tests/test_live_launcher.py` | Local Pinecone graph preflight and operator gates. |
| `tests/test_orchestration.py` | Existing Pinecone factory fixture checks forwarded snapshot pin. |
| `tests/test_ui.py` | Backend selection and independent authorization gates. |
| `tests/test_ui_app.py` | Optional graph/Pinecone selection and consent presentation. |
| `docs/decisions/Semantic-Graph-Pinecone.md` | This implementation report. |
| `docs/decisions/Controlled-Live-Graph-Wiring.md` | Link this extension while retaining original historical results. |

No dependency, offline launcher, Replit configuration, policy, manifest, frozen
evaluation, provider serialization, diagnostic-window authorization or publication
record changed. No files were staged and no provider calls were made.
