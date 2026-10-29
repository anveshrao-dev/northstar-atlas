# Interactive policy search without answer generation

The Streamlit Experience selector now offers **Policy search — retrieval only**
alongside Offline demo and Live Q&A. Offline demo remains the initial default
and continues to display clearly labeled authored examples. Policy search accepts
free-text questions with the existing 2,000-character UI limit, even when live
inference is disabled or its allowance is exhausted.

## Retrieval and presentation

The search path calls the existing `open_retriever(root, 'chroma')`, then
`RetrievalStore.search()` with `top_k=5`, the selected policy mode, the submission's
UTC reference date and a freshly verified corpus fingerprint. It uses dense
retrieval with the cached, pinned local MiniLM model. It never constructs
QAService, Nebius, a semantic graph or a budget coordinator. Pinecone is not
offered in this mode, and a forged Pinecone search submission is rejected even
when hosted retrieval is enabled for Live Q&A.

Up to five direct hits appear in rank order, with their exact passage text,
document/section/version/status citation and cosine similarity. Required sibling
passages appear in an initially expanded context section, explicitly distinct
from additional ranked hits. Source details retain chunk ID, relative source
path and source hash. Text is rendered inertly; retrieved Markdown/HTML/URLs are
not executed. No answer, paraphrase, clarification or policy conclusion is
generated from these passages. Streamlit removes outer whitespace in its visual
text elements; the stored passage strings remain unchanged.

Cosine similarity is labeled retrieval relevance, **not answer confidence**.
High similarity and valid source identity do not establish complete evidence
coverage or semantic answer correctness. Empty retrieval is reported as no
eligible passages returned, not proof that no applicable policy exists.

## Integrity, isolation and resources

Existing loader hashes, Chroma ready/publication state, record/vector digests,
embedding configuration, eligibility and sibling checks remain in place. Source
hashes are checked on each search even when resources are reused. Before display,
the public projection additionally rejects malformed scores, duplicate direct
IDs, ineligible/future approved evidence, incomplete sibling families and
conflicting source identities. One invalid group withholds the entire result;
there is no silent partial-evidence fallback. Drafts require explicit
`draft_review` selection and remain visibly proposed.

Search shares the existing serialized resource owner and session-local display
history. The retriever/model opens lazily on the first valid search, is reused
for later searches, and its context is closed on disconnect or a switch to an
answer workflow. Request IDs prevent automatic resubmission; busy requests are
rejected. Search does not consume, reset or initialize the live attempt counter
or durable ledger. Existing live opt-in, graph gates, pricing and reservation
behavior remain unchanged. No global inference flag is enabled by this feature.

Missing dependencies, cached model, index, stale source hashes or invalid
publication state yield a fixed safe error without raw diagnostics. A query
exceeding MiniLM's existing 256-token limit is rejected by its adapter without
truncation; users can try a shorter question. There are no downloads, indexing
actions or provider fallbacks in the search path.

## Setup and deployment limits

Use the existing local Streamlit startup command and choose the new Experience.
Keep live inference disabled. A verified local Chroma index, cached MiniLM model
and the existing full retrieval dependencies must already be provisioned. The
35-package Replit UI-only environment still supports authored Offline demo; it
does not acquire search dependencies or assets automatically. This change does
not deploy, change the offline launcher, enable hosted retrieval or install any
packages. A later hosted search setup requires separate review.

The chosen extension adds a retrieval-only branch to the existing resource owner
and an additive `PublicView.passages` field. Routing through QAService would risk
entering generation/budget logic, while a separate app would duplicate session
and resource management. Existing answer presentation and public contracts keep
their previous defaults.

## Offline validation

Tests use authored retrieval fixtures and temporary Chroma stores with fake
embeddings; they do not establish real-model retrieval quality. Coverage includes
free input, mode/backend restrictions, exact source display, proposed labels,
required context, safe failures, duplicate submissions, resource reuse and
cleanup, live allowance isolation, actual top-five Chroma retrieval, effective
dates, changed sources and incomplete publications. Existing production vectors
and model files are not changed.

Validation on October 29, 2026:

- Focused UI/retrieval suite: **85 run, 84 passed, one existing cached-model test
  skipped**, in 16.860 seconds with sockets blocked.
- Full network-blocked suite: **913 run, 900 passed, 13 existing opt-in tests
  skipped**, in 88.817 seconds. No failures or errors. The skips cover cached
  MiniLM embeddings/refinement and cached-model Chroma/Pinecone integrations;
  none of the 14 new tests are skipped.
- M1.8 validator: all 15 questions, source hashes, manifest pin, evidence sections,
  modes and behaviors pass.
- UI imports and an authored demo answer pass in a separate process that rejects
  backend/model/graph/provider imports.
- The 35-package Replit UI export is byte-for-byte unchanged, SHA-256
  `0d7a42c564358febb8bde9167fc108c7e022953946cd9fef5e8d1ec38f1e1faf`.
- Git whitespace and six-file scope checks pass; the index remains empty.
  Streamlit's expected bare-mode test warnings do not indicate provider calls.

Reproduce with the existing environment from the authorized project root:

```bash
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export NORTHSTAR_CONTROLLED_LIVE=0 TMPDIR="$PWD/.local/tmp"
.venv/bin/python -B -m unittest discover -s tests -p 'test_ui*.py'
.venv/bin/python -B -m unittest discover -s tests -p test_policy_search.py
.venv/bin/python -B -m unittest discover -s tests -p test_vector_store.py
.venv/bin/python -B -m unittest discover -s tests
.venv/bin/python -B scripts/validate_evaluations.py
```

The reported full run additionally used the existing socket-blocking wrapper
around unittest discovery, including spawned processes. No Nebius/Pinecone
requests, real-ledger operations, installs, staging, commits or deployments were
performed. Real cached-model retrieval quality was not evaluated in this task.
