# Northstar Atlas — Enterprise Knowledge & Operations Assistant

Northstar Atlas answers questions about synthetic enterprise policies using
retrieved evidence, explicit policy status, and verifiable source references.
It models Northstar Technologies, a fictional hybrid-cloud SaaS company, across
HR, IT Operations, Platform Engineering, SRE, Information Security and Compliance.

Example questions include when HR must notify IT about a transfer, how a
regenerated Terraform plan affects change approvals, and whether the corpus
establishes an approved evidence-retention period. The application can return an
answer, ask for clarification, or abstain when the supplied evidence is inadequate.

**Current status:** a working local RAG and Streamlit MVP, with Chroma as the
default backend and an optional Pinecone integration. This is a development system
using fictional documents, not production policy advice. Project roots and
Pinecone identity are explicitly validated without personal paths or account IDs
in source. Alternate checkouts have offline fixture coverage; Linux/Replit
deployment and a fresh dependency/model installation remain unvalidated.

## Implemented architecture

```text
Six policy Markdown files + SHA-256 corpus manifest
  → verified document loading
  → 151 canonical heading-aware chunks
  → model-aware refinement: 189 embedding-ready chunks
  → local MiniLM embeddings
  → Chroma persistence / optional immutable Pinecone namespace

Question + approved_only or draft_review
  → query embedding and top-5 dense retrieval
  → required sibling context and source-identity checks
  → deduplicated, token-budgeted evidence package
  → Nebius Qwen structured response
  → citation, quotation, status and contract validation
  → QAService public result → Streamlit
```

Ingestion/indexing is separate from asking questions. The browser does not
provision cloud resources, download models or rebuild an index. Rejected responses
are withheld; provider and retrieval failures remain safe errors rather than
being presented as policy abstentions.

| Component | Current implementation |
| --- | --- |
| Python and environment | Python 3.12; uv and a committed dependency lock |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2`, local CPU, normalized 384-dimensional vectors |
| MiniLM revision | `1110a243fdf4706b3f48f1d95db1a4f5529b4d41` |
| Token compatibility | Actual MiniLM tokenizer; 256-token input limit including special tokens; no silent truncation |
| Local vector store | Chroma 1.5.9 with supplied vectors, cosine distance and persistence in `.local/chroma/` |
| Managed alternative | Pinecone SDK 10.0.0; Azure/cosine/384 dimensions; explicit publication and integrity checks |
| Generation | Nebius Token Factory, `Qwen/Qwen3-30B-A3B-Instruct-2507`, HTTPX adapter |
| Answer contract | `grounded-answer-v2`, including caller-reviewed obligations and evidence-backed abstention |
| Browser interface | Streamlit 1.66.1; separate answer, Sources, Evidence details and Validation details |
| Testing | Standard-library unittest; deterministic providers and optional cached-model checks |

LangChain, BM25, hybrid retrieval, reranking and LangGraph are not part of the
current runtime. See the [architecture and experiment guide](docs/README.md).

## Corpus and policy authority

| Document | Domain / policy | Current source status |
| --- | --- | --- |
| HR-001 | Human Resources — Employee Role Change & Offboarding | v1.0, Approved (Synthetic), effective 2026-10-01 |
| IT-001 | IT Operations — Identity & Access Management | v0.1, Draft — Proposed (Synthetic) |
| PLAT-001 | Platform Engineering — Production Deployment & Change Management | v0.1, Draft — Proposed (Synthetic) |
| SRE-001 | Site Reliability Engineering — Incident Response & Escalation | v0.1, Draft — Proposed (Synthetic) |
| SEC-001 | Information Security — Privileged Access & MFA | v0.1, Draft — Proposed (Synthetic) |
| COMP-001 | Compliance — Access Review & Audit Evidence | v0.1, Draft — Proposed (Synthetic) |

`approved_only` admits only approved policies effective on the reference date.
`draft_review` also admits drafts with explicit proposed labels. Accepting a draft
as a project deliverable does not make it an effective rule. The manifest and
source hashes must agree; review reports and evaluation answers are not ingested.

## Grounding safeguards and limits

Retrieved text is untrusted evidence, not instructions. Context assembly verifies
source identity and required sibling relationships, retains conditions and
exceptions, and excludes whole critical groups when they cannot fit the budget.
Generated citations must identify supplied evidence with the correct document,
section, version and status. Quotations must match an exact or whitespace-normalized
contiguous span; changed words and fabricated quotations are rejected.

Caller-supplied reviewed obligations can require specific evidence coverage.
Application-owned extractive statements are copied from verified source spans;
evidence-linked paraphrases still need semantic review. The ordinary chatbot does
not silently select evaluation obligations: its coverage remains `not_assessed`.
Citation membership and quotation matching do **not** prove that an answer is
correct, complete or faithful. A documented policy gap is distinguished from
insufficient retrieval evidence; source silence does not prove no rule exists
elsewhere.

## Local setup

Run setup from the intended project checkout; verify `pwd -P` before writing
files. The application defaults to its installed source checkout. An operator
can explicitly select a complete checkout with the process environment variable
`NORTHSTAR_PROJECT_ROOT`; it is never read from `.env` or accepted from chat input.
Explicit root arguments must match that authority. The root must already exist,
contain the Northstar project configuration/source layout and a valid corpus
manifest, and pass all source SHA-256 checks. Symlink roots and escaping paths are
rejected. A wheel installation needs this explicit setting and a complete checkout;
there is no search for a convenient parent directory or automatic root creation.
These application checks do not replace operating-system filesystem isolation.

### Environment and uv

Prerequisites: Git, an existing Python 3.12 interpreter and uv. The verified local
interpreter is 3.12.15; other Python minor versions are excluded by `pyproject.toml`.
Reuse the existing interpreter rather than downloading another one.

From the authorized checkout, keep storage inside the project:

```bash
pwd -P
export NORTHSTAR_PROJECT_ROOT="$PWD"
mkdir -p .local/tmp
export UV_CACHE_DIR="$PWD/.local/uv-cache"
export UV_PYTHON_INSTALL_DIR="$PWD/.local/uv-python"
export UV_PYTHON_BIN_DIR="$PWD/.local/bin"
export UV_PROJECT_ENVIRONMENT="$PWD/.venv"
export UV_PYTHON_DOWNLOADS=never
export TMPDIR="$PWD/.local/tmp"
export PYTHONDONTWRITEBYTECODE=1
```

These are session settings; do not add them to global shell configuration.
Check that writable destinations do not follow symlinks outside the checkout.
If Python 3.12 is not discoverable, set `UV_PYTHON` privately to its existing
absolute executable path. Never commit that personal path.

If uv is already available, reuse it. Otherwise, after approving a tooling
download, the [official installer](https://docs.astral.sh/uv/getting-started/installation/)
can be installed locally without shell-profile changes using
[`UV_UNMANAGED_INSTALL`](https://docs.astral.sh/uv/reference/installer/).
The example retains uv 0.12.23, used with this project's pinned build backend:

```bash
# Optional tooling download, not needed when uv is already installed.
# Do not overwrite an existing installer file; inspect it before execution.
test ! -e .local/uv-install.sh && curl -LsSf https://astral.sh/uv/0.12.23/install.sh -o .local/uv-install.sh
# Review .local/uv-install.sh locally before the next command.
UV_UNMANAGED_INSTALL="$PWD/.local/bin" sh .local/uv-install.sh
export PATH="$PWD/.local/bin:$PATH"
```

No package installation or download is performed merely by opening the UI.
For an approved dependency setup on the validated platform:

```bash
uv sync --locked --no-python-downloads --extra ui --extra llm
.venv/bin/python -B --version
.venv/bin/python -B -c 'import northstar_atlas; print("Package import OK")'
```

The base install alone does not include the UI or LLM extras. Full development
tests also use the optional Pinecone package; install all extras only when needed:
`uv sync --locked --no-python-downloads --all-extras`. Never copy a Mac `.venv`
to Linux. Review the Linux footprint before synchronization on a new platform.

### Offline browser demonstration

With dependencies installed and the authorized root satisfied:

```bash
export NORTHSTAR_UI_ENABLE_LIVE=0
export NORTHSTAR_UI_ENABLE_PINECONE=0
.venv/bin/python -B -m streamlit run src/northstar_atlas/ui/app.py
```

Open the loopback URL displayed by Streamlit and choose **Offline demo**.
Demonstrations are clearly labeled authored examples, not live retrieval or model
outputs. They require no API keys, MiniLM cache or vector index. The project config
binds `127.0.0.1`, disables usage telemetry and detailed browser errors, and keeps
CORS/XSRF protections enabled. Proposed-policy warnings remain visible.

### Prepare local retrieval for live Q&A

This is separate from the offline demo. Validate the source corpus first:

```bash
.venv/bin/python -B scripts/validate_evaluations.py
.venv/bin/python -B -m northstar_atlas.ingestion --mode draft_review
```

If the pinned MiniLM snapshot is missing, explicitly provision it after approving
the model download. This command uses the project's `.local/huggingface/` cache:

```bash
# Explicit model download; skip when the approved snapshot is already cached.
.venv/bin/python -B -m northstar_atlas.local_embeddings download
```

Runtime embedding uses cached files only. To create or intentionally refresh the
local index, after approving that local database write:

```bash
.venv/bin/python -B -m northstar_atlas.vector_store index
.venv/bin/python -B -m northstar_atlas.vector_store status
.venv/bin/python -B -m northstar_atlas.vector_store search   "Who approves a Terraform production plan?" --mode draft_review
```

The current corpus yields 189 records. Indexing verifies hashes, preserves
provenance, upserts deterministically, removes obsolete records and marks incomplete
attempts. If status is already ready and current, do not reindex unnecessarily.
Similarity is a ranking score, not answer confidence.

### Credentials and paid inference

Create an ignored `.env` from `.env.example` only if no `.env` already exists;
edit it privately. Never overwrite existing credentials or print the file.
Configure `NEBIUS_API_KEY` and the reviewed model. The Nebius loader also supports
process environment variables, checks Git exclusion, and rejects conflicting
values. `.env.example` identifies active settings and reserved placeholders;
`CHROMA_PERSIST_DIR` and `EMBEDDING_MODEL` are not runtime overrides today.
`VECTOR_STORE` is read from the process environment by library/CLI callers, not
from `.env`; Streamlit has an explicit backend selector.

Verify account/model availability and current pricing separately before paid use.
For a separately approved local live session, restart Streamlit with:

```bash
export NORTHSTAR_UI_ENABLE_LIVE=1
export NORTHSTAR_UI_ENABLE_PINECONE=0
.venv/bin/python -B -m streamlit run src/northstar_atlas/ui/app.py
```

Choose **Live Q&A** and explicitly check **Enable paid answers for this session**.
Both server authorization and session consent are required. These gates are read
from the process environment, not `.env`. Every submitted question is independent;
chat history is display-only. The UI permits at most five attempted questions per
server process, one request at a time, with no automatic retries. Its budgets are
32,768 context tokens, 2,048 output tokens and a 512-token margin. Prompt preparation
checks the full reservation; MiniLM's separate 256-token query limit still applies.
A process restart resets the in-memory allowance: this is not a durable spending
cap or authentication system.

### Optional Pinecone

The adapter requires an explicit reviewed target; blank placeholders do not select
an account. Store nonsecret target settings privately in the ignored file
`.local/pinecone/target.json`, using the exact owner-verified values:

```json
{
  "name": "your-reviewed-index-name",
  "cloud": "azure",
  "region": "eastus2",
  "organization": "Your verified organization",
  "project_name": "your-verified-project",
  "project_id": "replace-with-the-verified-project-uuid",
  "plan": "Builder"
}
```

This example is intentionally incomplete. Azure/eastus2, 384 dimensions and cosine
remain the supported design. Unknown/duplicate fields, malformed identity and
conflicts with `.env` are rejected. Configuration does not constitute account
confirmation or permission to create a resource.

An authorized existing setup requires the Pinecone extra, privately configured
key/index/cloud/region, a compatible published namespace, and validated local
confirmation/snapshot/publication metadata under `.local/pinecone/`. The separate
`target-confirmation.json` must bind the exact organization, project, index,
placement and credential digest after explicit owner Console verification.
Project-key APIs do not independently establish organization/project identity.
There is no automatic confirmation or resource selection. Historical
`project-confirmation.json` alone is not sufficient for the new index-bound check.
The existing development target was migrated offline while preserving that
original record; a new checkout/account needs its own explicit confirmation.
The current
Pinecone loader requires `.env`; it is not environment-only yet. After verifying
that setup and approving remote reads, `NORTHSTAR_UI_ENABLE_PINECONE=1` exposes
Pinecone in the UI. Selection alone does not connect; a live submission can incur
remote read usage. Pinecone does not remove the local query-embedding requirement.

## Tests and evaluation

Use the existing environment with all extras, from the authorized root and with
the local temporary/cache settings above. The tests use deterministic fake
providers; they do not require paid inference:

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
.venv/bin/python -B -m unittest discover -s tests -v
.venv/bin/python -B scripts/validate_evaluations.py
```

After the approved MiniLM snapshot is available, opt into cached-model integration
checks with `NORTHSTAR_RUN_MODEL_TESTS=1` before running unittest. Historical replay
tests skip when private `.local/` artifacts are absent; do not publish raw provider
responses just to avoid skips. Root/configuration tests use temporary checkouts
inside `.local/tmp`. A passing prepared-workspace run does not establish
fresh-install or Linux reproducibility.

The frozen [15-question dataset](evaluations/README.md) has 40 required evidence
references. In the historical dense-retrieval baseline and backend comparison:

| Metric, all 15 questions | Direct top 5, both stores | With required siblings, both stores |
| --- | ---: | ---: |
| Recall@5 | 55.22% | 64.67% |
| Hit@5 | 86.67% | 93.33% |
| Full evidence coverage | 26.67% | 40.00% |

All top-five ID sets matched; two order changes affected rank-sensitive metrics.
Chroma's measured adapter latency was lower in that local experiment. See the
[comparison report](docs/decisions/M3.4-Chroma-Pinecone-Comparison.md) for provenance,
MRR, category breakdowns and timing boundaries. These are conservative complete
source-section coverage metrics, not semantic correctness or answer faithfulness.
Fifteen synthetic development questions and selected live smoke tests cannot
establish general answer quality. Failures and corrective experiments are summarized
in the [engineering guide](docs/README.md#experiments-and-observed-limitations).

## Repository map

- `src/northstar_atlas/`: ingestion, embeddings, vector stores, evaluation,
  context/grounding, provider adapter, QAService and UI.
- `data/policies/` and `data/corpus_manifest.json`: synthetic source documents
  and their verified metadata/hashes.
- `tests/`, `evaluations/`, `scripts/`: offline tests, frozen ground truth and
  validation tooling.
- `docs/`: public architecture/experiment guide and selected engineering records.
- `.local/` and `.venv/`: ignored runtime data, caches, diagnostics and environment.

Never commit credentials, Streamlit secrets, model downloads, vector databases or
raw provider diagnostics. The publication list is explicitly reviewed; a broad
`git add .` is not the publication workflow. Historical documents excluded from
that list remain in the local development checkout.

## Roadmap and portability

- **Advanced retrieval:** measure lexical/BM25 and hybrid baselines; evaluate
  reranking against the same evidence requirements.
- **LangGraph orchestration:** introduce explicit workflow orchestration only
  after the simpler pipeline's behavior and failure handling are understood.
- **Evaluation:** expand the corpus/questions and human-reviewed evidence and
  answer-quality checks; distinguish structural support from semantic faithfulness.
- **Hosted delivery:** Linux CPU packaging,
  hosted secrets, model provisioning, durable retrieval state and protected access.

Replit import/deployment is not complete. Current Linux x86_64 lock metadata
selects roughly 2.46 GiB of Torch/NVIDIA wheel downloads before other packages,
although embeddings run on CPU. No CPU-wheel migration or dependency-pin change
has been made. Hosted memory/startup, Chroma durability, environment-only Pinecone
configuration, public-interface binding and durable access/budget controls need
separate validation. Do not expose the local live UI publicly by changing only
its bind address. No open-source license has been selected yet.
