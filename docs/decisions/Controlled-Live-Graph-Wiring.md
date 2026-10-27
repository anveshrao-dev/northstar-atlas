# Controlled live semantic-graph wiring

## Scope

The optional semantic graph can now use the existing Nebius adapter through the
shared version-2 budget coordinator. This completes application wiring, not live
activation. Standard RAG remains the default. No ledger was initialized, migrated,
repriced or reset; no provider calls were made during implementation.

The earlier [budget implementation report](Shared-Durable-Live-Budget.md) records
the previously closed graph gate. This follow-up replaces that hard-coded gate
with an explicit operator gate and mandatory per-invocation checks. It does not
change graph nodes, eligibility, grounding, semantic regeneration or retrieval
defaults. The offline launcher and `.replit` remain unchanged.

## Invocation and resource ownership

`ResourceOwner.submit()` reserves the complete graph envelope before opening the
controlled graph wrapper. The reservation is **seven calls: 184,064 input-bound
tokens plus 20,736 output tokens, 204,800 in aggregate**. Current input/output
prices and upward per-stage rounding determine its maximum cost. A startup
readiness check is advisory; the atomic reservation independently checks the
available balance under the shared lock.

`open_live_graph_service()` accepts only the semantic profile, Chroma, the pinned
Qwen model, and an open graph reservation from the same coordinator and checkout.
Every `ask()` validates the question/mode/date binding, current pricing, frozen
assets and private access before provider construction or routing. It calls the
existing `open_graph_qa_service()` with an injected, budget-required Nebius adapter.
That generic factory's restrictions remain unchanged. The provider uses one
transport attempt; no automatic retries or fallback are added.

The existing invocation hooks bind router, grader 1/2, generator 0/1 and semantic
checker 1/2 to durable single-use stage permits. The budget still enforces order,
exact token ceilings and mandatory usage accounting. Invalid responses remain
withheld. Timeouts, unknown usage and uncertain persistence retain the full
workflow reservation; successful settlement releases only uncalled slots.

Only graph/provider resources are cached. No invocation handle is cached. Each
submission rechecks assets/access and obtains its own reservation. Session history,
duplicate IDs, nonblocking process serialization and disconnect cleanup retain
their existing behavior. The UI renders only the existing validated public result.

## Frozen local assets

The wrapper verifies the existing 189-chunk snapshot by its exact byte digest,
then reuses catalog verification against current manifest and source hashes:

- Snapshot ID: `e55dcdeca17453b27d45f4d3663ef72b9ba3e635f43689899a8d66b52cf27860`.
- Snapshot SHA-256: `6a752f13edef08fd73619f2b63a274ace8058e44592d62c97005fbc6b392ae27`.
- Corpus fingerprint: `20f5ec1d95989e7c6f6b12579d089dea5be0fe815fd5ba357902d195d49dfac9`.
- MiniLM revision: `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`, normalized 384-dimensional
  vectors and 256-token embedding inputs.

The snapshot remains under `.local/pinecone/snapshots/` for historical compatibility;
reading this local JSON does not use Pinecone or its credentials. Chroma must report
ready state, the matching fingerprint and both expected/actual count 189. All pinned
model files must already exist. Official model-cache links to blobs inside the
checkout are allowed for reading; links outside it are rejected. No download or
reindex path exists in this opener. Local readiness is checked on every invocation,
including reuse; this adds local verification overhead.

## Operator gates and activation prerequisites

All four process-level flags must equal `1` for graph selection to become available:
`NORTHSTAR_CONTROLLED_LIVE`, `NORTHSTAR_UI_ENABLE_LIVE`,
`NORTHSTAR_UI_ENABLE_GRAPH`, and `NORTHSTAR_PRIVATE_PREVIEW_CONFIRMED`.
These are operator configuration, not `.env` values, question parameters or UI
authorization. Paid-session consent remains mandatory. Selecting graph with Pinecone
is disabled and rejected; there is no silent backend switch.

The flags alone cannot authorize execution. A compatible version-2 ledger, matching
scope/billing identity, current pricing, enough capacity for the entire envelope,
verified local assets, valid credentials and successful independent private-preview
checks are still required. Access checks repeat before each provider stage. A price
expiring during access verification fails closed before transport.

`NORTHSTAR_BUDGET_MAX_ATTEMPTS` optionally supplies a separately approved lifetime
call cap as a strict positive decimal integer. It must exactly match the existing
ledger. Absence keeps the existing **three-call** default; the dollar ceiling remains
**$0.10**. No larger operational cap is selected by this change. The default cannot
reserve seven calls, and changing an environment flag never rewrites ledger history.
Limit changes and any version-1 transition still require independent approval.

With graph authorization configured, the separate live launcher's preflight checks
only local assets, Secret presence and full-envelope budget capacity. It does not
construct a provider or contact external endpoints before reservation. Actual
private-access checks occur on submission under the reservation. Standard-only
startup retains its existing external access/model-catalog checks. Do not execute
the live launcher as an offline test command.

Before Replit activation: approve and execute a history-preserving ledger transition
separately, review fresh pricing and limits, verify filesystem locking/durability and
private access, synchronize reviewed source, provision the full pinned runtime and
existing Chroma/model/snapshot assets separately, then authorize a bounded live test.
The 35-package offline UI environment cannot execute the semantic graph. Current
Replit assets, ledger state, account identity and pricing were not inspected here.

## Offline validation

Tests exercise the real graph and Nebius serialization/permit path using HTTPX mock
transport, authored responses, contained synthetic ledgers and blocked network
sockets. Cases cover seven ordered calls, duplicate rejection, shared session limits,
resource reuse/cleanup, uncertain transport, expired/corrupt/insufficient budgets,
revoked operator authorization, wrong/closed reservations, private-access failure,
snapshot/index mismatch, internal/external model links and Streamlit selection.
Existing budget tests cover concurrent processes, restart recovery, duplicate stage
permits, persistence failures and token ceilings; existing graph tests cover policy
isolation and all earlier profiles.

Focused tests can be run without installing packages:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m unittest discover -s tests -p test_live_graph.py
.venv/bin/python -B scripts/validate_evaluations.py
```

The network-blocked full regression run on 2026-10-26 completed **686 tests:
673 passed, 13 skipped**. All 14 new wiring/UI tests passed. The unchanged skips
are two opt-in cached MiniLM tests, nine refinement tests, one offline Pinecone
corpus round-trip and one Chroma integration test. No budget, graph, transport
permit, lifecycle or UI wiring test was skipped. The Streamlit harness emitted its
existing bare-mode ScriptRunContext warning; no application failure occurred.

The frozen evaluator validator passed all 15 questions, category counts 3/6/3/3,
source hashes, evidence references and modes. Shell syntax, changed-file scan and
documentation links passed. The UI-only export remains byte-for-byte identical:
35 packages, SHA-256
`0d7a42c564358febb8bde9167fc108c7e022953946cd9fef5e8d1ec38f1e1faf`.
Dependency manifests, policies, corpus manifest, evaluation data, `.replit`, offline
setup/launcher and Git index match the pre-task state. Private diagnostics remain
under ignored `.local/reviews/live-graph-wiring/`.

These tests establish offline behavior, not real provider availability, Replit
durability, billing accuracy, retrieval completeness or semantic faithfulness.

Independent read-only review found no blocking issues. One minor test improvement
is deferred: explicitly invalidate assets between two cached submissions and assert
that the second makes no HTTP call. The runtime already revalidates on each ask;
current tests separately cover asset rejection and cached-resource reuse.

## Task file inventory

- Added `src/northstar_atlas/live_graph.py`, `tests/test_live_graph.py` and this report.
- Modified `src/northstar_atlas/live_budget.py`, `src/northstar_atlas/live_preflight.py`,
  `src/northstar_atlas/ui/runtime.py`, `src/northstar_atlas/ui/app.py` and
  `tests/test_ui_app.py`.
- Updated comments only in `scripts/run_replit_live.sh` to describe the new
  preflight branch. Its startup commands and the offline launcher are unchanged.
