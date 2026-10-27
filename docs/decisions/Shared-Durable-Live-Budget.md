# Shared durable live-inference accounting

## Scope and activation boundary

This implementation selectively adapts the reviewed controlled-live donor's
POSIX locking, atomic writes, pricing checks and private-preview verification.
It preserves the newer routing, grading, semantic checking and UI implementation.
Standard RAG remains the default. No real ledger was read, initialized, migrated,
reset or repriced during this work. No inference or provider requests were made.

At this budget implementation checkpoint, live graph use was hard-disabled in the
UI and configured graph factory. The subsequent
[controlled-live wiring](Controlled-Live-Graph-Wiring.md) adds a conditional
operator gate and per-invocation checks through a separate wrapper; it does not
activate Replit or establish live verification. Live graph use remains disabled
by default. Adding a coordinator is not activation approval. The existing
three-call / $0.10 defaults cannot reserve a seven-call workflow; actual limit
changes require separate approval and must match the reviewed ledger.

## Ledger and lifecycle

`live_budget.PersistentBudget` uses `.local/controlled-live/budget.json` and
`budget.lock`, requiring a 0700 directory and 0600 regular files. Missing files,
symlinks, invalid permissions, unknown schemas, corrupt data, identity mismatch,
unpriced or expired state deny execution. It never creates an initial ledger.

Version 1 is supported for read-only history inspection, including expired price
history. Execution requires version 2. The version 2 format retains the entire
original version 1 document in `legacy`, with its historical costs and attempt
records unchanged and included in shared totals. It records operator-bound scope,
billing scope, provider/model, approved caps, price revisions, active price ID and
workflow records. No migration or pricing-setting command is provided.

Each workflow records its invocation ID, hashed request binding, profile, mode,
backend, price ID, timestamps, status and calls. The binding covers the question,
mode, backend, effective-date reference, model and manifest hash; no question,
prompt, response or credential is stored. Calls record stage/ordinal, input/output
bounds, status, timestamps and numeric usage. Price records use the donor's
official-source and maximum 24-hour validity rules. Historical prices are not
replaced when evaluating old reservations.

One nonblocking `flock` is held for the whole workflow. Reservation and per-stage
start records are durable before transport. Writes use a same-directory 0600
temporary file, file `fsync`, atomic replacement and directory `fsync`. Cooperating
processes on the same durable filesystem share the limits; independently copied
ledgers or replicas without shared POSIX locking do not.

Final settlement first creates and syncs a `settlement.pending` guard. An ambiguous
ledger replacement or interrupted guard cleanup blocks authorization after restart,
even if the ledger already contains a `closed` record. Guard removal follows a
durable ledger commit; cleanup failures attempt to restore the guard and block the
current coordinator. There is no automatic deletion/recovery of an orphaned guard.
These guarantees require working filesystem locks, atomic replacement and `fsync`;
an unavailable/corrupt filesystem must not be restored from an older ledger copy.

Duplicate invocation IDs cannot be replayed. Restarted processes retain pending
reservations without automatically resuming, refunding or deleting them. Known
terminal outcomes release only uncalled stage slots; used stages retain their
maximum reservation, regardless of lower actual token usage. Timeouts, missing
usage, provider errors and uncertain persistence retain the complete workflow
envelope. A failed settlement withholds an otherwise successful UI response.
Recovery and historical reconciliation require an independently reviewed operator
procedure; there is no automatic refund or migration.

## Workflow reservations

| Profile / stage | Maximum calls | Input bound per call | Output per call |
|---|---:|---:|---:|
| `linear_v1` generator | 1 | 30,720 | 2,048 |
| Semantic graph router | 1 | 7,936 | 256 |
| Semantic graph graders | 2 | 28,672 | 4,096 |
| Semantic graph generators | 2 | 30,720 | 2,048 |
| Semantic graph checkers | 2 | 28,672 | 4,096 |

The graph reservation covers **seven calls, 184,064 input-bound tokens and 20,736
output tokens: 204,800 total across calls**. Each stage is priced separately with
upward micro-dollar rounding; the full sum is reserved before resource/provider
opening in the UI. Prices are rechecked before each call. The 32,768 generation
context does not authorize 32,768 input tokens plus output. Lowering output does
not increase the reserved stage input ceiling.

`reserve(WorkflowRequest(...))` yields an invocation-local `WorkflowReservation`.
Its `invoke(provider, stage, ordinal, request)` verifies the pinned model, order,
counts, complete prompt bound, stage input/output ceilings and single attempt.
The Nebius boundary consumes a single-use transport permit for that exact request.
This adds no second charge. Invalid/duplicate stages cannot send requests.
Application graph transitions still enforce negative-check regeneration semantics;
the budget coordinator does not decide policy answers or semantic correctness.

## Integration

- `QAService.ask()` and `GraphQAService.ask()` accept optional `reservation` and
  `invocation_id` arguments. A service configured with `budget` can own reservation
  settlement, or the UI can own it and pass the same handle. No nested reservation
  or double charge occurs. Handles are never cached on reusable providers.
- `PhaseBInvocation` receives its explicit handle across the graph's fresh
  execution context; the existing in-memory graph limits remain independent.
- `ResourceOwner` retains the process counter, mutex, duplicate checks, connection
  cleanup and session-local public history. Controlled mode lazily constructs one
  shared coordinator and reserves before service setup. Only settled safe results
  reach presentation. Offline demos never reserve or initialize provider resources.
- Controlled Nebius requests require a stage permit. Ordinary linear requests and
  the older graph profiles retain their existing behavior outside controlled mode.
- The original graph UI gate remained closed at this checkpoint. The subsequent
  [wiring report](Controlled-Live-Graph-Wiring.md) describes the implemented
  conditional gate; configured live semantic execution requires Chroma and does
  not add Pinecone support.

## Controlled launcher and security

The separate, explicit `bash scripts/run_replit_live.sh` entry point uses the
existing environment, never installs packages, and never initializes or migrates a
ledger. **Do not run it as an offline validation command:** standard-only startup
retains public DNS/private-preview, retrieval-backend and Nebius read-only catalog
checks. After the controlled-live wiring follow-up, graph-authorized startup checks
only local assets, credential presence and full-envelope budget capacity. External
private-access checks occur on submission under the reservation, before provider
construction. Inference still requires UI opt-in and a successful reservation.

The operator supplies `NORTHSTAR_BUDGET_SCOPE`, `NORTHSTAR_BILLING_SCOPE` and an
already reviewed compatible ledger. Billing scope is an operator assertion, not
proof of the provider account's identity. The launcher also requires
`NORTHSTAR_PRIVATE_PREVIEW_CONFIRMED=1`, `REPLIT_DEV_DOMAIN`, and the existing Nebius
Secret. Pinecone configuration/Secrets are required only when explicitly selected
using `VECTOR_STORE=pinecone`; Chroma remains the default. Credentials are never
placed in the ledger or UI.

Private access checks retain the donor's independent public DNS, pinned public TCP
peer, verified TLS hostname, no-cookie/no-credential requests, no redirect following,
and denial checks across four Streamlit paths. The checks repeat before inference.
Unavailable or ambiguous checks fail closed. CORS/XSRF stay enabled. The existing
`.replit`, offline launcher and locked UI-only export are unchanged.

## Validation and remaining prerequisites

The following results are the original budget implementation checkpoint, before
the [graph wiring validation](Controlled-Live-Graph-Wiring.md#offline-validation).
Validation uses contained synthetic ledgers, fake graph providers, mocked HTTP,
process concurrency/crash tests, and blocked external socket connections. Final
validation on 2026-10-26: **672 tests run; 659 passed, 13 skipped**, including
51 new budget/workflow/access/launcher tests. The skips are the existing opt-in
cached MiniLM checks: two embedding tests, nine refinement tests, one offline
Pinecone corpus round-trip and one Chroma integration test. No budget test was
skipped. These model-dependent checks were not needed to change token accounting,
and their model/dependency code remains unchanged.

The M1.8 validator passed all 15 questions, category counts 3/6/3/3, manifest/source
hashes, section references, modes and behaviors. Shell syntax and diff whitespace
checks passed. A focused scan of the changed files found no credential literals,
private key blocks or personal absolute paths. Git staging remained empty.

The offline locked UI export remains exactly **35 packages**, byte-for-byte equal
to the reviewed export, SHA-256
`0d7a42c564358febb8bde9167fc108c7e022953946cd9fef5e8d1ec38f1e1faf`.
`pyproject.toml`, `uv.lock`, `.replit`, offline setup/launcher, all six policies,
manifest and frozen evaluation files match their pre-task hashes. Existing
uncommitted Phase A/B/C1 and retrieval work was preserved.

Independent review identified one important settlement failure: a final atomic
replacement could expose `closed` before a failed directory sync. Failing
regressions reproduced the post-replacement and interrupted-cleanup paths before
the durable pending guard was added. The corrected tests also cover failed sync
after guard removal. All now pass with restart authorization blocked.

Reproduce the offline regression and validator from the checkout (no installation):

```bash
export TMPDIR="$PWD/.local/tmp"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export HF_HOME="$PWD/.local/huggingface" NORTHSTAR_CONTROLLED_LIVE=0
.venv/bin/python -B - <<'PY'
import socket, unittest
from unittest.mock import patch
with patch.object(socket.socket, 'connect', side_effect=AssertionError('network blocked')), \
     patch.object(socket.socket, 'connect_ex', side_effect=AssertionError('network blocked')):
    result = unittest.TextTestRunner(verbosity=2).run(
        unittest.defaultTestLoader.discover('tests'))
raise SystemExit(not result.wasSuccessful())
PY
.venv/bin/python -B scripts/validate_evaluations.py
bash -n scripts/run_replit_live.sh
```

## Exact task file inventory

Modified existing implementation files:

- `src/northstar_atlas/nebius.py`
- `src/northstar_atlas/qa.py`
- `src/northstar_atlas/orchestration.py`
- `src/northstar_atlas/orchestration_assessment.py`
- `src/northstar_atlas/ui/runtime.py`
- `src/northstar_atlas/ui/app.py`

Added files:

- `src/northstar_atlas/live_budget.py`
- `src/northstar_atlas/live_preflight.py`
- `scripts/run_replit_live.sh`
- `tests/budget_fixture.py`
- `tests/test_live_budget.py`
- `tests/test_budget_workflows.py`
- `tests/test_live_launcher.py`
- `tests/test_private_preview_access.py`
- `docs/decisions/Shared-Durable-Live-Budget.md`

## Remaining activation prerequisites

Before activation: review a backup-preserving version 1 to version 2 transition,
prove history/cost conservation against the real ledger without changing it here,
review current prices and billing identity, approve any necessary limits, validate
the durable filesystem/locking model and private preview, and separately authorize
live graph calls. Local tests do not establish Replit filesystem durability,
external privacy, provider billing accuracy or semantic answer faithfulness.

No dependencies, corpus, evaluation questions, publication state, retrieval defaults
or grounding contracts are changed by this work.
