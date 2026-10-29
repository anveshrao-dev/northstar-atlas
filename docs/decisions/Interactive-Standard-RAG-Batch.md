# Private interactive Standard RAG batches

The existing Live Q&A interface accepts arbitrary enterprise-policy questions,
but a one-shot diagnostic approval binds one exact question and ledger snapshot.
A separate batch approval now permits five private Standard RAG submissions while
preserving that one-shot behavior. This implementation has only been exercised
offline with synthetic ledgers and mocked providers. No live batch is approved,
applied or executed by its presence in source.

## Authority and accounting

`northstar_atlas.demo_batch` accepts only the dedicated demo billing scope and an
existing validated version-2 ledger. The supported transitions are **2→7**, then
**7→12** after the first batch's five submissions have been consumed. The second
transition requires another preparation, candidate review, application approval
and live authorization. No further transition, initialization, rollback, refund
or pricing update is exposed.

The initial source must contain the two existing accounted LINEAR workflows. An
optional preserved zero-attempt version-1 donor must pass the authoritative
validator. Every historical workflow, attempt, status, usage record, reservation,
legacy document and pricing revision remains unchanged. The original separate
historical authority and its uncertain graph workflows are not accepted or read.
The cumulative estimated-cost ceiling remains **$0.10**.

Each submission reserves the existing LINEAR envelope: one generator call,
30,720 input-bound tokens and 2,048 output tokens. The existing verified pricing
revision must still be current; preparation independently binds its digest and
checks that five full reservations fit the cumulative ceiling. A reservation
must be at most **$0.01**. Historical reservations are never repriced. If pricing
expires, execution fails closed; pricing refresh requires separate review.

Batch progress counts appended workflows, including closed zero-call setup or
context failures. Such failures cannot reopen a batch submission even if the
existing ledger counts zero provider attempts. Started or uncertain requests keep
the existing conservative accounting. No automatic retry is introduced.

## Approval contract and persistence

The private version-5 batch plan is an authorization artifact, not a new ledger
schema. It binds:

- The existing authority, scope, billing scope and directory/lock device+inode
  identities, plus source bytes and ledger identity at application.
- Prior approval bytes, exact candidate digest, old/new caps and fixed cost caps.
- A unique batch identifier, LINEAR/Pinecone, pinned model, corpus-manifest hash,
  five submissions and both approved-only/draft-review modes.
- The unchanged active pricing revision digest.

The operator must stop competing writers and automatic restarts. Preparation is
read-only by default. `--write-artifacts` creates private create-only backups of
the source ledger and prior window, candidate and plan under the existing budget
authority. Application takes the existing exclusive lock and verifies exact
source bytes, identities, prior approval and candidate again before replacement.
The existing settlement guard, atomic writes, directory fsync and readback are
retained. An interrupted application remains blocked for independent recovery;
never restore an old ledger after spending has begun.

During execution, the ledger's immutable history and configuration are compared
with the digest-approved candidate. Only up to five new LINEAR/Pinecone workflows
may follow it. Every actual request has a unique invocation ID and the existing
question/mode/backend/reference-date/corpus binding. Requests without the batch
approval, graph requests, other backends and duplicate IDs fail closed. Removing
the approval window does not reopen ordinary spending. Filesystem identity changes
still require independent recovery; inode-only matches are not accepted.

This is coordination among reviewed writers, not an OS sandbox against an
operator who can rewrite trusted source or private authority files. Do not run
unreviewed clients or competing entrypoints against this authority.

## Interactive execution and presentation

With explicit batch activation, **Live Q&A** is fixed to Standard RAG and Pinecone.
Both policy modes remain available; reference date is the UI's current UTC date.
Source eligibility, publication confirmation, remote audit, sibling context,
context budget, deterministic grounding and citations use the existing pipeline.
The model and prompt contracts are unchanged. Ordinary-query obligation coverage
remains `not_assessed`; citations do not prove semantic completeness.

The UI displays the shared durable remaining submission count and disables input
when exhausted or unavailable. A lock held by another process also makes status
unavailable; it never creates a fresh allowance. Session opt-in and duplicate
submission guards remain mandatory. Clearing history or restarting does not reset
the batch. Concurrent processes share the same existing ledger and lock.

Each question opens a reservation-bound QAService, asks exactly once and closes it.
The full envelope is persisted before private-access checks, remote retrieval or
provider construction. The existing private-preview verifier runs before setup
and again before the provider call. The existing one-attempt Nebius adapter and
Pinecone integrity checks are retained. Service cleanup and ledger settlement must
finish before the typed `.public` result is durably captured under private
`.local/linear-demo-results/`. Only then may the UI project the result. Formatting
errors cannot invoke QA again. Safe failures never publish rejected candidates.

The current implementation opens resources per submission, rather than caching
reservation-bound providers across questions. This avoids stale permits but may
increase model/loading latency. These costs have not been measured live.

Outside explicitly activated batch Live Q&A, existing UI defaults and behavior
remain unchanged. Offline authored demo remains the initial selection. Policy
search remains retrieval-only. LangGraph is not authorized by a batch plan.
The offline launcher, Replit workflow, dependencies and UI-only export are unchanged.

## Operator sequence (not performed)

Use the existing dedicated authority configuration and independently review the
actual current state. The example commands contain placeholders, not deployment
identity or pricing. Run from the authorized checkout with the existing Python
3.12 environment. Keep all plans/backups private and out of Git.

1. Stop all writers and automatic restarts. Confirm the correct dedicated ledger,
   original two calls, existing cap 2, previous approval, current pricing and
   unchanged $0.10 ceiling. Configure `NORTHSTAR_BUDGET_MAX_ATTEMPTS=2` for preparation.
2. Dry-run the exact proposed batch:

   ```sh
   .venv/bin/python -B -m northstar_atlas.demo_batch --root "$PWD" prepare \
     --batch-id '<unique-batch-id>' --previous-plan '<prior-plan-sha256>' \
     --confirmed-billing-scope northstar-atlas-week2-demo \
     --approve-max-attempts 7 --approve-max-reserved-usd 0.100000 \
     --verified-pricing-sha256 '<independently-verified-active-price-digest>' \
     --writers-stopped
   ```

3. After review, repeat with `--write-artifacts`. Independently inspect source and
   candidate preservation, five-call cost and complete plan. Approve that concrete
   plan digest separately, then apply under the unchanged old-cap configuration:

   ```sh
   .venv/bin/python -B -m northstar_atlas.demo_batch --root "$PWD" apply \
     --previous-plan '<prior-plan-sha256>' --plan '<new-plan-sha256>' \
     --approve '<same-reviewed-new-plan-sha256>' --writers-stopped
   ```

4. Obtain separate approval for five live submissions and the enabled policy
   modes. The execution approval is the SHA-256 of the ASCII string
   `interactive-demo-batch-run:<new-plan-sha256>`. Configure trusted process
   settings (never chat input):

   ```text
   NORTHSTAR_BUDGET_MAX_ATTEMPTS=7
   NORTHSTAR_DEMO_BATCH_ENABLED=1
   NORTHSTAR_DEMO_BATCH_PLAN=<approved-plan-sha256>
   NORTHSTAR_DEMO_BATCH_RUN_APPROVAL=<separately-approved-execution-digest>
   NORTHSTAR_UI_ENABLE_GRAPH=0
   NORTHSTAR_PRIVATE_PREVIEW_CONFIRMED=1
   VECTOR_STORE=pinecone
   ```

   Retain the existing budget scope, billing scope, authority authorization and
   credential configuration. Never put secrets or private identity in documentation.
5. Start the existing `scripts/run_replit_live.sh` only after operational approval.
   It sets the existing controlled-live/UI gates. Batch startup checks local
   authorization, remaining allowance and Secret presence; external private-access
   and Pinecone checks are deferred until a reserved question. Select Live Q&A,
   consent to paid answers and choose a policy scope. No inference is authorized
   merely by selecting a mode.
6. After five submissions, stop. Any second batch requires separate review and
   the same sequence for 7→12, a new batch ID and a new plan. Restart the server
   after an approved amendment so cached coordinators cannot retain stale limits.
   No automatic extension or reset is provided.

## Offline validation

Focused tests use synthetic two-call histories and mocked transports. They cover
history/candidate/backup conservation, exact approvals, stale pricing, changed
source bytes and lock identity, uncertain outcomes, failed persistence, duplicate
and concurrent processes, restart persistence, five-submission exhaustion,
zero-call failures, separately approved future batches, full QA execution,
private-access denial, settled result capture, formatter failure and UI controls.
Streamlit tests distinguish authored demo from newly generated fake-provider
answers; they establish pipeline behavior, not live answer quality.

Reproduction:

```sh
PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  .venv/bin/python -B -m unittest discover -s tests -p 'test_demo_batch*.py'
.venv/bin/python -B scripts/validate_evaluations.py
```

Validation completed on 2026-10-29:

- 18 new focused tests passed (16 batch/runtime and 2 Streamlit interaction tests).
- Full network-blocked suite: **972 run, 959 passed, 13 skipped**, 126.208 seconds.
  Skips are the existing explicitly opt-in cached-model integrations: two MiniLM,
  nine refiner, one Pinecone offline corpus and one Chroma test. No new skip was added.
- M1.8 validator: all 15 questions, source hashes, evidence sections, modes and
  expected behaviors passed.
- Replit UI-only export: 35 packages, byte-identical SHA-256
  `0d7a42c564358febb8bde9167fc108c7e022953946cd9fef5e8d1ec38f1e1faf`.
- Git whitespace and candidate publication scans passed. Dependencies, policies,
  evaluation data, offline launcher and Replit configuration remain unchanged.
- Independent code and documentation review found no blocking issues.

The new AppTest fixture restores Python's original `__main__` after each test,
preventing temporary Streamlit scripts from affecting later multiprocessing tests.
No real ledger, pricing, Secrets, providers or indexes were accessed or changed.

Changed files are [batch authorization and execution](../../src/northstar_atlas/demo_batch.py),
[window dispatch](../../src/northstar_atlas/diagnostic_window.py),
[controlled preflight](../../src/northstar_atlas/live_preflight.py),
[UI runtime](../../src/northstar_atlas/ui/runtime.py),
[UI presentation](../../src/northstar_atlas/ui/app.py),
[batch tests](../../tests/test_demo_batch.py),
[interactive tests](../../tests/test_demo_batch_ui.py), and this report.
