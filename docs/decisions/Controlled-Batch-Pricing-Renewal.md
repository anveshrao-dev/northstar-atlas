# Controlled batch pricing renewal

## Scope and approval boundary

An expired price must not be edited in an existing batch or historical workflow.
The pricing successor preserves the unused 7→12 batch, including its original
batch ID and seven-workflow baseline. It changes neither the twelve-call lifetime
cap nor the $0.10 cumulative estimated-cost ceiling. It creates no spending pool
and performs no provider calls. This implementation has only synthetic offline
verification; it does not establish the state of any deployed authority.

The [interactive batch contract](Interactive-Standard-RAG-Batch.md), private-preview
access checks, [settled result capture](Controlled-Pinecone-Demonstration.md),
Pinecone publication checks and one-attempt Nebius adapter remain in force.
No UI, provider, retrieval, model, dependency or ledger-schema change is required.

## Successor contract

`demo_batch_pricing.py` provides `prepare_renewal`, `validate_renewal_plan` and
`apply_renewal`. Its approval has version 6 and kind
`v2-unused-demo-batch-pricing-renewal`; the accounting document remains version 2.
`demo_batch.py` and `diagnostic_window.py` recognize this distinct approval type.

Preparation requires the original version-5 7→12 batch approval, cap 12, exactly
seven terminal LINEAR/Pinecone workflows and an unchanged corpus. The existing
ledger bytes must equal the original approved candidate bytes. Even one appended
zero-call submission makes the batch ineligible. Reserved workflows and pending
settlement guards are rejected. Historical uncertain workflows remain uncertain
with their full original accounting. A valid zero-attempt legacy donor is retained.

The original candidate is validated as history with `current=False`, solely to
allow its expired price to be inspected. The successor and every live reservation
still require current pricing. The successor appends exactly one unique pricing
revision and updates `active_pricing`; all other ledger fields remain identical.
Its independently reviewed canonical pricing digest is bound into the approval.
The existing validator requires an official Nebius source URL, valid positive
rates and a current interval no longer than 24 hours. A source URL and digest are
not independent proof of current prices: verification remains an operator duty.

Each LINEAR request reserves 30,720 input plus 2,048 output tokens. At the new rates,
the conservative reservation per call must be at most $0.01, and the historical
reserved amount plus five such reservations must fit within $0.10. Historical
prices, usage, statuses and reservations are never recalculated at the new rates.

The successor binds source, candidate, original approval and new pricing digests,
the same scope, billing scope, model and request contract, and current device/inode
identities for the authority, `.local` directory, budget directory, lock and source
ledger. It does not repair changed filesystem identities. The original plan and
candidate remain untouched. Private create-only artifacts include byte-exact
ledger/window backups, candidate and plan, under `demo-batch-pricing-<plan-digest>`
inside the existing budget directory (directory 0700; files 0600).

At runtime, the successor chain is checked against the preserved original
candidate. The successor's seven-workflow prefix is immutable. Only five appended
LINEAR/Pinecone submissions are allowed, in approved-only or draft-review mode.
Duplicate IDs, unrelated profiles/backends, changed corpus and old approval hashes
are rejected. Zero-call failures consume a batch submission; uncertain calls retain
their bounded reservation. A second pricing renewal is explicitly out of scope.

## Locking and failure handling

All writers and automatic restarts must be stopped. Preparation and application
use the existing exclusive lock without replacing its inode. Dry-run creates no
artifacts and writes neither the ledger nor the window. Actual preparation writes
only private artifacts. Separate digest approval is required for application.

Application rechecks original bytes, identities, pricing and the complete candidate
under the lock. The existing settlement guard covers window replacement, atomic
ledger replacement, fsync and readback. These two files are not a single atomic
transaction: a crash between replacements leaves the guard and fails closed.
`apply.started` prevents replay; `apply.completed` records successful readback.
Neither file alone overrides a pending guard. A failure after ledger replacement
can leave new bytes installed but unavailable. Keep writers stopped and seek
independent recovery review; never delete the guard or automatically restore a
backup, particularly after any new reservation. No refund or retry is implied.

## Later Replit operator procedure

These commands are a runbook, not authorization to execute them. Use the reviewed
source in the authoritative Replit checkout, its existing Python 3.12 environment
and existing private authority settings. Never copy or initialize accounting on
another machine. No real renewal was prepared during implementation.

1. **Authorize maintenance and pricing review separately.** Stop competing writers
   and automatic restarts. Independently verify the existing seven-workflow,
   unused 7→12 batch, original approval/candidate, preserved history, authority
   identities and absent settlement guard. Verify current rates for the existing
   approved model using official pricing after separate network approval. Prepare
   a private 0600 JSON pricing revision inside the authorized authority root:
   `id`, `input_per_million_usd`, `output_per_million_usd`, `source`, `confirmed_at`,
   `valid_until`. Timestamps must be timezone-aware; validity cannot exceed 24 hours.
   Approve its SHA-256 over sorted-key, compact JSON (the existing `_encode` format),
   not an arbitrary whitespace variant of the file. Retain verification evidence
   privately; the program does not fetch or independently certify prices.
   After setting `PRICE_FILE` to the independently reviewed private revision,
   compute its canonical digest without printing its contents:

   ```sh
   .venv/bin/python -B - "$PRICE_FILE" <<'PY'
   import sys
   from pathlib import Path
   from northstar_atlas.budget_transition import _decode, _encode
   from northstar_atlas.diagnostic_window import sha
   print(sha(_encode(_decode(Path(sys.argv[1]).read_bytes()))))
   PY
   ```
2. Retain the existing authority, scope and billing configuration. Set
   `NORTHSTAR_BUDGET_MAX_ATTEMPTS=12` without raising any limits. With existing
   process variables `OLD_PLAN`, `PRICE_FILE`, `PRICE_DIGEST` and `BILLING_SCOPE`
   populated from reviewed non-public metadata, run the read-only dry-run:

   ```sh
   .venv/bin/python -B -m northstar_atlas.demo_batch_pricing --root "$PWD" prepare \
     --previous-plan "$OLD_PLAN" --pricing "$PRICE_FILE" \
     --verified-pricing-sha256 "$PRICE_DIGEST" \
     --confirmed-billing-scope "$BILLING_SCOPE" --writers-stopped
   ```

   `PRICE_FILE` must be an absolute path to the existing private file within the
   authorized authority root. Missing authority or artifacts fail closed.
3. **Approve private artifact preparation.** Repeat that command with
   `--write-artifacts`. Review the reported source/candidate/plan digests and
   byte-exact backups. Verify that only the pricing revision list and active
   pointer differ, the original request is unchanged, and five reservations fit.
   Preserve the original artifacts. Do not rerun the earlier cap amendment.
4. **Approve the concrete successor plan separately.** Set `NEW_PLAN` to its
   reviewed digest and `APPROVED_PLAN` to the independently approved same digest:

   ```sh
   .venv/bin/python -B -m northstar_atlas.demo_batch_pricing --root "$PWD" apply \
     --previous-plan "$OLD_PLAN" --plan "$NEW_PLAN" \
     --approve "$APPROVED_PLAN" --writers-stopped
   ```

   Reverify unchanged history/accounting, cap 12, ceiling $0.10, exact candidate
   checksum, current active pricing, absence of the settlement guard and five
   submissions remaining. A changed source or expired price requires review,
   never a forced application or manual artifact edit.
5. **Obtain separate live-execution approval.** The former activation no longer
   works. Use `NORTHSTAR_DEMO_BATCH_PLAN=<successor-plan-digest>` and its separate
   execution digest, computed by `demo_batch.run_approval` as SHA-256 of
   `interactive-demo-batch-run:<successor-plan-digest>`. Preserve all existing
   batch gates, `VECTOR_STORE=pinecone`, graph disabled, approved model and corpus,
   genuine Replit Private Preview verification and Pinecone target/remote audit.
   A process flag alone does not prove access protection. After explicit startup
   approval, use the unchanged launcher with the separately approved execution
   digest in `APPROVED_RUN_DIGEST`:

   ```sh
   export NORTHSTAR_BUDGET_MAX_ATTEMPTS=12
   export NORTHSTAR_DEMO_BATCH_ENABLED=1
   export NORTHSTAR_DEMO_BATCH_PLAN="$NEW_PLAN"
   export NORTHSTAR_DEMO_BATCH_RUN_APPROVAL="$APPROVED_RUN_DIGEST"
   export NORTHSTAR_UI_ENABLE_GRAPH=0
   export NORTHSTAR_PRIVATE_PREVIEW_CONFIRMED=1
   export VECTOR_STORE=pinecone
   bash scripts/run_replit_live.sh
   ```

   Do not spoof Replit settings to launch on another platform. Keep existing
   credential configuration private and unchanged. No command in this report
   grants approval to submit a paid question.
6. Stop on exhaustion, expired pricing or an accounting/access/integrity failure.
   No whole-request retry or additional allowance is authorized. A future renewal
   or partially used batch requires a separate design and approval.

## Offline validation

Synthetic tests cover byte-exact backups, history/uncertain-accounting conservation,
expired donor versus current successor prices, digest approval, authority/lock
identity changes, private permissions, tampered artifacts, pending settlements,
partial application/readback failures, five submissions, zero-call exhaustion,
duplicates, concurrent processes, restart, old-approval rejection and renewed
runtime access checks plus settlement/capture before display. Existing batch UI,
provider, budget and graph regressions remain required. Tests use fake providers
and mocked transport; no real accounting or external service is accessed.

Validation completed on 2026-11-10 with the existing Python 3.12 environment:

- Focused `test_demo_batch*.py`: **42 passed, 0 failures/errors, 0 skipped**,
  including 24 new renewal tests and 18 existing batch/runtime/UI tests.
- Full network-blocked suite: **996 run, 983 passed, 0 failures/errors, 13 skipped**
  in 118.577 seconds. Skips remain two opt-in MiniLM tests, nine refiner checks,
  one Pinecone offline corpus integration and one Chroma/model integration.
- M1.8: all 15 questions, source hashes, manifest pin, evidence sections, modes
  and expected behaviors passed. No answer-quality measurement was performed.
- Locked, offline Replit UI export: 35 packages, unchanged SHA-256
  `0d7a42c564358febb8bde9167fc108c7e022953946cd9fef5e8d1ec38f1e1faf`.
- Python syntax, local documentation links, containment, whitespace and targeted
  publication-content checks passed. Git commands were deliberately not used.

Tests ran with socket connection/DNS operations denied and model libraries in
offline mode; provider interactions used fixtures/mocked transports. Reproduce
test discovery with an equivalent outbound-network blocker around unittest:

```sh
PYTHONDONTWRITEBYTECODE=1 TMPDIR="$PWD/.local/tmp" \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  .venv/bin/python -B -m unittest discover -s tests -p 'test_demo_batch*.py'
PYTHONDONTWRITEBYTECODE=1 TMPDIR="$PWD/.local/tmp" \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  .venv/bin/python -B -m unittest discover -s tests
.venv/bin/python -B scripts/validate_evaluations.py
```

The environment flags alone are not a network firewall. The implementation run
also denied `socket.connect`, `connect_ex`, `create_connection` and `getaddrinfo`
in the test runner and spawned Python workers. Existing Replit launcher tests use
inert subprocess shims; no dependency installation or provider call is made.

Remaining operational prerequisites are independent current-pricing verification,
confirmation of the real unused batch and unchanged authority identities, stopped
writers, concrete artifact/application approval, and separate live authorization.
Neither the Mac checkout nor these tests establish those deployment facts.
