# Controlled diagnostic retest

This implementation supports one separately authorized semantic LangGraph
diagnostic after an uncertain workflow. It is offline validated; no real ledger
has been amended and no additional inference has been authorized or sent.

## Scope and invariants

- The separate `budget_amendment` command accepts only an existing valid v2
  ledger with eight accounted lifetime slots and the existing $0.10 ceiling.
  It proposes exactly 15 lifetime slots, leaving seven for one graph workflow.
- Historical `legacy`, workflows, call statuses, usage, reservations and pricing
  revisions are preserved as values, without reconciliation, refunds or
  repricing. A private backup preserves the entire original file byte for byte.
- A new independently verified pricing revision can be appended and made active.
  Existing revisions are immutable. Current pricing must have at most 24 hours
  of validity and remain valid at application, reservation and each stage.
- The original v1 migration command is unchanged and is never used to amend v2.
  Only its create-only private-file helpers are reused.
- There is no rollback, ledger creation, automatic recovery or limit-adjustment
  path in the diagnostic runner.

The shared coordinator still reserves seven calls, **184,064 input-bound tokens
and 20,736 output tokens, totaling 204,800**. Each stage estimate is rounded up
to a microdollar using the active revision. The candidate must leave enough of
the unchanged $0.10 cumulative ceiling for that complete reservation. These are
conservative estimates, not billing measurements. No current provider pricing
was verified during this offline implementation.

## Amendment approval and private artifacts

`prepare` defaults to a dry run with no filesystem writes. `--write-artifacts`
creates one private `amendment-<plan-sha256>/` directory beneath the existing
authoritative budget directory. It contains `backup.v2.json`,
`candidate.v2.json` and `plan.json`; directories are mode 0700 and files 0600.
Existing artifacts cannot be overwritten.

The exact plan digest binds source bytes, candidate bytes, original ledger inode,
unchanged lock inode, budget directory identity, canonical authority root,
scope, independently confirmed billing scope, pinned model, old/new caps,
active pricing digest and reviewed invocation contract. `apply` requires separate
approval of that exact plan after candidate review. Approval digests bind an
operator decision; they are not secrets or substitutes for access control.

The existing exclusive lock remains held across authoritative validation,
settlement-guard creation, create-only apply receipts, window installation,
atomic ledger replacement, directory fsync and exact readback. A crash or failed
write leaves either the original state or a guarded, uncertain state requiring
independent recovery review. `settlement.pending` is never automatically cleared
on recovery. An `apply.completed` receipt alone cannot override that guard.

All writers and automatic restarts must be stopped for prepare and apply. The
CLI requires the explicit `--writers-stopped` attestation; it does not inspect
or terminate processes. Conflicting cooperating writers fail the nonblocking
lock check. Unreviewed direct filesystem writes are outside the guarantee.

## Durable diagnostic exclusion

The candidate adds one explicit, optional v2 extension:
`diagnostic_request_sha256`, the SHA-256 of the canonical reviewed request
contract. Existing ledgers without this field retain their behavior. Older
strict v2 readers reject the extension, including if configured for 15 slots.

Applying the candidate also installs a private `diagnostic-window.json` containing
the exact approved plan. Both the durable ledger restriction and the window are
required for the diagnostic. Updated ordinary reservations and readiness checks
reject diagnostic ledgers. Deleting, losing, corrupting or changing permissions
on the window cannot reopen ordinary spending. Nothing removes this restriction
automatically after success or failure.

The approved request contract contains exactly:

| Field | Constraint |
| --- | --- |
| `invocation_id` | Operator-selected, unique existing-coordinator identifier |
| `question_sha256` | SHA-256 of exact UTF-8 question bytes, including whitespace |
| `mode` | Explicit `approved_only` or `draft_review` |
| `backend` | `chroma` |
| `profile` | `llm_routing_grading_semantic_v1` |
| `reference_date` | Explicit ISO date |
| `model` | Existing pinned Qwen model ID |
| `corpus_sha256` | SHA-256 of the verified application's corpus manifest bytes |

No raw question is stored in the plan. Source/manifest verification is repeated
under the lock. The first reservation requires the exact approved candidate
bytes, then uses the existing durable invocation ID and stage permits.
Later invocations are rejected, even when a pre-provider failure consumed zero
call slots or a successful four-call path left some of the seven slots unused.

Stopping other entrypoints remains a deployment prerequisite, including
Streamlit and any older or unbudgeted clients. A filesystem protocol cannot
prevent a separately authorized client from bypassing the coordinator.

## Operator command sequence — requires separate live approval

These commands are documentation, not an instruction to execute a real amendment
or live test. `$APP_ROOT` is the validated application checkout; shared authority
selection and billing/scope configuration must already be independently reviewed.
Private input paths below must be absolute paths beneath that checkout's `.local/`,
with restrictive ownership and permissions. Do not put credentials in command
arguments. `request-contract.json` follows the table above and is reviewed before
preparation; it is not generated from model output.

Keep the operator cap setting at **8** during preparation/application:

```bash
python -m northstar_atlas.budget_amendment --root "$APP_ROOT" prepare \
  --contract "$APP_ROOT/.local/operator/request-contract.json" \
  --confirmed-billing-scope "$REVIEWED_BILLING_SCOPE" \
  --approve-max-attempts 15 --approve-max-reserved-usd 0.100000 \
  --writers-stopped
```

If pricing requires renewal, add `--pricing-revision <private-json-path>` and
`--verified-pricing-sha256 <reviewed-canonical-json-sha256>`. The JSON includes a
new unique `id`, input/output per-million USD rates, official source URL,
`confirmed_at` and `valid_until`. The tool validates the revision and digest;
an operator must independently verify provider pricing and model applicability.

Repeat preparation with `--write-artifacts` only after reviewing the dry run.
Independently review backup, exact candidate, plan digest and conservation of
history. Then separately authorize the concrete candidate:

```bash
python -m northstar_atlas.budget_amendment --root "$APP_ROOT" apply \
  --plan "$REVIEWED_PLAN_SHA256" --approve "$REVIEWED_PLAN_SHA256" \
  --writers-stopped
```

After confirmed readback and explicit live authorization, the isolated runner's
operator cap must match **15**. Existing private-preview and controlled-graph
gates remain required; the runner does not enable or alter them. Keep Streamlit
and all other entrypoints stopped. The separate run approval is SHA-256 of the
ASCII string `diagnostic-run:` followed by the approved plan SHA-256. This binds
run approval to the same reviewed candidate, rather than treating amendment
approval as inference authorization.

```bash
python -m northstar_atlas.diagnostic_retest --root "$APP_ROOT" \
  --plan "$REVIEWED_PLAN_SHA256" --approve-run "$REVIEWED_RUN_SHA256" \
  --invocation-id "$REVIEWED_INVOCATION_ID" \
  --question-file "$APP_ROOT/.local/operator/question.txt" \
  --question-sha256 "$REVIEWED_QUESTION_SHA256" \
  --mode draft_review --backend chroma \
  --profile llm_routing_grading_semantic_v1 \
  --reference-date "$REVIEWED_REFERENCE_DATE" \
  --allow-live --other-writers-stopped
```

The runner reserves before provider construction or private-preview access I/O.
It uses the existing frozen snapshot, pinned MiniLM assets, Chroma checks,
reservation-bound Nebius adapter, router, grader, generator and semantic checker.
Exactly-one transport attempt per stage and the existing bounded internal
regeneration remain unchanged. There is no outer retry or replacement ID.

The process exits after one workflow. It prints only a fixed result summary;
the Python result exposes accepted `PublicAnswer` fields only, never rejected
candidates. Existing bounded provider diagnostics and durable call records stay
private. No raw prompts, answers, response payloads or exception strings are
added to diagnostic storage. Unknown transport/usage outcomes retain the full
workflow reservation. A known returned response rejected by graph validation
retains the existing closed-call accounting; the persistent window still denies
another request. Do not mistake an older latest-failure file for a new event:
correlate its stage timestamp with the new workflow's private call record.

## Recovery and activation boundaries

Do not automatically restore a backup, remove the window/ledger restriction,
clear a settlement guard, refund an uncertain workflow or re-run application.
Before any new reservation, a failed amendment may be investigated for a
separately reviewed recovery. After any new reservation, restoring the old
backup would erase spending history and is prohibited. This tool deliberately
provides no rollback command.

Real activation still requires review of deployed writer versions and shutdown,
authoritative ledger/lock identity, historical aggregates, fresh independently
verified prices, concrete candidate approval, successful guarded readback,
current Chroma/model assets, private-preview access and separate one-workflow
inference approval. No provider prices, real ledger state or deployment access
were inspected here.

## Offline validation

Tests use temporary synthetic ledgers and blocked sockets. Coverage includes
unchanged uncertain history, exact cap/cost limits, append-only pricing and
expiry, candidate/source/inode binding, private file modes and symlinks, pending
state, interprocess lock contention, crash after replacement, failed persistence,
missing-window fail-closed behavior, duplicate identities, full seven-stage
execution with mocked HTTP, timeout/HTTP/envelope failures, safe withholding,
operator/access gates and no workflow retries.

Focused commands:

```bash
PYTHONDONTWRITEBYTECODE=1 python -B -m unittest discover -s tests -p test_budget_amendment.py
PYTHONDONTWRITEBYTECODE=1 python -B -m unittest discover -s tests -p test_diagnostic_retest.py
PYTHONDONTWRITEBYTECODE=1 python -B scripts/validate_evaluations.py
```

Use a file-based network-blocked runner with an `if __name__ == '__main__'`
guard for full-suite execution: spawned concurrency tests cannot restart a
stdin-based test runner. The initial stdin-based run was stopped after that
harness failure; the final file-based run completed successfully.

Validation on the final source:

| Check | Result |
| --- | --- |
| New amendment regressions | 23 passed |
| New one-shot runner regressions | 10 passed |
| Complete network-blocked suite | 820 discovered; 807 passed, 13 skipped; 137.2 seconds |
| Frozen evaluation validator | 15 questions, source hashes, manifest, evidence references, modes and behaviors passed |
| Dependency export | 35 packages; exact approved digest unchanged |
| Publication checks | Local documentation links, whitespace, targeted credential/personal-path scan passed |

The 13 existing opt-in skips are two real MiniLM checks, nine tokenizer/refiner
checks, one cached-model Pinecone-store check and one real Chroma integration
check. They were not enabled or downloaded for this budget-only change.
Streamlit's usual bare-mode test warnings were observed; they were not failures.

Independent review identified and resolved two issues before final validation:
the ledger now retains the diagnostic restriction if the window file is lost,
and manifest validation/hashing occurs under the authoritative lock. Dedicated
regressions cover both. Offline checks do not establish live provider behavior,
pricing accuracy, deployment writer exclusion or distributed-filesystem safety.

The 35-package Replit UI-only export remains unchanged at SHA-256
`0d7a42c564358febb8bde9167fc108c7e022953946cd9fef5e8d1ec38f1e1faf`.
No dependency, model, retrieval, policy, evaluation, offline-launcher or provider
prompt changes are part of this implementation. Related implementation:
[shared budget](Shared-Durable-Live-Budget.md),
[graph wiring](Controlled-Live-Graph-Wiring.md),
[provider diagnostics](Provider-Failure-Diagnostics.md).
