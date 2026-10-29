# Pinecone semantic graph authorization recovery

This operator-only path is implemented and offline validated. It does not
authorize deployment, amend a real ledger, refresh real pricing or activate
inference. Existing Standard RAG, graph profiles, UI defaults and earlier
amendment commands retain their behavior.

## Narrow recovery contract

`northstar_atlas.pinecone_graph_recovery` handles the reported device-only
filesystem change. It requires an existing valid version-2 ledger with cap 15,
15 accounted slots, one non-pending legacy attempt and exactly two uncertain
semantic graph workflows. Its prior diagnostic window must be the existing
`v2-diagnostic-amendment`, linked to the latest historical invocation. Both
directory and lock device identities must have changed while their inode
numbers remain unchanged. Other discrepancies require separate review.

Matching inodes do **not** establish authority. Ordinary ledger operations
continue to reject the stale approval. An explicit recovery operation obtains
the same existing, nonblocking exclusive lock, with all writers and automatic
restarts stopped. It validates canonical containment, ownership, restrictive
permissions, regular files, single links and no symlinks. It binds full device
and inode identities for the authority root, `.local`, budget directory, lock
and original ledger. The lock is neither created nor replaced.

Independent continuity review must establish the authoritative path, scope,
billing scope, independently retained prior approval digest, current source
digest and current identities. The private continuity record has exactly:

```text
version = 1
authority = canonical authorized budget-root path
scope, billing_scope = independently confirmed existing identities
source_sha256 = reviewed exact ledger-byte digest
previous_window_sha256 = independently retained prior approval digest
identities_sha256 = digest of the current five identity pairs
independent_evidence_sha256 = digest of the independently reviewed continuity evidence
```

Its canonical JSON digest needs explicit operator approval. `inspect` reports
observed digests and aggregate accounting; these observations are not a
substitute for independent evidence. The program checks binding and consistency,
not the truth or independence of an operator's external evidence. No field is
automatically asserted, and missing or malformed evidence fails closed.

## Candidate and accounting

Preparation changes only the lifetime cap from 15 to 22, the existing diagnostic
request digest, and appends/selects one new independently verified pricing
revision. The cumulative estimated-cost ceiling remains `$0.100000`. Every
legacy/workflow record, call, status, usage, timestamp and original pricing
revision remains identical. Both uncertain reservations remain fully accounted.
The ledger stays version 2; the new **window**, not ledger, is version 3 with kind
`v2-pinecone-graph-recovery`.

The new request binds a unique invocation ID, exact UTF-8 question SHA-256,
`approved_only`, explicit reference date, `pinecone`,
`llm_routing_grading_semantic_v1`, the existing model, corpus manifest digest,
and the frozen snapshot ID, byte digest and corpus fingerprint from the existing
[controlled graph implementation](Controlled-Live-Graph-Wiring.md). The snapshot
file is verified locally. No IDs, questions, dates or alternative assets are
generated or selected automatically.

Pricing must match its separately reviewed digest, be currently valid for no
more than 24 hours, and use a new revision ID. Historical prices are not changed
or used to reprice history. The complete seven-stage reservation is:

- One router, up to two graders, two generators and two semantic checkers.
- 184,064 input-bound tokens plus 20,736 output tokens: 204,800 total.
- The sum of existing per-stage, upward-rounded input/output cost estimates.

Historical reserved cost plus this estimate must fit `$0.10` at preparation,
application and reservation. Current prices were not verified during offline
implementation. The ledger covers Nebius inference estimates, not Pinecone
request charges; no current monetary estimate is implied here.

## Preparation and separate approvals

The application root remains separate from any explicitly authorized budget
authority root. Reuse the existing scope/billing/authority configuration; never
create another ledger. Keep `NORTHSTAR_BUDGET_MAX_ATTEMPTS=15` for `inspect`,
`prepare` and `apply`. Inputs must be private regular files inside the
application's `.local/`. Commands below are templates, not execution approval.

```bash
python -m northstar_atlas.pinecone_graph_recovery --root "$APP_ROOT" inspect \
  --writers-stopped --recover-identity

python -m northstar_atlas.pinecone_graph_recovery --root "$APP_ROOT" prepare \
  --contract "$APP_ROOT/.local/operator/request.json" \
  --continuity "$APP_ROOT/.local/operator/continuity.json" \
  --approve-continuity "$REVIEWED_CONTINUITY_SHA256" \
  --approve-max-attempts 22 --approve-max-reserved-usd 0.100000 \
  --pricing-revision "$APP_ROOT/.local/operator/price.json" \
  --verified-pricing-sha256 "$REVIEWED_PRICE_SHA256" \
  --writers-stopped --recover-identity
```

The request JSON has exactly `invocation_id`, `question_sha256`, `mode`,
`backend`, `profile`, `reference_date`, `model`, `corpus_sha256`, `snapshot_id`,
`snapshot_sha256` and `corpus_fingerprint`. `request_contract()` builds this
from a caller-supplied `WorkflowRequest` and verified local snapshot; it grants
no approval. All approval JSON digests use sorted keys, compact separators,
UTF-8 encoding, no non-finite values and no trailing newline. Question identity
uses the original bytes, including whitespace.

Preparation defaults to a write-free dry run. After separate preparation
authorization, `--write-artifacts` creates a private, create-only
`pinecone-graph-recovery-<plan-sha256>/` directory beneath the existing authority.
It contains `backup.v2.json`, `backup.window.json`, `candidate.v2.json`,
`continuity.json` and `plan.json`. Backups preserve exact original bytes and are
verified on creation and application. Files use mode 0600; the directory 0700.
Original historical artifact directories remain unchanged.

Review the concrete candidate, conservation, all source/identity/pricing/request
bindings and both backups. Application requires both its exact plan digest and
a distinct rebind approval: SHA-256 of ASCII `pinecone-graph-rebind:` followed by
that plan digest (`rebind_approval()`). These digest values bind operator intent;
they are not signatures or authorization to bypass independent review.

```bash
python -m northstar_atlas.pinecone_graph_recovery --root "$APP_ROOT" apply \
  --plan "$REVIEWED_PLAN_SHA256" --approve-plan "$REVIEWED_PLAN_SHA256" \
  --approve-rebind "$REVIEWED_REBIND_SHA256" \
  --writers-stopped --recover-identity
```

Under the same exclusive lock, application revalidates source bytes, both
backups, current identities, fresh pricing, candidate conservation and approvals.
It durably installs the existing settlement guard, records `apply.started`,
atomically replaces/fsyncs the active window and ledger, verifies exact readback,
records `apply.completed`, then removes/fsyncs the guard. This is not a two-file
atomic transaction: the guard prevents intermediate states from authorizing
requests. Failed replacement, fsync, readback or receipt persistence fails closed.
Crash artifacts remain for independent review; no automatic retry or restoration
is provided. A future identity change invalidates this new approval again.

## One separately authorized live invocation

After successful application and separate live approval, the operator configures
cap 22, existing controlled-live/graph/Pinecone opt-ins and private-preview
protection. No command sets these flags. Keep other entrypoints and automatic
restarts stopped. The run approval is SHA-256 of ASCII `pinecone-graph-run:`
followed by the approved plan digest (`run_approval()`).

```bash
python -m northstar_atlas.pinecone_graph_recovery --root "$APP_ROOT" run \
  --plan "$REVIEWED_PLAN_SHA256" --approve-run "$REVIEWED_RUN_SHA256" \
  --invocation-id "$REVIEWED_INVOCATION_ID" \
  --question-file "$APP_ROOT/.local/operator/question.txt" \
  --question-sha256 "$REVIEWED_QUESTION_SHA256" \
  --mode approved_only --backend pinecone \
  --profile llm_routing_grading_semantic_v1 \
  --reference-date "$REVIEWED_REFERENCE_DATE" --allow-live --other-writers-stopped
```

The normal runtime window validation atomically reserves the full seven-call
envelope before provider construction, credential loading or access-check I/O.
The existing `open_live_graph_service` validates frozen assets, pinned MiniLM,
private preview, confirmed Pinecone target, active publication and remote audit.
Audit failure prevents inference. No vectors are uploaded or reindexed. Existing
audit behavior may mark invalid local publication state incomplete; recovery
does not bypass or repair that behavior. Each Nebius call retains its durable
stage permit, single transport attempt, token ceiling and sanitized diagnostics.
The router schema correction and all grounding/semantic validators are reused.

The runner exits after one accepted result, rejection or uncertainty. It does
not repair output, retry the workflow or allocate another ID. Bounded internal
retrieval/regeneration remains part of the same reservation. Only validated
public fields are returned to Python callers; the CLI prints a safe summary.
A pre-provider failure may close with zero calls under existing accounting, but
its recorded ID and changed candidate digest still prevent reuse. Removing the
window does not remove the ledger's restriction. Unrelated linear and graph
submissions remain denied even if unused slots remain after success.

## Recovery boundaries and activation prerequisites

Never restore old backups after spending or clear guards automatically. This
utility offers no rollback command. Partial application, failed persistence or
changed identities require a separate evidence-based recovery review. File
locks coordinate reviewed writers on the same filesystem, not independent
replicas, arbitrary clients or unreviewed direct writes. Stopped writers and
disabled automatic restarts remain deployment requirements, not a CLI proof.

Live activation still requires approved source synchronization, independent
authority continuity, reviewed request/candidate/rebind approvals, fresh prices,
private access, confirmed existing Pinecone publication assets and successful
remote audit, then separate run authorization. Offline fixtures do not establish
current account availability, runtime identity continuity, provider compatibility
or answer correctness. No real ledger or provider was accessed for this work.

## Offline validation

Validation used synthetic ledgers, mocked HTTP, real graph orchestration and
blocked sockets; no provider inference or cloud retrieval was performed.

- Recovery-specific suite: **33 passed** in 12.914 seconds.
- Focused recovery/budget/amendment/diagnostic/linear-demo/Pinecone/graph suite:
  **190 passed** in 40.427 seconds.
- Full network-blocked regression suite: **899 run, 886 passed, 13 skipped**
  in 94.326 seconds; no failures or errors. The existing skips are two cached
  MiniLM embedding tests, nine cached-model refinement tests, one full-corpus
  Pinecone offline integration and one cached-model Chroma integration. These
  require separate opt-in; no new recovery test is skipped.
- M1.8 validator: all 15 frozen questions, source hashes, manifest pin, evidence
  references, retrieval modes and expected behaviors pass.
- Replit UI export: unchanged, 35 packages; SHA-256
  `0d7a42c564358febb8bde9167fc108c7e022953946cd9fef5e8d1ec38f1e1faf`.
- Independent read-only implementation review found no blocking correctness or
  security issues. Documentation link, five-file scope and whitespace checks pass.

The tests cover exact history/backups, default dry run, separate approval
digests, missing/incorrect continuity, source changes, parent/device/inode/lock
changes, permissions, ownership, hardlinks, artifact symlinks, spawned-process
contention, a process crash after ledger replacement, failed window replacement,
fsync, ledger write, readback and completion receipt. Runner tests exercise a
complete seven-stage graph, separate live gates, duplicate/unrelated denial,
private-preview and remote-audit failures, expired pricing, altered snapshots,
router timeout/HTTP rejection/malformed envelopes and sanitized diagnostics.

Reproduce individual checks from the authorized checkout with its existing
environment (the full reported run additionally used a socket-blocking wrapper
around discovery, including spawned processes):

```bash
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export NORTHSTAR_CONTROLLED_LIVE=0 TMPDIR="$PWD/.local/tmp"
.venv/bin/python -B -m unittest discover -s tests -p test_pinecone_graph_recovery.py
.venv/bin/python -B -m unittest discover -s tests
.venv/bin/python -B scripts/validate_evaluations.py
```

Only the five approved implementation/test/report files changed. No dependency,
policy, frozen evaluation, retrieval default, offline launcher, publication state
or Git index was changed. All operational activation approvals remain outstanding.
