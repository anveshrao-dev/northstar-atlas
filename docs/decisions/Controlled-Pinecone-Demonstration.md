# Controlled Pinecone Standard RAG demonstration

This is an offline-validated operator path for one separately approved Standard
RAG invocation. It has not amended a real ledger or run a live demonstration.
It does not enable the UI or change the default retrieval backend.

## Narrow amendment

`northstar_atlas.linear_demo` accepts only an existing valid version-2 authority
with exactly 15 accounted lifetime slots: one completed legacy attempt and two
uncertain seven-slot semantic graph workflows. The existing graph diagnostic
window must be present, private, valid, and approved by its exact plan digest.
The ledger's restriction must match that window, and its latest workflow must
be the previously approved graph invocation. No new ledger is created.

The only candidate changes are:

- Lifetime cap 15 → 16; the cumulative estimated-cost ceiling stays $0.100000.
- A new diagnostic request digest, bound to the approved LINEAR/Pinecone request.
- One independently verified, uniquely identified pricing revision, made active.

All legacy and workflow records remain identical as values, including calls,
statuses, usage, timestamps, original pricing revisions, and uncertain
reservations. A byte-exact backup preserves the original document formatting.
Historical costs are not repriced or refunded. The new revision must be current,
have at most 24-hour validity, and match the separately reviewed canonical JSON
SHA-256. Approval of a digest is not independent verification of its prices.

The unchanged LINEAR envelope reserves one generator call: 30,720 input-bound
tokens and 2,048 output tokens. Cost is rounded up to a microdollar using the
existing coordinator. Historical reserved cost plus this envelope must fit the
unchanged ceiling. QAService retains its normal 1,024-token generation output
limit; the larger reservation is the existing conservative LINEAR allowance.
The budget accounts for Nebius inference estimates, not Pinecone read-unit fees.

## Preparation, approval and guarded replacement

All writers and automatic restarts must be stopped. The existing exclusive
lock is held for authoritative validation and application. Preparation defaults
to a read-only dry run. `--write-artifacts` creates a private, create-only
`linear-demo-<plan-sha256>/` directory under the existing budget authority, with:

- `backup.v2.json`: exact original ledger bytes.
- `backup.window.json`: exact original diagnostic-window bytes.
- `candidate.v2.json`: the proposed ledger.
- `plan.json`: the complete approval binding.

The plan binds source/candidate digests, authority and directory identities,
ledger/lock identities, prior window digest, billing scope, model, exact limits,
pricing revision, invocation ID, question SHA-256, explicit policy mode,
`pinecone` backend, `linear_v1` profile, reference date and corpus manifest digest.
Question text is not stored in the plan. Whitespace is part of question identity.

Application requires separate approval of the concrete plan digest. It
reconstructs and validates the candidate, verifies original bytes and identities,
then uses the existing durable settlement guard. The new private window is
atomically installed and fsynced, followed by atomic ledger replacement, fsync
and exact readback. Any uncertain write leaves the guard in place. This is not
a multi-file filesystem transaction: the guard makes intermediate states deny
all execution. There is no automatic recovery, rollback or retry.

The existing graph amendment and runner remain graph-only. The new version-2
plan has a distinct `v2-linear-demo-amendment` kind. Shared window validation
recognizes only the two exact plan/profile/backend combinations. Older readers
that do not recognize this new plan reject it rather than enabling ordinary
spending.

## One-shot execution

The runner requires separate run approval: SHA-256 of the ASCII string
`linear-demo-run:` followed by the approved plan digest. It also requires the
exact invocation ID and question digest, explicit mode/backend/profile/date,
controlled-live opt-in and a stopped-writer attestation.

Before provider construction or access-check I/O, it reserves the full LINEAR
envelope under the authoritative lock. It requires the existing private-preview
check as the coordinator's pre-request hook and verifies it before setup and
again through the stage permit. Pricing is checked on reservation, setup and
the generator stage. No separate spending pool or process-local override exists.

The existing `open_qa_service(backend='pinecone')` path loads the confirmed target,
checks owner confirmation, connects to the intended existing index, loads its
active publication and performs the existing remote integrity audit before
retrieval/generation. This audit reads remote records; it does not upsert,
rebuild or provision an index. Existing audit failure behavior can mark local
publication state incomplete; this implementation does not bypass that behavior.
Cached MiniLM assets and all existing publication/configuration files remain
prerequisites. No model provisioning or index repair is performed here.

QAService uses the unchanged Nebius serializer, one HTTP attempt, reservation
stage permit, grounding, citation validation and safe public-result contract.
Only accepted public fields are returned to Python callers. After private result
capture, the CLI prints the accepted public answer and a fixed outcome summary.
Failures expose only the existing safe summary, not rejected candidates, raw
responses or internal diagnostics.
Ordinary-question obligation coverage remains `not_assessed`.

After success, rejection, provider uncertainty, or a pre-provider failure, the
invocation cannot be repeated. A pre-provider failure can close with zero calls,
but its durable identity and changed ledger digest still deny another attempt.
The window and ledger restriction remain active. Removing the window cannot
enable ordinary Standard RAG or LangGraph reservations. Provider failures retain
conservative accounting and the existing bounded sanitized diagnostics.

## Operator commands — not authorization to execute

Use the same reviewed authority, scope and billing scope as the existing ledger.
Keep `NORTHSTAR_BUDGET_MAX_ATTEMPTS=15` for preparation and application. Input
files must be private regular files beneath the application's `.local/`.
The contract has the same fields as the
[graph diagnostic request](Controlled-Diagnostic-Retest.md), with backend
`pinecone` and profile `linear_v1`.

```bash
python -m northstar_atlas.linear_demo --root "$APP_ROOT" prepare \
  --contract "$APP_ROOT/.local/operator/linear-request.json" \
  --previous-plan "$PREVIOUS_APPROVED_PLAN_SHA256" \
  --confirmed-billing-scope "$REVIEWED_BILLING_SCOPE" \
  --approve-max-attempts 16 --approve-max-reserved-usd 0.100000 \
  --pricing-revision "$APP_ROOT/.local/operator/reviewed-price.json" \
  --verified-pricing-sha256 "$REVIEWED_PRICE_SHA256" --writers-stopped
```

Review the dry run. Repeat with `--write-artifacts` only to prepare the private
candidate. Independently review both backups, historical conservation, pricing,
request identity and the candidate before separately approving application:

```bash
python -m northstar_atlas.linear_demo --root "$APP_ROOT" apply \
  --previous-plan "$PREVIOUS_APPROVED_PLAN_SHA256" \
  --plan "$REVIEWED_DEMO_PLAN_SHA256" --approve "$REVIEWED_DEMO_PLAN_SHA256" \
  --writers-stopped
```

Only after successful guarded readback and explicit live authorization, set the
operator cap to 16. Set `NORTHSTAR_CONTROLLED_LIVE=1`; require the existing private
preview configuration and separately verified Secrets/target/publication assets.
Keep every other live entrypoint and automatic restart stopped:

```bash
python -m northstar_atlas.linear_demo --root "$APP_ROOT" run \
  --plan "$REVIEWED_DEMO_PLAN_SHA256" --approve-run "$REVIEWED_RUN_SHA256" \
  --invocation-id "$REVIEWED_INVOCATION_ID" \
  --question-file "$APP_ROOT/.local/operator/question.txt" \
  --question-sha256 "$REVIEWED_QUESTION_SHA256" \
  --mode approved_only --backend pinecone --profile linear_v1 \
  --reference-date "$REVIEWED_REFERENCE_DATE" --allow-live --other-writers-stopped
```

The mode must match the reviewed contract; `draft_review` requires explicit
selection and preserves proposed-policy labels. No command changes graph gates,
removes restrictions or authorizes an additional invocation.

## Recovery and remaining prerequisites

Never restore a backup after new spending, remove settlement guards, refund
uncertain reservations, or rerun with a new identity automatically. A partial
application requires independent recovery review with all writers stopped.
Old window and ledger backups are private history, not instructions to roll back.
Filesystem locks coordinate reviewed writers on one filesystem, not unrelated
clients, independent replicas or direct unreviewed writes.

Live operation still needs separate approval of the exact request and candidate,
current independently verified prices, sufficient remaining estimated budget,
deployed writer exclusion, private-preview protection, and existing Pinecone
target confirmation plus successful remote audit. Offline tests do not establish
account compatibility, live model behavior, semantic answer completeness or
provider billing. No real authority, secrets, prices or publication records were
read or changed to build this feature.

## Offline validation

Focused tests use temporary synthetic ledgers, socket blocks, HTTPX mock
transports, the actual QAService and a fake Pinecone adapter. They exercise
history conservation, approval binding, single-call reservations, identity and
backend/profile isolation, private-preview failure, remote audit ordering,
uncertain outcomes, duplicate rejection, locking and guarded persistence.

```bash
PYTHONDONTWRITEBYTECODE=1 python -B -m unittest discover -s tests -p test_linear_demo.py
PYTHONDONTWRITEBYTECODE=1 python -B scripts/validate_evaluations.py
```

Full-suite execution used a file-based network-blocked runner with a main guard
so spawned concurrency tests also ran safely. Final results:

| Check | Result |
| --- | --- |
| New focused demonstration tests | 23 passed |
| Complete network-blocked regression suite | 846 discovered; 833 passed; 13 existing opt-in checks skipped; 105.5 seconds |
| Frozen evaluation validator | 15 questions, source hashes, manifest, evidence references, modes and behaviors passed |
| Replit UI-only export | 35 packages; byte-for-byte unchanged |
| New report links and candidate whitespace/path/credential-pattern checks | Passed |

The 13 skips are two cached MiniLM checks, nine tokenizer/refiner checks, one
cached-model Pinecone-store check and one real Chroma integration check. No
assets were downloaded or opt-in provider checks enabled. Streamlit emitted its
usual bare-mode context warnings; they were not failures. Existing graph,
amendment, Standard RAG and provider regressions remain passing.

The unchanged UI export SHA-256 is
`0d7a42c564358febb8bde9167fc108c7e022953946cd9fef5e8d1ec38f1e1faf`.
No dependencies, policies, frozen questions, Nebius serialization, graph
prompts, retrieval algorithms, index contents or live gates changed.


## Durable public-result capture

The one-shot runner now wraps its existing authorized execution with
`northstar_atlas.result_capture.capture_result`. This selectively adapts the
transferred recorder without introducing a second runner or spending authority.
It reads `QASuccess.public` and `QAFailure.public`; neither result type exposes
an `.answer` or `.error` attribute.

After the initial explicit run-approval checks, a create-only, fsynced
`in_progress` marker is claimed at
`.local/linear-demo-results/<invocation-id>.json` under the application root.
The existing authoritative reservation, private-preview checks, Pinecone
confirmation/audit, service initialization, and single `service.ask()` invocation
then run unchanged. Accepted content is recorded only after service cleanup and
durable reservation finalization succeed. Formatting and CLI presentation follow
successful record persistence. The result directory is owner-only (0700), files
are 0600, and symlinks and unsafe writable parent directories are rejected.
Existing `.local/` need not be made private: the result subdirectory is private.

The bounded JSON record includes accepted public text, answer kind, text digest,
citation identities, document versions/status, source hashes, chunk IDs, and
allowlisted stage/backend/mode/call-count/token-usage fields. It does not serialize
the complete diagnostics object. Failures record only an allowlisted error code;
raised exceptions produce a fixed `outcome_unknown` record and propagate to the
existing safe-error boundary. Exception names, messages, rejected model content,
credentials, raw prompts and raw provider responses are not recorded.

Atomic replacement checks the claimed file identity, fsyncs the new file and its
directory, and precedes publication. Existing claims/results are never overwritten
by a repeat invocation. The ledger's duplicate rejection remains authoritative,
including if a result artifact is removed. Result records are not authorization,
settlement evidence, refunds, or a new spending pool. Formatter failures retain the
saved result. Interrupted or failed persistence leaves a blocking artifact and
withholds the answer; operators must inspect accounting without automatically
retrying or changing invocation IDs.

The record limit is 32 KiB, with at most 8,000 answer characters and 40 citations;
oversized content fails closed rather than being truncated. A disk failure or
process interruption can prevent final capture even after inference completes.
An `in_progress`/`outcome_unknown` artifact is not proof that no call occurred.
This recorder improves retention, not answer faithfulness, and adds no permission
to invoke an exhausted or otherwise unauthorized ledger.

Reproduction uses the existing environment with network access blocked:

```bash
python -B -m unittest discover -s tests -p test_result_capture.py
python -B -m unittest discover -s tests -p test_linear_demo.py
python -B scripts/validate_evaluations.py
```

Live execution still requires separate approval of the exact request and invocation,
a compatible existing authorization and remaining reservation allowance, current
verified pricing, private-preview protection, valid credentials, and the existing
Pinecone target/publication/remote-audit checks. No real ledger was accessed or amended,
and no provider calls were made during this offline integration.


Result-capture validation: 37 focused tests passed (28 runner tests and nine
recorder tests). The complete network-blocked suite discovered 927 tests:
914 passed, 13 existing opt-in integration checks skipped, zero failures
(110.167 seconds). The skips remain two MiniLM checks, nine refiner checks,
one cached-model Pinecone-store check, and one cached-model Chroma check.
The M1.8 validator passed all 15 questions and verified source hashes, manifest,
evidence references, modes and behaviors. Existing Streamlit bare-mode warnings
were non-failing. Documentation links, Git whitespace and publication-pattern
checks passed. Dependency manifests, policies, frozen evaluation data, retrieval,
QAService, budget schema and authorization logic were unchanged.
