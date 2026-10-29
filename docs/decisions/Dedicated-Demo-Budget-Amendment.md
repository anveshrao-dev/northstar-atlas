# Dedicated Standard RAG demo budget amendment

This operator-only, offline-tested path increases an **existing** version-2 demo
ledger from one lifetime inference slot to two. It does not initialize a ledger,
create another spending authority, change the historical graph authority, or
make provider calls. Application requires approval of the concrete candidate;
live execution requires a separate approval afterward.

## Accepted source and allowed changes

`northstar_atlas.demo_budget_amendment` requires the existing billing scope
`northstar-atlas-week2-demo`, a one-slot policy, the unchanged $0.100000 cumulative
estimated-cost ceiling, and exactly one consumed LINEAR workflow whose reserved
estimate is $0.003687. The first workflow must have one issued generator call
and be closed or uncertain, with no pending writer. Its complete record,
including usage, status and original pricing, is preserved. An uncertain call
is not refunded or reclassified.

The narrowly supported source has `legacy: null` and no existing diagnostic
window or request restriction. Existing windows, unknown layouts, pending
settlement markers, and incompatible histories fail closed; they are not removed
or bypassed. The actual deployed demo ledger was not inspected in this task.
Compatibility must be checked against this contract before any real application.

The candidate changes only:

- `max_attempts`: 1 → 2.
- `diagnostic_request_sha256`: the digest of one reviewed request contract.
- Optionally, an append-only independently verified pricing revision and its
  active pointer. Historical pricing revisions are never edited or repriced.

The request binds a unique invocation ID, SHA-256 of the exact UTF-8 question,
`linear_v1`, `pinecone`, `approved_only`, an explicit reference date, the existing
model identity, and the application corpus-manifest digest. Draft review,
Chroma, graph profiles, altered questions/dates and duplicate IDs are rejected.
No question text is embedded in amendment artifacts.

The unchanged LINEAR envelope is one generator call with an upper bound of
30,720 input tokens and 2,048 output tokens. The new conservative reservation
must be **at most $0.010000**, including microdollar rounding. Together with the
preserved $0.003687, this is at most $0.013687, within the unchanged $0.10 ceiling.
These are application reservation estimates, not measured provider billing or a
limit on Pinecone read fees or other applications.

The original authority containing two uncertain graph workflows is not an
accepted input to this tool. Its records and reservations remain untouched;
nothing is imported from it, reconciled, or transferred into the demo ledger.

## Locking, approval and persistence

Stop competing writers and automatic restarts first. The tool requires the
explicit `--writers-stopped` acknowledgment and uses the existing exclusive lock
without replacing its inode. This coordinates reviewed cooperating writers;
it is not an operating-system barrier against arbitrary clients.

Preparation defaults to a read-only dry run. `--write-artifacts` creates only
private, create-only artifacts beneath the existing authority:

```text
.local/controlled-live/demo-amendment-<plan-sha256>/
  backup.v2.json
  candidate.v2.json
  plan.json
```

The backup preserves exact source bytes and is verified after writing. Artifact
directories are 0700 and files 0600. The digest-approved plan binds source and
candidate digests, application corpus/request, scope, billing scope, model,
limits, active pricing, and current device-and-inode identities of the authority,
`.local`, budget directory, lock and source ledger. Paths and identity values
are private operator metadata, not publication material.

Application requires a separately approved exact plan SHA-256. It reconstructs
the allowed candidate, revalidates pricing and history, and rejects changed
source bytes, changed identities, altered backups/candidates or repeated apply.
It holds the existing lock through settlement-guard creation, restriction-window
installation, atomic ledger replacement, fsync and readback. The window is a
new strict `v2-dedicated-demo-amendment` approval kind (version 4); the ledger
schema remains version 2. The recorded current identities remain mandatory at
runtime, including across process restarts.

A failed write or uncertain readback leaves the existing settlement guard and
fails closed. There is no rollback command or automatic backup restoration.
Never restore the pre-amendment ledger after a new reservation. Preserve failed
artifacts and obtain an independent recovery review with writers stopped.

## Concrete operator approval procedure

These commands document a future approved operation; implementation approval
alone does not authorize using them on a real ledger or invoking inference.

1. Confirm the existing **demo** authority and scope. Keep the historical graph
   authority excluded. Use the existing `NORTHSTAR_BUDGET_SCOPE`,
   `NORTHSTAR_BILLING_SCOPE` and, if applicable, the reviewed external authority
   root plus its exact confirmation. Set `NORTHSTAR_BUDGET_MAX_ATTEMPTS=1` for
   preparation/application. Do not create any missing ledger or lock.
2. Independently review the precise question and reference date. Prepare a
   private JSON contract using `linear_demo.request_contract(budget, request)`
   with a `WorkflowRequest` for `linear_v1`, `pinecone`, `approved_only` and one
   new approved ID. The question digest must match the exact question-file bytes;
   no trimming or newline normalization is performed.
3. Independently verify the active pricing revision. Supply its canonical JSON
   SHA-256 (sorted keys, compact separators, including the revision ID). Pricing
   must be current with at most 24-hour validity. To use a new verified revision,
   pass its private JSON file as `--pricing-revision`; its ID must be new and it
   is appended without modifying old prices. No prices are fetched automatically.
4. Run the dry run, then review the source identity, history, request binding,
   cost and candidate digests:

```bash
python -m northstar_atlas.demo_budget_amendment --root "$APP_ROOT" prepare \
  --contract "$APP_ROOT/.local/operator/demo-request.json" \
  --confirmed-billing-scope northstar-atlas-week2-demo \
  --approve-max-attempts 2 --approve-max-reserved-usd 0.100000 \
  --verified-pricing-sha256 "$VERIFIED_PRICING_SHA256" --writers-stopped
```

5. After permission to prepare the concrete artifacts, repeat with
   `--write-artifacts`. Review `backup.v2.json`, `candidate.v2.json`, `plan.json`
   and the reported digests. Obtain separate approval of **that exact plan**.
6. Apply only the approved plan, still using the one-slot configured policy:

```bash
python -m northstar_atlas.demo_budget_amendment --root "$APP_ROOT" apply \
  --plan "$REVIEWED_PLAN_SHA256" --approve "$REVIEWED_PLAN_SHA256" \
  --writers-stopped
```

7. Stop after successful application and obtain separate authorization for the
   one live request. Configure the cap as two only after the approved readback.
   The existing [one-shot runner](Controlled-Pinecone-Demonstration.md) accepts
   this new restricted approval with the named billing scope. Its run digest is
   `linear_demo.run_approval(REVIEWED_PLAN_SHA256)`, which is distinct from the
   application approval. It still requires explicit live flags, stopped writers,
   private-preview checks, current pricing, credentials, existing Pinecone target
   confirmation, publication integrity and remote audit.

Ordinary interactive and graph entrypoints remain blocked by the installed
window. The runner reserves the one-call envelope before provider construction,
uses one transport attempt, records the typed public result after settlement,
and never retries after failure. Subsequent requests are rejected, even if the
new invocation fails before transport. No new ID is generated automatically.

## Offline validation and limitations

Focused synthetic tests exercise exact history and cost conservation, private
backups, dry-run behavior, approval and request binding, current/appended pricing,
exact one-cent rounding, corrupt/pending history, changed identities, process
lock contention, failed persistence, unchanged historical graph authority,
duplicate rejection, timeout accounting, and the actual one-shot QA runner with
a mocked transport and effective approved fixture evidence.

```bash
python -B -m unittest discover -s tests -p test_demo_budget_amendment.py
python -B -m unittest discover -s tests -p test_budget_amendment.py
python -B -m unittest discover -s tests -p test_live_budget.py
python -B -m unittest discover -s tests -p test_linear_demo.py
python -B -m unittest discover -s tests -p test_result_capture.py
python -B scripts/validate_evaluations.py
```

Run tests with sockets blocked and caches/temp files confined to `.local/`.
Synthetic tests establish contract behavior, not current deployment identity,
pricing accuracy, provider availability, or real answer quality. A future real
operation must independently establish those prerequisites and obtain the
separate candidate and inference approvals. No real ledger, provider, vector
index, policy, embedding or dependency was changed during implementation.

Validation results: all 23 new amendment tests passed. The combined amendment,
budget, linear-runner and result-capture selection passed **103 tests** in
21.865 seconds with sockets blocked. The full network-blocked suite discovered
**950 tests: 937 passed, 13 existing opt-in checks skipped**, zero failures,
in 98.410 seconds. Skips cover two cached MiniLM checks, nine refiner checks,
one cached-model Pinecone-store check and one cached-model Chroma check.
Existing Streamlit bare-mode warnings were non-failing.

The M1.8 validator passed all 15 frozen questions and source-hash, manifest,
evidence-section, mode and behavior checks. No changes appeared in dependency
manifests, source policies, frozen evaluations, embeddings, vector adapters,
PersistentBudget, result capture or Replit configuration. Link, whitespace and
publication-pattern checks passed. An independent read-only code review found
no critical or important blockers. No files were staged or committed.
