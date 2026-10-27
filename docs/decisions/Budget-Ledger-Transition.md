# History-preserving budget ledger transition

## Scope and design

`python -m northstar_atlas.budget_transition` is an operator-only tool for an
existing valid version-1 ledger. It is not imported by the UI or inference path,
does not initialize a ledger, and does not enable live execution.

It reuses the [shared coordinator](Shared-Durable-Live-Budget.md) and
[budget-authority selection](Shared-Budget-Authority-Root.md). Application-root
verification, explicit external-root authorization, ownership, permissions,
containment and symlink protections remain mandatory. All operations acquire the
existing exclusive `budget.lock`; its inode is not replaced.

A two-step prepare/apply protocol lets an operator review a concrete candidate
before authorizing replacement. An interactive single-step confirmation would
make independent review harder; automatic conversion on startup would bypass
the required exclusion and approval boundaries. Neither is implemented.

## Preconditions and approval contract

In the original all-writers-stopped mode, before even preparing, independently stop **all old and new live writers**, disable
automatic restart, and confirm every checkout uses the same authoritative ledger
and lock. Each command in that mode requires `--all-writers-stopped`. This is an operator
attestation, not process discovery or proof of shutdown; an exclusive lock alone
cannot prevent an old launcher from restarting later.

The separate [cooperating-writer mode](#cooperating-writer-exclusion) below permits
preparation and application under a reviewed lock/schema contract. It never
interprets cooperation as a shutdown attestation. Rollback still requires all
writers stopped, regardless of the preparation mode.

The trusted process environment supplies the existing budget scope, independently
verified billing scope and approved lifetime call cap using
`NORTHSTAR_BUDGET_SCOPE`, `NORTHSTAR_BILLING_SCOPE` and
`NORTHSTAR_BUDGET_MAX_ATTEMPTS`. The tool never sets them. Preparation separately
requires matching billing-scope confirmation and explicit approval of both caps.
The cumulative dollar cap must match the coordinator's policy; the CLI retains
the existing `$0.100000` ceiling. Defaults and spending limits were not raised.

Provide a private JSON pricing file under the application root's `.local/` with
exactly these fields: `id`, `input_per_million_usd`, `output_per_million_usd`,
`source`, `confirmed_at`, `valid_until`. It must be a single-link regular file,
owned by the effective user, mode `0600`, without symlinks.

The operator must separately verify the rates, model/account applicability and
official pricing source. `--verified-pricing-sha256` binds that confirmation to
SHA-256 of UTF-8 `json.dumps(pricing, sort_keys=True, separators=(',', ':'),
allow_nan=False)` bytes, not the file's optional indentation. Existing validation
requires approved source hosts, positive rates, timestamps current at preparation
and application, and a validity interval no longer than 24 hours. The hash and
source URL do not independently establish that the supplied rates are true; this
tool makes no provider calls or pricing lookup.

## Commands and review artifacts

The placeholders below require separately reviewed deployment values. These
commands are an operator procedure, not authorization to run a real transition.
First perform a no-write dry-run from the authorized application checkout:

```bash
.venv/bin/python -B -m northstar_atlas.budget_transition --root "$PWD" prepare \
  --pricing-file .local/reviewed-pricing.json \
  --verified-pricing-sha256 '<reviewed-canonical-pricing-digest>' \
  --confirm-billing-scope '<independently-confirmed-billing-scope>' \
  --approve-max-attempts '<approved-lifetime-call-cap>' \
  --approve-max-reserved-usd '0.100000' \
  --all-writers-stopped --dry-run
```

Repeat the same command without `--dry-run` to prepare private review artifacts.
Neither form replaces the authoritative ledger. Dry-run creates no files at all.
Preparation creates a `0700` directory beneath the existing authority's
`.local/controlled-live/transition-<plan-sha256>/`, containing `0600` files:

- `backup.v1.json`: exact original bytes, including whitespace; fsynced, read back
  and checksum-verified before a candidate is accepted for review.
- `candidate.v2.json`: fully validated version-2 candidate, containing the entire
  parsed original document under `legacy` and an empty `workflows` list.
- `plan.json`: binds the authority path, original lock device/inode, original-byte
  hash, candidate-byte hash and verified pricing digest. Its own SHA-256 is the
  approval identifier.

These files contain private deployment identity and accounting history. Keep
them out of Git and public logs. Existing or partial preparations are never
overwritten. Review the full candidate and backup privately, then provide separate
approval of the exact plan digest:

```bash
.venv/bin/python -B -m northstar_atlas.budget_transition --root "$PWD" apply \
  --plan '<reviewed-plan-sha256>' --approve '<reviewed-plan-sha256>' \
  --all-writers-stopped
```

Apply revalidates every artifact, the unchanged original ledger, current pricing,
runtime identities/caps and original lock identity under the lock. Tampering,
stale approval, changed source bytes, replacement locks and repeat applications
are rejected. It never edits historical attempt IDs, statuses, reserved costs or
pricing. Historical charges continue to count against the new lifetime limits.

An open v1 `reserved` attempt blocks transition and requires separate review.
A completed `failed_or_uncertain` attempt is preserved with its entire original
reserved charge; it is not turned into success, refunded or repriced. Expired
historical pricing remains unchanged while a separately verified current revision
governs future workflows.

## Atomic write and recovery

The tool durably creates the existing global settlement guard before writing
`apply.started`. It then reuses the coordinator's temporary-file write, fsync,
atomic replacement and directory fsync. Exact-byte readback and normal v2
validation must pass before `apply.completed` is persisted and the guard is
removed durably. The lock remains held throughout.

A crash during guarded settlement, before successful guard cleanup, leaves
transition operations and current v2 writers blocked. Caught settlement failures
also retain or restore the guard. A crash before guard creation can leave the
original ledger without a marker; a crash after durable cleanup can leave a
successfully installed v2 ledger eligible under normal readiness checks. Legacy
v1 writers do not recognize the settlement guard; their behavior is described
below. In stopped-writer mode, keep the independently established shutdown in effect.
Do not delete a settlement guard, overwrite partial artifacts or retry with a
different plan to bypass uncertain state. There is no automatic recovery or
restore operation. Keep writers stopped, retain the backup and receipts, and
review which bytes reached durable storage before any separately approved
recovery. A failed preparation before authoritative mutation can leave a private
partial bundle; it is also retained for review rather than overwritten.

## Rollback boundary

Rollback requires separate approval: SHA-256 of UTF-8
`rollback:<plan-sha256>` bytes, also reported by preparation. It requires a
completed application, no settlement guard, and an authoritative ledger exactly
equal to the original approved candidate with **zero version-2 workflow records**.
Even a closed zero-call reservation prohibits rollback. Changed pricing,
identities, caps or any other ledger edits also prohibit it.

```bash
.venv/bin/python -B -m northstar_atlas.budget_transition --root "$PWD" rollback \
  --plan '<reviewed-plan-sha256>' --approve '<separately-approved-rollback-digest>' \
  --all-writers-stopped
```

The tool restores the byte-exact backup using the same guarded atomic mechanism,
records rollback receipts and verifies the result. Expired new pricing alone does
not prevent this pre-reservation rollback; it never permits erasing spending.
The original plan cannot be applied again after rollback. No automatic restore
exists, and post-reservation rollback is forbidden rather than treated as a way
to recover allowance. Keep writers stopped after rollback; v1 is read-only to the
current coordinator.

## Cooperating-writer exclusion

This additional mode is explicitly selected with `--cooperating-writers` instead
of `--all-writers-stopped`. There is no automatic mode selection or fallback.
Both prepare and apply also require `--acknowledge-cooperating-writers-only`, a
private `--writer-provenance` file and an independently approved
`--approve-writer-provenance` SHA-256 digest. No application request imports the
transition tool, and this mode does not enable the live graph gate.

The operator's acknowledgment covers **only reviewed writers that cooperate with
the same existing lock and validate the authoritative schema before inference**.
It does not discover processes, authenticate running code, block a writer using a
different ledger/lock, or protect against administrators replacing paths. A hash
of an invented attestation does not establish compatibility. Independently review
the deployed ledger implementation, provider integration and budget-enforcement
configuration, including restarted instances and every checkout, before approving
their provenance. An unbudgeted provider path does not satisfy this contract.

The provenance file is exact-byte hashed, not whitespace-normalized or
canonicalized. It must be a private, owned, single-link regular `0600` JSON file
under the application root's `.local/`, with no symlink or traversal. Its exact
schema is:

- `version`: integer `1`; `kind`: `reviewed-v1-lock-cooperation`.
- `authority_root`: the exact canonical, independently authorized budget root.
- `directory_identity`: `[st_dev, st_ino]` of its `.local/controlled-live/`.
- `lock_identity`: `[st_dev, st_ino]` of its existing `budget.lock`.
- `writers`: one to 32 reviewed legacy writer identities, each with a unique
  `writer_id` (one to 80 letters, digits, underscores or hyphens),
  `ledger_source_sha256` and `provider_source_sha256` (64 lowercase hex digits).
- `guarantees`: exactly these six fields, all literal JSON `true`:
  `all_deployed_writers_reviewed`, `same_ledger_and_lock`,
  `exclusive_lock_through_inference_and_settlement`,
  `fresh_v1_schema_check_before_inference`,
  `no_cached_authorization_or_unlocked_writes`,
  `no_lock_or_directory_replacement`.

The review must establish that all deployed legacy writers and their provider
paths match the listed hashes and guarantees; current v2 writers must use the
reviewed shared coordinator. The tool validates this independently approved
attestation and filesystem identity. It does not execute supplied code, download
a writer, or independently attest deployed processes. An unreviewed, malformed,
incomplete, mismatched or changed provenance file fails closed.

### Commands and candidate binding

Using separately reviewed pricing, identity, caps and provenance, prepare a dry
run without claiming that writers are stopped:

```bash
.venv/bin/python -B -m northstar_atlas.budget_transition --root "$PWD" prepare \
  --pricing-file .local/reviewed-pricing.json \
  --verified-pricing-sha256 '<reviewed-canonical-pricing-digest>' \
  --confirm-billing-scope '<independently-confirmed-billing-scope>' \
  --approve-max-attempts '<approved-lifetime-call-cap>' \
  --approve-max-reserved-usd '0.100000' \
  --cooperating-writers --acknowledge-cooperating-writers-only \
  --writer-provenance .local/reviewed-writers.json \
  --approve-writer-provenance '<independently-approved-exact-file-sha256>' \
  --dry-run
```

Repeat without `--dry-run` to create a private review packet. This adds
`writer-provenance.json`, a byte-exact copy, to the existing backup/candidate/plan
files. The cooperating-mode **plan format** is version 2, binding exclusion mode,
provenance hash and directory identity as well as all existing plan fields. The
original stopped-mode plan format and approval digests remain unchanged. Both
produce the same validated v2 ledger schema; provenance is not inserted into
accounting history.

After reviewing the concrete candidate, apply with separate plan approval and
the same independently reviewed provenance:

```bash
.venv/bin/python -B -m northstar_atlas.budget_transition --root "$PWD" apply \
  --plan '<reviewed-plan-sha256>' --approve '<reviewed-plan-sha256>' \
  --cooperating-writers --acknowledge-cooperating-writers-only \
  --writer-provenance .local/reviewed-writers.json \
  --approve-writer-provenance '<independently-approved-exact-file-sha256>'
```

The unchanged exclusive lock is acquired before authoritative validation and
held through replacement, file/directory fsync, byte readback, schema validation,
receipts and settlement-guard cleanup. Contention fails without waiting or making
an inference request. The same inode/ownership/permission protections remain.
Source bytes are checked again before replacement. Any legitimate v1 spending
between prepare and apply invalidates the old candidate; it must never be
overwritten with an earlier backup. Changed provenance, directory/lock identity,
history, billing scope, limits or stale pricing also invalidate application.
Applying a plan under the other exclusion mode is rejected.

### Compatibility evidence and failure boundaries

The reviewed legacy source was verified against its review manifest and checksum
inventory. Relevant identities are source hashes, not credentials or deployment
identifiers:

| Reviewed source | SHA-256 |
| --- | --- |
| Legacy `live_budget.py` | `83099dc0ac95feb766b457bf3a82e989e8804f718f0a3a049aa8e9f9dc985cb6` |
| Legacy `nebius.py` | `13038d5ad8b48e7a572adb8e5abf99c2343a1f2ca53821d078a34875439641f1` |

Legacy `PersistentBudget.attempt()` acquires nonblocking exclusive `flock`, reads
and validates the v1 schema, durably records the reservation, then yields to the
provider while retaining the lock through outcome settlement. The reviewed
provider wraps its generation POST in that context **when configured with the
reviewed inference budget**. Its unbudgeted path uses `nullcontext()` and is not
covered: the deployment review must exclude that path before approval. A restarted cooperating writer is
denied while transition holds the lock. After v2 installation, its strict v1
schema check rejects the ledger before fake or real inference can begin.

**A lock is not a durable shutdown switch.** The legacy implementation ignores
`settlement.pending`. If the transition crashes or fails *before* replacement,
the OS releases the lock and the unchanged valid v1 ledger can authorize a later
legacy request under its original limits. Such spending changes the source bytes
and must be conserved. Where guarded settlement left a recovery marker, the
transition tool and v2 coordinator remain blocked; neither silently retries or
rolls back. If failure occurs *after* v2 replacement but before successful guard
cleanup, legacy writers reject its schema and current writers reject the recovery
marker. If the process dies before creating the marker, v1 may remain unchanged;
if it dies after durable cleanup, v2 may be fully installed even though the caller
received no success message. Inspect actual state before separate recovery
approval. A corrupt ledger authorizes neither implementation.

On **any** denied or uncertain cooperating-mode operation, stop all writers and
disable restarts before private recovery review. Do not delete markers, force
the candidate onto changed history or restore old bytes to recover allowance.
The mode makes no permanent-exclusion guarantee after a process crash. Operators
requiring that guarantee must use the original shutdown mode with external
restart inhibition. Rollback supports cooperating-mode packets only after the
separate stopped-writers attestation and existing distinct rollback approval;
all zero-reservation/exact-candidate restrictions remain. Restoring v1 can make
legacy inference eligible again, so do not restart writers automatically.

## Implementation and validation

### Original stopped-writer implementation

New files: `src/northstar_atlas/budget_transition.py`,
`tests/test_budget_transition.py`, and this report. Modified:
`src/northstar_atlas/live_budget.py`, extracting its unchanged schema checks for
candidate validation, adding an identity-checked byte read and exact-byte writes,
and optional guarded pre-replacement/readback callbacks. Existing request paths
retain their defaults. The earlier authority-root changes remain intact.

Validation used synthetic temporary ledgers only, blocked sockets, fake providers
and offline model settings. Transition tests exercise actual CLI commands against
synthetic checkouts, POSIX lock contention, a child-process crash, backup bytes,
approval binding, history/cost conservation, expiry, invalid history, permissions,
markers, failed receipt/write/readback, duplicate attempts and rollback denial.
The receipt-failure regressions first reproduced an ordering gap; moving receipt
creation after the durable global guard resolved it.

Focused reproduction using the existing environment:

```bash
TMPDIR="$PWD/.local/tmp" PYTHONDONTWRITEBYTECODE=1 \
  .venv/bin/python -B -m unittest discover -s tests -p 'test_budget_transition.py' -v
.venv/bin/python -B scripts/validate_evaluations.py
```

Final offline regression: **740 tests run; 727 passed, 13 skipped**, with no
failures or errors. All 29 transition tests passed. The unchanged skips are
opt-in cached-model tests: two embedding tests, nine refinement tests, one
Pinecone corpus round-trip and one Chroma/model integration. No accounting or
transition test was skipped. Independent review found no remaining blocker after
the receipt-ordering correction and independently passed the 27 non-CLI
transition tests. The frozen 15-question validator passed. The 35-package
Replit UI-only dependency export
remains byte-for-byte unchanged (SHA-256
`0d7a42c564358febb8bde9167fc108c7e022953946cd9fef5e8d1ec38f1e1faf`).

### Cooperating-mode extension

Only `src/northstar_atlas/budget_transition.py`, the new
`tests/test_budget_cooperating.py`, and this report changed. The shared coordinator,
provider, live gate, application dependencies and Replit workflows are unchanged.
The separate mode and provenance checks reuse the original locking, schema,
backup, approval and settlement mechanisms. Original stopped-mode commands and
plan digests remain compatible.

All **23 cooperating-mode tests** and **29 existing transition tests** passed.
They cover explicit approval, profile separation, invalid provenance, source and
lock/directory changes, contention, restarted writers during replacement,
directory sync and readback, crashes before/after replacement, recovery guards,
history/cost conservation, stale pricing/billing, no ledger initialization and
stopped-only rollback. Public tests use a small synthetic legacy protocol probe;
they do not depend on private source bundles.

Separately, four checks executed the checksum-verified legacy implementation with
synthetic ledgers and fake inference bodies: a positive authorization control,
lock exclusion followed by v2 schema rejection, a pre-replacement crash, and a
post-replacement crash. These confirmed the documented legacy recovery-marker
limitation. Manifest and source checksum checks passed without changing the
reviewed source. No real ledger or provider was accessed.

Final network-blocked regression: **763 tests run; 750 passed, 13 skipped; no
failures or errors**. The skips remain the opt-in cached-model checks listed in
the historical results; no budget or transition test was skipped. Streamlit's
bare-mode `ScriptRunContext` warnings occurred without test failures. The frozen
15-question validator passed. The 35-package Replit UI-only export retained the
exact SHA-256 above. Git whitespace and local report-link validation passed.
Independent safety review found no blocking defect under the explicitly attested
cooperation model; its diagnostic, test-probe and crash-boundary clarifications
were incorporated.

Additional focused reproduction:

```bash
TMPDIR="$PWD/.local/tmp" PYTHONDONTWRITEBYTECODE=1 \
  .venv/bin/python -B -m unittest discover -s tests -p 'test_budget_cooperating.py' -v
```

## Remaining live prerequisites

No real ledger was read, copied, transitioned or repriced. No provider was
contacted and no live gate was changed. Before a real transition, separately
approve writer shutdown or independently reviewed cooperating-writer exclusion,
the authoritative filesystem and lock semantics,
verified backup retention, account/billing identity, fresh pricing, cumulative
limits including prior consumption, and the concrete prepared plan. Verify
remaining allowance after transition before any independently authorized startup.

This is a single-filesystem POSIX procedure for cooperating, trusted operators.
It cannot authenticate pricing or billing ownership, prove all writers are
stopped, coordinate independent ledger copies or protect against a malicious
operator with the same filesystem access. Actual Replit state and migration
readiness remain unverified.
