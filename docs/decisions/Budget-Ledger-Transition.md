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
the required shutdown and approval boundaries. Neither is implemented.

## Preconditions and approval contract

Before even preparing, independently stop **all old and new live writers**, disable
automatic restart, and confirm every checkout uses the same authoritative ledger
and lock. Each command requires `--all-writers-stopped`. This is an operator
attestation, not process discovery or proof of shutdown; an exclusive lock alone
cannot prevent an old launcher from restarting later.

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

A crash or failed write/readback/receipt/guard cleanup leaves execution blocked.
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

## Implementation and validation

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

## Remaining live prerequisites

No real ledger was read, copied, transitioned or repriced. No provider was
contacted and no live gate was changed. Before a real transition, separately
approve writer shutdown, the authoritative filesystem and lock semantics,
verified backup retention, account/billing identity, fresh pricing, cumulative
limits including prior consumption, and the concrete prepared plan. Verify
remaining allowance after transition before any independently authorized startup.

This is a single-filesystem POSIX procedure for cooperating, trusted operators.
It cannot authenticate pricing or billing ownership, prove all writers are
stopped, coordinate independent ledger copies or protect against a malicious
operator with the same filesystem access. Actual Replit state and migration
readiness remain unverified.
