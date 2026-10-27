# Separate application and budget-authority roots

## Decision and scope

An isolated application checkout can use an existing, explicitly authorized ledger
under another trusted root. The change extends the existing
[durable coordinator](Shared-Durable-Live-Budget.md); it does not introduce a second
accounting system, initialize a ledger or migrate history.

`PersistentBudget.root` continues to identify the application checkout. Source
verification, the corpus manifest binding, Chroma, MiniLM and frozen-snapshot
checks continue to use that root. `PersistentBudget.authority_root` controls only
the existing `.local/controlled-live/` persistence directory. A budget authority
is storage authority, not an alternative source of policy truth.

Only `live_budget.py` changes in production. Existing linear, graph and UI callers
reuse `configured_budget()` and their existing reservation hooks. The
[controlled graph gate](Controlled-Live-Graph-Wiring.md) remains conditional and
disabled by default; Standard RAG remains the default workflow.

## Operator configuration

Leave both new variables unset to retain application-local storage. To select an
external authority, the trusted process environment must contain both:

```bash
NORTHSTAR_BUDGET_AUTHORITY_ROOT=/trusted/existing-project
NORTHSTAR_BUDGET_AUTHORITY_ROOT_CONFIRMED=/trusted/existing-project
```

These are placeholders, not deployment settings. Confirmation must equal the
exact canonical absolute root; `1`, a different path, empty values and confirmation
without a selection are rejected. An explicit selection equal to the application
root may omit confirmation. No `.env`, chat input or model output selects this
authority. Changing or revoking either setting invalidates an already configured
coordinator instead of redirecting it.

Trusted Python callers can supply `authority_root` and
`authorized_authority_root` to `PersistentBudget`; external selections require
the same exact-path confirmation. This is operator authorization, not an
independent authentication mechanism. Existing scope, billing identity, pricing
and operator-limit checks still apply.

All live processes and checkouts sharing an allowance must select the same root.
Both linear and graph workflows then use the same `budget.json`, `budget.lock`
and settlement guard. A valid application-local ledger is never a fallback when
an explicitly selected authority fails validation.

## Filesystem checks

- Reject missing, relative or noncanonical roots, explicit traversal, aliases and
  symlinks. All persistence paths must remain contained under the selected root.
- Require the authority root and `.local/` to be directories owned by the effective
  user, without group/other write permissions or special mode bits.
- Require `.local/controlled-live/` to be owned by that user with mode `0700`.
- Require existing `budget.json` and `budget.lock` to be regular, single-link files
  owned by that user with mode `0600`. Reject FIFOs, symlinks and hardlinks.
- Recheck paths before locking, reading and writing. Pin directory and lock
  device/inode identities for the coordinator lifetime. Check opened file
  descriptors against current paths. Atomic ledger replacement remains allowed.

These checks apply to the default location too. Permissions on existing storage
are not changed; no lock, directory or initial ledger is created. Existing reservation
writes, atomic replacements and settlement guards occur only at the selected
authority after validation. Missing, corrupt, incompatible, unpriced or expired
state denies inference.

POSIX locking coordinates cooperating processes on one filesystem with suitable
locking and durability semantics. It does not coordinate separately copied
ledgers, independent replicas or protect against a malicious operator with the
same filesystem identity. Selecting a trusted root does not provide OS-level
filesystem isolation.

## Accounting and compatibility

Invocation bindings still use the application's verified manifest, question,
mode, backend, reference date and model identity. Duplicate invocation IDs remain
global to the selected ledger. Stage ordering, single-use transport permits,
durable writes and conservative timeout/crash accounting are unchanged.

The semantic workflow still reserves seven calls with an aggregate envelope of
184,064 input plus 20,736 output tokens: 204,800 total. Linear requests share the
same ledger limits. This change neither sets pricing nor raises limits; the
existing default three-call allowance does not authorize a seven-call workflow.

Schema version 2 is unchanged. Version-1 history remains readable and is not
rewritten; it cannot authorize execution. No migration command was added or run.

## Offline validation

Validation used synthetic roots and ledgers inside the checkout's ignored
temporary directory, fake providers, mocked HTTP transports and blocked sockets.
No real ledger, provider or publication state was accessed.

- Complete suite: **711 tests run; 698 passed, 13 skipped**. No failures or errors.
- New coverage: 24 authority tests plus one external-authority graph wiring test.
- Focused budget suites: 52 tests passed.
- Frozen evidence validator: all 15 questions, source hashes, manifest pin,
  evidence references, modes and behaviors passed.
- The 35-package Replit UI export is byte-for-byte unchanged: SHA-256
  `0d7a42c564358febb8bde9167fc108c7e022953946cd9fef5e8d1ec38f1e1faf`.

Tests cover same-root behavior; explicit external authorization; configuration
revocation; path, ownership and permission failures; missing/corrupt/expired
state; no local fallback or second ledger; two-process lock contention; duplicate
IDs; process-crash persistence; shared linear/seven-stage limits; timeout and
settlement failures; v1 read-only history; application source-hash binding; and
existing offline UI behavior. Graph fixture setup clears ambient authority
settings so an operator's real ledger cannot be selected by those tests.

The 13 unchanged skips are opt-in cached-model checks: two MiniLM embedding tests,
nine refinement tests, one offline Pinecone corpus round-trip and one Chroma/model
integration test. No budget, graph wiring or offline launcher test was skipped.
External graph wiring asserts application-root asset selection with mocked asset
checks; it does not establish that a Replit deployment has those assets.

Focused reproduction, using the existing environment and project-local temporary
storage:

```bash
TMPDIR="$PWD/.local/tmp" PYTHONDONTWRITEBYTECODE=1 \
  .venv/bin/python -B -m unittest discover -s tests -p 'test_*budget*.py' -v
TMPDIR="$PWD/.local/tmp" PYTHONDONTWRITEBYTECODE=1 \
  .venv/bin/python -B -m unittest discover -s tests -p 'test_live_graph.py' -v
.venv/bin/python -B scripts/validate_evaluations.py
```

The full regression run additionally blocked `socket.connect` and `connect_ex`
and set model libraries to offline mode. These tests do not launch controlled live
inference. The existing offline launcher and dependency pins are unchanged.

## Files and remaining activation prerequisites

Changed files: `src/northstar_atlas/live_budget.py`,
`tests/test_budget_authority.py`, `tests/test_live_graph.py`, and this report.

Before deployment, separately review the actual authoritative location and
filesystem permissions, shared POSIX locking, compatible billing/scope identity,
current approved pricing and remaining allowance. Configure the same authority
for every participating process; do not copy a ledger to create another writer.

If the real ledger is still v1, an independently approved history-preserving v2
transition remains necessary, with a verified backup and reconciliation of all
historical attempts and costs. This task supplies neither that transition nor a
rollback that could erase consumed allowance. Uncertain records and settlement
guards still require reviewed recovery, not automatic reset or refund.

Application corpus/model/index verification, private-preview access and explicit
live authorization remain separate prerequisites. No live activation, pricing
refresh, migration or Replit verification is claimed by these offline results.
