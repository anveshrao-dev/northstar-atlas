# Private provider failure diagnostics

## Failure boundary and correction

The Nebius adapter previously collapsed response-envelope and usage failures into
the same safe `ProviderError`. `WorkflowReservation.invoke()` retained only
`failed_or_uncertain`; the graph's invocation wrapper then converted exceptions
to `_ProviderFailure` and public `provider_failed`. That intentionally safe public
conversion discarded the category needed for diagnosis.

The adapter now attaches a fixed category without changing its public error
messages. The durable invocation boundary saves sanitized metadata before the
graph conversion. No graph routing, prompt, model, retrieval, grounding or public
result contract changed. The ledger remains version 2 with its existing schema;
no migration or historical rewrite is needed.

## Diagnostic contract

Only these fields are serialized:

| Field | Meaning |
| --- | --- |
| `version` | Diagnostic format version `1`; separate from ledger version |
| `category` | One of the fixed categories below |
| `stage` | `router`, `grader`, `generator` or `semantic_checker` |
| `ordinal` | Existing validated stage ordinal, 0–2 |
| `started_at` | Existing application-recorded stage start timestamp |
| `transport_attempted` | Whether execution entered the HTTP transport; `null` for an uninstrumented adapter |
| `http_status` | Observed integer HTTP status, or `null` when unavailable |

Categories are `transport_timeout`, `connection_error`, `transport_error`
(other HTTP transport errors), `http_rejection`,
`response_envelope_validation`, `usage_validation`, `request_validation`
(adapter rejection before dispatch), and `unexpected_internal`.

A context-local observation is created for each durable stage invocation. The
HTTP transport wrapper marks dispatch before calling the underlying transport
and records status after headers return. A body-read timeout can therefore have
HTTP status 200; a connection error normally has no status. Consuming a stage
permit does **not** imply transport dispatch. Dispatch does **not** establish that
the provider received the request, completed inference or billed for it.

Unknown usage remains rejected by the budget coordinator and is classified as
`usage_validation`. Unexpected internal exceptions are not relabeled as provider
usage errors. The envelope category describes the outer provider response and
finish state; downstream routing, grounding and citation validation keep their
existing rejection paths.

## Privacy, storage and accounting

The latest failure is stored at `.local/controlled-live/provider-failure.json`
under the configured **budget-authority root**, which may differ from the source
checkout. Storage uses the existing exclusive budget lock, a 0700 directory and
a 0600 regular file. Existing paths must pass ownership, permissions, hard-link
and symlink checks. Non-regular files are rejected without blocking settlement.

The record is at most 512 bytes. Atomic replacement uses one fixed
`provider-failure.pending` file, also at most 512 bytes, with file and directory
fsync. A crash may leave that pending file; subsequent diagnostic writes refuse
to overwrite it or allocate additional files. Review it privately before any
operator-directed cleanup. No automatic recovery or deletion of stale artifacts
is introduced.

No exception messages, stack traces, API keys, headers, prompts, response bodies,
provider payloads, invocation identifiers, billing identifiers or credential
fingerprints are recorded. Diagnostics are not added to UI/public responses or
logs. Their parent `.local/` remains ignored by Git.

Diagnostic storage is best-effort: a storage or path-validation failure preserves
the original provider exception and still runs existing accounting finalization.
It cannot authorize another call or release an uncertain reservation. A missing
or older diagnostic is not proof of success; use its timestamp when reviewing
the latest failed stage. Readiness failures before stage invocation and failures
while saving the initial durable stage are outside this provider diagnostic
boundary. Neither is represented as an HTTP attempt.

Controlled inference still allows exactly one transport attempt. There are no
automatic retries or refunds. Timeouts, missing usage and provider failures keep
the existing conservative full-workflow reservation. The seven-call envelope is
unchanged: 184,064 input tokens plus 20,736 output tokens, totaling 204,800.

## Offline validation

Tests use synthetic temporary checkouts/ledgers and mocked HTTP with socket
connections blocked. They cover every category, missing status, timeout after
headers, no retry, unknown usage, private bounded storage, unsafe file paths,
failed atomic replacement, stale pending files, external authority roots,
invocation isolation, sanitized output and full reservation retention. A graph
integration regression verifies that the router category survives privately
while the existing safe public error and rejected-answer withholding remain.

Focused provider/budget/graph validation: **192 passed**. The frozen dataset
validator passes all 15 questions, source hashes, manifest pins, section
references, modes and expected behaviors. Replit's 35-package UI-only export is
unchanged, SHA-256:
`0d7a42c564358febb8bde9167fc108c7e022953946cd9fef5e8d1ec38f1e1faf`.

Full network-blocked regression: **787 discovered, 774 passed, 13 skipped** in
61.5 seconds, with no failures or errors. The existing skips comprise two real
MiniLM embedding tests, nine real-tokenizer refinement tests, one cached-corpus
Pinecone integration and one cached-model Chroma integration. They require
explicit opt-in and were not enabled. Streamlit bare-mode context warnings are
expected in its offline UI tests. No provider calls or real-ledger access occurred.

Focused reproduction, using the existing environment without installation:

```bash
TMPDIR="$PWD/.local/tmp" PYTHONDONTWRITEBYTECODE=1 \
  .venv/bin/python -B -m unittest discover -s tests -p test_provider_diagnostics.py
.venv/bin/python -B scripts/validate_evaluations.py
```

## Limits and next live-test prerequisites

The earlier router failure cannot be classified retrospectively from the generic
`provider_failed` / `failed_or_uncertain` records alone. No original category or
HTTP status should be inferred. This task did not inspect or change the real
Replit ledger and made no provider requests.

Before another live test: review and synchronize these source changes; verify
the authoritative ledger/lock and private directory; preserve all existing
consumed or uncertain reservations; recheck existing pricing, access, assets and
workflow readiness under the approved policy; and obtain explicit authorization
for the next bounded live invocation. This correction does not activate live
execution, reset attempts or increase spending limits.

Related implementation: [controlled live graph wiring](Controlled-Live-Graph-Wiring.md)
and [shared durable budget](Shared-Durable-Live-Budget.md).
