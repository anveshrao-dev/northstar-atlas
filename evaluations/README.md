# Northstar Atlas Evaluation Dataset

This directory defines a fifteen-question development benchmark, authored before
retrieval and answer generation were implemented. Fixed questions, expected
evidence, and explicit failure conditions let later experiments compare configurations against
the same task rather than select examples after seeing favorable answers.

The dataset is synthetic, version 0.1, and subject to project review. It is not a
record of model performance, published company policy, or operational approval.

## 1. Files and Local Validation

- [questions.json](questions.json): questions, expected behavior, evidence labels,
  qualifiers, forbidden inferences, and the pinned corpus-manifest hash.
- [Corpus manifest](../data/corpus_manifest.json): the sole policy-file allowlist
  for this experiment; reviews and evaluation answers are not retrieval sources.
- [Validator](../scripts/validate_evaluations.py): standard-library structural and
  integrity checks, with no model calls, installations, or file writes.
- [Regression tests](../tests/test_validate_evaluations.py): valid/invalid local
  fixtures for the validator, not RAG/model evaluations.
- [Design decision](../docs/decisions/M1.8-Evaluation-Design.md): source methodology,
  all fifteen questions, planned comparisons, limitations, and retrieval-evaluation prerequisites.

From the authorized project root:

```bash
pwd
.venv/bin/python -B scripts/validate_evaluations.py
.venv/bin/python -B -m unittest discover -s tests -v
```

The existing environment is Python 3.12.15. The validator locates the repository
from its own script path rather than the caller's current directory. Exit 0 means
structural checks passed; errors go to stderr and produce a nonzero exit. It never
repairs a manifest, rewrites labels, or treats a structural pass as semantic proof.
Tests use copied fixtures under `.local/tmp/` and remove their temporary copies.
`-B` prevents bytecode cache writes. No pytest installation is needed.

## 2. Categories and Retrieval Modes

| Category | Count | Purpose | Mode | Expected behavior |
| --- | --- | --- | --- | --- |
| A | 3 | Precise HR notification/removal/termination lookup | approved_only | answer |
| B | 6 | Multi-document authority, implementation, and incident reasoning | draft_review | answer |
| C | 3 | Missing hosting, permission scope, or authorization context | draft_review | clarify |
| D | 3 | Unavailable approved values or detailed procedures | draft_review | abstain |

`approved_only` restricts candidate evidence to manifest-listed policies marked
**Approved (Synthetic)**. In this snapshot that is HR-001 v1.0 only. Its references
to companion drafts do not make those drafts eligible or effective. For example,
an answer can state HR's exception referral without inventing an approved IT
exception workflow. This is a synthetic approval label, not real-world authority.

`draft_review` permits all six manifest-listed sources, while preserving their
individual statuses. IT-001, PLAT-001, SRE-001, SEC-001, and COMP-001 are v0.1,
**Draft — Proposed (Synthetic)**. Answers must explicitly label their proposals
and distinguish inherited approved HR obligations. Presence, project acceptance,
and a policy's use of “must” are not operational effectiveness.

Future retrieval must filter eligible sources before ranking and preserve that
filter through fusion, reranking, and context assembly. Returning a draft in
approved-only mode is an authority-leakage failure even if the answer ignores it.
Reviews, this README, question text, expected answers, and the design report must
not enter the retrieval index; otherwise answer-key leakage invalidates the run.

## 3. JSON Schema and Provenance

The top-level object contains exactly these fields:

| Field | Meaning |
| --- | --- |
| schema_version | Integer 1; describes the data format |
| dataset_id | Stable experiment identifier, `northstar-atlas-m1.8` |
| dataset_version | String `0.1`; revise intentionally when benchmark labels change |
| corpus_manifest | Repository-relative path `data/corpus_manifest.json` |
| corpus_manifest_sha256 | SHA-256 of the exact manifest file bytes |
| questions | Exactly fifteen question objects |

Each question has these required fields:

| Field | Type and contract |
| --- | --- |
| question_id | Unique string `EVAL-NNN`; this release uses EVAL-001 through EVAL-015 |
| category | `A`, `B`, `C`, or `D` |
| question | Nonempty user-facing question |
| retrieval_mode | `approved_only` or `draft_review`, following §2 |
| expected_behavior | `answer`, `clarify`, or `abstain`, following §2 |
| expected_answer_summary | Nonempty semantic answer guide, not an exact-match string |
| required_evidence | Nonempty array of unique document/section reference objects |
| required_qualifiers | Nonempty array of conditions that a correct answer must preserve |
| forbidden_inferences | Nonempty array of claims an answer must not assert |
| notes | Nonempty explanation of the case's intent or scoring boundary |

A reference has exactly two string fields, for example:

```json
{"document_id": "HR-001", "section_id": "4.2"}
```

Section identifiers are strings, not decimal numbers or ranges. Represent multiple
required sections as separate objects. A top-level section includes its descendant
subsections, but merely returning its heading does not establish their content.
Use the manifest for version, status, filename, and content hash.

`acceptable_alternative_evidence` is optional and defaults to an empty array.
Every current question includes an empty array: no substitutes have been adjudicated.
For a future reviewed one-for-one substitution, each entry contains `replaces`
and `evidence`, each a reference object. `replaces` must identify a required unit;
`evidence` must resolve to a section eligible for the question's mode. An approved
alternative must support the same required facts, authority, and conditions.
The validator checks structure and eligibility, not semantic equivalence.

JSON keys must be unique; nonfinite values such as NaN are rejected. Unknown fields,
missing fields, malformed objects, invalid enums, duplicate question IDs/evidence,
wrong category counts, nonexistent citations, and inappropriate draft evidence fail
validation. Category B must require multiple documents. Every run verifies all six
policy hashes before using labels and checks the dataset's manifest pin.

If a source hash mismatches, stop and investigate. If a deliberate corpus change
updates the manifest, review affected labels before revising the dataset version
and manifest pin. Never bypass an integrity failure by automatically accepting new
hashes. A byte hash identifies a snapshot; it does not prove approval or truth.

## 4. What Counts as Supporting Evidence

An evidence unit is a required `(document_id, section_id)` pair within the pinned
corpus. It is satisfied only when retrieved text supports the relevant expected
claims with the necessary actor, authority, conditions, and exceptions. The text
must have provenance matching the manifest's document version/hash and status.

Preserve parent heading context and draft labels on future chunks. A parent title,
document ID mention, citation to another policy, approximate text match, or high
similarity score alone earns no credit for that other policy's substantive rule.
A section split across chunks may need multiple retrieved pieces; a piece missing
a required qualifier does not satisfy the whole unit. Do not look up unretrieved
policy text to rescue a retrieval miss during scoring.

For unsupported questions, the relevant units establish the explicit gap. For
example, COMP-001 §11.1 says no numeric period is set. This is evidence for a
limited answer, not evidence of an absent approved duration. Review reports and
general model knowledge cannot fill that gap.

## 5. Evidence-unit Recall@5

This benchmark defines Recall@5 at the **required section-unit** level, using the
first five ranked chunks actually supplied by the evaluated retrieval pipeline.
It is not document-title hit rate or the fraction of retrieved chunks that appear
relevant. Keep chunking and result assembly fixed for comparable runs.

For question `q`:

- `R(q)` is the set of required document/section units in `required_evidence`.
- `T5(q)` is the first five returned chunks, or all returned chunks if fewer than
  five. Duplicate results consume slots but do not earn repeated unit credit.
- `H5(q)` is the subset of `R(q)` fully supported by eligible text in `T5(q)`,
  judged under §4. A reviewed alternative may satisfy its mapped required unit,
  but it does not add another unit or increase the denominator.
- `Recall@5(q) = |H5(q)| / |R(q)|`.

All cases have at least one required unit, including clarify/abstain cases, so
there is no empty denominator. Extra unrelated sections earn no credit. A chunk
may support several units only when the actual supporting text for each is present;
metadata listing many parent/child IDs is not enough. Two fragments in the top five
may jointly satisfy one unit if together they preserve its necessary conditions.
Invalid provenance or an ineligible draft cannot contribute evidence credit.

For EVAL-005, the required units are PLAT-001 §6.1, PLAT-001 §4.3, and COMP-001
§8.2. If the first five chunks fully support the first two but omit the evidence
requirements in COMP, Recall@5 is `2/3`. Full-evidence coverage is false. Retrieving
two copies of §6.1 instead of §4.3 does not increase credit. Retrieving only the
sentence “No separate Change Management approval” without the relevant renewed
technical-review and scope/risk/window conditions does not fully satisfy §6.1.

Report:

1. Per-question Recall@5 and a missing-unit list.
2. Macro mean across all fifteen questions and separately for each category.
3. Full-evidence coverage: the fraction of questions with Recall@5 equal to 1.
4. Authority-leakage counts, independently of retrieval coverage.

Partial coverage is useful diagnostically, but it is not complete support for a
multi-section answer. Oversized chunks could inflate section coverage while adding
irrelevant context; record chunk sizes, overlap, and context tokens and compare
equivalent settings. Parent expansion or extra fetched text must be included in
the evaluated context budget, not silently added after the top-five cutoff.

Initial scoring requires human adjudication of support; automatic ID overlap can
be a structural diagnostic only. The original dataset release implemented neither
retrieval nor this scorer. The later [retrieval baseline](../docs/decisions/M2.5-Retrieval-Evaluation.md)
reports a distinct, automated complete-source-section coverage proxy.

## 6. Answer Faithfulness and Citation Assessment

Later evaluation should separate retrieval success from generation quality.
Freeze and record the actual context supplied to the model. Break an answer into
material factual/authority claims and classify each as supported, contradicted,
or unsupported by that eligible context. A true statement found elsewhere in the
corpus but absent from supplied context is not grounded in the experiment's context.

Use the expected summary as a meaning-based rubric. Assess:

- **Faithfulness:** supported material claims divided by all material claims;
  record contradictions separately. An empty response gets no automatic perfect
  score; mark claim faithfulness not applicable and fail required answer behavior.
- **Qualifier coverage:** which applicable `required_qualifiers` are correctly
  expressed; omitting a condition may make an otherwise familiar rule misleading.
- **Forbidden inferences:** any asserted listed claim is a question-level failure,
  even when many other claims are correct. Quoting and rejecting an inference is
  not asserting it.
- **Citation correctness:** fraction of claim–citation associations where the
  cited source/version/section exists in the pinned corpus, was supplied as eligible
  context, and supports that claim with correct status and conditions.
- **Citation completeness:** fraction of material evidence-requiring claims with
  at least one correct supporting citation. If such claims have no citations,
  completeness is zero; lack of citations is not perfect correctness.

Report exact counts and annotate failures. Use citations such as
`PLAT-001 v0.1 §6.1 (proposed)`; cite HR directly for approved lifecycle deadlines.
Mentioning a valid document somewhere in the answer does not support every claim.
No claim is made that a keyword overlap, fluent answer, or exact summary wording
proves faithfulness. Potential future automated judges need calibration against
human review; none is called or installed here.

## 7. Clarification, Abstention, and Overall Behavior

For `answer`, provide the supported substantive answer with the required conditions
and authority labels. An unnecessary blanket refusal fails behavior even if safe.

For `clarify`, explicitly identify the missing decision-relevant context and ask
for it, while explaining the supported conditional facts. Do not choose an owner,
grant, or permission on an unstated assumption. EVAL-010 needs hosting/component
and impact context; EVAL-011 needs requested access scope; EVAL-012 needs window
and recorded authority context. “Please clarify” without useful specifics is not
sufficient. This dataset does not evaluate multi-turn follow-up dialogs.

For `abstain`, explicitly decline to supply the requested unavailable approved
value/procedure, explain what the cited policies do support, and identify the
unresolved decision. An answer can be useful without supplying the missing value.
EVAL-014 must acknowledge both the MFA-method and rotation-interval gaps. Do not
replace unknown company policy with a common industry default. A bare refusal
without supported explanation is incomplete; a guessed value is a failure.

Record actual versus expected behavior and behavior accuracy overall and by
category. For a strict question-level answer pass, require correct behavior,
all essential expected facts and qualifiers, no forbidden inference, faithful
material claims, and correct supporting citations. This strict result is distinct
from partial retrieval recall. Failure to retrieve limitation evidence is not proof
that the corpus has no answer; assess the returned context and report the miss.

This dataset specifies no model pass rate or release threshold. Future thresholds
and a larger test set need review before production claims.

## 8. Reproducibility and Limits

Future run records should contain dataset and manifest hashes, per-policy hashes,
chunking configuration and chunk IDs, embedding/model versions, backend/index
settings, metadata filters, dense/lexical/fusion settings, candidate and final-k
budgets, reranker, generation prompt/parameters, timings, and per-question results.
Keep generated run artifacts inside `.local/`. Avoid putting expected answers
or this rubric into model context during the evaluated task.

Fifteen synthetic, overlapping questions are too few to establish broad statistical
reliability. They are a visible development set, not a hidden independent test set.
One case changes overall accuracy by about 6.7 percentage points and a three-case
category by about 33.3 points. Report counts, examine individual failures, and avoid
vendor-quality claims from tiny average differences. No real user distribution,
multilingual performance, large-corpus scaling, or production security evaluation
is represented. Human relevance judgments also need review and versioning.

The dataset validator checks structure and source integrity locally; it does not
run embeddings, retrieval, model evaluations or cloud services.
