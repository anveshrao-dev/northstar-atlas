# Retrieval and Bounded-Graph Evaluation

**Date:** 2026-10-24.

**Status:** Local evaluation complete; engineering handoff findings ready for review.

**Scope:** Retrieval strategies and the original optional `bounded_v1` LangGraph
profile as of 2026-10-24. No application changes or hosted inference. This report
does not evaluate the later LLM routing, grading or semantic-checking profiles.

## 1. Findings and handoff assessment

Hybrid has the best expanded evidence recall, **70.44%**, versus **64.67%** dense and **66.78%** BM25. All three fully cover only **6/15** frozen questions. Every strategy's top-five order and evidence metrics match the earlier [hybrid evaluation](M5.2-Hybrid-Retrieval-Results.md). No retrieval regression was observed on this snapshot.

The actual graph enforces the tested call bounds, rejects malformed or invalidly cited responses, and preserves source eligibility. Reviewed Terraform requirements and the evidence-backed retention limitation pass with real local retrieval and deterministic fake generation. These are offline contract results, not new Nebius results.

Two important limitations remain:

- Nonempty evidence is sufficient for ordinary-query routing, even when cross-policy evidence is missing. Such queries retain `coverage.status = not_assessed`; the graph does not automatically establish relevance, completeness, ambiguity, or unsupportedness.
- An adversarial fake claim inventing an approved retention duration, paired with a genuine quotation from the proposed Compliance policy, **passed ordinary-query structural validation**. The evaluator rejects its semantic assessment. Citation membership and quotation provenance do not enforce entailment or prevent contradictory free-form wording. This is a material limitation for unrestricted policy guidance, not a successful answer.

This engineering evaluation supports a controlled, clearly labeled MVP handoff.
It does **not** establish readiness for unrestricted authoritative enterprise
advice or production release. The findings support retaining the linear
QAService/Streamlit defaults while evaluating hybrid and graph behavior explicitly.
No deployment was performed as part of this evaluation. Later implementation and
live observations are documented separately in the [engineering guide](../README.md).

## 2. Frozen inputs and provenance

Reviewed inputs include the [dataset and methodology](../../evaluations/README.md), [dense baseline](M2.5-Retrieval-Evaluation.md), [managed/local comparison](M3.4-Chroma-Pinecone-Comparison.md), [hybrid results](M5.2-Hybrid-Retrieval-Results.md), [graph implementation](M6.2-LangGraph-Implementation.md), and existing QA/context/grounding contracts.

| Item | Evaluated value |
| --- | --- |
| Dataset | `northstar-atlas-m1.8`, version 0.1; 15 questions; categories A/B/C/D = 3/6/3/3 |
| Required evidence | 40 question/reference occurrences, 31 distinct document/section pairs |
| Source snapshot | Six manifest-verified policies; 151 canonical chunks; 189 refined records |
| Eligibility reference date | **2026-10-22**, deliberately retained for baseline comparability |
| Approved-only evidence | HR-001 v1.0, Approved (Synthetic), effective 2026-10-01 |
| Draft review | HR plus five v0.1 Draft — Proposed (Synthetic) policies |
| Embedding model | `sentence-transformers/all-MiniLM-L6-v2`, revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41` |
| Vectors | 384 dimensions, normalized; cosine distance for Chroma |
| Local runtime | Python 3.12.15, Intel macOS; Chroma 1.5.9, LangGraph 1.2.16, SDK 0.4.9 |
| Ranking configuration | Dense top 5; BM25 top 5; hybrid dense 20 + BM25 20, RRF constant 60, final top 5 |

Exact identities:

```text
questions.json SHA-256:
e33b1dbccd7e8df82651f8a58f7044b3e2c3032cce1a82686bc792f2c7a241b4
corpus_manifest.json SHA-256:
1c6efd57db22e900da7118eca62f8f075d96e5b846c1a30e3889628cd23860b1
corpus fingerprint:
20f5ec1d95989e7c6f6b12579d089dea5be0fe815fd5ba357902d195d49dfac9
immutable local snapshot ID:
e55dcdeca17453b27d45f4d3663ef72b9ba3e635f43689899a8d66b52cf27860
```

The ready 189-record Chroma receipt, corpus, configuration and frozen local snapshot were checked before and after evaluation. The existing comparison adapter checked actual selected text, metadata and vector direction against that snapshot; the retrieval experiment touched 149 distinct selected records through normal searches/context expansion. It did not rebuild the index or run a separate full-record audit. Database internal housekeeping bytes were not used as a reproducibility assertion; receipt identity, count and verified payloads were.

The local snapshot lives under `.local/pinecone/snapshots/`, but reading it invokes no Pinecone service. Pinecone was **not queried**. Its earlier measurements remain historical evidence only. Credentials were not loaded. Existing model files were used offline; no document embeddings were regenerated.

## 3. Scoring definitions and experiment boundaries

The unchanged evaluator uses `complete-source-section-v1`, the same deterministic proxy as M2.5/M3.4/M5.2:

- A unit is a document-qualified section. Coverage requires its canonical source spans, not merely its heading or a cross-reference to it. A parent reference requires the parent's own content and all dot-delimited descendant subsections. A subsection alone does not satisfy the parent.
- **Recall@5:** covered required units divided by all required units, macro-averaged across questions.
- **Hit@5:** whether at least one required unit is covered.
- **Full Evidence Coverage@5:** whether every required unit is covered.
- **Reciprocal Rank@5 / MRR@5:** reciprocal of the earliest direct-rank prefix covering a required unit, then averaged. Expanded scoring uses the sibling closure of that prefix; siblings never receive independent direct ranks or votes.
- Duplicate section references earn no repeated credit. Direct scoring uses only the five ranked chunks; expanded scoring adds their required, eligible sibling context.

Exact source-span coverage is a conservative proxy, not a semantic judgment. It can penalize a missing paragraph that is irrelevant to a particular question. Conversely, copying a whole section does not prove an answer interprets it correctly. Cosine similarity, BM25 scores, and RRF scores are neither answer confidence nor interchangeable scales.

Categories C and D are scored for evidence supporting clarification or corpus limitations, **not** for ordinary answerability. Overall scores include their evidence needs; answerable A+B scores are reported separately. The graph experiments do not feed expected answers, required evidence, or category labels into ordinary runtime routing. Explicit reviewed fixtures are identified separately.

## 4. Retrieval results

All percentages below are macro-averages. MRR is on a 0–1 scale.

| Strategy | Evidence | Recall@5 | Hit@5 | Full coverage@5 | MRR@5 |
| --- | --- | ---: | ---: | ---: | ---: |
| Dense | Direct | 55.22% | 86.67% | 26.67% | 0.6944 |
| Dense | Expanded | 64.67% | 93.33% | 40.00% | 0.8667 |
| BM25 | Direct | 51.56% | 80.00% | 26.67% | 0.6133 |
| BM25 | Expanded | 66.78% | 93.33% | 40.00% | 0.8333 |
| Hybrid | Direct | 57.44% | 86.67% | 26.67% | 0.6556 |
| Hybrid | Expanded | 70.44% | 100.00% | 40.00% | 0.8833 |

Sibling expansion improves recall by 9.44, 15.22 and 13.00 percentage points respectively. Hybrid's 100% expanded Hit@5 means each question has at least one covered unit; it does **not** mean all questions can be answered.

For the nine answerable A+B questions, direct/expanded recall is dense **67.96% / 80.00%**, BM25 **63.70% / 81.67%**, hybrid **67.96% / 82.22%**. All three have direct full coverage 3/9, expanded full coverage 5/9, and Hit@5 100%. Answerable expanded MRR is 1.0000, 0.8889, 0.9444 respectively.

### 4.1 Category breakdown

Each cell is **direct / expanded**. Categories: A precise HR (3), B cross-policy (6), C ambiguous (3), D unsupported requested values/procedures (3).

| Strategy | Category | Recall % | Hit % | Full coverage % | MRR |
| --- | --- | ---: | ---: | ---: | ---: |
| Dense | A | 100 / 100 | 100 / 100 | 100 / 100 | 1.0000 / 1.0000 |
| BM25 | A | 100 / 100 | 100 / 100 | 100 / 100 | 0.8333 / 0.8333 |
| Hybrid | A | 100 / 100 | 100 / 100 | 100 / 100 | 0.8333 / 0.8333 |
| Dense | B | 51.94 / 70.00 | 100 / 100 | 0 / 33.33 | 0.7639 / 1.0000 |
| BM25 | B | 45.56 / 72.50 | 100 / 100 | 0 / 33.33 | 0.7000 / 0.9167 |
| Hybrid | B | 51.94 / 73.33 | 100 / 100 | 0 / 33.33 | 0.7639 / 1.0000 |
| Dense | C | 22.22 / 33.33 | 66.67 / 100 | 0 / 0 | 0.4444 / 0.8333 |
| BM25 | C | 33.33 / 44.44 | 66.67 / 100 | 0 / 0 | 0.5000 / 1.0000 |
| Hybrid | C | 33.33 / 44.44 | 66.67 / 100 | 0 / 0 | 0.5000 / 1.0000 |
| Dense | D | 50.00 / 50.00 | 66.67 / 66.67 | 33.33 / 33.33 | 0.5000 / 0.5000 |
| BM25 | D | 33.33 / 44.44 | 33.33 / 66.67 | 33.33 / 33.33 | 0.3333 / 0.5000 |
| Hybrid | D | 50.00 / 61.11 | 66.67 / 100 | 33.33 / 33.33 | 0.4167 / 0.5833 |

### 4.2 Missing expanded evidence by question

These are exact required units still missing under the unchanged source-coverage proxy. A dash means none. Direct misses, partial spans, ranks, distances/scores and siblings are also preserved in the machine-readable result.

| Question | Dense | BM25 | Hybrid |
| --- | --- | --- | --- |
| 001–003: HR lifecycle | — | — | — |
| 004: transfer to SRE | HR-001 §§4.3, 4.4; SEC-001 §§5.2, 4.1 | HR-001 §4.3; SEC-001 §4.1 | HR-001 §§4.3, 4.4; SEC-001 §4.1 |
| 005: Terraform regeneration | — | — | — |
| 006: Jenkins ownership | —; prompt blocked | —; prompt blocked | —; prompt blocked |
| 007: AKS identity / Key Vault | IT-001 §5.2 | IT-001 §5.2; SEC-001 §§6.1, 7.1 | IT-001 §5.2; SEC-001 §7.1 |
| 008: failed deployment / incident | SRE-001 §§1.1, 3.2 | SRE-001 §1.1 | SRE-001 §1.1 |
| 009: certificate disruption | SRE-001 §5.2 | SRE-001 §5.2 | SRE-001 §5.2 |
| 010: unspecified Jenkins failure | SEC-001 §7.2; SRE-001 §1.1 | Same | Same; prompt blocked |
| 011: vague production access | IT-001 §5.1; SEC-001 §3.1 | Same | Same |
| 012: extended deployment | PLAT-001 §4.3; SRE-001 §7.2 | PLAT-001 §4.3 | PLAT-001 §4.3 |
| 013: approved retention duration | — | — | — |
| 014: exact MFA / rotation value | SEC-001 §8.2 | SEC-001 §§4.1, 8.2 | SEC-001 §8.2 |
| 015: approved GCP procedure | PLAT-001 §§1.1, 2; SEC-001 §6.1 | PLAT-001 §1.1; SEC-001 §6.1 | Same as BM25 |

Representative successes are the HR lifecycle questions, expanded Terraform conditions/evidence, and the retention policy gap. Representative weaknesses are cross-policy transfer/removal gates, missing workload/secret authorization evidence, and conditional scope/escalation evidence. No ordinary answer is certified by these retrieval successes.

### 4.3 Historical comparison

All six aggregate metric rows and all 45 question/strategy top-five orders match M5.2. Dense metrics also exactly match M2.5 and the Chroma side of M3.4. No tuning or dataset changes were made.

M3.4's historical Pinecone evidence coverage equals Chroma's, with direct MRR 0.6911 and expanded MRR 0.8333 versus Chroma 0.6944/0.8667. That earlier backend comparison does not establish current Pinecone readiness, latency or usage. No managed-backend check was needed or performed here.

## 5. Latency, context size and budget failures

Warm retrieval measurements use three passes, rotating strategy order, 45 samples per strategy. They include readiness checks, selected payload verification and sibling fetching; exclude one-time query encoding, evidence scoring, context preparation and file IO. Hybrid includes dense20, BM25-20 and fusion. Frozen ranks remained identical across passes. The existing adapter returns ranked hits and siblings together, so there is one measured retrieval latency per strategy, not a separately instrumented direct-only latency.

| Strategy | Mean ms | Median ms | Maximum ms |
| --- | ---: | ---: | ---: |
| Dense | 18.94 | 18.12 | 24.61 |
| BM25 | 7.50 | 7.91 | 10.98 |
| Hybrid | 40.90 | 41.39 | 78.30 |

The first correctness pass averaged 20.22/7.94/40.46 ms respectively. These are local observations, not service-level guarantees. BM25 construction took 8.46 ms. Cached MiniLM loading took 26.00 s; sixteen distinct query encodings were reused across all retrieval arms and timing passes. Query encoding mean/median/max was 358.81/14.26/5,524.88 ms, dominated by the first observed cold query. Total initial setup took 37.72 s. Graph evaluation was a separate run with cached-model query encoding; no document encoding occurred. No memory benchmark or Linux performance measurement was performed.

Context sizes below are before full-prompt acceptance. Expanded character counts include chunk context duplication; the artifact also records deduplicated source characters and exact spans.

| Strategy | Direct chars min / median / max | Expanded chars min / median / max | Expanded chunks min / median / max | Evidence bound mean / median / max |
| --- | --- | --- | --- | --- |
| Dense | 1,231 / 4,550 / 6,052 | 1,231 / 4,986 / 17,338 | 5 / 6 / 15 | 13,369 / 11,170 / 26,245 |
| BM25 | 1,185 / 4,973 / 6,076 | 1,185 / 5,741 / 18,532 | 5 / 6 / 16 | 13,922 / 12,122 / 28,701 |
| Hybrid | 1,185 / 4,549 / 6,076 | 1,185 / 6,183 / 17,731 | 5 / 7 / 15 | 14,160 / 13,104 / 26,592 |

The existing Qwen UTF-8/NFC byte bound is deliberately conservative; these numbers are **not actual tokenizer usage or billing tokens**. Default configured context is 32,768, output reservation 1,024, margin 512. Evidence-only budget is 31,232. No evidence group was excluded and no context warning occurred during the retrieval comparison. Nevertheless, instructions, question and structured-response schema overhead make four complete prompts too large:

| Question / strategy | Input bound including margin | Plus output reservation | Over 32,768 by |
| --- | ---: | ---: | ---: |
| EVAL-006 dense | 33,686 | 34,710 | 1,942 |
| EVAL-006 BM25 | 36,142 | 37,166 | 4,398 |
| EVAL-006 hybrid | 34,033 | 35,057 | 2,289 |
| EVAL-010 hybrid | 33,485 | 34,509 | 1,741 |

For these rows the diagnostic bound was calculated offline with a larger temporary validation ceiling solely to expose the number; the real evaluation retained the default ceiling and rejected generation. No LLM was called and no critical group was silently trimmed. Successful evidence assembly and an empty warnings list therefore do not imply a sendable prompt.

Full-prompt-ready counts are **14/15 dense, 14/15 BM25, 13/15 hybrid**. Counting blocked requests as delivering no evidence reduces expanded recall to **58.00%, 60.11%, 61.56%** respectively. This separate application-delivery diagnostic does not replace or redefine frozen retrieval Recall@5. Hybrid's evidence gain is partly offset by context cost.

## 6. Actual LangGraph evaluation

### 6.1 Real retrieval with fake generation on all frozen questions

Ran `GraphQAService` against actual local dense, BM25 and hybrid adapters for **45 question/strategy runs**. Each initial strategy had one alternate configured (dense→BM25, BM25→dense, hybrid→BM25), using unchanged question text, mode, date and top-k. The catalog revalidates eligible source identity and siblings. Generation used a deterministic fake that copies one contiguous retrieved paragraph with its real citation/quotation; it receives no expected answer, evidence rubric or category label.

This deliberately simple fake checks integration and structural publication, **not whether the returned paragraph answers the question**. It does not simulate intelligent ambiguity detection or abstention selection.

| Initial strategy | Structurally accepted fake `answer` | Context errors before generation | Retrieval attempts per question | Fake provider calls total |
| --- | ---: | ---: | --- | ---: |
| Dense | 14 | 1: EVAL-006 | 1 | 14 |
| BM25 | 14 | 1: EVAL-006 | 1 | 14 |
| Hybrid | 13 | 2: EVAL-006, EVAL-010 | 1 | 13 |

Every accepted ordinary result reports `not_assessed`. Accepted citations in approved-only runs are HR-001 only; proposed labels remain on draft citations. There are no source/citation integrity violations. Success follows `router → retriever → assessor → generator → validator`; prompt overflow follows `router → retriever → assessor → validator`, with zero provider calls.

No ordinary run took an alternate because retrieval was nonempty; partial required-evidence coverage is not automatically recognized as insufficiency. The C/D paragraph fixtures can be structurally accepted as `answer` even though the frozen behavioral expectation is `clarify`/`abstain`. This is **not** a measured real-model behavior failure or behavioral success rate. It demonstrates that the graph itself does not infer those decisions from the benchmark or enforce semantic relevance.

### 6.2 Controlled routes, reviewed requirements and fault injection

Twenty-four labeled scenarios used the real graph and existing validators. Most inject catalog-derived source fixtures or controlled errors; two reviewed cases use actual local retrieval. All assertions about expected behavior/call counts passed. The final adversarial case intentionally exposes a semantic weakness and must not be counted as a safe policy answer.

| Scenario | Observed result | Searches / fake inference calls |
| --- | --- | --- |
| Approved HR / proposed Compliance fixture | Validated source-extractive `answer`, correct labels | 1 / 1 each |
| Empty dense, relevant BM25 | Alternate succeeds; prior hits replaced | 2 / 1 |
| Two empty attempts | `abstain`, insufficient supplied evidence, no fabricated citation | 2 / 0 |
| Single-strategy empty plan | `abstain` after configured single attempt | 1 / 0 |
| Trusted question-scoped clarification | `clarify` without retrieval/provider | 0 / 0 |
| Fake generated clarification | Validated `clarify` | 1 / 1 |
| Fake no-evidence abstention | Validated `abstain`; no unsupported policy claims | 1 / 1 |
| Required reviewed source missing first, present second | Alternate succeeds; no preflight evidence injection | 2 / 1 |
| Required reviewed source absent twice | `context_failed`; cannot bypass obligations with generic abstention | 2 / 0 |
| Four Terraform obligations with real retrieval | `answer`, all application-owned source spans rendered, proposed citation | 1 / 1 |
| Retention limitation with real retrieval | Cited evidence-backed `abstain`, documented-gap basis | 1 / 1 |
| Missing required stop/amend obligation | `response_rejected`, no candidate published | 1 / 1 |
| Fabricated citation / quotation | `response_rejected` | 1 / 1 each |
| Malformed JSON / `finish_reason=length` | `response_rejected` | 1 / 1 each |
| Provider timeout | `provider_failed`, no retry or policy abstention | 1 / 1 |
| Draft leak in approved-only mode | `context_failed` before generation | 1 / 0 |
| Missing sibling / stale source hash | `context_failed` before generation | 1 / 0 each |
| Small context budget | `context_failed`, no silent evidence loss | 1 / 0 |
| Fabricated reviewed source statement | `configuration_failed` before retrieval | 0 / 0 |
| Unsupported claim with genuine quote, no reviewed obligations | Structurally accepted; **semantic assessment fails** | 1 / 1 |

Alternate-success trace: `router → retriever → assessor → retriever → assessor → generator → validator`. Exhaustion omits generator. Trusted clarification and invalid reviewed input take `router → validator`. Invalid retrieved payloads take `router → retriever → validator`. Invalid generated candidates reach validator as failures and expose only safe public errors; no rejected sentinel text appeared in results.

A three-strategy configuration is rejected before execution. No tested invocation exceeded two retrieval attempts or one provider call. Deterministic fixtures also confirm the linear QAService and graph return identical public contracts for identical approved evidence. The complete existing suite supplies broader router, backend, retrieval-failure, budget, grounding, UI and compatibility regressions.

### 6.3 Reviewed versus ordinary completeness

The Terraform diagnostic uses the existing four caller-reviewed obligations and exact source spans, with the application-owned extractive profile. All four IDs are covered, quoted source matching permits only whitespace normalization, and the rendered citation remains `PLAT-001 v0.1 §6.1 (proposed)`. Input bound is 22,224 before the 1,024 output reservation. The source statements preserve renewed technical approval, no automatic additional Change Management approval solely for regeneration, authorized scope/risk/window/conditions, and stop/amend/escalate boundaries. Source reproduction is established; general semantic completeness is not.

The retention diagnostic uses the existing evidence-linked limitation and restricted missing-information label. It returns `abstain`, cites `COMP-001 v0.1 §11.1 (proposed)`, and quotes the exact statement that no numeric period is set. Input bound is 19,980. It establishes a documented gap in supplied proposed evidence, not the nonexistence of a rule elsewhere. Coverage is `required_entries_present`, with `no_period_established` evidence-linked; its authored semantic interpretation is assessed separately from structural validation.

In the adversarial probe, an invented approved numeric duration was accompanied by a genuine COMP-001 quotation. The ordinary validator accepted membership/provenance, while coverage stayed `not_assessed`; the free-form claim contradicted the source and its proposed authority. No real model produced this fixture and it is not published here as policy guidance. The result confirms that existing semantic limitations are observable at the public-result boundary, not eliminated by adding a graph. Changing validators or the successful reviewed paths was outside this evaluation's scope.

## 7. Supplementary AKS temporary-access diagnostic

Exact question, kept outside the frozen dataset:

> An engineer needs temporary production access to troubleshoot an AKS issue. Who should approve it, and which team grants the access?

Mode: `draft_review`. The existing supplementary rubric has five references: IT-001 §§10.1, 9.1; SEC-001 §§9.1, 5.2, 6.1. It is a diagnostic rubric, not a complete executable authorization decision.

| Strategy | Direct / expanded coverage | Direct top-five section sequence |
| --- | --- | --- |
| Dense | 2/5 / 2/5 | IT-001 §10.1; SEC-001 §9.1; SEC-001 §3.1; PLAT-001 §14; another SEC-001 §3.1 chunk |
| BM25 | 1/5 / 1/5 | SEC-001 §3.2; SEC-001 §9.2; SEC-001 §9.1; COMP-001 §10.1; IT-001 §6.2 |
| Hybrid | 2/5 / 2/5 | SEC-001 §9.1; SRE-001 §2; IT-001 §10.1; COMP-001 §6.1; SEC-001 §3.2 |

All three miss the complete technical-provisioning/decision references in final context. Dense sees a SEC-001 §5.2 candidate at rank 9; BM25 sees SEC-001 §6.1 at rank 16; IT-001 §9.1 is absent from both top-20 candidate lists. Fusion does not promote those candidates into final top five. This is observed candidate/rank evidence; it does not isolate a model-training or tokenizer root cause.

Source inspection, **not retrieval credit**, establishes the proposed boundaries:

- [IT-001 §10.1](../../data/policies/IT-001.md) routes production AKS requests for Security assessment of actual scope; production access is not automatically administrative or routine development access.
- [SEC-001 §§5.2, 9.1](../../data/policies/SEC-001.md) separate business approval, Security authorization and technical fulfillment; temporary access needs applicable approvals, approved start/end, monitoring, and revocation ownership. Missing decisions or delegation block provisioning.
- IT-001 §9.1 assigns workforce identity/authorized standard group work to IT, and platform-owned mappings/grants/removals to Platform Engineering. SEC-001 §6.1 assigns cloud implementation within delegation to Cloud Platform and distinguishes Azure permissions from Kubernetes RBAC.
- SRE-001 §2 in hybrid context provides a broad ownership table, but does not replace the detailed IAM/cloud grant and delegation conditions. A temporary troubleshooting request is not automatically an emergency; BM25's emergency hit does not establish that classification.

The evidence partially covers Security authorization, but does not adequately supply both IAM identity responsibilities and the conditional Cloud Platform executor. The owning resource, requested permissions, business approver/delegation, applicable change controls and expiry decisions still need case-specific confirmation. All relevant policies here remain proposed. Do not label this question complete or infer a blanket grant by Security, IT or SRE. Sibling expansion protects selected section families; it does not automatically traverse every cross-policy referral.

## 8. Failure analysis and future work

| Failure class | Evidence | Implication |
| --- | --- | --- |
| Retrieval omission | EVAL-004, 007–012, 014–015 and supplementary AKS lack required cross-policy units | Better candidate coverage/selection is needed; a citation cannot repair a missing source |
| Fusion tradeoff | Hybrid improves aggregate recall but loses some dense-only EVAL-007 evidence and adds EVAL-010 prompt overflow | No strategy dominates every question; preserve per-question results |
| Context failure | EVAL-006 all arms, EVAL-010 hybrid | Complete evidence may be retrieved but unavailable to generation under current conservative budget |
| Ordinary sufficiency limitation | Nonempty partial evidence routes directly to generation | Bounded retries do not recover omissions unless insufficiency is actually detected or reviewed requirements are supplied |
| Generation contract failure | Missing obligation, invented citation/quote, malformed/truncated output | Safely rejected with no retry and no published candidate |
| Semantic failure | Adversarial ordinary claim with genuine support quote passes structure | Membership/provenance is not entailment; unsupported authority and values need separate semantic assessment |
| Operational failure | Injected provider timeout remains `provider_failed` | Do not disguise infrastructure failure as policy abstention |

Recommended future experiments, **not implemented**:

1. Expand independently reviewed semantic/obligation evaluation, including negative claims, draft-authority contradictions and conditional approval gates. Keep ordinary completeness explicitly unassessed until there is evidence for a stronger claim.
2. Investigate bounded multi-part retrieval or explicit cross-policy evidence requirements for ownership questions; keep source authority and delegation gates intact. Do not inject the frozen answer key into runtime retrieval.
3. Test candidate selection/reranking and context allocation against a separate diagnostic set. Compare per-question evidence gains, latency and full-prompt delivery, rather than aggregate recall alone.
4. Measure a verified generation tokenizer or carefully reviewed prompt/schema overhead reduction to recover budget-blocked cases. Do not drop sibling conditions or increase limits silently.
5. Use the frozen baseline for regression, add held-out questions without rewriting it, and conduct a separately approved live answer evaluation. Fake generation cannot establish model behavior, faithfulness, refusal reliability, pricing or hosted latency.

The small development dataset has been inspected repeatedly and is not a held-out estimate of enterprise performance. There are no statistical confidence claims. No reranker, alternative model, graph LLM planner, new cloud component or deployment change was introduced.

## 9. Historical validation and reproducible commands

The test counts and commands below belong to the 2026-10-24 evaluation checkpoint.
They do not validate the later semantic graph or establish current live-provider
behavior. The recorded retrieval measurements remain unchanged.

**Complete existing suite: 527 tests passed, 0 failures, 0 errors, 0 skips, 115.75 s.** Cached-model integration tests were enabled. The M1.8 validator passed all fifteen IDs, 3/6/3/3 categories, source hashes, manifest pin, evidence references, modes and expected behaviors. Validator success remains structural, not semantic.

Additional evaluation assertions: 45 actual graph/retrieval runs; 24 controlled scenarios with explicit expected outcomes; one rejected third-attempt configuration; identical linear/graph public-contract check. No claim that the semantic adversarial fixture is safe is included in these structural pass counts.

The full suite uses temporary test stores and in-memory provider fakes, including tests named for Pinecone operations. No existing index was reindexed and no hosted operation ran. Outbound Python socket connections were blocked during retrieval, graph evaluation and the suite; model/telemetry/tracing settings were offline. Streamlit bare-mode test warnings are expected; no skipped model tests or provider failures occurred in the validation run.

New artifacts are ignored under `.local/evaluations/m7.1/`:

- `run_retrieval.py`, `retrieval.json`, `retrieval.log`: unchanged evaluator, metrics, full per-question ranks/scores, missing/partial evidence, sibling/source spans, prompt diagnostics, timing passes and provenance.
- `run_graph.py`, `graph.json`, `graph.log`: actual graph executions, source-derived/fault fixtures, public results, traces, call counts and semantic-probe assessment. Raw prompts/credentials are not persisted.
- `run_validation.py`, `tests.log`, `validation.json`: full-suite and validator evidence.
- `before.json`, `analysis.json`, `integrity.json`: protected hashes, baseline comparison and explicit prompt-overflow diagnostics.

Commands used from the authorized checkout, with the existing environment:

```bash
pwd
export TMPDIR="$PWD/.local/tmp"
export PYTHONPYCACHEPREFIX="$PWD/.local/pycache"
export HF_HOME="$PWD/.local/huggingface"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export ANONYMIZED_TELEMETRY=False LANGSMITH_TRACING=false LANGCHAIN_TRACING_V2=false
export UV_CACHE_DIR="$PWD/.local/uv-cache"
export UV_PYTHON_INSTALL_DIR="$PWD/.local/uv-python"
export UV_PYTHON_BIN_DIR="$PWD/.local/bin"
export UV_PROJECT_ENVIRONMENT="$PWD/.venv"
export UV_PYTHON_DOWNLOADS=never
export NORTHSTAR_RUN_MODEL_TESTS=1

.venv/bin/python -B .local/evaluations/m7.1/run_retrieval.py
.venv/bin/python -B .local/evaluations/m7.1/run_graph.py
.venv/bin/python -B .local/evaluations/m7.1/run_validation.py
.venv/bin/python -B scripts/validate_evaluations.py
```

The retained local harnesses use exclusive output creation and refuse to overwrite these results. For a repeat, preserve them, copy the harnesses into a new ignored run directory and change only each `OUT` destination to a fresh contained filename. The retrieval harness calls the existing `evaluation.hybrid.evaluate` with the snapshot above, reference date 2026-10-22 and three warm passes. The graph harness reuses existing reviewed regression requirements explicitly; it never adds `src/` to `PYTHONPATH`. These local harnesses/results are evaluation artifacts, not new public runtime modules. A fresh checkout needs its separately provisioned existing model/index/snapshot; missing prerequisites must stop rather than download or rebuild automatically.

Protected-file verification found **zero changes across 98 inventoried source, test, policy, dataset, script, dependency, Git-index, baseline, receipt and snapshot files**. HEAD and Git index are unchanged. Existing pre-task M5/M6 working changes remain intact. No policy, frozen question, dependency, retrieval/grounding implementation, Replit configuration or historical result was modified. The only new public file is this report; all generated measurements and harnesses stay ignored.

Stop after this evaluation. Further implementation, hosted inference, commits, publication or deployment require separate authorization.
