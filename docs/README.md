# Architecture, experiments and documentation

This is the public engineering overview for the local Northstar Atlas MVP.
The [project README](../README.md) contains current operation and setup guidance.
Decision records below describe the state at the time of each experiment or
design. Dated proposals and prices are historical, not current operating guidance.
Historical milestone and experiment identifiers remain unchanged for artifact
provenance and reproducibility. Descriptive report filenames identify their scope;
dated contents retain the original methodology and findings.

## Current component boundaries

| Component | Responsibility and boundary |
| --- | --- |
| Ingestion and manifest | Load only six allowlisted Markdown policies; validate metadata and SHA-256 hashes. Reviews and evaluation answers are not source documents. |
| Canonical chunker | Preserve section IDs, headings, rules, tables and source order. The current corpus produces 151 canonical chunks. |
| MiniLM refinement | Use the pinned tokenizer to refine 25 oversized chunks into a 189-chunk embedding view. Track source spans and duplicated context; retain the canonical representation. |
| Embeddings | CPU MiniLM, 384 dimensions, L2 normalization and a 256-token input ceiling. Reject oversized/non-finite/incompatible inputs rather than truncate. |
| Vector stores | Chroma persists locally; Pinecone uses managed immutable namespace publication. Both retain source/model identity, status, section citations and required siblings. Incomplete indexes cannot masquerade as ready. |
| Lexical/hybrid retrieval | Standard-library BM25 over the same verified chunks; optional dense20 + BM2520 RRF60, final five; source/status/sibling checks remain mandatory. Dense remains the default. |
| Optional graph | `bounded_v1` permits one inference; Phase A `llm_routing_grading_v1` adds routing/grading with at most four calls; `llm_routing_grading_semantic_v1` adds semantic checking and one possible regeneration, at most seven calls. Each graph permits at most two retrieval attempts. Linear remains the default. |
| Context builder | Validate eligibility and source relationships, include required siblings, deduplicate, group evidence and budget labels/content together. Missing critical context blocks acceptance. |
| Generation/grounding | Provider-independent structured requests; Nebius adapter with bounded transport behavior. Validate response shape, supplied citations, quotation spans and caller-reviewed requirements; do not confuse structural validation with semantics. |
| QAService | Reuse initialized retrieval/model/provider resources; return immutable public success/error models separately from developer diagnostics. Support answer, clarify and abstain without treating infrastructure errors as policy claims. |
| Streamlit | Display validated statements, source labels and optional evidence/validation details. Default to authored offline demonstrations; require server and session opt-in for paid questions. |
| Controlled-live accounting | Shared version-2 durable ledger reserves one linear call or the complete seven-call semantic workflow before provider opening. Single-use stage permits, current pricing, private-access checks and explicit operator authorization are required. |

The public result's structured presentation is additive. Legacy rendered `text`
remains unchanged, while exact accepted statements and quotations are exposed
separately. The UI does not parse, summarize or rewrite a validated answer string.
Older history retains its complete legacy text. Rejected model candidates never
become public presentation fields.

The Streamlit workflow selector and controlled Chroma graph wiring are implemented
and offline validated. Live graph use remains disabled by default and has not been
verified on Replit or against a real provider. The UI's process-local question
counter is separate from durable provider-call accounting. The semantic checker
cannot override deterministic failures, establish support from unretrieved evidence
or replace caller-reviewed obligation coverage.

The [Pinecone credential loader](../src/northstar_atlas/vector_store/pinecone_cloud.py)
accepts all four required process variables when `.env` is absent. When the file
exists, containment, Git exclusion, parsing and environment/file conflict checks
remain mandatory; neither path bypasses target identity, owner confirmation or
credential-fingerprint binding. See the [credential tests](../tests/test_pinecone_cloud.py)
and [setup guidance](../README.md#optional-pinecone).

## Reviewed conditions and evidence-backed abstention

The standard `grounded-answer-v2` contract accepts caller-supplied reviewed
obligations. Every required ID must appear exactly once; missing, duplicated,
invented or waived requirements are rejected. Resolved statements need valid
source support. Unresolved obligations require explicit clarification or abstention.
Coverage is computed in application code, not accepted as a model's assertion.

For explicitly extractive obligations, the application-owned profile asks the
model to reference the trusted IDs and renders exact reviewed source spans.
The caller establishes applicability; the model cannot select new obligations
or turn irrelevant source text into an authoritative answer merely by naming an ID.
Version, status, hash, citation, eligibility and sibling checks still apply.
Copying text proves reproduction, not that the selection fully answers a question.

Evidence-linked statements permit supported paraphrases and retain their semantic
review limitation. Evidence-backed abstention requires the caller-reviewed
limitation's citation and quotation; missing-information labels are restricted.
No-evidence abstention is permitted without fabricated sources. An explicit
documented gap differs from failure to retrieve an answer and does not prove a
rule cannot exist outside the corpus. Ordinary UI questions do not receive the
hand-reviewed evaluation obligations automatically; coverage is `not_assessed`.

## Experiments and observed limitations

The following failures and corrections are part of the engineering record.
They have not been removed from history. Full operational diaries, request
authorizations, raw responses and temporary harnesses remain local; this summary
retains the findings without publishing account details or provider payloads.

| Experiment | Observation, response and remaining limit |
| --- | --- |
| Canonical chunks versus MiniLM input | 25 of 151 chunks exceeded 256 tokens. A separate refinement stage produced 189 embedding-ready chunks, with source reconstruction and explicit sibling dependencies. Exact reconstruction does not establish independent semantic completeness. |
| Dense retrieval baseline | Across 15 development questions, direct Recall@5 was 55.22%, expanded recall 64.67%; full evidence coverage rose from 26.67% to 40.00%. Required siblings repaired fragmentation but not every missing cross-policy dependency. |
| Chroma versus Pinecone | All 15 top-five chunk-ID sets and expanded sets matched; two ordering differences affected MRR. Managed storage did not improve evidence coverage in this experiment. |
| Dense/BM25/hybrid | Expanded recall is 64.67%/66.78%/70.44%, with full coverage 6/15 for all three. Hybrid still misses AKS provisioning evidence, regresses on some cases and increases context/latency cost. |
| Original `bounded_v1` graph | Optional five-node orchestration preserves eligibility and call bounds. Empty evidence or missing reviewed sources can trigger one alternate; ordinary nonempty evidence remains semantically not_assessed. |
| Final structural/semantic probe | A fake ordinary claim inventing an approved retention duration passed citation/quotation checks when paired with a genuine source quote. Its semantic assessment fails; graph routing and source copying do not prove entailment. Full-prompt checks also block EVAL-006 in all strategies and EVAL-010 in hybrid. |
| Quotation mismatch | A live Terraform quotation replaced line breaks with spaces and failed exact matching. Narrow whitespace normalization was added for contiguous source matching; changed/omitted/reordered words remain invalid. |
| Incomplete main answer | Terraform answers could cite or quote a requirement without explaining all mandatory conditions. Structured obligations made main-answer coverage explicit; earlier incomplete responses remained rejected. |
| Truncated JSON | A QAService Terraform response hit the 1,024-output-token ceiling. Incomplete JSON and `finish_reason="length"` remained rejected. A larger reservation addressed capacity but did not by itself solve exact extractive wording. |
| Preflight harness defect | Tuple versus JSON-list comparison stopped a retest before inference. Representation normalization preserved strict ID membership and order checks; generic grounding validators were not weakened to fix the harness. |
| Extractive generation mismatch | Larger responses still depended on the model reproducing reviewed wording. The application-owned profile subsequently rendered verified source statements instead; a live test accepted all four required Terraform IDs and their source bindings. |
| Uncited abstention | A retention response correctly avoided a duration but omitted required limitation evidence. Caller-reviewed evidence-backed abstention now requires the citation and quotation; no-evidence abstention remains separate. |
| Incorrect validation mode | A cited retention paraphrase was initially checked as extractive. Changing only its reviewed semantic requirement to `evidence_linked` allowed structural replay; the original extractive validator still rejects paraphrases. Semantic assessment remained separate. |
| Presentation cleanup | The combined answer text mixed primary statements and repeated evidence/technical details. An additive public projection and separate UI sections improved readability without changing generation, retrieval or validators. |

Recorded live QAService checks on 2026-10-23:

| Scenario | Accepted behavior | Structural result | Provider tokens, input / output | QA latency after initialization |
| --- | --- | --- | ---: | ---: |
| Terraform, application-owned profile | `answer` | 4/4 reviewed extractive IDs; exact source statements; PLAT-001 v0.1 proposed | 5,901 / 77 | 1.547 s |
| Retention, reviewed evidence-linked requirement | `abstain` | 1/1 reviewed limitation; COMP-001 v0.1 §11.1 proposed; valid exact quotation | 5,075 / 172 | 2.739 s |

These linear QAService observations do not verify the semantic graph. They are
individual smoke observations, not general success rates or latency
guarantees. Model/store initialization was measured separately at roughly 11–13
seconds in those Mac runs. The reviewed Terraform set was caller-supplied; the
ordinary UI does not infer it. The retention result did not invent a duration,
but structural validation alone does not establish entailment of all paraphrases.

The historical retrieval comparison measured mean adapter latency of 21.85 ms
for Chroma and 301.66 ms for Pinecone. It used one alternating pass, identical
query embeddings and scoring, and a fixed reference date. These are adapter
timings, including checks and context expansion, not isolated engine benchmarks.
See the [backend comparison report](decisions/M3.4-Chroma-Pinecone-Comparison.md) for the full
metric definitions and limitations. No new model or retrieval experiment was run
to prepare these public documents, and no historical result was promoted into a
production guarantee.

## Selected public decision records

| Topic | Records retained in the proposed publication set |
| --- | --- |
| Synthetic policy ownership and review | [HR](decisions/M1.1-HR-Policy-Review.md), [IT](decisions/M1.2-IT-Policy-Review.md), [Platform](decisions/M1.3-Platform-Policy-Review.md), [SRE](decisions/M1.4-SRE-Policy-Review.md), [Security](decisions/M1.5-Security-Policy-Review.md), [Compliance](decisions/M1.6-Compliance-Policy-Review.md) |
| Corpus and evidence labels | [Consistency](decisions/M1.7-Corpus-Consistency-Report.md), [evaluation design](decisions/M1.8-Evaluation-Design.md) |
| Ingestion and input compatibility | [Loader](decisions/M2.1-Document-Loading-Design.md), [canonical chunking](decisions/M2.2-Chunking-Design.md), [embeddings/refinement](decisions/M2.3-Embedding-Design.md) |
| Local retrieval | [Chroma design](decisions/M2.4-Chroma-Design.md), [retrieval baseline](decisions/M2.5-Retrieval-Evaluation.md) |
| Managed retrieval | [Pinecone design](decisions/M3.1-Pinecone-Design.md), [offline adapter](decisions/M3.2-Pinecone-Offline-Adapter.md), [backend comparison](decisions/M3.4-Chroma-Pinecone-Comparison.md) |
| Generation preparation | [Provider decision](decisions/M4.1-LLM-Provider-Design.md), [context/citations](decisions/M4.2-Context-Citation-Design.md) |
| Lexical and hybrid retrieval | [Design](decisions/M5.1-Hybrid-Retrieval-Design.md), [implementation and results](decisions/M5.2-Hybrid-Retrieval-Results.md) |
| Graph profiles and UI integration | [Original bounded design](decisions/M6.1-LangGraph-Design.md), [dependency resolution, routing/grading, semantic checking and integration history](decisions/M6.2-LangGraph-Implementation.md) |
| Controlled live execution | [Shared durable budget](decisions/Shared-Durable-Live-Budget.md), [conditional graph wiring and activation prerequisites](decisions/Controlled-Live-Graph-Wiring.md) |
| Final engineering evaluation | [Retrieval, graph behavior and limitations](decisions/Retrieval-and-Bounded-Graph-Evaluation.md) |

The full M3.3, M4.3, M4.4, M8.1 and M8.2 operational histories and the publication
audit are intentionally excluded from the proposed public set and ignored. They
remain at their original paths in the development checkout. This guide provides
their current architectural conclusions, unsuccessful experiments and limitations;
it does not claim those local files are available in a public clone. Historical
replay tests remain optional when their private artifacts are absent.

## Remaining release boundaries

Runtime root guards now share a source-anchored/operator-configured root validator,
and Pinecone account identity resides in ignored target configuration with a
separate index/key-bound owner confirmation. Source/manifest verification and
immutable publication checks remain intact. Alternate checkout fixtures pass;
this does not establish that a fresh install or Replit import works today.
Environment-only Pinecone credentials, the shared durable budget and conditional
live graph wiring are implemented and offline validated. Linux CPU packaging,
model provisioning, durable retrieval/ledger state and private-access guarantees
still require hosted validation. Live graph activation requires separate approval;
the offline UI-only environment cannot run it. The publication plan remains
private-first, with all included material suitable for eventual public release.
A license and public release remain separate decisions;
private repository visibility does not replace publication review.
Source policies, evaluation data and dependency pins remain unchanged by
publication preparation and the subsequent portability correction.
