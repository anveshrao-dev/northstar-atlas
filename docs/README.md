# Architecture, experiments and documentation

This is the public engineering overview for the local Northstar Atlas MVP.
The [project README](../README.md) contains current operation and setup guidance.
Decision records below describe the state at the time of each experiment or
design. Dated proposals and prices are historical, not current operating guidance.
Existing M-prefixed filenames and experiment identifiers remain stable for links,
artifact provenance and reproducibility; document titles describe their subject.

## Current component boundaries

| Component | Responsibility and boundary |
| --- | --- |
| Ingestion and manifest | Load only six allowlisted Markdown policies; validate metadata and SHA-256 hashes. Reviews and evaluation answers are not source documents. |
| Canonical chunker | Preserve section IDs, headings, rules, tables and source order. The current corpus produces 151 canonical chunks. |
| MiniLM refinement | Use the pinned tokenizer to refine 25 oversized chunks into a 189-chunk embedding view. Track source spans and duplicated context; retain the canonical representation. |
| Embeddings | CPU MiniLM, 384 dimensions, L2 normalization and a 256-token input ceiling. Reject oversized/non-finite/incompatible inputs rather than truncate. |
| Vector stores | Chroma persists locally; Pinecone uses managed immutable namespace publication. Both retain source/model identity, status, section citations and required siblings. Incomplete indexes cannot masquerade as ready. |
| Context builder | Validate eligibility and source relationships, include required siblings, deduplicate, group evidence and budget labels/content together. Missing critical context blocks acceptance. |
| Generation/grounding | Provider-independent structured requests; Nebius adapter with bounded transport behavior. Validate response shape, supplied citations, quotation spans and caller-reviewed requirements; do not confuse structural validation with semantics. |
| QAService | Reuse initialized retrieval/model/provider resources; return immutable public success/error models separately from developer diagnostics. Support answer, clarify and abstain without treating infrastructure errors as policy claims. |
| Streamlit | Display validated statements, source labels and optional evidence/validation details. Default to authored offline demonstrations; require server and session opt-in for paid questions. |

The public result's structured presentation is additive. Legacy rendered `text`
remains unchanged, while exact accepted statements and quotations are exposed
separately. The UI does not parse, summarize or rewrite a validated answer string.
Older history retains its complete legacy text. Rejected model candidates never
become public presentation fields.

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

These are individual smoke observations, not general success rates or latency
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
Linux CPU packaging, environment-only hosted
secrets, model provisioning, durable retrieval state and access controls remain
open. The repository will initially be private, with the first commit limited to
public-suitable content. A license and public release remain separate decisions;
private repository visibility does not replace publication review.
Source policies, evaluation data and dependency pins remain unchanged by
publication preparation and the subsequent portability correction.
