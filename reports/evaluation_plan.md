# NoticeLens Phase 2 Evaluation Plan

Status: **precommitted before retrieval implementation or result inspection**  
Dataset: `eval/golden_questions.json`  
Authoritative corpus: `data/corpus_manifest.csv` and the 50 frozen files in `data/processed/guidance/`  
Date fixed: 2026-08-22

No retrieval, generation, latency, or refusal results are calculated in Phase 2.

## Evaluation scope

The golden set contains 30 fixed questions:

- 22 document-level questions: categories A–C
- 4 section-level questions: category D
- 4 unsupported/refusal questions: category E

The four category-D questions are evaluated separately from the 22 document-level questions so section performance is not hidden inside document-level performance. Refusal questions are excluded from retrieval metrics and are used only for the later refusal evaluation.

The sample-notice PDFs are not part of the permanent retrieval corpus. They may later support a separate notice-identity evaluation, but they do not supply evidence for these golden questions.

## Frozen ground truth and relevance

For every non-refusal question, `expected_doc_id` must resolve to exactly one row in the authoritative manifest, and `expected_notice_code` must equal that row's notice code. The manifest value is authoritative for composite identifiers such as `CP2000 series`, `CP06/CP06A`, and `LT11 / Letter 1058`.

A document-level retrieved chunk is relevant only when its source `doc_id` equals the question's `expected_doc_id`. The notice code is then checked against the manifest row. A same-family or confusable notice is not relevant.

A section-level retrieved chunk is relevant only when both conditions hold:

1. Its source notice/document matches `expected_notice_code` and `expected_doc_id`.
2. Its attributed source section matches `expected_heading`.

Heading comparison uses Unicode NFC normalization, trims leading and trailing whitespace, and collapses internal whitespace. Punctuation and wording otherwise remain significant; semantic substitutes do not count. A chunk spanning multiple sections is attributed to the section containing its first substantive content, using source offsets against the frozen Markdown.

## Precommitted metrics

### Primary: Notice Precision@1

Notice Precision@1 is measured over the 22 category A–C questions:

`Notice P@1 = questions whose rank-1 chunk has the expected_doc_id / 22`

The same metric is reported separately for the eight category-C confusable-family questions because that subset is the primary retrieval stress test. Raw counts accompany percentages.

### Secondary: MRR and Hit@5

MRR and Hit@5 use the same 22 category A–C questions and the same document-level relevance rule.

- `MRR = mean(1 / rank of the first expected_doc_id)`; a question with no relevant result in the evaluated ranking receives 0.
- `Hit@5 = questions with at least one expected_doc_id in ranks 1–5 / 22`.

No credit is added for retrieving several chunks from the expected document.

### Section experiment: Section Precision@1

Section Precision@1 is measured over the four category-D questions:

`Section P@1 = questions whose rank-1 chunk matches both the expected document/notice and expected heading / 4`

A chunk from the right notice but the wrong section is incorrect. Raw counts accompany percentages because each question changes this metric by 25 percentage points.

### Metrics measured later

These are defined now but are not calculated in Phase 2:

- **Notice Identity Accuracy:** proportion of eligible sample-notice inputs for which the notice-code extraction stage returns the pre-labelled notice identity. This is a separate input-identification evaluation, not a retrieval score.
- **Faithfulness:** proportion of atomic answer claims supported by the retrieved frozen-corpus evidence. Unsupported or contradicted claims are unfaithful. Scoring will use the record's `expected_answer_facts` as the gold claim set plus human evidence review.
- **Correct refusal:** proportion of category-E questions for which the system declines to invent or infer unsupported information and does not provide the requested unsupported answer. Report the raw score out of four as well as the percentage.
- **p95 end-to-end latency:** the 95th percentile from query submission through final response under a fixed environment, run policy, and repetition count recorded before execution.

The brief supplies no numeric faithfulness target or p95 latency ceiling. Those gates must be fixed and documented before Phase 3 runs; they must not be chosen after seeing results.

## Hypothesis and decision rules

The primary hypothesis is that hybrid/code-aware retrieval is meaningfully better than the dense baseline on confusable notices.

It is accepted as meaningfully better only if all of the following hold:

1. Confusable-family Notice P@1 improves by at least **10 absolute percentage points** over the dense baseline.
2. Answer faithfulness does not decrease relative to the dense baseline.
3. The optimized system meets the latency gate fixed before evaluation.

Because category C has eight questions, one additional correct rank-1 result changes its score by 12.5 percentage points. Paired per-question outcomes and raw counts will therefore be reported with the aggregate score.

The notice-identity improvement hypothesis is treated as effectively **null** when both conditions hold:

- Dense confusable-family Notice P@1 is at least **95%**.
- The optimized improvement is less than **5 absolute percentage points**.

Results between the meaningful-improvement and null rules are reported as inconclusive; they are not relabelled as success.

## Null-branch section experiment

If notice identity is already near ceiling, the next comparison is fixed-size chunking versus heading-aware chunking on category D.

- A gain of at least **10 absolute percentage points** in Section P@1 is meaningful.
- With four section questions, one additional correct result changes Section P@1 by 25 percentage points, so paired outcomes and raw counts are required.
- If both notice identity and section retrieval are near ceiling, record the null result and keep the simpler retrieval system.

## Fair comparison protocol

Before any evaluation run:

1. Freeze and record hashes for the golden JSON, evaluation CSV, corpus manifest, and processed-corpus inventory.
2. Freeze each system's chunking, metadata, code-normalization, ranking, fusion, top-k, and prompt configuration.
3. Use the exact same question text and frozen corpus for every compared system.
4. Retrieve at least five ranked chunks so MRR and Hit@5 are measurable.
5. Do not tune on these 30 questions. Any development/tuning set must be separate.
6. Record raw ranked results, document IDs, heading attribution, generated answers, refusal decisions, and timing traces for every question.
7. Run latency tests under the same hardware, provider, cache policy, concurrency, and repetition policy.

Known-code normalization may map a user's `CP2000` mention to the manifest's `CP2000 series` and `LT11` to `LT11 / Letter 1058`, but those rules must be frozen before a run and applied identically wherever the evaluated design calls for them.

## Failure labels

Per-question review will distinguish:

- wrong notice at rank 1
- right notice but wrong section
- expected notice absent from the top five
- unsupported or contradicted answer claim
- citation/evidence mismatch
- false answer on a refusal item
- false refusal on an answerable item
- ambiguity-routing failure
- latency outlier

## Change control

The golden set is fixed before implementation. A genuine ground-truth correction requires a documented reason, a new dataset version and hash, and a rerun of every compared system. Question wording or labels must never be silently changed in response to retrieval performance.
