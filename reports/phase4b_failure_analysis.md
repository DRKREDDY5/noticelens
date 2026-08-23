# Phase 4B heading-aware chunking analysis

This experiment changes only chunk construction. It uses the same corpus, questions, embedding model, index, cosine metric, known-notice filter, and top-5 scoring.
The heading-aware treatment bundles section boundaries with a deterministic structural prefix; results are attributed to the strategy, not boundaries alone.
Execution mode: `query_only_recovery_after_completed_namespace_population`.
This report was completed by a query-only recovery after the initial attempt had populated the exact 580 deterministic heading IDs. The recovery embedded only the 15 questions and performed zero document embeddings or upserts. Stored-vector model/upsert provenance is inherited from that initial attempt; exact remote ID parity is mechanically verified. The initial attempt stopped during post-index querying when one raw Pinecone response failed the then-strict descending-score-order check.

## Chunk audit

- Heading-aware chunks: 580
- Oversized sections split: 33
- Chunks crossing heading boundaries: 0
- Unassigned useful pre-H2 bodies: 1 (CP2000 series; no heading path was invented)

## Metric comparison

| System | n | Section P@1 | Section MRR | Section Hit@5 |
|---|---:|---:|---:|---:|
| Fixed 220/40 | 15 | 80.0% | 0.8689 | 100.0% |
| Heading-aware | 15 | 93.3% | 0.9556 | 100.0% |

P@1 change: 13.33 points. MRR change: 0.0867. Hit@5 change: 0.0000.

## Frozen fixed-failure comparison

### S03 — CP503

- Expected section: How much time do I have?
- Fixed rank: 2
- Heading-aware rank: 1
- Fixed top-1 preview: information](https://www.irs.gov/payments/online-account-for-individuals) pertaining to your tax account. - Learn more about your [payment options](https://www.irs.gov/payments) and how to make a [payment arrangement](https://www.irs.gov/payments/payment-plans-installment-agreements). - [Request an appeal](https://www.irs.gov/appeals/preparing-a-request-for-
- Heading-aware top-1 preview: Understanding your CP503 notice Notice: CP503 Section: Frequently asked questions > How much time do I have? You must pay the entire balance by the due date shown on your notice to avoid additional penalties and interest.
- Outcome: The heading-aware strategy changed this from a rank-1 miss to a rank-1 section match.

### S07 — CP2000 series

- Expected section: Check and correct your records
- Fixed rank: 5
- Heading-aware rank: 1
- Fixed top-1 preview: # Understanding your CP2000 series notice Source: https://www.irs.gov/individuals/understanding-your-cp2000-series-notice The CP2000 notice series includes: CP2000, CP2000A, CP2000B, CP2000C, CP2000D and CP2000E. Learn what your notice is about and what to do. ## Why you received this notice The income or payment information we received from third parties, s
- Heading-aware top-1 preview: Understanding your CP2000 series notice Notice: CP2000 series Section: You may want to > Check and correct your records - [Get a transcript](https://www.irs.gov/individuals/get-transcript) of your original tax return, if needed. - Correct your copy of your tax return. Keep it and the notice for your records. - Check your tax returns from prior years. If they
- Outcome: The heading-aware strategy changed this from a rank-1 miss to a rank-1 section match.

### S11 — CP59

- Expected section: What should I do if I've just filed my tax return?
- Fixed rank: 3
- Heading-aware rank: 1
- Fixed top-1 preview: # Understanding your CP59 notice Source: https://www.irs.gov/individuals/understanding-your-cp59-notice ## What this notice is about We have no record that you filed your prior year personal tax return. ## What you need to do File your personal tax return immediately or explain to us why you don't need to file. Note: If you received an IRS-issued identity pr
- Heading-aware top-1 preview: Understanding your CP59 notice Notice: CP59 Section: Frequently asked questions > What should I do if I've just filed my tax return? You don't have to do anything if you filed your tax return within the last eight weeks.
- Outcome: The heading-aware strategy changed this from a rank-1 miss to a rank-1 section match.

## Heading-aware P@1 failures

### S06 — CP2501

- Question: The bank information is wrong. What should I send back to explain why I disagree?
- Expected path: `What you need to do > If you don’t agree or the information is incorrect`
- Heading-aware first correct rank: 3
- Rank-1 path: `Why you received this notice`
- Rank-1 preview: Understanding your CP2501 notice Notice: CP2501 Section: Why you received this notice The income or payment information we received from third parties, such as employers or financial institutions, doesn't match the information you reported on your tax return. This difference may increase or decrease your tax or may not change it at all. The notice explains p

## Paired outcome

- Improved: 3
- Unchanged: 11
- Regressed: 1
- New regressions versus fixed P@1: S06

## Decision

The precommitted >=10-point P@1 threshold was met.
Evidence-only retention decision: retain heading-aware chunking.

## Regression protection

- All approved Phase 1–4A file and tree hashes matched before and after.
- The baseline namespace remained exactly 350 frozen IDs.
- The exact preexisting 580-vector heading namespace was reused without document embedding or upsert.
- Provider response-order normalizations: 0 (descending Pinecone similarity only; no reranker).
- No index creation, deletion, clearing, update, baseline upsert, BM25, hybrid retrieval, reranking, rewriting, generation, LangGraph, or Streamlit was used.
