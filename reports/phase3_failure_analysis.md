# NoticeLens Phase 3 Baseline Failure Analysis

This report records the frozen dense baseline exactly as run. It does not tune or repair retrieval.

## Run contract

- Embedding model: `Qwen/Qwen3-Embedding-8B` (4096 dimensions)
- Chunking: `fixed_220_40` (220 tokens, 40 overlap)
- Pinecone: `noticelens-rag` / `baseline-fixed-dense` / cosine / top 5
- Query input: exact frozen question text only; no filters, rewriting, hybrid search, reranking, or generation
- Scored population: categories A-C only (n=22); category D is observational; category E was not queried

## Metric summary

| Scope | n | P@1 | MRR | Hit@5 |
|---|---:|---:|---:|---:|
| Overall A-C | 22 | 100.0% | 1.000 | 100.0% |
| A exact code | 6 | 100.0% | 1.000 | 100.0% |
| B everyday language | 8 | 100.0% | 1.000 | 100.0% |
| C confusable family | 8 | 100.0% | 1.000 | 100.0% |

## Failure counts

- Precision@1 failures: 0
- Hit@5 failures: 0
- Confusable-family rank-1 mismatches: 0
- Everyday-language semantic (category B) rank-1 failures: 0
- Naive-language rank-1 failures: 0
- Exact-code rank-1 failures: 0
- Expected notice recovered only at ranks 2-5: 0
- Semantically related/procedurally wrong candidates requiring human review: 0

A procedurally wrong-neighbor label is only a review flag when rank 1 is a declared confusable notice or shares the frozen notice family; it is not an automatic legal conclusion.

## Most interesting retrieval misses (up to five)

| ID | Scope | Expected | Rank 1 | Best expected rank | Declared relationship |
|---|---|---|---|---:|---|
| D01 | observational D | CP501 | CP101 | 2 | none |

Only 1 retrieval miss existed across scored A-C and observational D traces; no extra cases were invented.

## Every scored A-C Precision@1 failure

None.

## Null-hypothesis check

Dense confusable-family Precision@1 is 100.0%. Dense confusable-family retrieval may be near ceiling; later improvement must still follow the frozen comparison rules.

## Section-question observations

The four D questions were run, but fixed-size chunks have no reliable heading attribution. Their full top-5 traces are in `phase3_baseline_results.json`; no Section Precision@1 was fabricated.

- D01: expected `CP501` at rank 2; rank 1 was `CP101`; expected-heading literal visible in a retrieved expected-document chunk: false.
- D02: expected `CP2000 series` at rank 1; rank 1 was `CP2000 series`; expected-heading literal visible in a retrieved expected-document chunk: true.
- D03: expected `CP523` at rank 1; rank 1 was `CP523`; expected-heading literal visible in a retrieved expected-document chunk: true.
- D04: expected `CP59` at rank 1; rank 1 was `CP59`; expected-heading literal visible in a retrieved expected-document chunk: true.
- These are observations only, not formal Section Precision@1 scores.

## Deferred future experiments (not implemented)

Heading-aware chunks, BM25/hybrid retrieval, metadata filters, optional reranking, refusal/abstention routing, and any small agentic workflow remain later-phase candidates. No such improvement was implemented or tuned in this baseline.
