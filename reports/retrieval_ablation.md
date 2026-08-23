# NoticeLens Retrieval Ablation

Generated: `2026-08-23T00:29:21.143813Z`

This is an isolated experiment over frozen artifacts. It does not prove that any retrieval technique is universally better, and it did not alter the production retriever.

## Frozen protocol

- Corpus/chunks: the exact 580 `heading_aware_220_40` records from the approved 50-document IRS corpus.
- Benchmark: exact S01-S15 questions; exact notice-code restriction; final `top_k=5`; unchanged full-heading-path scoring.
- BM25: one global Okapi index over exact prefixed chunk text; NFKC + casefold + Unicode alphanumeric tokenizer; `k1=1.5`, `b=0.75`; exact-notice candidates only.
- Hybrid: equal-weight reciprocal-rank fusion of dense top 5 and BM25 top 5 with `k=60`; deterministic chunk-ID tie break.
- Reranking: `bge-reranker-v2-m3` reranks exactly the hybrid top 5; no candidate expansion, query rewriting, or per-question tuning.
- Latency: one pass over S01-S15. Dense latency is the frozen Phase 4B Pinecone-query measurement; BM25/fusion and hosted-rerank components were measured now. Hybrid totals are disclosed component sums; common query embedding time is excluded.
- Study-only latency guardrail: p95 <= 2.0s. This was frozen before results and is not a production SLA.

## Reranker capability

Nebius was preferred, but its live catalog exposed no reranker and the documented `/v1/rerank` probe returned 404. The existing Pinecone account exposed and successfully probed `bge-reranker-v2-m3`, so it was used as the fallback.

- Nebius reference: https://docs.tokenfactory.nebius.com/api-reference/inference/rerank-documents
- Pinecone reference: https://sdk.pinecone.io/python/how-to/inference/reranking.html
- Pinecone rerank calls in measured study: 15; rerank units: 15.
- Vector index queries/writes/deletes/creates in this runner: 0/0/0/0. Dense ranks were replayed from the frozen Phase 4B trace.

## Aggregate results

| Variant | Section P@1 | MRR | Hit@5 | Median retrieval latency | p95 retrieval latency | Remaining P@1 failures |
|---|---:|---:|---:|---:|---:|---|
| Heading-aware Dense | 93.33% (14/15) | 0.95556 | 100.00% | 0.097893s | 0.246287s | S06 |
| Heading-aware BM25 | 40.00% (6/15) | 0.61556 | 100.00% | 0.000094s | 0.000229s | S03, S04, S06, S08, S10, S11, S12, S13, S15 |
| Heading-aware Hybrid (RRF) | 66.67% (10/15) | 0.83333 | 100.00% | 0.098132s | 0.246513s | S03, S04, S06, S08, S11 |
| Heading-aware Hybrid + Reranker | 93.33% (14/15) | 0.96667 | 100.00% | 0.315216s | 0.506594s | S11 |

## Paired per-question comparison

Ranks are the first exact full-heading-path match within the final five.

| ID | Notice | Dense | BM25 | Hybrid | Hybrid + reranker |
|---|---|---:|---:|---:|---:|
| S01 | CP14 | 1 | 1 | 1 | 1 |
| S02 | CP501 | 1 | 1 | 1 | 1 |
| S03 | CP503 | 1 | 2 | 2 | 1 |
| S04 | CP504 | 1 | 4 | 2 | 1 |
| S05 | LT11 / Letter 1058 | 1 | 1 | 1 | 1 |
| S06 | CP2501 | 3 | 5 | 2 | 1 |
| S07 | CP2000 series | 1 | 1 | 1 | 1 |
| S08 | CP3219A | 1 | 4 | 2 | 1 |
| S09 | CP30 | 1 | 1 | 1 | 1 |
| S10 | CP523 | 1 | 2 | 1 | 1 |
| S11 | CP59 | 1 | 5 | 2 | 2 |
| S12 | CP515 | 1 | 2 | 1 | 1 |
| S13 | CP518 | 1 | 3 | 1 | 1 |
| S14 | CP2566 | 1 | 1 | 1 | 1 |
| S15 | CP3219N | 1 | 2 | 1 | 1 |

## Decision

Retain the simpler heading-aware dense retriever; no tested variant met the quality-and-latency rule.

- Heading-aware BM25: P@1 delta -53.33 points; fixes S06=False; new rank-1 regressions=S03, S04, S08, S10, S11, S12, S13, S15; quality eligible=False; latency reasonable=True.
- Heading-aware Hybrid (RRF): P@1 delta -26.67 points; fixes S06=False; new rank-1 regressions=S03, S04, S08, S11; quality eligible=False; latency reasonable=True.
- Heading-aware Hybrid + Reranker: P@1 delta +0.00 points; fixes S06=True; new rank-1 regressions=S11; quality eligible=False; latency reasonable=True.

`reports/final_retrieval_config.json` remained byte-identical at `5b7f834eb2cf65cd153c644bf62b7852116b99bfaa64a5ec90bf9cbb6fc9eb41`. Even if an experimental candidate qualifies, adoption requires a separate approval.

## LlamaIndex time-box estimate

A small LlamaIndex retriever comparison is conditionally feasible in about 40-60 minutes after the main app: install the minimal core/adapters, wrap the frozen registry or existing namespace without re-embedding, run the same 15 questions, and report. Stop at 60 minutes if adapter/version friction appears; it should not delay Streamlit.

## Integrity

- Frozen Phase 1-4B composite plus pinned Phase 5/5.1 inputs: PASS (`5df91e5deaba6fb47b69e6227718956575eea3780c83dbf3df0843d747be95fa`).
- Heading registry SHA-256: `3aecf5db7ee5fe857bdb99156c9bb5ba585f2e845aa6697cc4ece8902ac27572`.
- Benchmark SHA-256: `1090c8b41f0b007adfda1eb9882b0237d93416a3ce57857bc4da58a8947aafa8`.
- Frozen dense result SHA-256: `36188d2d9f273b0cbe81a77d59e980c11ae1bbdd7748797ee47f1b29a140d189`.
- Benchmark questions, chunks, embeddings, Pinecone namespaces, production source, and production configuration were not changed.
