# PHASE 5.1 — FINAL GENERATION MODEL REPORT

Phase 5.1 compared live generation models over the frozen NoticeLens retrieval system. No corpus, chunk, embedding, Pinecone, filter, graph, refusal, citation-metadata, or question configuration was changed.

## 1. Models tested

- `Qwen/Qwen3-30B-A3B-Instruct-2507` — fully_benchmarked
- `nvidia/Nemotron-3_5-Lightning` — probe_rejected
- `deepseek-ai/DeepSeek-V4-Flash-0731` — fully_benchmarked
- `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B` — probe_rejected

## Results

| Model | Faithfulness | Refusal | Answered | Gen median / p95 | E2E median / p95 | Structured success |
|---|---:|---:|---:|---:|---:|---:|
| `Qwen/Qwen3-30B-A3B-Instruct-2507` | 100.00% (23/23) | 4/4 | 9/10 | 7.929s / 18.957s | 9.775s / 22.050s | 100.00% |
| `deepseek-ai/DeepSeek-V4-Flash-0731` | 100.00% (6/6) | 4/4 | 3/10 | 10.567s / 17.766s | 11.904s / 18.950s | 83.33% |

## Decision

**Final model selected:** `Qwen/Qwen3-30B-A3B-Instruct-2507`

It satisfied formal faithfulness ≥95%, refusal 4/4, and exact citation provenance, then passed the material-grounding no-degradation safeguard before latency ranking. It had the lowest measured warm end-to-end p95 among the remaining eligible models.

- Faithfulness target met: **True**
- Warm p95 <6s target met: **False**
- Exact final measured warm end-to-end p95: **22.050282s**

## Method and safeguards

- Formal faithfulness uses each final app-rendered factual claim as one unit. Compound/list claims are all-or-nothing. A claim is supported only when deterministic citation/field provenance is valid and a separate blinded evaluator labels the entire claim supported.
- Faithfulness is support, not answer completeness or general correctness; answer coverage and false refusals are reported separately.
- The brief forbids choosing a faster model when grounding materially degrades. The approved 9/10 answerable reference is therefore a disclosed eligibility safeguard; structured-output reliability remains a later tie-breaker.
- The initial blinded evaluation had one failed batch; only its eight unjudged claims were replayed in smaller blinded batches with the same evaluator, prompt, schema, and evidence.
- Refusal scores are system-level because frozen deterministic policy routing skipped all generation-provider calls for E01–E04.
- Warm latency used two excluded warmups and 20 measured, sequential full-pipeline calls per model; p95 is nearest-rank. Cold measurements used three fresh client/probe/index setups.
- Final config generation_model changed: **False**; every retrieval field stayed identical.
- Tests passed: **120**, failed: **0**.
- Frozen retrieval artifacts remained unchanged: **True**.

No UI or post-Phase-5.1 architecture was built.
