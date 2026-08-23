# PHASE 5 — FINAL RAG CORE REPORT

1. **Final retrieval configuration:** `Qwen/Qwen3-Embedding-8B` (4096 dimensions), Pinecone index `noticelens-rag`, namespace `heading-aware-dense`, `heading_aware_220_40`, exact notice-code filter, top 5. BM25, hybrid retrieval, and reranking remain disabled because the frozen dense notice benchmark was already perfect and heading-aware section retrieval cleared the precommitted gain threshold.
2. **Nebius generation model:** `Qwen/Qwen3-30B-A3B-Instruct-2507`. It was present in the live catalog and passed the actual strict answer-schema probe. Selection reason: live catalog availability plus a successful strict structured-output probe; chosen as a latency-conscious instruction model for concise grounded RAG.
3. **Notice identity behavior:** local header-first deterministic parsing; 8/8 official samples identified correctly, with ambiguous and missing identities stopped before retrieval. Reviewed aliases route variants such as CP503C→CP503 and CP523H→CP523 without generic suffix stripping.
4. **Field extraction behavior:** local PDF text-layer extraction only (OCR requests: 0); code/date/deadline/amount/reference values are emitted only from explicit labels, each with confidence and source text. Missing, conflicting, invalid, relative, or placeholder-only values remain null.
5. **LangGraph nodes/edges:** `START → identify_notice`; unidentified/ambiguous → `clarify_or_fail → END`; identified → `retrieve_guidance`; insufficient/unsupported → `refuse → END`; sufficient → `generate_grounded_answer → END`. No router, per-chunk grader, retry loop, or cycle.
6. **Evidence sufficiency:** deterministic nonempty, same-notice, required-metadata checks with no score threshold. Categorical unsupported requests short-circuit provider calls. The structured model can still return insufficient, which becomes the exact refusal fallback.
7. **Citations:** final claims use application-owned LangChain `Document` metadata. Guidance claims are exact cited IRS excerpts; notice claims are rendered from deterministic field source text. The model cannot supply source URLs or citation metadata.
8. **Faithfulness:** formal frozen-plan faithfulness is **pending human review** and the ≥95% target is not claimed as met. Separately, 21/21 generated atomic claims passed exact-source citation support (100.00%); the auxiliary structured judge support rate was 100.00%. Answerable-response coverage was 9/10; non-answered question IDs: D01.
9. **Frozen refusal test:** 4/4 correct (100.00%); target met: TRUE.
10. **Warm retrieval latency:** median 1.128s; p95 1.919s (embedding and Pinecone reported separately in `phase5_latency.json`).
11. **Warm generation latency:** median 9.107s; p95 24.627s.
12. **Warm end-to-end median:** 10.649s.
13. **Warm end-to-end p95:** 25.092s.
14. **Frozen <6s target:** FALSE (18/20 measured warm requests answered; passing requires all requests answered and p95 strictly below 6 seconds).
15. **Tests:** 104 passed, 0 failed across the complete Phase 1–5 offline suite.
16. **Limitations:** text-layer PDFs only; no OCR fallback was justified by the samples; explicit variant routing is intentionally finite; retrieval is bounded to the frozen IRS corpus; exact-excerpt answers favor faithfulness over stylistic paraphrase; the same model's judge is auxiliary; independent human faithfulness review is still pending; no personalized tax/legal advice. Cold-start end-to-end (including initialization) median was 14.870s and p95 was 21.369s; 2/3 cold requests answered. Cold embedding, Pinecone, retrieval, and generation components are reported separately in `phase5_latency.json`.
17. **Freeze confirmation:** Phase 1–4B approved composite `5df91e5deaba6fb47b69e6227718956575eea3780c83dbf3df0843d747be95fa` remained byte-identical before and after Phase 5.

No UI, Streamlit app, Git initialization, BM25, hybrid retrieval, reranker, Mem0, LlamaIndex, Lyzr, or ElevenLabs component was added.
