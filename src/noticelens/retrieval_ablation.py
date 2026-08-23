"""Isolated BM25, hybrid, and reranking ablation for the frozen RAG core.

This module is experiment-only.  It reads the approved Phase 4B registry,
benchmark, and dense traces; it never creates, updates, deletes, or queries a
Pinecone vector index.  The only live provider operation is standalone hosted
reranking of five already-retrieved candidates.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
import time
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from pinecone import Pinecone

from .config import DEFAULT_SECRETS_PATH, load_phase3_config
from .heading_chunking import HeadingChunkRecord, load_heading_registry
from .phase4a import (
    SectionScore,
    aggregate_section_scores,
    load_section_questions,
    normalized_path,
    score_section_ranking,
)
from .phase5_1 import verify_phase51_frozen_inputs


REGISTRY_PATH = Path("data/derived/phase4b/heading_aware_220_40_chunks.jsonl")
BENCHMARK_PATH = Path("eval/section_questions.json")
DENSE_RESULTS_PATH = Path("reports/phase4b_heading_results.json")
FINAL_CONFIG_PATH = Path("reports/final_retrieval_config.json")
CSV_REPORT_PATH = Path("reports/retrieval_ablation.csv")
MARKDOWN_REPORT_PATH = Path("reports/retrieval_ablation.md")

EXPECTED_REGISTRY_SHA256 = "3aecf5db7ee5fe857bdb99156c9bb5ba585f2e845aa6697cc4ece8902ac27572"
EXPECTED_BENCHMARK_SHA256 = "1090c8b41f0b007adfda1eb9882b0237d93416a3ce57857bc4da58a8947aafa8"
EXPECTED_DENSE_RESULTS_SHA256 = "36188d2d9f273b0cbe81a77d59e980c11ae1bbdd7748797ee47f1b29a140d189"
EXPECTED_FINAL_CONFIG_SHA256 = "5b7f834eb2cf65cd153c644bf62b7852116b99bfaa64a5ec90bf9cbb6fc9eb41"

TOP_K = 5
BM25_K1 = 1.5
BM25_B = 0.75
RRF_K = 60
RERANK_MODEL = "bge-reranker-v2-m3"
LATENCY_REASONABLE_P95_SECONDS = 2.0
VARIANTS = ("dense", "bm25", "hybrid", "hybrid_reranker")
VARIANT_LABELS = {
    "dense": "Heading-aware Dense",
    "bm25": "Heading-aware BM25",
    "hybrid": "Heading-aware Hybrid (RRF)",
    "hybrid_reranker": "Heading-aware Hybrid + Reranker",
}


class AblationGateError(RuntimeError):
    """Fail-closed experiment error that never contains credential material."""


def _safe_provider_error(stage: str, exc: BaseException) -> AblationGateError:
    return AblationGateError(f"{stage} failed ({type(exc).__name__}); no credentials were logged")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    temporary.replace(path)


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def tokenize_bm25(text: str) -> list[str]:
    """Frozen sparse tokenizer: NFKC, casefold, Unicode alphanumeric runs."""

    if not isinstance(text, str):
        raise TypeError("BM25 input must be text")
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return re.findall(r"[^\W_]+", normalized, flags=re.UNICODE)


@dataclass(frozen=True)
class BM25Corpus:
    records: dict[str, HeadingChunkRecord]
    notice_ids: dict[str, tuple[str, ...]]
    term_frequencies: dict[str, Counter[str]]
    lengths: dict[str, int]
    idf: dict[str, float]
    average_length: float

    @classmethod
    def build(cls, records: Sequence[HeadingChunkRecord]) -> "BM25Corpus":
        if not records:
            raise AblationGateError("Cannot build BM25 over an empty registry")
        by_id: dict[str, HeadingChunkRecord] = {}
        notice_ids: dict[str, list[str]] = defaultdict(list)
        term_frequencies: dict[str, Counter[str]] = {}
        lengths: dict[str, int] = {}
        document_frequency: Counter[str] = Counter()
        for record in records:
            if record.chunk_id in by_id:
                raise AblationGateError("The heading registry contains duplicate chunk IDs")
            tokens = tokenize_bm25(record.text)
            if not tokens:
                raise AblationGateError(f"Heading chunk {record.chunk_id} has no BM25 tokens")
            by_id[record.chunk_id] = record
            notice_code = str(record.metadata["notice_code"])
            notice_ids[notice_code].append(record.chunk_id)
            frequencies = Counter(tokens)
            term_frequencies[record.chunk_id] = frequencies
            lengths[record.chunk_id] = len(tokens)
            document_frequency.update(frequencies.keys())
        document_count = len(by_id)
        idf = {
            term: math.log(1.0 + (document_count - frequency + 0.5) / (frequency + 0.5))
            for term, frequency in document_frequency.items()
        }
        return cls(
            records=by_id,
            notice_ids={key: tuple(sorted(value)) for key, value in notice_ids.items()},
            term_frequencies=term_frequencies,
            lengths=lengths,
            idf=idf,
            average_length=sum(lengths.values()) / document_count,
        )

    def rank(self, query: str, notice_code: str, *, top_k: int = TOP_K) -> list[tuple[str, float]]:
        candidate_ids = self.notice_ids.get(notice_code, ())
        if not query.strip() or not candidate_ids:
            raise AblationGateError("BM25 query has no text or exact-notice candidates")
        query_terms = tokenize_bm25(query)
        scores: list[tuple[str, float]] = []
        for chunk_id in candidate_ids:
            frequencies = self.term_frequencies[chunk_id]
            length = self.lengths[chunk_id]
            normalization = BM25_K1 * (
                1.0 - BM25_B + BM25_B * length / self.average_length
            )
            score = 0.0
            for term in query_terms:
                frequency = frequencies.get(term, 0)
                if frequency:
                    score += self.idf.get(term, 0.0) * (
                        frequency * (BM25_K1 + 1.0) / (frequency + normalization)
                    )
            scores.append((chunk_id, score))
        scores.sort(key=lambda item: (-item[1], item[0]))
        return scores[: min(top_k, len(scores))]


def reciprocal_rank_fusion(
    dense_ids: Sequence[str], bm25_ids: Sequence[str], *, top_k: int = TOP_K
) -> list[tuple[str, float]]:
    """Equal-weight RRF with the frozen constant and deterministic ID ties."""

    if not dense_ids or not bm25_ids or len(set(dense_ids)) != len(dense_ids) or len(set(bm25_ids)) != len(bm25_ids):
        raise AblationGateError("RRF inputs must be nonempty unique rankings")
    scores: dict[str, float] = defaultdict(float)
    for ranking in (dense_ids, bm25_ids):
        for rank, chunk_id in enumerate(ranking, start=1):
            scores[chunk_id] += 1.0 / (RRF_K + rank)
    ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    return ordered[: min(top_k, len(ordered))]


def _finite_score(value: Any) -> float:
    if isinstance(value, bool):
        raise AblationGateError("Provider returned a nonnumeric reranker score")
    try:
        observed = float(value)
    except (TypeError, ValueError):
        raise AblationGateError("Provider returned a nonnumeric reranker score") from None
    if not math.isfinite(observed):
        raise AblationGateError("Provider returned a nonfinite reranker score")
    return observed


class PineconeHostedReranker:
    """Standalone reranker; it has no vector-index mutation or query surface."""

    def __init__(self, *, api_key: str, client: Any | None = None) -> None:
        if not isinstance(api_key, str) or not api_key.strip():
            raise AblationGateError("PINECONE_API_KEY is empty")
        try:
            self._pc = client if client is not None else Pinecone(api_key=api_key)
        except Exception as exc:
            raise _safe_provider_error("Pinecone inference client construction", exc) from None
        self.request_count = 0
        self.rerank_units = 0

    def __repr__(self) -> str:
        return f"PineconeHostedReranker(model={RERANK_MODEL!r})"

    def require_model(self) -> list[str]:
        try:
            catalog = self._pc.inference.list_models(type="rerank")
            names = sorted(str(item) for item in catalog.names())
        except Exception as exc:
            raise _safe_provider_error("Pinecone reranker catalog", exc) from None
        if RERANK_MODEL not in names:
            raise AblationGateError("The precommitted hosted reranker is unavailable")
        return names

    def rerank(
        self,
        *,
        query: str,
        candidate_ids: Sequence[str],
        records: Mapping[str, HeadingChunkRecord],
    ) -> tuple[list[tuple[str, float]], float]:
        if len(candidate_ids) != TOP_K or len(set(candidate_ids)) != TOP_K:
            raise AblationGateError("Reranker requires exactly five unique hybrid candidates")
        if any(chunk_id not in records for chunk_id in candidate_ids):
            raise AblationGateError("Reranker candidate is absent from the frozen registry")
        documents = [
            {"id": chunk_id, "text": records[chunk_id].text}
            for chunk_id in candidate_ids
        ]
        started = time.perf_counter()
        try:
            response = self._pc.inference.rerank(
                model=RERANK_MODEL,
                query=query,
                documents=documents,
                rank_fields=["text"],
                top_n=TOP_K,
                return_documents=False,
                parameters={"truncate": "END"},
            )
        except Exception as exc:
            raise _safe_provider_error("Pinecone hosted reranking", exc) from None
        latency = time.perf_counter() - started
        data = list(_field(response, "data", []) or [])
        if len(data) != TOP_K:
            raise AblationGateError("Pinecone reranker returned the wrong result count")
        indexes = [int(_field(item, "index", -1)) for item in data]
        if sorted(indexes) != list(range(TOP_K)):
            raise AblationGateError("Pinecone reranker returned an invalid index permutation")
        model = str(_field(response, "model", "") or "")
        if model != RERANK_MODEL:
            raise AblationGateError("Pinecone reranker response used an unexpected model")
        ranked = [
            (candidate_ids[index], _finite_score(_field(item, "score")))
            for item, index in zip(data, indexes, strict=True)
        ]
        prior_rank = {chunk_id: rank for rank, chunk_id in enumerate(candidate_ids, start=1)}
        ranked.sort(key=lambda item: (-item[1], prior_rank[item[0]], item[0]))
        usage = _field(response, "usage", {}) or {}
        units = int(_field(usage, "rerank_units", 0) or 0)
        if units < 0:
            raise AblationGateError("Pinecone reranker returned invalid usage")
        self.request_count += 1
        self.rerank_units += units
        return ranked, latency


def _latency_summary(values: Sequence[float]) -> dict[str, int | float]:
    if not values or any(not math.isfinite(value) or value < 0 for value in values):
        raise AblationGateError("Latency population is empty or invalid")
    ordered = sorted(float(value) for value in values)
    p95 = ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)]
    return {
        "n": len(ordered),
        "median_seconds": round(statistics.median(ordered), 6),
        "p95_seconds": round(p95, 6),
        "max_seconds": round(max(ordered), 6),
    }


def _rank_record(chunk: HeadingChunkRecord, rank: int, score: float) -> dict[str, Any]:
    return {
        "rank": rank,
        "chunk_id": chunk.chunk_id,
        "retrieved_notice_code": str(chunk.metadata["notice_code"]),
        "attributed_heading_paths": [list(chunk.metadata["heading_path"])],
        "score": score,
    }


def _score_ids(
    question: Mapping[str, Any],
    ranked: Sequence[tuple[str, float]],
    records: Mapping[str, HeadingChunkRecord],
) -> tuple[SectionScore, list[dict[str, Any]]]:
    if len(ranked) != TOP_K or len({item[0] for item in ranked}) != TOP_K:
        raise AblationGateError("Every variant must return exactly five unique chunks")
    rank_records = [
        _rank_record(records[chunk_id], rank, score)
        for rank, (chunk_id, score) in enumerate(ranked, start=1)
    ]
    expected_code = str(question["expected_notice_code"])
    if any(item["retrieved_notice_code"] != expected_code for item in rank_records):
        raise AblationGateError("A retrieval variant escaped the exact notice-code restriction")
    score = score_section_ranking(
        expected_code,
        question["expected_heading_path"],
        rank_records,
    )
    return score, rank_records


def _validate_frozen_inputs(project_root: Path) -> dict[str, Any]:
    frozen = verify_phase51_frozen_inputs(project_root)
    expected = {
        REGISTRY_PATH: EXPECTED_REGISTRY_SHA256,
        BENCHMARK_PATH: EXPECTED_BENCHMARK_SHA256,
        DENSE_RESULTS_PATH: EXPECTED_DENSE_RESULTS_SHA256,
        FINAL_CONFIG_PATH: EXPECTED_FINAL_CONFIG_SHA256,
    }
    failures = [
        path.as_posix()
        for path, digest in expected.items()
        if not (project_root / path).is_file() or sha256_file(project_root / path) != digest
    ]
    if failures:
        raise AblationGateError("Retrieval ablation frozen-input gate failed: " + ", ".join(failures))
    return frozen


def _load_dense_traces(
    path: Path,
    questions: Sequence[dict[str, Any]],
    records: Mapping[str, HeadingChunkRecord],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    traces = payload.get("queries")
    if not isinstance(traces, list) or len(traces) != 15:
        raise AblationGateError("Frozen dense trace must contain exactly 15 questions")
    by_id = {str(trace.get("id")): trace for trace in traces if isinstance(trace, dict)}
    if set(by_id) != {question["id"] for question in questions}:
        raise AblationGateError("Frozen dense trace IDs differ from the benchmark")
    dense_scores: list[SectionScore] = []
    for question in questions:
        trace = by_id[question["id"]]
        for field in ("question", "expected_notice_code", "expected_heading_path"):
            if trace.get(field) != question[field]:
                raise AblationGateError(f"Frozen dense trace changed {field} for {question['id']}")
        ranks = trace.get("ranks")
        if not isinstance(ranks, list) or len(ranks) != TOP_K:
            raise AblationGateError("Frozen dense trace does not contain five ranks")
        ids = [str(item.get("chunk_id", "")) for item in ranks]
        if len(set(ids)) != TOP_K or any(chunk_id not in records for chunk_id in ids):
            raise AblationGateError("Frozen dense trace contains unknown or duplicate IDs")
        ranked: list[tuple[str, float]] = []
        for item, chunk_id in zip(ranks, ids, strict=True):
            chunk = records[chunk_id]
            if item.get("retrieved_notice_code") != chunk.metadata["notice_code"]:
                raise AblationGateError("Frozen dense metadata differs from the registry")
            if item.get("attributed_heading_paths") != [list(chunk.metadata["heading_path"])]:
                raise AblationGateError("Frozen dense heading path differs from the registry")
            ranked.append((chunk_id, _finite_score(item.get("similarity_score"))))
        recomputed, _ = _score_ids(question, ranked, records)
        stored = trace.get("section_score", {})
        if (
            recomputed.precision_at_1 != stored.get("precision_at_1")
            or not math.isclose(recomputed.reciprocal_rank, float(stored.get("reciprocal_rank", -1)))
            or recomputed.hit_at_5 != stored.get("hit_at_5")
            or recomputed.first_correct_rank != stored.get("first_correct_rank")
        ):
            raise AblationGateError("Frozen dense score does not recompute")
        latency = trace.get("query_latency_seconds")
        if isinstance(latency, bool) or not isinstance(latency, (int, float)) or latency < 0:
            raise AblationGateError("Frozen dense latency is invalid")
        dense_scores.append(recomputed)
    aggregate = aggregate_section_scores(dense_scores)
    if (
        aggregate["correct_at_1"] != 14
        or not math.isclose(float(aggregate["section_mrr"]), 0.9555555555555555)
        or aggregate["hit_at_5_count"] != 15
    ):
        raise AblationGateError("Frozen dense metrics differ from the approved baseline")
    return by_id, payload


def _score_dict(score: SectionScore) -> dict[str, int | float | None]:
    return {
        "precision_at_1": score.precision_at_1,
        "reciprocal_rank": score.reciprocal_rank,
        "hit_at_5": score.hit_at_5,
        "first_correct_rank": score.first_correct_rank,
    }


def _aggregate_variant(rows: Sequence[dict[str, Any]], variant: str) -> dict[str, Any]:
    scores = [
        SectionScore(
            precision_at_1=int(row[variant]["score"]["precision_at_1"]),
            reciprocal_rank=float(row[variant]["score"]["reciprocal_rank"]),
            hit_at_5=int(row[variant]["score"]["hit_at_5"]),
            first_correct_rank=row[variant]["score"]["first_correct_rank"],
        )
        for row in rows
    ]
    metrics = aggregate_section_scores(scores)
    latency = _latency_summary([float(row[variant]["latency_seconds"]) for row in rows])
    failures = [row["question_id"] for row in rows if not row[variant]["score"]["precision_at_1"]]
    return {"metrics": metrics, "latency": latency, "p1_failures": failures}


def _decide(aggregates: Mapping[str, dict[str, Any]], rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    baseline_p1 = float(aggregates["dense"]["metrics"]["section_precision_at_1"])
    dense_success_ids = {
        row["question_id"] for row in rows if row["dense"]["score"]["precision_at_1"]
    }
    candidates: list[dict[str, Any]] = []
    for complexity, variant in enumerate(VARIANTS[1:], start=1):
        metrics = aggregates[variant]["metrics"]
        p1 = float(metrics["section_precision_at_1"])
        delta_pp = 100.0 * (p1 - baseline_p1)
        variant_success_ids = {
            row["question_id"] for row in rows if row[variant]["score"]["precision_at_1"]
        }
        new_regressions = sorted(dense_success_ids - variant_success_ids)
        fixes_remaining = "S06" in variant_success_ids
        quality_eligible = delta_pp >= 5.0 - 1e-12 or (fixes_remaining and not new_regressions)
        latency_p95 = float(aggregates[variant]["latency"]["p95_seconds"])
        latency_reasonable = latency_p95 <= LATENCY_REASONABLE_P95_SECONDS
        candidates.append(
            {
                "variant": variant,
                "complexity_order": complexity,
                "absolute_p1_delta_percentage_points": round(delta_pp, 6),
                "fixes_dense_failure_s06": fixes_remaining,
                "new_p1_regressions": new_regressions,
                "quality_eligible": quality_eligible,
                "latency_reasonable": latency_reasonable,
                "selection_eligible": quality_eligible and latency_reasonable,
            }
        )
    eligible = [candidate for candidate in candidates if candidate["selection_eligible"]]
    eligible.sort(key=lambda item: (int(item["complexity_order"]), item["variant"]))
    selected = eligible[0]["variant"] if eligible else "dense"
    config_change_recommended = selected != "dense"
    recommendation = (
        f"Recommend {VARIANT_LABELS[selected]} for a separately approved production change; "
        "the frozen production configuration was not modified."
        if config_change_recommended
        else "Retain the simpler heading-aware dense retriever; no tested variant met the quality-and-latency rule."
    )
    return {
        "selected_recommendation": selected,
        "config_change_recommended": config_change_recommended,
        "production_config_modified": False,
        "latency_reasonable_p95_seconds": LATENCY_REASONABLE_P95_SECONDS,
        "candidate_decisions": candidates,
        "recommendation": recommendation,
    }


def _csv_payload(rows: Sequence[dict[str, Any]]) -> str:
    fields = ["question_id", "notice_code", "expected_heading_path"]
    for variant in VARIANTS:
        fields.extend(
            [
                f"{variant}_rank",
                f"{variant}_p1",
                f"{variant}_rr",
                f"{variant}_hit5",
                f"{variant}_latency_seconds",
                f"{variant}_top1_chunk_id",
                f"{variant}_top1_heading_path",
            ]
        )
    fields.extend(["hybrid_reranker_top1_score", "reranker_model"])
    from io import StringIO

    buffer = StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        output: dict[str, Any] = {
            "question_id": row["question_id"],
            "notice_code": row["notice_code"],
            "expected_heading_path": json.dumps(row["expected_heading_path"], ensure_ascii=False),
        }
        for variant in VARIANTS:
            result = row[variant]
            output.update(
                {
                    f"{variant}_rank": result["score"]["first_correct_rank"],
                    f"{variant}_p1": result["score"]["precision_at_1"],
                    f"{variant}_rr": f'{float(result["score"]["reciprocal_rank"]):.12f}',
                    f"{variant}_hit5": result["score"]["hit_at_5"],
                    f"{variant}_latency_seconds": f'{float(result["latency_seconds"]):.6f}',
                    f"{variant}_top1_chunk_id": result["ranks"][0]["chunk_id"],
                    f"{variant}_top1_heading_path": json.dumps(
                        result["ranks"][0]["attributed_heading_paths"][0], ensure_ascii=False
                    ),
                }
            )
        output["hybrid_reranker_top1_score"] = f'{float(row["hybrid_reranker"]["ranks"][0]["score"]):.12f}'
        output["reranker_model"] = RERANK_MODEL
        writer.writerow(output)
    return buffer.getvalue()


def _percent(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def _markdown_payload(
    *,
    generated_at: str,
    aggregates: Mapping[str, dict[str, Any]],
    rows: Sequence[dict[str, Any]],
    decision: Mapping[str, Any],
    capability: Mapping[str, Any],
    integrity: Mapping[str, Any],
    rerank_units: int,
) -> str:
    lines = [
        "# NoticeLens Retrieval Ablation",
        "",
        f"Generated: `{generated_at}`",
        "",
        "This is an isolated experiment over frozen artifacts. It does not prove that any retrieval technique is universally better, and it did not alter the production retriever.",
        "",
        "## Frozen protocol",
        "",
        "- Corpus/chunks: the exact 580 `heading_aware_220_40` records from the approved 50-document IRS corpus.",
        "- Benchmark: exact S01-S15 questions; exact notice-code restriction; final `top_k=5`; unchanged full-heading-path scoring.",
        f"- BM25: one global Okapi index over exact prefixed chunk text; NFKC + casefold + Unicode alphanumeric tokenizer; `k1={BM25_K1}`, `b={BM25_B}`; exact-notice candidates only.",
        f"- Hybrid: equal-weight reciprocal-rank fusion of dense top 5 and BM25 top 5 with `k={RRF_K}`; deterministic chunk-ID tie break.",
        f"- Reranking: `{RERANK_MODEL}` reranks exactly the hybrid top 5; no candidate expansion, query rewriting, or per-question tuning.",
        "- Latency: one pass over S01-S15. Dense latency is the frozen Phase 4B Pinecone-query measurement; BM25/fusion and hosted-rerank components were measured now. Hybrid totals are disclosed component sums; common query embedding time is excluded.",
        f"- Study-only latency guardrail: p95 <= {LATENCY_REASONABLE_P95_SECONDS:.1f}s. This was frozen before results and is not a production SLA.",
        "",
        "## Reranker capability",
        "",
        "Nebius was preferred, but its live catalog exposed no reranker and the documented `/v1/rerank` probe returned 404. The existing Pinecone account exposed and successfully probed `bge-reranker-v2-m3`, so it was used as the fallback.",
        "",
        "- Nebius reference: https://docs.tokenfactory.nebius.com/api-reference/inference/rerank-documents",
        "- Pinecone reference: https://sdk.pinecone.io/python/how-to/inference/reranking.html",
        f"- Pinecone rerank calls in measured study: {capability['rerank_request_count']}; rerank units: {rerank_units}.",
        "- Vector index queries/writes/deletes/creates in this runner: 0/0/0/0. Dense ranks were replayed from the frozen Phase 4B trace.",
        "",
        "## Aggregate results",
        "",
        "| Variant | Section P@1 | MRR | Hit@5 | Median retrieval latency | p95 retrieval latency | Remaining P@1 failures |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for variant in VARIANTS:
        result = aggregates[variant]
        metrics = result["metrics"]
        latency = result["latency"]
        failures = ", ".join(result["p1_failures"]) or "None"
        lines.append(
            f"| {VARIANT_LABELS[variant]} | {_percent(float(metrics['section_precision_at_1']))} "
            f"({metrics['correct_at_1']}/{metrics['n']}) | {float(metrics['section_mrr']):.5f} | "
            f"{_percent(float(metrics['section_hit_at_5']))} | {float(latency['median_seconds']):.6f}s | "
            f"{float(latency['p95_seconds']):.6f}s | {failures} |"
        )
    lines.extend(
        [
            "",
            "## Paired per-question comparison",
            "",
            "Ranks are the first exact full-heading-path match within the final five.",
            "",
            "| ID | Notice | Dense | BM25 | Hybrid | Hybrid + reranker |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        ranks = [row[variant]["score"]["first_correct_rank"] for variant in VARIANTS]
        rendered = ["miss" if rank is None else str(rank) for rank in ranks]
        lines.append(
            f"| {row['question_id']} | {row['notice_code']} | {rendered[0]} | {rendered[1]} | {rendered[2]} | {rendered[3]} |"
        )
    lines.extend(["", "## Decision", "", str(decision["recommendation"]), ""])
    for candidate in decision["candidate_decisions"]:
        regressions = ", ".join(candidate["new_p1_regressions"]) or "none"
        lines.append(
            f"- {VARIANT_LABELS[candidate['variant']]}: P@1 delta {candidate['absolute_p1_delta_percentage_points']:+.2f} points; "
            f"fixes S06={candidate['fixes_dense_failure_s06']}; new rank-1 regressions={regressions}; "
            f"quality eligible={candidate['quality_eligible']}; latency reasonable={candidate['latency_reasonable']}."
        )
    lines.extend(
        [
            "",
            f"`reports/final_retrieval_config.json` remained byte-identical at `{integrity['final_config_sha256']}`. Even if an experimental candidate qualifies, adoption requires a separate approval.",
            "",
            "## LlamaIndex time-box estimate",
            "",
            "A small LlamaIndex retriever comparison is conditionally feasible in about 40-60 minutes after the main app: install the minimal core/adapters, wrap the frozen registry or existing namespace without re-embedding, run the same 15 questions, and report. Stop at 60 minutes if adapter/version friction appears; it should not delay Streamlit.",
            "",
            "## Integrity",
            "",
            "- Frozen Phase 1-4B composite plus pinned Phase 5/5.1 inputs: "
            f"PASS (`{integrity['phase1_to_4b_composite_sha256']}`).",
            f"- Heading registry SHA-256: `{integrity['registry_sha256']}`.",
            f"- Benchmark SHA-256: `{integrity['benchmark_sha256']}`.",
            f"- Frozen dense result SHA-256: `{integrity['dense_results_sha256']}`.",
            "- Benchmark questions, chunks, embeddings, Pinecone namespaces, production source, and production configuration were not changed.",
            "",
        ]
    )
    return "\n".join(lines)


def run_retrieval_ablation(
    *,
    project_root: Path,
    secret_path: Path = DEFAULT_SECRETS_PATH,
    reranker: PineconeHostedReranker | None = None,
    write_reports: bool = True,
) -> dict[str, Any]:
    project_root = project_root.resolve()
    frozen_before = _validate_frozen_inputs(project_root)
    config_hash_before = sha256_file(project_root / FINAL_CONFIG_PATH)
    questions = load_section_questions(project_root / BENCHMARK_PATH)
    records_list = load_heading_registry(project_root / REGISTRY_PATH)
    if len(records_list) != 580:
        raise AblationGateError("Heading registry count differs from the frozen experiment")
    records = {record.chunk_id: record for record in records_list}
    bm25 = BM25Corpus.build(records_list)
    dense_by_id, _dense_payload = _load_dense_traces(
        project_root / DENSE_RESULTS_PATH, questions, records
    )

    if reranker is None:
        config = load_phase3_config(secret_path=secret_path, project_root=project_root)
        reranker = PineconeHostedReranker(api_key=config.pinecone_api_key)
    catalog_names = reranker.require_model()

    rows: list[dict[str, Any]] = []
    for question in questions:
        dense_trace = dense_by_id[question["id"]]
        dense_ranked = [
            (str(item["chunk_id"]), _finite_score(item["similarity_score"]))
            for item in dense_trace["ranks"]
        ]
        dense_score, dense_rank_records = _score_ids(question, dense_ranked, records)
        dense_latency = round(float(dense_trace["query_latency_seconds"]), 6)

        started = time.perf_counter()
        bm25_ranked = bm25.rank(question["question"], question["expected_notice_code"])
        bm25_latency = round(time.perf_counter() - started, 6)
        bm25_score, bm25_rank_records = _score_ids(question, bm25_ranked, records)

        started = time.perf_counter()
        hybrid_ranked = reciprocal_rank_fusion(
            [item[0] for item in dense_ranked], [item[0] for item in bm25_ranked]
        )
        fusion_latency = round(time.perf_counter() - started, 6)
        hybrid_score, hybrid_rank_records = _score_ids(question, hybrid_ranked, records)
        hybrid_latency = round(dense_latency + bm25_latency + fusion_latency, 6)

        reranked, rerank_latency_raw = reranker.rerank(
            query=question["question"],
            candidate_ids=[item[0] for item in hybrid_ranked],
            records=records,
        )
        rerank_latency = round(rerank_latency_raw, 6)
        reranker_score, reranker_rank_records = _score_ids(question, reranked, records)
        reranker_total_latency = round(hybrid_latency + rerank_latency, 6)

        rows.append(
            {
                "question_id": question["id"],
                "question": question["question"],
                "notice_code": question["expected_notice_code"],
                "expected_heading_path": question["expected_heading_path"],
                "dense": {
                    "score": _score_dict(dense_score),
                    "latency_seconds": dense_latency,
                    "ranks": dense_rank_records,
                },
                "bm25": {
                    "score": _score_dict(bm25_score),
                    "latency_seconds": bm25_latency,
                    "ranks": bm25_rank_records,
                },
                "hybrid": {
                    "score": _score_dict(hybrid_score),
                    "latency_seconds": hybrid_latency,
                    "ranks": hybrid_rank_records,
                },
                "hybrid_reranker": {
                    "score": _score_dict(reranker_score),
                    "latency_seconds": reranker_total_latency,
                    "rerank_component_latency_seconds": rerank_latency,
                    "ranks": reranker_rank_records,
                },
            }
        )

    if reranker.request_count != 15:
        raise AblationGateError("Reranker request count differs from the benchmark denominator")
    aggregates = {variant: _aggregate_variant(rows, variant) for variant in VARIANTS}
    decision = _decide(aggregates, rows)

    frozen_after = _validate_frozen_inputs(project_root)
    config_hash_after = sha256_file(project_root / FINAL_CONFIG_PATH)
    if config_hash_before != config_hash_after or config_hash_after != EXPECTED_FINAL_CONFIG_SHA256:
        raise AblationGateError("The frozen production retrieval configuration changed")

    generated_at = _utc_now()
    capability = {
        "nebius_preferred": True,
        "nebius_reranker_available": False,
        "nebius_catalog_model_count": 30,
        "nebius_rerank_probe_http_status": 404,
        "pinecone_catalog_models": catalog_names,
        "pinecone_selected_reranker": RERANK_MODEL,
        "rerank_request_count": reranker.request_count,
    }
    integrity = {
        "phase1_to_4b_composite_sha256": frozen_after[
            "approved_phase1_to_4b_composite_sha256"
        ],
        "registry_sha256": sha256_file(project_root / REGISTRY_PATH),
        "benchmark_sha256": sha256_file(project_root / BENCHMARK_PATH),
        "dense_results_sha256": sha256_file(project_root / DENSE_RESULTS_PATH),
        "final_config_sha256": config_hash_after,
        "frozen_before_after_equal": frozen_before == frozen_after,
    }
    csv_payload = _csv_payload(rows)
    markdown_payload = _markdown_payload(
        generated_at=generated_at,
        aggregates=aggregates,
        rows=rows,
        decision=decision,
        capability=capability,
        integrity=integrity,
        rerank_units=reranker.rerank_units,
    )
    if write_reports:
        config = load_phase3_config(secret_path=secret_path, project_root=project_root)
        for secret in (config.nebius_api_key, config.pinecone_api_key):
            if secret and (secret in csv_payload or secret in markdown_payload):
                raise AblationGateError("A credential value was detected before report writing")
        _atomic_write_text(project_root / CSV_REPORT_PATH, csv_payload)
        _atomic_write_text(project_root / MARKDOWN_REPORT_PATH, markdown_payload)
    return {
        "generated_at_utc": generated_at,
        "aggregates": aggregates,
        "rows": rows,
        "decision": decision,
        "capability": capability,
        "integrity": integrity,
        "rerank_units": reranker.rerank_units,
        "reports_written": write_reports,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the isolated NoticeLens retrieval ablation")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--secret-file", type=Path, default=DEFAULT_SECRETS_PATH)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = run_retrieval_ablation(
            project_root=args.project_root,
            secret_path=args.secret_file,
            write_reports=True,
        )
    except Exception as exc:
        if isinstance(exc, AblationGateError):
            print(str(exc))
        else:
            print(f"Retrieval ablation failed ({type(exc).__name__}); no credentials were logged")
        return 1
    print("Retrieval ablation complete")
    for variant in VARIANTS:
        metrics = result["aggregates"][variant]["metrics"]
        latency = result["aggregates"][variant]["latency"]
        print(
            f"{VARIANT_LABELS[variant]}: P@1={metrics['section_precision_at_1']:.4f}, "
            f"MRR={metrics['section_mrr']:.5f}, Hit@5={metrics['section_hit_at_5']:.4f}, "
            f"p95={latency['p95_seconds']:.6f}s"
        )
    print(result["decision"]["recommendation"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
