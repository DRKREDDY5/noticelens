"""Phase 3: frozen fixed-chunk dense retrieval baseline."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import re
import sys
import time
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .chunking import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    CHUNK_STRATEGY,
    TOKENIZER_NAME,
    TOKENIZER_REVISION,
    ChunkRecord,
    ChunkingGateError,
    atomic_write_json,
    atomic_write_text,
    build_fixed_chunks,
    load_qwen_tokenizer,
    sha256_file,
)
from .config import ConfigurationError, Phase3Config, load_phase3_config
from .evaluation import (
    EvaluationGateError,
    aggregate_scores,
    build_notice_alias_registry,
    group_metric_scores,
    heading_text_is_visible,
    load_golden_questions,
    questions_for_retrieval,
    read_manifest_rows,
    resolve_notice_alias,
    score_ranking,
)
from .providers import (
    EMBEDDING_DIMENSION,
    EMBEDDING_MODEL,
    INDEX_CLOUD,
    INDEX_METRIC,
    INDEX_NAME,
    INDEX_REGION,
    NAMESPACE,
    PINECONE_METADATA_KEYS,
    NebiusEmbeddings,
    PineconeBaselineStore,
    ProviderGateError,
    finite_score,
    latency_summary,
    match_field,
)


TOP_K = 5
DEFAULT_BATCH_SIZE = 8
REGISTRY_RELATIVE_PATH = Path("data/derived/phase3/fixed_220_40_chunks.jsonl")
CHUNK_AUDIT_RELATIVE_PATH = Path("reports/phase3_chunk_audit.json")
INDEX_STATS_RELATIVE_PATH = Path("reports/phase3_indexing_stats.json")
RESULTS_RELATIVE_PATH = Path("reports/phase3_baseline_results.json")
SUMMARY_RELATIVE_PATH = Path("reports/phase3_baseline_summary.csv")
FAILURES_RELATIVE_PATH = Path("reports/phase3_failure_analysis.md")


class Phase3GateError(RuntimeError):
    """Raised when an end-to-end Phase 3 invariant fails."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _tree_digest(directory: Path) -> tuple[str, int]:
    records: list[str] = []
    for path in sorted((path for path in directory.rglob("*") if path.is_file()), key=lambda item: item.as_posix()):
        relative = path.relative_to(directory).as_posix()
        records.append(f"{relative}|{sha256_file(path)}")
    payload = "\n".join(records).encode("utf-8")
    return hashlib.sha256(payload).hexdigest(), len(records)


def frozen_input_snapshot(project_root: Path) -> dict[str, Any]:
    processed_hash, processed_count = _tree_digest(project_root / "data" / "processed" / "guidance")
    paths = {
        "corpus_manifest": project_root / "data" / "corpus_manifest.csv",
        "sample_notice_manifest": project_root / "data" / "sample_notice_manifest.csv",
        "golden_questions": project_root / "eval" / "golden_questions.json",
        "phase2_evaluation_manifest": project_root / "reports" / "phase2_eval_manifest.csv",
        "evaluation_plan": project_root / "reports" / "evaluation_plan.md",
    }
    return {
        **{
            name: {
                "path": path.relative_to(project_root).as_posix(),
                "sha256": sha256_file(path),
            }
            for name, path in paths.items()
        },
        "processed_guidance_inventory": {
            "path": "data/processed/guidance",
            "file_count": processed_count,
            "tree_sha256": processed_hash,
        },
    }


def _assert_frozen_inputs_unchanged(project_root: Path, before: dict[str, Any]) -> None:
    after = frozen_input_snapshot(project_root)
    if after != before:
        raise Phase3GateError("A frozen Phase 1 or Phase 2 input changed during the Phase 3 run")


def _batch(values: Sequence[Any], batch_size: int) -> list[Sequence[Any]]:
    return [values[offset : offset + batch_size] for offset in range(0, len(values), batch_size)]


def _safe_metadata_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    try:
        return dict(value)
    except (TypeError, ValueError):
        raise Phase3GateError("Pinecone returned malformed metadata") from None


def _text_preview(text: str, limit: int = 360) -> str:
    return re.sub(r"\s+", " ", text).strip()[:limit]


def _write_indexing_stats(project_root: Path, stats: dict[str, Any]) -> None:
    atomic_write_json(project_root / INDEX_STATS_RELATIVE_PATH, stats)


def index_chunks(
    *,
    project_root: Path,
    config: Phase3Config,
    chunks: list[ChunkRecord],
    frozen_inputs: dict[str, Any],
    batch_size: int,
) -> tuple[NebiusEmbeddings, PineconeBaselineStore, dict[str, Any]]:
    """Probe the dimension, verify/create the index, then embed and upsert."""

    report_path = project_root / INDEX_STATS_RELATIVE_PATH
    stats: dict[str, Any] = {
        "schema_version": "1.0",
        "generated_at_utc": _utc_now(),
        "status": "started",
        "public_configuration": config.public_summary(),
        "embedding": {
            "provider": "Nebius Token Factory",
            "model": EMBEDDING_MODEL,
            "expected_dimension": EMBEDDING_DIMENSION,
            "batch_size": batch_size,
        },
        "pinecone": {
            "index_name": INDEX_NAME,
            "namespace": NAMESPACE,
            "expected_dimension": EMBEDDING_DIMENSION,
            "expected_metric": INDEX_METRIC,
            "creation_cloud": INDEX_CLOUD,
            "creation_region": INDEX_REGION,
        },
        "inputs": {
            "frozen": frozen_inputs,
            "chunk_registry_path": REGISTRY_RELATIVE_PATH.as_posix(),
            "chunk_registry_sha256": sha256_file(project_root / REGISTRY_RELATIVE_PATH),
            "local_expected_chunk_count": len(chunks),
        },
        "quality_gate_passed": False,
    }
    stage = "dimension_probe"
    try:
        embedder = NebiusEmbeddings(
            api_key=config.nebius_api_key,
            base_url=config.nebius_base_url,
            batch_size=batch_size,
        )
        observed_dimension, probe_latency = embedder.dimension_probe()
        stats["embedding"]["dimension_probe"] = {
            "request_success": True,
            "observed_dimension": observed_dimension,
            "latency_seconds": round(probe_latency, 6),
            "request_count": 1,
        }

        stage = "pinecone_index_compatibility"
        store = PineconeBaselineStore(api_key=config.pinecone_api_key)
        state = store.ensure_compatible_index()
        stats["pinecone"].update(
            {
                "existed_or_created": state.existed_or_created,
                "ready": state.ready,
                "observed_dimension": state.dimension,
                "observed_metric": state.metric,
                "observed_cloud": state.cloud,
                "observed_region": state.region,
            }
        )

        expected_ids = {chunk.chunk_id for chunk in chunks}
        if len(expected_ids) != len(chunks):
            raise Phase3GateError("Local chunk IDs are not unique")
        stage = "pinecone_namespace_preflight"
        stats["pinecone"].update(store.assert_safe_existing_namespace(expected_ids))

        stage = "embedding_and_upsert"
        embedded_count = 0
        upserted_count = 0
        upsert_latencies: list[float] = []
        embedding_latency_start = len(embedder.request_latencies)
        batches = _batch(chunks, batch_size)
        for batch_number, chunk_batch_raw in enumerate(batches, start=1):
            chunk_batch = list(chunk_batch_raw)
            vectors = embedder.embed_documents([chunk.text for chunk in chunk_batch])
            embedded_count += len(vectors)
            started = time.perf_counter()
            upserted_count += store.upsert_batch(chunk_batch, vectors)
            upsert_latencies.append(time.perf_counter() - started)
            if batch_number == 1 or batch_number % 10 == 0 or batch_number == len(batches):
                print(f"Phase 3 indexing progress: {upserted_count}/{len(chunks)} chunks")

        stage = "pinecone_namespace_parity"
        pinecone_count, observed_ids = store.wait_for_exact_parity(expected_ids)
        if embedded_count != len(chunks) or upserted_count != len(chunks) or observed_ids != expected_ids:
            raise Phase3GateError("Local/embed/upsert/Pinecone ID parity failed")

        bulk_embedding_latencies = embedder.request_latencies[embedding_latency_start:]
        stats["embedding"].update(
            {
                "embedded_chunk_count": embedded_count,
                "batch_count": len(batches),
                "bulk_request_latency": latency_summary(bulk_embedding_latencies),
            }
        )
        stats["pinecone"].update(
            {
                "upserted_chunk_count": upserted_count,
                "namespace_vector_count": pinecone_count,
                "exact_id_set_parity": True,
                "upsert_latency": latency_summary(upsert_latencies),
            }
        )
        stats["status"] = "completed"
        stats["completed_at_utc"] = _utc_now()
        stats["quality_gate_passed"] = True
        _write_indexing_stats(project_root, stats)
        return embedder, store, stats
    except (ProviderGateError, Phase3GateError, EvaluationGateError) as exc:
        stats["status"] = "failed"
        stats["failed_stage"] = stage
        stats["failure"] = str(exc)
        stats["completed_at_utc"] = _utc_now()
        _write_indexing_stats(project_root, stats)
        raise


def _validate_and_serialize_matches(
    *,
    matches: list[Any],
    chunks_by_id: dict[str, ChunkRecord],
) -> list[dict[str, Any]]:
    retrievals: list[dict[str, Any]] = []
    observed_ids: set[str] = set()
    for rank, match in enumerate(matches, start=1):
        chunk_id = str(match_field(match, "id", ""))
        if not chunk_id or chunk_id in observed_ids:
            raise Phase3GateError("Pinecone query returned a blank or duplicate chunk ID")
        observed_ids.add(chunk_id)
        if chunk_id not in chunks_by_id:
            raise Phase3GateError("Pinecone query returned an unknown or stale chunk ID")
        chunk = chunks_by_id[chunk_id]
        metadata = _safe_metadata_dict(match_field(match, "metadata", {}))
        if set(metadata) != set(PINECONE_METADATA_KEYS):
            raise Phase3GateError(f"Pinecone metadata schema mismatch for {chunk_id}")
        for key in PINECONE_METADATA_KEYS:
            if metadata[key] != chunk.metadata[key]:
                raise Phase3GateError(f"Pinecone/local metadata mismatch for {chunk_id} field {key}")
        score = finite_score(match_field(match, "score"))
        retrievals.append(
            {
                "rank": rank,
                "chunk_id": chunk_id,
                "retrieved_notice_code": metadata["notice_code"],
                "doc_id": metadata["doc_id"],
                "title": metadata["title"],
                "similarity_score": score,
                "text_preview": _text_preview(chunk.text),
            }
        )
    if [item["rank"] for item in retrievals] != list(range(1, TOP_K + 1)):
        raise Phase3GateError("Retrieval ranks must be exactly 1 through 5")
    return retrievals


def _confusable_rank1_code(
    question: dict[str, Any],
    rank1_doc_id: str,
    alias_registry: dict[str, str],
) -> str | None:
    for notice_code in question["confusable_with"]:
        if resolve_notice_alias(notice_code, alias_registry) == rank1_doc_id:
            return notice_code
    return None


def evaluate_baseline(
    *,
    project_root: Path,
    embedder: NebiusEmbeddings,
    store: PineconeBaselineStore,
    chunks: list[ChunkRecord],
    frozen_inputs: dict[str, Any],
    indexing_stats: dict[str, Any],
    batch_size: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    golden_path = project_root / "eval" / "golden_questions.json"
    manifest_path = project_root / "data" / "corpus_manifest.csv"
    questions = load_golden_questions(golden_path, manifest_path)
    retrieval_questions = questions_for_retrieval(questions)
    if any(question["id"].startswith("E") for question in retrieval_questions):
        raise Phase3GateError("A refusal question was incorrectly selected for Phase 3 retrieval")

    manifest_rows = read_manifest_rows(manifest_path)
    manifest_by_doc = {row["doc_id"]: row for row in manifest_rows}
    alias_registry = build_notice_alias_registry(manifest_rows)
    chunks_by_id = {chunk.chunk_id: chunk for chunk in chunks}
    if len(chunks_by_id) != len(chunks):
        raise Phase3GateError("Local registry has duplicate chunk IDs")

    # No prefixes, code normalization, filters, or rewrites: the exact frozen
    # question strings are the only inputs sent to the embedding endpoint.
    exact_questions = [question["question"] for question in retrieval_questions]
    embedding_latency_start = len(embedder.request_latencies)
    question_vectors = embedder.embed_documents(exact_questions)
    query_embedding_latencies = embedder.request_latencies[embedding_latency_start:]
    if len(question_vectors) != len(retrieval_questions):
        raise Phase3GateError("Question embedding count mismatch")

    query_results: list[dict[str, Any]] = []
    query_latencies: list[float] = []
    for ordinal, (question, vector) in enumerate(zip(retrieval_questions, question_vectors, strict=True), start=1):
        started = time.perf_counter()
        matches = store.query(vector, top_k=TOP_K)
        query_latencies.append(time.perf_counter() - started)
        retrievals = _validate_and_serialize_matches(matches=matches, chunks_by_id=chunks_by_id)
        ranked_doc_ids = [item["doc_id"] for item in retrievals]
        score = score_ranking(question["expected_doc_id"], ranked_doc_ids)
        rank1_doc = ranked_doc_ids[0]
        expected_family = manifest_by_doc[question["expected_doc_id"]]["notice_family"]
        rank1_family = manifest_by_doc[rank1_doc]["notice_family"]
        declared_confusable = _confusable_rank1_code(question, rank1_doc, alias_registry)
        query_result: dict[str, Any] = {
            "question_id": question["id"],
            "category": question["category"],
            "language_style": question["language_style"],
            "question": question["question"],
            "question_sha256": hashlib.sha256(question["question"].encode("utf-8")).hexdigest(),
            "expected_notice_code": question["expected_notice_code"],
            "expected_doc_id": question["expected_doc_id"],
            "confusable_with": question["confusable_with"],
            "retrievals": retrievals,
            "precision_at_1": score.precision_at_1,
            "reciprocal_rank": score.reciprocal_rank,
            "hit_at_5": score.hit_at_5,
            "first_expected_doc_rank": score.first_expected_rank,
            "declared_confusable_rank1": declared_confusable,
            "procedurally_wrong_neighbor_candidate": bool(
                not score.precision_at_1
                and (declared_confusable is not None or rank1_family == expected_family)
            ),
        }
        if question["id"].startswith("D"):
            expected_chunks = [
                chunks_by_id[item["chunk_id"]].text
                for item in retrievals
                if item["doc_id"] == question["expected_doc_id"]
            ]
            query_result["section_observation"] = {
                "expected_heading": question["expected_heading"],
                "formal_heading_attribution_available": False,
                "scored": False,
                "expected_heading_literal_visible_in_retrieved_expected_doc_chunks": any(
                    heading_text_is_visible(question["expected_heading"], text) for text in expected_chunks
                ),
                "limitation": (
                    "Fixed-size chunks carry no heading attribution; literal heading visibility is "
                    "reported only as an observation and is not Section Precision@1."
                ),
            }
        query_results.append(query_result)
        print(f"Phase 3 retrieval progress: {ordinal}/{len(retrieval_questions)} questions")

    metrics = group_metric_scores(query_results)
    section_results = [row for row in query_results if row["question_id"].startswith("D")]
    if len(section_results) != 4:
        raise Phase3GateError("Expected four section-question observations")
    confusable_p1 = float(metrics["by_category"]["C"]["precision_at_1"])
    near_ceiling = confusable_p1 >= 0.95
    result_document: dict[str, Any] = {
        "schema_version": "1.0",
        "generated_at_utc": _utc_now(),
        "run_configuration": {
            "embedding_model": EMBEDDING_MODEL,
            "embedding_provider": "Nebius Token Factory",
            "embedding_dimension": EMBEDDING_DIMENSION,
            "chunk_strategy": CHUNK_STRATEGY,
            "chunk_size_tokens": CHUNK_SIZE,
            "chunk_overlap_tokens": CHUNK_OVERLAP,
            "pinecone_index": INDEX_NAME,
            "pinecone_namespace": NAMESPACE,
            "similarity_metric": INDEX_METRIC,
            "top_k": TOP_K,
            "query_policy": "exact_frozen_question_text_only",
            "metadata_filters_used": False,
            "bm25_used": False,
            "hybrid_used": False,
            "reranking_used": False,
            "query_rewriting_used": False,
            "generation_used": False,
        },
        "inputs": frozen_inputs,
        "indexing_stats": {
            "path": INDEX_STATS_RELATIVE_PATH.as_posix(),
            "sha256": sha256_file(project_root / INDEX_STATS_RELATIVE_PATH),
            "local_chunk_count": len(chunks),
            "pinecone_vector_count": indexing_stats["pinecone"]["namespace_vector_count"],
        },
        "evaluation_scope": {
            "queried_categories": ["A", "B", "C", "D"],
            "metric_categories": ["A", "B", "C"],
            "excluded_categories": ["E"],
            "answerable_queries_run": 26,
            "notice_metric_denominator": 22,
            "section_observation_count": 4,
            "refusal_queries_run": 0,
        },
        "metrics": {"notice_retrieval": metrics},
        "section_observations": {
            "n": 4,
            "scored": False,
            "formal_section_precision_at_1": None,
            "limitation": (
                "The fixed baseline has no reliable heading attribution. Category D is retained as "
                "four observational traces and no Section Precision@1 is fabricated."
            ),
        },
        "null_hypothesis_check": {
            "dense_confusable_family_precision_at_1": confusable_p1,
            "near_ceiling_threshold": 0.95,
            "near_ceiling": near_ceiling,
            "measurable_headroom_to_perfect_precision": max(0.0, 1.0 - confusable_p1),
            "interpretation": (
                "Dense confusable-family retrieval may be near ceiling; later improvement must still "
                "follow the frozen comparison rules."
                if near_ceiling
                else "Dense confusable-family retrieval is below the near-ceiling threshold, leaving measurable headroom."
            ),
        },
        "latency": {
            "question_embedding_requests": latency_summary(query_embedding_latencies),
            "pinecone_query_requests": latency_summary(query_latencies),
            "not_end_to_end_answer_latency": True,
        },
        "queries": query_results,
        "quality_gate_passed": True,
    }
    return result_document, query_results


def _summary_rows(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    selections = [
        ("overall_document", "all", "all", metrics["overall"]),
        ("category", "A", "all", metrics["by_category"]["A"]),
        ("category", "B", "all", metrics["by_category"]["B"]),
        ("category", "C", "all", metrics["by_category"]["C"]),
        ("language_style", "all", "naive", metrics["by_language_style"]["naive"]),
        ("language_style", "all", "expert", metrics["by_language_style"]["expert"]),
    ]
    rows: list[dict[str, Any]] = []
    for scope, category, style, values in selections:
        rows.append(
            {
                "scope": scope,
                "category": category,
                "language_style": style,
                "question_count": values["n"],
                "p1_correct": values["correct_at_1"],
                "precision_at_1": f"{values['precision_at_1']:.6f}",
                "reciprocal_rank_sum": f"{values['reciprocal_rank_sum']:.6f}",
                "mrr": f"{values['mrr']:.6f}",
                "hit5_correct": values["hit_at_5_count"],
                "hit_at_5": f"{values['hit_at_5']:.6f}",
            }
        )
    return rows


def write_summary_csv(project_root: Path, metrics: dict[str, Any]) -> None:
    fields = (
        "scope",
        "category",
        "language_style",
        "question_count",
        "p1_correct",
        "precision_at_1",
        "reciprocal_rank_sum",
        "mrr",
        "hit5_correct",
        "hit_at_5",
    )
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(_summary_rows(metrics))
    atomic_write_text(project_root / SUMMARY_RELATIVE_PATH, buffer.getvalue())


def _percent(value: float) -> str:
    return f"{value * 100:.1f}%"


def _escape_markdown(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def write_failure_analysis(
    project_root: Path,
    results_document: dict[str, Any],
    query_results: list[dict[str, Any]],
) -> None:
    metrics = results_document["metrics"]["notice_retrieval"]
    scored = [row for row in query_results if row["question_id"][0] in "ABC"]
    p1_failures = [row for row in scored if not row["precision_at_1"]]
    hit5_failures = [row for row in scored if not row["hit_at_5"]]
    c_failures = [row for row in p1_failures if row["question_id"].startswith("C")]
    everyday_failures = [row for row in p1_failures if row["question_id"].startswith("B")]
    naive_failures = [row for row in p1_failures if row["language_style"] == "naive"]
    exact_failures = [row for row in p1_failures if row["question_id"].startswith("A")]
    recovered = [row for row in scored if not row["precision_at_1"] and row["hit_at_5"]]
    procedural_candidates = [row for row in p1_failures if row["procedurally_wrong_neighbor_candidate"]]
    section_misses = [
        row
        for row in query_results
        if row["question_id"].startswith("D") and not row["precision_at_1"]
    ]

    priority = ["C03", "C06", "C01", "C05", "B04"]
    ordered = sorted(
        p1_failures,
        key=lambda row: (
            priority.index(row["question_id"]) if row["question_id"] in priority else len(priority),
            row["question_id"],
        ),
    )
    interesting = (ordered + section_misses)[:5]

    lines = [
        "# NoticeLens Phase 3 Baseline Failure Analysis",
        "",
        "This report records the frozen dense baseline exactly as run. It does not tune or repair retrieval.",
        "",
        "## Run contract",
        "",
        f"- Embedding model: `{EMBEDDING_MODEL}` ({EMBEDDING_DIMENSION} dimensions)",
        f"- Chunking: `{CHUNK_STRATEGY}` ({CHUNK_SIZE} tokens, {CHUNK_OVERLAP} overlap)",
        f"- Pinecone: `{INDEX_NAME}` / `{NAMESPACE}` / cosine / top {TOP_K}",
        "- Query input: exact frozen question text only; no filters, rewriting, hybrid search, reranking, or generation",
        "- Scored population: categories A-C only (n=22); category D is observational; category E was not queried",
        "",
        "## Metric summary",
        "",
        "| Scope | n | P@1 | MRR | Hit@5 |",
        "|---|---:|---:|---:|---:|",
        f"| Overall A-C | {metrics['overall']['n']} | {_percent(metrics['overall']['precision_at_1'])} | {metrics['overall']['mrr']:.3f} | {_percent(metrics['overall']['hit_at_5'])} |",
        f"| A exact code | {metrics['by_category']['A']['n']} | {_percent(metrics['by_category']['A']['precision_at_1'])} | {metrics['by_category']['A']['mrr']:.3f} | {_percent(metrics['by_category']['A']['hit_at_5'])} |",
        f"| B everyday language | {metrics['by_category']['B']['n']} | {_percent(metrics['by_category']['B']['precision_at_1'])} | {metrics['by_category']['B']['mrr']:.3f} | {_percent(metrics['by_category']['B']['hit_at_5'])} |",
        f"| C confusable family | {metrics['by_category']['C']['n']} | {_percent(metrics['by_category']['C']['precision_at_1'])} | {metrics['by_category']['C']['mrr']:.3f} | {_percent(metrics['by_category']['C']['hit_at_5'])} |",
        "",
        "## Failure counts",
        "",
        f"- Precision@1 failures: {len(p1_failures)}",
        f"- Hit@5 failures: {len(hit5_failures)}",
        f"- Confusable-family rank-1 mismatches: {len(c_failures)}",
        f"- Everyday-language semantic (category B) rank-1 failures: {len(everyday_failures)}",
        f"- Naive-language rank-1 failures: {len(naive_failures)}",
        f"- Exact-code rank-1 failures: {len(exact_failures)}",
        f"- Expected notice recovered only at ranks 2-5: {len(recovered)}",
        f"- Semantically related/procedurally wrong candidates requiring human review: {len(procedural_candidates)}",
        "",
        "A procedurally wrong-neighbor label is only a review flag when rank 1 is a declared confusable notice or shares the frozen notice family; it is not an automatic legal conclusion.",
        "",
        "## Most interesting retrieval misses (up to five)",
        "",
    ]
    if not interesting:
        lines.append("No scored A-C or observational D rank-1 misses occurred.")
    else:
        lines.extend(
            [
                "| ID | Scope | Expected | Rank 1 | Best expected rank | Declared relationship |",
                "|---|---|---|---|---:|---|",
            ]
        )
        for row in interesting:
            rank1 = row["retrievals"][0]
            scope = "observational D" if row["question_id"].startswith("D") else "scored A-C"
            lines.append(
                f"| {row['question_id']} | {scope} | {_escape_markdown(row['expected_notice_code'])} | "
                f"{_escape_markdown(rank1['retrieved_notice_code'])} | "
                f"{row['first_expected_doc_rank'] or 'absent'} | "
                f"{_escape_markdown(row['declared_confusable_rank1'] or 'none')} |"
            )
    if 0 < len(interesting) < 5:
        miss_label = "miss" if len(interesting) == 1 else "misses"
        lines.extend(
            [
                "",
                f"Only {len(interesting)} retrieval {miss_label} existed across scored A-C and observational D traces; no extra cases were invented.",
            ]
        )

    lines.extend(["", "## Every scored A-C Precision@1 failure", ""])
    if not p1_failures:
        lines.extend(["None.", ""])
    for row in p1_failures:
        rank1 = row["retrievals"][0]
        lines.extend(
            [
                f"### {row['question_id']}",
                "",
                f"- Question: {row['question']}",
                f"- Expected notice: `{row['expected_notice_code']}` (`{row['expected_doc_id']}`)",
                f"- Rank-1 notice: `{rank1['retrieved_notice_code']}` (`{rank1['doc_id']}`)",
                f"- Rank-1 preview: {rank1['text_preview']}",
                f"- Best rank of expected notice: {row['first_expected_doc_rank'] or 'absent from top 5'}",
                f"- Declared confusable relationship: {row['declared_confusable_rank1'] or 'none'}",
                f"- Procedurally wrong-neighbor review flag: {str(row['procedurally_wrong_neighbor_candidate']).lower()}",
                "",
            ]
        )

    null_check = results_document["null_hypothesis_check"]
    lines.extend(
        [
            "## Null-hypothesis check",
            "",
            f"Dense confusable-family Precision@1 is {_percent(null_check['dense_confusable_family_precision_at_1'])}. "
            + null_check["interpretation"],
            "",
            "## Section-question observations",
            "",
            "The four D questions were run, but fixed-size chunks have no reliable heading attribution. Their full top-5 traces are in `phase3_baseline_results.json`; no Section Precision@1 was fabricated.",
            "",
        ]
    )
    section_rows = [row for row in query_results if row["question_id"].startswith("D")]
    for row in section_rows:
        rank1 = row["retrievals"][0]
        expected_rank = row["first_expected_doc_rank"] or "absent from top 5"
        visible = row["section_observation"][
            "expected_heading_literal_visible_in_retrieved_expected_doc_chunks"
        ]
        lines.append(
            f"- {row['question_id']}: expected `{row['expected_notice_code']}` at rank {expected_rank}; "
            f"rank 1 was `{rank1['retrieved_notice_code']}`; expected-heading literal visible in a "
            f"retrieved expected-document chunk: {str(visible).lower()}."
        )
    lines.extend(
        [
            "- These are observations only, not formal Section Precision@1 scores.",
            "",
            "## Deferred future experiments (not implemented)",
            "",
            "Heading-aware chunks, BM25/hybrid retrieval, metadata filters, optional reranking, refusal/abstention routing, and any small agentic workflow remain later-phase candidates. No such improvement was implemented or tuned in this baseline.",
            "",
        ]
    )
    atomic_write_text(project_root / FAILURES_RELATIVE_PATH, "\n".join(lines))


def run_phase3(
    *,
    project_root: Path,
    audit_only: bool = False,
    secret_path: Path | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> dict[str, Any]:
    project_root = project_root.resolve()
    if batch_size <= 0:
        raise Phase3GateError("batch_size must be positive")
    frozen_before = frozen_input_snapshot(project_root)
    if frozen_before["processed_guidance_inventory"]["file_count"] != 50:
        raise Phase3GateError("Frozen processed corpus must contain exactly 50 files")

    print(f"Loading frozen tokenizer {TOKENIZER_NAME} at revision {TOKENIZER_REVISION[:12]}...")
    tokenizer = load_qwen_tokenizer()
    chunks, audit = build_fixed_chunks(project_root=project_root, tokenizer=tokenizer, write_outputs=True)
    print(f"Chunk audit passed: {audit['inputs']['source_document_count']} documents, {len(chunks)} chunks")
    _assert_frozen_inputs_unchanged(project_root, frozen_before)
    if audit_only:
        return {"audit": audit, "chunks": len(chunks), "audit_only": True}

    config = load_phase3_config(
        secret_path=secret_path,
        project_root=project_root,
    )
    embedder, store, indexing_stats = index_chunks(
        project_root=project_root,
        config=config,
        chunks=chunks,
        frozen_inputs=frozen_before,
        batch_size=batch_size,
    )
    results_document, query_results = evaluate_baseline(
        project_root=project_root,
        embedder=embedder,
        store=store,
        chunks=chunks,
        frozen_inputs=frozen_before,
        indexing_stats=indexing_stats,
        batch_size=batch_size,
    )
    metrics = results_document["metrics"]["notice_retrieval"]
    atomic_write_json(project_root / RESULTS_RELATIVE_PATH, results_document)
    write_summary_csv(project_root, metrics)
    write_failure_analysis(project_root, results_document, query_results)
    _assert_frozen_inputs_unchanged(project_root, frozen_before)
    print("Phase 3 baseline reports written; no retrieval improvements were applied")
    return {
        "audit": audit,
        "indexing": indexing_stats,
        "results": results_document,
        "audit_only": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run NoticeLens Phase 3 baseline dense retrieval")
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="Build and validate fixed chunks without loading secrets or calling cloud providers",
    )
    parser.add_argument(
        "--secret-file",
        type=Path,
        default=None,
        help="Explicit external dotenv path (defaults to ~/.noticelens.env)",
    )
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    project_root = Path(__file__).resolve().parents[2]
    try:
        run_phase3(
            project_root=project_root,
            audit_only=args.audit_only,
            secret_path=args.secret_file,
            batch_size=args.batch_size,
        )
    except (
        ConfigurationError,
        ChunkingGateError,
        ProviderGateError,
        EvaluationGateError,
        Phase3GateError,
    ) as exc:
        print(f"Phase 3 stopped safely: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
