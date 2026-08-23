"""Phase 4A: fixed-chunk section attribution and read-only baseline evaluation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import io
import json
import math
import os
import re
import statistics
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pinecone import Pinecone

from .chunking import (
    CHUNK_STRATEGY,
    TOKENIZER_NAME,
    TOKENIZER_REVISION,
    ChunkRecord,
    atomic_write_json,
    atomic_write_text,
    load_chunk_registry,
    nearest_rank_percentile,
    sha256_file,
)
from .config import ConfigurationError, load_phase3_config
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
    ProviderGateError,
    finite_score,
    latency_summary,
)


TOP_K = 5
SECTION_BENCHMARK_FROZEN_SHA256 = "1090c8b41f0b007adfda1eb9882b0237d93416a3ce57857bc4da58a8947aafa8"
SECTION_BENCHMARK_PATH = Path("eval/section_questions.json")
REGISTRY_PATH = Path("data/derived/phase3/fixed_220_40_chunks.jsonl")
MAP_PATH = Path("reports/phase4_fixed_chunk_section_map.json")
RESULTS_PATH = Path("reports/phase4_fixed_section_results.json")
SUMMARY_PATH = Path("reports/phase4_fixed_section_summary.csv")
FAILURES_PATH = Path("reports/phase4_fixed_section_failures.md")

FROZEN_FILE_HASHES = {
    "data/corpus_manifest.csv": "f3c7c2e257a6f1dc17bf8e55ab745702299f2af2a5faf9623d390778045ea1a1",
    "data/sample_notice_manifest.csv": "0a92688fe008ee7fbc10fb5ec4b41733ecad250a47110335cc65e3d72dc99769",
    "eval/golden_questions.json": "a5e12ae768b8d43250fac99198efba35bd2d7f5db640e3aba1e8d6958920f391",
    "reports/phase2_eval_manifest.csv": "f73453102fbbe0cc2e18e214e2926b40c252830b9c4eed92792b2d01cca27ea9",
    "reports/evaluation_plan.md": "5065857f4c568e6915ed186467d0c6e55442df2682ec8e3b9224e4eeb2dce52e",
    "data/derived/phase3/fixed_220_40_chunks.jsonl": "4752049f3c435d79a83b5950d0882ce2ac61bb57a5e210d398b529306a9ab709",
    "reports/phase3_chunk_audit.json": "dfe675d414c5e5e1fa0cdd4f25cb30b72cc7e985e03e383057b1e4cd91b95746",
    "reports/phase3_indexing_stats.json": "329e4c0cfc569d5d13561e16f385ea82a512f8b44c98b6e5cc1806db7a5ffb7b",
    "reports/phase3_baseline_results.json": "1af861aaa152f0cff7c87f7c3496bd825eb9e300b524aa1b5f0a5f4a4104d338",
    "reports/phase3_baseline_summary.csv": "f5f5466b67d9912d37f647e1784a369eded30ca4ba77bad15c2b9e79cb2de5e1",
    "reports/phase3_failure_analysis.md": "a508397e16192c7fb8e447f206018659c300d06cc93e84c6e448b74794255abd",
}
FROZEN_PROCESSED_TREE_SHA256 = "36cf7f0c8a01879062a328dc009d6c6de0f4324f02f95450468b448e391ebac2"
FROZEN_PROCESSED_FILE_COUNT = 50

MEANINGFUL_GAIN_ABSOLUTE_POINTS = 10
NULL_FIXED_PRECISION_THRESHOLD = 0.95
NULL_IMPROVEMENT_LESS_THAN_POINTS = 5
PREDECLARED_HARDEST_RULE = (
    "no Section Hit@5 first; then first correct rank descending; then correct-vs-incorrect "
    "similarity margin ascending; then question ID"
)

BENCHMARK_FIELDS = {
    "id",
    "question",
    "language_style",
    "expected_doc_id",
    "expected_notice_code",
    "expected_heading",
    "expected_heading_path",
    "expected_evidence_excerpt",
    "rationale",
}

_ATX_HEADING = re.compile(r"^(#{1,6})[ \t]+(.+?)\s*$")
_FENCE = re.compile(r"^[ \t]*(```+|~~~+)")
_MARKDOWN_LINK = re.compile(r"!?\[([^\]]+)\]\([^)]*\)")
_HTML_TAG = re.compile(r"<[^>]+>")


class Phase4GateError(RuntimeError):
    """Raised when a Phase 4A freeze, attribution, or retrieval gate fails."""


@dataclass(frozen=True)
class HeadingSection:
    path: tuple[str, ...]
    raw_path: tuple[str, ...]
    levels: tuple[int, ...]
    heading_line: int
    char_start: int
    char_end: int
    token_start: int
    token_end: int


@dataclass(frozen=True)
class SectionScore:
    precision_at_1: int
    reciprocal_rank: float
    hit_at_5: int
    first_correct_rank: int | None


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def load_phase4_tokenizer() -> Any:
    """Load the frozen tokenizer strictly from the existing local cache."""

    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained(
            TOKENIZER_NAME,
            revision=TOKENIZER_REVISION,
            use_fast=False,
            trust_remote_code=False,
            local_files_only=True,
        )
    except Exception as exc:
        raise Phase4GateError(
            f"Frozen tokenizer is unavailable offline ({type(exc).__name__}); no network call was made"
        ) from None


def normalize_exact(value: str) -> str:
    """NFC + surrounding trim + collapsed whitespace; punctuation remains exact."""

    return " ".join(unicodedata.normalize("NFC", value).split())


def visible_markdown(value: str) -> str:
    """Produce visible text for heading/evidence verification only."""

    value = _MARKDOWN_LINK.sub(lambda match: match.group(1), value)
    value = _HTML_TAG.sub(" ", value)
    value = html.unescape(value)
    value = value.replace("`", "").replace("**", "").replace("__", "")
    return normalize_exact(value)


def _tree_digest(directory: Path) -> tuple[str, int]:
    records: list[str] = []
    files = sorted((path for path in directory.rglob("*") if path.is_file()), key=lambda path: path.as_posix())
    for path in files:
        records.append(f"{path.relative_to(directory).as_posix()}|{sha256_file(path)}")
    return hashlib.sha256("\n".join(records).encode("utf-8")).hexdigest(), len(files)


def verify_approved_frozen_artifacts(project_root: Path) -> dict[str, Any]:
    """Compare against approved hashes, not a snapshot taken during this run."""

    observed: dict[str, Any] = {}
    failures: list[str] = []
    for relative, expected_hash in FROZEN_FILE_HASHES.items():
        path = project_root / relative
        if not path.is_file():
            failures.append(f"missing:{relative}")
            continue
        actual_hash = sha256_file(path)
        observed[relative] = actual_hash
        if actual_hash != expected_hash:
            failures.append(f"hash:{relative}")
    tree_hash, file_count = _tree_digest(project_root / "data" / "processed" / "guidance")
    observed["data/processed/guidance"] = {
        "tree_sha256": tree_hash,
        "file_count": file_count,
    }
    if tree_hash != FROZEN_PROCESSED_TREE_SHA256 or file_count != FROZEN_PROCESSED_FILE_COUNT:
        failures.append("tree:data/processed/guidance")
    if failures:
        raise Phase4GateError("Approved Phase 1-3 freeze gate failed: " + ", ".join(failures))
    return observed


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    required = {"doc_id", "notice_code", "notice_family", "title", "source_url", "source_origin"}
    if len(rows) != 50 or any(not required.issubset(row) for row in rows):
        raise Phase4GateError("Frozen corpus manifest schema/count is invalid")
    if len({row["doc_id"] for row in rows}) != 50:
        raise Phase4GateError("Frozen corpus manifest doc IDs are not unique")
    return rows


def _heading_events(markdown: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    offset = 0
    active_fence: str | None = None
    stack: list[tuple[int, str, str]] = []
    for line_number, line in enumerate(markdown.splitlines(keepends=True), start=1):
        content = line.rstrip("\r\n")
        fence = _FENCE.match(content)
        if fence:
            marker = fence.group(1)[0]
            if active_fence is None:
                active_fence = marker
            elif marker == active_fence:
                active_fence = None
            offset += len(line)
            continue
        match = None if active_fence is not None else _ATX_HEADING.match(content)
        if match:
            level = len(match.group(1))
            raw_heading = normalize_exact(match.group(2))
            display_heading = visible_markdown(raw_heading)
            if level == 1:
                stack.clear()
                path: tuple[str, ...] = ()
                raw_path: tuple[str, ...] = ()
                levels: tuple[int, ...] = ()
            else:
                while stack and stack[-1][0] >= level:
                    stack.pop()
                stack.append((level, display_heading, raw_heading))
                path = tuple(item[1] for item in stack)
                raw_path = tuple(item[2] for item in stack)
                levels = tuple(item[0] for item in stack)
            events.append(
                {
                    "level": level,
                    "line": line_number,
                    "char_start": offset,
                    "path": path,
                    "raw_path": raw_path,
                    "levels": levels,
                }
            )
        offset += len(line)
    return events


def parse_markdown_sections(markdown: str, tokenizer: Any) -> tuple[list[int], list[HeadingSection]]:
    """Parse H2+ section intervals and map them to exact source-token offsets."""

    encoded = tokenizer(markdown, add_special_tokens=False, return_offsets_mapping=True)
    token_ids = list(encoded["input_ids"])
    offsets = [tuple(pair) for pair in encoded["offset_mapping"]]
    if token_ids != list(tokenizer.encode(markdown, add_special_tokens=False)):
        raise Phase4GateError("Tokenizer ID/offset encoding mismatch")
    events = _heading_events(markdown)
    if sum(event["level"] == 1 for event in events) != 1:
        raise Phase4GateError("Each frozen Markdown document must contain exactly one H1")

    sections: list[HeadingSection] = []
    for index, event in enumerate(events):
        if not event["path"]:
            continue
        raw_end = events[index + 1]["char_start"] if index + 1 < len(events) else len(markdown)
        segment = markdown[event["char_start"] : raw_end]
        nonspace = [position for position, character in enumerate(segment) if not character.isspace()]
        if not nonspace:
            continue
        char_start = event["char_start"] + nonspace[0]
        char_end = event["char_start"] + nonspace[-1] + 1
        intersecting = [
            token_index
            for token_index, (start, end) in enumerate(offsets)
            if end > start and end > char_start and start < char_end
        ]
        if not intersecting:
            raise Phase4GateError(f"Heading at line {event['line']} has no source-token interval")
        sections.append(
            HeadingSection(
                path=event["path"],
                raw_path=event["raw_path"],
                levels=event["levels"],
                heading_line=event["line"],
                char_start=char_start,
                char_end=char_end,
                token_start=min(intersecting),
                token_end=max(intersecting) + 1,
            )
        )
    return token_ids, sections


def attribute_chunk(
    chunk: ChunkRecord,
    sections: Sequence[HeadingSection],
) -> list[dict[str, Any]]:
    attributions: list[dict[str, Any]] = []
    seen_paths: set[tuple[str, ...]] = set()
    for section in sections:
        overlap_start = max(chunk.token_start, section.token_start)
        overlap_end = min(chunk.token_end, section.token_end)
        if overlap_end <= overlap_start or section.path in seen_paths:
            continue
        seen_paths.add(section.path)
        attributions.append(
            {
                "path": list(section.path),
                "levels": list(section.levels),
                "raw_path": list(section.raw_path),
                "heading_line": section.heading_line,
                "overlap_token_start": overlap_start,
                "overlap_token_end": overlap_end,
                "overlap_token_count": overlap_end - overlap_start,
            }
        )
    return attributions


def build_fixed_chunk_section_map(
    *,
    project_root: Path,
    tokenizer: Any,
    chunks: list[ChunkRecord],
    manifest_rows: list[dict[str, str]],
    frozen_hashes: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, list[HeadingSection]]]:
    """Create annotation-only heading paths without changing frozen chunks."""

    chunks_by_doc: dict[str, list[ChunkRecord]] = defaultdict(list)
    for chunk in chunks:
        chunks_by_doc[str(chunk.metadata["doc_id"])].append(chunk)
    manifest_by_doc = {row["doc_id"]: row for row in manifest_rows}
    if set(chunks_by_doc) != set(manifest_by_doc):
        raise Phase4GateError("Registry document set does not equal the frozen manifest")

    records: list[dict[str, Any]] = []
    sections_by_doc: dict[str, list[HeadingSection]] = {}
    heading_counts = Counter()
    for row in manifest_rows:
        doc_id = row["doc_id"]
        path = project_root / "data" / "processed" / "guidance" / f"{doc_id}.md"
        payload = path.read_bytes()
        markdown = payload.decode("utf-8")
        token_ids, sections = parse_markdown_sections(markdown, tokenizer)
        sections_by_doc[doc_id] = sections
        heading_counts.update(section.levels[-1] for section in sections)
        doc_chunks = sorted(chunks_by_doc[doc_id], key=lambda item: item.token_start)
        if not doc_chunks or doc_chunks[0].token_start != 0 or doc_chunks[-1].token_end != len(token_ids):
            raise Phase4GateError(f"Frozen chunk boundaries do not cover source EOF for {doc_id}")
        for prior, current in zip(doc_chunks, doc_chunks[1:]):
            if current.token_start != prior.token_end - 40:
                raise Phase4GateError(f"Frozen chunk overlap changed for {doc_id}")
        for chunk in doc_chunks:
            source_text = tokenizer.decode(
                token_ids[chunk.token_start : chunk.token_end],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            if source_text != chunk.text:
                raise Phase4GateError(f"Frozen chunk text no longer reconstructs for {chunk.chunk_id}")
            attributions = attribute_chunk(chunk, sections)
            records.append(
                {
                    "chunk_id": chunk.chunk_id,
                    "doc_id": doc_id,
                    "notice_code": row["notice_code"],
                    "token_start": chunk.token_start,
                    "token_end": chunk.token_end,
                    "heading_paths": attributions,
                    "spans_multiple_heading_paths": len(attributions) > 1,
                }
            )

    if [record["chunk_id"] for record in records] != [chunk.chunk_id for chunk in chunks]:
        raise Phase4GateError("Attribution-map record order/ID set differs from the frozen registry")
    unattributed = [record["chunk_id"] for record in records if not record["heading_paths"]]
    path_distribution = Counter(len(record["heading_paths"]) for record in records)
    multi_count = sum(record["spans_multiple_heading_paths"] for record in records)
    quality_gate = len(records) == 350 and not unattributed and len({record["chunk_id"] for record in records}) == 350
    result = {
        "schema_version": "1.0",
        "generated_at_utc": utc_now(),
        "annotation_only": True,
        "inputs": {
            "processed_guidance_tree_sha256": FROZEN_PROCESSED_TREE_SHA256,
            "chunk_registry_path": REGISTRY_PATH.as_posix(),
            "chunk_registry_sha256": FROZEN_FILE_HASHES[REGISTRY_PATH.as_posix()],
            "section_benchmark_path": SECTION_BENCHMARK_PATH.as_posix(),
            "section_benchmark_sha256": SECTION_BENCHMARK_FROZEN_SHA256,
            "approved_frozen_hashes": frozen_hashes,
            "tokenizer_name": TOKENIZER_NAME,
            "tokenizer_revision": TOKENIZER_REVISION,
        },
        "attribution_policy": {
            "h1_document_title_included": False,
            "path_scope": "H2-H6 level-stack hierarchy",
            "heading_line_ownership": "new_section",
            "chunk_interval": "half_open_source_token_span",
            "intersection_rule": "positive overlap with a token intersecting non-whitespace section content",
            "normalization": "Unicode NFC, trim, collapse whitespace; punctuation and case otherwise exact",
            "multi_section_policy": "record_all_matching_heading_paths",
        },
        "summary": {
            "document_count": len(manifest_rows),
            "chunk_count": len(records),
            "unattributed_chunk_count": len(unattributed),
            "single_heading_path_chunk_count": len(records) - multi_count,
            "multi_heading_path_chunk_count": multi_count,
            "total_chunk_heading_path_associations": sum(
                len(record["heading_paths"]) for record in records
            ),
            "heading_path_count_distribution": {
                str(key): path_distribution[key] for key in sorted(path_distribution)
            },
            "parsed_h2_h6_section_count": len([section for values in sections_by_doc.values() for section in values]),
            "terminal_heading_level_counts": {
                f"H{key}": heading_counts[key] for key in sorted(heading_counts)
            },
        },
        "unattributed_chunk_ids": unattributed,
        "chunks": records,
        "quality_gate_passed": quality_gate,
    }
    if not quality_gate:
        raise Phase4GateError("Fixed-chunk section-attribution quality gate failed")
    return result, sections_by_doc


def load_section_questions(path: Path) -> list[dict[str, Any]]:
    questions = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(questions, list) or len(questions) != 15:
        raise Phase4GateError("Section benchmark must be a 15-record top-level array")
    expected_ids = {f"S{number:02d}" for number in range(1, 16)}
    observed_ids: list[str] = []
    for record in questions:
        if not isinstance(record, dict) or set(record) != BENCHMARK_FIELDS:
            raise Phase4GateError("Section benchmark record schema differs from the frozen contract")
        observed_ids.append(record["id"])
        for field in (
            "id",
            "question",
            "language_style",
            "expected_doc_id",
            "expected_notice_code",
            "expected_heading",
            "expected_evidence_excerpt",
            "rationale",
        ):
            if not isinstance(record[field], str) or not record[field].strip():
                raise Phase4GateError(f"Section benchmark has blank/invalid {field}")
        if record["language_style"] not in {"naive", "expert"}:
            raise Phase4GateError(f"Invalid language style for {record['id']}")
        if not isinstance(record["expected_heading_path"], list) or not record["expected_heading_path"]:
            raise Phase4GateError(f"Invalid expected heading path for {record['id']}")
        if any(not isinstance(item, str) or not item.strip() for item in record["expected_heading_path"]):
            raise Phase4GateError(f"Invalid expected heading path component for {record['id']}")
    if len(set(observed_ids)) != 15 or set(observed_ids) != expected_ids:
        raise Phase4GateError("Section benchmark IDs must be exactly S01-S15")
    if Counter(record["language_style"] for record in questions) != {"naive": 10, "expert": 5}:
        raise Phase4GateError("Section benchmark style split must be exactly 10 naive / 5 expert")
    if len({normalize_exact(record["question"]).casefold() for record in questions}) != 15:
        raise Phase4GateError("Section benchmark contains duplicate normalized questions")
    return questions


def validate_section_benchmark(
    *,
    questions: list[dict[str, Any]],
    manifest_rows: list[dict[str, str]],
    sections_by_doc: dict[str, list[HeadingSection]],
    project_root: Path,
) -> dict[str, Any]:
    manifest_by_doc = {row["doc_id"]: row for row in manifest_rows}
    families = Counter()
    for record in questions:
        question_id = record["id"]
        doc_id = record["expected_doc_id"]
        if doc_id not in manifest_by_doc:
            raise Phase4GateError(f"Unknown expected document for {question_id}")
        manifest_row = manifest_by_doc[doc_id]
        if record["expected_notice_code"] != manifest_row["notice_code"]:
            raise Phase4GateError(f"Expected notice code mismatch for {question_id}")
        expected_path = tuple(normalize_exact(item) for item in record["expected_heading_path"])
        if normalize_exact(record["expected_heading"]) != expected_path[-1]:
            raise Phase4GateError(f"Expected heading/path terminal mismatch for {question_id}")
        if normalize_exact(record["question"]).casefold() == expected_path[-1].casefold():
            raise Phase4GateError(f"Question mechanically copies its expected heading for {question_id}")
        matches = [section for section in sections_by_doc[doc_id] if section.path == expected_path]
        if len(matches) != 1:
            raise Phase4GateError(
                f"Expected full heading path for {question_id} resolves {len(matches)} times"
            )
        markdown = (
            project_root / "data" / "processed" / "guidance" / f"{doc_id}.md"
        ).read_text(encoding="utf-8")
        section = matches[0]
        section_visible = visible_markdown(markdown[section.char_start : section.char_end])
        evidence_visible = visible_markdown(record["expected_evidence_excerpt"])
        if evidence_visible not in section_visible:
            raise Phase4GateError(f"Evidence excerpt does not occur in expected section for {question_id}")
        families[manifest_row["notice_family"]] += 1

    unique_notices = len({record["expected_notice_code"] for record in questions})
    if unique_notices < 10 or len(families) < 2:
        raise Phase4GateError("Section benchmark lacks required notice/family coverage")
    return {
        "question_count": len(questions),
        "unique_notice_count": unique_notices,
        "language_style_counts": dict(Counter(record["language_style"] for record in questions)),
        "notice_family_counts": dict(sorted(families.items())),
        "all_expected_headings_verified": True,
        "all_evidence_excerpts_verified": True,
        "selected_before_attribution_or_retrieval": True,
        "selection_basis_attestation": "processed guidance evidence only",
        "not_selected_because_of_known_fixed_retrieval_failure_attestation": True,
    }


def normalized_path(path: Sequence[str]) -> tuple[str, ...]:
    return tuple(normalize_exact(item) for item in path)


def score_section_ranking(
    expected_notice_code: str,
    expected_heading_path: Sequence[str],
    ranked: Sequence[dict[str, Any]],
) -> SectionScore:
    target_path = normalized_path(expected_heading_path)
    first_rank: int | None = None
    for rank, item in enumerate(ranked[:TOP_K], start=1):
        paths = {normalized_path(path) for path in item["attributed_heading_paths"]}
        correct = item["retrieved_notice_code"] == expected_notice_code and target_path in paths
        if correct:
            first_rank = rank
            break
    return SectionScore(
        precision_at_1=int(first_rank == 1),
        reciprocal_rank=0.0 if first_rank is None else 1.0 / first_rank,
        hit_at_5=int(first_rank is not None),
        first_correct_rank=first_rank,
    )


def aggregate_section_scores(scores: Sequence[SectionScore]) -> dict[str, int | float]:
    if not scores:
        raise Phase4GateError("Cannot aggregate an empty section score set")
    n = len(scores)
    p1_count = sum(score.precision_at_1 for score in scores)
    rr_sum = sum(score.reciprocal_rank for score in scores)
    hit_count = sum(score.hit_at_5 for score in scores)
    return {
        "n": n,
        "correct_at_1": p1_count,
        "section_precision_at_1": p1_count / n,
        "reciprocal_rank_sum": rr_sum,
        "section_mrr": rr_sum / n,
        "hit_at_5_count": hit_count,
        "section_hit_at_5": hit_count / n,
    }


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _safe_provider_failure(stage: str, exc: BaseException) -> ProviderGateError:
    # Provider exception text can contain request details. Persist only its type.
    return ProviderGateError(f"{stage} failed ({type(exc).__name__}); no credentials were logged")


def _namespace_count(stats: Any, namespace: str) -> int:
    namespaces = _field(stats, "namespaces", {}) or {}
    summary = namespaces.get(namespace) if hasattr(namespaces, "get") else None
    if summary is None:
        return 0
    return int(_field(summary, "vector_count", 0) or 0)


def _flatten_list_page(page: Any) -> list[str]:
    if isinstance(page, str):
        return [page]
    if isinstance(page, Mapping):
        values = page.get("vectors") or page.get("ids") or []
    else:
        values = _field(page, "vectors", page)
    if isinstance(values, str):
        return [values]
    try:
        iterator = iter(values)
    except TypeError:
        return []
    result: list[str] = []
    for value in iterator:
        if isinstance(value, str):
            result.append(value)
        else:
            vector_id = _field(value, "id")
            if vector_id is not None:
                result.append(str(vector_id))
    return result


def _validate_vector(vector: Sequence[float]) -> None:
    if isinstance(vector, (str, bytes)) or len(vector) != EMBEDDING_DIMENSION:
        observed = 0 if isinstance(vector, (str, bytes)) else len(vector)
        raise ProviderGateError(
            f"Embedding dimension mismatch: expected {EMBEDDING_DIMENSION}, observed {observed}"
        )
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) for value in vector):
        raise ProviderGateError("Embedding contains a non-finite or non-numeric value")


@dataclass(frozen=True)
class ReadOnlyIndexState:
    index: Any
    ready: bool
    dimension: int
    metric: str
    vector_type: str
    cloud: str | None
    region: str | None


class ReadOnlyPineconeBaseline:
    """Query-only access to the already-approved Phase 3 Pinecone index.

    Deliberately exposes no create, configure, upsert, update, clear, or delete
    operation. An absent or incompatible index is a hard stop.
    """

    def __init__(self, *, api_key: str, client: Any | None = None) -> None:
        if not isinstance(api_key, str) or not api_key.strip():
            raise ProviderGateError("PINECONE_API_KEY is empty")
        try:
            self._pc = client if client is not None else Pinecone(api_key=api_key)
        except Exception as exc:
            raise _safe_provider_failure("Pinecone read-only client construction", exc) from None
        self.state: ReadOnlyIndexState | None = None

    def __repr__(self) -> str:
        return f"ReadOnlyPineconeBaseline(index_name={INDEX_NAME!r}, namespace={NAMESPACE!r})"

    def require_existing_index(self) -> ReadOnlyIndexState:
        try:
            exists = bool(self._pc.indexes.exists(INDEX_NAME))
            if not exists:
                raise ProviderGateError(
                    "The frozen Pinecone index does not exist; Phase 4A will not create it"
                )
            description = self._pc.indexes.describe(INDEX_NAME)
        except ProviderGateError:
            raise
        except Exception as exc:
            raise _safe_provider_failure("Pinecone read-only index verification", exc) from None

        dimension = int(_field(description, "dimension", 0) or 0)
        metric = str(_field(description, "metric", "")).lower()
        vector_type = str(_field(description, "vector_type", "dense") or "dense").lower()
        status = _field(description, "status", {}) or {}
        ready = bool(_field(status, "ready", False))
        host = str(_field(description, "host", "") or "")
        if dimension != EMBEDDING_DIMENSION or metric != INDEX_METRIC:
            raise ProviderGateError(
                "Existing Pinecone index is incompatible "
                f"(dimension={dimension}, metric={metric!r}); it was not modified"
            )
        if vector_type not in {"", "dense"}:
            raise ProviderGateError(
                f"Existing Pinecone index vector type is {vector_type!r}; it was not modified"
            )
        if not ready or not host:
            raise ProviderGateError("The frozen Pinecone index is not ready for read-only queries")
        spec = _field(description, "spec", {}) or {}
        serverless = _field(spec, "serverless", {}) or {}
        cloud = _field(serverless, "cloud")
        region = _field(serverless, "region")
        if (
            str(cloud or "").lower() != INDEX_CLOUD
            or str(region or "").lower() != INDEX_REGION
        ):
            raise ProviderGateError(
                "Existing Pinecone index location differs from the frozen Phase 3 index "
                f"({INDEX_CLOUD}/{INDEX_REGION}); it was not modified"
            )
        try:
            index = self._pc.index(host=host)
        except Exception as exc:
            raise _safe_provider_failure("Pinecone read-only data-plane connection", exc) from None
        self.state = ReadOnlyIndexState(
            index=index,
            ready=ready,
            dimension=dimension,
            metric=metric,
            vector_type=vector_type or "dense",
            cloud=None if cloud is None else str(cloud),
            region=None if region is None else str(region),
        )
        return self.state

    @property
    def index(self) -> Any:
        if self.state is None:
            raise ProviderGateError("Pinecone index has not passed the read-only compatibility gate")
        return self.state.index

    def namespace_snapshot(self) -> tuple[int, set[str]]:
        try:
            count = _namespace_count(self.index.describe_index_stats(), NAMESPACE)
            ids: set[str] = set()
            for page in self.index.list(namespace=NAMESPACE):
                page_ids = _flatten_list_page(page)
                if len(set(page_ids)) != len(page_ids):
                    raise ProviderGateError("Pinecone namespace listing contains duplicate IDs")
                ids.update(page_ids)
        except ProviderGateError:
            raise
        except Exception as exc:
            raise _safe_provider_failure("Pinecone read-only namespace snapshot", exc) from None
        if count != len(ids):
            raise ProviderGateError(
                f"Pinecone namespace count/list mismatch: count={count}, listed={len(ids)}"
            )
        return count, ids

    def query_known_notice(
        self,
        vector: Sequence[float],
        *,
        notice_code: str,
        eligible_chunk_count: int,
    ) -> list[Any]:
        _validate_vector(vector)
        if not isinstance(notice_code, str) or not notice_code.strip():
            raise ProviderGateError("A nonempty exact notice code is required for the Phase 4A filter")
        expected_returned = min(TOP_K, eligible_chunk_count)
        if expected_returned <= 0:
            raise ProviderGateError(f"No frozen chunks are eligible for notice code {notice_code!r}")
        try:
            response = self.index.query(
                namespace=NAMESPACE,
                vector=list(vector),
                top_k=TOP_K,
                filter={"notice_code": {"$eq": notice_code}},
                include_metadata=True,
                include_values=False,
            )
        except Exception as exc:
            raise _safe_provider_failure("Pinecone filtered section query", exc) from None
        matches = list(_field(response, "matches", []) or [])
        if len(matches) != expected_returned:
            raise ProviderGateError(
                f"Pinecone returned {len(matches)} filtered matches for {notice_code!r}; "
                f"expected {expected_returned}"
            )
        match_ids = [str(_field(match, "id", "") or "") for match in matches]
        if any(not match_id for match_id in match_ids) or len(set(match_ids)) != len(match_ids):
            raise ProviderGateError("Pinecone filtered results contain blank or duplicate chunk IDs")
        scores = [finite_score(_field(match, "score")) for match in matches]
        if any(left < right for left, right in zip(scores, scores[1:])):
            raise ProviderGateError("Pinecone filtered results are not in descending score order")
        return matches


def _safe_metadata(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    try:
        return dict(value)
    except (TypeError, ValueError):
        raise Phase4GateError("Pinecone returned malformed metadata") from None


def _text_preview(text: str, limit: int = 360) -> str:
    return re.sub(r"\s+", " ", text).strip()[:limit]


def _id_set_sha256(ids: set[str]) -> str:
    return hashlib.sha256("\n".join(sorted(ids)).encode("utf-8")).hexdigest()


def _map_by_chunk(section_map: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    records = section_map.get("chunks")
    if not isinstance(records, list) or len(records) != 350:
        raise Phase4GateError("Fixed-chunk section map must contain exactly 350 records")
    result: dict[str, dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict):
            raise Phase4GateError("Fixed-chunk section map contains a malformed record")
        chunk_id = record.get("chunk_id")
        paths = record.get("heading_paths")
        if not isinstance(chunk_id, str) or not chunk_id or not isinstance(paths, list) or not paths:
            raise Phase4GateError("Fixed-chunk section map contains an invalid ID/path set")
        if chunk_id in result:
            raise Phase4GateError("Fixed-chunk section map contains duplicate chunk IDs")
        result[chunk_id] = record
    if not section_map.get("quality_gate_passed"):
        raise Phase4GateError("Fixed-chunk section map quality gate is not passed")
    return result


def _validate_existing_map(existing: Mapping[str, Any], candidate: Mapping[str, Any]) -> None:
    comparable_keys = (
        "schema_version",
        "annotation_only",
        "inputs",
        "attribution_policy",
        "summary",
        "unattributed_chunk_ids",
        "chunks",
        "quality_gate_passed",
    )
    if any(existing.get(key) != candidate.get(key) for key in comparable_keys):
        raise Phase4GateError("Existing Phase 4A section map differs from deterministic reconstruction")
    _map_by_chunk(existing)


def _section_margin(ranks: Sequence[dict[str, Any]]) -> float | None:
    correct = [float(rank["similarity_score"]) for rank in ranks if rank["correct"]]
    incorrect = [float(rank["similarity_score"]) for rank in ranks if not rank["correct"]]
    if not correct or not incorrect:
        return None
    return max(correct) - max(incorrect)


def _hardest_key(trace: Mapping[str, Any]) -> tuple[int, int, float, str]:
    score = trace["section_score"]
    no_hit_first = 0 if score["hit_at_5"] == 0 else 1
    first_rank = score["first_correct_rank"]
    descending_rank = -(TOP_K + 1 if first_rank is None else int(first_rank))
    margin = trace.get("correct_vs_incorrect_similarity_margin")
    margin_value = math.inf if margin is None else float(margin)
    return no_hit_first, descending_rank, margin_value, str(trace["id"])


def _trace_one_query(
    *,
    question: dict[str, Any],
    manifest_row: dict[str, str],
    vector: Sequence[float],
    store: ReadOnlyPineconeBaseline,
    chunks_by_id: dict[str, ChunkRecord],
    map_by_id: dict[str, dict[str, Any]],
    eligible_ids: set[str],
) -> tuple[dict[str, Any], float]:
    started = time.perf_counter()
    matches = store.query_known_notice(
        vector,
        notice_code=question["expected_notice_code"],
        eligible_chunk_count=len(eligible_ids),
    )
    latency = time.perf_counter() - started
    rank_records: list[dict[str, Any]] = []
    for rank, match in enumerate(matches, start=1):
        chunk_id = str(_field(match, "id", "") or "")
        if chunk_id not in eligible_ids or chunk_id not in chunks_by_id or chunk_id not in map_by_id:
            raise Phase4GateError(
                f"Filtered result for {question['id']} contains an unknown or wrong-notice chunk ID"
            )
        metadata = _safe_metadata(_field(match, "metadata", {}))
        if set(metadata) != set(PINECONE_METADATA_KEYS):
            raise Phase4GateError(f"Filtered result for {question['id']} has unexpected metadata")
        chunk = chunks_by_id[chunk_id]
        expected_metadata = {key: chunk.metadata[key] for key in PINECONE_METADATA_KEYS}
        if metadata != expected_metadata or metadata["chunk_id"] != chunk_id:
            raise Phase4GateError(f"Filtered result metadata does not match the registry for {chunk_id}")
        if (
            metadata["notice_code"] != question["expected_notice_code"]
            or metadata["doc_id"] != question["expected_doc_id"]
        ):
            raise Phase4GateError(f"Notice-only filter returned an unexpected document for {question['id']}")
        map_record = map_by_id[chunk_id]
        if (
            map_record["doc_id"] != metadata["doc_id"]
            or map_record["notice_code"] != metadata["notice_code"]
            or map_record["token_start"] != chunk.token_start
            or map_record["token_end"] != chunk.token_end
        ):
            raise Phase4GateError(f"Section map does not reconcile with frozen chunk {chunk_id}")
        paths = [entry["path"] for entry in map_record["heading_paths"]]
        notice_match = metadata["notice_code"] == question["expected_notice_code"]
        section_match = normalized_path(question["expected_heading_path"]) in {
            normalized_path(path) for path in paths
        }
        rank_records.append(
            {
                "rank": rank,
                "chunk_id": chunk_id,
                "doc_id": metadata["doc_id"],
                "retrieved_notice_code": metadata["notice_code"],
                "title": metadata["title"],
                "similarity_score": finite_score(_field(match, "score")),
                "text_preview": _text_preview(chunk.text),
                "attributed_heading_paths": paths,
                "spans_multiple_heading_paths": bool(map_record["spans_multiple_heading_paths"]),
                "notice_match": notice_match,
                "section_match": section_match,
                "correct": notice_match and section_match,
            }
        )

    score = score_section_ranking(
        question["expected_notice_code"], question["expected_heading_path"], rank_records
    )
    if any(rank["correct"] for rank in rank_records) != bool(score.hit_at_5):
        raise Phase4GateError(f"Section score invariant failed for {question['id']}")
    trace = {
        **question,
        "notice_family": manifest_row["notice_family"],
        "retrieval_filter": {"notice_code": {"$eq": question["expected_notice_code"]}},
        "eligible_chunk_count": len(eligible_ids),
        "returned_chunk_count": len(rank_records),
        "ranks": rank_records,
        "section_score": {
            "precision_at_1": score.precision_at_1,
            "reciprocal_rank": score.reciprocal_rank,
            "hit_at_5": score.hit_at_5,
            "first_correct_rank": score.first_correct_rank,
        },
        "correct_vs_incorrect_similarity_margin": _section_margin(rank_records),
        "query_latency_seconds": round(latency, 6),
    }
    return trace, latency


def _aggregate_traces(traces: Sequence[dict[str, Any]]) -> dict[str, Any]:
    def scores(values: Sequence[dict[str, Any]]) -> list[SectionScore]:
        return [
            SectionScore(
                precision_at_1=int(value["section_score"]["precision_at_1"]),
                reciprocal_rank=float(value["section_score"]["reciprocal_rank"]),
                hit_at_5=int(value["section_score"]["hit_at_5"]),
                first_correct_rank=value["section_score"]["first_correct_rank"],
            )
            for value in values
        ]

    if len(traces) != 15 or len({trace["id"] for trace in traces}) != 15:
        raise Phase4GateError("Phase 4A metrics require exactly 15 unique traces")
    by_style = {
        style: aggregate_section_scores(scores([trace for trace in traces if trace["language_style"] == style]))
        for style in ("naive", "expert")
    }
    families = sorted({trace["notice_family"] for trace in traces})
    by_family = {
        family: aggregate_section_scores(scores([trace for trace in traces if trace["notice_family"] == family]))
        for family in families
    }
    result = {
        "overall": aggregate_section_scores(scores(traces)),
        "by_language_style": by_style,
        "by_notice_family": by_family,
    }
    if result["overall"]["n"] != 15 or by_style["naive"]["n"] != 10 or by_style["expert"]["n"] != 5:
        raise Phase4GateError("Phase 4A metric denominators differ from the frozen benchmark")
    return result


def evaluate_fixed_sections(
    *,
    project_root: Path,
    config: Any,
    questions: list[dict[str, Any]],
    benchmark_audit: dict[str, Any],
    manifest_rows: list[dict[str, str]],
    chunks: list[ChunkRecord],
    section_map: dict[str, Any],
    section_map_sha256: str,
    frozen_before: dict[str, Any],
    store: ReadOnlyPineconeBaseline | None = None,
    embedder: NebiusEmbeddings | None = None,
) -> dict[str, Any]:
    chunks_by_id = {chunk.chunk_id: chunk for chunk in chunks}
    if len(chunks_by_id) != 350:
        raise Phase4GateError("The frozen Phase 3 registry must contain 350 unique chunks")
    map_by_id = _map_by_chunk(section_map)
    if set(map_by_id) != set(chunks_by_id):
        raise Phase4GateError("Section map and frozen chunk registry ID sets differ")
    manifest_by_doc = {row["doc_id"]: row for row in manifest_rows}
    eligible_by_code: dict[str, set[str]] = defaultdict(set)
    for chunk in chunks:
        eligible_by_code[str(chunk.metadata["notice_code"])].add(chunk.chunk_id)

    active_store = store or ReadOnlyPineconeBaseline(api_key=config.pinecone_api_key)
    state = active_store.require_existing_index()
    pre_count, pre_ids = active_store.namespace_snapshot()
    expected_ids = set(chunks_by_id)
    if pre_count != 350 or pre_ids != expected_ids:
        raise Phase4GateError("Frozen Pinecone namespace does not exactly match the 350-chunk registry")

    active_embedder = embedder or NebiusEmbeddings(
        api_key=config.nebius_api_key,
        base_url=config.nebius_base_url,
        batch_size=15,
    )
    exact_question_texts = [question["question"] for question in questions]
    vectors = active_embedder.embed_documents(exact_question_texts)
    if len(vectors) != 15:
        raise Phase4GateError("Nebius did not return exactly 15 section-query embeddings")
    for vector in vectors:
        _validate_vector(vector)

    traces: list[dict[str, Any]] = []
    for question, vector in zip(questions, vectors, strict=True):
        row = manifest_by_doc[question["expected_doc_id"]]
        eligible_ids = eligible_by_code[question["expected_notice_code"]]
        trace, _latency = _trace_one_query(
            question=question,
            manifest_row=row,
            vector=vector,
            store=active_store,
            chunks_by_id=chunks_by_id,
            map_by_id=map_by_id,
            eligible_ids=eligible_ids,
        )
        traces.append(trace)

    post_count, post_ids = active_store.namespace_snapshot()
    if post_count != pre_count or post_ids != pre_ids:
        raise Phase4GateError("Pinecone namespace changed during the read-only Phase 4A run")
    frozen_after = verify_approved_frozen_artifacts(project_root)
    if frozen_after != frozen_before:
        raise Phase4GateError("A frozen Phase 1-3 artifact changed during the Phase 4A run")
    if sha256_file(project_root / SECTION_BENCHMARK_PATH) != SECTION_BENCHMARK_FROZEN_SHA256:
        raise Phase4GateError("The frozen 15-question benchmark changed during retrieval")
    if sha256_file(project_root / MAP_PATH) != section_map_sha256:
        raise Phase4GateError("The frozen fixed-chunk section map changed during retrieval")

    metrics = _aggregate_traces(traces)
    hardest = sorted(traces, key=_hardest_key)[:5]
    public_config = config.public_summary()
    return {
        "schema_version": "1.0",
        "generated_at_utc": utc_now(),
        "phase": "4A-fixed-chunk-section-baseline",
        "experiment_contract": {
            "retrieval_mode": "dense_fixed_chunks_with_exact_notice_code_filter",
            "query_policy": "exact_frozen_question_text_only",
            "top_k": TOP_K,
            "section_correctness": "exact notice_code AND exact normalized full heading path",
            "heading_metadata_used_for_ranking": False,
            "generation_used": False,
            "fixed_document_vectors_reused": True,
            "blind_selection_attestation": benchmark_audit["selected_before_attribution_or_retrieval"],
            "hardest_five_rule": PREDECLARED_HARDEST_RULE,
        },
        "precommitted_thresholds": {
            "meaningful_heading_aware_gain_absolute_percentage_points": MEANINGFUL_GAIN_ABSOLUTE_POINTS,
            "section_null_fixed_precision_at_1_at_least": NULL_FIXED_PRECISION_THRESHOLD,
            "section_null_heading_aware_gain_less_than_percentage_points": NULL_IMPROVEMENT_LESS_THAN_POINTS,
            "decision_status": "not_adjudicated_until_heading_aware_section_run",
        },
        "inputs": {
            "section_benchmark": {
                "path": SECTION_BENCHMARK_PATH.as_posix(),
                "sha256": SECTION_BENCHMARK_FROZEN_SHA256,
                "audit": benchmark_audit,
            },
            "fixed_chunk_section_map": {
                "path": MAP_PATH.as_posix(),
                "sha256": section_map_sha256,
                "summary": section_map["summary"],
            },
            "chunk_registry": {
                "path": REGISTRY_PATH.as_posix(),
                "sha256": FROZEN_FILE_HASHES[REGISTRY_PATH.as_posix()],
                "count": len(chunks),
            },
            "approved_frozen_artifacts_before": frozen_before,
            "approved_frozen_artifacts_after": frozen_after,
        },
        "embedding": {
            "provider": "Nebius Token Factory",
            "model": EMBEDDING_MODEL,
            "dimension": EMBEDDING_DIMENSION,
            "base_url": public_config["nebius_base_url"],
            "secrets_source": public_config["secrets_source"],
            "exact_question_text_count": 15,
            "question_embedding_request_latency": latency_summary(active_embedder.request_latencies),
            "document_embedding_requests": 0,
            "document_vectors_reused": True,
        },
        "pinecone": {
            "index_name": INDEX_NAME,
            "namespace": NAMESPACE,
            "index_existed": True,
            "ready": state.ready,
            "dimension": state.dimension,
            "metric": state.metric,
            "vector_type": state.vector_type,
            "cloud": state.cloud,
            "region": state.region,
            "pre_query_vector_count": pre_count,
            "post_query_vector_count": post_count,
            "pre_query_id_set_sha256": _id_set_sha256(pre_ids),
            "post_query_id_set_sha256": _id_set_sha256(post_ids),
            "exact_id_parity_before_and_after": pre_ids == post_ids == expected_ids,
            "query_latency": latency_summary(
                [float(trace["query_latency_seconds"]) for trace in traces]
            ),
            "index_create_calls": 0,
            "document_upsert_calls": 0,
            "update_calls": 0,
            "delete_or_clear_calls": 0,
        },
        "metrics": {"section_retrieval": metrics},
        "failure_counts": {
            "precision_at_1_failures": sum(trace["section_score"]["precision_at_1"] == 0 for trace in traces),
            "hit_at_5_failures": sum(trace["section_score"]["hit_at_5"] == 0 for trace in traces),
        },
        "fixed_section_headroom_absolute_percentage_points": round(
            (1.0 - metrics["overall"]["section_precision_at_1"]) * 100.0, 6
        ),
        "hardest_five_question_ids": [trace["id"] for trace in hardest],
        "queries": traces,
        "quality_gate_passed": True,
    }


def _summary_rows(results: Mapping[str, Any]) -> list[dict[str, Any]]:
    metrics = results["metrics"]["section_retrieval"]
    rows: list[dict[str, Any]] = []

    def add(scope: str, style: str, family: str, values: Mapping[str, Any]) -> None:
        rows.append(
            {
                "scope": scope,
                "language_style": style,
                "notice_family": family,
                **{key: values[key] for key in (
                    "n",
                    "correct_at_1",
                    "section_precision_at_1",
                    "reciprocal_rank_sum",
                    "section_mrr",
                    "hit_at_5_count",
                    "section_hit_at_5",
                )},
            }
        )

    add("overall", "all", "all", metrics["overall"])
    for style, values in metrics["by_language_style"].items():
        add("language_style", style, "all", values)
    for family, values in metrics["by_notice_family"].items():
        add("notice_family", "all", family, values)
    return rows


def write_summary_csv(path: Path, results: Mapping[str, Any]) -> None:
    rows = _summary_rows(results)
    fieldnames = (
        "scope",
        "language_style",
        "notice_family",
        "n",
        "correct_at_1",
        "section_precision_at_1",
        "reciprocal_rank_sum",
        "section_mrr",
        "hit_at_5_count",
        "section_hit_at_5",
    )
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    atomic_write_text(path, output.getvalue())


def _percent(value: float) -> str:
    return f"{value * 100:.1f}%"


def write_failure_analysis(path: Path, results: Mapping[str, Any]) -> None:
    metrics = results["metrics"]["section_retrieval"]
    overall = metrics["overall"]
    traces = list(results["queries"])
    failures = [trace for trace in traces if trace["section_score"]["precision_at_1"] == 0]
    hardest_by_id = {trace["id"]: trace for trace in traces}
    lines = [
        "# Phase 4A fixed-chunk section failure analysis",
        "",
        "This is the frozen fixed-chunk, notice-filtered section baseline. It reused the 350 Phase 3 vectors and did not build heading-aware chunks.",
        "",
        "## Contract and thresholds",
        "",
        f"- Section correctness: exact notice code and exact normalized full heading path.",
        f"- Meaningful later heading-aware gain: at least {MEANINGFUL_GAIN_ABSOLUTE_POINTS} absolute percentage points.",
        f"- Section null branch: fixed P@1 at least {_percent(NULL_FIXED_PRECISION_THRESHOLD)} and later gain under {NULL_IMPROVEMENT_LESS_THAN_POINTS} points.",
        "- The null decision is not made in Phase 4A because the heading-aware comparator has not been built.",
        "",
        "## Aggregate results",
        "",
        "| Scope | n | Section P@1 | Section MRR | Section Hit@5 |",
        "|---|---:|---:|---:|---:|",
        f"| Overall | {overall['n']} | {_percent(overall['section_precision_at_1'])} | {overall['section_mrr']:.4f} | {_percent(overall['section_hit_at_5'])} |",
    ]
    for style, values in metrics["by_language_style"].items():
        lines.append(
            f"| {style.title()} | {values['n']} | {_percent(values['section_precision_at_1'])} | {values['section_mrr']:.4f} | {_percent(values['section_hit_at_5'])} |"
        )
    for family, values in metrics["by_notice_family"].items():
        lines.append(
            f"| Family: {family} | {values['n']} | {_percent(values['section_precision_at_1'])} | {values['section_mrr']:.4f} | {_percent(values['section_hit_at_5'])} |"
        )
    lines.extend(
        [
            "",
            "## Rank-1 failures",
            "",
        ]
    )
    if not failures:
        lines.append("No Section P@1 failures occurred on the frozen 15-question benchmark.")
    for trace in failures:
        rank1 = trace["ranks"][0]
        top = ", ".join(
            f"r{item['rank']}:{'match' if item['correct'] else 'miss'}" for item in trace["ranks"]
        )
        lines.extend(
            [
                f"### {trace['id']} — {trace['expected_notice_code']}",
                "",
                f"- Question: {trace['question']}",
                f"- Expected path: `{' > '.join(trace['expected_heading_path'])}`",
                f"- Expected evidence: {trace['expected_evidence_excerpt']}",
                f"- Rank 1 paths: `{json.dumps(rank1['attributed_heading_paths'], ensure_ascii=False)}`",
                f"- Rank 1 score/preview: {rank1['similarity_score']:.6f} — {rank1['text_preview']}",
                f"- First correct rank: {trace['section_score']['first_correct_rank']}",
                f"- Top candidates: {top}",
                "",
            ]
        )
    lines.extend(
        [
            "## Predeclared five hardest",
            "",
            f"Ordering rule: {PREDECLARED_HARDEST_RULE}.",
            "",
        ]
    )
    for question_id in results["hardest_five_question_ids"]:
        trace = hardest_by_id[question_id]
        lines.append(
            f"- {question_id}: “{trace['question']}” — first correct rank="
            f"{trace['section_score']['first_correct_rank']}, "
            f"Hit@5={trace['section_score']['hit_at_5']}, margin={trace['correct_vs_incorrect_similarity_margin']}"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"Fixed-chunk Section P@1 leaves {results['fixed_section_headroom_absolute_percentage_points']:.1f} absolute percentage points to perfect on this benchmark.",
            "This is the only retrieval evidence collected here; no heading-aware, hybrid, reranked, generated, or agentic system was implemented.",
            "",
            "## Integrity confirmation",
            "",
            "- Frozen Phase 1–3 hashes passed before and after retrieval.",
            "- The existing namespace contained the same exact 350 IDs before and after.",
            "- Document embeddings, index creation, upserts, updates, and deletes were all zero.",
            "- API credentials were neither written nor logged.",
            "",
        ]
    )
    atomic_write_text(path, "\n".join(lines))


def prepare_offline_phase4a(project_root: Path) -> tuple[
    dict[str, Any],
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, str]],
    list[ChunkRecord],
    dict[str, Any],
]:
    frozen = verify_approved_frozen_artifacts(project_root)
    benchmark_path = project_root / SECTION_BENCHMARK_PATH
    if sha256_file(benchmark_path) != SECTION_BENCHMARK_FROZEN_SHA256:
        raise Phase4GateError("Section benchmark hash differs from its pre-attribution freeze")
    questions = load_section_questions(benchmark_path)
    manifest_rows = read_manifest(project_root / "data" / "corpus_manifest.csv")
    chunks = load_chunk_registry(project_root / REGISTRY_PATH)
    if len(chunks) != 350:
        raise Phase4GateError("The frozen Phase 3 registry no longer contains exactly 350 chunks")

    tokenizer = load_phase4_tokenizer()
    candidate, sections_by_doc = build_fixed_chunk_section_map(
        project_root=project_root,
        tokenizer=tokenizer,
        chunks=chunks,
        manifest_rows=manifest_rows,
        frozen_hashes=frozen,
    )
    benchmark_audit = validate_section_benchmark(
        questions=questions,
        manifest_rows=manifest_rows,
        sections_by_doc=sections_by_doc,
        project_root=project_root,
    )
    map_path = project_root / MAP_PATH
    if map_path.is_file():
        existing = json.loads(map_path.read_text(encoding="utf-8"))
        if not isinstance(existing, dict):
            raise Phase4GateError("Existing Phase 4A section map is not a JSON object")
        _validate_existing_map(existing, candidate)
        section_map = existing
    else:
        atomic_write_json(map_path, candidate)
        section_map = candidate
    _map_by_chunk(section_map)
    if sha256_file(benchmark_path) != SECTION_BENCHMARK_FROZEN_SHA256:
        raise Phase4GateError("Section benchmark changed during offline attribution")
    if verify_approved_frozen_artifacts(project_root) != frozen:
        raise Phase4GateError("A frozen Phase 1-3 artifact changed during offline attribution")
    return frozen, benchmark_audit, questions, manifest_rows, chunks, section_map


def run_phase4a(
    *,
    project_root: Path,
    offline_only: bool = False,
    secret_path: Path | None = None,
) -> dict[str, Any]:
    project_root = project_root.resolve()
    frozen, benchmark_audit, questions, manifest_rows, chunks, section_map = prepare_offline_phase4a(
        project_root
    )
    section_map_sha256 = sha256_file(project_root / MAP_PATH)
    offline_summary = {
        "status": "offline_ready" if offline_only else "offline_gate_passed",
        "section_benchmark_sha256": SECTION_BENCHMARK_FROZEN_SHA256,
        "section_benchmark_audit": benchmark_audit,
        "section_map_sha256": section_map_sha256,
        "section_map_summary": section_map["summary"],
        "network_calls_made": 0,
    }
    if offline_only:
        return offline_summary

    config = load_phase3_config(secret_path=secret_path, project_root=project_root)
    results = evaluate_fixed_sections(
        project_root=project_root,
        config=config,
        questions=questions,
        benchmark_audit=benchmark_audit,
        manifest_rows=manifest_rows,
        chunks=chunks,
        section_map=section_map,
        section_map_sha256=section_map_sha256,
        frozen_before=frozen,
    )
    atomic_write_json(project_root / RESULTS_PATH, results)
    write_summary_csv(project_root / SUMMARY_PATH, results)
    write_failure_analysis(project_root / FAILURES_PATH, results)

    if verify_approved_frozen_artifacts(project_root) != frozen:
        raise Phase4GateError("A frozen Phase 1-3 artifact changed while writing Phase 4A reports")
    if sha256_file(project_root / SECTION_BENCHMARK_PATH) != SECTION_BENCHMARK_FROZEN_SHA256:
        raise Phase4GateError("The section benchmark changed while writing Phase 4A reports")
    if sha256_file(project_root / MAP_PATH) != section_map_sha256:
        raise Phase4GateError("The attribution map changed while writing Phase 4A reports")
    return results


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Phase 4A fixed-chunk section baseline")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="NoticeLens project root",
    )
    parser.add_argument(
        "--secret-file",
        type=Path,
        default=None,
        help="Optional explicit dotenv path outside the project workspace",
    )
    parser.add_argument(
        "--offline-only",
        action="store_true",
        help="Validate/freeze the benchmark and attribution map without provider calls",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    try:
        result = run_phase4a(
            project_root=args.project_root,
            offline_only=args.offline_only,
            secret_path=args.secret_file,
        )
    except (Phase4GateError, ProviderGateError, ConfigurationError) as exc:
        print(f"Phase 4A STOP: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # Never print a potentially credential-bearing exception message.
        print(
            f"Phase 4A STOP: unexpected {type(exc).__name__}; no credentials were logged",
            file=sys.stderr,
        )
        return 2
    if args.offline_only:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        metrics = result["metrics"]["section_retrieval"]["overall"]
        print(
            "Phase 4A complete: "
            f"Section P@1={metrics['section_precision_at_1']:.4f}, "
            f"MRR={metrics['section_mrr']:.4f}, "
            f"Hit@5={metrics['section_hit_at_5']:.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
