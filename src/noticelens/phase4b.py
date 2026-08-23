"""Phase 4B: heading-aware dense section retrieval experiment."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import re
import statistics
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pinecone import Pinecone

from .chunking import (
    ChunkRecord,
    atomic_write_json,
    atomic_write_text,
    load_chunk_registry,
    nearest_rank_percentile,
    sha256_file,
)
from .config import ConfigurationError, load_phase3_config
from .heading_chunking import (
    HEADING_CHUNK_STRATEGY,
    HEADING_PINECONE_METADATA_KEYS,
    HeadingChunkRecord,
    HeadingChunkingGateError,
    build_heading_chunks,
    iter_heading_jsonl,
    load_heading_registry,
    load_heading_tokenizer,
    write_heading_audit,
    write_heading_registry,
)
from .phase4a import (
    BENCHMARK_FIELDS,
    MEANINGFUL_GAIN_ABSOLUTE_POINTS,
    NULL_FIXED_PRECISION_THRESHOLD,
    NULL_IMPROVEMENT_LESS_THAN_POINTS,
    SectionScore,
    aggregate_section_scores,
    load_section_questions,
    normalized_path,
    score_section_ranking,
)
from .providers import (
    EMBEDDING_DIMENSION,
    EMBEDDING_MODEL,
    INDEX_CLOUD,
    INDEX_METRIC,
    INDEX_NAME,
    INDEX_REGION,
    NAMESPACE as BASELINE_NAMESPACE,
    NebiusEmbeddings,
    ProviderGateError,
    finite_score,
    latency_summary,
)


HEADING_NAMESPACE = "heading-aware-dense"
TOP_K = 5
EMBEDDING_BATCH_SIZE = 16
EXPECTED_FIXED_VECTOR_COUNT = 350
EXPECTED_HEADING_CHUNK_COUNT = 580

HEADING_REGISTRY_PATH = Path("data/derived/phase4b/heading_aware_220_40_chunks.jsonl")
CHUNK_AUDIT_PATH = Path("reports/phase4b_chunk_audit.json")
RESULTS_PATH = Path("reports/phase4b_heading_results.json")
SUMMARY_PATH = Path("reports/phase4b_heading_summary.csv")
COMPARISON_PATH = Path("reports/phase4b_comparison.csv")
FAILURES_PATH = Path("reports/phase4b_failure_analysis.md")
SECTION_QUESTIONS_PATH = Path("eval/section_questions.json")
FIXED_RESULTS_PATH = Path("reports/phase4_fixed_section_results.json")
FIXED_MAP_PATH = Path("reports/phase4_fixed_chunk_section_map.json")
FIXED_REGISTRY_PATH = Path("data/derived/phase3/fixed_220_40_chunks.jsonl")

FROZEN_FILE_HASHES = {
    "data/corpus_manifest.csv": "f3c7c2e257a6f1dc17bf8e55ab745702299f2af2a5faf9623d390778045ea1a1",
    "data/sample_notice_manifest.csv": "0a92688fe008ee7fbc10fb5ec4b41733ecad250a47110335cc65e3d72dc99769",
    "reports/phase1_acquisition_ledger.csv": "96ab9daf63a043052027f4af76ac7be625d6ceb025d2d81f0c99738e0c1c4cb8",
    "reports/phase1_quality_report.json": "95aefed07a14fe75acfc4be45cb0335434efff5d55225e1d31481f43d025af3c",
    "eval/golden_questions.json": "a5e12ae768b8d43250fac99198efba35bd2d7f5db640e3aba1e8d6958920f391",
    "reports/phase2_eval_manifest.csv": "f73453102fbbe0cc2e18e214e2926b40c252830b9c4eed92792b2d01cca27ea9",
    "reports/evaluation_plan.md": "5065857f4c568e6915ed186467d0c6e55442df2682ec8e3b9224e4eeb2dce52e",
    "data/derived/phase3/fixed_220_40_chunks.jsonl": "4752049f3c435d79a83b5950d0882ce2ac61bb57a5e210d398b529306a9ab709",
    "reports/phase3_chunk_audit.json": "dfe675d414c5e5e1fa0cdd4f25cb30b72cc7e985e03e383057b1e4cd91b95746",
    "reports/phase3_indexing_stats.json": "329e4c0cfc569d5d13561e16f385ea82a512f8b44c98b6e5cc1806db7a5ffb7b",
    "reports/phase3_baseline_results.json": "1af861aaa152f0cff7c87f7c3496bd825eb9e300b524aa1b5f0a5f4a4104d338",
    "reports/phase3_baseline_summary.csv": "f5f5466b67d9912d37f647e1784a369eded30ca4ba77bad15c2b9e79cb2de5e1",
    "reports/phase3_failure_analysis.md": "a508397e16192c7fb8e447f206018659c300d06cc93e84c6e448b74794255abd",
    "eval/section_questions.json": "1090c8b41f0b007adfda1eb9882b0237d93416a3ce57857bc4da58a8947aafa8",
    "reports/phase4_fixed_chunk_section_map.json": "ae919b13f37fbd6ba2a092d369bdfcf899a655f54c0c91ebd83c7b42eaf3f3d4",
    "reports/phase4_fixed_section_results.json": "e3d86aa5390549da47e6b3dc1bf0ea220baa338b30353b8117d0b4743cc1df36",
    "reports/phase4_fixed_section_summary.csv": "556e94fa1e957011efe08b34ac08ad94e2e5bb725be6c8c6a19726a7e225c322",
    "reports/phase4_fixed_section_failures.md": "3022d0baf78c47cb5809c153a52a0efa9ea681afb210ce4a785562e3dbf13ee8",
}
FROZEN_TREE_HASHES = {
    "data/raw/guidance": (50, "53265c67d68fb34354ad186b6a7157093cfde6ea00ac491da2dab8618f2a2a65"),
    "data/raw/sample_notices": (8, "b4f901bd79e0d290ff6095ac2cdc2a0d6f3cb818aa5557edd14773809804106b"),
    "data/processed/guidance": (50, "36cf7f0c8a01879062a328dc009d6c6de0f4324f02f95450468b448e391ebac2"),
}
FROZEN_COMPOSITE_SHA256 = "2561f0d6d9b3f781b6f0a77007c36f7c642c50821ee8495a0cb4087b5e701f9c"
EXPECTED_BASELINE_ID_SET_SHA256 = "e218c17fd6e5d98eb7c9c82c1b95e08aa9224d561feebc7ce729f6adab6e741e"
EXPECTED_HEADING_REGISTRY_SHA256 = "3aecf5db7ee5fe857bdb99156c9bb5ba585f2e845aa6697cc4ece8902ac27572"
EXPECTED_HEADING_AUDIT_SHA256 = "88265f578c3f65b9eae8a42b2ab4c52e1e295f87c1f0f679c5169a427869cc4e"
EXPECTED_HEADING_ID_SET_SHA256 = "60bbf276bc752bfaf8f1e3ca8904fc88d688ce07e8c32b4224dc1e667b26912e"

FIXED_METRICS = {
    "section_precision_at_1": 0.8,
    "section_mrr": 0.8688888888888889,
    "section_hit_at_5": 1.0,
    "correct_at_1": 12,
    "n": 15,
}
SPECIAL_FIXED_RANKS = {"S03": 2, "S07": 5, "S11": 3}
PAIR_CLASSIFICATION_RULE = (
    "IMPROVED when fixed correct_at_1=0 and heading correct_at_1=1; "
    "REGRESSED when fixed=1 and heading=0; otherwise UNCHANGED"
)
RANK_CHANGE_RULE = "effective fixed rank minus effective heading rank; a top-5 miss has effective rank 6"


class Phase4BGateError(RuntimeError):
    """Raised when a Phase 4B freeze, indexing, or evaluation gate fails."""


def utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _tree_digest(directory: Path) -> tuple[str, int]:
    files = sorted((path for path in directory.rglob("*") if path.is_file()), key=lambda path: path.as_posix())
    records = [
        f"{path.relative_to(directory).as_posix()}|{sha256_file(path)}" for path in files
    ]
    return hashlib.sha256("\n".join(records).encode("utf-8")).hexdigest(), len(files)


def verify_frozen_artifacts(project_root: Path) -> dict[str, Any]:
    observed: dict[str, Any] = {}
    failures: list[str] = []
    composite_records: list[str] = []
    for relative, expected_hash in FROZEN_FILE_HASHES.items():
        path = project_root / relative
        if not path.is_file():
            failures.append(f"missing:{relative}")
            continue
        actual = sha256_file(path)
        observed[relative] = actual
        composite_records.append(f"{relative}|{actual}")
        if actual != expected_hash:
            failures.append(f"hash:{relative}")
    for relative, (expected_count, expected_hash) in FROZEN_TREE_HASHES.items():
        actual_hash, actual_count = _tree_digest(project_root / relative)
        observed[relative] = {"file_count": actual_count, "tree_sha256": actual_hash}
        composite_records.append(f"{relative}|tree|{actual_count}|{actual_hash}")
        if actual_count != expected_count or actual_hash != expected_hash:
            failures.append(f"tree:{relative}")
    composite = hashlib.sha256("\n".join(composite_records).encode("utf-8")).hexdigest()
    observed["approved_composite_sha256"] = composite
    if composite != FROZEN_COMPOSITE_SHA256:
        failures.append("approved_composite")
    if failures:
        raise Phase4BGateError("Frozen Phase 1-4A gate failed: " + ", ".join(failures))
    return observed


def _read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    required = {"doc_id", "notice_code", "notice_family", "title", "source_url", "source_origin"}
    if len(rows) != 50 or any(not required.issubset(row) for row in rows):
        raise Phase4BGateError("Frozen manifest schema/count is invalid")
    return rows


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _safe_provider_failure(stage: str, exc: BaseException) -> ProviderGateError:
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
    values = page.get("vectors") or page.get("ids") or [] if isinstance(page, Mapping) else _field(page, "vectors", page)
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
class Phase4BIndexState:
    index: Any
    ready: bool
    dimension: int
    metric: str
    vector_type: str
    cloud: str
    region: str


class HeadingAwarePineconeStore:
    """Existing-index client that can write only the new heading namespace."""

    def __init__(self, *, api_key: str, client: Any | None = None) -> None:
        if not isinstance(api_key, str) or not api_key.strip():
            raise ProviderGateError("PINECONE_API_KEY is empty")
        if HEADING_NAMESPACE in {"", "__default__", BASELINE_NAMESPACE}:
            raise ProviderGateError("Heading-aware namespace is not safely isolated")
        try:
            self._pc = client if client is not None else Pinecone(api_key=api_key)
        except Exception as exc:
            raise _safe_provider_failure("Pinecone Phase 4B client construction", exc) from None
        self.state: Phase4BIndexState | None = None
        self.provider_response_reordered_query_count = 0

    def __repr__(self) -> str:
        return f"HeadingAwarePineconeStore(index_name={INDEX_NAME!r}, namespace={HEADING_NAMESPACE!r})"

    def require_existing_index(self) -> Phase4BIndexState:
        try:
            if not bool(self._pc.indexes.exists(INDEX_NAME)):
                raise ProviderGateError("The frozen Pinecone index is missing; Phase 4B will not create it")
            description = self._pc.indexes.describe(INDEX_NAME)
        except ProviderGateError:
            raise
        except Exception as exc:
            raise _safe_provider_failure("Pinecone Phase 4B index verification", exc) from None
        dimension = int(_field(description, "dimension", 0) or 0)
        metric = str(_field(description, "metric", "")).lower()
        vector_type = str(_field(description, "vector_type", "dense") or "dense").lower()
        status = _field(description, "status", {}) or {}
        ready = bool(_field(status, "ready", False))
        host = str(_field(description, "host", "") or "")
        spec = _field(description, "spec", {}) or {}
        serverless = _field(spec, "serverless", {}) or {}
        cloud = str(_field(serverless, "cloud", "") or "").lower()
        region = str(_field(serverless, "region", "") or "").lower()
        if dimension != EMBEDDING_DIMENSION or metric != INDEX_METRIC or vector_type not in {"", "dense"}:
            raise ProviderGateError("Existing Pinecone index is incompatible; it was not modified")
        if cloud != INDEX_CLOUD or region != INDEX_REGION:
            raise ProviderGateError("Existing Pinecone index location differs from frozen Phase 3; it was not modified")
        if not ready or not host:
            raise ProviderGateError("Existing Pinecone index is not ready")
        try:
            index = self._pc.index(host=host)
        except Exception as exc:
            raise _safe_provider_failure("Pinecone Phase 4B data-plane connection", exc) from None
        self.state = Phase4BIndexState(
            index=index,
            ready=ready,
            dimension=dimension,
            metric=metric,
            vector_type=vector_type or "dense",
            cloud=cloud,
            region=region,
        )
        return self.state

    @property
    def index(self) -> Any:
        if self.state is None:
            raise ProviderGateError("Pinecone index has not passed the Phase 4B gate")
        return self.state.index

    def _namespace_components(self, namespace: str) -> tuple[int, set[str]]:
        if namespace not in {BASELINE_NAMESPACE, HEADING_NAMESPACE}:
            raise ProviderGateError("Phase 4B refused an unauthorized namespace")
        try:
            count = _namespace_count(self.index.describe_index_stats(), namespace)
            ids: set[str] = set()
            for page in self.index.list(namespace=namespace):
                for vector_id in _flatten_list_page(page):
                    if vector_id in ids:
                        raise ProviderGateError(f"Pinecone namespace {namespace!r} listed duplicate IDs")
                    ids.add(vector_id)
        except ProviderGateError:
            raise
        except Exception as exc:
            raise _safe_provider_failure(f"Pinecone snapshot for {namespace}", exc) from None
        return count, ids

    def namespace_snapshot(self, namespace: str) -> tuple[int, set[str]]:
        count, ids = self._namespace_components(namespace)
        if count != len(ids):
            raise ProviderGateError(
                f"Pinecone namespace {namespace!r} count/list mismatch: {count} vs {len(ids)}"
            )
        return count, ids

    def preflight_target(self, expected_ids: set[str]) -> dict[str, int]:
        if not expected_ids:
            raise ProviderGateError("Heading-aware expected ID set is empty")
        count, ids = self.namespace_snapshot(HEADING_NAMESPACE)
        if count or ids:
            raise ProviderGateError(
                "The new heading-aware namespace is not empty; nothing was overwritten, deleted, or written"
            )
        return {"preexisting_vector_count": 0, "preexisting_expected_ids": 0}

    def upsert_heading_batch(
        self,
        chunks: Sequence[HeadingChunkRecord],
        vectors: Sequence[Sequence[float]],
    ) -> int:
        if not chunks or len(chunks) != len(vectors):
            raise ProviderGateError("Heading chunk/vector batch is empty or mismatched")
        records: list[dict[str, Any]] = []
        for chunk, vector in zip(chunks, vectors, strict=True):
            _validate_vector(vector)
            metadata = {key: chunk.metadata[key] for key in HEADING_PINECONE_METADATA_KEYS}
            if tuple(metadata) != HEADING_PINECONE_METADATA_KEYS:
                raise ProviderGateError(f"Unexpected heading metadata schema for {chunk.chunk_id}")
            records.append({"id": chunk.chunk_id, "values": list(vector), "metadata": metadata})
        try:
            response = self.index.upsert(vectors=records, namespace=HEADING_NAMESPACE)
        except Exception as exc:
            raise _safe_provider_failure("Pinecone heading-aware upsert", exc) from None
        raw_count = _field(response, "upserted_count")
        if isinstance(raw_count, bool) or not isinstance(raw_count, int) or raw_count != len(records):
            raise ProviderGateError("Pinecone heading-aware upsert count mismatch")
        return raw_count

    def wait_for_target_parity(
        self,
        expected_ids: set[str],
        *,
        timeout_seconds: float = 180.0,
        poll_seconds: float = 3.0,
    ) -> tuple[int, set[str]]:
        if timeout_seconds <= 0 or poll_seconds <= 0:
            raise ValueError("Parity timeout and poll interval must be positive")
        deadline = time.monotonic() + timeout_seconds
        last_count = -1
        while time.monotonic() < deadline:
            count, ids = self._namespace_components(HEADING_NAMESPACE)
            last_count = count
            if count == len(ids) == len(expected_ids) and ids == expected_ids:
                return count, ids
            time.sleep(poll_seconds)
        raise ProviderGateError(
            f"Heading namespace parity timed out: expected {len(expected_ids)}, observed {last_count}"
        )

    def query_known_notice(
        self,
        vector: Sequence[float],
        *,
        notice_code: str,
        eligible_chunk_count: int,
    ) -> list[Any]:
        _validate_vector(vector)
        expected_returned = min(TOP_K, eligible_chunk_count)
        if not notice_code.strip() or expected_returned <= 0:
            raise ProviderGateError("Heading query has no valid notice code or eligible chunks")
        try:
            response = self.index.query(
                namespace=HEADING_NAMESPACE,
                vector=list(vector),
                top_k=TOP_K,
                filter={"notice_code": {"$eq": notice_code}},
                include_metadata=True,
                include_values=False,
            )
        except Exception as exc:
            raise _safe_provider_failure("Pinecone heading-aware filtered query", exc) from None
        matches = list(_field(response, "matches", []) or [])
        if len(matches) != expected_returned:
            raise ProviderGateError(
                f"Pinecone returned {len(matches)} heading matches for {notice_code!r}; expected {expected_returned}"
            )
        ids = [str(_field(match, "id", "") or "") for match in matches]
        scores = [finite_score(_field(match, "score")) for match in matches]
        if any(not value for value in ids) or len(set(ids)) != len(ids):
            raise ProviderGateError("Heading query returned blank or duplicate IDs")
        ordered = sorted(
            zip(matches, ids, scores, strict=True),
            key=lambda item: (-item[2], item[1]),
        )
        ordered_ids = [item[1] for item in ordered]
        if ordered_ids != ids:
            self.provider_response_reordered_query_count += 1
        # Pinecone documents query matches as similarity-ranked. Normalize any
        # transient response-order anomaly to that provider score (with a stable
        # ID tie-break); this does not introduce a second scorer or reranker.
        return [item[0] for item in ordered]


def _safe_metadata(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    try:
        return dict(value)
    except (TypeError, ValueError):
        raise Phase4BGateError("Pinecone returned malformed metadata") from None


def _text_preview(text: str, limit: int = 360) -> str:
    return re.sub(r"\s+", " ", text).strip()[:limit]


def _id_set_sha256(ids: set[str]) -> str:
    return hashlib.sha256("\n".join(sorted(ids)).encode("utf-8")).hexdigest()


def _audit_comparable(value: Mapping[str, Any]) -> dict[str, Any]:
    return {key: item for key, item in value.items() if key != "generated_at_utc"}


def prepare_heading_chunks(
    project_root: Path,
) -> tuple[dict[str, Any], list[HeadingChunkRecord], dict[str, Any]]:
    frozen = verify_frozen_artifacts(project_root)
    tokenizer = load_heading_tokenizer()
    candidate_chunks, candidate_audit = build_heading_chunks(
        project_root=project_root,
        tokenizer=tokenizer,
        frozen_inputs=frozen,
    )
    if len(candidate_chunks) != EXPECTED_HEADING_CHUNK_COUNT:
        raise Phase4BGateError(
            f"Heading-aware chunk count changed: expected {EXPECTED_HEADING_CHUNK_COUNT}, observed {len(candidate_chunks)}"
        )
    registry_path = project_root / HEADING_REGISTRY_PATH
    candidate_registry = "\n".join(iter_heading_jsonl(candidate_chunks)) + "\n"
    if registry_path.is_file():
        if registry_path.read_text(encoding="utf-8") != candidate_registry:
            raise Phase4BGateError("Existing heading-aware registry differs from deterministic reconstruction")
    else:
        write_heading_registry(registry_path, candidate_chunks)
    chunks = load_heading_registry(registry_path)
    if [chunk.as_dict() for chunk in chunks] != [chunk.as_dict() for chunk in candidate_chunks]:
        raise Phase4BGateError("Saved heading registry does not round-trip exactly")
    candidate_audit["outputs"] = {
        "heading_registry_path": HEADING_REGISTRY_PATH.as_posix(),
        "heading_registry_sha256": sha256_file(registry_path),
    }
    audit_path = project_root / CHUNK_AUDIT_PATH
    if audit_path.is_file():
        existing = json.loads(audit_path.read_text(encoding="utf-8"))
        if not isinstance(existing, dict) or _audit_comparable(existing) != _audit_comparable(candidate_audit):
            raise Phase4BGateError("Existing Phase 4B chunk audit differs from deterministic reconstruction")
        audit = existing
    else:
        write_heading_audit(audit_path, candidate_audit)
        audit = candidate_audit
    if not audit.get("quality_gate_passed"):
        raise Phase4BGateError("Heading-aware chunk audit did not pass")
    if verify_frozen_artifacts(project_root) != frozen:
        raise Phase4BGateError("A frozen artifact changed during heading-aware chunk preparation")
    return frozen, chunks, audit


def load_and_validate_fixed_results(
    project_root: Path,
    questions: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    fixed = json.loads((project_root / FIXED_RESULTS_PATH).read_text(encoding="utf-8"))
    fixed_map = json.loads((project_root / FIXED_MAP_PATH).read_text(encoding="utf-8"))
    fixed_chunks = load_chunk_registry(project_root / FIXED_REGISTRY_PATH)
    fixed_chunks_by_id = {chunk.chunk_id: chunk for chunk in fixed_chunks}
    map_records = fixed_map.get("chunks", []) if isinstance(fixed_map, dict) else []
    fixed_map_by_id = {
        record["chunk_id"]: record for record in map_records if isinstance(record, dict) and "chunk_id" in record
    }
    if (
        len(fixed_chunks_by_id) != EXPECTED_FIXED_VECTOR_COUNT
        or len(fixed_map_by_id) != EXPECTED_FIXED_VECTOR_COUNT
        or set(fixed_chunks_by_id) != set(fixed_map_by_id)
    ):
        raise Phase4BGateError("Frozen Phase 4A map and fixed registry do not reconcile")
    if not isinstance(fixed, dict) or len(fixed.get("queries", [])) != 15:
        raise Phase4BGateError("Frozen Phase 4A results schema/count is invalid")
    questions_by_id = {question["id"]: question for question in questions}
    fixed_by_id: dict[str, dict[str, Any]] = {}
    scores: list[SectionScore] = []
    for trace in fixed["queries"]:
        question_id = trace.get("id")
        if question_id not in questions_by_id or question_id in fixed_by_id:
            raise Phase4BGateError("Frozen Phase 4A trace IDs are missing or duplicated")
        question = questions_by_id[question_id]
        if any(trace.get(field) != question[field] for field in BENCHMARK_FIELDS):
            raise Phase4BGateError(f"Frozen Phase 4A trace differs from benchmark for {question_id}")
        ranks = trace.get("ranks")
        if not isinstance(ranks, list) or [rank.get("rank") for rank in ranks] != list(range(1, 6)):
            raise Phase4BGateError(f"Frozen Phase 4A ranks are invalid for {question_id}")
        for rank in ranks:
            chunk_id = rank.get("chunk_id")
            if chunk_id not in fixed_chunks_by_id:
                raise Phase4BGateError(f"Frozen Phase 4A trace contains unknown chunk {chunk_id}")
            chunk = fixed_chunks_by_id[chunk_id]
            map_record = fixed_map_by_id[chunk_id]
            expected_paths = [entry["path"] for entry in map_record["heading_paths"]]
            if (
                rank.get("doc_id") != chunk.metadata["doc_id"]
                or rank.get("retrieved_notice_code") != chunk.metadata["notice_code"]
                or rank.get("title") != chunk.metadata["title"]
                or rank.get("attributed_heading_paths") != expected_paths
                or rank.get("text_preview") != _text_preview(chunk.text)
            ):
                raise Phase4BGateError(f"Frozen Phase 4A rank evidence does not reconcile for {chunk_id}")
        score = score_section_ranking(
            question["expected_notice_code"], question["expected_heading_path"], ranks
        )
        stored = trace.get("section_score", {})
        if stored != {
            "precision_at_1": score.precision_at_1,
            "reciprocal_rank": score.reciprocal_rank,
            "hit_at_5": score.hit_at_5,
            "first_correct_rank": score.first_correct_rank,
        }:
            raise Phase4BGateError(f"Frozen Phase 4A score mismatch for {question_id}")
        scores.append(score)
        fixed_by_id[question_id] = trace
    aggregate = aggregate_section_scores(scores)
    if (
        aggregate["n"] != FIXED_METRICS["n"]
        or aggregate["correct_at_1"] != FIXED_METRICS["correct_at_1"]
        or not math.isclose(aggregate["section_precision_at_1"], FIXED_METRICS["section_precision_at_1"])
        or not math.isclose(aggregate["section_mrr"], FIXED_METRICS["section_mrr"])
        or not math.isclose(aggregate["section_hit_at_5"], FIXED_METRICS["section_hit_at_5"])
    ):
        raise Phase4BGateError("Frozen Phase 4A aggregate metrics changed")
    for question_id, expected_rank in SPECIAL_FIXED_RANKS.items():
        if fixed_by_id[question_id]["section_score"]["first_correct_rank"] != expected_rank:
            raise Phase4BGateError(f"Frozen special-case rank changed for {question_id}")
    return fixed, fixed_by_id


def _trace_heading_query(
    *,
    question: dict[str, Any],
    manifest_row: dict[str, str],
    vector: Sequence[float],
    store: HeadingAwarePineconeStore,
    chunks_by_id: dict[str, HeadingChunkRecord],
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
        if chunk_id not in eligible_ids or chunk_id not in chunks_by_id:
            raise Phase4BGateError(f"Heading query returned an unknown or wrong-notice ID for {question['id']}")
        chunk = chunks_by_id[chunk_id]
        metadata = _safe_metadata(_field(match, "metadata", {}))
        if set(metadata) != set(HEADING_PINECONE_METADATA_KEYS):
            raise Phase4BGateError(f"Heading query metadata schema is invalid for {question['id']}")
        expected_metadata = {key: chunk.metadata[key] for key in HEADING_PINECONE_METADATA_KEYS}
        if metadata != expected_metadata:
            raise Phase4BGateError(f"Heading query metadata differs from local registry for {chunk_id}")
        if metadata["notice_code"] != question["expected_notice_code"] or metadata["doc_id"] != question["expected_doc_id"]:
            raise Phase4BGateError(f"Notice-code filter returned the wrong document for {question['id']}")
        heading_path = list(metadata["heading_path"])
        notice_match = metadata["notice_code"] == question["expected_notice_code"]
        section_match = normalized_path(heading_path) == normalized_path(question["expected_heading_path"])
        rank_records.append(
            {
                "rank": rank,
                "chunk_id": chunk_id,
                "doc_id": metadata["doc_id"],
                "retrieved_notice_code": metadata["notice_code"],
                "title": metadata["title"],
                "heading": metadata["heading"],
                "heading_path": heading_path,
                "section_index": int(metadata["section_index"]),
                "subchunk_index": int(metadata["subchunk_index"]),
                "similarity_score": finite_score(_field(match, "score")),
                "text_preview": _text_preview(chunk.text),
                "attributed_heading_paths": [heading_path],
                "notice_match": notice_match,
                "section_match": section_match,
                "correct": notice_match and section_match,
            }
        )
    score = score_section_ranking(
        question["expected_notice_code"], question["expected_heading_path"], rank_records
    )
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
        "query_latency_seconds": round(latency, 6),
    }
    return trace, latency


def aggregate_heading_traces(traces: Sequence[dict[str, Any]]) -> dict[str, Any]:
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

    if len(traces) != 15 or {trace["id"] for trace in traces} != {f"S{number:02d}" for number in range(1, 16)}:
        raise Phase4BGateError("Heading metrics require exactly S01-S15")
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
        raise Phase4BGateError("Heading metric denominators differ from the frozen benchmark")
    return result


def build_paired_comparison(
    questions: Sequence[dict[str, Any]],
    fixed_by_id: Mapping[str, dict[str, Any]],
    heading_by_id: Mapping[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    rows: list[dict[str, Any]] = []
    for question in questions:
        question_id = question["id"]
        fixed = fixed_by_id[question_id]
        heading = heading_by_id[question_id]
        fixed_rank = fixed["section_score"]["first_correct_rank"]
        heading_rank = heading["section_score"]["first_correct_rank"]
        fixed_correct = int(fixed["section_score"]["precision_at_1"])
        heading_correct = int(heading["section_score"]["precision_at_1"])
        if fixed_correct == 0 and heading_correct == 1:
            classification = "IMPROVED"
        elif fixed_correct == 1 and heading_correct == 0:
            classification = "REGRESSED"
        else:
            classification = "UNCHANGED"
        effective_fixed = TOP_K + 1 if fixed_rank is None else int(fixed_rank)
        effective_heading = TOP_K + 1 if heading_rank is None else int(heading_rank)
        rows.append(
            {
                "question_id": question_id,
                "question": question["question"],
                "expected_notice": question["expected_notice_code"],
                "expected_heading": question["expected_heading"],
                "expected_heading_path": question["expected_heading_path"],
                "fixed_rank": fixed_rank,
                "heading_aware_rank": heading_rank,
                "fixed_correct_at_1": fixed_correct,
                "heading_correct_at_1": heading_correct,
                "rank_change": effective_fixed - effective_heading,
                "classification": classification,
                "fixed_top_1_preview": fixed["ranks"][0]["text_preview"],
                "heading_aware_top_1_preview": heading["ranks"][0]["text_preview"],
                "fixed_top_1_paths": fixed["ranks"][0]["attributed_heading_paths"],
                "heading_aware_top_1_path": heading["ranks"][0]["heading_path"],
            }
        )
    counts = Counter(row["classification"] for row in rows)
    result_counts = {name.lower(): counts[name] for name in ("IMPROVED", "UNCHANGED", "REGRESSED")}
    if result_counts["improved"] - result_counts["regressed"] != (
        sum(row["heading_correct_at_1"] for row in rows) - sum(row["fixed_correct_at_1"] for row in rows)
    ):
        raise Phase4BGateError("Paired classifications do not reconcile with P@1 change")
    return rows, result_counts


def evaluate_and_index(
    *,
    project_root: Path,
    config: Any,
    frozen_before: dict[str, Any],
    chunks: list[HeadingChunkRecord],
    chunk_audit: dict[str, Any],
    questions: list[dict[str, Any]],
    fixed_results: dict[str, Any],
    fixed_by_id: dict[str, dict[str, Any]],
    resume_query_only: bool = False,
) -> dict[str, Any]:
    fixed_chunks = load_chunk_registry(project_root / FIXED_REGISTRY_PATH)
    fixed_ids = {chunk.chunk_id for chunk in fixed_chunks}
    if len(fixed_ids) != EXPECTED_FIXED_VECTOR_COUNT or _id_set_sha256(fixed_ids) != EXPECTED_BASELINE_ID_SET_SHA256:
        raise Phase4BGateError("Frozen local fixed-chunk ID set is invalid")
    chunks_by_id = {chunk.chunk_id: chunk for chunk in chunks}
    if len(chunks_by_id) != EXPECTED_HEADING_CHUNK_COUNT:
        raise Phase4BGateError("Heading-aware IDs are not unique or complete")
    expected_heading_ids = set(chunks_by_id)
    if _id_set_sha256(expected_heading_ids) != EXPECTED_HEADING_ID_SET_SHA256:
        raise Phase4BGateError("Heading-aware local ID set differs from the pre-index freeze")

    store = HeadingAwarePineconeStore(api_key=config.pinecone_api_key)
    state = store.require_existing_index()
    baseline_before_count, baseline_before_ids = store.namespace_snapshot(BASELINE_NAMESPACE)
    if baseline_before_count != EXPECTED_FIXED_VECTOR_COUNT or baseline_before_ids != fixed_ids:
        raise Phase4BGateError("Baseline namespace differs from the frozen 350-vector registry")
    if resume_query_only:
        target_initial_count, target_initial_ids = store.namespace_snapshot(HEADING_NAMESPACE)
        if target_initial_count != len(expected_heading_ids) or target_initial_ids != expected_heading_ids:
            raise Phase4BGateError(
                "Query-only recovery requires exact 580-vector heading namespace parity; nothing was modified"
            )
        target_preflight_initial = {
            "existing_vector_count": target_initial_count,
            "exact_expected_id_parity": True,
            "mutation_allowed": False,
        }
    else:
        target_preflight_initial = store.preflight_target(expected_heading_ids)
    if [question.get("id") for question in questions] != [f"S{number:02d}" for number in range(1, 16)]:
        raise Phase4BGateError("Phase 4B questions must be embedded in exact frozen S01-S15 order")
    registry_hash_before = sha256_file(project_root / HEADING_REGISTRY_PATH)
    audit_hash_before = sha256_file(project_root / CHUNK_AUDIT_PATH)

    embedder = NebiusEmbeddings(
        api_key=config.nebius_api_key,
        base_url=config.nebius_base_url,
        batch_size=EMBEDDING_BATCH_SIZE,
    )
    if resume_query_only:
        chunk_vectors: list[list[float]] = []
        chunk_embedding_latencies: list[float] = []
        embedded_count = 0
    else:
        chunk_embedding_latency_start = len(embedder.request_latencies)
        chunk_vectors = embedder.embed_documents([chunk.text for chunk in chunks])
        chunk_embedding_latencies = embedder.request_latencies[chunk_embedding_latency_start:]
        embedded_count = len(chunk_vectors)
        if embedded_count != len(chunks):
            raise Phase4BGateError("Heading-aware embedding count mismatch")

    question_latency_start = len(embedder.request_latencies)
    question_vectors = embedder.embed_documents([question["question"] for question in questions])
    question_embedding_latencies = embedder.request_latencies[question_latency_start:]
    if len(question_vectors) != 15:
        raise Phase4BGateError("Nebius did not return exactly 15 question embeddings")

    if verify_frozen_artifacts(project_root) != frozen_before:
        raise Phase4BGateError("A frozen Phase 1-4A artifact changed before the Phase 4B remote action")
    if (
        sha256_file(project_root / HEADING_REGISTRY_PATH) != registry_hash_before
        or sha256_file(project_root / CHUNK_AUDIT_PATH) != audit_hash_before
    ):
        raise Phase4BGateError("Heading registry or audit changed before the Phase 4B remote action")
    baseline_prewrite_count, baseline_prewrite_ids = store.namespace_snapshot(BASELINE_NAMESPACE)
    if baseline_prewrite_count != baseline_before_count or baseline_prewrite_ids != baseline_before_ids:
        raise Phase4BGateError("Baseline namespace changed before the Phase 4B remote action")
    if resume_query_only:
        target_prewrite_count, target_prewrite_ids = store.namespace_snapshot(HEADING_NAMESPACE)
        if target_prewrite_count != len(expected_heading_ids) or target_prewrite_ids != expected_heading_ids:
            raise Phase4BGateError("Heading namespace changed before query-only recovery")
        target_preflight_prewrite = {
            "existing_vector_count": target_prewrite_count,
            "exact_expected_id_parity": True,
            "mutation_allowed": False,
        }
    else:
        target_preflight_prewrite = store.preflight_target(expected_heading_ids)

    upserted_count = 0
    upsert_latencies: list[float] = []
    if resume_query_only:
        target_count, target_ids = store.namespace_snapshot(HEADING_NAMESPACE)
    else:
        for offset in range(0, len(chunks), EMBEDDING_BATCH_SIZE):
            batch = chunks[offset : offset + EMBEDDING_BATCH_SIZE]
            vectors = chunk_vectors[offset : offset + EMBEDDING_BATCH_SIZE]
            started = time.perf_counter()
            upserted_count += store.upsert_heading_batch(batch, vectors)
            upsert_latencies.append(time.perf_counter() - started)
        if upserted_count != len(chunks):
            raise Phase4BGateError("Heading-aware upsert count mismatch")
        target_count, target_ids = store.wait_for_target_parity(expected_heading_ids)
    if target_count != len(chunks) or target_ids != expected_heading_ids:
        raise Phase4BGateError("Heading-aware Pinecone vector parity failed")
    baseline_after_index_count, baseline_after_index_ids = store.namespace_snapshot(BASELINE_NAMESPACE)
    if baseline_after_index_count != baseline_before_count or baseline_after_index_ids != baseline_before_ids:
        raise Phase4BGateError("Baseline namespace changed during heading-aware indexing")

    manifest_rows = _read_manifest(project_root / "data" / "corpus_manifest.csv")
    manifest_by_doc = {row["doc_id"]: row for row in manifest_rows}
    eligible_by_code: dict[str, set[str]] = defaultdict(set)
    for chunk in chunks:
        eligible_by_code[str(chunk.metadata["notice_code"])].add(chunk.chunk_id)
    traces: list[dict[str, Any]] = []
    query_latencies: list[float] = []
    for question, vector in zip(questions, question_vectors, strict=True):
        trace, query_latency = _trace_heading_query(
            question=question,
            manifest_row=manifest_by_doc[question["expected_doc_id"]],
            vector=vector,
            store=store,
            chunks_by_id=chunks_by_id,
            eligible_ids=eligible_by_code[question["expected_notice_code"]],
        )
        traces.append(trace)
        query_latencies.append(query_latency)

    target_after_count, target_after_ids = store.namespace_snapshot(HEADING_NAMESPACE)
    baseline_after_count, baseline_after_ids = store.namespace_snapshot(BASELINE_NAMESPACE)
    if target_after_count != target_count or target_after_ids != target_ids:
        raise Phase4BGateError("Heading namespace changed unexpectedly during evaluation")
    if baseline_after_count != baseline_before_count or baseline_after_ids != baseline_before_ids:
        raise Phase4BGateError("Baseline namespace changed during Phase 4B evaluation")
    frozen_after = verify_frozen_artifacts(project_root)
    if frozen_after != frozen_before:
        raise Phase4BGateError("A frozen Phase 1-4A artifact changed during Phase 4B")
    if (
        sha256_file(project_root / HEADING_REGISTRY_PATH) != registry_hash_before
        or sha256_file(project_root / CHUNK_AUDIT_PATH) != audit_hash_before
    ):
        raise Phase4BGateError("Heading registry or audit changed during Phase 4B")

    metrics = aggregate_heading_traces(traces)
    heading_by_id = {trace["id"]: trace for trace in traces}
    comparison_rows, classification_counts = build_paired_comparison(
        questions, fixed_by_id, heading_by_id
    )
    heading_overall = metrics["overall"]
    p1_delta_points = (
        heading_overall["section_precision_at_1"] - FIXED_METRICS["section_precision_at_1"]
    ) * 100.0
    mrr_delta = heading_overall["section_mrr"] - FIXED_METRICS["section_mrr"]
    hit_delta = heading_overall["section_hit_at_5"] - FIXED_METRICS["section_hit_at_5"]
    threshold_met = p1_delta_points + 1e-12 >= MEANINGFUL_GAIN_ABSOLUTE_POINTS
    new_regressions = [
        row["question_id"] for row in comparison_rows if row["classification"] == "REGRESSED"
    ]
    heading_p1_failures = [
        trace["id"] for trace in traces if trace["section_score"]["precision_at_1"] == 0
    ]
    execution_mode = (
        "query_only_recovery_after_completed_namespace_population"
        if resume_query_only
        else "initial_embedding_indexing_and_evaluation"
    )
    return {
        "schema_version": "1.0",
        "generated_at_utc": utc_now(),
        "phase": "4B-heading-aware-chunking-experiment",
        "execution": {
            "mode": execution_mode,
            "query_only_recovery": resume_query_only,
            "initial_attempt_reached_exact_heading_id_parity": resume_query_only,
            "initial_attempt_stopped_before_result_reports": resume_query_only,
            "initial_attempt_interruption": (
                "post-index query evaluation stopped when one raw Pinecone response did not satisfy "
                "the then-strict descending-score-order validator"
                if resume_query_only
                else None
            ),
            "initial_indexing_latency_available": not resume_query_only,
            "recovery_evidence": (
                "The existing heading-aware namespace exactly matched all 580 deterministic local IDs "
                "before and after the query-only recovery; no document vectors were embedded or written "
                "during this recovery run."
                if resume_query_only
                else None
            ),
        },
        "experiment_contract": {
            "execution_mode": execution_mode,
            "research_question": "Does respecting IRS heading boundaries improve known-notice section retrieval?",
            "isolated_variable": "chunking strategy",
            "treatment_scope": "logical heading boundaries plus deterministic title/code/path prefix",
            "chunk_strategy": HEADING_CHUNK_STRATEGY,
            "query_policy": "exact frozen question text with exact known notice_code filter only",
            "provider_order_normalization": (
                "returned matches are ordered by descending Pinecone cosine similarity score with "
                "chunk_id as a deterministic tie-break; no second scorer or reranker is used"
            ),
            "top_k": TOP_K,
            "embedding_model_unchanged": True,
            "generation_used": False,
            "bm25_hybrid_reranking_query_rewriting_used": False,
            "pair_classification_rule": PAIR_CLASSIFICATION_RULE,
            "rank_change_rule": RANK_CHANGE_RULE,
        },
        "precommitted_thresholds": {
            "meaningful_p1_gain_absolute_percentage_points": MEANINGFUL_GAIN_ABSOLUTE_POINTS,
            "minimum_heading_p1_from_fixed_80_percent": 0.9,
            "minimum_correct_at_1_with_15_questions_to_meet_threshold": 14,
            "weak_null_fixed_near_ceiling_threshold": NULL_FIXED_PRECISION_THRESHOLD,
            "weak_null_gain_less_than_percentage_points": NULL_IMPROVEMENT_LESS_THAN_POINTS,
            "weak_null_condition_applicable": False,
            "threshold_met": threshold_met,
        },
        "inputs": {
            "approved_frozen_before": frozen_before,
            "approved_frozen_after": frozen_after,
            "section_questions": {
                "path": SECTION_QUESTIONS_PATH.as_posix(),
                "sha256": FROZEN_FILE_HASHES[SECTION_QUESTIONS_PATH.as_posix()],
                "count": 15,
            },
            "frozen_fixed_results": {
                "path": FIXED_RESULTS_PATH.as_posix(),
                "sha256": FROZEN_FILE_HASHES[FIXED_RESULTS_PATH.as_posix()],
                "metrics": FIXED_METRICS,
            },
            "heading_registry": {
                "path": HEADING_REGISTRY_PATH.as_posix(),
                "sha256": registry_hash_before,
                "count": len(chunks),
            },
            "heading_chunk_audit": {
                "path": CHUNK_AUDIT_PATH.as_posix(),
                "sha256": audit_hash_before,
                "quality_gate_passed": chunk_audit["quality_gate_passed"],
            },
        },
        "embedding": {
            "provider": "Nebius Token Factory",
            "model": EMBEDDING_MODEL,
            "dimension": EMBEDDING_DIMENSION,
            "document_chunk_count": len(chunks),
            "document_chunks_embedded": embedded_count,
            "document_chunks_embedded_current_run": embedded_count,
            "document_vectors_reused": len(chunks) if resume_query_only else 0,
            "document_embedding_requests_current_run": len(chunk_embedding_latencies),
            "document_embedding_latency": latency_summary(chunk_embedding_latencies),
            "document_embedding_latency_available": not resume_query_only,
            "question_count_embedded": 15,
            "question_embedding_latency": latency_summary(question_embedding_latencies),
        },
        "pinecone": {
            "index_name": INDEX_NAME,
            "dimension": state.dimension,
            "metric": state.metric,
            "cloud": state.cloud,
            "region": state.region,
            "baseline_namespace": BASELINE_NAMESPACE,
            "heading_namespace": HEADING_NAMESPACE,
            "baseline_before_count": baseline_before_count,
            "baseline_after_index_count": baseline_after_index_count,
            "baseline_after_query_count": baseline_after_count,
            "baseline_exact_id_parity": baseline_before_ids == baseline_after_index_ids == baseline_after_ids == fixed_ids,
            "baseline_id_set_sha256": _id_set_sha256(baseline_after_ids),
            "heading_preflight": {
                "initial": target_preflight_initial,
                "immediately_before_first_write_or_recovery_query": target_preflight_prewrite,
            },
            "heading_local_count": len(chunks),
            "heading_embedded_count": embedded_count,
            "heading_embedded_count_current_run": embedded_count,
            "heading_upserted_count": upserted_count,
            "heading_upserted_count_current_run": upserted_count,
            "heading_upsert_calls_current_run": len(upsert_latencies),
            "heading_namespace_count": target_after_count,
            "heading_exact_id_parity": target_after_ids == expected_heading_ids,
            "heading_id_set_sha256": _id_set_sha256(target_after_ids),
            "upsert_latency": latency_summary(upsert_latencies),
            "upsert_latency_available": not resume_query_only,
            "query_latency": latency_summary(query_latencies),
            "provider_response_reordered_query_count": store.provider_response_reordered_query_count,
            "index_create_calls": 0,
            "baseline_upsert_calls": 0,
            "delete_clear_update_calls": 0,
        },
        "metrics": {"heading_aware_section_retrieval": metrics},
        "direct_comparison": {
            "fixed": FIXED_METRICS,
            "heading_aware": heading_overall,
            "absolute_p1_improvement_percentage_points": p1_delta_points,
            "mrr_improvement": mrr_delta,
            "hit_at_5_change": hit_delta,
            "classification_counts": classification_counts,
            "heading_p1_failure_ids": heading_p1_failures,
            "new_regression_ids": new_regressions,
            "retain_heading_aware_based_on_precommitted_evidence": threshold_met,
        },
        "paired_comparison": comparison_rows,
        "queries": traces,
        "quality_gate_passed": True,
    }


def _summary_rows(results: Mapping[str, Any]) -> list[dict[str, Any]]:
    metrics = results["metrics"]["heading_aware_section_retrieval"]
    rows: list[dict[str, Any]] = []

    def add(scope: str, style: str, family: str, values: Mapping[str, Any]) -> None:
        rows.append(
            {
                "scope": scope,
                "language_style": style,
                "notice_family": family,
                **{key: values[key] for key in (
                    "n", "correct_at_1", "section_precision_at_1", "reciprocal_rank_sum",
                    "section_mrr", "hit_at_5_count", "section_hit_at_5",
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
    fields = (
        "scope", "language_style", "notice_family", "n", "correct_at_1",
        "section_precision_at_1", "reciprocal_rank_sum", "section_mrr",
        "hit_at_5_count", "section_hit_at_5",
    )
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(_summary_rows(results))
    atomic_write_text(path, output.getvalue())


def write_comparison_csv(path: Path, results: Mapping[str, Any]) -> None:
    fields = (
        "question_id", "expected_notice", "expected_heading", "fixed_rank",
        "heading_aware_rank", "fixed_correct_at_1", "heading_correct_at_1",
        "rank_change", "classification",
    )
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    writer.writerows(results["paired_comparison"])
    atomic_write_text(path, output.getvalue())


def _percent(value: float) -> str:
    return f"{value * 100:.1f}%"


def write_failure_analysis(path: Path, results: Mapping[str, Any], chunk_audit: Mapping[str, Any]) -> None:
    comparison = results["direct_comparison"]
    heading_metrics = results["metrics"]["heading_aware_section_retrieval"]
    paired = {row["question_id"]: row for row in results["paired_comparison"]}
    traces = {trace["id"]: trace for trace in results["queries"]}
    lines = [
        "# Phase 4B heading-aware chunking analysis",
        "",
        "This experiment changes only chunk construction. It uses the same corpus, questions, embedding model, index, cosine metric, known-notice filter, and top-5 scoring.",
        "The heading-aware treatment bundles section boundaries with a deterministic structural prefix; results are attributed to the strategy, not boundaries alone.",
        f"Execution mode: `{results['execution']['mode']}`.",
        (
            "This report was completed by a query-only recovery after the initial attempt had populated "
            "the exact 580 deterministic heading IDs. The recovery embedded only the 15 questions and "
            "performed zero document embeddings or upserts. Stored-vector model/upsert provenance is "
            "inherited from that initial attempt; exact remote ID parity is mechanically verified. The "
            "initial attempt stopped during post-index querying when one raw Pinecone response failed the "
            "then-strict descending-score-order check."
            if results["execution"]["query_only_recovery"]
            else "This report was completed during the initial embedding, indexing, and evaluation run."
        ),
        "",
        "## Chunk audit",
        "",
        f"- Heading-aware chunks: {chunk_audit['total_heading_aware_chunks']}",
        f"- Oversized sections split: {chunk_audit['logical_sections']['oversized_sections_requiring_subchunking']}",
        f"- Chunks crossing heading boundaries: {chunk_audit['integrity']['chunks_crossing_heading_boundaries']['count']}",
        f"- Unassigned useful pre-H2 bodies: {chunk_audit['logical_sections']['unassigned_useful_pre_h2_content_count']} (CP2000 series; no heading path was invented)",
        "",
        "## Metric comparison",
        "",
        "| System | n | Section P@1 | Section MRR | Section Hit@5 |",
        "|---|---:|---:|---:|---:|",
        f"| Fixed 220/40 | 15 | {_percent(FIXED_METRICS['section_precision_at_1'])} | {FIXED_METRICS['section_mrr']:.4f} | {_percent(FIXED_METRICS['section_hit_at_5'])} |",
        f"| Heading-aware | 15 | {_percent(heading_metrics['overall']['section_precision_at_1'])} | {heading_metrics['overall']['section_mrr']:.4f} | {_percent(heading_metrics['overall']['section_hit_at_5'])} |",
        "",
        f"P@1 change: {comparison['absolute_p1_improvement_percentage_points']:.2f} points. MRR change: {comparison['mrr_improvement']:.4f}. Hit@5 change: {comparison['hit_at_5_change']:.4f}.",
        "",
        "## Frozen fixed-failure comparison",
        "",
    ]
    for question_id in ("S03", "S07", "S11"):
        row = paired[question_id]
        if row["fixed_correct_at_1"] == 0 and row["heading_correct_at_1"] == 1:
            outcome = "The heading-aware strategy changed this from a rank-1 miss to a rank-1 section match."
        elif row["fixed_correct_at_1"] == row["heading_correct_at_1"]:
            outcome = "The heading-aware strategy did not change the rank-1 correctness outcome."
        else:
            outcome = "The heading-aware strategy regressed the rank-1 correctness outcome."
        lines.extend(
            [
                f"### {question_id} — {row['expected_notice']}",
                "",
                f"- Expected section: {row['expected_heading']}",
                f"- Fixed rank: {row['fixed_rank']}",
                f"- Heading-aware rank: {row['heading_aware_rank']}",
                f"- Fixed top-1 preview: {row['fixed_top_1_preview']}",
                f"- Heading-aware top-1 preview: {row['heading_aware_top_1_preview']}",
                f"- Outcome: {outcome}",
                "",
            ]
        )
    heading_failures = [trace for trace in traces.values() if trace["section_score"]["precision_at_1"] == 0]
    lines.extend(["## Heading-aware P@1 failures", ""])
    if not heading_failures:
        lines.append("No heading-aware Section P@1 failures occurred.")
    for trace in heading_failures:
        top = trace["ranks"][0]
        lines.extend(
            [
                f"### {trace['id']} — {trace['expected_notice_code']}",
                "",
                f"- Question: {trace['question']}",
                f"- Expected path: `{' > '.join(trace['expected_heading_path'])}`",
                f"- Heading-aware first correct rank: {trace['section_score']['first_correct_rank']}",
                f"- Rank-1 path: `{' > '.join(top['heading_path'])}`",
                f"- Rank-1 preview: {top['text_preview']}",
                "",
            ]
        )
    counts = comparison["classification_counts"]
    lines.extend(
        [
            "## Paired outcome",
            "",
            f"- Improved: {counts['improved']}",
            f"- Unchanged: {counts['unchanged']}",
            f"- Regressed: {counts['regressed']}",
            f"- New regressions versus fixed P@1: {', '.join(comparison['new_regression_ids']) or 'none'}",
            "",
            "## Decision",
            "",
            f"The precommitted >=10-point P@1 threshold was {'met' if results['precommitted_thresholds']['threshold_met'] else 'not met'}.",
            f"Evidence-only retention decision: {'retain heading-aware chunking' if comparison['retain_heading_aware_based_on_precommitted_evidence'] else 'do not retain heading-aware chunking'}.",
            "",
            "## Regression protection",
            "",
            "- All approved Phase 1–4A file and tree hashes matched before and after.",
            "- The baseline namespace remained exactly 350 frozen IDs.",
            (
                "- The exact preexisting 580-vector heading namespace was reused without document embedding or upsert."
                if results["execution"]["query_only_recovery"]
                else "- The heading namespace alone received heading-aware upserts and reached exact local/remote ID parity."
            ),
            f"- Provider response-order normalizations: {results['pinecone']['provider_response_reordered_query_count']} (descending Pinecone similarity only; no reranker).",
            "- No index creation, deletion, clearing, update, baseline upsert, BM25, hybrid retrieval, reranking, rewriting, generation, LangGraph, or Streamlit was used.",
            "",
        ]
    )
    atomic_write_text(path, "\n".join(lines))


def run_phase4b(
    *,
    project_root: Path,
    offline_only: bool = False,
    resume_query_only: bool = False,
    secret_path: Path | None = None,
) -> dict[str, Any]:
    if offline_only and resume_query_only:
        raise Phase4BGateError("--offline-only and --resume-query-only are mutually exclusive")
    project_root = project_root.resolve()
    if resume_query_only:
        recovery_artifacts = {
            HEADING_REGISTRY_PATH: EXPECTED_HEADING_REGISTRY_SHA256,
            CHUNK_AUDIT_PATH: EXPECTED_HEADING_AUDIT_SHA256,
        }
        for relative, expected_hash in recovery_artifacts.items():
            path = project_root / relative
            if not path.is_file() or sha256_file(path) != expected_hash:
                raise Phase4BGateError(
                    f"Query-only recovery requires the exact pre-indexed artifact: {relative.as_posix()}"
                )
    frozen, chunks, chunk_audit = prepare_heading_chunks(project_root)
    questions = load_section_questions(project_root / SECTION_QUESTIONS_PATH)
    fixed_results, fixed_by_id = load_and_validate_fixed_results(project_root, questions)
    offline_result = {
        "status": "offline_ready" if offline_only else "offline_gate_passed",
        "heading_registry_sha256": sha256_file(project_root / HEADING_REGISTRY_PATH),
        "chunk_audit_sha256": sha256_file(project_root / CHUNK_AUDIT_PATH),
        "chunk_audit": {
            "documents_processed": chunk_audit["documents_processed"],
            "total_heading_aware_chunks": chunk_audit["total_heading_aware_chunks"],
            "chunks_per_document": chunk_audit["chunks_per_document"],
            "content_tokens_per_chunk": chunk_audit["content_tokens_per_chunk"],
            "logical_sections": chunk_audit["logical_sections"],
            "integrity": chunk_audit["integrity"],
            "quality_gate_passed": chunk_audit["quality_gate_passed"],
        },
        "frozen_fixed_metrics": FIXED_METRICS,
        "network_calls_made": 0,
    }
    if offline_only:
        return offline_result

    existing_reports = [
        path.as_posix()
        for path in (RESULTS_PATH, SUMMARY_PATH, COMPARISON_PATH, FAILURES_PATH)
        if (project_root / path).exists()
    ]
    if existing_reports:
        raise Phase4BGateError(
            "Phase 4B result reports already exist and will not be overwritten: " + ", ".join(existing_reports)
        )
    config = load_phase3_config(secret_path=secret_path, project_root=project_root)
    results = evaluate_and_index(
        project_root=project_root,
        config=config,
        frozen_before=frozen,
        chunks=chunks,
        chunk_audit=chunk_audit,
        questions=questions,
        fixed_results=fixed_results,
        fixed_by_id=fixed_by_id,
        resume_query_only=resume_query_only,
    )
    atomic_write_json(project_root / RESULTS_PATH, results)
    write_summary_csv(project_root / SUMMARY_PATH, results)
    write_comparison_csv(project_root / COMPARISON_PATH, results)
    write_failure_analysis(project_root / FAILURES_PATH, results, chunk_audit)
    if verify_frozen_artifacts(project_root) != frozen:
        raise Phase4BGateError("A frozen artifact changed while writing Phase 4B reports")
    return results


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Phase 4B heading-aware experiment")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--secret-file", type=Path, default=None)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--offline-only", action="store_true")
    mode.add_argument(
        "--resume-query-only",
        action="store_true",
        help="reuse an exact existing 580-ID heading namespace and perform no document embedding or upsert",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    try:
        result = run_phase4b(
            project_root=args.project_root,
            offline_only=args.offline_only,
            resume_query_only=args.resume_query_only,
            secret_path=args.secret_file,
        )
    except (
        Phase4BGateError,
        HeadingChunkingGateError,
        ProviderGateError,
        ConfigurationError,
    ) as exc:
        print(f"Phase 4B STOP: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(
            f"Phase 4B STOP: unexpected {type(exc).__name__}; no credentials were logged",
            file=sys.stderr,
        )
        return 2
    if args.offline_only:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        comparison = result["direct_comparison"]
        heading = comparison["heading_aware"]
        print(
            "Phase 4B complete: "
            f"Section P@1={heading['section_precision_at_1']:.4f}, "
            f"MRR={heading['section_mrr']:.4f}, Hit@5={heading['section_hit_at_5']:.4f}, "
            f"P@1 delta={comparison['absolute_p1_improvement_percentage_points']:.2f} points"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
