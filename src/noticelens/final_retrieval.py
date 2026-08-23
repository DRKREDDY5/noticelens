"""Read-only production retrieval over the frozen heading-aware namespace."""

from __future__ import annotations

import hashlib
import math
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from pinecone import Pinecone

from .chunking import sha256_file
from .heading_chunking import HEADING_PINECONE_METADATA_KEYS, HeadingChunkRecord, load_heading_registry
from .providers import (
    EMBEDDING_DIMENSION,
    EMBEDDING_MODEL,
    INDEX_CLOUD,
    INDEX_METRIC,
    INDEX_NAME,
    INDEX_REGION,
    NebiusEmbeddings,
    ProviderGateError,
)


PRODUCTION_NAMESPACE = "heading-aware-dense"
TOP_K = 5
EXPECTED_VECTOR_COUNT = 580
REGISTRY_RELATIVE_PATH = Path("data/derived/phase4b/heading_aware_220_40_chunks.jsonl")
EXPECTED_REGISTRY_SHA256 = "3aecf5db7ee5fe857bdb99156c9bb5ba585f2e845aa6697cc4ece8902ac27572"
EXPECTED_ID_SET_SHA256 = "60bbf276bc752bfaf8f1e3ca8904fc88d688ce07e8c32b4224dc1e667b26912e"


class FinalRetrievalError(RuntimeError):
    """A safe, fail-closed production retrieval error."""


def _safe_error(stage: str, exc: BaseException) -> FinalRetrievalError:
    return FinalRetrievalError(f"{stage} failed ({type(exc).__name__}); no credentials were logged")


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _ids_from_page(page: Any) -> list[str]:
    if isinstance(page, str):
        return [page]
    values = page.get("vectors") or page.get("ids") or [] if isinstance(page, Mapping) else _field(page, "vectors", page)
    if isinstance(values, str):
        return [values]
    result: list[str] = []
    try:
        iterator = iter(values)
    except TypeError:
        return result
    for value in iterator:
        if isinstance(value, str):
            result.append(value)
        else:
            vector_id = _field(value, "id")
            if vector_id is not None:
                result.append(str(vector_id))
    return result


def _namespace_count(stats: Any, namespace: str) -> int:
    namespaces = _field(stats, "namespaces", {}) or {}
    summary = namespaces.get(namespace) if hasattr(namespaces, "get") else None
    return 0 if summary is None else int(_field(summary, "vector_count", 0) or 0)


def _id_set_sha256(ids: set[str]) -> str:
    return hashlib.sha256("\n".join(sorted(ids)).encode("utf-8")).hexdigest()


def _validate_vector(vector: Sequence[float]) -> None:
    if isinstance(vector, (str, bytes)) or len(vector) != EMBEDDING_DIMENSION:
        observed = 0 if isinstance(vector, (str, bytes)) else len(vector)
        raise FinalRetrievalError(
            f"Embedding dimension mismatch: expected {EMBEDDING_DIMENSION}, observed {observed}"
        )
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        for value in vector
    ):
        raise FinalRetrievalError("Embedding contains a non-finite or non-numeric value")


@dataclass(frozen=True)
class RetrievalResult:
    documents: list[Document]
    embedding_seconds: float
    pinecone_seconds: float


class ReadOnlyHeadingStore:
    """Existing-index Pinecone client with no mutating method surface."""

    def __init__(self, *, api_key: str, client: Any | None = None) -> None:
        if not isinstance(api_key, str) or not api_key.strip():
            raise FinalRetrievalError("PINECONE_API_KEY is empty")
        try:
            self._pc = client if client is not None else Pinecone(api_key=api_key)
        except Exception as exc:
            raise _safe_error("Pinecone client construction", exc) from None
        self._index: Any | None = None

    def require_frozen_index(self) -> dict[str, object]:
        try:
            if not bool(self._pc.indexes.exists(INDEX_NAME)):
                raise FinalRetrievalError("The approved Pinecone index is missing")
            description = self._pc.indexes.describe(INDEX_NAME)
        except FinalRetrievalError:
            raise
        except Exception as exc:
            raise _safe_error("Pinecone index verification", exc) from None
        dimension = int(_field(description, "dimension", 0) or 0)
        metric = str(_field(description, "metric", "") or "").lower()
        vector_type = str(_field(description, "vector_type", "dense") or "dense").lower()
        status = _field(description, "status", {}) or {}
        ready = bool(_field(status, "ready", False))
        host = str(_field(description, "host", "") or "")
        spec = _field(description, "spec", {}) or {}
        serverless = _field(spec, "serverless", {}) or {}
        cloud = str(_field(serverless, "cloud", "") or "").lower()
        region = str(_field(serverless, "region", "") or "").lower()
        if (
            dimension != EMBEDDING_DIMENSION
            or metric != INDEX_METRIC
            or vector_type not in {"", "dense"}
            or cloud != INDEX_CLOUD
            or region != INDEX_REGION
            or not ready
            or not host
        ):
            raise FinalRetrievalError("The approved Pinecone index is incompatible or not ready")
        try:
            self._index = self._pc.index(host=host)
        except Exception as exc:
            raise _safe_error("Pinecone data-plane connection", exc) from None
        return {
            "name": INDEX_NAME,
            "dimension": dimension,
            "metric": metric,
            "namespace": PRODUCTION_NAMESPACE,
            "ready": ready,
        }

    @property
    def index(self) -> Any:
        if self._index is None:
            raise FinalRetrievalError("Pinecone index has not passed the frozen gate")
        return self._index

    def assert_namespace(self, expected_ids: set[str]) -> dict[str, object]:
        try:
            count = _namespace_count(self.index.describe_index_stats(), PRODUCTION_NAMESPACE)
            observed_list: list[str] = []
            for page in self.index.list(namespace=PRODUCTION_NAMESPACE):
                observed_list.extend(_ids_from_page(page))
        except Exception as exc:
            if isinstance(exc, FinalRetrievalError):
                raise
            raise _safe_error("Pinecone production namespace verification", exc) from None
        observed_ids = set(observed_list)
        if len(observed_list) != len(observed_ids):
            raise FinalRetrievalError("Pinecone production namespace listed duplicate vector IDs")
        if count != len(observed_ids) or observed_ids != expected_ids:
            raise FinalRetrievalError("Pinecone production namespace does not match the frozen registry")
        return {
            "vector_count": count,
            "exact_id_parity": True,
            "id_set_sha256": _id_set_sha256(observed_ids),
        }

    def query(self, vector: Sequence[float], *, notice_code: str, eligible_count: int) -> list[Any]:
        _validate_vector(vector)
        if not notice_code.strip() or eligible_count <= 0:
            raise FinalRetrievalError("The notice has no eligible frozen guidance chunks")
        expected_matches = min(TOP_K, eligible_count)
        try:
            response = self.index.query(
                namespace=PRODUCTION_NAMESPACE,
                vector=list(vector),
                top_k=TOP_K,
                filter={"notice_code": {"$eq": notice_code}},
                include_metadata=True,
                include_values=False,
            )
        except Exception as exc:
            raise _safe_error("Pinecone filtered guidance query", exc) from None
        matches = list(_field(response, "matches", []) or [])
        if len(matches) != expected_matches:
            raise FinalRetrievalError(
                f"Pinecone returned {len(matches)} matches; expected {expected_matches}"
            )
        decorated: list[tuple[float, str, Any]] = []
        for match in matches:
            vector_id = str(_field(match, "id", "") or "")
            try:
                score = float(_field(match, "score"))
            except (TypeError, ValueError):
                raise FinalRetrievalError("Pinecone returned a non-numeric similarity score") from None
            if not vector_id or not math.isfinite(score):
                raise FinalRetrievalError("Pinecone returned a blank ID or non-finite score")
            decorated.append((score, vector_id, match))
        if len({item[1] for item in decorated}) != len(decorated):
            raise FinalRetrievalError("Pinecone returned duplicate guidance chunks")
        return [item[2] for item in sorted(decorated, key=lambda item: (-item[0], item[1]))]


class FinalHeadingRetriever:
    """Qwen embedding + exact-notice filtered heading-aware retriever."""

    def __init__(
        self,
        *,
        project_root: Path,
        nebius_api_key: str,
        pinecone_api_key: str,
        embedding_client: NebiusEmbeddings | None = None,
        store: ReadOnlyHeadingStore | None = None,
    ) -> None:
        registry_path = project_root / REGISTRY_RELATIVE_PATH
        if sha256_file(registry_path) != EXPECTED_REGISTRY_SHA256:
            raise FinalRetrievalError("The approved heading-aware registry hash changed")
        chunks = load_heading_registry(registry_path)
        self._records = {chunk.chunk_id: chunk for chunk in chunks}
        if len(self._records) != EXPECTED_VECTOR_COUNT:
            raise FinalRetrievalError("The approved heading-aware registry count changed")
        expected_ids = set(self._records)
        if _id_set_sha256(expected_ids) != EXPECTED_ID_SET_SHA256:
            raise FinalRetrievalError("The approved heading-aware registry ID set changed")
        self._eligible_counts = Counter(
            str(chunk.metadata["notice_code"]) for chunk in self._records.values()
        )
        self.embeddings = embedding_client or NebiusEmbeddings(api_key=nebius_api_key, batch_size=16)
        if self.embeddings.model != EMBEDDING_MODEL or self.embeddings.expected_dimension != EMBEDDING_DIMENSION:
            raise FinalRetrievalError("The final embedding configuration differs from the approved baseline")
        self.store = store or ReadOnlyHeadingStore(api_key=pinecone_api_key)
        self.index_state = self.store.require_frozen_index()
        self.namespace_state = self.store.assert_namespace(expected_ids)

    def retrieve(self, question: str, *, notice_code: str) -> RetrievalResult:
        if not isinstance(question, str) or not question.strip():
            raise FinalRetrievalError("The retrieval question is empty")
        eligible = int(self._eligible_counts.get(notice_code, 0))
        started_embedding = time.perf_counter()
        vector = self.embeddings.embed_query(question)
        embedding_seconds = time.perf_counter() - started_embedding
        _validate_vector(vector)
        started_query = time.perf_counter()
        matches = self.store.query(vector, notice_code=notice_code, eligible_count=eligible)
        pinecone_seconds = time.perf_counter() - started_query
        documents: list[Document] = []
        for match in matches:
            chunk_id = str(_field(match, "id", "") or "")
            record = self._records.get(chunk_id)
            if record is None:
                raise FinalRetrievalError("Pinecone returned a chunk absent from the frozen registry")
            raw_metadata = _field(match, "metadata", {}) or {}
            if not isinstance(raw_metadata, Mapping) or set(raw_metadata) != set(HEADING_PINECONE_METADATA_KEYS):
                raise FinalRetrievalError("Pinecone returned an invalid heading-aware metadata schema")
            expected = {key: record.metadata[key] for key in HEADING_PINECONE_METADATA_KEYS}
            if dict(raw_metadata) != expected:
                raise FinalRetrievalError("Pinecone metadata differs from the frozen local registry")
            if expected["notice_code"] != notice_code:
                raise FinalRetrievalError("Pinecone returned evidence for a different notice code")
            score = float(_field(match, "score"))
            metadata = dict(expected)
            metadata["similarity_score"] = score
            documents.append(Document(page_content=record.text, metadata=metadata))
        return RetrievalResult(
            documents=documents,
            embedding_seconds=embedding_seconds,
            pinecone_seconds=pinecone_seconds,
        )


def evidence_is_sufficient(documents: Sequence[Document], *, notice_code: str) -> bool:
    """Deterministic metadata/content gate; deliberately has no score cutoff."""

    required = {
        "doc_id",
        "notice_code",
        "title",
        "source_url",
        "chunk_id",
        "heading",
        "heading_path",
    }
    if not documents:
        return False
    for document in documents:
        if not isinstance(document, Document) or not document.page_content.strip():
            return False
        metadata = document.metadata
        if not required.issubset(metadata) or any(metadata.get(key) in (None, "") for key in required):
            return False
        if metadata.get("notice_code") != notice_code:
            return False
        path = metadata.get("heading_path")
        if not isinstance(path, list) or not path or any(not isinstance(item, str) or not item for item in path):
            return False
    return True

