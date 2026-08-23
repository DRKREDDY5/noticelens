"""Secret-safe Nebius embeddings and Pinecone access for Phase 3."""

from __future__ import annotations

import math
import statistics
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from langchain_core.embeddings import Embeddings
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec

from .chunking import ChunkRecord, nearest_rank_percentile
from .evaluation import validate_embedding_dimension


EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-8B"
EMBEDDING_DIMENSION = 4096
NEBIUS_BASE_URL = "https://api.tokenfactory.nebius.com/v1"
INDEX_NAME = "noticelens-rag"
NAMESPACE = "baseline-fixed-dense"
INDEX_METRIC = "cosine"
INDEX_CLOUD = "aws"
INDEX_REGION = "us-east-1"
PINECONE_METADATA_KEYS = (
    "doc_id",
    "notice_code",
    "notice_family",
    "title",
    "source_url",
    "chunk_id",
    "chunk_strategy",
)


class ProviderGateError(RuntimeError):
    """Provider failure whose message is safe to persist in a report."""


def _safe_failure(stage: str, exc: BaseException) -> ProviderGateError:
    # Never include provider exception text: it can contain request details.
    return ProviderGateError(f"{stage} failed ({type(exc).__name__}); no credentials were logged")


def latency_summary(values: Sequence[float]) -> dict[str, float | int]:
    if not values:
        return {"request_count": 0, "total_seconds": 0.0, "p50_seconds": 0.0, "p95_seconds": 0.0, "max_seconds": 0.0}
    return {
        "request_count": len(values),
        "total_seconds": round(sum(values), 6),
        "p50_seconds": round(float(statistics.median(values)), 6),
        "p95_seconds": round(nearest_rank_percentile(values, 0.95), 6),
        "max_seconds": round(max(values), 6),
    }


class NebiusEmbeddings(Embeddings):
    """Small LangChain embedding adapter over Nebius's OpenAI-compatible API."""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = NEBIUS_BASE_URL,
        model: str = EMBEDDING_MODEL,
        expected_dimension: int = EMBEDDING_DIMENSION,
        batch_size: int = 8,
    ) -> None:
        if not api_key.strip():
            raise ProviderGateError("NEBIUS_API_KEY is empty")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.model = model
        self.expected_dimension = expected_dimension
        self.batch_size = batch_size
        self.base_url = base_url
        try:
            self._client = OpenAI(
                api_key=api_key,
                base_url=base_url,
                max_retries=2,
                timeout=120.0,
            )
        except Exception as exc:
            raise _safe_failure("Nebius client construction", exc) from None
        self.request_latencies: list[float] = []

    def __repr__(self) -> str:
        return (
            f"NebiusEmbeddings(model={self.model!r}, expected_dimension="
            f"{self.expected_dimension}, batch_size={self.batch_size})"
        )

    def _request(self, texts: list[str]) -> list[list[float]]:
        if not texts or any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ProviderGateError("Embedding inputs must be nonempty strings")
        started = time.perf_counter()
        try:
            response = self._client.embeddings.create(
                model=self.model,
                input=texts,
                encoding_format="float",
            )
        except Exception as exc:
            raise _safe_failure("Nebius embedding request", exc) from None
        finally:
            self.request_latencies.append(time.perf_counter() - started)

        ordered = sorted(response.data, key=lambda item: item.index)
        observed_indices = [int(item.index) for item in ordered]
        if observed_indices != list(range(len(texts))):
            raise ProviderGateError("Nebius embedding response indices are incomplete or duplicated")
        vectors = [list(item.embedding) for item in ordered]
        if len(vectors) != len(texts):
            raise ProviderGateError(
                f"Nebius embedding count mismatch: expected {len(texts)}, observed {len(vectors)}"
            )
        for vector in vectors:
            validate_embedding_dimension(vector, self.expected_dimension)
        return vectors

    def dimension_probe(self, text: str = "IRS CP503 unpaid balance notice") -> tuple[int, float]:
        before = len(self.request_latencies)
        vector = self._request([text])[0]
        latency = self.request_latencies[before]
        return validate_embedding_dimension(vector, self.expected_dimension), latency

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for offset in range(0, len(texts), self.batch_size):
            vectors.extend(self._request(texts[offset : offset + self.batch_size]))
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self._request([text])[0]


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _namespace_count(stats: Any, namespace: str) -> int:
    namespaces = _field(stats, "namespaces", {}) or {}
    summary = namespaces.get(namespace) if hasattr(namespaces, "get") else None
    if summary is None:
        return 0
    return int(_field(summary, "vector_count", 0) or 0)


def _flatten_list_page(page: Any) -> Iterable[str]:
    if isinstance(page, str):
        yield page
        return
    if isinstance(page, Mapping):
        values = page.get("vectors") or page.get("ids") or []
    else:
        values = _field(page, "vectors", page)
    if isinstance(values, str):
        yield values
        return
    try:
        iterator = iter(values)
    except TypeError:
        return
    for value in iterator:
        if isinstance(value, str):
            yield value
        elif isinstance(value, Mapping):
            vector_id = value.get("id")
            if vector_id is not None:
                yield str(vector_id)
        else:
            vector_id = _field(value, "id")
            if vector_id is not None:
                yield str(vector_id)


@dataclass(frozen=True)
class PineconeIndexState:
    index: Any
    existed_or_created: str
    host: str
    ready: bool
    dimension: int
    metric: str
    cloud: str | None
    region: str | None


class PineconeBaselineStore:
    """Pinecone v9 baseline index helper with exact namespace safeguards."""

    def __init__(
        self,
        *,
        api_key: str,
        index_name: str = INDEX_NAME,
        namespace: str = NAMESPACE,
    ) -> None:
        if not api_key.strip():
            raise ProviderGateError("PINECONE_API_KEY is empty")
        if not namespace.strip() or namespace == "__default__":
            raise ProviderGateError("Phase 3 requires a dedicated, non-default namespace")
        self.index_name = index_name
        self.namespace = namespace
        try:
            self._pc = Pinecone(api_key=api_key)
        except Exception as exc:
            raise _safe_failure("Pinecone client construction", exc) from None
        self.state: PineconeIndexState | None = None

    def __repr__(self) -> str:
        return f"PineconeBaselineStore(index_name={self.index_name!r}, namespace={self.namespace!r})"

    def ensure_compatible_index(self) -> PineconeIndexState:
        try:
            exists = bool(self._pc.indexes.exists(self.index_name))
            disposition = "reused" if exists else "created"
            if not exists:
                self._pc.indexes.create(
                    name=self.index_name,
                    vector_type="dense",
                    dimension=EMBEDDING_DIMENSION,
                    metric=INDEX_METRIC,
                    spec=ServerlessSpec(cloud=INDEX_CLOUD, region=INDEX_REGION),
                    deletion_protection="enabled",
                    timeout=300,
                )
            description = self._pc.indexes.describe(self.index_name)
        except Exception as exc:
            raise _safe_failure("Pinecone index verification/creation", exc) from None

        observed_dimension = int(_field(description, "dimension", 0) or 0)
        observed_metric = str(_field(description, "metric", "")).lower()
        observed_vector_type = str(_field(description, "vector_type", "dense") or "dense").lower()
        if observed_dimension != EMBEDDING_DIMENSION or observed_metric != INDEX_METRIC:
            raise ProviderGateError(
                "Existing Pinecone index is incompatible "
                f"(dimension={observed_dimension}, metric={observed_metric!r}); it was not modified"
            )
        if observed_vector_type not in {"", "dense"}:
            raise ProviderGateError(
                f"Existing Pinecone index vector type is {observed_vector_type!r}; it was not modified"
            )
        host = str(_field(description, "host", ""))
        status = _field(description, "status", {})
        ready = bool(_field(status, "ready", False))
        spec = _field(description, "spec", {}) or {}
        serverless = _field(spec, "serverless", {}) or {}
        cloud = _field(serverless, "cloud")
        region = _field(serverless, "region")
        normalized_cloud = "" if cloud is None else str(cloud).lower()
        normalized_region = "" if region is None else str(region).lower()
        if not host or not ready:
            raise ProviderGateError("Pinecone index did not become ready within the creation timeout")
        try:
            index = self._pc.index(host=host)
        except Exception as exc:
            raise _safe_failure("Pinecone data-plane connection", exc) from None
        self.state = PineconeIndexState(
            index=index,
            existed_or_created=disposition,
            host=host,
            ready=ready,
            dimension=observed_dimension,
            metric=observed_metric,
            cloud=normalized_cloud or None,
            region=normalized_region or None,
        )
        return self.state

    @property
    def index(self) -> Any:
        if self.state is None:
            raise ProviderGateError("Pinecone index has not been verified")
        return self.state.index

    def vector_count(self) -> int:
        try:
            return _namespace_count(self.index.describe_index_stats(), self.namespace)
        except Exception as exc:
            raise _safe_failure("Pinecone namespace count", exc) from None

    def list_vector_ids(self) -> set[str]:
        """List all IDs in the dedicated serverless namespace."""

        try:
            pages = self.index.list(namespace=self.namespace)
            ids: set[str] = set()
            for page in pages:
                ids.update(_flatten_list_page(page))
            return ids
        except Exception as exc:
            raise _safe_failure("Pinecone namespace ID reconciliation", exc) from None

    def assert_safe_existing_namespace(self, expected_ids: set[str]) -> dict[str, int]:
        current_count = self.vector_count()
        current_ids = self.list_vector_ids()
        if len(current_ids) != current_count:
            raise ProviderGateError(
                "Pinecone namespace list/count disagree before indexing; no vectors were written"
            )
        unknown = current_ids - expected_ids
        if unknown:
            raise ProviderGateError(
                f"Pinecone namespace contains {len(unknown)} stale or unknown IDs; no vectors were written"
            )
        return {"preexisting_vector_count": current_count, "preexisting_matching_ids": len(current_ids)}

    def upsert_batch(self, chunks: Sequence[ChunkRecord], vectors: Sequence[Sequence[float]]) -> int:
        if len(chunks) != len(vectors):
            raise ProviderGateError("Chunk/vector batch length mismatch")
        if not chunks:
            raise ProviderGateError("Empty Pinecone upsert batches are not allowed")
        records: list[dict[str, Any]] = []
        for chunk, vector in zip(chunks, vectors, strict=True):
            validate_embedding_dimension(vector, EMBEDDING_DIMENSION)
            try:
                metadata = {key: chunk.metadata[key] for key in PINECONE_METADATA_KEYS}
            except KeyError as exc:
                raise ProviderGateError(
                    f"Chunk {chunk.chunk_id} is missing required Pinecone metadata: {exc.args[0]}"
                ) from None
            if tuple(metadata) != PINECONE_METADATA_KEYS:
                raise ProviderGateError(f"Unexpected Pinecone metadata schema for {chunk.chunk_id}")
            records.append({"id": chunk.chunk_id, "values": list(vector), "metadata": metadata})
        try:
            response = self.index.upsert(vectors=records, namespace=self.namespace)
        except Exception as exc:
            raise _safe_failure("Pinecone vector upsert", exc) from None
        raw_count = _field(response, "upserted_count")
        if isinstance(raw_count, bool) or not isinstance(raw_count, int):
            raise ProviderGateError("Pinecone upsert response omitted a valid upserted count")
        observed = raw_count
        if observed != len(records):
            raise ProviderGateError(
                f"Pinecone upsert count mismatch: expected {len(records)}, observed {observed}"
            )
        return observed

    def wait_for_exact_parity(
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
            last_count = self.vector_count()
            if last_count == len(expected_ids):
                observed_ids = self.list_vector_ids()
                if observed_ids == expected_ids:
                    return last_count, observed_ids
            time.sleep(poll_seconds)
        raise ProviderGateError(
            f"Pinecone namespace parity timed out: expected {len(expected_ids)}, observed {last_count}"
        )

    def query(self, vector: Sequence[float], *, top_k: int = 5) -> list[Any]:
        if top_k != 5:
            raise ProviderGateError("The frozen Phase 3 retriever requires top_k=5")
        validate_embedding_dimension(vector, EMBEDDING_DIMENSION)
        try:
            response = self.index.query(
                namespace=self.namespace,
                vector=list(vector),
                top_k=top_k,
                include_metadata=True,
                include_values=False,
            )
        except Exception as exc:
            raise _safe_failure("Pinecone similarity query", exc) from None
        matches = list(_field(response, "matches", []) or [])
        if len(matches) != top_k:
            raise ProviderGateError(
                f"Pinecone returned {len(matches)} matches; exactly {top_k} are required"
            )
        return matches


def match_field(match: Any, name: str, default: Any = None) -> Any:
    return _field(match, name, default)


def finite_score(value: Any) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        raise ProviderGateError("Pinecone returned a non-numeric similarity score") from None
    if not math.isfinite(score):
        raise ProviderGateError("Pinecone returned a non-finite similarity score")
    return score
