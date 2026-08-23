"""Deterministic fixed-token chunking for the Phase 3 dense baseline."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import statistics
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langchain_text_splitters.base import TextSplitter


TOKENIZER_NAME = "Qwen/Qwen3-Embedding-8B"
TOKENIZER_REVISION = "7bd6fbe3c54b9ec2b4b1cc3a052720a76fcf0d90"
CHUNK_SIZE = 220
CHUNK_OVERLAP = 40
CHUNK_STRIDE = CHUNK_SIZE - CHUNK_OVERLAP
CHUNK_STRATEGY = "fixed_220_40"

LOCAL_METADATA_KEYS = (
    "doc_id",
    "notice_code",
    "notice_family",
    "title",
    "source_url",
    "source_origin",
    "chunk_id",
    "chunk_strategy",
)

REQUIRED_MANIFEST_COLUMNS = (
    "doc_id",
    "notice_code",
    "notice_family",
    "title",
    "source_url",
    "source_origin",
    "content_hash",
)


class ChunkingGateError(RuntimeError):
    """Raised when frozen-input or chunk-quality validation fails."""


@dataclass(frozen=True)
class TokenWindow:
    """One decoded source-token window."""

    text: str
    token_start: int
    token_end: int
    token_count: int
    source_token_count: int
    roundtrip_exact: bool


@dataclass(frozen=True)
class ChunkRecord:
    """One local chunk registry record."""

    chunk_id: str
    text: str
    token_start: int
    token_end: int
    token_count: int
    source_token_count: int
    text_sha256: str
    metadata: dict[str, str | int]

    def as_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "token_start": self.token_start,
            "token_end": self.token_end,
            "token_count": self.token_count,
            "source_token_count": self.source_token_count,
            "text_sha256": self.text_sha256,
            "metadata": self.metadata,
        }


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def nearest_rank_percentile(values: Sequence[int | float], percentile: float) -> float:
    """Return a predeclared nearest-rank percentile."""

    if not values:
        raise ValueError("Cannot calculate a percentile for an empty sequence")
    if not 0 < percentile <= 1:
        raise ValueError("percentile must be in (0, 1]")
    ordered = sorted(values)
    return float(ordered[math.ceil(percentile * len(ordered)) - 1])


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


class FixedTokenTextSplitter(TextSplitter):
    """LangChain-compatible splitter with exact token windows.

    It treats each complete Markdown file as a flat token stream. Headings and
    notice identifiers are ordinary text and never affect a boundary.
    """

    def __init__(
        self,
        tokenizer: Any,
        *,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if not 0 <= chunk_overlap < chunk_size:
            raise ValueError("chunk_overlap must be nonnegative and smaller than chunk_size")
        super().__init__(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.tokenizer = tokenizer

    def _encode(self, text: str) -> list[int]:
        return list(self.tokenizer.encode(text, add_special_tokens=False))

    def _decode(self, token_ids: Sequence[int]) -> str:
        return self.tokenizer.decode(
            list(token_ids),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

    def split_windows(self, text: str) -> list[TokenWindow]:
        source_ids = self._encode(text)
        if not source_ids:
            return []

        windows: list[TokenWindow] = []
        start = 0
        while start < len(source_ids):
            end = min(start + self._chunk_size, len(source_ids))
            candidate = self._decode(source_ids[start:end])
            encoded_candidate = self._encode(candidate)

            # Decoding a slice can very rarely tokenize to more tokens as a
            # standalone string. Shrink deterministically until the actual
            # embedding input respects the 220-token cap.
            while len(encoded_candidate) > self._chunk_size and end > start:
                end -= 1
                candidate = self._decode(source_ids[start:end])
                encoded_candidate = self._encode(candidate)

            if end <= start or not candidate.strip():
                raise ChunkingGateError("Tokenizer produced an empty fixed-size chunk")

            windows.append(
                TokenWindow(
                    text=candidate,
                    token_start=start,
                    token_end=end,
                    token_count=len(encoded_candidate),
                    source_token_count=end - start,
                    roundtrip_exact=encoded_candidate == source_ids[start:end],
                )
            )
            if end == len(source_ids):
                break
            next_start = end - self._chunk_overlap
            if next_start <= start:
                raise ChunkingGateError("Chunker failed to make forward progress")
            start = next_start
        return windows

    def split_text(self, text: str) -> list[str]:
        return [window.text for window in self.split_windows(text)]


def load_qwen_tokenizer() -> Any:
    """Load the frozen tokenizer revision without remote code execution."""

    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        TOKENIZER_NAME,
        revision=TOKENIZER_REVISION,
        # Loading the monolithic fast-tokenizer artifact hit a Windows-native
        # allocator failure here. Building the model tokenizer from the same
        # frozen revision's vocab/merges produces the authoritative token IDs.
        use_fast=False,
        trust_remote_code=False,
    )


def _read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = tuple(reader.fieldnames or ())
        missing_columns = [name for name in REQUIRED_MANIFEST_COLUMNS if name not in fieldnames]
        if missing_columns:
            raise ChunkingGateError(
                "Corpus manifest is missing required columns: " + ", ".join(missing_columns)
            )
        rows = [dict(row) for row in reader]

    if len(rows) != 50:
        raise ChunkingGateError(f"Frozen corpus must contain 50 rows, found {len(rows)}")
    doc_ids = [row["doc_id"].strip() for row in rows]
    if len(doc_ids) != len(set(doc_ids)):
        raise ChunkingGateError("Corpus manifest has duplicate doc_id values")
    for row_number, row in enumerate(rows, start=2):
        blank = [name for name in REQUIRED_MANIFEST_COLUMNS if not row.get(name, "").strip()]
        if blank:
            raise ChunkingGateError(
                f"Corpus manifest row {row_number} has blank required fields: {', '.join(blank)}"
            )
        content_hash = row["content_hash"].strip().lower()
        if len(content_hash) != 64 or any(char not in "0123456789abcdef" for char in content_hash):
            raise ChunkingGateError(f"Corpus manifest row {row_number} has an invalid content hash")
    return rows


def _build_chunk_id(doc_id: str, ordinal: int, text_sha256: str) -> str:
    return f"{doc_id}__{CHUNK_STRATEGY}__{ordinal:06d}__{text_sha256}"


def _summarize_numeric(values: Sequence[int]) -> dict[str, int | float]:
    if not values:
        return {"min": 0, "median": 0, "max": 0}
    return {
        "min": min(values),
        "median": statistics.median(values),
        "max": max(values),
    }


def build_fixed_chunks(
    *,
    project_root: Path,
    tokenizer: Any,
    write_outputs: bool = True,
) -> tuple[list[ChunkRecord], dict[str, Any]]:
    """Validate the frozen corpus, chunk it, and create the local audit."""

    project_root = project_root.resolve()
    manifest_path = project_root / "data" / "corpus_manifest.csv"
    processed_dir = project_root / "data" / "processed" / "guidance"
    output_path = project_root / "data" / "derived" / "phase3" / "fixed_220_40_chunks.jsonl"
    audit_path = project_root / "reports" / "phase3_chunk_audit.json"

    rows = _read_manifest(manifest_path)
    expected_paths = {processed_dir / f"{row['doc_id'].strip()}.md" for row in rows}
    actual_paths = set(processed_dir.glob("*.md"))
    missing_files = sorted(str(path.relative_to(project_root)) for path in expected_paths - actual_paths)
    extra_files = sorted(str(path.relative_to(project_root)) for path in actual_paths - expected_paths)
    if missing_files or extra_files:
        raise ChunkingGateError(
            f"Frozen processed-file set mismatch (missing={len(missing_files)}, extra={len(extra_files)})"
        )

    splitter = FixedTokenTextSplitter(tokenizer)
    chunks: list[ChunkRecord] = []
    chunks_per_document: list[int] = []
    zero_chunk_doc_ids: list[str] = []
    source_hashes: dict[str, str] = {}
    roundtrip_mismatch_count = 0
    overlap_mismatch_count = 0

    for row in rows:
        doc_id = row["doc_id"].strip()
        markdown_path = processed_dir / f"{doc_id}.md"
        payload = markdown_path.read_bytes()
        observed_hash = sha256_bytes(payload)
        expected_hash = row["content_hash"].strip().lower()
        if observed_hash != expected_hash:
            raise ChunkingGateError(f"Frozen content hash mismatch for {doc_id}")
        try:
            markdown = payload.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ChunkingGateError(f"Frozen Markdown is not valid UTF-8 for {doc_id}") from exc
        source_hashes[doc_id] = observed_hash

        windows = splitter.split_windows(markdown)
        chunks_per_document.append(len(windows))
        if not windows:
            zero_chunk_doc_ids.append(doc_id)
        prior_window: TokenWindow | None = None
        for ordinal, window in enumerate(windows):
            if not window.roundtrip_exact:
                roundtrip_mismatch_count += 1
            if prior_window is not None:
                if window.token_start != prior_window.token_end - CHUNK_OVERLAP:
                    overlap_mismatch_count += 1
            prior_window = window

            text_hash = sha256_bytes(window.text.encode("utf-8"))
            chunk_id = _build_chunk_id(doc_id, ordinal, text_hash)
            metadata: dict[str, str | int] = {
                "doc_id": doc_id,
                "notice_code": row["notice_code"].strip(),
                "notice_family": row["notice_family"].strip(),
                "title": row["title"].strip(),
                "source_url": row["source_url"].strip(),
                "source_origin": row["source_origin"].strip(),
                "chunk_id": chunk_id,
                "chunk_strategy": CHUNK_STRATEGY,
            }
            if tuple(metadata) != LOCAL_METADATA_KEYS:
                raise ChunkingGateError(f"Unexpected local metadata schema for {chunk_id}")
            chunks.append(
                ChunkRecord(
                    chunk_id=chunk_id,
                    text=window.text,
                    token_start=window.token_start,
                    token_end=window.token_end,
                    token_count=window.token_count,
                    source_token_count=window.source_token_count,
                    text_sha256=text_hash,
                    metadata=metadata,
                )
            )

    chunk_ids = [chunk.chunk_id for chunk in chunks]
    duplicate_ids = sorted({chunk_id for chunk_id in chunk_ids if chunk_ids.count(chunk_id) > 1})
    empty_ids = [chunk.chunk_id for chunk in chunks if not chunk.text.strip()]
    missing_doc_ids = [chunk.chunk_id for chunk in chunks if not str(chunk.metadata["doc_id"]).strip()]
    missing_notice_codes = [chunk.chunk_id for chunk in chunks if not str(chunk.metadata["notice_code"]).strip()]
    missing_titles = [chunk.chunk_id for chunk in chunks if not str(chunk.metadata["title"]).strip()]
    token_counts = [chunk.token_count for chunk in chunks]

    quality_gate_passed = not any(
        (
            duplicate_ids,
            empty_ids,
            missing_doc_ids,
            missing_notice_codes,
            missing_titles,
            zero_chunk_doc_ids,
            overlap_mismatch_count,
        )
    ) and all(0 < count <= CHUNK_SIZE for count in token_counts)

    try:
        import transformers

        transformers_version = transformers.__version__
    except Exception:
        transformers_version = "unknown"

    audit: dict[str, Any] = {
        "schema_version": "1.0",
        "generated_at_utc": utc_now(),
        "inputs": {
            "corpus_manifest_path": str(manifest_path.relative_to(project_root)).replace("\\", "/"),
            "corpus_manifest_sha256": sha256_file(manifest_path),
            "processed_guidance_path": str(processed_dir.relative_to(project_root)).replace("\\", "/"),
            "source_document_count": len(rows),
            "source_hashes_verified": len(source_hashes),
            "source_hashes": source_hashes,
            "missing_files": missing_files,
            "extra_files": extra_files,
        },
        "chunker": {
            "chunk_strategy": CHUNK_STRATEGY,
            "tokenizer_name": TOKENIZER_NAME,
            "tokenizer_revision": TOKENIZER_REVISION,
            "transformers_version": transformers_version,
            "tokenizer_class": type(tokenizer).__name__,
            "is_fast": bool(getattr(tokenizer, "is_fast", False)),
            "target_tokens": CHUNK_SIZE,
            "overlap_tokens": CHUNK_OVERLAP,
            "stride_tokens": CHUNK_STRIDE,
            "add_special_tokens": False,
            "heading_boundaries_used": False,
            "notice_code_boundaries_used": False,
        },
        "total_chunks": len(chunks),
        "chunks_per_document": _summarize_numeric(chunks_per_document),
        "tokens_per_chunk": {
            **_summarize_numeric(token_counts),
            "p95": nearest_rank_percentile(token_counts, 0.95) if token_counts else 0,
            "p95_method": "nearest_rank",
        },
        "integrity": {
            "empty_chunks": {"count": len(empty_ids), "ids": empty_ids},
            "duplicate_chunk_ids": {"count": len(duplicate_ids), "ids": duplicate_ids},
            "missing_doc_ids": {"count": len(missing_doc_ids), "ids": missing_doc_ids},
            "missing_notice_codes": {
                "count": len(missing_notice_codes),
                "ids": missing_notice_codes,
            },
            "missing_titles": {"count": len(missing_titles), "ids": missing_titles},
            "zero_chunk_documents": {
                "count": len(zero_chunk_doc_ids),
                "doc_ids": zero_chunk_doc_ids,
            },
            "overlap_mismatches": overlap_mismatch_count,
            "roundtrip_token_mismatches": roundtrip_mismatch_count,
            "max_token_limit_violations": sum(count > CHUNK_SIZE for count in token_counts),
        },
        "quality_gate_passed": quality_gate_passed,
    }

    if write_outputs:
        lines = [json.dumps(chunk.as_dict(), ensure_ascii=False, separators=(",", ":")) for chunk in chunks]
        atomic_write_text(output_path, "\n".join(lines) + ("\n" if lines else ""))
        atomic_write_json(audit_path, audit)

    if not quality_gate_passed:
        raise ChunkingGateError("Phase 3 chunk quality gate failed; cloud calls are blocked")
    return chunks, audit


def load_chunk_registry(path: Path) -> list[ChunkRecord]:
    """Load and minimally validate a previously generated registry."""

    records: list[ChunkRecord] = []
    ordinals_by_doc: dict[str, list[int]] = defaultdict(list)
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            metadata = value["metadata"]
            if tuple(metadata) != LOCAL_METADATA_KEYS:
                raise ChunkingGateError(f"Registry line {line_number} has unexpected metadata")
            record = ChunkRecord(
                chunk_id=value["chunk_id"],
                text=value["text"],
                token_start=int(value["token_start"]),
                token_end=int(value["token_end"]),
                token_count=int(value["token_count"]),
                source_token_count=int(value["source_token_count"]),
                text_sha256=value["text_sha256"],
                metadata=metadata,
            )
            if record.chunk_id != metadata["chunk_id"]:
                raise ChunkingGateError(f"Registry line {line_number} has inconsistent chunk IDs")
            if sha256_bytes(record.text.encode("utf-8")) != record.text_sha256:
                raise ChunkingGateError(f"Registry line {line_number} has a text hash mismatch")
            if not record.text.strip():
                raise ChunkingGateError(f"Registry line {line_number} has empty text")
            if not 0 < record.token_count <= CHUNK_SIZE:
                raise ChunkingGateError(f"Registry line {line_number} has an invalid token count")
            if record.token_start < 0 or record.token_end <= record.token_start:
                raise ChunkingGateError(f"Registry line {line_number} has an invalid token span")
            if record.source_token_count != record.token_end - record.token_start:
                raise ChunkingGateError(f"Registry line {line_number} has an inconsistent source token count")
            if metadata["chunk_strategy"] != CHUNK_STRATEGY:
                raise ChunkingGateError(f"Registry line {line_number} has an invalid chunk strategy")
            try:
                id_doc, id_strategy, id_ordinal, id_hash = record.chunk_id.rsplit("__", 3)
                ordinal = int(id_ordinal)
            except (TypeError, ValueError):
                raise ChunkingGateError(f"Registry line {line_number} has a malformed chunk ID") from None
            if ordinal < 0:
                raise ChunkingGateError(f"Registry line {line_number} has a negative chunk ordinal")
            expected_id = _build_chunk_id(str(metadata["doc_id"]), ordinal, record.text_sha256)
            if (
                id_doc != metadata["doc_id"]
                or id_strategy != CHUNK_STRATEGY
                or id_hash != record.text_sha256
                or record.chunk_id != expected_id
            ):
                raise ChunkingGateError(f"Registry line {line_number} has a non-deterministic chunk ID")
            ordinals_by_doc[str(metadata["doc_id"])].append(ordinal)
            records.append(record)
    if len({record.chunk_id for record in records}) != len(records):
        raise ChunkingGateError("Registry contains duplicate chunk IDs")
    for doc_id, ordinals in ordinals_by_doc.items():
        if sorted(ordinals) != list(range(len(ordinals))):
            raise ChunkingGateError(f"Registry has noncontiguous or duplicate ordinals for {doc_id}")
    return records


def iter_jsonl(records: Iterable[ChunkRecord]) -> Iterable[str]:
    """Expose deterministic JSONL serialization for offline tests."""

    for record in records:
        yield json.dumps(record.as_dict(), ensure_ascii=False, separators=(",", ":"))
