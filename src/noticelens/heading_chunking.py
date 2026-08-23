"""Deterministic heading-aware chunks for the Phase 4B experiment."""

from __future__ import annotations

import csv
import hashlib
import html
import json
import re
import statistics
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .chunking import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    CHUNK_STRIDE,
    TOKENIZER_NAME,
    TOKENIZER_REVISION,
    atomic_write_json,
    atomic_write_text,
    nearest_rank_percentile,
    sha256_bytes,
    sha256_file,
    utc_now,
)


HEADING_CHUNK_STRATEGY = "heading_aware_220_40"
HEADING_LOCAL_METADATA_KEYS = (
    "doc_id",
    "notice_code",
    "notice_family",
    "title",
    "source_url",
    "source_origin",
    "chunk_id",
    "chunk_strategy",
    "heading",
    "heading_path",
    "section_index",
    "subchunk_index",
)
HEADING_PINECONE_METADATA_KEYS = HEADING_LOCAL_METADATA_KEYS

_ATX_HEADING = re.compile(r"^(#{1,6})[ \t]+(.+?)\s*$")
_FENCE = re.compile(r"^[ \t]*(```+|~~~+)")
_MARKDOWN_LINK = re.compile(r"!?\[([^\]]+)\]\([^)]*\)")
_HTML_TAG = re.compile(r"<[^>]+>")


class HeadingChunkingGateError(RuntimeError):
    """Raised when heading-aware chunk construction violates the frozen contract."""


@dataclass(frozen=True)
class LogicalSection:
    """One H2+ section body bounded by the next source heading."""

    section_index: int
    heading: str
    heading_path: tuple[str, ...]
    raw_heading_path: tuple[str, ...]
    heading_levels: tuple[int, ...]
    heading_line: int
    content_char_start: int
    content_char_end: int
    content: str


@dataclass(frozen=True)
class HeadingChunkRecord:
    """One heading-aware embedding record and its source-section provenance."""

    chunk_id: str
    text: str
    content_text: str
    text_sha256: str
    content_sha256: str
    content_token_start: int
    content_token_end: int
    content_token_count: int
    embedding_token_count: int
    section_content_token_count: int
    section_subchunk_count: int
    source_heading_line: int
    metadata: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "content_text": self.content_text,
            "text_sha256": self.text_sha256,
            "content_sha256": self.content_sha256,
            "content_token_start": self.content_token_start,
            "content_token_end": self.content_token_end,
            "content_token_count": self.content_token_count,
            "embedding_token_count": self.embedding_token_count,
            "section_content_token_count": self.section_content_token_count,
            "section_subchunk_count": self.section_subchunk_count,
            "source_heading_line": self.source_heading_line,
            "metadata": self.metadata,
        }


def normalize_heading(value: str) -> str:
    return " ".join(unicodedata.normalize("NFC", value).split())


def visible_heading(value: str) -> str:
    value = _MARKDOWN_LINK.sub(lambda match: match.group(1), value)
    value = _HTML_TAG.sub(" ", value)
    value = html.unescape(value)
    value = value.replace("`", "").replace("**", "").replace("__", "")
    return normalize_heading(value)


def load_heading_tokenizer() -> Any:
    """Load the Phase 3 tokenizer revision cache-only and without remote code."""

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
        raise HeadingChunkingGateError(
            f"Frozen tokenizer is unavailable offline ({type(exc).__name__}); no network call was made"
        ) from None


def _heading_events(markdown: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    offset = 0
    active_fence: str | None = None
    stack: list[tuple[int, str, str]] = []
    section_index = 0
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
            raw_heading = normalize_heading(match.group(2))
            display_heading = visible_heading(raw_heading)
            if level == 1:
                stack.clear()
                path: tuple[str, ...] = ()
                raw_path: tuple[str, ...] = ()
                levels: tuple[int, ...] = ()
                logical_index: int | None = None
            else:
                while stack and stack[-1][0] >= level:
                    stack.pop()
                stack.append((level, display_heading, raw_heading))
                path = tuple(item[1] for item in stack)
                raw_path = tuple(item[2] for item in stack)
                levels = tuple(item[0] for item in stack)
                logical_index = section_index
                section_index += 1
            events.append(
                {
                    "level": level,
                    "heading": display_heading,
                    "path": path,
                    "raw_path": raw_path,
                    "levels": levels,
                    "line": line_number,
                    "char_start": offset,
                    "char_end": offset + len(line),
                    "section_index": logical_index,
                }
            )
        offset += len(line)
    if active_fence is not None:
        raise HeadingChunkingGateError("Markdown contains an unclosed fenced block")
    return events


def parse_logical_sections(markdown: str) -> tuple[str, list[LogicalSection], dict[str, int]]:
    """Return the H1 title and nonempty, independently bounded H2+ bodies."""

    events = _heading_events(markdown)
    h1_events = [event for event in events if event["level"] == 1]
    if len(h1_events) != 1:
        raise HeadingChunkingGateError("Each frozen Markdown document must contain exactly one H1")
    title = h1_events[0]["heading"]
    h2_plus = [event for event in events if event["path"]]
    full_paths = [tuple(event["path"]) for event in h2_plus]
    if len(set(full_paths)) != len(full_paths):
        raise HeadingChunkingGateError("A document contains duplicate full H2+ heading paths")

    sections: list[LogicalSection] = []
    empty_body_count = 0
    for index, event in enumerate(events):
        if not event["path"]:
            continue
        next_start = events[index + 1]["char_start"] if index + 1 < len(events) else len(markdown)
        raw_body = markdown[event["char_end"] : next_start]
        if not raw_body.strip():
            empty_body_count += 1
            continue
        left_trimmed = len(raw_body) - len(raw_body.lstrip())
        right_trimmed = len(raw_body.rstrip())
        content_start = event["char_end"] + left_trimmed
        content_end = event["char_end"] + right_trimmed
        content = markdown[content_start:content_end]
        if not content.strip():
            raise HeadingChunkingGateError("A nonempty logical section normalized to empty content")
        sections.append(
            LogicalSection(
                section_index=int(event["section_index"]),
                heading=event["heading"],
                heading_path=tuple(event["path"]),
                raw_heading_path=tuple(event["raw_path"]),
                heading_levels=tuple(event["levels"]),
                heading_line=int(event["line"]),
                content_char_start=content_start,
                content_char_end=content_end,
                content=content,
            )
        )
    first_h2_start = h2_plus[0]["char_start"] if h2_plus else len(markdown)
    pre_h2 = markdown[h1_events[0]["char_end"] : first_h2_start]
    useful_pre_h2_lines = [
        line.strip()
        for line in pre_h2.splitlines()
        if line.strip() and not line.strip().startswith("Source:")
    ]
    return title, sections, {
        "h1_count": len(h1_events),
        "h2_h6_heading_count": len(h2_plus),
        "nonempty_logical_section_count": len(sections),
        "empty_body_heading_count": empty_body_count,
        "unassigned_useful_pre_h2_content": int(bool(useful_pre_h2_lines)),
    }


def structural_prefix(title: str, notice_code: str, heading_path: Sequence[str]) -> str:
    if not title.strip() or not notice_code.strip() or not heading_path:
        raise HeadingChunkingGateError("Structural prefix fields must be nonempty")
    path_text = " > ".join(heading_path)
    return f"{title}\n\nNotice: {notice_code}\n\nSection:\n{path_text}\n\n"


def _content_windows(token_ids: Sequence[int]) -> list[tuple[int, int]]:
    if not token_ids:
        raise HeadingChunkingGateError("A logical section produced zero content tokens")
    windows: list[tuple[int, int]] = []
    start = 0
    while start < len(token_ids):
        end = min(start + CHUNK_SIZE, len(token_ids))
        windows.append((start, end))
        if end == len(token_ids):
            break
        next_start = end - CHUNK_OVERLAP
        if next_start <= start:
            raise HeadingChunkingGateError("Oversized-section splitter failed to make progress")
        start = next_start
    return windows


def _chunk_id(
    doc_id: str,
    section_index: int,
    subchunk_index: int,
    text_sha256: str,
) -> str:
    return (
        f"{doc_id}__{HEADING_CHUNK_STRATEGY}__s{section_index:04d}__"
        f"c{subchunk_index:04d}__{text_sha256}"
    )


def chunk_section(
    *,
    tokenizer: Any,
    manifest_row: dict[str, str],
    document_title: str,
    section: LogicalSection,
) -> list[HeadingChunkRecord]:
    token_ids = list(tokenizer.encode(section.content, add_special_tokens=False))
    windows = _content_windows(token_ids)
    prefix = structural_prefix(document_title, manifest_row["notice_code"], section.heading_path)
    chunks: list[HeadingChunkRecord] = []
    for subchunk_index, (start, end) in enumerate(windows):
        if len(windows) == 1:
            content_text = section.content
        else:
            content_text = tokenizer.decode(
                token_ids[start:end],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
        if not content_text.strip():
            raise HeadingChunkingGateError("Oversized-section splitting produced an empty chunk")
        standalone_count = len(tokenizer.encode(content_text, add_special_tokens=False))
        if standalone_count > CHUNK_SIZE:
            raise HeadingChunkingGateError(
                f"Standalone content exceeds {CHUNK_SIZE} tokens after deterministic decoding"
            )
        if len(windows) > 1 and standalone_count != end - start:
            raise HeadingChunkingGateError("Oversized-section token decode/encode is not exact")
        embedding_text = prefix + content_text
        text_hash = sha256_bytes(embedding_text.encode("utf-8"))
        chunk_id = _chunk_id(
            manifest_row["doc_id"], section.section_index, subchunk_index, text_hash
        )
        metadata: dict[str, Any] = {
            "doc_id": manifest_row["doc_id"],
            "notice_code": manifest_row["notice_code"],
            "notice_family": manifest_row["notice_family"],
            "title": manifest_row["title"],
            "source_url": manifest_row["source_url"],
            "source_origin": manifest_row["source_origin"],
            "chunk_id": chunk_id,
            "chunk_strategy": HEADING_CHUNK_STRATEGY,
            "heading": section.heading,
            "heading_path": list(section.heading_path),
            "section_index": section.section_index,
            "subchunk_index": subchunk_index,
        }
        if tuple(metadata) != HEADING_LOCAL_METADATA_KEYS:
            raise HeadingChunkingGateError("Heading-aware metadata schema is not deterministic")
        chunks.append(
            HeadingChunkRecord(
                chunk_id=chunk_id,
                text=embedding_text,
                content_text=content_text,
                text_sha256=text_hash,
                content_sha256=sha256_bytes(content_text.encode("utf-8")),
                content_token_start=start,
                content_token_end=end,
                content_token_count=standalone_count,
                embedding_token_count=len(tokenizer.encode(embedding_text, add_special_tokens=False)),
                section_content_token_count=len(token_ids),
                section_subchunk_count=len(windows),
                source_heading_line=section.heading_line,
                metadata=metadata,
            )
        )
    for left, right in zip(chunks, chunks[1:]):
        if right.content_token_start != left.content_token_end - CHUNK_OVERLAP:
            raise HeadingChunkingGateError("Oversized-section overlap is not exactly 40 source tokens")
        if left.metadata["section_index"] != right.metadata["section_index"]:
            raise HeadingChunkingGateError("An overlap crossed a logical heading boundary")
    return chunks


def _read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    required = {
        "doc_id",
        "notice_code",
        "notice_family",
        "title",
        "source_url",
        "source_origin",
        "content_hash",
    }
    if len(rows) != 50 or any(not required.issubset(row) for row in rows):
        raise HeadingChunkingGateError("Frozen manifest schema/count is invalid")
    if len({row["doc_id"] for row in rows}) != 50:
        raise HeadingChunkingGateError("Frozen manifest doc IDs are not unique")
    return rows


def _stats(values: Sequence[int]) -> dict[str, int | float]:
    if not values:
        raise HeadingChunkingGateError("Cannot calculate chunk statistics for an empty sequence")
    return {
        "min": min(values),
        "median": float(statistics.median(values)),
        "p95": nearest_rank_percentile(values, 0.95),
        "max": max(values),
        "p95_method": "nearest_rank",
    }


def build_heading_chunks(
    *,
    project_root: Path,
    tokenizer: Any,
    frozen_inputs: dict[str, Any],
) -> tuple[list[HeadingChunkRecord], dict[str, Any]]:
    manifest_path = project_root / "data" / "corpus_manifest.csv"
    rows = _read_manifest(manifest_path)
    chunks: list[HeadingChunkRecord] = []
    per_doc_counts: dict[str, int] = {}
    heading_inventory = Counter()
    total_sections = 0
    nonempty_sections = 0
    empty_body_headings = 0
    oversized_sections = 0
    oversized_section_chunks = 0
    unassigned_pre_h2_documents: list[str] = []
    markdown_h1_manifest_title_mismatches: list[str] = []
    source_hash_failures: list[str] = []

    for row in rows:
        doc_id = row["doc_id"]
        path = project_root / "data" / "processed" / "guidance" / f"{doc_id}.md"
        if not path.is_file() or sha256_file(path) != row["content_hash"]:
            source_hash_failures.append(doc_id)
            continue
        markdown = path.read_text(encoding="utf-8")
        markdown_h1, sections, inventory = parse_logical_sections(markdown)
        if markdown_h1 != row["title"]:
            markdown_h1_manifest_title_mismatches.append(doc_id)
        total_sections += inventory["h2_h6_heading_count"]
        nonempty_sections += inventory["nonempty_logical_section_count"]
        empty_body_headings += inventory["empty_body_heading_count"]
        if inventory["unassigned_useful_pre_h2_content"]:
            unassigned_pre_h2_documents.append(doc_id)
        heading_inventory.update(level for section in sections for level in section.heading_levels[-1:])
        doc_chunks: list[HeadingChunkRecord] = []
        for section in sections:
            section_chunks = chunk_section(
                tokenizer=tokenizer,
                manifest_row=row,
                document_title=row["title"],
                section=section,
            )
            if section_chunks[0].section_content_token_count > CHUNK_SIZE:
                oversized_sections += 1
                oversized_section_chunks += len(section_chunks)
            doc_chunks.extend(section_chunks)
        if not doc_chunks:
            raise HeadingChunkingGateError(f"Document {doc_id} produced no heading-aware chunks")
        per_doc_counts[doc_id] = len(doc_chunks)
        chunks.extend(doc_chunks)

    duplicate_ids = [
        chunk_id for chunk_id, count in Counter(chunk.chunk_id for chunk in chunks).items() if count > 1
    ]
    empty_chunks = [chunk.chunk_id for chunk in chunks if not chunk.content_text.strip()]
    no_heading = [chunk.chunk_id for chunk in chunks if not chunk.metadata["heading_path"]]
    oversized_chunks = [
        chunk.chunk_id for chunk in chunks if chunk.content_token_count > CHUNK_SIZE
    ]
    crossing_chunks: list[str] = []  # Impossible by construction: each call receives one section only.
    missing_docs = sorted(set(row["doc_id"] for row in rows) - set(per_doc_counts))
    content_counts = [chunk.content_token_count for chunk in chunks]
    embedding_counts = [chunk.embedding_token_count for chunk in chunks]
    counts_per_document = list(per_doc_counts.values())
    quality_gate = not any(
        (
            source_hash_failures,
            duplicate_ids,
            empty_chunks,
            no_heading,
            oversized_chunks,
            crossing_chunks,
            missing_docs,
        )
    ) and len(per_doc_counts) == 50

    audit = {
        "schema_version": "1.0",
        "generated_at_utc": utc_now(),
        "phase": "4B-heading-aware-chunking",
        "inputs": frozen_inputs,
        "chunker": {
            "chunk_strategy": HEADING_CHUNK_STRATEGY,
            "tokenizer_name": TOKENIZER_NAME,
            "tokenizer_revision": TOKENIZER_REVISION,
            "logical_boundaries": "nonempty H2-H6 section bodies",
            "document_title_source": "authoritative corpus manifest title",
            "markdown_h1_manifest_title_mismatch_count": len(markdown_h1_manifest_title_mismatches),
            "markdown_h1_manifest_title_mismatch_doc_ids": markdown_h1_manifest_title_mismatches,
            "content_max_tokens": CHUNK_SIZE,
            "oversized_section_overlap_tokens": CHUNK_OVERLAP,
            "oversized_section_stride_tokens": CHUNK_STRIDE,
            "small_section_merging": False,
            "structural_prefix": "document title, notice code, full heading path",
        },
        "documents_processed": len(per_doc_counts),
        "total_heading_aware_chunks": len(chunks),
        "chunks_per_document": {
            "min": min(counts_per_document),
            "median": float(statistics.median(counts_per_document)),
            "max": max(counts_per_document),
            "by_doc_id": per_doc_counts,
        },
        "content_tokens_per_chunk": _stats(content_counts),
        "embedding_tokens_per_chunk_including_prefix": _stats(embedding_counts),
        "logical_sections": {
            "h2_h6_heading_count": total_sections,
            "nonempty_section_count": nonempty_sections,
            "empty_body_heading_count_skipped": empty_body_headings,
            "unassigned_useful_pre_h2_content_count": len(unassigned_pre_h2_documents),
            "unassigned_useful_pre_h2_document_ids": unassigned_pre_h2_documents,
            "terminal_heading_level_counts_for_nonempty_sections": {
                f"H{level}": heading_inventory[level] for level in sorted(heading_inventory)
            },
            "oversized_sections_requiring_subchunking": oversized_sections,
            "chunks_emitted_from_oversized_sections": oversized_section_chunks,
        },
        "integrity": {
            "chunks_with_no_heading_path": {"count": len(no_heading), "ids": no_heading},
            "duplicate_chunk_ids": {"count": len(duplicate_ids), "ids": duplicate_ids},
            "empty_chunks": {"count": len(empty_chunks), "ids": empty_chunks},
            "content_chunks_over_220_tokens": {"count": len(oversized_chunks), "ids": oversized_chunks},
            "chunks_crossing_heading_boundaries": {"count": len(crossing_chunks), "ids": crossing_chunks},
            "boundary_proof": "every chunk window is produced from one LogicalSection content token sequence",
            "source_hash_failures": source_hash_failures,
            "missing_document_ids": missing_docs,
        },
        "quality_gate_passed": quality_gate,
    }
    if not quality_gate:
        raise HeadingChunkingGateError("Heading-aware chunk quality gate failed")
    return chunks, audit


def iter_heading_jsonl(records: Iterable[HeadingChunkRecord]) -> Iterable[str]:
    for record in records:
        yield json.dumps(record.as_dict(), ensure_ascii=False, separators=(",", ":"))


def write_heading_registry(path: Path, records: Sequence[HeadingChunkRecord]) -> None:
    atomic_write_text(path, "\n".join(iter_heading_jsonl(records)) + "\n")


def load_heading_registry(path: Path) -> list[HeadingChunkRecord]:
    records: list[HeadingChunkRecord] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            metadata = value["metadata"]
            if tuple(metadata) != HEADING_LOCAL_METADATA_KEYS:
                raise HeadingChunkingGateError(
                    f"Heading registry line {line_number} has unexpected metadata"
                )
            record = HeadingChunkRecord(
                chunk_id=value["chunk_id"],
                text=value["text"],
                content_text=value["content_text"],
                text_sha256=value["text_sha256"],
                content_sha256=value["content_sha256"],
                content_token_start=int(value["content_token_start"]),
                content_token_end=int(value["content_token_end"]),
                content_token_count=int(value["content_token_count"]),
                embedding_token_count=int(value["embedding_token_count"]),
                section_content_token_count=int(value["section_content_token_count"]),
                section_subchunk_count=int(value["section_subchunk_count"]),
                source_heading_line=int(value["source_heading_line"]),
                metadata=metadata,
            )
            if record.chunk_id != metadata["chunk_id"]:
                raise HeadingChunkingGateError(f"Heading registry line {line_number} has inconsistent IDs")
            if sha256_bytes(record.text.encode("utf-8")) != record.text_sha256:
                raise HeadingChunkingGateError(f"Heading registry line {line_number} has a text hash mismatch")
            if sha256_bytes(record.content_text.encode("utf-8")) != record.content_sha256:
                raise HeadingChunkingGateError(f"Heading registry line {line_number} has a content hash mismatch")
            expected_id = _chunk_id(
                str(metadata["doc_id"]),
                int(metadata["section_index"]),
                int(metadata["subchunk_index"]),
                record.text_sha256,
            )
            if record.chunk_id != expected_id:
                raise HeadingChunkingGateError(f"Heading registry line {line_number} has a non-deterministic ID")
            if (
                not record.text.strip()
                or not record.content_text.strip()
                or not 0 < record.content_token_count <= CHUNK_SIZE
                or record.content_token_start < 0
                or record.content_token_end <= record.content_token_start
                or record.content_token_end > record.section_content_token_count
                or record.section_subchunk_count <= 0
                or metadata["chunk_strategy"] != HEADING_CHUNK_STRATEGY
                or metadata["heading"] != metadata["heading_path"][-1]
            ):
                raise HeadingChunkingGateError(f"Heading registry line {line_number} violates its contract")
            records.append(record)
    if not records or len({record.chunk_id for record in records}) != len(records):
        raise HeadingChunkingGateError("Heading registry is empty or contains duplicate IDs")
    groups: dict[tuple[str, int], list[HeadingChunkRecord]] = defaultdict(list)
    for record in records:
        groups[(str(record.metadata["doc_id"]), int(record.metadata["section_index"]))].append(record)
    for key, values in groups.items():
        ordered = sorted(values, key=lambda record: int(record.metadata["subchunk_index"]))
        if [int(record.metadata["subchunk_index"]) for record in ordered] != list(range(len(ordered))):
            raise HeadingChunkingGateError(f"Heading registry has noncontiguous subchunks for {key}")
        if any(record.section_subchunk_count != len(ordered) for record in ordered):
            raise HeadingChunkingGateError(f"Heading registry subchunk total is inconsistent for {key}")
        for left, right in zip(ordered, ordered[1:]):
            if right.content_token_start != left.content_token_end - CHUNK_OVERLAP:
                raise HeadingChunkingGateError(f"Heading registry overlap is invalid for {key}")
    return records


def write_heading_audit(path: Path, audit: dict[str, Any]) -> None:
    atomic_write_json(path, audit)
