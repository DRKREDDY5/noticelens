"""Phase 1 acquisition, HTML cleaning, and corpus quality validation.

The two checked-in CSV manifests are the sole source of network URLs. This
module deliberately uses only the Python standard library so Phase 1 does not
pull in any retrieval or application dependencies.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import statistics
import time
import unicodedata
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Iterable, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin, urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener


IRS_HOST = "www.irs.gov"
USER_AGENT = "NoticeLens/0.1 (educational corpus acquisition)"
CLEANER_VERSION = "phase1-html-v2"
GUIDANCE_REQUIRED_COLUMNS = (
    "doc_id",
    "notice_code",
    "notice_family",
    "title",
    "source_url",
    "source_origin",
    "source_format",
    "heading_structure_present",
    "verified_loaded",
    "retrieved_at",
    "priority_tier",
)
GUIDANCE_OUTPUT_COLUMNS = (
    "download_status",
    "cleaned_character_count",
    "headings_found",
    "content_hash",
)
SAMPLE_REQUIRED_COLUMNS = (
    "notice_code",
    "filename",
    "source_url",
    "language",
    "verification_status",
)
MANDATORY_FAMILIES = {
    "balance_collection": ("CP14", "CP501", "CP503", "CP504", "LT11"),
    "underreporter_deficiency": ("CP2501", "CP2000", "CP3219A"),
    "installment_agreement": ("CP521", "CP523"),
    "non_filer": ("CP59", "CP515", "CP518", "CP2566", "CP3219N"),
}
NAVIGATION_MARKERS = (
    "skip to main content",
    "main navigation",
    "footer navigation",
    "search irs.gov",
    "submit search",
)
IGNORED_TAGS = {"script", "style", "noscript", "svg", "template"}
BLOCK_TAGS = {"h2", "h3", "h4", "p", "li", "dt", "dd", "blockquote", "pre"}
INLINE_TAGS = {"a", "abbr", "b", "button", "code", "em", "i", "small", "span", "strong", "sub", "sup", "time", "u"}


class ManifestError(ValueError):
    """Raised when an authoritative manifest violates its expected schema."""


@dataclass(frozen=True)
class ContentBlock:
    kind: str
    text: str
    list_depth: int = 0
    marker: str = "-"
    is_faq: bool = False


@dataclass
class CleanResult:
    doc_id: str
    title: str
    page_title_found: bool
    canonical_url: str
    markdown: str
    character_count: int
    content_hash: str
    heading_counts: dict[str, int]
    faq_questions: int
    paragraph_count: int
    list_item_count: int
    source_list_item_count: int
    table_count: int
    source_text_token_coverage: float
    problems: list[str] = field(default_factory=list)

    @property
    def headings_found(self) -> int:
        """Measured H2/H3 count used in the manifest."""

        return self.heading_counts.get("h2", 0) + self.heading_counts.get("h3", 0)


@dataclass
class DownloadResult:
    key: str
    destination: Path
    ok: bool
    byte_count: int = 0
    content_type: str = ""
    payload_hash: str = ""
    final_url: str = ""
    mode: str = ""
    observed_at: str = ""
    error: str = ""


def _normalise_text(value: str) -> str:
    value = unicodedata.normalize("NFKC", value)
    value = value.replace("\u200b", "").replace("\u00ad", "")
    value = re.sub(r"\s+", " ", value).strip()
    value = re.sub(r"\[\s+", "[", value)
    value = re.sub(r"\s+\]", "]", value)
    return value


class IRSArticleParser(HTMLParser):
    """Extract only the IRS article, retaining its ordered block structure."""

    def __init__(self, source_url: str) -> None:
        super().__init__(convert_charrefs=True)
        self.source_url = source_url
        self.in_article = False
        self.article_depth = 0
        self.ignored_depth = 0
        self.page_title = ""
        self.canonical_url = ""
        self._title_active = False
        self._title_parts: list[str] = []
        self.blocks: list[ContentBlock] = []
        self._block_tag: str | None = None
        self._block_kind: str | None = None
        self._block_parts: list[str] = []
        self._block_list_depth = 0
        self._block_marker = "-"
        self._block_is_faq = False
        self._link_stack: list[tuple[bool, str]] = []
        self._list_stack: list[dict[str, int | str]] = []
        self.article_text_parts: list[str] = []
        self._in_table = False
        self._table_rows: list[tuple[list[str], list[bool]]] = []
        self._table_row: list[str] | None = None
        self._table_row_headers: list[bool] | None = None
        self._table_cell_parts: list[str] | None = None
        self._table_cell_is_header = False
        self._table_link_stack: list[tuple[bool, str]] = []
        self._table_caption_parts: list[str] | None = None
        self.source_list_item_count = 0
        self.source_table_count = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        attributes = {key.lower(): value or "" for key, value in attrs}

        if tag == "link" and not self.canonical_url:
            relationships = set(attributes.get("rel", "").casefold().split())
            if "canonical" in relationships and attributes.get("href"):
                self.canonical_url = urljoin(self.source_url, attributes["href"])

        if tag in IGNORED_TAGS:
            self.ignored_depth += 1
            return
        if self.ignored_depth:
            return

        if tag == "h1" and not self.page_title and not self._title_active:
            self._title_active = True
            self._title_parts = []

        if tag == "article":
            self.article_depth += 1
            self.in_article = True
            return
        if not self.in_article:
            return

        if self._in_table:
            self._handle_table_start(tag, attributes)
            return

        if tag == "li":
            self.source_list_item_count += 1

        if self._block_tag == "__loose__" and tag not in INLINE_TAGS and tag != "br":
            self._finish_block()

        if tag == "table" and self._block_tag is None:
            self._in_table = True
            self.source_table_count += 1
            self._table_rows = []
            self._table_row = None
            self._table_row_headers = None
            self._table_caption_parts = None
            return

        if tag in {"ul", "ol"}:
            if self._block_tag == "li":
                self._finish_block()
            self._list_stack.append({"tag": tag, "counter": 0})
            return

        if tag in BLOCK_TAGS and self._block_tag is None:
            kind = tag
            if tag == "dt":
                kind = "h3"
            elif tag == "dd":
                kind = "p"
            self._block_tag = tag
            self._block_kind = kind
            self._block_parts = []
            self._link_stack = []
            self._block_list_depth = len(self._list_stack)
            self._block_marker = "-"
            classes = set(attributes.get("class", "").split())
            self._block_is_faq = "accordion-heading" in classes
            if tag == "li" and self._list_stack:
                current_list = self._list_stack[-1]
                if current_list["tag"] == "ol":
                    current_list["counter"] = int(current_list["counter"]) + 1
                    self._block_marker = f"{current_list['counter']}."
            return

        if self._block_tag is not None:
            if tag == "br":
                self._block_parts.append("<br>")
            elif tag == "a":
                href = attributes.get("href", "").strip()
                if href:
                    href = urljoin(self.source_url, href)
                    self._block_parts.append("[")
                    self._link_stack.append((True, href))
                else:
                    self._link_stack.append((False, ""))
            elif self._block_tag == "li" and tag in {"div", "p"} and self._block_parts:
                if self._block_parts[-1] != "<br>":
                    self._block_parts.append("<br>")

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs)
        self.handle_endtag(tag)

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in IGNORED_TAGS:
            if self.ignored_depth:
                self.ignored_depth -= 1
            return
        if self.ignored_depth:
            return

        if tag == "h1" and self._title_active:
            self.page_title = _normalise_text("".join(self._title_parts))
            self._title_active = False

        if not self.in_article:
            return

        if self._in_table:
            self._handle_table_end(tag)
            return

        if tag == "a" and self._block_tag is not None and self._link_stack:
            has_link, href = self._link_stack.pop()
            if has_link:
                self._block_parts.append(f"]({href})")

        if self._block_tag == tag:
            self._finish_block()
        elif self._block_tag == "__loose__" and tag not in INLINE_TAGS:
            self._finish_block()

        if tag in {"ul", "ol"} and self._list_stack:
            self._list_stack.pop()

        if tag == "article":
            self.article_depth -= 1
            if self.article_depth <= 0:
                self.article_depth = 0
                self.in_article = False

    def handle_data(self, data: str) -> None:
        if self.ignored_depth:
            return
        if self._title_active:
            self._title_parts.append(data)
        if self.in_article:
            self.article_text_parts.append(data)
            if self._in_table:
                if self._table_cell_parts is not None:
                    self._table_cell_parts.append(data)
                elif self._table_caption_parts is not None:
                    self._table_caption_parts.append(data)
            elif self._block_tag is not None:
                self._block_parts.append(data)
            elif data.strip():
                self._block_tag = "__loose__"
                self._block_kind = "p"
                self._block_parts = [data]

    def close(self) -> None:
        super().close()
        if self._block_tag is not None:
            self._finish_block()
        if self._title_active and not self.page_title:
            self.page_title = _normalise_text("".join(self._title_parts))
            self._title_active = False

    def _finish_block(self) -> None:
        text = _normalise_text("".join(self._block_parts))
        if text and self._block_kind:
            block = ContentBlock(
                kind=self._block_kind,
                text=text,
                list_depth=self._block_list_depth,
                marker=self._block_marker,
                is_faq=self._block_is_faq,
            )
            if not self.blocks or self.blocks[-1] != block:
                self.blocks.append(block)
        self._block_tag = None
        self._block_kind = None
        self._block_parts = []
        self._link_stack = []
        self._block_list_depth = 0
        self._block_marker = "-"
        self._block_is_faq = False

    def _handle_table_start(self, tag: str, attributes: dict[str, str]) -> None:
        if tag == "tr":
            self._table_row = []
            self._table_row_headers = []
        elif tag in {"th", "td"}:
            self._table_cell_parts = []
            self._table_cell_is_header = tag == "th"
            self._table_link_stack = []
        elif tag == "caption":
            self._table_caption_parts = []
        elif tag == "br" and self._table_cell_parts is not None:
            self._table_cell_parts.append("<br>")
        elif tag == "a" and self._table_cell_parts is not None:
            href = attributes.get("href", "").strip()
            if href:
                self._table_cell_parts.append("[")
                self._table_link_stack.append((True, urljoin(self.source_url, href)))
            else:
                self._table_link_stack.append((False, ""))

    def _handle_table_end(self, tag: str) -> None:
        if tag == "a" and self._table_cell_parts is not None and self._table_link_stack:
            has_link, href = self._table_link_stack.pop()
            if has_link:
                self._table_cell_parts.append(f"]({href})")
            return
        if tag in {"th", "td"} and self._table_cell_parts is not None:
            value = _normalise_text("".join(self._table_cell_parts)).replace("|", r"\|")
            if self._table_row is None:
                self._table_row = []
                self._table_row_headers = []
            self._table_row.append(value)
            assert self._table_row_headers is not None
            self._table_row_headers.append(self._table_cell_is_header)
            self._table_cell_parts = None
            self._table_cell_is_header = False
            self._table_link_stack = []
            return
        if tag == "tr" and self._table_row is not None:
            if any(self._table_row):
                self._table_rows.append((self._table_row, self._table_row_headers or []))
            self._table_row = None
            self._table_row_headers = None
            return
        if tag == "caption" and self._table_caption_parts is not None:
            caption = _normalise_text("".join(self._table_caption_parts))
            if caption:
                self.blocks.append(ContentBlock(kind="p", text=caption))
            self._table_caption_parts = None
            return
        if tag == "table":
            self._finish_table()

    def _finish_table(self) -> None:
        if self._table_row is not None and any(self._table_row):
            self._table_rows.append((self._table_row, self._table_row_headers or []))
        if self._table_rows:
            column_count = max(len(cells) for cells, _ in self._table_rows)
            first_cells, first_headers = self._table_rows[0]
            if any(first_headers):
                header = first_cells + [""] * (column_count - len(first_cells))
                data_rows = self._table_rows[1:]
            else:
                header = [f"Column {index}" for index in range(1, column_count + 1)]
                data_rows = self._table_rows
            lines = [
                "| " + " | ".join(header) + " |",
                "| " + " | ".join("---" for _ in range(column_count)) + " |",
            ]
            for cells, _ in data_rows:
                padded = cells + [""] * (column_count - len(cells))
                lines.append("| " + " | ".join(padded) + " |")
            self.blocks.append(ContentBlock(kind="table", text="\n".join(lines)))
        self._in_table = False
        self._table_rows = []
        self._table_row = None
        self._table_row_headers = None
        self._table_cell_parts = None
        self._table_link_stack = []
        self._table_caption_parts = None


def _decode_html(payload: bytes) -> str:
    head = payload[:4096].decode("ascii", errors="ignore")
    match = re.search(r"charset\s*=\s*['\"]?([\w.-]+)", head, flags=re.IGNORECASE)
    candidates = [match.group(1)] if match else []
    candidates.extend(["utf-8", "windows-1252"])
    for encoding in candidates:
        try:
            return payload.decode(encoding)
        except (UnicodeDecodeError, LookupError):
            continue
    return payload.decode("utf-8", errors="replace")


def _ensure_blank_line(lines: list[str]) -> None:
    if lines and lines[-1] != "":
        lines.append("")


def _render_markdown(title: str, source_url: str, blocks: Sequence[ContentBlock]) -> str:
    lines = [f"# {title}", "", f"Source: {source_url}"]
    previous_kind = ""
    for block in blocks:
        if block.kind in {"h2", "h3", "h4"}:
            _ensure_blank_line(lines)
            level = int(block.kind[-1])
            lines.append(f"{'#' * level} {block.text}")
        elif block.kind == "li":
            if previous_kind != "li":
                _ensure_blank_line(lines)
            indent = "  " * max(0, block.list_depth - 1)
            lines.append(f"{indent}{block.marker} {block.text}")
        elif block.kind == "blockquote":
            _ensure_blank_line(lines)
            lines.append(f"> {block.text}")
        elif block.kind == "pre":
            _ensure_blank_line(lines)
            lines.extend(["```", block.text, "```"])
        elif block.kind == "table":
            _ensure_blank_line(lines)
            lines.extend(block.text.splitlines())
        else:
            _ensure_blank_line(lines)
            lines.append(block.text)
        previous_kind = block.kind
    return "\n".join(lines).strip() + "\n"


def clean_html(
    payload: bytes,
    *,
    doc_id: str,
    source_url: str,
    fallback_title: str,
    expected_notice_code: str = "",
) -> CleanResult:
    """Convert a raw IRS notice-guidance page into deterministic Markdown."""

    parser = IRSArticleParser(source_url)
    parser.feed(_decode_html(payload))
    parser.close()
    title = parser.page_title or _normalise_text(fallback_title)
    markdown = _render_markdown(title, source_url, parser.blocks)
    heading_counts = Counter(block.kind for block in parser.blocks if block.kind in {"h2", "h3", "h4"})
    faq_questions = sum(block.is_faq for block in parser.blocks)
    paragraph_count = sum(block.kind == "p" for block in parser.blocks)
    list_item_count = sum(block.kind == "li" for block in parser.blocks)
    table_count = sum(block.kind == "table" for block in parser.blocks)
    raw_tokens = Counter(re.findall(r"[\w]+", _normalise_text(" ".join(parser.article_text_parts)).casefold()))
    captured_text = " ".join(block.text for block in parser.blocks)
    captured_text = re.sub(r"\]\([^)]+\)", "]", captured_text)
    captured_tokens = Counter(re.findall(r"[\w]+", _normalise_text(captured_text).casefold()))
    covered_tokens = sum(min(count, captured_tokens[token]) for token, count in raw_tokens.items())
    source_text_token_coverage = covered_tokens / sum(raw_tokens.values()) if raw_tokens else 0.0
    problems: list[str] = []

    if not parser.page_title:
        problems.append("page H1 title was not found; manifest title used")
    if not parser.canonical_url:
        problems.append("canonical page URL was not found")
    elif parser.canonical_url.rstrip("/") != source_url.rstrip("/"):
        problems.append(f"canonical URL does not match manifest URL: {parser.canonical_url}")
    if not parser.blocks:
        problems.append("IRS article element contained no supported content blocks")
    if len(markdown) < 500:
        problems.append(f"cleaned content is too short ({len(markdown)} characters)")
    if heading_counts.get("h2", 0) == 0:
        problems.append("no H2 headings were preserved")
    if list_item_count != parser.source_list_item_count:
        problems.append(
            f"preserved {list_item_count} of {parser.source_list_item_count} source list items"
        )
    if table_count != parser.source_table_count:
        problems.append(f"preserved {table_count} of {parser.source_table_count} source tables")
    if expected_notice_code:
        expected_aliases = _notice_aliases(expected_notice_code)
        searchable = re.sub(r"[^A-Z0-9]", "", f"{title} {markdown}".upper())
        missing_aliases = sorted(alias for alias in expected_aliases if alias not in searchable)
        if missing_aliases:
            problems.append("expected notice code not found in cleaned page: " + ", ".join(missing_aliases))
    if source_text_token_coverage < 0.98:
        problems.append(f"only {source_text_token_coverage:.1%} of source article tokens were preserved")
    if "\ufffd" in markdown:
        problems.append("cleaned content contains Unicode replacement characters")
    lower_markdown = markdown.casefold()
    nav_hits = [marker for marker in NAVIGATION_MARKERS if marker in lower_markdown]
    if nav_hits:
        problems.append("possible global navigation contamination: " + ", ".join(nav_hits))

    rendered_h2 = len(re.findall(r"(?m)^## [^#]", markdown))
    rendered_h3 = len(re.findall(r"(?m)^### [^#]", markdown))
    if rendered_h2 != heading_counts.get("h2", 0) or rendered_h3 != heading_counts.get("h3", 0):
        problems.append("rendered heading counts do not match extracted H2/H3 counts")

    return CleanResult(
        doc_id=doc_id,
        title=title,
        page_title_found=bool(parser.page_title),
        canonical_url=parser.canonical_url,
        markdown=markdown,
        character_count=len(markdown),
        content_hash=hashlib.sha256(markdown.encode("utf-8")).hexdigest(),
        heading_counts=dict(heading_counts),
        faq_questions=faq_questions,
        paragraph_count=paragraph_count,
        list_item_count=list_item_count,
        source_list_item_count=parser.source_list_item_count,
        table_count=table_count,
        source_text_token_coverage=source_text_token_coverage,
        problems=problems,
    )


def _read_manifest(path: Path, required_columns: Sequence[str]) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        missing = [column for column in required_columns if column not in fieldnames]
        if missing:
            raise ManifestError(f"{path} is missing columns: {', '.join(missing)}")
        rows = [{key: (value or "").strip() for key, value in row.items()} for row in reader]
    if not rows:
        raise ManifestError(f"{path} contains no data rows")
    return rows, fieldnames


def _validate_irs_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme.lower() != "https" or (parsed.hostname or "").lower() != IRS_HOST:
        raise ManifestError(f"URL is not an allowed IRS HTTPS URL: {url}")
    if parsed.username or parsed.password or parsed.port not in (None, 443):
        raise ManifestError(f"URL contains unsupported authority components: {url}")


def _same_source_url(requested_url: str, final_url: str) -> bool:
    requested = urlparse(requested_url)
    final = urlparse(final_url)
    return (
        requested.scheme.casefold(),
        (requested.hostname or "").casefold(),
        requested.path.rstrip("/") or "/",
        requested.query,
    ) == (
        final.scheme.casefold(),
        (final.hostname or "").casefold(),
        final.path.rstrip("/") or "/",
        final.query,
    )


def _validate_manifests(
    guidance_rows: Sequence[dict[str, str]],
    sample_rows: Sequence[dict[str, str]],
) -> None:
    if len(guidance_rows) != 50:
        raise ManifestError(f"expected 50 guidance rows, found {len(guidance_rows)}")
    for row_number, row in enumerate(guidance_rows, start=2):
        blanks = [column for column in GUIDANCE_REQUIRED_COLUMNS if not row[column]]
        if blanks:
            raise ManifestError(f"guidance row {row_number} has blank fields: {', '.join(blanks)}")
    ids = [row["doc_id"] for row in guidance_rows]
    urls = [row["source_url"] for row in guidance_rows]
    if len(set(ids)) != len(ids):
        raise ManifestError("guidance manifest contains duplicate doc_id values")
    if len(set(urls)) != len(urls):
        raise ManifestError("guidance manifest contains duplicate source_url values")
    for row in guidance_rows:
        if not re.fullmatch(r"irs_[a-z0-9_]+", row["doc_id"]):
            raise ManifestError(f"unsafe doc_id: {row['doc_id']}")
        _validate_irs_url(row["source_url"])
        if row["source_origin"] != "IRS" or row["source_format"].casefold() != "html":
            raise ManifestError(f"unexpected source metadata for {row['doc_id']}")
        if row["heading_structure_present"].casefold() not in {"yes", "no"}:
            raise ManifestError(f"invalid heading_structure_present for {row['doc_id']}")
        if row["verified_loaded"].casefold() not in {"yes", "no"}:
            raise ManifestError(f"invalid verified_loaded for {row['doc_id']}")
        if row["priority_tier"].casefold() not in {"core", "general"}:
            raise ManifestError(f"invalid priority_tier for {row['doc_id']}")
        try:
            date.fromisoformat(row["retrieved_at"])
        except ValueError as exc:
            raise ManifestError(f"invalid retrieved_at for {row['doc_id']}") from exc

    sample_urls = [row["source_url"] for row in sample_rows]
    filenames = [row["filename"] for row in sample_rows]
    sample_codes = [row["notice_code"] for row in sample_rows]
    if len({url.casefold() for url in sample_urls}) != len(sample_urls):
        raise ManifestError("sample manifest contains duplicate URL values")
    if len({filename.casefold() for filename in filenames}) != len(filenames):
        raise ManifestError("sample manifest contains duplicate filename values")
    if len({code.casefold() for code in sample_codes}) != len(sample_codes):
        raise ManifestError("sample manifest contains duplicate notice_code values")
    for row_number, row in enumerate(sample_rows, start=2):
        blanks = [column for column in SAMPLE_REQUIRED_COLUMNS if not row[column]]
        if blanks:
            raise ManifestError(f"sample row {row_number} has blank fields: {', '.join(blanks)}")
        _validate_irs_url(row["source_url"])
        filename = row["filename"]
        if Path(filename).name != filename or not re.fullmatch(r"[A-Za-z0-9_.-]+\.pdf", filename, re.IGNORECASE):
            raise ManifestError(f"unsafe sample filename: {filename}")
        if Path(urlparse(row["source_url"]).path).name != filename:
            raise ManifestError(f"sample filename does not match URL: {filename}")
        if row["verification_status"].casefold() != "verified":
            raise ManifestError(f"sample is not verified: {filename}")


class _RestrictedRedirectHandler(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[no-untyped-def]
        _validate_irs_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _download_once(url: str, *, expected_kind: str, timeout: float) -> tuple[bytes, str, str]:
    _validate_irs_url(url)
    request = Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "text/html,application/xhtml+xml" if expected_kind == "html" else "application/pdf",
            "Accept-Language": "en-US,en;q=0.9",
        },
    )
    opener = build_opener(_RestrictedRedirectHandler())
    with opener.open(request, timeout=timeout) as response:
        final_url = response.geturl()
        _validate_irs_url(final_url)
        if not _same_source_url(url, final_url):
            raise OSError(f"unexpected redirect target: {final_url}")
        status = response.getcode()
        if status != 200:
            raise OSError(f"HTTP status {status}")
        content_type = response.headers.get_content_type().lower()
        max_bytes = 5_000_000 if expected_kind == "html" else 30_000_000
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = response.read(64 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > max_bytes:
                raise OSError(f"response exceeds {max_bytes} byte limit")
            chunks.append(chunk)
    payload = b"".join(chunks)
    if expected_kind == "html":
        if content_type not in {"text/html", "application/xhtml+xml"}:
            raise OSError(f"unexpected HTML content type: {content_type}")
        if len(payload) < 1_000 or b"<html" not in payload[:10_000].lower():
            raise OSError("response does not look like meaningful HTML")
    else:
        if len(payload) < 1_000 or not payload.startswith(b"%PDF-"):
            raise OSError("response does not have a valid PDF signature")
        if b"%%EOF" not in payload[-2048:]:
            raise OSError("PDF response is missing a trailing EOF marker")
        if content_type not in {"application/pdf", "application/octet-stream"}:
            raise OSError(f"unexpected PDF content type: {content_type}")
    return payload, content_type, final_url


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def _download_with_retries(
    key: str,
    url: str,
    destination: Path,
    expected_kind: str,
    *,
    timeout: float = 45.0,
    attempts: int = 3,
) -> DownloadResult:
    last_error = ""
    for attempt in range(1, attempts + 1):
        try:
            payload, content_type, final_url = _download_once(
                url,
                expected_kind=expected_kind,
                timeout=timeout,
            )
            _atomic_write_bytes(destination, payload)
            return DownloadResult(
                key=key,
                destination=destination,
                ok=True,
                byte_count=len(payload),
                content_type=content_type,
                payload_hash=hashlib.sha256(payload).hexdigest(),
                final_url=final_url,
                mode="network",
                observed_at=datetime.now(timezone.utc).isoformat(),
            )
        except (HTTPError, URLError, TimeoutError, OSError, ManifestError, ValueError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt < attempts:
                time.sleep(1.5 * attempt)
    return DownloadResult(
        key=key,
        destination=destination,
        ok=False,
        mode="network",
        observed_at=datetime.now(timezone.utc).isoformat(),
        error=last_error,
    )


def _download_batch(
    jobs: Sequence[tuple[str, str, Path, str]],
    *,
    workers: int,
) -> dict[str, DownloadResult]:
    results: dict[str, DownloadResult] = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_download_with_retries, key, url, destination, kind): key
            for key, url, destination, kind in jobs
        }
        for future in as_completed(futures):
            result = future.result()
            results[result.key] = result
            state = "ok" if result.ok else f"failed ({result.error})"
            print(f"[{len(results):02d}/{len(jobs):02d}] {result.key}: {state}", flush=True)
    return results


def _inspect_existing(
    key: str,
    path: Path,
    expected_kind: str,
    *,
    expected_hash: str,
) -> DownloadResult:
    try:
        payload = path.read_bytes()
    except OSError as exc:
        return DownloadResult(
            key=key,
            destination=path,
            ok=False,
            mode="cached",
            observed_at=datetime.now(timezone.utc).isoformat(),
            error=str(exc),
        )
    if expected_kind == "html":
        valid = len(payload) >= 1_000 and b"<html" in payload[:10_000].lower()
        content_type = "text/html"
    else:
        valid = (
            len(payload) >= 1_000
            and payload.startswith(b"%PDF-")
            and b"%%EOF" in payload[-2048:]
        )
        content_type = "application/pdf"
    payload_hash = hashlib.sha256(payload).hexdigest() if valid else ""
    if valid and not expected_hash:
        valid = False
        error = "no pinned acquisition hash is available for cached validation"
    elif valid and payload_hash != expected_hash:
        valid = False
        error = "cached file SHA-256 does not match the acquisition ledger"
    else:
        error = "" if valid else f"existing file is not valid {expected_kind}"
    return DownloadResult(
        key=key,
        destination=path,
        ok=valid,
        byte_count=len(payload),
        content_type=content_type,
        payload_hash=payload_hash if valid else "",
        mode="cached",
        observed_at=datetime.now(timezone.utc).isoformat(),
        error=error,
    )


def _read_acquisition_hashes(path: Path) -> dict[tuple[str, str], str]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return {
            (row.get("asset_type", ""), row.get("asset_id", "")): row.get("raw_content_hash", "")
            for row in reader
            if row.get("status") == "success"
        }


def _write_acquisition_ledger(
    path: Path,
    *,
    root: Path,
    guidance_rows: Sequence[dict[str, str]],
    sample_rows: Sequence[dict[str, str]],
    guidance_results: dict[str, DownloadResult],
    sample_results: dict[str, DownloadResult],
) -> None:
    fieldnames = [
        "asset_type",
        "asset_id",
        "requested_url",
        "final_url",
        "local_path",
        "status",
        "content_type",
        "byte_count",
        "raw_content_hash",
        "acquisition_mode",
        "observed_at",
    ]
    ledger_rows: list[dict[str, object]] = []
    for asset_type, rows, results, id_field in (
        ("guidance", guidance_rows, guidance_results, "doc_id"),
        ("sample_notice", sample_rows, sample_results, "notice_code"),
    ):
        for row in rows:
            asset_id = row[id_field]
            result = results[asset_id]
            ledger_rows.append(
                {
                    "asset_type": asset_type,
                    "asset_id": asset_id,
                    "requested_url": row["source_url"],
                    "final_url": result.final_url,
                    "local_path": result.destination.relative_to(root).as_posix(),
                    "status": "success" if result.ok else "failed",
                    "content_type": result.content_type,
                    "byte_count": result.byte_count,
                    "raw_content_hash": result.payload_hash,
                    "acquisition_mode": result.mode,
                    "observed_at": result.observed_at,
                }
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(ledger_rows)
    os.replace(temporary, path)


def _write_guidance_manifest(
    path: Path,
    rows: Sequence[dict[str, str]],
    original_fields: Sequence[str],
) -> None:
    fieldnames = list(original_fields)
    for column in GUIDANCE_OUTPUT_COLUMNS:
        if column not in fieldnames:
            fieldnames.append(column)
    temporary = path.with_name(path.name + ".part")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _body_fingerprint(markdown: str) -> str:
    lines = markdown.splitlines()
    body = "\n".join(line for line in lines if not line.startswith("# ") and not line.startswith("Source: ")).strip()
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def _notice_aliases(notice_code: str) -> set[str]:
    matches = re.findall(r"(?:CP|LT)\s*\d+[A-Z]?", notice_code.upper())
    return {re.sub(r"\s+", "", match) for match in matches}


def _mandatory_coverage(
    guidance_rows: Sequence[dict[str, str]],
    clean_results: dict[str, CleanResult],
) -> dict[str, dict[str, object]]:
    available_codes: set[str] = set()
    for row in guidance_rows:
        if row["doc_id"] in clean_results and not clean_results[row["doc_id"]].problems:
            available_codes.update(_notice_aliases(row["notice_code"]))
    coverage: dict[str, dict[str, object]] = {}
    for family, required in MANDATORY_FAMILIES.items():
        present = [code for code in required if code in available_codes]
        missing = [code for code in required if code not in available_codes]
        coverage[family] = {
            "required": list(required),
            "present": present,
            "missing": missing,
            "covered": not missing,
        }
    return coverage


def _duplicate_groups(clean_results: Iterable[CleanResult], *, body_only: bool) -> list[list[str]]:
    grouped: defaultdict[str, list[str]] = defaultdict(list)
    for result in clean_results:
        key = _body_fingerprint(result.markdown) if body_only else result.content_hash
        grouped[key].append(result.doc_id)
    return sorted((sorted(ids) for ids in grouped.values() if len(ids) > 1), key=lambda ids: ids[0])


def _atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def run_phase1(root: Path, *, skip_download: bool = False, workers: int = 4) -> dict[str, object]:
    root = root.resolve()
    guidance_manifest = root / "data" / "corpus_manifest.csv"
    sample_manifest = root / "data" / "sample_notice_manifest.csv"
    raw_guidance = root / "data" / "raw" / "guidance"
    processed_guidance = root / "data" / "processed" / "guidance"
    raw_samples = root / "data" / "raw" / "sample_notices"
    report_path = root / "reports" / "phase1_quality_report.json"
    acquisition_ledger_path = root / "reports" / "phase1_acquisition_ledger.csv"

    guidance_rows, guidance_fields = _read_manifest(guidance_manifest, GUIDANCE_REQUIRED_COLUMNS)
    sample_rows, _ = _read_manifest(sample_manifest, SAMPLE_REQUIRED_COLUMNS)
    _validate_manifests(guidance_rows, sample_rows)
    raw_guidance.mkdir(parents=True, exist_ok=True)
    processed_guidance.mkdir(parents=True, exist_ok=True)
    raw_samples.mkdir(parents=True, exist_ok=True)

    guidance_jobs = [
        (row["doc_id"], row["source_url"], raw_guidance / f"{row['doc_id']}.html", "html")
        for row in guidance_rows
    ]
    sample_jobs = [
        (row["notice_code"], row["source_url"], raw_samples / row["filename"], "pdf")
        for row in sample_rows
    ]

    if skip_download:
        pinned_hashes = _read_acquisition_hashes(acquisition_ledger_path)
        guidance_downloads = {
            key: _inspect_existing(
                key,
                destination,
                kind,
                expected_hash=pinned_hashes.get(("guidance", key), ""),
            )
            for key, _, destination, kind in guidance_jobs
        }
        sample_downloads = {
            key: _inspect_existing(
                key,
                destination,
                kind,
                expected_hash=pinned_hashes.get(("sample_notice", key), ""),
            )
            for key, _, destination, kind in sample_jobs
        }
    else:
        print("Downloading guidance pages from corpus_manifest.csv", flush=True)
        guidance_downloads = _download_batch(guidance_jobs, workers=workers)
        print("Downloading sample notices from sample_notice_manifest.csv", flush=True)
        sample_downloads = _download_batch(sample_jobs, workers=min(workers, len(sample_jobs)))
        _write_acquisition_ledger(
            acquisition_ledger_path,
            root=root,
            guidance_rows=guidance_rows,
            sample_rows=sample_rows,
            guidance_results=guidance_downloads,
            sample_results=sample_downloads,
        )

    clean_results: dict[str, CleanResult] = {}
    cleaning_failures: dict[str, str] = {}
    for row in guidance_rows:
        doc_id = row["doc_id"]
        download = guidance_downloads[doc_id]
        if download.ok:
            row["download_status"] = "downloaded" if download.mode == "network" else "validated_cached"
        else:
            row["download_status"] = "failed"
        row["cleaned_character_count"] = "0"
        row["headings_found"] = "0"
        row["content_hash"] = ""
        if not download.ok:
            cleaning_failures[doc_id] = f"download failed: {download.error}"
            continue
        try:
            result = clean_html(
                download.destination.read_bytes(),
                doc_id=doc_id,
                source_url=row["source_url"],
                fallback_title=row["title"],
                expected_notice_code=row["notice_code"],
            )
            output_path = processed_guidance / f"{doc_id}.md"
            _atomic_write_bytes(output_path, result.markdown.encode("utf-8"))
            clean_results[doc_id] = result
            row["cleaned_character_count"] = str(result.character_count)
            row["headings_found"] = str(result.headings_found)
            row["content_hash"] = result.content_hash
            if result.problems:
                cleaning_failures[doc_id] = "; ".join(result.problems)
        except (OSError, UnicodeError, ValueError) as exc:
            cleaning_failures[doc_id] = f"cleaning failed: {type(exc).__name__}: {exc}"

    _write_guidance_manifest(guidance_manifest, guidance_rows, guidance_fields)

    clean_values = list(clean_results.values())
    lengths = [result.character_count for result in clean_values]
    heading_totals = Counter()
    for result in clean_values:
        heading_totals.update(result.heading_counts)
    exact_duplicates = _duplicate_groups(clean_values, body_only=False)
    body_duplicates = _duplicate_groups(clean_values, body_only=True)
    coverage = _mandatory_coverage(guidance_rows, clean_results)
    failed_guidance = {
        key: result.error for key, result in guidance_downloads.items() if not result.ok
    }
    failed_samples = {
        key: result.error for key, result in sample_downloads.items() if not result.ok
    }
    expected_raw_guidance = {result.destination.name for result in guidance_downloads.values() if result.ok}
    expected_processed_guidance = {f"{doc_id}.md" for doc_id in clean_results}
    expected_sample_files = {result.destination.name for result in sample_downloads.values() if result.ok}
    actual_raw_guidance = {path.name for path in raw_guidance.iterdir() if path.is_file()}
    actual_processed_guidance = {path.name for path in processed_guidance.iterdir() if path.is_file()}
    actual_sample_files = {path.name for path in raw_samples.iterdir() if path.is_file()}
    reconciliation = {
        "missing_raw_guidance": sorted(expected_raw_guidance - actual_raw_guidance),
        "unexpected_or_stale_raw_guidance": sorted(actual_raw_guidance - expected_raw_guidance),
        "missing_processed_guidance": sorted(expected_processed_guidance - actual_processed_guidance),
        "unexpected_or_stale_processed_guidance": sorted(
            actual_processed_guidance - expected_processed_guidance
        ),
        "missing_sample_notices": sorted(expected_sample_files - actual_sample_files),
        "unexpected_or_stale_sample_notices": sorted(actual_sample_files - expected_sample_files),
    }
    artifacts_reconciled = not any(reconciliation.values())

    report: dict[str, object] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_policy": "URLs read only from the two authoritative manifests",
        "run_mode": "cached_validation" if skip_download else "network_acquisition",
        "acquisition_ledger": acquisition_ledger_path.relative_to(root).as_posix(),
        "guidance": {
            "manifest_rows": len(guidance_rows),
            "available": sum(result.ok for result in guidance_downloads.values()),
            "downloaded": sum(
                result.ok and result.mode == "network" for result in guidance_downloads.values()
            ),
            "validated_cached": sum(
                result.ok and result.mode == "cached" for result in guidance_downloads.values()
            ),
            "download_failures": failed_guidance,
            "successfully_cleaned": len(clean_results) - sum(bool(result.problems) for result in clean_values),
            "processed_files_written": len(clean_results),
            "cleaning_problems": cleaning_failures,
            "median_cleaned_length": statistics.median(lengths) if lengths else 0,
            "minimum_cleaned_length": min(lengths) if lengths else 0,
            "maximum_cleaned_length": max(lengths) if lengths else 0,
            "heading_counts": {
                "h1_titles": sum(result.page_title_found for result in clean_values),
                "h2": heading_totals.get("h2", 0),
                "h3": heading_totals.get("h3", 0),
                "h4": heading_totals.get("h4", 0),
                "h2_h3_total": heading_totals.get("h2", 0) + heading_totals.get("h3", 0),
                "faq_questions": sum(result.faq_questions for result in clean_values),
            },
            "paragraph_blocks_extracted": sum(result.paragraph_count for result in clean_values),
            "source_list_items": sum(result.source_list_item_count for result in clean_values),
            "list_items_preserved": sum(result.list_item_count for result in clean_values),
            "tables_preserved": sum(result.table_count for result in clean_values),
            "cleaner_version": CLEANER_VERSION,
            "source_text_token_coverage": {
                "minimum": min((result.source_text_token_coverage for result in clean_values), default=0),
                "median": statistics.median(
                    [result.source_text_token_coverage for result in clean_values]
                ) if clean_values else 0,
            },
        },
        "sample_notices": {
            "manifest_rows": len(sample_rows),
            "available": sum(result.ok for result in sample_downloads.values()),
            "downloaded": sum(
                result.ok and result.mode == "network" for result in sample_downloads.values()
            ),
            "validated_cached": sum(
                result.ok and result.mode == "cached" for result in sample_downloads.values()
            ),
            "download_failures": failed_samples,
            "kept_outside_permanent_guidance_corpus": True,
        },
        "mandatory_family_coverage": coverage,
        "duplicates": {
            "exact_cleaned_documents": exact_duplicates,
            "same_body_excluding_title_and_source": body_duplicates,
        },
        "artifact_reconciliation": reconciliation,
        "quality": {
            "cleaner_scope": "content inside the IRS article element only; global nav, search, side navigation, and footer excluded",
            "meaningful_content_threshold_characters": 500,
            "all_guidance_available": not failed_guidance and len(guidance_downloads) == 50,
            "all_pages_meaningful_and_heading_preserved": not cleaning_failures and len(clean_results) == 50,
            "no_duplicate_cleaned_documents": not exact_duplicates and not body_duplicates,
            "mandatory_families_complete": all(bool(details["covered"]) for details in coverage.values()),
            "all_sample_pdfs_available": not failed_samples and len(sample_downloads) == len(sample_rows),
            "artifacts_reconciled": artifacts_reconciled,
        },
    }
    quality = report["quality"]
    assert isinstance(quality, dict)
    report["quality_gate_passed"] = all(
        bool(quality[key])
        for key in (
            "all_guidance_available",
            "all_pages_meaningful_and_heading_preserved",
            "no_duplicate_cleaned_documents",
            "mandatory_families_complete",
            "all_sample_pdfs_available",
            "artifacts_reconciled",
        )
    )
    _atomic_write_json(report_path, report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run NoticeLens Phase 1 corpus acquisition and cleaning")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="project root (defaults to the repository containing src/)",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="clean and validate existing raw files without network access",
    )
    parser.add_argument("--workers", type=int, default=4, help="parallel download workers (default: 4)")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not 1 <= args.workers <= 8:
        raise SystemExit("--workers must be between 1 and 8")
    report = run_phase1(args.root, skip_download=args.skip_download, workers=args.workers)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["quality_gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
