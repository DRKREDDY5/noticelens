"""Local, deterministic PDF and IRS notice-field extraction for NoticeLens."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

from pypdf import PdfReader


MAX_PDF_BYTES = 12 * 1024 * 1024
MAX_PDF_PAGES = 25
MIN_USABLE_TEXT_CHARACTERS = 100

IdentityStatus = Literal["identified", "unidentified", "ambiguous"]


class NoticeInputError(RuntimeError):
    """Raised when an uploaded notice cannot be handled safely."""


@dataclass(frozen=True)
class ExtractedNotice:
    """Text extracted from a PDF without OCR or persistence."""

    display_name: str
    pages: tuple[str, ...]
    text: str
    extraction_method: str = "pypdf_text_layer"


@dataclass(frozen=True)
class NoticeIdentity:
    status: IdentityStatus
    notice_code: str | None
    retrieval_notice_code: str | None
    confidence: float
    evidence_text: str | None

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ExtractedField:
    value: str | None
    confidence: float
    source_text: str | None

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class NoticeFields:
    notice_code: ExtractedField
    notice_date: ExtractedField
    due_or_response_date: ExtractedField
    amount: ExtractedField
    reference_number: ExtractedField

    def as_dict(self) -> dict[str, dict[str, object]]:
        # ``asdict`` recursively converts each ExtractedField to a plain dict.
        return asdict(self)


# These aliases are reviewed, evidence-backed routing decisions. Never infer a
# generic suffix-stripping rule: CP01 and CP01A, for example, are distinct.
EXPLICIT_GUIDANCE_ROUTES: dict[str, str] = {
    "CP06": "CP06/CP06A",
    "CP06A": "CP06/CP06A",
    "CP12": "CP12 series",
    "CP12G": "CP12G/CP12U",
    "CP12U": "CP12G/CP12U",
    "CP24": "CP24 series",
    "CP400": "CP400/CP401",
    "CP401": "CP400/CP401",
    "CP2000": "CP2000 series",
    "CP2000A": "CP2000 series",
    "CP2000B": "CP2000 series",
    "CP2000C": "CP2000 series",
    "CP2000D": "CP2000 series",
    "CP2000E": "CP2000 series",
    "CP503C": "CP503",
    "CP523H": "CP523",
    "LT11": "LT11 / Letter 1058",
    "LETTER1058": "LT11 / Letter 1058",
}

_CODE_PATTERN = re.compile(
    r"(?ix)\b(?:"
    r"(?P<cp>CP\s*[- ]?\s*\d{1,4}\s*[A-Z]?)|"
    r"(?P<lt>LT\s*[- ]?\s*\d{1,4}\s*[A-Z]?)|"
    r"(?P<letter>(?:LTR|LETTER)\s*[- ]?\s*1058)"
    r")\b"
)
_HEADER_LABEL_PATTERN = re.compile(
    r"(?i)(?:IRS\s+)?(?:notice|letter)(?:\s+(?:number|type))?\s*[:#-]?\s*"
)
_MONTH_DATE = (
    r"(?:January|February|March|April|May|June|July|August|September|October|November|December)"
    r"\s+\d{1,2},\s+\d{4}"
)
_NUMERIC_DATE = r"\d{1,2}[/-]\d{1,2}[/-](?:\d{2}|\d{4})"
_DATE_VALUE = rf"(?:{_MONTH_DATE}|{_NUMERIC_DATE})"
_MONEY_VALUE = r"\$\s*(?:\d+|\d{1,3}(?:,\d{3})+)\.\d{2}"


def extract_pdf_text(path: Path) -> ExtractedNotice:
    """Extract an existing PDF text layer and fail rather than invoking OCR."""

    resolved = path.resolve()
    if not resolved.is_file():
        raise NoticeInputError("The notice PDF does not exist")
    size = resolved.stat().st_size
    if size <= 0 or size > MAX_PDF_BYTES:
        raise NoticeInputError("The notice PDF size is invalid or exceeds the local limit")
    try:
        with resolved.open("rb") as handle:
            if handle.read(5) != b"%PDF-":
                raise NoticeInputError("The uploaded file is not a PDF")
        reader = PdfReader(resolved, strict=True)
        if reader.is_encrypted:
            raise NoticeInputError("Encrypted notice PDFs are not supported")
        if not reader.pages or len(reader.pages) > MAX_PDF_PAGES:
            raise NoticeInputError("The notice PDF page count is invalid or exceeds the local limit")
        pages = tuple((page.extract_text() or "").replace("\x00", "") for page in reader.pages)
    except NoticeInputError:
        raise
    except Exception as exc:
        raise NoticeInputError(f"The notice PDF text layer could not be extracted ({type(exc).__name__})") from None
    text = "\n\n".join(pages)
    if len(re.sub(r"\s+", "", text)) < MIN_USABLE_TEXT_CHARACTERS:
        raise NoticeInputError("The notice PDF has no usable text layer; OCR was not attempted")
    return ExtractedNotice(display_name=resolved.name, pages=pages, text=text)


def normalize_notice_code(value: str) -> str | None:
    """Normalize supported public IRS code spellings without guessing suffixes."""

    normalized = unicodedata.normalize("NFKC", value).upper().strip()
    normalized = re.sub(r"[\s-]+", "", normalized)
    if re.fullmatch(r"(?:LTR|LETTER)1058", normalized):
        return "LETTER1058"
    if re.fullmatch(r"CP\d{1,4}[A-Z]?", normalized):
        return normalized
    if re.fullmatch(r"LT\d{1,4}[A-Z]?", normalized):
        return normalized
    return None


def route_notice_code(notice_code: str, available_guidance_codes: set[str]) -> str | None:
    """Resolve a detected code to an exact frozen Pinecone metadata value."""

    if notice_code in available_guidance_codes:
        return notice_code
    routed = EXPLICIT_GUIDANCE_ROUTES.get(notice_code)
    return routed if routed in available_guidance_codes else None


def _candidate_lines(page: str) -> list[tuple[int, str]]:
    compact = [re.sub(r"\s+", " ", line).strip() for line in page.splitlines() if line.strip()]
    # PDF layout extraction may emit many empty rows. Header proximity is
    # therefore measured using nonempty-line ordinals.
    return list(enumerate(compact, start=1))


def identify_notice(text: str, *, available_guidance_codes: set[str]) -> NoticeIdentity:
    """Identify a notice from labeled header occurrences, never from the question."""

    # Callers pass the first extracted page. Limiting the scan by line count is
    # safer than splitting at a blank line because PDF extractors often insert
    # blank lines inside a page header.
    lines = _candidate_lines(text)[:80]
    high_confidence: dict[str, tuple[float, str, int]] = {}
    for line_index, (line_number, line) in enumerate(lines):
        matches = list(_CODE_PATTERN.finditer(line))
        header_sequence = False
        prior_end = 0
        for match_index, match in enumerate(matches):
            raw = match.group(0)
            canonical = normalize_notice_code(raw)
            if canonical is None:
                continue
            prefix = line[: match.start()].strip()
            previous_line = lines[line_index - 1][1] if line_index else ""
            adjacent_label = bool(re.fullmatch(r"(?i)(?:notice|letter)", previous_line))
            directly_labeled = bool(_HEADER_LABEL_PATTERN.fullmatch(prefix)) or adjacent_label
            if match_index == 0:
                header_sequence = directly_labeled
            else:
                connector = line[prior_end : match.start()]
                header_sequence = header_sequence and bool(
                    re.fullmatch(r"(?i)\s*(?:/|,|and|&)\s*", connector)
                )
            labeled = directly_labeled or header_sequence
            near_header = line_number <= 40
            exact_code_line = normalize_notice_code(line) == canonical
            confidence = (
                0.99
                if labeled and near_header
                else (0.96 if exact_code_line and near_header else 0.55)
            )
            if confidence < 0.90:
                continue
            safe_evidence = f"Notice {canonical}" if labeled else canonical
            current = high_confidence.get(canonical)
            if current is None or line_number < current[2] or (
                line_number == current[2] and confidence > current[0]
            ):
                high_confidence[canonical] = (confidence, safe_evidence, line_number)
            prior_end = match.end()

    if not high_confidence:
        return NoticeIdentity("unidentified", None, None, 0.0, None)
    earliest_candidate_line = min(value[2] for value in high_confidence.values())
    high_confidence = {
        code: value
        for code, value in high_confidence.items()
        if value[2] <= earliest_candidate_line + 4
    }
    grouped: dict[str, list[tuple[str, float, str]]] = {}
    for code, (confidence, evidence, _) in high_confidence.items():
        # LT11, LTR 1058, and Letter 1058 are alternate public labels for one
        # notice. Other codes remain distinct even when routed to a shared
        # guidance page (for example CP06 and CP06A).
        identity_key = "LT11" if code == "LETTER1058" else code
        grouped.setdefault(identity_key, []).append((code, confidence, evidence))
    if len(grouped) > 1:
        evidence = " | ".join(value[1] for _, value in sorted(high_confidence.items()))
        return NoticeIdentity("ambiguous", None, None, max(value[0] for value in high_confidence.values()), evidence)
    candidates = next(iter(grouped.values()))
    notice_code, confidence, evidence = max(candidates, key=lambda item: (item[1], item[0] != "LETTER1058"))
    retrieval_code = route_notice_code(notice_code, available_guidance_codes)
    return NoticeIdentity("identified", notice_code, retrieval_code, confidence, evidence)


def _normalized_extraction_text(text: str) -> str:
    repaired = unicodedata.normalize("NFKC", text)
    repaired = re.sub(r"(?i)Noti\s*\n\s*ce", "Notice", repaired)
    repaired = re.sub(r"(?i)A\s+mount", "Amount", repaired)
    repaired = re.sub(r"\s+", " ", repaired)
    return repaired.strip()


def _unique_field_match(
    text: str,
    patterns: tuple[re.Pattern[str], ...],
    *,
    confidence: float,
    normalize_value: object | None = None,
) -> ExtractedField:
    matches: list[tuple[str, str]] = []
    for pattern in patterns:
        for match in pattern.finditer(text):
            value = match.group("value").strip()
            if callable(normalize_value):
                value = normalize_value(value)
                if value is None:
                    continue
            source = match.group(0).strip()[:280]
            matches.append((value, source))
    unique_values = {value for value, _ in matches}
    if len(unique_values) != 1:
        return ExtractedField(None, 0.0, None)
    value = next(iter(unique_values))
    source = next(source for candidate, source in matches if candidate == value)
    return ExtractedField(value, confidence, source)


def _canonical_date(value: str) -> str | None:
    for pattern in ("%B %d, %Y", "%m/%d/%Y", "%m-%d-%Y", "%m/%d/%y", "%m-%d-%y"):
        try:
            parsed = datetime.strptime(value, pattern)
            return f"{parsed.strftime('%B')} {parsed.day}, {parsed.year}"
        except ValueError:
            continue
    return None


def _canonical_amount(value: str) -> str:
    return "$" + value.replace("$", "").replace(" ", "")


_NOTICE_DATE_PATTERNS = (
    re.compile(rf"(?i)\bNotice\s+date\s*:?[ ]*(?P<value>{_DATE_VALUE})\b"),
)
_DUE_DATE_PATTERNS = (
    re.compile(rf"(?i)\b(?:must\s+pay|pay|respond|reply)\s+by\s+(?P<value>{_DATE_VALUE})\b"),
    re.compile(rf"(?i)\bAmount\s+due\s*,?\s*to\s+be\s+received\s+by\s+(?P<value>{_DATE_VALUE})\b"),
    re.compile(rf"(?i)\b(?:response|payment)\s+(?:is\s+)?due\s+(?:by\s+)?(?P<value>{_DATE_VALUE})\b"),
    re.compile(rf"(?i)\bSend\s+us\b[^.!?]{{0,180}}?\bby\s+(?P<value>{_DATE_VALUE})\b"),
    re.compile(rf"(?i)\b(?:request\s+a\s+hearing|Hearing,)\s*,?\s*by\s+(?P<value>{_DATE_VALUE})\b"),
)
_AMOUNT_PATTERNS = (
    re.compile(
        rf"(?i)\bAmount\s+due(?:\s*,?\s*to\s+be\s+received\s+by\s+{_DATE_VALUE})?\s*:?[ ]*(?P<value>{_MONEY_VALUE})"
    ),
    re.compile(rf"(?i)\bAmount\s+due\s+immediately\s*:?[ ]*(?P<value>{_MONEY_VALUE})"),
    re.compile(
        r"(?i)\bTotal\s+amount\s+due\s+if\s+we\s+terminate\s+your\s+installment\s+agreement"
        rf".{{0,180}}?(?P<value>{_MONEY_VALUE})"
    ),
)
_REFERENCE_PATTERNS = (
    re.compile(r"(?i)\bCase\s+reference\s+number\s*:?[ ]*(?P<value>[A-Z0-9-]{3,40})\b"),
    re.compile(r"(?i)\bReference\s+number\s*:?[ ]*(?P<value>[A-Z0-9-]{3,40})\b"),
)


def extract_notice_fields(text: str, identity: NoticeIdentity) -> NoticeFields:
    """Extract only explicit, uniquely labeled values; never calculate them."""

    normalized = _normalized_extraction_text(text)
    code = ExtractedField(
        identity.notice_code if identity.status == "identified" else None,
        identity.confidence if identity.status == "identified" else 0.0,
        identity.evidence_text if identity.status == "identified" else None,
    )
    reference = _unique_field_match(normalized, _REFERENCE_PATTERNS, confidence=0.97)
    if reference.value is not None and (
        re.fullmatch(r"(?i)[NX-]+", reference.value)
        or reference.value.casefold() in {"unavailable", "unknown", "none", "na", "n/a"}
    ):
        # Official samples may use a placeholder such as "nnnn". It proves a
        # field location, not a real taxpayer-specific reference value.
        reference = ExtractedField(None, 0.0, None)
    return NoticeFields(
        notice_code=code,
        notice_date=_unique_field_match(
            normalized,
            _NOTICE_DATE_PATTERNS,
            confidence=0.98,
            normalize_value=_canonical_date,
        ),
        due_or_response_date=_unique_field_match(
            normalized,
            _DUE_DATE_PATTERNS,
            confidence=0.95,
            normalize_value=_canonical_date,
        ),
        amount=_unique_field_match(
            normalized,
            _AMOUNT_PATTERNS,
            confidence=0.95,
            normalize_value=_canonical_amount,
        ),
        reference_number=reference,
    )


_REDACTIONS = (
    (re.compile(r"(?i)\b(?:\d|N){3}-(?:\d|N){2}-(?:\d|N){4}\b"), "[REDACTED TAXPAYER ID]"),
    (re.compile(r"(?i)\b(?:\d|N){2}-(?:\d|N){7}\b"), "[REDACTED TAXPAYER ID]"),
    (re.compile(r"(?i)\b(?:\d|x){3}-(?:\d|x){3}-(?:\d|x){4}\b"), "[REDACTED PHONE]"),
    (re.compile(r"(?i)\b(?:account|caller|taxpayer|social security|employer)\s+(?:ID|identification|number)\s*:?\s*\S+"), "[REDACTED IDENTIFIER]"),
    (re.compile(r"(?i)\b\d{1,6}\s+[A-Z0-9 .'-]+\s+(?:ST(?:REET)?|AVE(?:NUE)?|RD|ROAD|BLVD|DR(?:IVE)?|LN|LANE|CT|COURT)\b.*"), "[REDACTED ADDRESS]"),
    (re.compile(r"(?i)\bP\.?\s*O\.?\s+BOX\s+\d+\b.*"), "[REDACTED ADDRESS]"),
    (re.compile(r"(?im)^[A-Z .'-]+,\s*[A-Z]{2}\s+\d{5}(?:-\d{4})?$"), "[REDACTED ADDRESS]"),
    (re.compile(r"\b\d{9}\b"), "[REDACTED IDENTIFIER]"),
    (re.compile(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b"), "[REDACTED EMAIL]"),
)
_NOTICE_CONTEXT_STOPWORDS = {
    "about", "after", "again", "also", "does", "from", "have", "into", "notice",
    "should", "that", "their", "there", "these", "they", "this", "what", "when", "where",
    "which", "with", "would", "your", "irs",
}


def _looks_sensitive_line(line: str) -> bool:
    if "[REDACTED" in line:
        return True
    if re.search(
        r"(?i)\b(?:(?:taxpayer|account|caller|social security|employer)\s+"
        r"(?:ID|identification|number)|SSN|TIN|EIN)\b",
        line,
    ):
        return True
    if re.search(r"\b[A-Z]{2}\s+\d{5}(?:-\d{4})?\b", line) or re.search(r"\b\d{7,12}\b", line):
        return True
    if re.search(
        r"(?i)\b(?:street|st|avenue|ave|road|rd|boulevard|blvd|drive|dr|lane|ln|court|ct|"
        r"terrace|way|parkway|circle|place|highway|suite|apartment|apt)\b",
        line,
    ):
        return True
    # Conservative name-line suppression. False positives merely reduce the
    # optional notice excerpt; official guidance remains available.
    if re.fullmatch(r"[A-Z][a-z.'-]+(?:\s+(?:[A-Z]\.?|[A-Z][a-z.'-]+)){1,3}", line):
        return True
    if re.fullmatch(r"[A-Z][A-Z.'-]*(?:\s+[A-Z][A-Z.'-]*){1,4}", line):
        return True
    return False


def redact_notice_context(text: str) -> str:
    redacted = text
    for pattern, replacement in _REDACTIONS:
        redacted = pattern.sub(replacement, redacted)
    safe_lines: list[str] = []
    for line in redacted.splitlines():
        compact = re.sub(r"\s+", " ", line).strip()
        if re.fullmatch(r"(?i)(?:TAXPAYER NAME|ADDRESS|CITY,? STATE ZIP)", compact):
            continue
        if re.search(r"(?i)\b(?:taxpayer|social security|employer)\s+ID\s+number\b", compact):
            continue
        if _looks_sensitive_line(compact):
            continue
        safe_lines.append(compact)
    return "\n".join(line for line in safe_lines if line)


def relevant_notice_field_names(question: str) -> set[str]:
    """Return the deterministic minimum notice-field scope for a question."""

    normalized = " ".join(question.casefold().split())
    relevant = {"notice_code"}
    if re.search(r"\b(?:guidance|webpage|website)\s+(?:alone|only)\b", normalized):
        return relevant
    if re.search(
        r"\b(?:summarize|summary of|what does|what is)\s+(?:this|my|the uploaded)\s+"
        r"(?:notice|letter|pdf|document)\b",
        normalized,
    ):
        return {
            "notice_code",
            "notice_date",
            "due_or_response_date",
            "amount",
            "reference_number",
        }
    if re.search(r"\b(?:notice date|date (?:is )?printed|when (?:was|is) .*\bdated)\b", normalized):
        relevant.add("notice_date")
    if re.search(r"\b(?:due date|response date|reply date|deadline|by what date|when do i need to)\b", normalized):
        relevant.add("due_or_response_date")
    if re.search(r"\b(?:amount|how much|what do i owe|what is (?:my|the) total|total amount)\b", normalized):
        relevant.add("amount")
    if re.search(r"\b(?:reference number|case number)\b", normalized):
        relevant.add("reference_number")
    return relevant


def select_relevant_notice_context(
    notice_text: str,
    question: str,
    fields: NoticeFields,
    *,
    max_characters: int = 4_000,
) -> str:
    """Select small, query-relevant local notice spans before model use."""

    redacted = redact_notice_context(notice_text)
    lines = [line for line in redacted.splitlines() if line]
    terms = {
        token.casefold()
        for token in re.findall(r"[A-Za-z0-9$]+", question)
        if len(token) >= 4 and token.casefold() not in _NOTICE_CONTEXT_STOPWORDS
    }
    scored: list[tuple[int, int, str]] = []
    for index, line in enumerate(lines):
        words = {token.casefold() for token in re.findall(r"[A-Za-z0-9$]+", line)}
        score = len(terms & words)
        # Do not blanket-send the notice header: real headers contain names,
        # addresses, account identifiers, and other context irrelevant to the
        # user's question. Explicit identity/field snippets are added below.
        if score >= 2 and index >= 20 and not _looks_sensitive_line(line):
            scored.append((score, index, line))
    selected_indexes = {index for _, index, _ in sorted(scored, key=lambda item: (-item[0], item[1]))[:30]}
    relevant_field_names = relevant_notice_field_names(question)
    field_snippets: list[str] = []
    for field_name, field in fields.as_dict().items():
        if field_name not in relevant_field_names:
            continue
        source = field.get("source_text")
        if not source:
            continue
        safe_source = redact_notice_context(str(source)).strip()
        if safe_source and not _looks_sensitive_line(safe_source):
            field_snippets.append(safe_source)
    selected = field_snippets + [lines[index] for index in sorted(selected_indexes)]
    deduplicated = list(dict.fromkeys(selected))
    return "\n".join(deduplicated)[:max_characters]
