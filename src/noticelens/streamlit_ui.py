"""Pure helpers for the NoticeLens Streamlit presentation layer.

This module deliberately contains no alternate retrieval or generation logic.
Every analysis and chat request delegates to the frozen Phase 5 ``NoticeLensCore``.
"""

from __future__ import annotations

import csv
import json
import math
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Sequence

from .notice_input import (
    MAX_PDF_BYTES,
    ExtractedNotice,
    NoticeInputError,
    extract_pdf_text,
)

if TYPE_CHECKING:
    from .phase5 import CoreRun


DEFAULT_ANALYSIS_QUESTION = (
    "What does this notice mean, what does my notice state, and what does official IRS guidance say?"
)
SUGGESTED_QUESTIONS = (
    "What does this notice mean?",
    "What does IRS guidance say about responding?",
    "What happens at this stage?",
    "Show me the supporting evidence.",
)


class StreamlitUiError(RuntimeError):
    """A local presentation/integrity error safe to classify for the UI."""


@dataclass(frozen=True)
class SampleNotice:
    notice_code: str
    label: str
    filename: str
    source_url: str
    path: Path


@dataclass(frozen=True)
class EvidenceCard:
    claim_id: str
    claim_text: str
    citation_id: str
    notice_code: str
    source_title: str
    heading: str
    heading_path: tuple[str, ...]
    evidence_excerpt: str
    source_url: str
    chunk_id: str


@dataclass(frozen=True)
class ProductSnapshot:
    corpus_documents: int
    sample_notices: int
    final_config: Mapping[str, Any]
    notice_dense_p1: float
    notice_dense_mrr: float
    notice_dense_hit5: float
    fixed_section_p1: float
    fixed_section_mrr: float
    fixed_section_hit5: float
    heading_section_p1: float
    heading_section_mrr: float
    heading_section_hit5: float
    heading_gain_percentage_points: float
    ablation_p1: Mapping[str, float]
    faithfulness: float
    refusal_rate: float
    warm_median_seconds: float
    warm_p95_seconds: float
    latency_target_seconds: float
    latency_target_met: bool


def _json(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise StreamlitUiError(f"A frozen UI artifact could not be read ({type(exc).__name__})") from None
    if not isinstance(value, Mapping):
        raise StreamlitUiError("A frozen UI artifact has an invalid schema")
    return value


def _expect_close(name: str, observed: float, expected: float) -> float:
    value = float(observed)
    if not math.isclose(value, expected, rel_tol=0.0, abs_tol=1e-9):
        raise StreamlitUiError(f"The frozen {name} value changed")
    return value


def _validated_child(parent: Path, filename: str) -> Path:
    if not filename or Path(filename).name != filename or Path(filename).suffix.casefold() != ".pdf":
        raise StreamlitUiError("The sample notice manifest contains an invalid filename")
    root = parent.resolve()
    candidate = (root / filename).resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        raise StreamlitUiError("A sample notice path escaped its approved directory") from None
    if not candidate.is_file():
        raise StreamlitUiError("An approved sample notice PDF is missing")
    return candidate


def load_sample_notices(project_root: Path) -> tuple[SampleNotice, ...]:
    """Load the eight verified local samples without exposing local paths."""

    manifest = project_root / "data/sample_notice_manifest.csv"
    sample_root = project_root / "data/raw/sample_notices"
    try:
        with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except Exception as exc:
        raise StreamlitUiError(f"The sample notice catalog could not be read ({type(exc).__name__})") from None
    expected_columns = {
        "notice_code",
        "filename",
        "source_url",
        "language",
        "verification_status",
    }
    if len(rows) != 8 or not rows or set(rows[0]) != expected_columns:
        raise StreamlitUiError("The frozen sample notice catalog changed")
    samples: list[SampleNotice] = []
    seen_codes: set[str] = set()
    seen_files: set[str] = set()
    for row in rows:
        code = str(row.get("notice_code", "")).strip()
        filename = str(row.get("filename", "")).strip()
        source_url = str(row.get("source_url", "")).strip()
        if (
            not code
            or code in seen_codes
            or filename in seen_files
            or row.get("language") != "English"
            or row.get("verification_status") != "verified"
            or not source_url.startswith("https://www.irs.gov/")
        ):
            raise StreamlitUiError("The frozen sample notice catalog is invalid")
        seen_codes.add(code)
        seen_files.add(filename)
        samples.append(
            SampleNotice(
                notice_code=code,
                label=f"{code} · Official IRS sample",
                filename=filename,
                source_url=source_url,
                path=_validated_child(sample_root, filename),
            )
        )
    return tuple(samples)


def extract_uploaded_notice(data: bytes, display_name: str) -> ExtractedNotice:
    """Extract a PDF through the production parser and remove the temp file."""

    if not isinstance(data, bytes) or not data or len(data) > MAX_PDF_BYTES:
        raise NoticeInputError("The uploaded notice PDF size is invalid or exceeds the local limit")
    if data[:5] != b"%PDF-":
        raise NoticeInputError("The uploaded file is not a PDF")
    safe_name = Path(display_name or "uploaded-notice.pdf").name
    if Path(safe_name).suffix.casefold() != ".pdf":
        raise NoticeInputError("The uploaded file is not a PDF")
    with tempfile.TemporaryDirectory(prefix="noticelens_upload_") as temporary:
        path = Path(temporary) / "notice.pdf"
        path.write_bytes(data)
        extracted = extract_pdf_text(path)
    return replace(extracted, display_name=safe_name)


def analyze_sample(core: Any, sample: SampleNotice, question: str = DEFAULT_ANALYSIS_QUESTION) -> tuple[ExtractedNotice, CoreRun]:
    extracted = extract_pdf_text(sample.path)
    return extracted, core.run_extracted(extracted, question)


def analyze_upload(
    core: Any,
    data: bytes,
    display_name: str,
    question: str = DEFAULT_ANALYSIS_QUESTION,
) -> tuple[ExtractedNotice, CoreRun]:
    extracted = extract_uploaded_notice(data, display_name)
    return extracted, core.run_extracted(extracted, question)


def ask_notice(core: Any, extracted: ExtractedNotice, question: str) -> CoreRun:
    """Route every chat turn through the same frozen notice-specific graph."""

    if not isinstance(question, str) or not question.strip():
        raise StreamlitUiError("A chat question is required")
    return core.run_extracted(extracted, question.strip())


def evidence_status(run: CoreRun) -> str:
    if run.response.status == "answered" and run.response.claims:
        return "GROUNDED"
    if run.identity.status in {"unidentified", "ambiguous"}:
        return "NOTICE IDENTITY UNCLEAR"
    return "INSUFFICIENT EVIDENCE"


def _normalized_excerpt(claim_text: str) -> str:
    prefix = "IRS guidance says: "
    return claim_text[len(prefix) :] if claim_text.startswith(prefix) else claim_text


def build_evidence_cards(run: CoreRun) -> tuple[EvidenceCard, ...]:
    """Bind displayed citation metadata back to retrieved app-owned documents."""

    documents = {str(doc.metadata.get("chunk_id", "")): doc for doc in run.documents}
    citations = {citation.citation_id: citation for citation in run.response.citations}
    cards: list[EvidenceCard] = []
    for claim in run.response.claims:
        if claim.evidence_type != "guidance":
            continue
        for citation_id in claim.citation_ids:
            citation = citations.get(citation_id)
            if citation is None:
                raise StreamlitUiError("A response references an unknown citation")
            document = documents.get(citation.chunk_id)
            if document is None:
                raise StreamlitUiError("A citation is not backed by a retrieved document")
            metadata = document.metadata
            expected = {
                "notice_code": citation.notice_code,
                "title": citation.source_title,
                "heading": citation.heading,
                "heading_path": citation.heading_path,
                "source_url": citation.source_url,
                "chunk_id": citation.chunk_id,
            }
            if any(metadata.get(key) != value for key, value in expected.items()):
                raise StreamlitUiError("Citation metadata differs from the retrieved document")
            if not citation.source_url.startswith("https://www.irs.gov/"):
                raise StreamlitUiError("A citation URL is not an official IRS source")
            cards.append(
                EvidenceCard(
                    claim_id=claim.claim_id,
                    claim_text=claim.text,
                    citation_id=citation.citation_id,
                    notice_code=citation.notice_code,
                    source_title=citation.source_title,
                    heading=citation.heading,
                    heading_path=tuple(citation.heading_path),
                    evidence_excerpt=_normalized_excerpt(claim.text),
                    source_url=citation.source_url,
                    chunk_id=citation.chunk_id,
                )
            )
    return tuple(cards)


def build_trace_rows(run: CoreRun) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    for rank, document in enumerate(run.documents, start=1):
        metadata = document.metadata
        required = ("notice_code", "heading", "title", "source_url", "chunk_id", "similarity_score")
        if any(metadata.get(name) in (None, "") for name in required):
            raise StreamlitUiError("A retrieval trace record is incomplete")
        source_url = str(metadata["source_url"])
        if not source_url.startswith("https://www.irs.gov/"):
            raise StreamlitUiError("A retrieval trace contains a non-IRS source")
        score = float(metadata["similarity_score"])
        if not math.isfinite(score):
            raise StreamlitUiError("A retrieval trace contains a non-finite score")
        preview = " ".join(str(document.page_content).split())
        rows.append(
            {
                "Rank": rank,
                "Notice code": str(metadata["notice_code"]),
                "Heading": str(metadata["heading"]),
                "Similarity": score,
                "Source": str(metadata["title"]),
                "Source URL": source_url,
                "Chunk ID": str(metadata["chunk_id"]),
                "Preview": preview[:420],
            }
        )
    return tuple(rows)


def notice_detail_rows(run: CoreRun) -> tuple[tuple[str, str], ...]:
    fields = run.response.notice_fields
    return (
        ("Notice date", fields.notice_date.value or "Not confidently identified"),
        ("Amount", fields.amount.value or "Not confidently identified"),
        ("Due / response date", fields.due_or_response_date.value or "Not confidently identified"),
        ("Reference number", fields.reference_number.value or "Not confidently identified"),
    )


def safe_error_message(error: BaseException) -> str:
    """Return a stable category message without provider text, paths, or secrets."""

    if isinstance(error, NoticeInputError):
        return "This PDF could not be analyzed. Use a valid, unencrypted PDF with a usable text layer."

    # Import core exception types only when classifying a runtime core failure.
    # This keeps ordinary app startup independent of experiment/tokenizer code.
    from .final_retrieval import FinalRetrievalError
    from .grounded_generation import GenerationGateError
    from .phase5 import Phase5GateError

    if isinstance(error, FinalRetrievalError):
        return "Official IRS guidance retrieval is temporarily unavailable. Please try again later."
    if isinstance(error, GenerationGateError):
        return "The evidence-backed explanation could not be generated. Please try again later."
    if isinstance(error, Phase5GateError):
        return "NoticeLens could not start because its frozen configuration or provider setup is unavailable."
    return "NoticeLens could not complete this request safely. Please try again later."


def _count_rows(path: Path) -> int:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            return len(list(csv.DictReader(handle)))
    except Exception as exc:
        raise StreamlitUiError(f"A frozen catalog could not be read ({type(exc).__name__})") from None


def load_product_snapshot(project_root: Path) -> ProductSnapshot:
    """Load and validate only frozen local configuration and report values."""

    reports = project_root / "reports"
    config = _json(reports / "final_retrieval_config.json")
    expected_config = {
        "embedding_model": "Qwen/Qwen3-Embedding-8B",
        "embedding_dimension": 4096,
        "generation_model": "Qwen/Qwen3-30B-A3B-Instruct-2507",
        "pinecone_index": "noticelens-rag",
        "production_namespace": "heading-aware-dense",
        "chunk_strategy": "heading_aware_220_40",
        "top_k": 5,
        "metadata_filter": "exact notice_code equality",
        "bm25": False,
        "hybrid_retrieval": False,
        "reranking": False,
    }
    if any(config.get(key) != value for key, value in expected_config.items()):
        raise StreamlitUiError("The final production RAG configuration changed")

    phase3 = _json(reports / "phase3_baseline_results.json")
    phase4a = _json(reports / "phase4_fixed_section_results.json")
    phase4b = _json(reports / "phase4b_heading_results.json")
    faith = _json(reports / "phase5_1_faithfulness.json")
    latency = _json(reports / "phase5_1_latency.json")

    notice = phase3["metrics"]["notice_retrieval"]["overall"]
    fixed = phase4a["metrics"]["section_retrieval"]["overall"]
    heading = phase4b["metrics"]["heading_aware_section_retrieval"]["overall"]
    gain = phase4b["direct_comparison"]["absolute_p1_improvement_percentage_points"]
    selected = str(faith["decision"]["selected_model"])
    selected_summary = faith["models"][selected]["summary"]
    refusal = faith["models"][selected]["refusal"]
    warm = latency["models"][selected]["warm_measured"]["summary"]["end_to_end"]
    latency_target_met = bool(latency["decision"]["latency_target_met"])

    ablation_rows: list[dict[str, str]]
    try:
        with (reports / "retrieval_ablation.csv").open("r", encoding="utf-8", newline="") as handle:
            ablation_rows = list(csv.DictReader(handle))
    except Exception as exc:
        raise StreamlitUiError(f"The retrieval ablation could not be read ({type(exc).__name__})") from None
    if len(ablation_rows) != 15:
        raise StreamlitUiError("The frozen retrieval ablation question count changed")
    ablation = {
        "Dense": sum(int(row["dense_p1"]) for row in ablation_rows) / 15,
        "BM25": sum(int(row["bm25_p1"]) for row in ablation_rows) / 15,
        "Hybrid": sum(int(row["hybrid_p1"]) for row in ablation_rows) / 15,
        "Hybrid + reranker": sum(int(row["hybrid_reranker_p1"]) for row in ablation_rows) / 15,
    }

    validated_ablation = {
        "Dense": _expect_close("dense ablation P@1", ablation["Dense"], 14 / 15),
        "BM25": _expect_close("BM25 ablation P@1", ablation["BM25"], 6 / 15),
        "Hybrid": _expect_close("hybrid ablation P@1", ablation["Hybrid"], 10 / 15),
        "Hybrid + reranker": _expect_close(
            "hybrid reranker ablation P@1", ablation["Hybrid + reranker"], 14 / 15
        ),
    }
    corpus_documents = _count_rows(project_root / "data/corpus_manifest.csv")
    sample_notices = _count_rows(project_root / "data/sample_notice_manifest.csv")
    if corpus_documents != 50 or sample_notices != 8:
        raise StreamlitUiError("The frozen corpus or sample inventory changed")
    return ProductSnapshot(
        corpus_documents=corpus_documents,
        sample_notices=sample_notices,
        final_config=dict(config),
        notice_dense_p1=_expect_close("notice dense P@1", notice["precision_at_1"], 1.0),
        notice_dense_mrr=_expect_close("notice dense MRR", notice["mrr"], 1.0),
        notice_dense_hit5=_expect_close("notice dense Hit@5", notice["hit_at_5"], 1.0),
        fixed_section_p1=_expect_close("fixed section P@1", fixed["section_precision_at_1"], 0.8),
        fixed_section_mrr=_expect_close("fixed section MRR", fixed["section_mrr"], 0.8688888888888889),
        fixed_section_hit5=_expect_close("fixed section Hit@5", fixed["section_hit_at_5"], 1.0),
        heading_section_p1=_expect_close(
            "heading-aware section P@1", heading["section_precision_at_1"], 14 / 15
        ),
        heading_section_mrr=_expect_close(
            "heading-aware section MRR", heading["section_mrr"], 0.9555555555555555
        ),
        heading_section_hit5=_expect_close(
            "heading-aware section Hit@5", heading["section_hit_at_5"], 1.0
        ),
        heading_gain_percentage_points=_expect_close("heading-aware gain", gain, 13.33333333333333),
        ablation_p1=validated_ablation,
        faithfulness=_expect_close("formal faithfulness", selected_summary["faithfulness"], 1.0),
        refusal_rate=_expect_close("correct refusal rate", refusal["rate"], 1.0),
        warm_median_seconds=_expect_close("warm median latency", warm["median_seconds"], 9.775036),
        warm_p95_seconds=_expect_close("warm p95 latency", warm["p95_seconds"], 22.050282),
        latency_target_seconds=6.0,
        latency_target_met=latency_target_met,
    )
