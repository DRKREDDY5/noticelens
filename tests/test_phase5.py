from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from langchain_core.documents import Document
from pydantic import ValidationError


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from noticelens.final_retrieval import (  # noqa: E402
    EXPECTED_ID_SET_SHA256,
    EXPECTED_VECTOR_COUNT,
    PRODUCTION_NAMESPACE,
    TOP_K,
    FinalHeadingRetriever,
    FinalRetrievalError,
    ReadOnlyHeadingStore,
    RetrievalResult,
    evidence_is_sufficient,
)
from noticelens.grounded_generation import (  # noqa: E402
    INSUFFICIENT_ANSWER,
    GenerationGateError,
    ModelSelection,
    NebiusGroundedGenerator,
    build_grounded_response,
)
from noticelens.heading_chunking import (  # noqa: E402
    HEADING_PINECONE_METADATA_KEYS,
    load_heading_registry,
)
from noticelens.notice_input import (  # noqa: E402
    ExtractedField,
    ExtractedNotice,
    NoticeFields,
    NoticeIdentity,
    NoticeInputError,
    extract_notice_fields,
    extract_pdf_text,
    identify_notice,
    normalize_notice_code,
    redact_notice_context,
    relevant_notice_field_names,
    route_notice_code,
    select_relevant_notice_context,
)
from noticelens.phase5 import (  # noqa: E402
    CoreRun,
    PHASE4B_FROZEN_HASHES,
    PHASE5_FROZEN_COMPOSITE_SHA256,
    NoticeLensCore,
    Phase5GateError,
    Phase5Secrets,
    classify_unsupported_question,
    load_phase5_secrets,
    verify_phase5_frozen_inputs,
)
from noticelens.phase5_evaluation import (  # noqa: E402
    COLD_REPETITIONS,
    FAITHFULNESS_SAMPLE_MAP,
    FINAL_CONFIG_PATH,
    FINAL_REPORT_PATH,
    GENERATION_REPORT_PATH,
    LATENCY_CASE_IDS,
    LATENCY_REPORT_PATH,
    REFUSAL_REPORT_PATH,
    REFUSAL_SAMPLE_MAP,
    WARMUP_REPETITIONS,
    WARM_MEASURED_REPETITIONS,
    WARM_P95_TARGET_SECONDS,
    _latency_stats,
    evaluate_generation,
    evaluate_latency,
    evaluate_refusals,
    write_reports,
)
from noticelens.phase5_models import (  # noqa: E402
    Citation,
    ClaimJudgment,
    DraftClaim,
    FaithfulnessJudgment,
    GenerationDraft,
    GroundedClaim,
    GroundedResponse,
    ResponseNoticeFields,
)
from noticelens.providers import (  # noqa: E402
    EMBEDDING_DIMENSION,
    EMBEDDING_MODEL,
    INDEX_CLOUD,
    INDEX_METRIC,
    INDEX_NAME,
    INDEX_REGION,
)


REGISTRY_PATH = ROOT / "data/derived/phase4b/heading_aware_220_40_chunks.jsonl"
SAMPLE_DIR = ROOT / "data/raw/sample_notices"

with (ROOT / "data/corpus_manifest.csv").open(encoding="utf-8-sig", newline="") as handle:
    GUIDANCE_CODES = {row["notice_code"] for row in csv.DictReader(handle)}

REGISTRY_RECORDS = load_heading_registry(REGISTRY_PATH)
RECORDS_BY_CODE: dict[str, list[object]] = {}
for _record in REGISTRY_RECORDS:
    RECORDS_BY_CODE.setdefault(str(_record.metadata["notice_code"]), []).append(_record)

_FROZEN_BEFORE: dict[str, object] | None = None


def setUpModule() -> None:
    global _FROZEN_BEFORE
    _FROZEN_BEFORE = verify_phase5_frozen_inputs(ROOT)


def tearDownModule() -> None:
    after = verify_phase5_frozen_inputs(ROOT)
    if _FROZEN_BEFORE != after:
        raise AssertionError("Approved Phase 1-4B artifacts changed during Phase 5 tests")


def make_fields(
    *,
    code: str = "CP503",
    notice_date: str | None = "January 2, 2020",
    due_date: str | None = "February 3, 2020",
    amount: str | None = "$9,533.53",
    reference: str | None = None,
) -> NoticeFields:
    def field(value: str | None, source: str) -> ExtractedField:
        return ExtractedField(value, 0.95 if value is not None else 0.0, source if value is not None else None)

    return NoticeFields(
        notice_code=field(code, f"Notice {code}"),
        notice_date=field(notice_date, f"Notice date {notice_date}"),
        due_or_response_date=field(due_date, f"Respond by {due_date}"),
        amount=field(amount, f"Amount due {amount}"),
        reference_number=field(reference, f"Reference number {reference}"),
    )


def make_document(
    *,
    code: str = "CP503",
    chunk_id: str = "chunk-cp503-1",
    text: str = "The unpaid balance remains, and the IRS has not received a payment or response.",
) -> Document:
    return Document(
        page_content=text,
        metadata={
            "doc_id": "irs_cp503",
            "notice_code": code,
            "notice_family": "balance_collection",
            "title": "Understanding your CP503 notice",
            "source_url": "https://www.irs.gov/individuals/understanding-your-cp503-notice",
            "source_origin": "IRS",
            "chunk_id": chunk_id,
            "chunk_strategy": "heading_aware_220_40",
            "heading": "What is the notice telling me?",
            "heading_path": ["What is the notice telling me?"],
            "section_index": 1,
            "subchunk_index": 0,
            "similarity_score": -100.0,
        },
    )


class StubEmbeddings:
    model = EMBEDDING_MODEL
    expected_dimension = EMBEDDING_DIMENSION

    def __init__(self) -> None:
        self.queries: list[str] = []

    def embed_query(self, question: str) -> list[float]:
        self.queries.append(question)
        return [0.0] * EMBEDDING_DIMENSION


class StubStore:
    def __init__(self, *, code: str = "CP503", tamper_metadata: bool = False, unknown_id: bool = False) -> None:
        self.code = code
        self.tamper_metadata = tamper_metadata
        self.unknown_id = unknown_id
        self.query_calls: list[tuple[str, int]] = []
        self.expected_ids: set[str] | None = None

    def require_frozen_index(self) -> dict[str, object]:
        return {"name": INDEX_NAME, "namespace": PRODUCTION_NAMESPACE, "ready": True}

    def assert_namespace(self, expected_ids: set[str]) -> dict[str, object]:
        self.expected_ids = set(expected_ids)
        return {"vector_count": len(expected_ids), "exact_id_parity": True}

    def query(self, vector: list[float], *, notice_code: str, eligible_count: int) -> list[dict[str, object]]:
        if len(vector) != EMBEDDING_DIMENSION:
            raise AssertionError("Production retriever supplied a non-4096-dimensional vector")
        self.query_calls.append((notice_code, eligible_count))
        records = RECORDS_BY_CODE[notice_code][:TOP_K]
        matches: list[dict[str, object]] = []
        for rank, record in enumerate(records):
            metadata = {key: record.metadata[key] for key in HEADING_PINECONE_METADATA_KEYS}
            if self.tamper_metadata and rank == 0:
                metadata["source_url"] = "https://attacker.invalid/invented"
            matches.append(
                {
                    "id": "unknown-frozen-id" if self.unknown_id and rank == 0 else record.chunk_id,
                    "score": 1.0 - rank / 10,
                    "metadata": metadata,
                }
            )
        return matches


class StubRetriever:
    def __init__(self, documents: list[Document]) -> None:
        self.documents = documents
        self.calls: list[tuple[str, str]] = []

    def retrieve(self, question: str, *, notice_code: str) -> RetrievalResult:
        self.calls.append((question, notice_code))
        return RetrievalResult(self.documents, embedding_seconds=0.01, pinecone_seconds=0.02)


class StubGenerator:
    def __init__(self, draft: GenerationDraft) -> None:
        self.draft = draft
        self.calls: list[dict[str, object]] = []

    def generate_draft(self, **kwargs: object) -> tuple[GenerationDraft, float]:
        self.calls.append(dict(kwargs))
        return self.draft, 0.03


def guidance_draft(document: Document | None = None) -> GenerationDraft:
    document = document or make_document()
    quote = document.page_content
    return GenerationDraft(
        status="answer",
        claims=[
            DraftClaim(
                statement="The model's prose is untrusted and will not be rendered.",
                evidence_type="guidance",
                evidence_quote=quote,
                support_chunk_ids=[str(document.metadata["chunk_id"])],
                notice_field_names=[],
            )
        ],
    )


SAMPLE_IDENTITY_ROUTE = {
    "cp05a_english.pdf": ("CP05A", "CP05A"),
    "cp523h_english.pdf": ("CP523H", "CP523"),
    "lt11_english.pdf": ("LT11", "LT11 / Letter 1058"),
    "cp44_english.pdf": ("CP44", "CP44"),
    "cp503c.pdf": ("CP503C", "CP503"),
    "cp501_english.pdf": ("CP501", "CP501"),
    "cp59_english.pdf": ("CP59", "CP59"),
}


def make_evaluation_answered_run(
    question_id: str,
    filename: str,
    *,
    claim_count: int = 1,
    unsupported_last_claim: bool = False,
) -> CoreRun:
    public_code, retrieval_code = SAMPLE_IDENTITY_ROUTE[filename]
    evidence_text = (
        f"Official IRS evidence for {question_id} says the taxpayer should follow the instructions "
        "shown on the notice."
    )
    document = make_document(
        code=retrieval_code,
        chunk_id=f"{question_id}-frozen-chunk",
        text=evidence_text,
    )
    citation = Citation(
        citation_id="C1",
        notice_code=retrieval_code,
        source_title=str(document.metadata["title"]),
        heading=str(document.metadata["heading"]),
        heading_path=list(document.metadata["heading_path"]),
        source_url=str(document.metadata["source_url"]),
        chunk_id=str(document.metadata["chunk_id"]),
    )
    claims: list[GroundedClaim] = []
    for index in range(1, claim_count + 1):
        claim_text = f"IRS guidance says: {evidence_text}"
        if unsupported_last_claim and index == claim_count:
            claim_text = "IRS guidance says: This sentence is absent from the cited frozen evidence."
        claims.append(
            GroundedClaim(
                claim_id=f"CL{index}",
                text=claim_text,
                evidence_type="guidance",
                citation_ids=["C1"],
                notice_field_names=[],
            )
        )
    fields = make_fields(code=public_code, notice_date=None, due_date=None, amount=None)
    response = GroundedResponse(
        status="answered",
        notice_code=public_code,
        answer=" ".join(claim.text for claim in claims),
        claims=claims,
        citations=[citation],
        notice_fields=ResponseNoticeFields.from_notice_fields(fields),
    )
    return CoreRun(
        response=response,
        timings={
            "embedding_seconds": 0.01,
            "pinecone_seconds": 0.02,
            "generation_seconds": 0.03,
            "request_end_to_end_seconds": 0.08,
        },
        identity=NoticeIdentity("identified", public_code, retrieval_code, 0.99, f"Notice {public_code}"),
        fields=fields,
        documents=[document],
        policy_refusal_reason=None,
    )


class BatchJudge:
    def __init__(self, *, mismatch_question: str | None = None, unavailable_question: str | None = None) -> None:
        self.mismatch_question = mismatch_question
        self.unavailable_question = unavailable_question
        self.calls: list[list[dict[str, object]]] = []

    def judge_claims(self, *, claims: list[dict[str, object]]) -> tuple[FaithfulnessJudgment, float]:
        self.calls.append(claims)
        question_id = str(claims[0]["claim_id"]).split(":", 1)[0]
        if question_id == self.unavailable_question:
            raise GenerationGateError("auxiliary judge unavailable without provider details")
        ids = [str(claim["claim_id"]) for claim in claims]
        if question_id == self.mismatch_question:
            ids = [f"{question_id}:WRONG"]
        return (
            FaithfulnessJudgment(
                judgments=[
                    ClaimJudgment(claim_id=claim_id, supported=True, explanation="Supported by attached evidence.")
                    for claim_id in ids
                ]
            ),
            0.05,
        )


class GenerationEvaluationCore:
    def __init__(
        self,
        *,
        judge: BatchJudge,
        two_claim_question: str | None = None,
        unsupported_question: str | None = None,
    ) -> None:
        self.generator = judge
        self.two_claim_question = two_claim_question
        self.unsupported_question = unsupported_question
        self.calls: list[tuple[str, str]] = []
        golden = json.loads((ROOT / "eval/golden_questions.json").read_text(encoding="utf-8"))
        self._id_by_question = {str(row["question"]): str(row["id"]) for row in golden}

    def run_pdf(self, path: Path, question: str) -> CoreRun:
        question_id = self._id_by_question[question]
        self.calls.append((path.name, question_id))
        claim_count = 2 if question_id == self.two_claim_question else 1
        return make_evaluation_answered_run(
            question_id,
            path.name,
            claim_count=claim_count,
            unsupported_last_claim=question_id == self.unsupported_question,
        )


REFUSAL_POLICY_REASONS = {
    "E01": "unsupported_intent_or_fraud_inference",
    "E02": "taxpayer_specific_fact_excluded_by_source_scope",
    "E03": "out_of_domain_financial_recommendation",
    "E04": "fabrication_request",
}


class RefusalEvaluationCore:
    def __init__(self, *, answered_failure_id: str | None = None) -> None:
        self.answered_failure_id = answered_failure_id
        self.calls: list[tuple[str, str]] = []
        golden = json.loads((ROOT / "eval/golden_questions.json").read_text(encoding="utf-8"))
        self._id_by_question = {str(row["question"]): str(row["id"]) for row in golden}

    def run_pdf(self, path: Path, question: str) -> CoreRun:
        question_id = self._id_by_question[question]
        self.calls.append((path.name, question_id))
        if question_id == self.answered_failure_id:
            return make_evaluation_answered_run(question_id, path.name)
        public_code, retrieval_code = SAMPLE_IDENTITY_ROUTE[path.name]
        fields = make_fields(code=public_code, notice_date=None, due_date=None, amount=None)
        response = GroundedResponse(
            status="refused",
            notice_code=public_code,
            answer=INSUFFICIENT_ANSWER,
            claims=[],
            citations=[],
            notice_fields=ResponseNoticeFields.from_notice_fields(fields),
        )
        return CoreRun(
            response=response,
            timings={
                "embedding_seconds": 0.0,
                "pinecone_seconds": 0.0,
                "generation_seconds": 0.0,
                "request_end_to_end_seconds": 0.01,
            },
            identity=NoticeIdentity(
                "identified",
                public_code,
                retrieval_code,
                0.99,
                f"Notice {public_code}",
            ),
            fields=fields,
            documents=[],
            policy_refusal_reason=REFUSAL_POLICY_REASONS[question_id],
        )


def make_latency_run(duration: float) -> CoreRun:
    base = make_evaluation_answered_run("C01", "cp503c.pdf")
    return CoreRun(
        response=base.response,
        timings={
            "pdf_extraction_seconds": 0.01,
            "identity_and_fields_seconds": 0.01,
            "embedding_seconds": 0.10,
            "pinecone_seconds": 0.20,
            "generation_seconds": 0.30,
            "validation_and_render_seconds": 0.01,
            "request_end_to_end_seconds": duration,
        },
        identity=base.identity,
        fields=base.fields,
        documents=base.documents,
        policy_refusal_reason=None,
    )


class SequenceLatencyCore:
    def __init__(self, durations: list[float], statuses: list[str] | None = None) -> None:
        self.durations = durations
        self.statuses = statuses or ["answered"] * len(durations)
        self.calls: list[tuple[str, str]] = []

    def run_pdf(self, path: Path, question: str) -> CoreRun:
        index = len(self.calls)
        self.calls.append((path.name, question))
        run = make_latency_run(self.durations[index])
        if self.statuses[index] != "answered":
            run = CoreRun(
                response=run.response.model_copy(
                    update={
                        "status": "refused",
                        "answer": INSUFFICIENT_ANSWER,
                        "claims": [],
                        "citations": [],
                    }
                ),
                timings=run.timings,
                identity=run.identity,
                fields=run.fields,
                documents=run.documents,
                policy_refusal_reason=None,
            )
        return run


class PdfIdentityAndFieldTests(unittest.TestCase):
    EXPECTED = {
        "cp05a_english.pdf": ("CP05A", "CP05A", "March 7, 2019", "April 21, 2019", None, None),
        "cp131_english.pdf": ("CP131", "CP131", "January 28, 2019", None, "$0.00", None),
        "cp44_english.pdf": ("CP44", "CP44", "October 10, 2013", None, None, None),
        "cp501_english.pdf": (
            "CP501",
            "CP501",
            "January 28, 2019",
            "February 19, 2019",
            "$9,533.53",
            None,
        ),
        "cp503c.pdf": (
            "CP503C",
            "CP503",
            "August 10, 2018",
            "August 20, 2018",
            "$9,533.53",
            None,
        ),
        "cp523h_english.pdf": ("CP523H", "CP523", "January 30, 2019", None, "$9,533.53", None),
        "cp59_english.pdf": ("CP59", "CP59", "January 28, 2019", None, None, None),
        "lt11_english.pdf": (
            "LT11",
            "LT11 / Letter 1058",
            "March 2, 2020",
            "April 1, 2020",
            "$4,823.12",
            None,
        ),
    }

    @classmethod
    def setUpClass(cls) -> None:
        with (ROOT / "data/sample_notice_manifest.csv").open(encoding="utf-8-sig", newline="") as handle:
            cls.manifest_rows = list(csv.DictReader(handle))
        cls.results: dict[str, tuple[ExtractedNotice, NoticeIdentity, NoticeFields]] = {}
        for row in cls.manifest_rows:
            extracted = extract_pdf_text(SAMPLE_DIR / row["filename"])
            identity = identify_notice(extracted.pages[0], available_guidance_codes=GUIDANCE_CODES)
            fields = extract_notice_fields(extracted.text, identity)
            cls.results[row["filename"]] = (extracted, identity, fields)

    def test_all_eight_manifest_pdfs_use_existing_text_layers_without_ocr(self) -> None:
        self.assertEqual(len(self.manifest_rows), 8)
        self.assertEqual(set(self.results), set(self.EXPECTED))
        for filename, (extracted, _identity, _fields) in self.results.items():
            with self.subTest(filename=filename):
                self.assertEqual(extracted.display_name, filename)
                self.assertEqual(extracted.extraction_method, "pypdf_text_layer")
                self.assertTrue(extracted.pages)
                self.assertGreater(len("".join(extracted.text.split())), 100)

    def test_all_sample_identities_and_explicit_retrieval_routes_are_exact(self) -> None:
        for filename, expected in self.EXPECTED.items():
            _extracted, identity, _fields = self.results[filename]
            with self.subTest(filename=filename):
                self.assertEqual(identity.status, "identified")
                self.assertEqual(identity.notice_code, expected[0])
                self.assertEqual(identity.retrieval_notice_code, expected[1])
                self.assertGreaterEqual(identity.confidence, 0.90)
                self.assertNotRegex(identity.evidence_text or "", r"\d{3}-\d{2}-\d{4}")

    def test_all_sample_fields_are_deterministic_and_null_safe(self) -> None:
        for filename, expected in self.EXPECTED.items():
            _extracted, _identity, fields = self.results[filename]
            observed = (
                fields.notice_code.value,
                self.results[filename][1].retrieval_notice_code,
                fields.notice_date.value,
                fields.due_or_response_date.value,
                fields.amount.value,
                fields.reference_number.value,
            )
            with self.subTest(filename=filename):
                self.assertEqual(observed, expected)
                self.assertEqual(
                    set(fields.as_dict()),
                    {"notice_code", "notice_date", "due_or_response_date", "amount", "reference_number"},
                )
                for field_value in fields.as_dict().values():
                    self.assertEqual(set(field_value), {"value", "confidence", "source_text"})
                    self.assertGreaterEqual(float(field_value["confidence"]), 0.0)
                    self.assertLessEqual(float(field_value["confidence"]), 1.0)
                    if field_value["value"] is None:
                        self.assertEqual(field_value, {"value": None, "confidence": 0.0, "source_text": None})

    def test_normalization_supports_required_cp_lt_and_letter_spellings(self) -> None:
        cases = {
            "CP503": "CP503",
            "CP 503": "CP503",
            "CP-503": "CP503",
            "CP2000A": "CP2000A",
            "LT11": "LT11",
            "LT 11": "LT11",
            "LT-11": "LT11",
            "LTR 1058": "LETTER1058",
            "Letter 1058": "LETTER1058",
        }
        for raw, canonical in cases.items():
            with self.subTest(raw=raw):
                self.assertEqual(normalize_notice_code(raw), canonical)
        for invalid in ("", "503", "CP-", "Letter 9999", "CP503AA"):
            with self.subTest(invalid=invalid):
                self.assertIsNone(normalize_notice_code(invalid))

    def test_routing_is_explicit_and_never_generically_strips_suffixes(self) -> None:
        self.assertEqual(route_notice_code("CP503C", GUIDANCE_CODES), "CP503")
        self.assertEqual(route_notice_code("CP523H", GUIDANCE_CODES), "CP523")
        self.assertEqual(route_notice_code("LETTER1058", GUIDANCE_CODES), "LT11 / Letter 1058")
        self.assertEqual(route_notice_code("CP2000A", GUIDANCE_CODES), "CP2000 series")
        self.assertEqual(route_notice_code("CP01A", GUIDANCE_CODES), "CP01A")
        self.assertIsNone(route_notice_code("CP9999A", GUIDANCE_CODES))

    def test_header_edge_cases_ambiguity_missing_and_later_reference(self) -> None:
        supported_headers = (
            "Notice: CP503",
            "Notice CP 503",
            "Notice CP-503",
            "CP503",
            "LTR 1058",
            "Letter 1058",
        )
        for header in supported_headers:
            with self.subTest(header=header):
                identity = identify_notice(header, available_guidance_codes=GUIDANCE_CODES)
                self.assertEqual(identity.status, "identified")

        ambiguous = identify_notice(
            "Department of the Treasury\nNotice CP503 / CP501\nTax year 2020",
            available_guidance_codes=GUIDANCE_CODES,
        )
        self.assertEqual(ambiguous.status, "ambiguous")
        self.assertIsNone(ambiguous.notice_code)
        self.assertIsNone(ambiguous.retrieval_notice_code)

        missing = identify_notice(
            "Department of the Treasury\nWe previously sent Notice CP501 about this account.",
            available_guidance_codes=GUIDANCE_CODES,
        )
        self.assertEqual(missing.status, "unidentified")

        later_reference = identify_notice(
            "Notice CP503\nTax year 2020\nNotice date January 2, 2020\nPage 1\nBalance due\n"
            "For more information, see Notice CP501.",
            available_guidance_codes=GUIDANCE_CODES,
        )
        self.assertEqual(later_reference.status, "identified")
        self.assertEqual(later_reference.notice_code, "CP503")

        aliases = identify_notice(
            "Notice LT11 / Letter 1058",
            available_guidance_codes=GUIDANCE_CODES,
        )
        self.assertEqual(aliases.status, "identified")
        self.assertEqual(aliases.retrieval_notice_code, "LT11 / Letter 1058")

    def test_known_header_without_frozen_guidance_never_guesses_a_route(self) -> None:
        identity = identify_notice("Notice CP9999", available_guidance_codes=GUIDANCE_CODES)
        self.assertEqual(identity.status, "identified")
        self.assertEqual(identity.notice_code, "CP9999")
        self.assertIsNone(identity.retrieval_notice_code)

    def test_relative_language_does_not_create_a_date_or_amount(self) -> None:
        text = (
            "Notice CP503\nNotice date January 2, 2020\n"
            "Please respond within 30 days. Pay the amount shown on your account."
        )
        identity = identify_notice(text, available_guidance_codes=GUIDANCE_CODES)
        fields = extract_notice_fields(text, identity)
        self.assertEqual(fields.notice_date.value, "January 2, 2020")
        self.assertIsNone(fields.due_or_response_date.value)
        self.assertIsNone(fields.amount.value)

    def test_textless_pdf_fails_clearly_and_does_not_attempt_ocr(self) -> None:
        fake_reader = SimpleNamespace(
            is_encrypted=False,
            pages=[SimpleNamespace(extract_text=lambda: "")],
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "textless.pdf"
            path.write_bytes(b"%PDF-1.4\nplaceholder")
            with patch("noticelens.notice_input.PdfReader", return_value=fake_reader):
                with self.assertRaisesRegex(NoticeInputError, "OCR was not attempted"):
                    extract_pdf_text(path)

    def test_non_pdf_fails_before_extraction(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "not-a-pdf.txt"
            path.write_text("this is not a PDF", encoding="utf-8")
            with patch("noticelens.notice_input.PdfReader") as reader:
                with self.assertRaisesRegex(NoticeInputError, "not a PDF"):
                    extract_pdf_text(path)
            reader.assert_not_called()

    def test_notice_context_redacts_common_pii_and_keeps_safe_facts(self) -> None:
        raw = (
            "John Q. Taxpayer\n"
            "123 Main Street, Springfield, IL 12345\n"
            "P.O. Box 1234\n"
            "Taxpayer ID number 123-45-6789\n"
            "Caller ID 987654321\n"
            "Phone 212-555-0199\n"
            "Email john.taxpayer@example.com\n"
            "The unpaid balance remains."
        )
        redacted = redact_notice_context(raw)
        for sensitive in (
            "John Q. Taxpayer",
            "123 Main",
            "P.O. Box",
            "123-45-6789",
            "987654321",
            "212-555-0199",
            "john.taxpayer@example.com",
        ):
            self.assertNotIn(sensitive, redacted)
        self.assertIn("The unpaid balance remains.", redacted)

    def test_relevant_context_includes_only_requested_notice_fields(self) -> None:
        extracted, _identity, fields = self.results["cp501_english.pdf"]
        context = select_relevant_notice_context(
            extracted.text,
            "What amount do I owe and when is it due?",
            fields,
        )
        self.assertIn("$9,533.53", context)
        self.assertIn("February 19, 2019", context)
        self.assertNotRegex(context, r"\d{3}-\d{2}-\d{4}")
        self.assertNotIn("TAXPAYER NAME", context)
        self.assertEqual(
            relevant_notice_field_names("What happens if I ignore the IRS guidance?"),
            {"notice_code"},
        )


class ReadOnlyRetrievalTests(unittest.TestCase):
    @staticmethod
    def compatible_client() -> tuple[MagicMock, MagicMock]:
        client = MagicMock()
        index = MagicMock()
        client.indexes.exists.return_value = True
        client.indexes.describe.return_value = {
            "dimension": EMBEDDING_DIMENSION,
            "metric": INDEX_METRIC,
            "vector_type": "dense",
            "host": "offline-host",
            "status": {"ready": True},
            "spec": {"serverless": {"cloud": INDEX_CLOUD, "region": INDEX_REGION}},
        }
        client.index.return_value = index
        return client, index

    def test_store_has_no_mutating_surface_and_uses_exact_filtered_query(self) -> None:
        client, index = self.compatible_client()
        index.query.return_value = {
            "matches": [
                {"id": f"chunk-{rank}", "score": 1.0 - rank / 10, "metadata": {}}
                for rank in range(TOP_K)
            ]
        }
        store = ReadOnlyHeadingStore(api_key="sentinel-pinecone", client=client)
        state = store.require_frozen_index()
        self.assertEqual(state["namespace"], PRODUCTION_NAMESPACE)
        matches = store.query([0.0] * EMBEDDING_DIMENSION, notice_code="CP503", eligible_count=6)
        self.assertEqual(len(matches), TOP_K)
        index.query.assert_called_once_with(
            namespace="heading-aware-dense",
            vector=[0.0] * EMBEDDING_DIMENSION,
            top_k=5,
            filter={"notice_code": {"$eq": "CP503"}},
            include_metadata=True,
            include_values=False,
        )
        for method in ("upsert", "delete", "update"):
            self.assertFalse(hasattr(store, method))
            getattr(index, method).assert_not_called()
        client.indexes.create.assert_not_called()

    def test_missing_index_fails_closed_and_is_never_created(self) -> None:
        client, _index = self.compatible_client()
        client.indexes.exists.return_value = False
        store = ReadOnlyHeadingStore(api_key="sentinel-pinecone", client=client)
        with self.assertRaisesRegex(FinalRetrievalError, "missing"):
            store.require_frozen_index()
        client.indexes.create.assert_not_called()

    def test_namespace_reconciles_all_580_frozen_ids_without_writes(self) -> None:
        client, index = self.compatible_client()
        expected_ids = {record.chunk_id for record in REGISTRY_RECORDS}
        index.describe_index_stats.return_value = {
            "namespaces": {PRODUCTION_NAMESPACE: {"vector_count": len(expected_ids)}}
        }
        index.list.return_value = [
            {"vectors": [{"id": vector_id} for vector_id in sorted(expected_ids)]}
        ]
        store = ReadOnlyHeadingStore(api_key="sentinel-pinecone", client=client)
        store.require_frozen_index()
        snapshot = store.assert_namespace(expected_ids)
        self.assertEqual(snapshot["vector_count"], EXPECTED_VECTOR_COUNT)
        self.assertEqual(snapshot["id_set_sha256"], EXPECTED_ID_SET_SHA256)
        self.assertTrue(snapshot["exact_id_parity"])
        index.list.assert_called_once_with(namespace="heading-aware-dense")
        index.upsert.assert_not_called()
        index.delete.assert_not_called()

    def test_final_retriever_uses_qwen_and_builds_documents_from_frozen_registry(self) -> None:
        embeddings = StubEmbeddings()
        store = StubStore()
        retriever = FinalHeadingRetriever(
            project_root=ROOT,
            nebius_api_key="unused-sentinel-nebius",
            pinecone_api_key="unused-sentinel-pinecone",
            embedding_client=embeddings,
            store=store,
        )
        result = retriever.retrieve("What does this second reminder mean?", notice_code="CP503")
        self.assertEqual(embeddings.queries, ["What does this second reminder mean?"])
        self.assertEqual(store.query_calls, [("CP503", len(RECORDS_BY_CODE["CP503"]))])
        self.assertEqual(len(result.documents), TOP_K)
        expected_records = {record.chunk_id: record for record in RECORDS_BY_CODE["CP503"][:TOP_K]}
        for document in result.documents:
            chunk_id = str(document.metadata["chunk_id"])
            self.assertEqual(document.page_content, expected_records[chunk_id].text)
            self.assertEqual(document.metadata["notice_code"], "CP503")
            self.assertIn("similarity_score", document.metadata)

    def test_final_retriever_rejects_unknown_ids_and_metadata_tampering(self) -> None:
        for store in (StubStore(unknown_id=True), StubStore(tamper_metadata=True)):
            with self.subTest(store=type(store).__name__, mode=(store.unknown_id, store.tamper_metadata)):
                retriever = FinalHeadingRetriever(
                    project_root=ROOT,
                    nebius_api_key="unused-sentinel-nebius",
                    pinecone_api_key="unused-sentinel-pinecone",
                    embedding_client=StubEmbeddings(),
                    store=store,
                )
                with self.assertRaises(FinalRetrievalError):
                    retriever.retrieve("question", notice_code="CP503")

    def test_evidence_gate_requires_content_matching_code_and_complete_metadata(self) -> None:
        valid = make_document()
        self.assertTrue(evidence_is_sufficient([valid], notice_code="CP503"))
        self.assertFalse(evidence_is_sufficient([], notice_code="CP503"))
        self.assertFalse(evidence_is_sufficient([make_document(code="CP501")], notice_code="CP503"))
        self.assertFalse(
            evidence_is_sufficient(
                [Document(page_content="   ", metadata=dict(valid.metadata))],
                notice_code="CP503",
            )
        )
        required = ("doc_id", "notice_code", "title", "source_url", "chunk_id", "heading", "heading_path")
        for key in required:
            metadata = dict(valid.metadata)
            metadata.pop(key)
            with self.subTest(missing=key):
                self.assertFalse(
                    evidence_is_sufficient(
                        [Document(page_content=valid.page_content, metadata=metadata)],
                        notice_code="CP503",
                    )
                )

    def test_evidence_gate_deliberately_ignores_similarity_score_thresholds(self) -> None:
        low_score = make_document()
        low_score.metadata["similarity_score"] = -1_000_000.0
        self.assertTrue(evidence_is_sufficient([low_score], notice_code="CP503"))


class StructuredGroundingTests(unittest.TestCase):
    def test_live_model_probe_exercises_the_real_answer_branch_contract(self) -> None:
        model = "Qwen/Qwen3-30B-A3B-Instruct-2507"
        client = MagicMock()
        client.models.list.return_value = SimpleNamespace(data=[SimpleNamespace(id=model)])
        generator = NebiusGroundedGenerator(api_key="sentinel-nebius", client=client)
        probe_draft = GenerationDraft(
            status="answer",
            claims=[
                DraftClaim(
                    statement="The sample form is blue.",
                    evidence_type="guidance",
                    evidence_quote="The sample form is blue.",
                    support_chunk_ids=["noticelens-capability-probe-chunk"],
                    notice_field_names=[],
                )
            ],
        )
        with patch.object(generator, "_structured_request", return_value=(probe_draft, 0.01)) as request:
            selection = generator.select_live_model()
        self.assertEqual(selection.selected_model, model)
        self.assertTrue(generator.catalog_verified)
        messages = request.call_args.kwargs["messages"]
        self.assertIn("you MUST return status=answer", messages[0]["content"])
        payload = json.loads(messages[1]["content"])
        self.assertEqual(payload["user_question"], "According to the supplied evidence, what color is the sample form?")
        self.assertEqual(
            payload["official_irs_evidence"][0]["chunk_id"],
            "noticelens-capability-probe-chunk",
        )

    def test_guidance_claim_is_rendered_from_exact_quote_and_document_metadata(self) -> None:
        document = make_document()
        fields = make_fields()
        response = build_grounded_response(
            draft=guidance_draft(document),
            public_notice_code="CP503",
            fields=fields,
            documents=[document],
            permitted_notice_field_names={"notice_code"},
        )
        self.assertEqual(response.status, "answered")
        self.assertEqual(response.answer, f"IRS guidance says: {document.page_content}")
        self.assertEqual(len(response.claims), 1)
        self.assertEqual(response.claims[0].citation_ids, ["C1"])
        self.assertEqual(
            response.citations[0].model_dump(),
            {
                "citation_id": "C1",
                "notice_code": document.metadata["notice_code"],
                "source_title": document.metadata["title"],
                "heading": document.metadata["heading"],
                "heading_path": document.metadata["heading_path"],
                "source_url": document.metadata["source_url"],
                "chunk_id": document.metadata["chunk_id"],
            },
        )

    def test_model_cannot_supply_citation_metadata_or_unknown_support(self) -> None:
        with self.assertRaises(ValidationError):
            DraftClaim.model_validate(
                {
                    "statement": "claim",
                    "evidence_type": "guidance",
                    "evidence_quote": "A sufficiently long exact evidence quote from a document.",
                    "support_chunk_ids": ["chunk-1"],
                    "notice_field_names": [],
                    "source_url": "https://attacker.invalid/source",
                }
            )
        document = make_document()
        draft = GenerationDraft(
            status="answer",
            claims=[
                DraftClaim(
                    statement="claim",
                    evidence_type="guidance",
                    evidence_quote=document.page_content,
                    support_chunk_ids=["unknown-chunk"],
                    notice_field_names=[],
                )
            ],
        )
        response = build_grounded_response(
            draft=draft,
            public_notice_code="CP503",
            fields=make_fields(),
            documents=[document],
            permitted_notice_field_names={"notice_code"},
        )
        self.assertEqual(response.status, "refused")
        self.assertEqual(response.answer, INSUFFICIENT_ANSWER)
        self.assertEqual(response.claims, [])
        self.assertEqual(response.citations, [])

    def test_app_binder_ignores_irrelevant_draft_fields_and_uses_only_exact_guidance(self) -> None:
        document = make_document()
        draft = GenerationDraft(
            status="answer",
            claims=[
                DraftClaim(
                    statement="Untrusted model summary.",
                    evidence_type="guidance",
                    evidence_quote=document.page_content,
                    support_chunk_ids=[str(document.metadata["chunk_id"])],
                    # Provider JSON schemas require this field to exist. If a
                    # model fills it anyway, it cannot affect rendered guidance.
                    notice_field_names=["amount"],
                )
            ],
        )
        response = build_grounded_response(
            draft=draft,
            public_notice_code="CP503",
            fields=make_fields(amount="$9,533.53"),
            documents=[document],
            permitted_notice_field_names={"amount"},
        )
        self.assertEqual(response.status, "answered")
        self.assertEqual(response.answer, f"IRS guidance says: {document.page_content}")
        self.assertNotIn("$9,533.53", response.answer)

    def test_non_exact_or_too_short_guidance_quote_fails_closed(self) -> None:
        document = make_document()
        for quote in ("not present in evidence at all", "The unpaid balance"):
            draft = GenerationDraft(
                status="answer",
                claims=[
                    DraftClaim(
                        statement="claim",
                        evidence_type="guidance",
                        evidence_quote=quote,
                        support_chunk_ids=[str(document.metadata["chunk_id"])],
                        notice_field_names=[],
                    )
                ],
            )
            with self.subTest(quote=quote):
                response = build_grounded_response(
                    draft=draft,
                    public_notice_code="CP503",
                    fields=make_fields(),
                    documents=[document],
                    permitted_notice_field_names={"notice_code"},
                )
                self.assertEqual(response.status, "refused")
                self.assertEqual(response.answer, INSUFFICIENT_ANSWER)

    def test_untrusted_model_statement_cannot_fabricate_date_or_amount(self) -> None:
        fields = make_fields(notice_date=None, due_date=None, amount="$9,533.53")
        draft = GenerationDraft(
            status="answer",
            claims=[
                DraftClaim(
                    statement="Pay $999,999.99 by December 31, 2099.",
                    evidence_type="notice",
                    evidence_quote=None,
                    support_chunk_ids=[],
                    notice_field_names=["amount"],
                )
            ],
        )
        response = build_grounded_response(
            draft=draft,
            public_notice_code="CP503",
            fields=fields,
            documents=[make_document()],
            permitted_notice_field_names={"amount"},
        )
        self.assertEqual(response.status, "answered")
        self.assertIn("$9,533.53", response.answer)
        self.assertNotIn("$999,999.99", response.answer)
        self.assertNotIn("December 31, 2099", response.answer)

    def test_null_or_unpermitted_notice_field_claim_fails_closed(self) -> None:
        draft = GenerationDraft(
            status="answer",
            claims=[
                DraftClaim(
                    statement="A made-up deadline.",
                    evidence_type="notice",
                    evidence_quote=None,
                    support_chunk_ids=[],
                    notice_field_names=["due_or_response_date"],
                )
            ],
        )
        for allowed in ({"due_or_response_date"}, {"notice_code"}):
            with self.subTest(allowed=allowed):
                response = build_grounded_response(
                    draft=draft,
                    public_notice_code="CP503",
                    fields=make_fields(due_date=None),
                    documents=[make_document()],
                    permitted_notice_field_names=allowed,
                )
                self.assertEqual(response.status, "refused")
                self.assertEqual(response.answer, INSUFFICIENT_ANSWER)

    def test_structured_response_forbids_extra_fields_and_unknown_citations(self) -> None:
        fields = ResponseNoticeFields.from_notice_fields(make_fields())
        citation = Citation(
            citation_id="C1",
            notice_code="CP503",
            source_title="Understanding your CP503 notice",
            heading="What this notice is about",
            heading_path=["What this notice is about"],
            source_url="https://www.irs.gov/individuals/understanding-your-cp503-notice",
            chunk_id="chunk-1",
        )
        with self.assertRaises(ValidationError):
            GroundedResponse.model_validate(
                {
                    "status": "answered",
                    "notice_code": "CP503",
                    "answer": "answer",
                    "claims": [],
                    "citations": [],
                    "notice_fields": fields.model_dump(),
                    "model_secret_guess": "not allowed",
                }
            )
        with self.assertRaises(ValidationError):
            GroundedResponse(
                status="answered",
                notice_code="CP503",
                answer="answer",
                claims=[
                    GroundedClaim(
                        claim_id="CL1",
                        text="claim",
                        evidence_type="guidance",
                        citation_ids=["C2"],
                        notice_field_names=[],
                    )
                ],
                citations=[citation],
                notice_fields=fields,
            )

    def test_citations_reject_non_irs_urls(self) -> None:
        with self.assertRaises(ValidationError):
            Citation(
                citation_id="C1",
                notice_code="CP503",
                source_title="Invented",
                heading="Invented",
                heading_path=["Invented"],
                source_url="https://attacker.invalid/not-irs",
                chunk_id="chunk-1",
            )

    def test_generation_payload_omits_field_source_text_and_null_fields(self) -> None:
        pii_sentinel = "PII_SENTINEL_123-45-6789_JANE_TAXPAYER"
        fields = NoticeFields(
            notice_code=ExtractedField("CP503", 0.99, "Notice CP503"),
            notice_date=ExtractedField(None, 0.0, None),
            due_or_response_date=ExtractedField(None, 0.0, None),
            amount=ExtractedField("$9,533.53", 0.95, f"Amount due $9,533.53 {pii_sentinel}"),
            reference_number=ExtractedField(None, 0.0, None),
        )
        generator = NebiusGroundedGenerator(
            api_key="sentinel-nebius",
            model="Qwen/Qwen3-30B-A3B-Instruct-2507",
            client=MagicMock(),
        )
        generator.catalog_verified = True
        insufficient = GenerationDraft(status="insufficient", claims=[])
        with patch.object(generator, "_structured_request", return_value=(insufficient, 0.01)) as request:
            generator.generate_draft(
                question="How much does my notice say I owe?",
                notice_context="Safe minimal notice context",
                notice_fields=fields,
                permitted_notice_field_names={"amount", "due_or_response_date"},
                documents=[make_document()],
            )
        messages = request.call_args.kwargs["messages"]
        self.assertEqual(request.call_args.kwargs["max_tokens"], 2400)
        system_prompt = messages[0]["content"]
        self.assertIn("you MUST return status=answer", system_prompt)
        self.assertIn("only when none of the supplied evidence", system_prompt)
        self.assertIn("Markdown bullets are allowed", system_prompt)
        self.assertIn("authoritative IRS source evidence", system_prompt)
        self.assertIn("cover every responsive item", system_prompt)
        payload_text = messages[1]["content"]
        payload = json.loads(payload_text)
        self.assertEqual(
            payload["deterministically_extracted_notice_fields"],
            {"amount": {"value": "$9,533.53", "confidence": 0.95}},
        )
        self.assertNotIn("source_text", payload_text)
        self.assertNotIn(pii_sentinel, payload_text)
        self.assertNotIn("due_or_response_date", payload["deterministically_extracted_notice_fields"])


class GraphAndRefusalTests(unittest.TestCase):
    def setUp(self) -> None:
        self.document = make_document()
        self.extracted = ExtractedNotice(
            display_name="offline.pdf",
            pages=("Notice CP503\nNotice date January 2, 2020",),
            text=(
                "Notice CP503\nNotice date January 2, 2020\n"
                "Amount due, to be received by February 3, 2020 $9,533.53"
            ),
        )

    def core(self, documents: list[Document], draft: GenerationDraft | None = None) -> tuple[NoticeLensCore, StubRetriever, StubGenerator]:
        retriever = StubRetriever(documents)
        generator = StubGenerator(draft or guidance_draft(self.document))
        core = NoticeLensCore(
            project_root=ROOT,
            retriever=retriever,
            generator=generator,
            available_guidance_codes={"CP503", "CP501"},
        )
        return core, retriever, generator

    def test_graph_topology_is_the_exact_minimal_notice_flow(self) -> None:
        core, _retriever, _generator = self.core([self.document])
        graph = core.graph.get_graph()
        self.assertEqual(
            set(graph.nodes),
            {
                "__start__",
                "identify_notice",
                "clarify_or_fail",
                "retrieve_guidance",
                "refuse",
                "generate_grounded_answer",
                "__end__",
            },
        )
        self.assertEqual(
            {(edge.source, edge.target) for edge in graph.edges},
            {
                ("__start__", "identify_notice"),
                ("identify_notice", "clarify_or_fail"),
                ("identify_notice", "retrieve_guidance"),
                ("clarify_or_fail", "__end__"),
                ("retrieve_guidance", "refuse"),
                ("retrieve_guidance", "generate_grounded_answer"),
                ("refuse", "__end__"),
                ("generate_grounded_answer", "__end__"),
            },
        )

    def test_identified_sufficient_route_retrieves_and_generates_once(self) -> None:
        core, retriever, generator = self.core([self.document])
        run = core.run_extracted(self.extracted, "What does this second reminder mean?")
        self.assertEqual(run.identity.status, "identified")
        self.assertEqual(run.response.status, "answered")
        self.assertEqual(len(retriever.calls), 1)
        self.assertEqual(retriever.calls[0][1], "CP503")
        self.assertEqual(len(generator.calls), 1)
        self.assertIn("identity_and_fields_seconds", run.timings)
        self.assertIn("embedding_seconds", run.timings)
        self.assertIn("pinecone_seconds", run.timings)
        self.assertIn("generation_seconds", run.timings)
        self.assertIn("graph_end_to_end_seconds", run.timings)

    def test_unidentified_and_ambiguous_routes_never_retrieve_or_generate(self) -> None:
        core, retriever, generator = self.core([self.document])
        unidentified = core.run_extracted(
            ExtractedNotice("missing.pdf", ("Department of the Treasury",), "Department of the Treasury"),
            "What is this?",
        )
        self.assertEqual(unidentified.response.status, "unidentified")
        self.assertEqual(retriever.calls, [])
        self.assertEqual(generator.calls, [])

        ambiguous = core.run_extracted(
            ExtractedNotice("ambiguous.pdf", ("Notice CP503 / CP501",), "Notice CP503 / CP501"),
            "What is this?",
        )
        self.assertEqual(ambiguous.response.status, "ambiguous")
        self.assertEqual(retriever.calls, [])
        self.assertEqual(generator.calls, [])

    def test_empty_or_wrong_notice_evidence_routes_to_exact_refusal(self) -> None:
        for documents in ([], [make_document(code="CP501")]):
            with self.subTest(document_count=len(documents)):
                core, retriever, generator = self.core(documents)
                run = core.run_extracted(self.extracted, "What does this second reminder mean?")
                self.assertEqual(len(retriever.calls), 1)
                self.assertEqual(generator.calls, [])
                self.assertEqual(run.response.status, "refused")
                self.assertEqual(run.response.answer, INSUFFICIENT_ANSWER)
                self.assertEqual(run.response.claims, [])
                self.assertEqual(run.response.citations, [])

    def test_frozen_e01_to_e04_are_four_for_four_deterministic_refusals(self) -> None:
        benchmark = json.loads((ROOT / "eval/golden_questions.json").read_text(encoding="utf-8"))
        frozen = {item["id"]: item for item in benchmark if item["id"] in {"E01", "E02", "E03", "E04"}}
        self.assertEqual(set(frozen), {"E01", "E02", "E03", "E04"})
        expected_reasons = {
            "E01": "unsupported_intent_or_fraud_inference",
            "E02": "taxpayer_specific_fact_excluded_by_source_scope",
            "E03": "out_of_domain_financial_recommendation",
            "E04": "fabrication_request",
        }
        core, retriever, generator = self.core([self.document])
        for question_id in ("E01", "E02", "E03", "E04"):
            with self.subTest(question_id=question_id):
                item = frozen[question_id]
                self.assertTrue(item["should_refuse"])
                run = core.run_extracted(self.extracted, item["question"])
                self.assertEqual(run.policy_refusal_reason, expected_reasons[question_id])
                self.assertEqual(run.response.status, "refused")
                self.assertEqual(run.response.answer, INSUFFICIENT_ANSWER)
                self.assertEqual(run.response.claims, [])
                self.assertEqual(run.response.citations, [])
        # Categorical policy refusals short-circuit the external retrieval call
        # inside the required retrieve_guidance node.
        self.assertEqual(retriever.calls, [])
        self.assertEqual(generator.calls, [])

    def test_refusal_rules_are_categorical_and_do_not_overrefuse_normal_guidance(self) -> None:
        cases = {
            "Does this notice prove I am guilty of tax evasion?": "unsupported_intent_or_fraud_inference",
            "Using the website alone, give my exact deadline.": "taxpayer_specific_fact_excluded_by_source_scope",
            "Recommend which bitcoin token I should buy for profit.": "out_of_domain_financial_recommendation",
            "Fabricate a confident penalty for me.": "fabrication_request",
            "Should I cash out retirement savings to pay this?": "personalized_tax_or_financial_advice",
        }
        for question, reason in cases.items():
            with self.subTest(question=question):
                self.assertEqual(classify_unsupported_question(question), reason)
        self.assertIsNone(classify_unsupported_question("What does IRS guidance say CP503 is about?"))
        self.assertIsNone(
            classify_unsupported_question("Using IRS guidance alone, what response deadline does CP503 describe?")
        )
        self.assertEqual(
            relevant_notice_field_names("Using IRS guidance alone, what due date applies?"),
            {"notice_code"},
        )


class Phase5GenerationEvaluationTests(unittest.TestCase):
    def test_objective_ten_item_subset_and_metric_labels_are_truthful(self) -> None:
        judge = BatchJudge()
        core = GenerationEvaluationCore(judge=judge, two_claim_question="A02")
        report, runs = evaluate_generation(project_root=ROOT, core=core)

        self.assertEqual(list(runs), list(FAITHFULNESS_SAMPLE_MAP))
        self.assertEqual(
            core.calls,
            [(filename, question_id) for question_id, filename in FAITHFULNESS_SAMPLE_MAP.items()],
        )
        self.assertEqual(report["subset_selection"]["question_ids"], list(FAITHFULNESS_SAMPLE_MAP))
        self.assertTrue(report["subset_selection"]["selected_before_generation"])
        self.assertEqual(report["subset_selection"]["question_count"], 10)
        self.assertEqual(
            report["subset_selection"]["expert_count"] + report["subset_selection"]["naive_count"],
            10,
        )

        results = report["results"]
        self.assertEqual(results["answerable_questions"], 10)
        self.assertEqual(results["answered_questions"], 10)
        self.assertEqual(results["answerable_response_rate"], 1.0)
        self.assertEqual(results["atomic_claims"], 11)
        self.assertEqual(results["deterministically_supported_claims"], 11)
        self.assertEqual(results["citation_support_rate"], 1.0)
        self.assertIsNone(results["formal_faithfulness"])
        self.assertEqual(results["formal_faithfulness_status"], "pending_human_review")
        self.assertEqual(results["frozen_faithfulness_target"], 0.95)
        self.assertIsNone(results["frozen_faithfulness_target_met"])
        self.assertFalse(report["method"]["human_evidence_review_completed"])
        self.assertEqual(report["method"]["formal_faithfulness_status"], "pending_human_review")
        self.assertNotIn("faithfulness", results)

        self.assertEqual(len(judge.calls), 10)
        self.assertEqual(
            [str(item["claim_id"]) for item in judge.calls[0]],
            ["A02:CL1", "A02:CL2"],
        )
        for question_id, batch in zip(FAITHFULNESS_SAMPLE_MAP, judge.calls, strict=True):
            self.assertTrue(batch)
            self.assertTrue(all(str(item["claim_id"]).startswith(f"{question_id}:") for item in batch))
        self.assertEqual(results["auxiliary_judge_evaluated_claims"], 11)
        self.assertEqual(results["auxiliary_judge_coverage"], 1.0)
        self.assertEqual(results["auxiliary_judge_support_rate"], 1.0)
        self.assertEqual(results["auxiliary_judge_failures"], [])

    def test_citation_support_is_not_relabelled_as_formal_faithfulness(self) -> None:
        judge = BatchJudge()
        core = GenerationEvaluationCore(judge=judge, unsupported_question="A02")
        report, _runs = evaluate_generation(project_root=ROOT, core=core)
        results = report["results"]
        self.assertEqual(results["atomic_claims"], 10)
        self.assertEqual(results["deterministically_supported_claims"], 9)
        self.assertEqual(results["citation_support_rate"], 0.9)
        self.assertIsNone(results["formal_faithfulness"])
        self.assertEqual(results["formal_faithfulness_status"], "pending_human_review")
        self.assertIsNone(results["frozen_faithfulness_target_met"])
        self.assertEqual(results["auxiliary_judge_support_rate"], 1.0)

    def test_auxiliary_judge_is_batched_by_question_and_failures_are_nonfatal(self) -> None:
        judge = BatchJudge(mismatch_question="A02", unavailable_question="B04")
        core = GenerationEvaluationCore(judge=judge, two_claim_question="A02")
        report, _runs = evaluate_generation(project_root=ROOT, core=core)
        results = report["results"]

        self.assertEqual(len(judge.calls), 10)
        self.assertEqual(results["atomic_claims"], 11)
        self.assertEqual(results["deterministically_supported_claims"], 11)
        self.assertEqual(results["citation_support_rate"], 1.0)
        self.assertEqual(results["auxiliary_judge_evaluated_claims"], 8)
        self.assertEqual(results["auxiliary_judge_supported_claims"], 8)
        self.assertAlmostEqual(results["auxiliary_judge_coverage"], 8 / 11)
        self.assertEqual(results["auxiliary_judge_support_rate"], 1.0)
        self.assertEqual(
            results["auxiliary_judge_failures"],
            [
                {"question_id": "A02", "error_type": "Phase5GateError"},
                {"question_id": "B04", "error_type": "GenerationGateError"},
            ],
        )
        question_by_id = {row["question_id"]: row for row in report["questions"]}
        self.assertTrue(
            all(
                claim["auxiliary_model_judgment"] == {"status": "unavailable"}
                for claim in question_by_id["A02"]["claim_evidence_review"]
            )
        )
        self.assertEqual(
            question_by_id["B04"]["claim_evidence_review"][0]["auxiliary_model_judgment"],
            {"status": "unavailable"},
        )


class Phase5RefusalEvaluationTests(unittest.TestCase):
    def test_refusal_report_uses_unchanged_questions_and_valid_identity_routes(self) -> None:
        core = RefusalEvaluationCore()
        report = evaluate_refusals(project_root=ROOT, core=core)
        self.assertEqual(report["correct_refusals"], 4)
        self.assertEqual(report["question_count"], 4)
        self.assertEqual(report["correct_refusal_rate"], 1.0)
        self.assertTrue(report["target_met"])
        self.assertFalse(report["question_text_modified"])
        self.assertEqual(report["context_policy"]["mapping_frozen_before_execution"], REFUSAL_SAMPLE_MAP)
        self.assertEqual(
            core.calls,
            [(filename, question_id) for question_id, filename in REFUSAL_SAMPLE_MAP.items()],
        )
        golden = {
            row["id"]: row
            for row in json.loads((ROOT / "eval/golden_questions.json").read_text(encoding="utf-8"))
        }
        for row in report["questions"]:
            public_code, retrieval_code = SAMPLE_IDENTITY_ROUTE[row["sample_filename"]]
            with self.subTest(question_id=row["question_id"]):
                self.assertEqual(row["question"], golden[row["question_id"]]["question"])
                self.assertEqual(row["identity_status"], "identified")
                self.assertEqual(row["detected_notice_code"], public_code)
                self.assertEqual(row["retrieval_notice_code"], retrieval_code)
                self.assertEqual(row["status"], "refused")
                self.assertEqual(row["answer"], INSUFFICIENT_ANSWER)
                self.assertEqual(row["claims"], [])
                self.assertEqual(row["citations"], [])
                self.assertEqual(row["retrieved_chunk_ids"], [])
                self.assertTrue(row["provider_calls_skipped"])
                self.assertTrue(row["correct_refusal"])

    def test_refusal_report_serializes_an_actual_bad_response_instead_of_hardcoding_success(self) -> None:
        core = RefusalEvaluationCore(answered_failure_id="E04")
        report = evaluate_refusals(project_root=ROOT, core=core)
        self.assertEqual(report["correct_refusals"], 3)
        self.assertEqual(report["correct_refusal_rate"], 0.75)
        self.assertFalse(report["target_met"])
        row = next(item for item in report["questions"] if item["question_id"] == "E04")
        self.assertEqual(row["identity_status"], "identified")
        self.assertEqual(row["detected_notice_code"], "CP501")
        self.assertEqual(row["retrieval_notice_code"], "CP501")
        self.assertEqual(row["status"], "answered")
        self.assertNotEqual(row["answer"], INSUFFICIENT_ANSWER)
        self.assertEqual(len(row["claims"]), 1)
        self.assertEqual(len(row["citations"]), 1)
        self.assertEqual(row["retrieved_chunk_ids"], ["E04-frozen-chunk"])
        self.assertFalse(row["provider_calls_skipped"])
        self.assertFalse(row["correct_refusal"])


class Phase5LatencyEvaluationTests(unittest.TestCase):
    @staticmethod
    def selection(model: str = "Qwen/Qwen3-30B-A3B-Instruct-2507") -> ModelSelection:
        return ModelSelection(
            selected_model=model,
            reason="offline test selection",
            checked_at_utc="2026-08-22T00:00:00Z",
            catalog_model_count=1,
            catalog_id_set_sha256="0" * 64,
            structured_output_probe_success=True,
            structured_output_probe_seconds=0.01,
            probe_attempts=({"model": model, "success": True},),
        )

    def evaluate_with_measured_durations(self, measured: list[float]) -> tuple[dict[str, object], SequenceLatencyCore, MagicMock]:
        selected_model = "Qwen/Qwen3-30B-A3B-Instruct-2507"
        warm_core = SequenceLatencyCore([0.1] * WARMUP_REPETITIONS + measured)
        cold_cores = [SequenceLatencyCore([1.0 + index]) for index in range(COLD_REPETITIONS)]
        with patch(
            "noticelens.phase5_evaluation.create_live_core",
            side_effect=[(core, self.selection(selected_model)) for core in cold_cores],
        ) as create_core:
            report = evaluate_latency(
                project_root=ROOT,
                secrets=Phase5Secrets("sentinel-nebius", "sentinel-pinecone"),
                warm_core=warm_core,
                selected_model=selected_model,
            )
        return report, warm_core, create_core

    def test_nearest_rank_latency_statistics_are_exact(self) -> None:
        stats = _latency_stats([float(value) for value in range(1, 21)])
        self.assertEqual(
            stats,
            {"n": 20, "median_seconds": 10.5, "p95_seconds": 19.0, "max_seconds": 20.0},
        )
        self.assertEqual(
            _latency_stats([]),
            {"n": 0, "median_seconds": 0.0, "p95_seconds": 0.0, "max_seconds": 0.0},
        )

    def test_protocol_counts_components_and_strict_warm_target(self) -> None:
        measured = [1.0] * 18 + [5.999, 10.0]
        report, warm_core, create_core = self.evaluate_with_measured_durations(measured)
        protocol = report["protocol_frozen_before_execution"]
        self.assertEqual(protocol["cold_repetitions"], 3)
        self.assertEqual(protocol["warmup_repetitions"], 2)
        self.assertEqual(protocol["warm_measured_repetitions"], 20)
        self.assertEqual(protocol["case_ids"], list(LATENCY_CASE_IDS))
        self.assertEqual(protocol["p95_method"], "nearest_rank")
        self.assertEqual(protocol["target_seconds"], WARM_P95_TARGET_SECONDS)
        self.assertEqual(create_core.call_count, 3)
        self.assertEqual(len(warm_core.calls), 22)
        self.assertEqual(len(report["cold"]["runs"]), 3)
        self.assertEqual(report["cold"]["answered_count"], 3)
        self.assertEqual(len(report["warm"]["warmup_runs"]), 2)
        self.assertEqual(len(report["warm"]["runs"]), 20)
        self.assertEqual(report["warm"]["answered_count"], 20)
        self.assertEqual(report["cold"]["request_end_to_end"]["n"], 3)
        for component in ("embedding", "pinecone", "retrieval", "generation", "end_to_end"):
            self.assertEqual(report["warm"][component]["n"], 20)
        self.assertEqual(report["warm"]["retrieval"]["median_seconds"], 0.3)
        self.assertEqual(report["warm"]["end_to_end"]["median_seconds"], 1.0)
        self.assertEqual(report["warm"]["end_to_end"]["p95_seconds"], 5.999)
        self.assertEqual(report["warm"]["end_to_end"]["max_seconds"], 10.0)
        self.assertTrue(report["warm_end_to_end_p95_target_met"])

    def test_exactly_six_seconds_does_not_meet_strict_less_than_target(self) -> None:
        report, _warm_core, _create_core = self.evaluate_with_measured_durations([6.0] * 20)
        self.assertEqual(report["warm"]["end_to_end"]["p95_seconds"], 6.0)
        self.assertFalse(report["warm_end_to_end_p95_target_met"])

    def test_nonanswered_warm_request_is_measured_and_fails_target_without_aborting(self) -> None:
        selected_model = "Qwen/Qwen3-30B-A3B-Instruct-2507"
        statuses = ["answered"] * (WARMUP_REPETITIONS + WARM_MEASURED_REPETITIONS)
        statuses[-1] = "refused"
        warm_core = SequenceLatencyCore([1.0] * len(statuses), statuses=statuses)
        cold_cores = [SequenceLatencyCore([1.0]) for _ in range(COLD_REPETITIONS)]
        with patch(
            "noticelens.phase5_evaluation.create_live_core",
            side_effect=[(core, self.selection(selected_model)) for core in cold_cores],
        ):
            report = evaluate_latency(
                project_root=ROOT,
                secrets=Phase5Secrets("sentinel-nebius", "sentinel-pinecone"),
                warm_core=warm_core,
                selected_model=selected_model,
            )
        self.assertEqual(report["warm"]["end_to_end"]["n"], 20)
        self.assertEqual(report["warm"]["answered_count"], 19)
        self.assertEqual(report["warm"]["answered_rate"], 0.95)
        self.assertFalse(report["warm_end_to_end_p95_target_met"])


class Phase5ReportWriterTests(unittest.TestCase):
    @staticmethod
    def payloads() -> tuple[ModelSelection, dict[str, object], dict[str, object], dict[str, object], dict[str, object]]:
        selection = Phase5LatencyEvaluationTests.selection()
        generation = {
            "schema_version": "1.0",
            "results": {
                "answerable_questions": 10,
                "answered_questions": 10,
                "atomic_claims": 10,
                "deterministically_supported_claims": 10,
                "citation_support_rate": 1.0,
                "formal_faithfulness": None,
                "formal_faithfulness_status": "pending_human_review",
                "frozen_faithfulness_target": 0.95,
                "frozen_faithfulness_target_met": None,
                "auxiliary_judge_support_rate": 1.0,
            },
            "method": {"human_evidence_review_completed": False},
            "questions": [],
        }
        refusal = {
            "schema_version": "1.0",
            "correct_refusals": 4,
            "question_count": 4,
            "correct_refusal_rate": 1.0,
            "target_met": True,
            "questions": [],
        }
        latency = {
            "schema_version": "1.0",
            "cold": {
                "answered_count": 3,
                "end_to_end_including_initialization": {
                    "n": 3,
                    "median_seconds": 3.0,
                    "p95_seconds": 4.0,
                    "max_seconds": 4.0,
                },
            },
            "warm": {
                "answered_count": 20,
                "retrieval": {"n": 20, "median_seconds": 0.3, "p95_seconds": 0.4, "max_seconds": 0.5},
                "generation": {"n": 20, "median_seconds": 1.0, "p95_seconds": 1.5, "max_seconds": 2.0},
                "end_to_end": {"n": 20, "median_seconds": 1.5, "p95_seconds": 2.0, "max_seconds": 2.5},
            },
            "warm_end_to_end_p95_target_met": True,
        }
        input_eval = {"identified_correctly": 8, "sample_count": 8, "ocr_requests": 0}
        return selection, generation, refusal, latency, input_eval

    def test_report_schema_configuration_and_truthful_pending_status(self) -> None:
        selection, generation, refusal, latency, input_eval = self.payloads()
        frozen = {"approved_phase1_to_4b_composite_sha256": PHASE5_FROZEN_COMPOSITE_SHA256}
        nebius = "SENTINEL_REPORT_NEBIUS_SECRET"
        pinecone = "SENTINEL_REPORT_PINECONE_SECRET"
        with tempfile.TemporaryDirectory() as temporary:
            project_root = Path(temporary)
            paths = write_reports(
                project_root=project_root,
                model_selection=selection,
                frozen_before=frozen,
                frozen_after=dict(frozen),
                tests={"passed": 100, "failed": 0},
                input_eval=input_eval,
                generation_eval=generation,
                refusal_eval=refusal,
                latency_eval=latency,
                secrets=Phase5Secrets(nebius, pinecone),
            )
            self.assertEqual(
                {path.relative_to(project_root) for path in paths},
                {
                    GENERATION_REPORT_PATH,
                    REFUSAL_REPORT_PATH,
                    LATENCY_REPORT_PATH,
                    FINAL_CONFIG_PATH,
                    FINAL_REPORT_PATH,
                },
            )
            generation_json = json.loads((project_root / GENERATION_REPORT_PATH).read_text(encoding="utf-8"))
            refusal_json = json.loads((project_root / REFUSAL_REPORT_PATH).read_text(encoding="utf-8"))
            latency_json = json.loads((project_root / LATENCY_REPORT_PATH).read_text(encoding="utf-8"))
            config_json = json.loads((project_root / FINAL_CONFIG_PATH).read_text(encoding="utf-8"))
            markdown = (project_root / FINAL_REPORT_PATH).read_text(encoding="utf-8")
            self.assertIsNone(generation_json["results"]["formal_faithfulness"])
            self.assertEqual(
                generation_json["results"]["formal_faithfulness_status"],
                "pending_human_review",
            )
            self.assertIsNone(generation_json["results"]["frozen_faithfulness_target_met"])
            self.assertEqual(generation_json["results"]["citation_support_rate"], 1.0)
            self.assertEqual(generation_json["offline_tests"], {"passed": 100, "failed": 0})
            self.assertEqual(refusal_json, refusal)
            self.assertEqual(latency_json, latency)
            self.assertEqual(
                config_json,
                {
                    "embedding_model": EMBEDDING_MODEL,
                    "embedding_dimension": 4096,
                    "generation_model": selection.selected_model,
                    "pinecone_index": "noticelens-rag",
                    "production_namespace": "heading-aware-dense",
                    "chunk_strategy": "heading_aware_220_40",
                    "top_k": 5,
                    "metadata_filter": "exact notice_code equality",
                    "bm25": False,
                    "hybrid_retrieval": False,
                    "reranking": False,
                    "decision_basis": config_json["decision_basis"],
                },
            )
            self.assertIn("formal frozen-plan faithfulness is **pending human review**", markdown)
            self.assertIn("target is not claimed as met", markdown)
            self.assertIn("exact-source citation support", markdown)
            self.assertIn("4/4 correct", markdown)
            self.assertIn("Warm end-to-end p95:** 2.000s", markdown)
            for path in paths:
                text = path.read_text(encoding="utf-8")
                self.assertNotIn(nebius, text)
                self.assertNotIn(pinecone, text)

    def test_report_writer_rejects_freeze_mismatch_before_writing(self) -> None:
        selection, generation, refusal, latency, input_eval = self.payloads()
        before = {"approved_phase1_to_4b_composite_sha256": PHASE5_FROZEN_COMPOSITE_SHA256}
        after = {"approved_phase1_to_4b_composite_sha256": "changed"}
        with tempfile.TemporaryDirectory() as temporary:
            project_root = Path(temporary)
            with self.assertRaisesRegex(Phase5GateError, "frozen"):
                write_reports(
                    project_root=project_root,
                    model_selection=selection,
                    frozen_before=before,
                    frozen_after=after,
                    tests={"passed": 100, "failed": 0},
                    input_eval=input_eval,
                    generation_eval=generation,
                    refusal_eval=refusal,
                    latency_eval=latency,
                    secrets=Phase5Secrets("sentinel-nebius", "sentinel-pinecone"),
                )
            self.assertFalse((project_root / "reports").exists())

    def test_report_writer_scans_every_artifact_for_secret_sentinels(self) -> None:
        selection, generation, refusal, latency, input_eval = self.payloads()
        sentinel = "SENTINEL_SECRET_MUST_TRIGGER_ARTIFACT_SCAN"
        generation["poisoned_test_value"] = sentinel
        frozen = {"approved_phase1_to_4b_composite_sha256": PHASE5_FROZEN_COMPOSITE_SHA256}
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(Phase5GateError) as captured:
                write_reports(
                    project_root=Path(temporary),
                    model_selection=selection,
                    frozen_before=frozen,
                    frozen_after=dict(frozen),
                    tests={"passed": 100, "failed": 0},
                    input_eval=input_eval,
                    generation_eval=generation,
                    refusal_eval=refusal,
                    latency_eval=latency,
                    secrets=Phase5Secrets(sentinel, "second-sentinel-secret"),
                )
        self.assertNotIn(sentinel, str(captured.exception))


class ExternalSecretSafetyTests(unittest.TestCase):
    def test_loader_uses_only_explicit_external_file_and_repr_hides_sentinels(self) -> None:
        nebius = "SENTINEL_NEBIUS_SECRET_DO_NOT_LOG"
        pinecone = "SENTINEL_PINECONE_SECRET_DO_NOT_LOG"
        with tempfile.TemporaryDirectory() as temporary:
            external = Path(temporary) / ".noticelens.env"
            external.write_text(
                f"NEBIUS_API_KEY={nebius}\nPINECONE_API_KEY={pinecone}\n",
                encoding="utf-8",
            )
            injected_environment: dict[str, str] = {}
            secrets = load_phase5_secrets(
                project_root=ROOT,
                external_path=external,
                environ=injected_environment,
            )
        self.assertEqual(injected_environment["NEBIUS_API_KEY"], nebius)
        self.assertEqual(injected_environment["PINECONE_API_KEY"], pinecone)
        self.assertNotIn(nebius, repr(secrets))
        self.assertNotIn(pinecone, repr(secrets))
        self.assertEqual(secrets.public_summary(), {"secrets_source": "external_local_file"})
        public = json.dumps(secrets.public_summary())
        self.assertNotIn(nebius, public)
        self.assertNotIn(pinecone, public)

    def test_loader_rejects_any_project_local_dotenv_before_reading_it(self) -> None:
        local_path = ROOT / "forbidden-local.env"
        with patch("noticelens.phase5.dotenv_values") as dotenv_loader:
            with self.assertRaisesRegex(Phase5GateError, "inside the project"):
                load_phase5_secrets(
                    project_root=ROOT,
                    external_path=local_path,
                    environ={},
                )
        dotenv_loader.assert_not_called()

    def test_project_has_no_local_env_and_gitignore_covers_dotenv_files(self) -> None:
        ignore = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
        self.assertIn(".env", ignore)
        self.assertIn("*.env", ignore)
        self.assertFalse((ROOT / ".env").exists())
        self.assertFalse((ROOT / "data/.env").exists())

    def test_provider_exception_and_generator_repr_never_echo_secret(self) -> None:
        sentinel = "SENTINEL_NEBIUS_SECRET_IN_EXCEPTION"
        client = MagicMock()
        client.models.list.side_effect = RuntimeError(f"provider accidentally echoed {sentinel}")
        generator = NebiusGroundedGenerator(api_key=sentinel, client=client)
        self.assertNotIn(sentinel, repr(generator))
        with self.assertRaises(GenerationGateError) as captured:
            generator.select_live_model()
        self.assertNotIn(sentinel, str(captured.exception))


class FrozenPhaseOneThroughFourBTests(unittest.TestCase):
    def test_exact_approved_files_trees_and_composite_remain_frozen(self) -> None:
        observed = verify_phase5_frozen_inputs(ROOT)
        self.assertEqual(
            observed["approved_phase1_to_4b_composite_sha256"],
            PHASE5_FROZEN_COMPOSITE_SHA256,
        )
        self.assertEqual(observed["phase4b_files"], PHASE4B_FROZEN_HASHES)
        # 18 pinned files + 3 pinned trees + the approved composite digest.
        self.assertEqual(len(observed["phase1_to_4a"]), 22)
        for frozen_tree in ("data/raw/guidance", "data/raw/sample_notices", "data/processed/guidance"):
            self.assertIn(frozen_tree, observed["phase1_to_4a"])
        self.assertEqual(len(REGISTRY_RECORDS), 580)
        self.assertEqual(len({record.chunk_id for record in REGISTRY_RECORDS}), 580)


if __name__ == "__main__":
    unittest.main()
