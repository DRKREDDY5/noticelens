from __future__ import annotations

import ast
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import streamlit as st
from langchain_core.documents import Document
from streamlit.testing.v1 import AppTest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from noticelens.final_retrieval import FinalRetrievalError  # noqa: E402
from noticelens.grounded_generation import GenerationGateError  # noqa: E402
from noticelens.notice_input import (  # noqa: E402
    ExtractedField,
    ExtractedNotice,
    NoticeFields,
    NoticeIdentity,
    NoticeInputError,
)
from noticelens.phase5 import CoreRun, Phase5GateError  # noqa: E402
from noticelens.phase5_models import (  # noqa: E402
    Citation,
    GroundedClaim,
    GroundedResponse,
    ResponseNoticeFields,
)
from noticelens.streamlit_ui import (  # noqa: E402
    DEFAULT_ANALYSIS_QUESTION,
    StreamlitUiError,
    analyze_sample,
    analyze_upload,
    ask_notice,
    build_evidence_cards,
    build_trace_rows,
    evidence_status,
    extract_uploaded_notice,
    load_product_snapshot,
    load_sample_notices,
    notice_detail_rows,
    safe_error_message,
)


def _extracted() -> ExtractedNotice:
    return ExtractedNotice(
        display_name="cp501_english.pdf",
        pages=("Notice CP501\nA sufficiently long official sample first page for testing.",),
        text="Notice CP501\nA sufficiently long official sample first page for testing.",
    )


def _run(*, source_url: str = "https://www.irs.gov/individuals/understanding-your-cp501-notice") -> CoreRun:
    identity = NoticeIdentity(
        status="identified",
        notice_code="CP501",
        retrieval_notice_code="CP501",
        confidence=0.99,
        evidence_text="Notice CP501",
    )
    fields = NoticeFields(
        notice_code=ExtractedField("CP501", 0.99, "Notice CP501"),
        notice_date=ExtractedField("January 2, 2026", 0.98, "Notice date: January 2, 2026"),
        due_or_response_date=ExtractedField(None, 0.0, None),
        amount=ExtractedField("$125.00", 0.95, "Amount due: $125.00"),
        reference_number=ExtractedField(None, 0.0, None),
    )
    heading_path = ["Understanding your CP501 notice", "What you need to do"]
    document = Document(
        page_content=(
            "Notice: CP501\nTitle: Understanding your CP501 notice\n"
            "Heading: What you need to do\nPay the amount you owe or contact the IRS."
        ),
        metadata={
            "doc_id": "irs_cp501",
            "notice_code": "CP501",
            "title": "Understanding your CP501 notice",
            "source_url": source_url,
            "chunk_id": "irs_cp501::h0002::s000",
            "heading": "What you need to do",
            "heading_path": heading_path,
            "similarity_score": 0.912345,
        },
    )
    citation = Citation(
        citation_id="C1",
        notice_code="CP501",
        source_title="Understanding your CP501 notice",
        heading="What you need to do",
        heading_path=heading_path,
        source_url=source_url,
        chunk_id="irs_cp501::h0002::s000",
    )
    claim = GroundedClaim(
        claim_id="CL1",
        text="IRS guidance says: Pay the amount you owe or contact the IRS.",
        evidence_type="guidance",
        citation_ids=["C1"],
        notice_field_names=[],
    )
    response = GroundedResponse(
        status="answered",
        notice_code="CP501",
        answer=claim.text,
        claims=[claim],
        citations=[citation],
        notice_fields=ResponseNoticeFields.from_notice_fields(fields),
    )
    return CoreRun(
        response=response,
        timings={"embedding_seconds": 0.01, "pinecone_seconds": 0.02},
        identity=identity,
        fields=fields,
        documents=[document],
        policy_refusal_reason=None,
    )


class RecordingCore:
    def __init__(self, run: CoreRun | None = None) -> None:
        self.result = run or _run()
        self.calls: list[tuple[ExtractedNotice, str]] = []

    def run_extracted(self, extracted: ExtractedNotice, question: str) -> CoreRun:
        self.calls.append((extracted, question))
        return self.result


class SampleAndUploadTests(unittest.TestCase):
    def test_exact_eight_verified_samples_are_contained_and_paths_are_not_labels(self) -> None:
        samples = load_sample_notices(PROJECT_ROOT)
        self.assertEqual(len(samples), 8)
        self.assertEqual(
            [sample.notice_code for sample in samples],
            ["CP501", "CP59", "CP05A", "CP523H", "CP503C", "LT11", "CP131", "CP44"],
        )
        approved_root = (PROJECT_ROOT / "data/raw/sample_notices").resolve()
        for sample in samples:
            self.assertEqual(sample.path.resolve().relative_to(approved_root).parts, (sample.filename,))
            self.assertTrue(sample.path.is_file())
            self.assertTrue(sample.source_url.startswith("https://www.irs.gov/"))
            self.assertNotIn(str(PROJECT_ROOT), sample.label)
            self.assertNotIn(sample.filename, sample.label)

    def test_sample_manifest_rejects_a_traversal_filename(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "data/raw/sample_notices").mkdir(parents=True)
            manifest = PROJECT_ROOT / "data/sample_notice_manifest.csv"
            text = manifest.read_text(encoding="utf-8")
            (root / "data").mkdir(exist_ok=True)
            (root / "data/sample_notice_manifest.csv").write_text(
                text.replace("cp501_english.pdf", "../outside.pdf", 1),
                encoding="utf-8",
            )
            with self.assertRaises(StreamlitUiError):
                load_sample_notices(root)

    def test_upload_uses_a_temporary_pdf_and_removes_it_after_extraction(self) -> None:
        observed: dict[str, Path] = {}

        def fake_extract(path: Path) -> ExtractedNotice:
            observed["file"] = path
            observed["directory"] = path.parent
            self.assertTrue(path.is_file())
            self.assertEqual(path.read_bytes(), b"%PDF-offline-test")
            return _extracted()

        with patch("noticelens.streamlit_ui.extract_pdf_text", side_effect=fake_extract) as extractor:
            extracted = extract_uploaded_notice(b"%PDF-offline-test", "../../safe-name.pdf")
        extractor.assert_called_once()
        self.assertEqual(extracted.display_name, "safe-name.pdf")
        self.assertFalse(observed["file"].exists())
        self.assertFalse(observed["directory"].exists())

    def test_invalid_upload_is_rejected_before_pdf_parsing_or_temp_storage(self) -> None:
        with patch("noticelens.streamlit_ui.extract_pdf_text") as extractor:
            for data, name in ((b"not-a-pdf", "notice.pdf"), (b"%PDF-content", "notice.txt"), (b"", "notice.pdf")):
                with self.subTest(name=name, data=data[:5]):
                    with self.assertRaises(NoticeInputError):
                        extract_uploaded_notice(data, name)
        extractor.assert_not_called()

    def test_analysis_and_chat_helpers_delegate_every_turn_to_the_frozen_core(self) -> None:
        core = RecordingCore()
        extracted = _extracted()
        sample = load_sample_notices(PROJECT_ROOT)[0]
        with patch("noticelens.streamlit_ui.extract_pdf_text", return_value=extracted) as parser:
            sample_extracted, sample_run = analyze_sample(core, sample)
        parser.assert_called_once_with(sample.path)
        self.assertIs(sample_extracted, extracted)
        self.assertIs(sample_run, core.result)
        self.assertEqual(core.calls[-1], (extracted, DEFAULT_ANALYSIS_QUESTION))

        with patch("noticelens.streamlit_ui.extract_uploaded_notice", return_value=extracted) as parser:
            upload_extracted, upload_run = analyze_upload(core, b"%PDF-test", "notice.pdf", "Explain it")
        parser.assert_called_once_with(b"%PDF-test", "notice.pdf")
        self.assertIs(upload_extracted, extracted)
        self.assertIs(upload_run, core.result)
        self.assertEqual(core.calls[-1], (extracted, "Explain it"))

        self.assertIs(ask_notice(core, extracted, "  What happens next?  "), core.result)
        self.assertEqual(core.calls[-1], (extracted, "What happens next?"))
        with self.assertRaises(StreamlitUiError):
            ask_notice(core, extracted, "   ")


class EvidenceAndTraceTests(unittest.TestCase):
    def test_citation_cards_and_trace_rows_bind_only_to_retrieved_documents(self) -> None:
        run = _run()
        cards = build_evidence_cards(run)
        self.assertEqual(len(cards), 1)
        self.assertEqual(cards[0].chunk_id, run.documents[0].metadata["chunk_id"])
        self.assertEqual(cards[0].source_url, run.documents[0].metadata["source_url"])
        self.assertEqual(cards[0].heading_path, tuple(run.documents[0].metadata["heading_path"]))
        self.assertEqual(cards[0].evidence_excerpt, "Pay the amount you owe or contact the IRS.")
        self.assertEqual(evidence_status(run), "GROUNDED")

        rows = build_trace_rows(run)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["Rank"], 1)
        self.assertEqual(rows[0]["Similarity"], 0.912345)
        self.assertEqual(rows[0]["Chunk ID"], cards[0].chunk_id)
        self.assertLessEqual(len(rows[0]["Preview"]), 420)

    def test_citation_metadata_mismatch_and_unbacked_chunk_are_rejected(self) -> None:
        run = _run()
        run.documents[0].metadata["heading"] = "Tampered heading"
        with self.assertRaises(StreamlitUiError):
            build_evidence_cards(run)

        run = _run()
        run.documents.clear()
        with self.assertRaises(StreamlitUiError):
            build_evidence_cards(run)

    def test_trace_rejects_non_irs_source_missing_metadata_and_nonfinite_score(self) -> None:
        run = _run()
        run.documents[0].metadata["source_url"] = "https://example.com/not-irs"
        with self.assertRaises(StreamlitUiError):
            build_trace_rows(run)

        run = _run()
        del run.documents[0].metadata["heading"]
        with self.assertRaises(StreamlitUiError):
            build_trace_rows(run)

        run = _run()
        run.documents[0].metadata["similarity_score"] = float("nan")
        with self.assertRaises(StreamlitUiError):
            build_trace_rows(run)

    def test_notice_details_are_null_safe(self) -> None:
        rows = dict(notice_detail_rows(_run()))
        self.assertEqual(rows["Notice date"], "January 2, 2026")
        self.assertEqual(rows["Amount"], "$125.00")
        self.assertEqual(rows["Due / response date"], "Not confidently identified")
        self.assertEqual(rows["Reference number"], "Not confidently identified")


class SnapshotAndSafetyTests(unittest.TestCase):
    def test_snapshot_matches_the_frozen_configuration_metrics_and_limitations(self) -> None:
        snapshot = load_product_snapshot(PROJECT_ROOT)
        self.assertEqual(snapshot.corpus_documents, 50)
        self.assertEqual(snapshot.sample_notices, 8)
        self.assertEqual(snapshot.final_config["embedding_model"], "Qwen/Qwen3-Embedding-8B")
        self.assertEqual(snapshot.final_config["embedding_dimension"], 4096)
        self.assertEqual(snapshot.final_config["generation_model"], "Qwen/Qwen3-30B-A3B-Instruct-2507")
        self.assertEqual(snapshot.final_config["chunk_strategy"], "heading_aware_220_40")
        self.assertEqual(snapshot.final_config["top_k"], 5)
        self.assertFalse(snapshot.final_config["bm25"])
        self.assertFalse(snapshot.final_config["hybrid_retrieval"])
        self.assertFalse(snapshot.final_config["reranking"])
        self.assertAlmostEqual(snapshot.notice_dense_p1, 1.0)
        self.assertAlmostEqual(snapshot.fixed_section_p1, 0.8)
        self.assertAlmostEqual(snapshot.heading_section_p1, 14 / 15)
        self.assertAlmostEqual(snapshot.heading_gain_percentage_points, 13.33333333333333)
        self.assertEqual(dict(snapshot.ablation_p1), {
            "Dense": 14 / 15,
            "BM25": 6 / 15,
            "Hybrid": 10 / 15,
            "Hybrid + reranker": 14 / 15,
        })
        self.assertEqual(snapshot.faithfulness, 1.0)
        self.assertEqual(snapshot.refusal_rate, 1.0)
        self.assertAlmostEqual(snapshot.warm_median_seconds, 9.775036)
        self.assertAlmostEqual(snapshot.warm_p95_seconds, 22.050282)
        self.assertEqual(snapshot.latency_target_seconds, 6.0)
        self.assertFalse(snapshot.latency_target_met)

    def test_snapshot_fails_closed_when_a_frozen_value_is_changed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shutil.copytree(PROJECT_ROOT / "reports", root / "reports")
            (root / "data").mkdir()
            shutil.copy2(PROJECT_ROOT / "data/corpus_manifest.csv", root / "data/corpus_manifest.csv")
            shutil.copy2(PROJECT_ROOT / "data/sample_notice_manifest.csv", root / "data/sample_notice_manifest.csv")
            path = root / "reports/final_retrieval_config.json"
            text = path.read_text(encoding="utf-8").replace('"top_k": 5', '"top_k": 6')
            path.write_text(text, encoding="utf-8")
            with self.assertRaises(StreamlitUiError):
                load_product_snapshot(root)

    def test_safe_errors_never_echo_exception_text_paths_tracebacks_or_secret_shapes(self) -> None:
        sentinel = "FAKE_SECRET_VALUE_123"
        unsafe = f"NEBIUS_API_KEY={sentinel} at C:/Users/example/.noticelens.env\nTraceback"
        cases = (
            NoticeInputError(unsafe),
            FinalRetrievalError(unsafe),
            GenerationGateError(unsafe),
            Phase5GateError(unsafe),
            RuntimeError(unsafe),
        )
        for error in cases:
            with self.subTest(error_type=type(error).__name__):
                message = safe_error_message(error)
                self.assertNotIn(sentinel, message)
                self.assertNotIn("NEBIUS_API_KEY", message)
                self.assertNotIn("C:/Users", message)
                self.assertNotIn("Traceback", message)
                self.assertNotIn(str(error), message)


class OfflineStreamlitTests(unittest.TestCase):
    def tearDown(self) -> None:
        st.cache_data.clear()
        st.cache_resource.clear()

    def test_no_notice_app_has_exactly_three_tabs_and_makes_no_provider_call(self) -> None:
        st.cache_data.clear()
        st.cache_resource.clear()
        with patch("noticelens.phase5.load_phase5_secrets", side_effect=AssertionError("provider forbidden")) as secrets, patch(
            "noticelens.phase5.create_live_core", side_effect=AssertionError("provider forbidden")
        ) as creator:
            app = AppTest.from_file(PROJECT_ROOT / "app.py", default_timeout=12).run(timeout=12)
        self.assertEqual(len(app.exception), 0)
        self.assertEqual([tab.label for tab in app.tabs], ["Analyze Notice", "Ask NoticeLens", "RAG X-Ray"])
        self.assertIn(
            "Analyze a notice first to ask evidence-grounded questions.",
            [message.value for message in app.info],
        )
        self.assertEqual(len(app.chat_input), 0)
        secrets.assert_not_called()
        creator.assert_not_called()

    def test_sample_analysis_and_active_chat_use_one_injected_core(self) -> None:
        st.cache_data.clear()
        st.cache_resource.clear()
        core = RecordingCore()
        selected = SimpleNamespace(selected_model="Qwen/Qwen3-30B-A3B-Instruct-2507")
        with patch("noticelens.phase5.load_phase5_secrets", return_value=object()) as secrets, patch(
            "noticelens.phase5.create_live_core", return_value=(core, selected)
        ) as creator:
            app = AppTest.from_file(PROJECT_ROOT / "app.py", default_timeout=15).run(timeout=15)
            app.button[0].click()
            app = app.run(timeout=15)
            self.assertEqual(len(app.exception), 0)
            self.assertEqual(len(core.calls), 1)
            self.assertEqual(core.calls[0][1], DEFAULT_ANALYSIS_QUESTION)
            self.assertEqual(len(app.chat_input), 1)
            self.assertTrue(any(markdown.value == "### CP501" for markdown in app.markdown))
            self.assertTrue(any("Evidence 1" in expander.label for expander in app.expander))

            app.chat_input[0].set_value("What happens at this stage?")
            app = app.run(timeout=15)
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(core.calls), 2)
        self.assertEqual(core.calls[1][1], "What happens at this stage?")
        self.assertTrue(any(message.name == "user" for message in app.chat_message))
        self.assertTrue(any(message.name == "assistant" for message in app.chat_message))
        rendered_text = [*app.markdown, *app.text]
        self.assertTrue(any("Pay the amount you owe" in text.value for text in rendered_text))
        secrets.assert_called_once()
        creator.assert_called_once()

    def test_app_sources_are_responsive_and_have_no_direct_provider_or_mutation_path(self) -> None:
        app_source = (PROJECT_ROOT / "app.py").read_text(encoding="utf-8")
        helper_source = (PROJECT_ROOT / "src/noticelens/streamlit_ui.py").read_text(encoding="utf-8")
        combined = app_source + "\n" + helper_source
        self.assertIn("@media (max-width: 640px)", app_source)
        self.assertIn("overflow-x: hidden", app_source)
        self.assertIn('[data-testid="stDataFrame"]', app_source)
        self.assertIn("overflow: auto", app_source)
        self.assertIn('[data-testid="stChatInput"] { width: 100%; }', app_source)
        self.assertIn("flex: 1 1 100%", app_source)

        trees = (ast.parse(app_source), ast.parse(helper_source))
        imported_modules = {
            node.module
            for tree in trees
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module
        }
        self.assertFalse(
            imported_modules.intersection(
                {
                    "openai",
                    "pinecone",
                    "noticelens.providers",
                    "noticelens.retrieval_ablation",
                    "noticelens.phase3",
                    "noticelens.phase4a",
                    "noticelens.phase4b",
                }
            )
        )
        called_attributes = {
            node.func.attr
            for tree in trees
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        }
        self.assertFalse(
            called_attributes.intersection(
                {"upsert", "delete", "delete_index", "create_index", "configure_index", "rerank"}
            )
        )
        self.assertNotIn("st.text_input", combined)
        for prohibited in ("Mem0", "LlamaIndex", "Lyzr", "ElevenLabs"):
            self.assertNotIn(prohibited, combined)


if __name__ == "__main__":
    unittest.main()
