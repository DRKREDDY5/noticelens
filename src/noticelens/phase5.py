"""Phase 5 final NoticeLens RAG core and minimal LangGraph."""

from __future__ import annotations

import csv
import hashlib
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, MutableMapping

from dotenv import dotenv_values, load_dotenv
from langgraph.graph import END, START, StateGraph

from .chunking import sha256_file
from .final_retrieval import FinalHeadingRetriever, RetrievalResult, evidence_is_sufficient
from .grounded_generation import (
    INSUFFICIENT_ANSWER,
    ModelSelection,
    NebiusGroundedGenerator,
    build_grounded_response,
    refusal_response,
)
from .notice_input import (
    ExtractedNotice,
    NoticeFields,
    NoticeIdentity,
    extract_notice_fields,
    extract_pdf_text,
    identify_notice,
    relevant_notice_field_names,
    select_relevant_notice_context,
)
from .phase4b import (
    FROZEN_FILE_HASHES,
    FROZEN_TREE_HASHES,
    _tree_digest,
    verify_frozen_artifacts,
)
from .phase5_models import GroundedResponse, NoticeLensState, ResponseNoticeFields


DEFAULT_EXTERNAL_SECRETS_PATH = Path("C:/Users/donur/.noticelens.env")
PHASE4B_FROZEN_HASHES = {
    "data/derived/phase4b/heading_aware_220_40_chunks.jsonl": "3aecf5db7ee5fe857bdb99156c9bb5ba585f2e845aa6697cc4ece8902ac27572",
    "reports/phase4b_chunk_audit.json": "88265f578c3f65b9eae8a42b2ab4c52e1e295f87c1f0f679c5169a427869cc4e",
    "reports/phase4b_heading_results.json": "36188d2d9f273b0cbe81a77d59e980c11ae1bbdd7748797ee47f1b29a140d189",
    "reports/phase4b_heading_summary.csv": "357b4dd9f930f210e4fc957fa339b6a180d44908bdcd9098efb1b8df14de8617",
    "reports/phase4b_comparison.csv": "31541ba4c1533354f6c7631fa853c6a914fba247781655832c1462dbac10b3a2",
    "reports/phase4b_failure_analysis.md": "049305e35ba8bc22dcedca21978d3f71e33e6c8b2755101157ebe9a25df0a516",
}
PHASE5_FROZEN_COMPOSITE_SHA256 = "5df91e5deaba6fb47b69e6227718956575eea3780c83dbf3df0843d747be95fa"
REQUIRED_SECRET_NAMES = ("NEBIUS_API_KEY", "PINECONE_API_KEY")


class Phase5GateError(RuntimeError):
    """A safe Phase 5 configuration or frozen-input failure."""


@dataclass(frozen=True)
class Phase5Secrets:
    nebius_api_key: str = field(repr=False)
    pinecone_api_key: str = field(repr=False)

    def public_summary(self) -> dict[str, str]:
        return {"secrets_source": "external_local_file"}


def _within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def load_phase5_secrets(
    *,
    project_root: Path,
    external_path: Path = DEFAULT_EXTERNAL_SECRETS_PATH,
    environ: MutableMapping[str, str] | None = None,
) -> Phase5Secrets:
    """Load only the explicit external dotenv, then access the two env names."""

    candidate = external_path.expanduser().resolve()
    if _within(candidate, project_root.resolve()):
        raise Phase5GateError("Refusing to load credentials from inside the project workspace")
    if not candidate.is_file():
        raise Phase5GateError("The configured external credential file does not exist")
    if environ is None:
        # The real application path intentionally populates os.environ and then
        # accesses only the two documented names.
        load_dotenv(dotenv_path=candidate, override=False, verbose=False)
        source: Mapping[str, str] = os.environ
    else:
        values = dotenv_values(candidate)
        for name in REQUIRED_SECRET_NAMES:
            if not environ.get(name, "").strip():
                value = values.get(name)
                if isinstance(value, str) and value.strip():
                    environ[name] = value.strip()
        source = environ
    missing = [name for name in REQUIRED_SECRET_NAMES if not source.get(name, "").strip()]
    if missing:
        raise Phase5GateError("Missing required environment variable(s): " + ", ".join(missing))
    return Phase5Secrets(
        nebius_api_key=source["NEBIUS_API_KEY"].strip(),
        pinecone_api_key=source["PINECONE_API_KEY"].strip(),
    )


def verify_phase5_frozen_inputs(project_root: Path) -> dict[str, Any]:
    """Verify every approved Phase 1-4B file/tree against pinned hashes."""

    phase1_to_4a = verify_frozen_artifacts(project_root)
    failures: list[str] = []
    phase4b_observed: dict[str, str] = {}
    for relative, expected in PHASE4B_FROZEN_HASHES.items():
        path = project_root / relative
        if not path.is_file():
            failures.append(f"missing:{relative}")
            continue
        actual = sha256_file(path)
        phase4b_observed[relative] = actual
        if actual != expected:
            failures.append(f"hash:{relative}")
    composite_records: list[str] = []
    for relative in FROZEN_FILE_HASHES:
        composite_records.append(f"{relative}|{sha256_file(project_root / relative)}")
    for relative in PHASE4B_FROZEN_HASHES:
        if (project_root / relative).is_file():
            composite_records.append(f"{relative}|{sha256_file(project_root / relative)}")
    for relative in FROZEN_TREE_HASHES:
        digest, count = _tree_digest(project_root / relative)
        composite_records.append(f"{relative}|tree|{count}|{digest}")
    composite = hashlib.sha256("\n".join(composite_records).encode("utf-8")).hexdigest()
    if composite != PHASE5_FROZEN_COMPOSITE_SHA256:
        failures.append("phase5_approved_composite")
    if failures:
        raise Phase5GateError("Frozen Phase 1-4B gate failed: " + ", ".join(failures))
    return {
        "phase1_to_4a": phase1_to_4a,
        "phase4b_files": phase4b_observed,
        "approved_phase1_to_4b_composite_sha256": composite,
    }


def _manifest_codes(path: Path) -> set[str]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    codes = {str(row.get("notice_code", "")).strip() for row in rows}
    if len(rows) != 50 or "" in codes:
        raise Phase5GateError("The frozen corpus manifest code inventory is invalid")
    return codes


def classify_unsupported_question(question: str) -> str | None:
    """Categorical policy rules; no benchmark IDs or notice-specific tuning."""

    normalized = " ".join(question.casefold().split())
    if re.search(r"\b(?:invent|fabricate|make[- ]?up|made[- ]?up)\b", normalized):
        return "fabrication_request"
    if re.search(r"\b(?:crypto|cryptocurrency|bitcoin|stock|token)\b", normalized) and re.search(
        r"\b(?:buy|sell|profit|invest|recommend|fastest)\b", normalized
    ):
        return "out_of_domain_financial_recommendation"
    if re.search(r"\b(?:fraud|tax evasion)\b", normalized) and re.search(
        r"\b(?:prove|proof|intent|intentionally|decided|committed|guilty)\b", normalized
    ):
        return "unsupported_intent_or_fraud_inference"
    if re.search(r"\b(?:guidance|webpage|website)\s+(?:alone|only)\b", normalized) and re.search(
        r"\b(?:exact|personal|my)\b", normalized
    ) and re.search(r"\b(?:amount|owe|date|deadline|due)\b", normalized):
        return "taxpayer_specific_fact_excluded_by_source_scope"
    if re.search(r"\bshould i\b", normalized) and re.search(
        r"\b(?:cash out|retirement|borrow|loan|sell|invest|bankruptcy)\b", normalized
    ):
        return "personalized_tax_or_financial_advice"
    return None


@dataclass(frozen=True)
class CoreRun:
    response: GroundedResponse
    timings: dict[str, float]
    identity: NoticeIdentity
    fields: NoticeFields
    documents: list[Any]
    policy_refusal_reason: str | None


class NoticeLensCore:
    """Minimal notice-specific graph with deterministic gates."""

    def __init__(
        self,
        *,
        project_root: Path,
        retriever: Any,
        generator: NebiusGroundedGenerator,
        available_guidance_codes: set[str] | None = None,
    ) -> None:
        self.project_root = project_root
        self.retriever = retriever
        self.generator = generator
        self.available_guidance_codes = available_guidance_codes or _manifest_codes(
            project_root / "data/corpus_manifest.csv"
        )
        workflow = StateGraph(NoticeLensState)
        workflow.add_node("identify_notice", self._identify_notice_node)
        workflow.add_node("clarify_or_fail", self._clarify_or_fail_node)
        workflow.add_node("retrieve_guidance", self._retrieve_guidance_node)
        workflow.add_node("refuse", self._refuse_node)
        workflow.add_node("generate_grounded_answer", self._generate_grounded_answer_node)
        workflow.add_edge(START, "identify_notice")
        workflow.add_conditional_edges(
            "identify_notice",
            lambda state: "identified" if state["notice_identity"].status == "identified" else "not_identified",
            {"identified": "retrieve_guidance", "not_identified": "clarify_or_fail"},
        )
        workflow.add_conditional_edges(
            "retrieve_guidance",
            lambda state: "sufficient" if state.get("evidence_sufficient", False) else "insufficient",
            {"sufficient": "generate_grounded_answer", "insufficient": "refuse"},
        )
        workflow.add_edge("clarify_or_fail", END)
        workflow.add_edge("refuse", END)
        workflow.add_edge("generate_grounded_answer", END)
        self.graph = workflow.compile()

    @staticmethod
    def _timings(state: NoticeLensState, **updates: float) -> dict[str, float]:
        result = dict(state.get("timings", {}))
        result.update(updates)
        return result

    def _identify_notice_node(self, state: NoticeLensState) -> dict[str, Any]:
        started = time.perf_counter()
        identity = identify_notice(
            state["notice_first_page"],
            available_guidance_codes=self.available_guidance_codes,
        )
        fields = extract_notice_fields(state["notice_text"], identity)
        return {
            "notice_identity": identity,
            "notice_fields": fields,
            "policy_refusal_reason": classify_unsupported_question(state["question"]),
            "status": identity.status,
            "timings": self._timings(state, identity_and_fields_seconds=time.perf_counter() - started),
        }

    def _clarify_or_fail_node(self, state: NoticeLensState) -> dict[str, Any]:
        identity = state["notice_identity"]
        fields = state["notice_fields"]
        answer = (
            "I couldn't identify an IRS notice code near the notice header."
            if identity.status == "unidentified"
            else "I found multiple plausible IRS notice codes near the header. Please provide a clearer notice."
        )
        response = GroundedResponse(
            status=identity.status,
            notice_code=identity.notice_code,
            answer=answer,
            claims=[],
            citations=[],
            notice_fields=ResponseNoticeFields.from_notice_fields(fields),
        )
        return {"answer": response, "status": response.status}

    def _retrieve_guidance_node(self, state: NoticeLensState) -> dict[str, Any]:
        identity = state["notice_identity"]
        if state.get("policy_refusal_reason") is not None or identity.retrieval_notice_code is None:
            documents: list[Any] = []
            embedding_seconds = 0.0
            pinecone_seconds = 0.0
        else:
            result: RetrievalResult = self.retriever.retrieve(
                state["question"],
                notice_code=identity.retrieval_notice_code,
            )
            documents = result.documents
            embedding_seconds = result.embedding_seconds
            pinecone_seconds = result.pinecone_seconds
        sufficient = (
            state.get("policy_refusal_reason") is None
            and identity.retrieval_notice_code is not None
            and evidence_is_sufficient(documents, notice_code=identity.retrieval_notice_code)
        )
        return {
            "retrieved_documents": documents,
            "evidence_sufficient": sufficient,
            "timings": self._timings(
                state,
                embedding_seconds=embedding_seconds,
                pinecone_seconds=pinecone_seconds,
            ),
        }

    def _refuse_node(self, state: NoticeLensState) -> dict[str, Any]:
        response = refusal_response(
            notice_code=state["notice_identity"].notice_code,
            fields=state["notice_fields"],
        )
        return {"answer": response, "status": response.status}

    def _generate_grounded_answer_node(self, state: NoticeLensState) -> dict[str, Any]:
        permitted_fields = relevant_notice_field_names(state["question"])
        notice_context = select_relevant_notice_context(
            state["notice_text"],
            state["question"],
            state["notice_fields"],
        )
        draft, generation_seconds = self.generator.generate_draft(
            question=state["question"],
            notice_context=notice_context,
            notice_fields=state["notice_fields"],
            permitted_notice_field_names=permitted_fields,
            documents=state["retrieved_documents"],
        )
        started_validation = time.perf_counter()
        response = build_grounded_response(
            draft=draft,
            public_notice_code=state["notice_identity"].notice_code or "",
            fields=state["notice_fields"],
            documents=state["retrieved_documents"],
            permitted_notice_field_names=permitted_fields,
        )
        return {
            "answer": response,
            "status": response.status,
            "timings": self._timings(
                state,
                generation_seconds=generation_seconds,
                validation_and_render_seconds=time.perf_counter() - started_validation,
            ),
        }

    def run_extracted(self, extracted: ExtractedNotice, question: str) -> CoreRun:
        if not isinstance(question, str) or not question.strip() or len(question) > 4_000:
            raise Phase5GateError("The question must be a nonempty string of at most 4,000 characters")
        started = time.perf_counter()
        final_state = self.graph.invoke(
            {
                "notice_text": extracted.text,
                "notice_first_page": extracted.pages[0],
                "question": question,
                "retrieved_documents": [],
                "evidence_sufficient": False,
                "timings": {},
            }
        )
        timings = dict(final_state.get("timings", {}))
        timings["graph_end_to_end_seconds"] = time.perf_counter() - started
        return CoreRun(
            response=final_state["answer"],
            timings=timings,
            identity=final_state["notice_identity"],
            fields=final_state["notice_fields"],
            documents=list(final_state.get("retrieved_documents", [])),
            policy_refusal_reason=final_state.get("policy_refusal_reason"),
        )

    def run_pdf(self, path: Path, question: str) -> CoreRun:
        started = time.perf_counter()
        extract_started = time.perf_counter()
        extracted = extract_pdf_text(path)
        extraction_seconds = time.perf_counter() - extract_started
        run = self.run_extracted(extracted, question)
        timings = dict(run.timings)
        timings["pdf_extraction_seconds"] = extraction_seconds
        timings["request_end_to_end_seconds"] = time.perf_counter() - started
        return CoreRun(
            response=run.response,
            timings=timings,
            identity=run.identity,
            fields=run.fields,
            documents=run.documents,
            policy_refusal_reason=run.policy_refusal_reason,
        )


def create_live_core(
    *,
    project_root: Path,
    secrets: Phase5Secrets,
) -> tuple[NoticeLensCore, ModelSelection]:
    """Freeze-gate first, then perform catalog/probe and read-only retrieval setup."""

    verify_phase5_frozen_inputs(project_root)
    generator = NebiusGroundedGenerator(api_key=secrets.nebius_api_key)
    selection = generator.select_live_model()
    retriever = FinalHeadingRetriever(
        project_root=project_root,
        nebius_api_key=secrets.nebius_api_key,
        pinecone_api_key=secrets.pinecone_api_key,
    )
    return NoticeLensCore(project_root=project_root, retriever=retriever, generator=generator), selection
