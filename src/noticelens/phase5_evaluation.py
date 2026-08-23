"""Frozen Phase 5 evaluation, latency protocol, and report writers."""

from __future__ import annotations

import json
import os
import platform
import re
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

from .chunking import atomic_write_json, atomic_write_text, nearest_rank_percentile, sha256_file
from .final_retrieval import (
    EMBEDDING_DIMENSION,
    EMBEDDING_MODEL,
    INDEX_NAME,
    PRODUCTION_NAMESPACE,
    TOP_K,
)
from .grounded_generation import (
    INSUFFICIENT_ANSWER,
    ModelSelection,
)
from .heading_chunking import HEADING_CHUNK_STRATEGY
from .notice_input import extract_pdf_text, extract_notice_fields, identify_notice
from .phase5 import (
    CoreRun,
    Phase5GateError,
    Phase5Secrets,
    create_live_core,
    verify_phase5_frozen_inputs,
)


GENERATION_REPORT_PATH = Path("reports/phase5_generation_eval.json")
REFUSAL_REPORT_PATH = Path("reports/phase5_refusal_eval.json")
LATENCY_REPORT_PATH = Path("reports/phase5_latency.json")
FINAL_REPORT_PATH = Path("reports/phase5_final_rag_report.md")
FINAL_CONFIG_PATH = Path("reports/final_retrieval_config.json")
GOLDEN_PATH = Path("eval/golden_questions.json")
EXPECTED_GOLDEN_SHA256 = "a5e12ae768b8d43250fac99198efba35bd2d7f5db640e3aba1e8d6958920f391"
FAITHFULNESS_TARGET = 0.95
REFUSAL_TARGET = 1.0
WARM_P95_TARGET_SECONDS = 6.0

# Frozen before generation: selected solely because a corresponding official
# sample PDF exists locally for the question's expected notice. No answer or
# generation result influenced membership.
FAITHFULNESS_SAMPLE_MAP = {
    "A02": "cp05a_english.pdf",
    "A05": "cp523h_english.pdf",
    "A06": "lt11_english.pdf",
    "B04": "cp44_english.pdf",
    "C01": "cp503c.pdf",
    "C02": "lt11_english.pdf",
    "C05": "cp523h_english.pdf",
    "D01": "cp501_english.pdf",
    "D03": "cp523h_english.pdf",
    "D04": "cp59_english.pdf",
}
REFUSAL_SAMPLE_MAP = {
    "E01": "cp503c.pdf",
    "E02": "cp501_english.pdf",
    "E03": "cp501_english.pdf",
    "E04": "cp501_english.pdf",
}
LATENCY_CASE_IDS = ("A02", "B04", "C01")
COLD_REPETITIONS = 3
WARMUP_REPETITIONS = 2
WARM_MEASURED_REPETITIONS = 20


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _load_golden(project_root: Path) -> dict[str, dict[str, Any]]:
    path = project_root / GOLDEN_PATH
    if sha256_file(path) != EXPECTED_GOLDEN_SHA256:
        raise Phase5GateError("The frozen Phase 2 golden question hash changed")
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or len(rows) != 30:
        raise Phase5GateError("The frozen Phase 2 golden question set is invalid")
    by_id = {str(row.get("id", "")): row for row in rows}
    if len(by_id) != 30:
        raise Phase5GateError("The frozen Phase 2 golden question IDs are invalid")
    return by_id


def _sample_path(project_root: Path, filename: str) -> Path:
    path = project_root / "data/raw/sample_notices" / filename
    if not path.is_file():
        raise Phase5GateError(f"Frozen official sample PDF is missing: {filename}")
    return path


def _round_timings(values: dict[str, float]) -> dict[str, float]:
    return {key: round(float(value), 6) for key, value in values.items()}


def _normalized(value: str) -> str:
    return " ".join(value.split())


def _run_tests(project_root: Path) -> dict[str, Any]:
    environment = dict(os.environ)
    environment.pop("NEBIUS_API_KEY", None)
    environment.pop("PINECONE_API_KEY", None)
    existing = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = str(project_root / "src") + (os.pathsep + existing if existing else "")
    completed = subprocess.run(
        [sys.executable, "-B", "-m", "unittest", "discover", "-s", "tests", "-v"],
        cwd=project_root,
        env=environment,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    combined = completed.stdout + "\n" + completed.stderr
    match = re.search(r"Ran\s+(\d+)\s+tests?", combined)
    count = int(match.group(1)) if match else 0
    if completed.returncode != 0 or not re.search(r"\bOK\b", combined) or count <= 0:
        raise Phase5GateError("The complete offline test suite failed; live evaluation was not started")
    return {"passed": count, "failed": 0, "command": "python -B -m unittest discover -s tests -v"}


def evaluate_notice_inputs(project_root: Path) -> dict[str, Any]:
    import csv

    with (project_root / "data/corpus_manifest.csv").open(encoding="utf-8-sig", newline="") as handle:
        available = {row["notice_code"] for row in csv.DictReader(handle)}
    with (project_root / "data/sample_notice_manifest.csv").open(encoding="utf-8-sig", newline="") as handle:
        samples = list(csv.DictReader(handle))
    results: list[dict[str, Any]] = []
    for sample in samples:
        extracted = extract_pdf_text(_sample_path(project_root, sample["filename"]))
        identity = identify_notice(extracted.pages[0], available_guidance_codes=available)
        fields = extract_notice_fields(extracted.text, identity)
        correct = identity.status == "identified" and identity.notice_code == sample["notice_code"]
        results.append(
            {
                "filename": sample["filename"],
                "expected_notice_code": sample["notice_code"],
                "identified_notice_code": identity.notice_code,
                "retrieval_notice_code": identity.retrieval_notice_code,
                "status": identity.status,
                "confidence": identity.confidence,
                "correct": correct,
                "extraction_method": extracted.extraction_method,
                "page_count": len(extracted.pages),
                "fields": fields.as_dict(),
            }
        )
    correct_count = sum(1 for row in results if row["correct"])
    return {
        "sample_count": len(results),
        "identified_correctly": correct_count,
        "notice_identity_accuracy": correct_count / len(results),
        "ocr_requests": 0,
        "ambiguous": sum(row["status"] == "ambiguous" for row in results),
        "unidentified": sum(row["status"] == "unidentified" for row in results),
        "results": results,
    }


def _claim_support_record(run: CoreRun, question_id: str) -> list[dict[str, Any]]:
    response = run.response
    documents = {str(document.metadata["chunk_id"]): document for document in run.documents}
    citations = {citation.citation_id: citation for citation in response.citations}
    fields = run.fields.as_dict()
    rows: list[dict[str, Any]] = []
    for claim in response.claims:
        evidence: list[dict[str, str]] = []
        if claim.evidence_type == "guidance":
            cited_chunks = [citations[citation_id].chunk_id for citation_id in claim.citation_ids]
            quote = claim.text.removeprefix("IRS guidance says:").strip().rstrip(".")
            deterministic_supported = bool(quote) and all(
                chunk_id in documents and _normalized(quote) in _normalized(documents[chunk_id].page_content)
                for chunk_id in cited_chunks
            )
            for chunk_id in cited_chunks:
                evidence.append(
                    {
                        "evidence_type": "official_irs_guidance",
                        "source_id": chunk_id,
                        "text": documents[chunk_id].page_content,
                    }
                )
        else:
            sources = [str(fields[name]["source_text"] or "") for name in claim.notice_field_names]
            deterministic_supported = bool(sources) and all(source and source.rstrip(".") in claim.text for source in sources)
            for name, source in zip(claim.notice_field_names, sources, strict=True):
                evidence.append(
                    {
                        "evidence_type": "uploaded_notice_field",
                        "source_id": name,
                        "text": source,
                    }
                )
        rows.append(
            {
                "claim_id": f"{question_id}:{claim.claim_id}",
                "claim": claim.text,
                "evidence": evidence,
                "deterministic_exact_source_supported": deterministic_supported,
            }
        )
    return rows


def evaluate_generation(
    *,
    project_root: Path,
    core: Any,
) -> tuple[dict[str, Any], dict[str, CoreRun]]:
    golden = _load_golden(project_root)
    runs: dict[str, CoreRun] = {}
    per_question: list[dict[str, Any]] = []
    all_claims: list[dict[str, Any]] = []
    for question_id, filename in FAITHFULNESS_SAMPLE_MAP.items():
        row = golden[question_id]
        run = core.run_pdf(_sample_path(project_root, filename), row["question"])
        runs[question_id] = run
        claim_rows = _claim_support_record(run, question_id)
        all_claims.extend(claim_rows)
        per_question.append(
            {
                "question_id": question_id,
                "question": row["question"],
                "language_style": row["language_style"],
                "expected_doc_id": row["expected_doc_id"],
                "expected_notice_code": row["expected_notice_code"],
                "expected_answer_facts": row["expected_answer_facts"],
                "sample_filename": filename,
                "detected_notice_code": run.identity.notice_code,
                "retrieval_notice_code": run.identity.retrieval_notice_code,
                "status": run.response.status,
                "response": run.response.model_dump(mode="json"),
                "retrieved_chunk_ids": [str(document.metadata["chunk_id"]) for document in run.documents],
                "claim_evidence_review": claim_rows,
                "timings": _round_timings(run.timings),
            }
        )
    supported = sum(bool(row["deterministic_exact_source_supported"]) for row in all_claims)
    total = len(all_claims)
    answered = sum(run.response.status == "answered" for run in runs.values())
    faithfulness = supported / total if total else 0.0
    judge_supported = 0
    judge_evaluated = 0
    judge_seconds = 0.0
    judge_failures: list[dict[str, str]] = []
    # The model judge is explicitly auxiliary. Batch per question to constrain
    # output size, and never let an auxiliary outage erase the primary evidence
    # ledger or the honest pending-human-review status.
    for question in per_question:
        claim_rows = question["claim_evidence_review"]
        payload = [
            {"claim_id": row["claim_id"], "claim": row["claim"], "evidence": row["evidence"]}
            for row in claim_rows
        ]
        if not payload:
            continue
        try:
            judgment, elapsed = core.generator.judge_claims(claims=payload)
            judge_seconds += elapsed
            expected_ids = [str(row["claim_id"]) for row in payload]
            observed_ids = [item.claim_id for item in judgment.judgments]
            if observed_ids != expected_ids or len(observed_ids) != len(set(observed_ids)):
                raise Phase5GateError("Auxiliary judge IDs did not exactly match the submitted batch")
            judge_by_id = {item.claim_id: item for item in judgment.judgments}
            for claim in claim_rows:
                item = judge_by_id[claim["claim_id"]]
                claim["auxiliary_model_judgment"] = {
                    "status": "available",
                    "supported": item.supported,
                    "explanation": item.explanation,
                }
                judge_evaluated += 1
                judge_supported += int(item.supported)
        except Exception as exc:
            judge_failures.append(
                {"question_id": question["question_id"], "error_type": type(exc).__name__}
            )
            for claim in claim_rows:
                claim["auxiliary_model_judgment"] = {"status": "unavailable"}
    return (
        {
            "schema_version": "1.0",
            "generated_at_utc": _utc_now(),
            "frozen_golden_sha256": EXPECTED_GOLDEN_SHA256,
            "subset_selection": {
                "rule": "all frozen answerable questions whose expected notice has an existing official sample PDF",
                "selected_before_generation": True,
                "question_ids": list(FAITHFULNESS_SAMPLE_MAP),
                "question_count": len(FAITHFULNESS_SAMPLE_MAP),
                "expert_count": sum(golden[qid]["language_style"] == "expert" for qid in FAITHFULNESS_SAMPLE_MAP),
                "naive_count": sum(golden[qid]["language_style"] == "naive" for qid in FAITHFULNESS_SAMPLE_MAP),
            },
            "method": {
                "citation_support_audit": "atomic final claims are app-rendered exact excerpts from cited frozen IRS chunks or deterministic notice-field source text; support is verified by exact normalized containment",
                "expected_answer_facts_included_for_review": True,
                "auxiliary": "same selected structured-output model judges every claim against only its attached evidence",
                "human_evidence_review_completed": False,
                "formal_faithfulness_status": "pending_human_review",
                "human_review_limitation": "the frozen Phase 2 metric requires expected_answer_facts plus human evidence review; citation support and the auxiliary judge are reported separately and are not relabelled as formal faithfulness",
            },
            "results": {
                "answerable_questions": len(runs),
                "answered_questions": answered,
                "answerable_response_rate": answered / len(runs),
                "atomic_claims": total,
                "deterministically_supported_claims": supported,
                "citation_support_rate": faithfulness,
                "formal_faithfulness": None,
                "formal_faithfulness_status": "pending_human_review",
                "frozen_faithfulness_target": FAITHFULNESS_TARGET,
                "frozen_faithfulness_target_met": None,
                "auxiliary_judge_supported_claims": judge_supported,
                "auxiliary_judge_evaluated_claims": judge_evaluated,
                "auxiliary_judge_coverage": judge_evaluated / total if total else 0.0,
                "auxiliary_judge_support_rate": judge_supported / judge_evaluated if judge_evaluated else None,
                "auxiliary_judge_seconds": round(judge_seconds, 6),
                "auxiliary_judge_failures": judge_failures,
            },
            "questions": per_question,
        },
        runs,
    )


def evaluate_refusals(*, project_root: Path, core: Any) -> dict[str, Any]:
    golden = _load_golden(project_root)
    rows: list[dict[str, Any]] = []
    for question_id, filename in REFUSAL_SAMPLE_MAP.items():
        row = golden[question_id]
        run = core.run_pdf(_sample_path(project_root, filename), row["question"])
        provider_calls_skipped = (
            not run.documents
            and run.timings.get("embedding_seconds", 0.0) == 0.0
            and run.timings.get("pinecone_seconds", 0.0) == 0.0
            and run.timings.get("generation_seconds", 0.0) == 0.0
        )
        correct = (
            run.response.status == "refused"
            and run.response.answer == INSUFFICIENT_ANSWER
            and not run.response.claims
            and not run.response.citations
            and run.policy_refusal_reason is not None
            and provider_calls_skipped
        )
        rows.append(
            {
                "question_id": question_id,
                "question": row["question"],
                "sample_filename": filename,
                "detected_notice_code": run.identity.notice_code,
                "retrieval_notice_code": run.identity.retrieval_notice_code,
                "identity_status": run.identity.status,
                "policy_category": run.policy_refusal_reason,
                "status": run.response.status,
                "answer": run.response.answer,
                "claims": [claim.model_dump(mode="json") for claim in run.response.claims],
                "citations": [citation.model_dump(mode="json") for citation in run.response.citations],
                "retrieved_chunk_ids": [str(document.metadata.get("chunk_id", "")) for document in run.documents],
                "provider_calls_skipped": provider_calls_skipped,
                "correct_refusal": correct,
                "timings": _round_timings(run.timings),
            }
        )
    correct_count = sum(row["correct_refusal"] for row in rows)
    return {
        "schema_version": "1.0",
        "generated_at_utc": _utc_now(),
        "frozen_golden_sha256": EXPECTED_GOLDEN_SHA256,
        "question_text_modified": False,
        "context_policy": {
            "rule": "each unchanged refusal question is paired with a valid official sample PDF so refusal is based on categorical support/scope, not missing notice identity",
            "mapping_frozen_before_execution": REFUSAL_SAMPLE_MAP,
        },
        "correct_refusals": correct_count,
        "question_count": len(rows),
        "correct_refusal_rate": correct_count / len(rows),
        "target": REFUSAL_TARGET,
        "target_met": correct_count == len(rows),
        "questions": rows,
    }


def _latency_stats(values: Sequence[float]) -> dict[str, float | int]:
    if not values:
        return {"n": 0, "median_seconds": 0.0, "p95_seconds": 0.0, "max_seconds": 0.0}
    return {
        "n": len(values),
        "median_seconds": round(float(statistics.median(values)), 6),
        "p95_seconds": round(float(nearest_rank_percentile(values, 0.95)), 6),
        "max_seconds": round(float(max(values)), 6),
    }


def _latency_record(case_id: str, run: CoreRun, *, initialization_seconds: float = 0.0) -> dict[str, Any]:
    timing = run.timings
    request = float(timing["request_end_to_end_seconds"])
    return {
        "case_id": case_id,
        "status": run.response.status,
        "initialization_seconds": round(initialization_seconds, 6),
        "pdf_extraction_seconds": round(float(timing.get("pdf_extraction_seconds", 0.0)), 6),
        "identity_and_fields_seconds": round(float(timing.get("identity_and_fields_seconds", 0.0)), 6),
        "embedding_seconds": round(float(timing.get("embedding_seconds", 0.0)), 6),
        "pinecone_seconds": round(float(timing.get("pinecone_seconds", 0.0)), 6),
        "retrieval_seconds": round(
            float(timing.get("embedding_seconds", 0.0)) + float(timing.get("pinecone_seconds", 0.0)), 6
        ),
        "generation_seconds": round(float(timing.get("generation_seconds", 0.0)), 6),
        "validation_and_render_seconds": round(float(timing.get("validation_and_render_seconds", 0.0)), 6),
        "request_end_to_end_seconds": round(request, 6),
        "cold_end_to_end_including_initialization_seconds": round(initialization_seconds + request, 6),
    }


def evaluate_latency(
    *,
    project_root: Path,
    secrets: Phase5Secrets,
    warm_core: Any,
    selected_model: str,
) -> dict[str, Any]:
    golden = _load_golden(project_root)
    cases = {
        question_id: (_sample_path(project_root, FAITHFULNESS_SAMPLE_MAP[question_id]), golden[question_id]["question"])
        for question_id in LATENCY_CASE_IDS
    }
    cold: list[dict[str, Any]] = []
    for question_id in LATENCY_CASE_IDS:
        started = time.perf_counter()
        core, selection = create_live_core(project_root=project_root, secrets=secrets)
        initialization_seconds = time.perf_counter() - started
        if selection.selected_model != selected_model:
            raise Phase5GateError("The live generation model changed during cold latency measurement")
        path, question = cases[question_id]
        run = core.run_pdf(path, question)
        cold.append(_latency_record(question_id, run, initialization_seconds=initialization_seconds))

    warmups: list[dict[str, Any]] = []
    for index in range(WARMUP_REPETITIONS):
        question_id = LATENCY_CASE_IDS[index % len(LATENCY_CASE_IDS)]
        path, question = cases[question_id]
        run = warm_core.run_pdf(path, question)
        warmups.append(_latency_record(question_id, run))

    warm: list[dict[str, Any]] = []
    for index in range(WARM_MEASURED_REPETITIONS):
        question_id = LATENCY_CASE_IDS[index % len(LATENCY_CASE_IDS)]
        path, question = cases[question_id]
        run = warm_core.run_pdf(path, question)
        warm.append(_latency_record(question_id, run))

    def component(rows: list[dict[str, Any]], key: str) -> dict[str, float | int]:
        return _latency_stats([float(row[key]) for row in rows])

    warm_end_to_end = component(warm, "request_end_to_end_seconds")
    cold_answered = sum(row["status"] == "answered" for row in cold)
    warmup_answered = sum(row["status"] == "answered" for row in warmups)
    warm_answered = sum(row["status"] == "answered" for row in warm)
    return {
        "schema_version": "1.0",
        "generated_at_utc": _utc_now(),
        "protocol_frozen_before_execution": {
            "cold_repetitions": COLD_REPETITIONS,
            "cold_policy": "new clients, live catalog/schema probe, frozen hash gate, Pinecone namespace verification, then one request",
            "warmup_repetitions": WARMUP_REPETITIONS,
            "warm_measured_repetitions": WARM_MEASURED_REPETITIONS,
            "warm_policy": "one reused core, sequential concurrency=1, no response cache, three cases cycled",
            "status_policy": (
                "all scheduled responses remain in the latency denominator; the warm target requires both "
                "20/20 answered responses and nearest-rank p95 strictly below 6 seconds"
            ),
            "case_ids": list(LATENCY_CASE_IDS),
            "p95_method": "nearest_rank",
            "target_seconds": WARM_P95_TARGET_SECONDS,
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "generation_model": selected_model,
            "embedding_model": EMBEDDING_MODEL,
        },
        "cold": {
            "runs": cold,
            "answered_count": cold_answered,
            "answered_rate": cold_answered / len(cold),
            "initialization": component(cold, "initialization_seconds"),
            "embedding": component(cold, "embedding_seconds"),
            "pinecone": component(cold, "pinecone_seconds"),
            "retrieval": component(cold, "retrieval_seconds"),
            "generation": component(cold, "generation_seconds"),
            "request_end_to_end": component(cold, "request_end_to_end_seconds"),
            "end_to_end_including_initialization": component(
                cold, "cold_end_to_end_including_initialization_seconds"
            ),
        },
        "warm": {
            "warmup_runs": warmups,
            "warmup_answered_count": warmup_answered,
            "runs": warm,
            "answered_count": warm_answered,
            "answered_rate": warm_answered / len(warm),
            "embedding": component(warm, "embedding_seconds"),
            "pinecone": component(warm, "pinecone_seconds"),
            "retrieval": component(warm, "retrieval_seconds"),
            "generation": component(warm, "generation_seconds"),
            "end_to_end": warm_end_to_end,
        },
        "warm_end_to_end_p95_target_met": (
            warm_answered == WARM_MEASURED_REPETITIONS
            and float(warm_end_to_end["p95_seconds"]) < WARM_P95_TARGET_SECONDS
        ),
    }


def _assert_secret_absence_in_memory(payloads: Sequence[str], secrets: Phase5Secrets) -> None:
    values = (secrets.nebius_api_key, secrets.pinecone_api_key)
    for text in payloads:
        if any(value and value in text for value in values):
            raise Phase5GateError("A generated artifact contained a credential value")


def write_reports(
    *,
    project_root: Path,
    model_selection: ModelSelection,
    frozen_before: dict[str, Any],
    frozen_after: dict[str, Any],
    tests: dict[str, Any],
    input_eval: dict[str, Any],
    generation_eval: dict[str, Any],
    refusal_eval: dict[str, Any],
    latency_eval: dict[str, Any],
    secrets: Phase5Secrets,
) -> list[Path]:
    if frozen_before != frozen_after:
        raise Phase5GateError("A frozen Phase 1-4B artifact changed during Phase 5")
    final_config = {
        "embedding_model": EMBEDDING_MODEL,
        "embedding_dimension": EMBEDDING_DIMENSION,
        "generation_model": model_selection.selected_model,
        "pinecone_index": INDEX_NAME,
        "production_namespace": PRODUCTION_NAMESPACE,
        "chunk_strategy": HEADING_CHUNK_STRATEGY,
        "top_k": TOP_K,
        "metadata_filter": "exact notice_code equality",
        "bm25": False,
        "hybrid_retrieval": False,
        "reranking": False,
        "decision_basis": (
            "Frozen dense notice retrieval already achieved 100% Precision@1/MRR/Hit@5, and the "
            "heading-aware strategy exceeded the precommitted section gain threshold; added retrieval "
            "complexity is not justified by measured evidence."
        ),
    }
    generation_eval["model_selection"] = model_selection.as_dict()
    generation_eval["notice_input_evaluation"] = input_eval
    generation_eval["frozen_inputs_before"] = frozen_before
    generation_eval["frozen_inputs_after"] = frozen_after
    generation_eval["offline_tests"] = tests
    faith = generation_eval["results"]
    warm = latency_eval["warm"]
    auxiliary_rate = faith["auxiliary_judge_support_rate"]
    auxiliary_text = "unavailable" if auxiliary_rate is None else f"{auxiliary_rate:.2%}"
    unanswered_ids = [
        str(question["question_id"])
        for question in generation_eval["questions"]
        if question["status"] != "answered"
    ]
    unanswered_text = ", ".join(unanswered_ids) if unanswered_ids else "none"
    cold = latency_eval["cold"]
    report = f"""# PHASE 5 — FINAL RAG CORE REPORT

1. **Final retrieval configuration:** `{EMBEDDING_MODEL}` ({EMBEDDING_DIMENSION} dimensions), Pinecone index `{INDEX_NAME}`, namespace `{PRODUCTION_NAMESPACE}`, `{HEADING_CHUNK_STRATEGY}`, exact notice-code filter, top {TOP_K}. BM25, hybrid retrieval, and reranking remain disabled because the frozen dense notice benchmark was already perfect and heading-aware section retrieval cleared the precommitted gain threshold.
2. **Nebius generation model:** `{model_selection.selected_model}`. It was present in the live catalog and passed the actual strict answer-schema probe. Selection reason: {model_selection.reason}.
3. **Notice identity behavior:** local header-first deterministic parsing; {input_eval['identified_correctly']}/{input_eval['sample_count']} official samples identified correctly, with ambiguous and missing identities stopped before retrieval. Reviewed aliases route variants such as CP503C→CP503 and CP523H→CP523 without generic suffix stripping.
4. **Field extraction behavior:** local PDF text-layer extraction only (OCR requests: {input_eval['ocr_requests']}); code/date/deadline/amount/reference values are emitted only from explicit labels, each with confidence and source text. Missing, conflicting, invalid, relative, or placeholder-only values remain null.
5. **LangGraph nodes/edges:** `START → identify_notice`; unidentified/ambiguous → `clarify_or_fail → END`; identified → `retrieve_guidance`; insufficient/unsupported → `refuse → END`; sufficient → `generate_grounded_answer → END`. No router, per-chunk grader, retry loop, or cycle.
6. **Evidence sufficiency:** deterministic nonempty, same-notice, required-metadata checks with no score threshold. Categorical unsupported requests short-circuit provider calls. The structured model can still return insufficient, which becomes the exact refusal fallback.
7. **Citations:** final claims use application-owned LangChain `Document` metadata. Guidance claims are exact cited IRS excerpts; notice claims are rendered from deterministic field source text. The model cannot supply source URLs or citation metadata.
8. **Faithfulness:** formal frozen-plan faithfulness is **pending human review** and the ≥{FAITHFULNESS_TARGET:.0%} target is not claimed as met. Separately, {faith['deterministically_supported_claims']}/{faith['atomic_claims']} generated atomic claims passed exact-source citation support ({faith['citation_support_rate']:.2%}); the auxiliary structured judge support rate was {auxiliary_text}. Answerable-response coverage was {faith['answered_questions']}/{faith['answerable_questions']}; non-answered question IDs: {unanswered_text}.
9. **Frozen refusal test:** {refusal_eval['correct_refusals']}/{refusal_eval['question_count']} correct ({refusal_eval['correct_refusal_rate']:.2%}); target met: {str(refusal_eval['target_met']).upper()}.
10. **Warm retrieval latency:** median {warm['retrieval']['median_seconds']:.3f}s; p95 {warm['retrieval']['p95_seconds']:.3f}s (embedding and Pinecone reported separately in `phase5_latency.json`).
11. **Warm generation latency:** median {warm['generation']['median_seconds']:.3f}s; p95 {warm['generation']['p95_seconds']:.3f}s.
12. **Warm end-to-end median:** {warm['end_to_end']['median_seconds']:.3f}s.
13. **Warm end-to-end p95:** {warm['end_to_end']['p95_seconds']:.3f}s.
14. **Frozen <6s target:** {str(latency_eval['warm_end_to_end_p95_target_met']).upper()} ({warm['answered_count']}/{warm['end_to_end']['n']} measured warm requests answered; passing requires all requests answered and p95 strictly below 6 seconds).
15. **Tests:** {tests['passed']} passed, {tests['failed']} failed across the complete Phase 1–5 offline suite.
16. **Limitations:** text-layer PDFs only; no OCR fallback was justified by the samples; explicit variant routing is intentionally finite; retrieval is bounded to the frozen IRS corpus; exact-excerpt answers favor faithfulness over stylistic paraphrase; the same model's judge is auxiliary; independent human faithfulness review is still pending; no personalized tax/legal advice. Cold-start end-to-end (including initialization) median was {cold['end_to_end_including_initialization']['median_seconds']:.3f}s and p95 was {cold['end_to_end_including_initialization']['p95_seconds']:.3f}s; {cold['answered_count']}/{cold['end_to_end_including_initialization']['n']} cold requests answered. Cold embedding, Pinecone, retrieval, and generation components are reported separately in `phase5_latency.json`.
17. **Freeze confirmation:** Phase 1–4B approved composite `{frozen_after['approved_phase1_to_4b_composite_sha256']}` remained byte-identical before and after Phase 5.

No UI, Streamlit app, Git initialization, BM25, hybrid retrieval, reranker, Mem0, LlamaIndex, Lyzr, or ElevenLabs component was added.
"""
    paths = [
        project_root / GENERATION_REPORT_PATH,
        project_root / REFUSAL_REPORT_PATH,
        project_root / LATENCY_REPORT_PATH,
        project_root / FINAL_CONFIG_PATH,
        project_root / FINAL_REPORT_PATH,
    ]
    serialized = [
        json.dumps(generation_eval, ensure_ascii=False, indent=2) + "\n",
        json.dumps(refusal_eval, ensure_ascii=False, indent=2) + "\n",
        json.dumps(latency_eval, ensure_ascii=False, indent=2) + "\n",
        json.dumps(final_config, ensure_ascii=False, indent=2) + "\n",
        report,
    ]
    # Scan before the first write so a poisoned payload can never briefly land
    # in the project or OneDrive sync folder.
    _assert_secret_absence_in_memory(serialized, secrets)
    for path, text in zip(paths, serialized, strict=True):
        atomic_write_text(path, text)
    return paths


def run_phase5(project_root: Path, secrets: Phase5Secrets) -> dict[str, Any]:
    frozen_before = verify_phase5_frozen_inputs(project_root)
    tests = _run_tests(project_root)
    input_eval = evaluate_notice_inputs(project_root)
    if input_eval["notice_identity_accuracy"] != 1.0:
        raise Phase5GateError("Official sample notice identity gate failed")
    core, selection = create_live_core(project_root=project_root, secrets=secrets)
    generation_eval, _ = evaluate_generation(project_root=project_root, core=core)
    refusal_eval = evaluate_refusals(project_root=project_root, core=core)
    latency_eval = evaluate_latency(
        project_root=project_root,
        secrets=secrets,
        warm_core=core,
        selected_model=selection.selected_model,
    )
    frozen_after = verify_phase5_frozen_inputs(project_root)
    paths = write_reports(
        project_root=project_root,
        model_selection=selection,
        frozen_before=frozen_before,
        frozen_after=frozen_after,
        tests=tests,
        input_eval=input_eval,
        generation_eval=generation_eval,
        refusal_eval=refusal_eval,
        latency_eval=latency_eval,
        secrets=secrets,
    )
    return {
        "reports": [str(path.relative_to(project_root)) for path in paths],
        "model_selection": selection.as_dict(),
        "identity": {
            "correct": input_eval["identified_correctly"],
            "n": input_eval["sample_count"],
        },
        "faithfulness": generation_eval["results"],
        "refusal": {
            "correct": refusal_eval["correct_refusals"],
            "n": refusal_eval["question_count"],
        },
        "latency": latency_eval["warm"],
        "latency_target_met": latency_eval["warm_end_to_end_p95_target_met"],
        "tests": tests,
        "frozen_composite": frozen_after["approved_phase1_to_4b_composite_sha256"],
    }
