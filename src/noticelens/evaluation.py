"""Frozen Phase 3 evaluation loading, normalization, and scoring."""

from __future__ import annotations

import csv
import json
import math
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


CATEGORY_NAMES = {
    "A": "exact_notice_identifier",
    "B": "everyday_language_semantic",
    "C": "confusable_family",
    "D": "section_level_retrieval",
    "E": "unsupported_refusal",
}
DOCUMENT_METRIC_CATEGORIES = {"A", "B", "C"}
SECTION_CATEGORY = "D"
REFUSAL_CATEGORY = "E"
_NOTICE_PATTERN = re.compile(r"\b(CP|LT|LETTER)\s*[- ]?\s*(\d+)([A-Z]?)\b", re.IGNORECASE)


class EvaluationGateError(RuntimeError):
    """Raised when frozen evaluation data or ranked results are invalid."""


@dataclass(frozen=True)
class RetrievalScore:
    precision_at_1: int
    reciprocal_rank: float
    hit_at_5: int
    first_expected_rank: int | None


def normalize_notice_code(value: str) -> str | None:
    """Normalize one CP/LT/Letter identifier without inventing aliases."""

    normalized = unicodedata.normalize("NFKC", value).upper().strip()
    match = _NOTICE_PATTERN.fullmatch(normalized)
    if not match:
        return None
    prefix, digits, suffix = match.groups()
    return f"{prefix}{digits}{suffix}"


def notice_tokens(value: str) -> list[str]:
    """Extract explicit identifiers from a manifest composite/series value."""

    normalized = unicodedata.normalize("NFKC", value).upper()
    return [f"{prefix}{digits}{suffix}" for prefix, digits, suffix in _NOTICE_PATTERN.findall(normalized)]


def build_notice_alias_registry(
    manifest_rows: list[dict[str, str]],
) -> dict[str, str]:
    """Map only explicit manifest aliases to doc IDs, failing on collisions."""

    registry: dict[str, str] = {}
    for row in manifest_rows:
        doc_id = row.get("doc_id", "").strip()
        notice_code = row.get("notice_code", "").strip()
        if not doc_id or not notice_code:
            raise EvaluationGateError("Manifest rows require doc_id and notice_code")
        tokens = notice_tokens(notice_code)
        if not tokens:
            raise EvaluationGateError(f"No supported notice identifier in manifest row {doc_id}")
        for token in tokens:
            prior = registry.get(token)
            if prior is not None and prior != doc_id:
                raise EvaluationGateError(f"Notice alias collision for {token}: {prior}, {doc_id}")
            registry[token] = doc_id
    return registry


def resolve_notice_alias(value: str, registry: dict[str, str]) -> str | None:
    token = normalize_notice_code(value)
    return None if token is None else registry.get(token)


def read_manifest_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    if not rows:
        raise EvaluationGateError("Corpus manifest is empty")
    return rows


def load_golden_questions(
    golden_path: Path,
    manifest_path: Path,
) -> list[dict[str, Any]]:
    """Load and validate the frozen 30-question top-level JSON array."""

    questions = json.loads(golden_path.read_text(encoding="utf-8"))
    if not isinstance(questions, list) or len(questions) != 30:
        raise EvaluationGateError("Golden questions must be a 30-record top-level array")
    manifest_rows = read_manifest_rows(manifest_path)
    manifest_by_doc = {row["doc_id"]: row for row in manifest_rows}
    if len(manifest_by_doc) != len(manifest_rows):
        raise EvaluationGateError("Manifest doc IDs are not unique")

    required = {
        "id",
        "category",
        "question",
        "language_style",
        "expected_doc_id",
        "expected_notice_code",
        "expected_heading",
        "confusable_with",
        "expected_answer_facts",
        "should_refuse",
        "rationale",
    }
    expected_ids = {
        *(f"A{number:02d}" for number in range(1, 7)),
        *(f"B{number:02d}" for number in range(1, 9)),
        *(f"C{number:02d}" for number in range(1, 9)),
        *(f"D{number:02d}" for number in range(1, 5)),
        *(f"E{number:02d}" for number in range(1, 5)),
    }
    seen_ids: set[str] = set()
    counts: dict[str, int] = defaultdict(int)
    for record in questions:
        if not isinstance(record, dict) or set(record) != required:
            raise EvaluationGateError("Golden question schema differs from the frozen 11-field contract")
        question_id = record["id"]
        if not isinstance(question_id, str) or not re.fullmatch(r"[A-E]\d{2}", question_id):
            raise EvaluationGateError(f"Invalid golden question ID: {question_id!r}")
        if question_id in seen_ids:
            raise EvaluationGateError(f"Duplicate golden question ID: {question_id}")
        seen_ids.add(question_id)
        letter = question_id[0]
        counts[letter] += 1
        if record["category"] != CATEGORY_NAMES[letter]:
            raise EvaluationGateError(f"Category mismatch for {question_id}")
        if record["language_style"] not in {"naive", "expert"}:
            raise EvaluationGateError(f"Invalid language style for {question_id}")
        if not isinstance(record["question"], str) or not record["question"].strip():
            raise EvaluationGateError(f"Blank question for {question_id}")
        if not isinstance(record["confusable_with"], list):
            raise EvaluationGateError(f"Invalid confusable list for {question_id}")
        if any(not isinstance(value, str) or not value.strip() for value in record["confusable_with"]):
            raise EvaluationGateError(f"Invalid confusable notice value for {question_id}")
        if not isinstance(record["rationale"], str) or not record["rationale"].strip():
            raise EvaluationGateError(f"Blank rationale for {question_id}")
        if not isinstance(record["expected_answer_facts"], list):
            raise EvaluationGateError(f"Invalid expected facts for {question_id}")

        if letter == REFUSAL_CATEGORY:
            if record["should_refuse"] is not True:
                raise EvaluationGateError(f"Refusal record {question_id} is not labelled for refusal")
            if any(record[field] is not None for field in ("expected_doc_id", "expected_notice_code", "expected_heading")):
                raise EvaluationGateError(f"Refusal record {question_id} has retrieval ground truth")
            if record["expected_answer_facts"]:
                raise EvaluationGateError(f"Refusal record {question_id} has expected answer facts")
        else:
            if record["should_refuse"] is not False:
                raise EvaluationGateError(f"Answerable record {question_id} is labelled for refusal")
            doc_id = record["expected_doc_id"]
            if doc_id not in manifest_by_doc:
                raise EvaluationGateError(f"Unknown expected doc ID for {question_id}")
            manifest_code = manifest_by_doc[doc_id]["notice_code"]
            if record["expected_notice_code"] != manifest_code:
                raise EvaluationGateError(f"Expected notice code mismatch for {question_id}")
            if not isinstance(record["expected_heading"], str) or not record["expected_heading"].strip():
                raise EvaluationGateError(f"Blank expected heading for {question_id}")
            facts = record["expected_answer_facts"]
            if not 2 <= len(facts) <= 4 or any(not isinstance(fact, str) or not fact.strip() for fact in facts):
                raise EvaluationGateError(f"Invalid expected answer facts for {question_id}")

    if counts != {"A": 6, "B": 8, "C": 8, "D": 4, "E": 4}:
        raise EvaluationGateError(f"Frozen category counts changed: {dict(counts)}")
    if seen_ids != expected_ids:
        raise EvaluationGateError("Frozen golden question ID set changed")
    return questions


def questions_for_retrieval(questions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected = [record for record in questions if record["id"][0] in "ABCD" and not record["should_refuse"]]
    if len(selected) != 26:
        raise EvaluationGateError(f"Expected 26 answerable A-D questions, found {len(selected)}")
    return selected


def score_ranking(expected_doc_id: str, ranked_doc_ids: list[str]) -> RetrievalScore:
    """Score the first five retrieved chunks by source-document identity."""

    top_five = ranked_doc_ids[:5]
    first_rank: int | None = None
    for rank, doc_id in enumerate(top_five, start=1):
        if doc_id == expected_doc_id:
            first_rank = rank
            break
    return RetrievalScore(
        precision_at_1=int(bool(top_five) and top_five[0] == expected_doc_id),
        reciprocal_rank=0.0 if first_rank is None else 1.0 / first_rank,
        hit_at_5=int(first_rank is not None),
        first_expected_rank=first_rank,
    )


def aggregate_scores(scores: list[RetrievalScore]) -> dict[str, int | float]:
    if not scores:
        raise EvaluationGateError("Cannot aggregate an empty retrieval score set")
    n = len(scores)
    correct_at_1 = sum(score.precision_at_1 for score in scores)
    reciprocal_rank_sum = sum(score.reciprocal_rank for score in scores)
    hits = sum(score.hit_at_5 for score in scores)
    return {
        "n": n,
        "correct_at_1": correct_at_1,
        "precision_at_1": correct_at_1 / n,
        "reciprocal_rank_sum": reciprocal_rank_sum,
        "mrr": reciprocal_rank_sum / n,
        "hit_at_5_count": hits,
        "hit_at_5": hits / n,
    }


def validate_embedding_dimension(embedding: Any, expected_dimension: int = 4096) -> int:
    """Validate a single provider embedding response without logging its values."""

    if not isinstance(embedding, (list, tuple)):
        raise EvaluationGateError("Embedding response is not a vector sequence")
    observed = len(embedding)
    if observed != expected_dimension:
        raise EvaluationGateError(
            f"Embedding dimension mismatch: expected {expected_dimension}, observed {observed}"
        )
    for item in embedding:
        if isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(float(item)):
            raise EvaluationGateError("Embedding response contains a non-finite numeric value")
    return observed


def normalize_heading_text(value: str) -> str:
    return " ".join(unicodedata.normalize("NFC", value).split())


def heading_text_is_visible(expected_heading: str, chunk_text: str) -> bool:
    """Literal observation only; this does not attribute a chunk to a heading."""

    normalized_heading = normalize_heading_text(expected_heading)
    return bool(normalized_heading) and normalized_heading in normalize_heading_text(chunk_text)


def group_metric_scores(query_results: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate the frozen A-C metrics and predeclared breakdowns."""

    metric_results = [row for row in query_results if row["question_id"][0] in DOCUMENT_METRIC_CATEGORIES]
    if len(metric_results) != 22:
        raise EvaluationGateError(f"Expected 22 A-C metric records, found {len(metric_results)}")
    expected_ids = {
        *(f"A{number:02d}" for number in range(1, 7)),
        *(f"B{number:02d}" for number in range(1, 9)),
        *(f"C{number:02d}" for number in range(1, 9)),
    }
    observed_ids = [row["question_id"] for row in metric_results]
    if len(set(observed_ids)) != len(observed_ids) or set(observed_ids) != expected_ids:
        raise EvaluationGateError("A-C result IDs are duplicated or incomplete")

    def scores_for(rows: list[dict[str, Any]]) -> list[RetrievalScore]:
        scores: list[RetrievalScore] = []
        for row in rows:
            retrievals = row.get("retrievals")
            if not isinstance(retrievals, list) or len(retrievals) != 5:
                raise EvaluationGateError(f"Result {row['question_id']} must contain exactly five retrievals")
            ranked_doc_ids = [retrieval.get("doc_id") for retrieval in retrievals]
            if any(not isinstance(doc_id, str) or not doc_id for doc_id in ranked_doc_ids):
                raise EvaluationGateError(f"Result {row['question_id']} has invalid retrieved doc IDs")
            score = score_ranking(row["expected_doc_id"], ranked_doc_ids)
            expected_fields = (
                ("precision_at_1", score.precision_at_1),
                ("reciprocal_rank", score.reciprocal_rank),
                ("hit_at_5", score.hit_at_5),
                ("first_expected_doc_rank", score.first_expected_rank),
            )
            for field, expected in expected_fields:
                if row.get(field) != expected:
                    raise EvaluationGateError(
                        f"Result {row['question_id']} has inconsistent precomputed field {field}"
                    )
            scores.append(score)
        return scores

    by_category: dict[str, Any] = {}
    for letter in "ABC":
        rows = [row for row in metric_results if row["question_id"].startswith(letter)]
        by_category[letter] = aggregate_scores(scores_for(rows))
    by_style: dict[str, Any] = {}
    for style in ("naive", "expert"):
        rows = [row for row in metric_results if row["language_style"] == style]
        by_style[style] = aggregate_scores(scores_for(rows))

    if by_category["A"]["n"] != 6 or by_category["B"]["n"] != 8 or by_category["C"]["n"] != 8:
        raise EvaluationGateError("Category metric denominators changed")
    if by_style["naive"]["n"] != 13 or by_style["expert"]["n"] != 9:
        raise EvaluationGateError("Language-style metric denominators changed")

    return {
        "overall": aggregate_scores(scores_for(metric_results)),
        "by_category": by_category,
        "by_language_style": by_style,
    }
