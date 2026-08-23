from __future__ import annotations

import hashlib
import json
import math
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from noticelens.heading_chunking import HeadingChunkRecord, load_heading_registry  # noqa: E402
from noticelens.phase4a import aggregate_section_scores, load_section_questions  # noqa: E402
from noticelens.retrieval_ablation import (  # noqa: E402
    BENCHMARK_PATH,
    BM25_B,
    BM25_K1,
    DENSE_RESULTS_PATH,
    EXPECTED_BENCHMARK_SHA256,
    EXPECTED_DENSE_RESULTS_SHA256,
    EXPECTED_FINAL_CONFIG_SHA256,
    EXPECTED_REGISTRY_SHA256,
    FINAL_CONFIG_PATH,
    LATENCY_REASONABLE_P95_SECONDS,
    REGISTRY_PATH,
    RERANK_MODEL,
    RRF_K,
    TOP_K,
    VARIANTS,
    AblationGateError,
    BM25Corpus,
    PineconeHostedReranker,
    _decide,
    _load_dense_traces,
    _score_ids,
    _validate_frozen_inputs,
    reciprocal_rank_fusion,
    run_retrieval_ablation,
    sha256_file,
    tokenize_bm25,
)


def heading_chunk(
    chunk_id: str,
    text: str,
    *,
    notice_code: str = "CP1",
    heading_path: Sequence[str] = ("Section",),
) -> HeadingChunkRecord:
    text_digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    tokens = tokenize_bm25(text)
    return HeadingChunkRecord(
        chunk_id=chunk_id,
        text=text,
        content_text=text,
        text_sha256=text_digest,
        content_sha256=text_digest,
        content_token_start=0,
        content_token_end=len(tokens),
        content_token_count=len(tokens),
        embedding_token_count=len(tokens),
        section_content_token_count=len(tokens),
        section_subchunk_count=1,
        source_heading_line=1,
        metadata={
            "chunk_id": chunk_id,
            "doc_id": f"doc-{notice_code}",
            "notice_code": notice_code,
            "heading": heading_path[-1],
            "heading_path": list(heading_path),
        },
    )


class FakeCatalog:
    def __init__(self, names: Sequence[str]) -> None:
        self._names = list(names)

    def names(self) -> list[str]:
        return list(self._names)


class FakeInference:
    def __init__(
        self,
        *,
        catalog_names: Sequence[str] = (RERANK_MODEL, "other-reranker"),
        rerank_data: Sequence[Any] | None = None,
        error_stage: str | None = None,
    ) -> None:
        self.catalog_names = list(catalog_names)
        self.rerank_data = list(rerank_data) if rerank_data is not None else None
        self.error_stage = error_stage
        self.catalog_types: list[str] = []
        self.rerank_calls: list[dict[str, Any]] = []

    def list_models(self, *, type: str) -> FakeCatalog:
        self.catalog_types.append(type)
        if self.error_stage == "catalog":
            raise RuntimeError("TOP_SECRET_VALUE")
        return FakeCatalog(self.catalog_names)

    def rerank(self, **kwargs: Any) -> Any:
        self.rerank_calls.append(kwargs)
        if self.error_stage == "rerank":
            raise RuntimeError("TOP_SECRET_VALUE")
        data = self.rerank_data
        if data is None:
            data = [SimpleNamespace(index=index, score=1.0 - index / 10.0) for index in range(TOP_K)]
        return SimpleNamespace(
            data=data,
            model=RERANK_MODEL,
            usage=SimpleNamespace(rerank_units=2),
        )


class FakePineconeClient:
    def __init__(self, inference: FakeInference) -> None:
        self.inference = inference


class BM25TokenizerAndScoringTests(unittest.TestCase):
    def test_tokenizer_is_nfkc_casefolded_unicode_alphanumeric_only(self) -> None:
        self.assertEqual(
            tokenize_bm25("ＴＡＸ Tax_tax Straße CAFÉ don’t"),
            ["tax", "tax", "tax", "strasse", "café", "don", "t"],
        )
        with self.assertRaisesRegex(TypeError, "must be text"):
            tokenize_bm25(123)  # type: ignore[arg-type]

    def test_bm25_uses_document_frequency_and_the_frozen_okapi_formula(self) -> None:
        records = [
            heading_chunk("a", "tax tax due"),
            heading_chunk("b", "tax paid"),
            heading_chunk("z", "penalty due", notice_code="CP2"),
        ]
        corpus = BM25Corpus.build(records)

        expected_idf = math.log(1.0 + (3 - 2 + 0.5) / (2 + 0.5))
        self.assertAlmostEqual(corpus.idf["tax"], expected_idf)
        expected_average_length = 7 / 3
        self.assertAlmostEqual(corpus.average_length, expected_average_length)
        normalization = BM25_K1 * (
            1.0 - BM25_B + BM25_B * 3 / expected_average_length
        )
        expected_score = expected_idf * (
            2 * (BM25_K1 + 1.0) / (2 + normalization)
        )
        ranked = corpus.rank("tax", "CP1", top_k=2)
        self.assertEqual([chunk_id for chunk_id, _score in ranked], ["a", "b"])
        self.assertAlmostEqual(ranked[0][1], expected_score)

    def test_bm25_ties_are_by_chunk_id_and_notice_filter_is_exact(self) -> None:
        corpus = BM25Corpus.build(
            [
                heading_chunk("b", "same words", notice_code="CP1"),
                heading_chunk("a", "same words", notice_code="CP1"),
                heading_chunk("foreign", "same same same words", notice_code="CP10"),
            ]
        )

        ranked = corpus.rank("same words", "CP1", top_k=TOP_K)

        self.assertEqual([chunk_id for chunk_id, _score in ranked], ["a", "b"])
        self.assertEqual(ranked[0][1], ranked[1][1])
        self.assertNotIn("foreign", {chunk_id for chunk_id, _score in ranked})
        with self.assertRaisesRegex(AblationGateError, "exact-notice candidates"):
            corpus.rank("same", "cp1")


class ReciprocalRankFusionTests(unittest.TestCase):
    def test_rrf_uses_k60_deterministic_ties_and_returns_only_top_five(self) -> None:
        dense = ["b", "a", "c", "d", "e"]
        sparse = ["a", "b", "f", "g", "h"]

        fused = reciprocal_rank_fusion(dense, sparse)

        self.assertEqual(RRF_K, 60)
        self.assertEqual(len(fused), TOP_K)
        self.assertEqual([chunk_id for chunk_id, _score in fused], ["a", "b", "c", "f", "d"])
        self.assertAlmostEqual(fused[0][1], 1 / 61 + 1 / 62)
        self.assertEqual(fused[0][1], fused[1][1])

    def test_rrf_rejects_empty_or_duplicate_input_rankings(self) -> None:
        for dense, sparse in (([], ["a"]), (["a"], []), (["a", "a"], ["b"])):
            with self.subTest(dense=dense, sparse=sparse):
                with self.assertRaisesRegex(AblationGateError, "nonempty unique"):
                    reciprocal_rank_fusion(dense, sparse)


class FullHeadingPathScoringTests(unittest.TestCase):
    def setUp(self) -> None:
        self.records = {
            record.chunk_id: record
            for record in [
                heading_chunk("wrong-parent", "appeal", heading_path=("Other", "Appeal")),
                heading_chunk("exact", "appeal", heading_path=("FAQ", "Appeal")),
                heading_chunk("three", "three", heading_path=("FAQ", "Other 1")),
                heading_chunk("four", "four", heading_path=("FAQ", "Other 2")),
                heading_chunk("five", "five", heading_path=("FAQ", "Other 3")),
            ]
        }
        self.question = {
            "expected_notice_code": "CP1",
            "expected_heading_path": ["FAQ", "Appeal"],
        }

    def test_scorer_requires_the_complete_normalized_heading_path(self) -> None:
        ranked = [
            ("wrong-parent", 0.9),
            ("exact", 0.8),
            ("three", 0.7),
            ("four", 0.6),
            ("five", 0.5),
        ]

        score, rank_records = _score_ids(self.question, ranked, self.records)

        self.assertEqual(score.precision_at_1, 0)
        self.assertEqual(score.reciprocal_rank, 0.5)
        self.assertEqual(score.hit_at_5, 1)
        self.assertEqual(score.first_correct_rank, 2)
        self.assertEqual(rank_records[0]["attributed_heading_paths"], [["Other", "Appeal"]])

    def test_scorer_fails_closed_if_any_candidate_escapes_notice_restriction(self) -> None:
        foreign = heading_chunk("foreign", "appeal", notice_code="CP2", heading_path=("FAQ", "Appeal"))
        records = {**self.records, foreign.chunk_id: foreign}
        ranked = [
            ("exact", 0.9),
            ("foreign", 0.8),
            ("three", 0.7),
            ("four", 0.6),
            ("five", 0.5),
        ]

        with self.assertRaisesRegex(AblationGateError, "exact notice-code restriction"):
            _score_ids(self.question, ranked, records)


class HostedRerankerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.records = {
            f"c{index}": heading_chunk(f"c{index}", f"document {index}")
            for index in range(TOP_K)
        }
        self.candidate_ids = list(self.records)

    def test_catalog_and_rerank_contract_order_ties_and_usage(self) -> None:
        inference = FakeInference(
            rerank_data=[
                SimpleNamespace(index=4, score=0.7),
                SimpleNamespace(index=2, score=0.9),
                SimpleNamespace(index=0, score=0.9),
                SimpleNamespace(index=1, score=0.2),
                SimpleNamespace(index=3, score=0.1),
            ]
        )
        reranker = PineconeHostedReranker(
            api_key="test-placeholder",
            client=FakePineconeClient(inference),
        )

        catalog = reranker.require_model()
        ranked, latency = reranker.rerank(
            query="What should I do?",
            candidate_ids=self.candidate_ids,
            records=self.records,
        )

        self.assertEqual(catalog, [RERANK_MODEL, "other-reranker"])
        self.assertEqual(inference.catalog_types, ["rerank"])
        self.assertEqual([chunk_id for chunk_id, _score in ranked], ["c0", "c2", "c4", "c1", "c3"])
        self.assertGreaterEqual(latency, 0.0)
        self.assertEqual(reranker.request_count, 1)
        self.assertEqual(reranker.rerank_units, 2)
        call = inference.rerank_calls[0]
        self.assertEqual(call["model"], RERANK_MODEL)
        self.assertEqual(call["query"], "What should I do?")
        self.assertEqual(call["top_n"], TOP_K)
        self.assertEqual(call["rank_fields"], ["text"])
        self.assertFalse(call["return_documents"])
        self.assertEqual(call["parameters"], {"truncate": "END"})
        self.assertEqual(
            call["documents"],
            [{"id": chunk_id, "text": self.records[chunk_id].text} for chunk_id in self.candidate_ids],
        )

    def test_provider_errors_are_type_only_and_never_echo_exception_text(self) -> None:
        for stage in ("catalog", "rerank"):
            with self.subTest(stage=stage):
                inference = FakeInference(error_stage=stage)
                reranker = PineconeHostedReranker(
                    api_key="test-placeholder",
                    client=FakePineconeClient(inference),
                )
                with self.assertRaises(AblationGateError) as captured:
                    if stage == "catalog":
                        reranker.require_model()
                    else:
                        reranker.rerank(
                            query="query",
                            candidate_ids=self.candidate_ids,
                            records=self.records,
                        )
                message = str(captured.exception)
                self.assertIn("RuntimeError", message)
                self.assertIn("no credentials were logged", message)
                self.assertNotIn("TOP_SECRET_VALUE", message)

    def test_catalog_must_expose_the_precommitted_model(self) -> None:
        inference = FakeInference(catalog_names=["different-model"])
        reranker = PineconeHostedReranker(
            api_key="test-placeholder",
            client=FakePineconeClient(inference),
        )

        with self.assertRaisesRegex(AblationGateError, "precommitted hosted reranker is unavailable"):
            reranker.require_model()


class FrozenArtifactAndDenseParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.records_list = load_heading_registry(ROOT / REGISTRY_PATH)
        cls.records = {record.chunk_id: record for record in cls.records_list}
        cls.questions = load_section_questions(ROOT / BENCHMARK_PATH)

    def test_ablation_input_hashes_and_production_config_are_exactly_frozen(self) -> None:
        expected = {
            REGISTRY_PATH: EXPECTED_REGISTRY_SHA256,
            BENCHMARK_PATH: EXPECTED_BENCHMARK_SHA256,
            DENSE_RESULTS_PATH: EXPECTED_DENSE_RESULTS_SHA256,
            FINAL_CONFIG_PATH: EXPECTED_FINAL_CONFIG_SHA256,
        }
        config_before = (ROOT / FINAL_CONFIG_PATH).read_bytes()

        frozen = _validate_frozen_inputs(ROOT)

        self.assertTrue(frozen)
        for relative_path, digest in expected.items():
            with self.subTest(path=relative_path):
                self.assertEqual(sha256_file(ROOT / relative_path), digest)
        self.assertEqual((ROOT / FINAL_CONFIG_PATH).read_bytes(), config_before)
        config = json.loads(config_before)
        self.assertFalse(config["bm25"])
        self.assertFalse(config["hybrid_retrieval"])
        self.assertFalse(config["reranking"])
        self.assertEqual(config["top_k"], TOP_K)

    def test_frozen_dense_trace_recomputes_the_approved_metrics(self) -> None:
        by_id, payload = _load_dense_traces(
            ROOT / DENSE_RESULTS_PATH,
            self.questions,
            self.records,
        )
        recomputed = []
        for question in self.questions:
            trace = by_id[question["id"]]
            ranked = [
                (str(item["chunk_id"]), float(item["similarity_score"]))
                for item in trace["ranks"]
            ]
            score, _records = _score_ids(question, ranked, self.records)
            recomputed.append(score)
        aggregate = aggregate_section_scores(recomputed)

        self.assertEqual(len(self.records_list), 580)
        self.assertEqual(len(self.questions), 15)
        self.assertEqual(aggregate["correct_at_1"], 14)
        self.assertAlmostEqual(float(aggregate["section_precision_at_1"]), 14 / 15)
        self.assertAlmostEqual(float(aggregate["section_mrr"]), 0.9555555555555555)
        self.assertEqual(aggregate["hit_at_5_count"], 15)
        self.assertEqual(by_id["S06"]["section_score"]["first_correct_rank"], 3)
        self.assertEqual(
            payload["metrics"]["heading_aware_section_retrieval"]["overall"],
            aggregate,
        )

    def test_full_runner_with_fake_reranker_is_offline_and_config_byte_preserving(self) -> None:
        config_before = (ROOT / FINAL_CONFIG_PATH).read_bytes()
        inference = FakeInference()
        reranker = PineconeHostedReranker(
            api_key="test-placeholder",
            client=FakePineconeClient(inference),
        )

        result = run_retrieval_ablation(
            project_root=ROOT,
            reranker=reranker,
            write_reports=False,
        )

        self.assertFalse(result["reports_written"])
        self.assertEqual(reranker.request_count, 15)
        self.assertEqual(len(inference.rerank_calls), 15)
        self.assertEqual(len(result["rows"]), 15)
        self.assertEqual(set(result["aggregates"]), set(VARIANTS))
        self.assertTrue(result["integrity"]["frozen_before_after_equal"])
        self.assertEqual((ROOT / FINAL_CONFIG_PATH).read_bytes(), config_before)
        self.assertFalse((ROOT / FINAL_CONFIG_PATH).with_suffix(".json.part").exists())


def decision_rows(successes: dict[str, set[str]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for number in range(1, 16):
        question_id = f"S{number:02d}"
        row: dict[str, Any] = {"question_id": question_id}
        for variant in VARIANTS:
            correct = int(question_id in successes[variant])
            row[variant] = {"score": {"precision_at_1": correct}}
        rows.append(row)
    return rows


def decision_aggregates(
    successes: dict[str, set[str]],
    *,
    p95: dict[str, float] | None = None,
) -> dict[str, dict[str, Any]]:
    p95 = p95 or {variant: 0.5 for variant in VARIANTS}
    return {
        variant: {
            "metrics": {"section_precision_at_1": len(successes[variant]) / 15},
            "latency": {"p95_seconds": p95[variant]},
        }
        for variant in VARIANTS
    }


class RetrievalDecisionRuleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.all_ids = {f"S{number:02d}" for number in range(1, 16)}
        self.dense_ids = self.all_ids - {"S06"}

    def test_perfect_reranker_clears_five_point_rule_and_is_recommended_only(self) -> None:
        successes = {
            "dense": self.dense_ids,
            "bm25": self.dense_ids,
            "hybrid": self.dense_ids,
            "hybrid_reranker": self.all_ids,
        }

        decision = _decide(decision_aggregates(successes), decision_rows(successes))

        self.assertEqual(decision["selected_recommendation"], "hybrid_reranker")
        self.assertTrue(decision["config_change_recommended"])
        self.assertFalse(decision["production_config_modified"])
        candidate = decision["candidate_decisions"][-1]
        self.assertAlmostEqual(
            candidate["absolute_p1_delta_percentage_points"],
            100 / 15,
            places=6,
        )
        self.assertTrue(candidate["fixes_dense_failure_s06"])
        self.assertEqual(candidate["new_p1_regressions"], [])
        self.assertTrue(candidate["selection_eligible"])

    def test_fix_with_a_new_regression_or_unreasonable_latency_retains_dense(self) -> None:
        regressed_ids = (self.dense_ids - {"S01"}) | {"S06"}
        for reranker_ids, reranker_p95 in (
            (regressed_ids, 0.5),
            (self.all_ids, LATENCY_REASONABLE_P95_SECONDS + 0.001),
        ):
            with self.subTest(reranker_ids=reranker_ids, reranker_p95=reranker_p95):
                successes = {
                    "dense": self.dense_ids,
                    "bm25": self.dense_ids,
                    "hybrid": self.dense_ids,
                    "hybrid_reranker": reranker_ids,
                }
                p95 = {variant: 0.5 for variant in VARIANTS}
                p95["hybrid_reranker"] = reranker_p95

                decision = _decide(
                    decision_aggregates(successes, p95=p95),
                    decision_rows(successes),
                )

                self.assertEqual(decision["selected_recommendation"], "dense")
                self.assertFalse(decision["config_change_recommended"])
                self.assertFalse(decision["production_config_modified"])

    def test_when_multiple_candidates_qualify_the_simpler_variant_wins(self) -> None:
        successes = {
            "dense": self.dense_ids,
            "bm25": self.all_ids,
            "hybrid": self.all_ids,
            "hybrid_reranker": self.all_ids,
        }

        decision = _decide(decision_aggregates(successes), decision_rows(successes))

        self.assertEqual(decision["selected_recommendation"], "bm25")
        self.assertFalse(decision["production_config_modified"])


if __name__ == "__main__":
    unittest.main()
