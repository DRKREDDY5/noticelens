from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from noticelens.phase5_1 import (  # noqa: E402
    EXPECTED_GENERATION_SCHEMA_SHA256,
    EXPECTED_GOLDEN_SHA256,
    EXPECTED_PROMPT_SHA256,
    EXPECTED_RETRIEVAL_CONFIG,
    FAITHFULNESS_TARGET,
    Phase51GateError,
    QUALITY_CASE_IDS,
    REFUSAL_CASE_IDS,
    _build_comparison_rows,
    _recovery_claim_payload,
    _verify_recovery_evaluator_contract,
    retrieval_config_view,
    score_faithfulness,
    select_final_model,
    update_generation_model_config,
    validate_retrieval_config,
    verify_phase51_frozen_inputs,
)


CURRENT_MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"
FASTER_MODEL = "provider/faster-grounded-model"


def claim_row(
    *,
    deterministic: bool = True,
    semantic_label: str | None = "SUPPORTED",
) -> dict[str, object]:
    return {
        "claim_id": "opaque-claim",
        "claim_text": "A factual claim.",
        "cited_evidence": [{"source_id": "chunk-1", "text": "Evidence."}],
        "deterministic_provenance_valid": deterministic,
        "semantic_label": semantic_label,
    }


def final_config(model: str = CURRENT_MODEL) -> dict[str, object]:
    return {**EXPECTED_RETRIEVAL_CONFIG, "generation_model": model}


class ClaimLevelFaithfulnessTests(unittest.TestCase):
    def test_micro_average_uses_every_factual_claim_and_meets_inclusive_target(self) -> None:
        rows = [claim_row() for _ in range(19)] + [claim_row(semantic_label="UNSUPPORTED")]

        result = score_faithfulness(rows)

        self.assertEqual(result["factual_claims"], 20)
        self.assertEqual(result["deterministic_provenance_valid_claims"], 20)
        self.assertEqual(result["evaluator_covered_claims"], 20)
        self.assertEqual(result["formally_supported_claims"], 19)
        self.assertEqual(result["evaluator_coverage"], 1.0)
        self.assertEqual(result["citation_provenance_rate"], 1.0)
        self.assertAlmostEqual(float(result["faithfulness"]), FAITHFULNESS_TARGET)
        self.assertTrue(result["faithfulness_target_met"])

    def test_support_requires_both_deterministic_provenance_and_semantic_support(self) -> None:
        rows = [
            claim_row(deterministic=True, semantic_label="SUPPORTED"),
            claim_row(deterministic=False, semantic_label="SUPPORTED"),
            claim_row(deterministic=True, semantic_label="UNSUPPORTED"),
        ]

        result = score_faithfulness(rows)

        self.assertEqual(result["factual_claims"], 3)
        self.assertEqual(result["deterministic_provenance_valid_claims"], 2)
        self.assertEqual(result["formally_supported_claims"], 1)
        self.assertAlmostEqual(float(result["faithfulness"]), 1 / 3)
        self.assertFalse(result["faithfulness_target_met"])

    def test_incomplete_semantic_coverage_never_produces_a_formal_score(self) -> None:
        rows = [claim_row(), claim_row(semantic_label=None)]

        result = score_faithfulness(rows)

        self.assertEqual(result["evaluator_covered_claims"], 1)
        self.assertEqual(result["evaluator_coverage"], 0.5)
        self.assertIsNone(result["faithfulness"])
        self.assertFalse(result["faithfulness_target_met"])

    def test_zero_claims_cannot_inflate_faithfulness(self) -> None:
        result = score_faithfulness([])

        self.assertEqual(result["factual_claims"], 0)
        self.assertEqual(result["evaluator_coverage"], 0.0)
        self.assertIsNone(result["citation_provenance_rate"])
        self.assertIsNone(result["faithfulness"])
        self.assertFalse(result["faithfulness_target_met"])


class GenerationModelConfigurationTests(unittest.TestCase):
    def test_model_update_changes_only_generation_model(self) -> None:
        before = final_config()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "final_retrieval_config.json"
            path.write_text(json.dumps(before, indent=2) + "\n", encoding="utf-8")

            observed, changed = update_generation_model_config(path, FASTER_MODEL)
            persisted = json.loads(path.read_text(encoding="utf-8"))

        self.assertTrue(changed)
        self.assertEqual(observed, persisted)
        self.assertEqual(persisted["generation_model"], FASTER_MODEL)
        self.assertEqual(retrieval_config_view(persisted), retrieval_config_view(before))
        self.assertEqual(retrieval_config_view(persisted), EXPECTED_RETRIEVAL_CONFIG)
        self.assertEqual(
            {key for key in before if before[key] != persisted[key]},
            {"generation_model"},
        )

    def test_same_model_is_a_byte_preserving_no_op(self) -> None:
        before = final_config()
        original = json.dumps(before, separators=(",", ":")).encode("utf-8")
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "final_retrieval_config.json"
            path.write_bytes(original)

            observed, changed = update_generation_model_config(path, CURRENT_MODEL)

            self.assertFalse(changed)
            self.assertEqual(observed, before)
            self.assertEqual(path.read_bytes(), original)

    def test_retrieval_drift_or_schema_expansion_is_rejected_without_writing(self) -> None:
        invalid_configs = []
        changed_top_k = final_config()
        changed_top_k["top_k"] = 99
        invalid_configs.append(changed_top_k)
        extra_field = final_config()
        extra_field["new_retrieval_setting"] = True
        invalid_configs.append(extra_field)

        for config in invalid_configs:
            with self.subTest(config=config):
                original = (json.dumps(config, indent=2) + "\n").encode("utf-8")
                with tempfile.TemporaryDirectory() as temporary:
                    path = Path(temporary) / "final_retrieval_config.json"
                    path.write_bytes(original)

                    with self.assertRaisesRegex(Phase51GateError, "retrieval configuration"):
                        update_generation_model_config(path, FASTER_MODEL)

                    self.assertEqual(path.read_bytes(), original)

    def test_blank_generation_model_is_rejected_without_mutating_config(self) -> None:
        before = final_config()
        original = (json.dumps(before, indent=2) + "\n").encode("utf-8")
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "final_retrieval_config.json"
            path.write_bytes(original)

            with self.assertRaisesRegex(Phase51GateError, "generation model"):
                update_generation_model_config(path, "   ")

            self.assertEqual(path.read_bytes(), original)


class FinalModelSelectionTests(unittest.TestCase):
    @staticmethod
    def row(
        model_id: str,
        *,
        eligible: bool,
        end_to_end_p95: float,
        structured_rate: float,
        generation_p95: float,
        order: int,
    ) -> dict[str, object]:
        return {
            "model_id": model_id,
            "eligible": eligible,
            "warm_end_to_end_p95_seconds": end_to_end_p95,
            "structured_output_success_rate": structured_rate,
            "warm_generation_p95_seconds": generation_p95,
            "candidate_order": order,
        }

    def test_ineligible_speed_leader_cannot_beat_quality_gated_model(self) -> None:
        rows = [
            self.row(
                "fast-but-unfaithful",
                eligible=False,
                end_to_end_p95=1.0,
                structured_rate=1.0,
                generation_p95=0.8,
                order=0,
            ),
            self.row(
                "grounded-model",
                eligible=True,
                end_to_end_p95=7.0,
                structured_rate=1.0,
                generation_p95=6.0,
                order=1,
            ),
        ]

        self.assertEqual(select_final_model(rows), "grounded-model")

    def test_lowest_warm_end_to_end_p95_precedes_structured_reliability_tiebreak(self) -> None:
        rows = [
            self.row(
                "lower-p95",
                eligible=True,
                end_to_end_p95=4.0,
                structured_rate=0.95,
                generation_p95=3.5,
                order=1,
            ),
            self.row(
                "higher-p95",
                eligible=True,
                end_to_end_p95=4.1,
                structured_rate=1.0,
                generation_p95=3.0,
                order=0,
            ),
        ]

        self.assertEqual(select_final_model(rows), "lower-p95")

    def test_structured_reliability_breaks_equal_latency_tie_and_no_eligible_returns_none(self) -> None:
        rows = [
            self.row(
                "less-reliable",
                eligible=True,
                end_to_end_p95=4.0,
                structured_rate=0.95,
                generation_p95=3.0,
                order=0,
            ),
            self.row(
                "more-reliable",
                eligible=True,
                end_to_end_p95=4.0,
                structured_rate=1.0,
                generation_p95=3.5,
                order=1,
            ),
        ]

        self.assertEqual(select_final_model(rows), "more-reliable")
        self.assertIsNone(select_final_model([{**rows[0], "eligible": False}]))

    def test_material_answer_coverage_degradation_blocks_speed_eligibility(self) -> None:
        probe = {
            "model_id": "fast-but-materially-degraded",
            "role": "latency_candidate",
            "candidate_order": 1,
            "probe_success": True,
            "probe_seconds": 1.0,
        }
        rows = _build_comparison_rows(
            probe_records=[probe],
            quality={
                "fast-but-materially-degraded": {
                    "answered": 3,
                    "structured_successes": 5,
                }
            },
            faithfulness={
                "fast-but-materially-degraded": {
                    "faithfulness": 1.0,
                    "citation_provenance_rate": 1.0,
                    "factual_claims": 6,
                    "formally_supported_claims": 6,
                    "evaluator_coverage": 1.0,
                }
            },
            refusals={"fast-but-materially-degraded": {"correct": 4, "n": 4}},
            warm={
                "fast-but-materially-degraded": {
                    "summary": {
                        "structured_successes": 20,
                        "answered": 20,
                        "generation": {"median_seconds": 1.0, "p95_seconds": 2.0},
                        "end_to_end": {"n": 20, "median_seconds": 2.0, "p95_seconds": 3.0},
                    }
                }
            },
        )

        self.assertFalse(rows[0]["answer_coverage_no_degradation"])
        self.assertFalse(rows[0]["eligible"])


class EvaluatorRecoverySafetyTests(unittest.TestCase):
    def test_recovery_payload_contains_only_blinded_claim_and_its_cited_evidence(self) -> None:
        claim = {
            "blinded_evaluator_id": "J00017",
            "claim_text": "The deadline is stated in the cited passage.",
            "cited_evidence": [
                {
                    "source_id": "chunk-1",
                    "evidence_type": "guidance_chunk",
                    "text": "The cited passage states the deadline.",
                    "ignored_metadata": "must-not-leave-process",
                }
            ],
            "question_id": "hidden",
            "model_id": "hidden",
            "expected_answer_facts": ["hidden"],
        }

        payload = _recovery_claim_payload(claim)

        self.assertEqual(
            payload,
            {
                "claim_id": "J00017",
                "claim": "The deadline is stated in the cited passage.",
                "cited_evidence": [
                    {
                        "source_id": "chunk-1",
                        "evidence_type": "guidance_chunk",
                        "text": "The cited passage states the deadline.",
                    }
                ],
            },
        )

    def test_recovery_rejects_evaluator_contract_drift_before_provider_work(self) -> None:
        from noticelens.phase5_1 import (  # local import keeps the fixture tied to runtime constants
            EVALUATOR_SYSTEM_PROMPT,
            SemanticJudgmentBatch,
            _canonical_json,
            _sha256_text,
        )

        valid = {
            "system_prompt_sha256": _sha256_text(EVALUATOR_SYSTEM_PROMPT),
            "schema_sha256": _sha256_text(_canonical_json(SemanticJudgmentBatch.model_json_schema())),
            "temperature": 0,
            "max_tokens": 3200,
        }
        _verify_recovery_evaluator_contract(valid)
        for key, bad_value in (
            ("system_prompt_sha256", "0" * 64),
            ("schema_sha256", "1" * 64),
            ("temperature", 1),
            ("max_tokens", 1),
        ):
            with self.subTest(key=key):
                changed = {**valid, key: bad_value}
                with self.assertRaisesRegex(Phase51GateError, "evaluator contract"):
                    _verify_recovery_evaluator_contract(changed)


class RetrievalConfigurationPreservationTests(unittest.TestCase):
    def test_current_final_config_matches_the_frozen_retrieval_contract(self) -> None:
        path = ROOT / "reports/final_retrieval_config.json"
        before_bytes = path.read_bytes()
        config = json.loads(before_bytes.decode("utf-8"))

        validate_retrieval_config(config)

        self.assertEqual(retrieval_config_view(config), EXPECTED_RETRIEVAL_CONFIG)
        self.assertEqual(path.read_bytes(), before_bytes)

    def test_prompt_schema_question_maps_and_retrieval_inputs_pass_the_frozen_gate(self) -> None:
        snapshot = verify_phase51_frozen_inputs(ROOT)

        self.assertEqual(
            QUALITY_CASE_IDS,
            ("A02", "A05", "A06", "B04", "C01", "C02", "C05", "D01", "D03", "D04"),
        )
        self.assertEqual(REFUSAL_CASE_IDS, ("E01", "E02", "E03", "E04"))
        self.assertEqual(snapshot["generation_system_prompt_sha256"], EXPECTED_PROMPT_SHA256)
        self.assertEqual(snapshot["generation_schema_sha256"], EXPECTED_GENERATION_SCHEMA_SHA256)
        self.assertEqual(snapshot["golden_questions_sha256"], EXPECTED_GOLDEN_SHA256)
        self.assertEqual(snapshot["retrieval_config"], EXPECTED_RETRIEVAL_CONFIG)


if __name__ == "__main__":
    unittest.main()
