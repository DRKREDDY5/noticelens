from __future__ import annotations

import hashlib
import json
import sys
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import MagicMock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from noticelens.chunking import ChunkRecord  # noqa: E402
from noticelens.phase4a import (  # noqa: E402
    BENCHMARK_FIELDS,
    EMBEDDING_DIMENSION,
    INDEX_NAME,
    NAMESPACE,
    SECTION_BENCHMARK_FROZEN_SHA256,
    SECTION_BENCHMARK_PATH,
    Phase4GateError,
    ReadOnlyPineconeBaseline,
    SectionScore,
    _aggregate_traces,
    _hardest_key,
    aggregate_section_scores,
    attribute_chunk,
    load_section_questions,
    parse_markdown_sections,
    read_manifest,
    score_section_ranking,
    validate_section_benchmark,
    verify_approved_frozen_artifacts,
)
from noticelens.providers import ProviderGateError  # noqa: E402


class CharacterOffsetTokenizer:
    """Reversible character tokenizer with exact, fully offline offsets."""

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        if add_special_tokens:
            raise AssertionError("Phase 4A source mapping must not add special tokens")
        return [ord(character) for character in text]

    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool = False,
        return_offsets_mapping: bool = False,
    ) -> dict[str, list[object]]:
        token_ids = self.encode(text, add_special_tokens=add_special_tokens)
        result: dict[str, list[object]] = {"input_ids": token_ids}
        if return_offsets_mapping:
            result["offset_mapping"] = [(index, index + 1) for index in range(len(text))]
        return result

    def decode(
        self,
        token_ids: list[int],
        *,
        skip_special_tokens: bool,
        clean_up_tokenization_spaces: bool,
    ) -> str:
        if not skip_special_tokens or clean_up_tokenization_spaces:
            raise AssertionError("Unexpected decode policy")
        return "".join(chr(token_id) for token_id in token_ids)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class FrozenSectionBenchmarkTests(unittest.TestCase):
    def test_frozen_benchmark_schema_distribution_and_ground_truth(self) -> None:
        benchmark_path = ROOT / SECTION_BENCHMARK_PATH
        self.assertEqual(sha256(benchmark_path), SECTION_BENCHMARK_FROZEN_SHA256)

        questions = load_section_questions(benchmark_path)
        self.assertEqual([record["id"] for record in questions], [f"S{i:02d}" for i in range(1, 16)])
        self.assertTrue(all(set(record) == BENCHMARK_FIELDS for record in questions))
        self.assertEqual(Counter(record["language_style"] for record in questions), {"naive": 10, "expert": 5})
        self.assertEqual(len({record["expected_doc_id"] for record in questions}), 15)
        self.assertEqual(len({record["expected_notice_code"] for record in questions}), 15)

        manifest_rows = read_manifest(ROOT / "data" / "corpus_manifest.csv")
        manifest_by_doc = {row["doc_id"]: row for row in manifest_rows}
        expected_families = {
            "balance_collection": 4,
            "installment_agreement": 1,
            "levy_cdp": 1,
            "non_filer": 5,
            "penalty_estimated_tax": 1,
            "underreporter_deficiency": 3,
        }
        self.assertEqual(
            Counter(manifest_by_doc[record["expected_doc_id"]]["notice_family"] for record in questions),
            expected_families,
        )

        # Ground truth is checked independently of the Qwen tokenizer: heading
        # hierarchy and evidence containment are character-offset properties.
        tokenizer = CharacterOffsetTokenizer()
        sections_by_doc = {}
        for doc_id in {record["expected_doc_id"] for record in questions}:
            markdown = (ROOT / "data" / "processed" / "guidance" / f"{doc_id}.md").read_text(
                encoding="utf-8"
            )
            _, sections_by_doc[doc_id] = parse_markdown_sections(markdown, tokenizer)
        audit = validate_section_benchmark(
            questions=questions,
            manifest_rows=manifest_rows,
            sections_by_doc=sections_by_doc,
            project_root=ROOT,
        )
        self.assertEqual(audit["question_count"], 15)
        self.assertEqual(audit["unique_notice_count"], 15)
        self.assertEqual(audit["language_style_counts"], {"naive": 10, "expert": 5})
        self.assertEqual(audit["notice_family_counts"], expected_families)
        self.assertTrue(audit["all_expected_headings_verified"])
        self.assertTrue(audit["all_evidence_excerpts_verified"])

    def test_approved_phase1_through_phase3_hashes_are_still_frozen(self) -> None:
        observed = verify_approved_frozen_artifacts(ROOT)
        self.assertEqual(observed["data/processed/guidance"]["file_count"], 50)
        self.assertEqual(len(observed), 12)


class MapperHierarchyAndOffsetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tokenizer = CharacterOffsetTokenizer()
        self.markdown = (
            "# Page title\n"
            "Source: https://example.test\n"
            "## Parent section\n"
            "parent body\n"
            "#### [Deep label](https://example.test/deep)\n"
            "deep body\n"
            "```md\n"
            "### Not a real heading\n"
            "```\n"
            "### Sibling section\n"
            "sibling body\n"
        )

    def test_literal_h2_plus_stack_skipped_levels_links_fences_and_offsets(self) -> None:
        token_ids, sections = parse_markdown_sections(self.markdown, self.tokenizer)
        self.assertEqual(token_ids, [ord(character) for character in self.markdown])
        self.assertEqual(
            [section.path for section in sections],
            [
                ("Parent section",),
                ("Parent section", "Deep label"),
                ("Parent section", "Sibling section"),
            ],
        )
        self.assertEqual([section.levels for section in sections], [(2,), (2, 4), (2, 3)])
        self.assertEqual(
            sections[1].raw_path,
            ("Parent section", "[Deep label](https://example.test/deep)"),
        )
        self.assertNotIn("Not a real heading", {part for section in sections for part in section.path})

        parent_start = self.markdown.index("## Parent section")
        deep_start = self.markdown.index("#### [Deep label]")
        sibling_start = self.markdown.index("### Sibling section")
        self.assertEqual((sections[0].char_start, sections[0].token_start), (parent_start, parent_start))
        self.assertEqual((sections[1].char_start, sections[1].token_start), (deep_start, deep_start))
        self.assertEqual((sections[2].char_start, sections[2].token_start), (sibling_start, sibling_start))
        self.assertEqual(sections[0].char_end, self.markdown.index("\n#### [Deep label]"))
        self.assertTrue(self.markdown[sections[1].char_start : sections[1].char_end].endswith("```"))
        self.assertEqual(sections[2].char_end, len(self.markdown) - 1)
        self.assertTrue(all(section.token_end == section.char_end for section in sections))

    def test_half_open_chunk_intersections_record_every_applicable_path(self) -> None:
        _, sections = parse_markdown_sections(self.markdown, self.tokenizer)
        start = sections[0].token_start
        end = sections[2].token_start + 3
        chunk = ChunkRecord(
            chunk_id="synthetic",
            text=self.markdown[start:end],
            token_start=start,
            token_end=end,
            token_count=end - start,
            source_token_count=end - start,
            text_sha256="synthetic",
            metadata={},
        )
        attributions = attribute_chunk(chunk, sections)
        self.assertEqual(
            [entry["path"] for entry in attributions],
            [
                ["Parent section"],
                ["Parent section", "Deep label"],
                ["Parent section", "Sibling section"],
            ],
        )
        for entry, section in zip(attributions, sections, strict=True):
            expected_start = max(start, section.token_start)
            expected_end = min(end, section.token_end)
            self.assertEqual(entry["overlap_token_start"], expected_start)
            self.assertEqual(entry["overlap_token_end"], expected_end)
            self.assertEqual(entry["overlap_token_count"], expected_end - expected_start)

        boundary_chunk = ChunkRecord(
            chunk_id="boundary",
            text=self.markdown[start : sections[1].token_start],
            token_start=start,
            token_end=sections[1].token_start,
            token_count=sections[1].token_start - start,
            source_token_count=sections[1].token_start - start,
            text_sha256="boundary",
            metadata={},
        )
        self.assertEqual(
            [entry["path"] for entry in attribute_chunk(boundary_chunk, sections)],
            [["Parent section"]],
        )

    def test_frozen_map_reconciles_registry_and_preserves_literal_paths(self) -> None:
        section_map = json.loads(
            (ROOT / "reports" / "phase4_fixed_chunk_section_map.json").read_text(encoding="utf-8")
        )
        registry = [
            json.loads(line)
            for line in (ROOT / "data" / "derived" / "phase3" / "fixed_220_40_chunks.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        ]
        records = section_map["chunks"]
        self.assertEqual(len(records), len(registry), 350)
        self.assertEqual([record["chunk_id"] for record in records], [record["chunk_id"] for record in registry])

        distribution = Counter()
        association_count = 0
        linked_heading_seen = False
        skipped_level_seen = False
        for mapped, chunk in zip(records, registry, strict=True):
            metadata = chunk["metadata"]
            self.assertEqual(mapped["doc_id"], metadata["doc_id"])
            self.assertEqual(mapped["notice_code"], metadata["notice_code"])
            self.assertEqual(mapped["token_start"], chunk["token_start"])
            self.assertEqual(mapped["token_end"], chunk["token_end"])
            paths = mapped["heading_paths"]
            self.assertTrue(paths)
            self.assertEqual(mapped["spans_multiple_heading_paths"], len(paths) > 1)
            distribution[len(paths)] += 1
            association_count += len(paths)
            for attribution in paths:
                self.assertEqual(len(attribution["path"]), len(attribution["levels"]))
                self.assertEqual(len(attribution["path"]), len(attribution["raw_path"]))
                self.assertTrue(all(left < right for left, right in zip(attribution["levels"], attribution["levels"][1:])))
                self.assertNotIn(1, attribution["levels"])
                self.assertGreater(attribution["overlap_token_count"], 0)
                self.assertEqual(
                    attribution["overlap_token_count"],
                    attribution["overlap_token_end"] - attribution["overlap_token_start"],
                )
                self.assertGreaterEqual(attribution["overlap_token_start"], mapped["token_start"])
                self.assertLessEqual(attribution["overlap_token_end"], mapped["token_end"])
                linked_heading_seen |= attribution["path"] != attribution["raw_path"]
                skipped_level_seen |= any(
                    right - left > 1 for left, right in zip(attribution["levels"], attribution["levels"][1:])
                )

        self.assertTrue(linked_heading_seen)
        self.assertTrue(skipped_level_seen)
        self.assertEqual(distribution, {1: 48, 2: 99, 3: 97, 4: 68, 5: 26, 6: 9, 7: 2, 8: 1})
        self.assertEqual(association_count, 1015)
        self.assertEqual(
            section_map["summary"],
            {
                "document_count": 50,
                "chunk_count": 350,
                "unattributed_chunk_count": 0,
                "single_heading_path_chunk_count": 48,
                "multi_heading_path_chunk_count": 302,
                "total_chunk_heading_path_associations": 1015,
                "heading_path_count_distribution": {
                    "1": 48,
                    "2": 99,
                    "3": 97,
                    "4": 68,
                    "5": 26,
                    "6": 9,
                    "7": 2,
                    "8": 1,
                },
                "parsed_h2_h6_section_count": 595,
                "terminal_heading_level_counts": {"H2": 256, "H3": 230, "H4": 109},
            },
        )
        self.assertTrue(section_map["annotation_only"])
        self.assertTrue(section_map["quality_gate_passed"])
        self.assertEqual(section_map["unattributed_chunk_ids"], [])


class ReadOnlyPineconeTests(unittest.TestCase):
    @staticmethod
    def compatible_client() -> tuple[MagicMock, MagicMock]:
        client = MagicMock()
        index = MagicMock()
        client.indexes.exists.return_value = True
        client.indexes.describe.return_value = {
            "dimension": EMBEDDING_DIMENSION,
            "metric": "cosine",
            "vector_type": "dense",
            "host": "offline-host",
            "status": {"ready": True},
            "spec": {"serverless": {"cloud": "aws", "region": "us-east-1"}},
        }
        client.index.return_value = index
        ids = [f"chunk-{number}" for number in range(5)]
        index.describe_index_stats.return_value = {"namespaces": {NAMESPACE: {"vector_count": 5}}}
        index.list.return_value = [[{"id": chunk_id} for chunk_id in ids]]
        index.query.return_value = {
            "matches": [
                {"id": chunk_id, "score": 0.9 - number / 100}
                for number, chunk_id in enumerate(ids)
            ]
        }
        return client, index

    def test_exact_notice_filter_and_namespace_snapshots_make_no_mutations(self) -> None:
        secret = "PINECONE_PHASE4_SENTINEL"
        client, index = self.compatible_client()
        store = ReadOnlyPineconeBaseline(api_key=secret, client=client)
        state = store.require_existing_index()
        before = store.namespace_snapshot()
        vector = [0.0] * EMBEDDING_DIMENSION
        matches = store.query_known_notice(vector, notice_code="CP14", eligible_chunk_count=5)
        after = store.namespace_snapshot()

        self.assertEqual((state.dimension, state.metric, state.vector_type), (4096, "cosine", "dense"))
        self.assertEqual(before, after)
        self.assertEqual(before, (5, {f"chunk-{number}" for number in range(5)}))
        self.assertEqual(len(matches), 5)
        query_kwargs = index.query.call_args.kwargs
        self.assertEqual(query_kwargs["namespace"], NAMESPACE)
        self.assertEqual(query_kwargs["top_k"], 5)
        self.assertEqual(query_kwargs["filter"], {"notice_code": {"$eq": "CP14"}})
        self.assertEqual(set(query_kwargs["filter"]), {"notice_code"})
        self.assertTrue(query_kwargs["include_metadata"])
        self.assertFalse(query_kwargs["include_values"])
        self.assertEqual(len(query_kwargs["vector"]), EMBEDDING_DIMENSION)

        client.indexes.create.assert_not_called()
        client.indexes.configure.assert_not_called()
        client.indexes.delete.assert_not_called()
        index.upsert.assert_not_called()
        index.update.assert_not_called()
        index.delete.assert_not_called()
        self.assertNotIn(secret, repr(store))

    def test_missing_index_stops_without_creating_or_opening_one(self) -> None:
        client = MagicMock()
        client.indexes.exists.return_value = False
        store = ReadOnlyPineconeBaseline(api_key="offline-sentinel", client=client)
        with self.assertRaises(ProviderGateError):
            store.require_existing_index()
        client.indexes.describe.assert_not_called()
        client.indexes.create.assert_not_called()
        client.index.assert_not_called()

    def test_wrong_phase3_index_location_stops_without_opening_or_mutating(self) -> None:
        client, index = self.compatible_client()
        client.indexes.describe.return_value["spec"]["serverless"] = {
            "cloud": "gcp",
            "region": "europe-west4",
        }
        store = ReadOnlyPineconeBaseline(api_key="offline-sentinel", client=client)
        with self.assertRaises(ProviderGateError):
            store.require_existing_index()
        client.index.assert_not_called()
        client.indexes.create.assert_not_called()
        index.upsert.assert_not_called()

    def test_filtered_query_rejects_non_descending_or_wrong_sized_results(self) -> None:
        client, index = self.compatible_client()
        store = ReadOnlyPineconeBaseline(api_key="offline-sentinel", client=client)
        store.require_existing_index()
        vector = [0.0] * EMBEDDING_DIMENSION

        index.query.return_value = {
            "matches": [
                {"id": f"chunk-{number}", "score": score}
                for number, score in enumerate((0.9, 0.8, 0.81, 0.7, 0.6))
            ]
        }
        with self.assertRaises(ProviderGateError):
            store.query_known_notice(vector, notice_code="CP14", eligible_chunk_count=5)

        index.query.return_value = {"matches": [{"id": "only-one", "score": 0.9}]}
        with self.assertRaises(ProviderGateError):
            store.query_known_notice(vector, notice_code="CP14", eligible_chunk_count=5)


class SectionScoringTests(unittest.TestCase):
    @staticmethod
    def ranked_item(notice_code: str, *paths: list[str]) -> dict[str, object]:
        return {
            "retrieved_notice_code": notice_code,
            "attributed_heading_paths": list(paths),
        }

    def test_section_correctness_requires_exact_notice_and_full_normalized_path(self) -> None:
        expected_path = ["Parent section", "Caf\u00e9 options"]
        ranked = [
            self.ranked_item("CP999", ["Parent section", "Caf\u00e9 options"]),
            self.ranked_item("CP14", ["Caf\u00e9 options"]),
            self.ranked_item("CP14", ["  Parent   section ", "Cafe\u0301 options"]),
            self.ranked_item("CP14", ["Parent section", "Other options"]),
            self.ranked_item("CP14", ["Parent section"]),
        ]
        score = score_section_ranking("CP14", expected_path, ranked)
        self.assertEqual(score, SectionScore(0, 1 / 3, 1, 3))

        sixth_only = [
            self.ranked_item("CP14", ["Parent section", f"miss-{number}"])
            for number in range(5)
        ] + [self.ranked_item("CP14", expected_path)]
        self.assertEqual(score_section_ranking("CP14", expected_path, sixth_only), SectionScore(0, 0.0, 0, None))

    def test_section_aggregate_formulas(self) -> None:
        scores = [
            SectionScore(1, 1.0, 1, 1),
            SectionScore(0, 0.5, 1, 2),
            SectionScore(0, 0.2, 1, 5),
            SectionScore(0, 0.0, 0, None),
        ]
        result = aggregate_section_scores(scores)
        self.assertEqual(result["n"], 4)
        self.assertEqual(result["correct_at_1"], 1)
        self.assertAlmostEqual(float(result["section_precision_at_1"]), 0.25)
        self.assertAlmostEqual(float(result["reciprocal_rank_sum"]), 1.7)
        self.assertAlmostEqual(float(result["section_mrr"]), 0.425)
        self.assertEqual(result["hit_at_5_count"], 3)
        self.assertAlmostEqual(float(result["section_hit_at_5"]), 0.75)
        with self.assertRaises(Phase4GateError):
            aggregate_section_scores([])

    def test_trace_aggregates_use_frozen_style_and_family_denominators(self) -> None:
        traces = []
        for number in range(15):
            if number < 5:
                score = {"precision_at_1": 1, "reciprocal_rank": 1.0, "hit_at_5": 1, "first_correct_rank": 1}
            elif number < 10:
                score = {"precision_at_1": 0, "reciprocal_rank": 0.5, "hit_at_5": 1, "first_correct_rank": 2}
            else:
                score = {"precision_at_1": 0, "reciprocal_rank": 0.0, "hit_at_5": 0, "first_correct_rank": None}
            traces.append(
                {
                    "id": f"S{number + 1:02d}",
                    "language_style": "naive" if number < 10 else "expert",
                    "notice_family": "family_a" if number < 8 else "family_b",
                    "section_score": score,
                }
            )

        result = _aggregate_traces(traces)
        self.assertEqual(result["overall"]["n"], 15)
        self.assertAlmostEqual(float(result["overall"]["section_precision_at_1"]), 1 / 3)
        self.assertAlmostEqual(float(result["overall"]["section_mrr"]), 0.5)
        self.assertAlmostEqual(float(result["overall"]["section_hit_at_5"]), 2 / 3)
        self.assertEqual(result["by_language_style"]["naive"]["n"], 10)
        self.assertAlmostEqual(float(result["by_language_style"]["naive"]["section_mrr"]), 0.75)
        self.assertEqual(result["by_language_style"]["expert"]["n"], 5)
        self.assertEqual(result["by_language_style"]["expert"]["section_hit_at_5"], 0.0)
        self.assertEqual(result["by_notice_family"]["family_a"]["n"], 8)
        self.assertEqual(result["by_notice_family"]["family_b"]["n"], 7)

    def test_predeclared_hardest_order(self) -> None:
        def trace(question_id: str, hit: int, rank: int | None, margin: float | None) -> dict[str, object]:
            return {
                "id": question_id,
                "section_score": {"hit_at_5": hit, "first_correct_rank": rank},
                "correct_vs_incorrect_similarity_margin": margin,
            }

        traces = [
            trace("S04", 1, 2, -0.2),
            trace("S03", 0, None, None),
            trace("S02", 1, 5, 0.2),
            trace("S01", 1, 5, 0.1),
        ]
        self.assertEqual(
            [item["id"] for item in sorted(traces, key=_hardest_key)],
            ["S03", "S01", "S02", "S04"],
        )


if __name__ == "__main__":
    unittest.main()
