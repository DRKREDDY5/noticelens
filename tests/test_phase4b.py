from __future__ import annotations

import hashlib
import json
import statistics
import sys
import unittest
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from noticelens.chunking import (  # noqa: E402
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    load_chunk_registry,
    nearest_rank_percentile,
    sha256_file,
)
from noticelens.heading_chunking import (  # noqa: E402
    HEADING_CHUNK_STRATEGY,
    HEADING_LOCAL_METADATA_KEYS,
    HeadingChunkRecord,
    HeadingChunkingGateError,
    chunk_section,
    load_heading_registry,
    parse_logical_sections,
    structural_prefix,
)
from noticelens.phase4b import (  # noqa: E402
    EMBEDDING_DIMENSION,
    EXPECTED_HEADING_CHUNK_COUNT,
    EXPECTED_FIXED_VECTOR_COUNT,
    FIXED_METRICS,
    FIXED_REGISTRY_PATH,
    FROZEN_COMPOSITE_SHA256,
    FROZEN_FILE_HASHES,
    FROZEN_TREE_HASHES,
    HEADING_NAMESPACE,
    HEADING_REGISTRY_PATH,
    CHUNK_AUDIT_PATH,
    BASELINE_NAMESPACE,
    HeadingAwarePineconeStore,
    Phase4BGateError,
    SECTION_QUESTIONS_PATH,
    SPECIAL_FIXED_RANKS,
    aggregate_heading_traces,
    build_argument_parser,
    build_paired_comparison,
    evaluate_and_index,
    load_and_validate_fixed_results,
    verify_frozen_artifacts,
)
from noticelens.phase4a import load_section_questions  # noqa: E402
from noticelens.providers import ProviderGateError  # noqa: E402


EXPECTED_REGISTRY_SHA256 = "3aecf5db7ee5fe857bdb99156c9bb5ba585f2e845aa6697cc4ece8902ac27572"
EXPECTED_AUDIT_SHA256 = "88265f578c3f65b9eae8a42b2ab4c52e1e295f87c1f0f679c5169a427869cc4e"
EXPECTED_FROZEN_COMPOSITE_SHA256 = (
    "2561f0d6d9b3f781b6f0a77007c36f7c642c50821ee8495a0cb4087b5e701f9c"
)


class VocabularyTokenizer:
    """Tiny reversible whitespace tokenizer for fully offline boundary tests."""

    def __init__(self) -> None:
        self._token_to_id: dict[str, int] = {}
        self._id_to_token: dict[int, str] = {}

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        if add_special_tokens:
            raise AssertionError("Heading chunking must not add tokenizer special tokens")
        result: list[int] = []
        for token in text.split():
            if token not in self._token_to_id:
                token_id = len(self._token_to_id)
                self._token_to_id[token] = token_id
                self._id_to_token[token_id] = token
            result.append(self._token_to_id[token])
        return result

    def decode(
        self,
        token_ids: list[int],
        *,
        skip_special_tokens: bool,
        clean_up_tokenization_spaces: bool,
    ) -> str:
        if not skip_special_tokens or clean_up_tokenization_spaces:
            raise AssertionError("Unexpected heading-aware decode policy")
        return " ".join(self._id_to_token[token_id] for token_id in token_ids)


def manifest_row(
    *,
    doc_id: str = "irs_cp14",
    notice_code: str = "CP14",
    title: str = "Understanding your CP14 notice",
) -> dict[str, str]:
    return {
        "doc_id": doc_id,
        "notice_code": notice_code,
        "notice_family": "balance_collection",
        "title": title,
        "source_url": "https://www.irs.gov/individuals/understanding-your-cp14-notice",
        "source_origin": "IRS",
    }


def one_heading_chunk() -> HeadingChunkRecord:
    tokenizer = VocabularyTokenizer()
    _title, sections, _inventory = parse_logical_sections(
        "# Extracted page title\n## What you need to do\nPay the amount shown.\n"
    )
    return chunk_section(
        tokenizer=tokenizer,
        manifest_row=manifest_row(),
        document_title="Understanding your CP14 notice",
        section=sections[0],
    )[0]


class FrozenHeadingArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.registry_path = ROOT / HEADING_REGISTRY_PATH
        cls.audit_path = ROOT / CHUNK_AUDIT_PATH
        cls.records = load_heading_registry(cls.registry_path)
        cls.audit = json.loads(cls.audit_path.read_text(encoding="utf-8"))

    def test_registry_and_audit_are_the_exact_580_chunk_freeze(self) -> None:
        self.assertEqual(EXPECTED_HEADING_CHUNK_COUNT, 580)
        self.assertEqual(sha256_file(self.registry_path), EXPECTED_REGISTRY_SHA256)
        self.assertEqual(sha256_file(self.audit_path), EXPECTED_AUDIT_SHA256)
        self.assertEqual(
            self.audit["outputs"],
            {
                "heading_registry_path": HEADING_REGISTRY_PATH.as_posix(),
                "heading_registry_sha256": EXPECTED_REGISTRY_SHA256,
            },
        )
        self.assertEqual(len(self.records), 580)
        self.assertEqual(len({record.chunk_id for record in self.records}), 580)
        self.assertEqual(
            hashlib.sha256(
                "\n".join(sorted(record.chunk_id for record in self.records)).encode("utf-8")
            ).hexdigest(),
            "60bbf276bc752bfaf8f1e3ca8904fc88d688ce07e8c32b4224dc1e667b26912e",
        )

        per_doc = Counter(str(record.metadata["doc_id"]) for record in self.records)
        self.assertEqual(len(per_doc), 50)
        self.assertEqual(dict(per_doc), self.audit["chunks_per_document"]["by_doc_id"])
        self.assertEqual(
            (min(per_doc.values()), statistics.median(per_doc.values()), max(per_doc.values())),
            (6, 11.0, 27),
        )
        self.assertEqual(
            {key: self.audit["chunks_per_document"][key] for key in ("min", "median", "max")},
            {"min": 6, "median": 11.0, "max": 27},
        )

        content_counts = [record.content_token_count for record in self.records]
        embedding_counts = [record.embedding_token_count for record in self.records]
        self.assertEqual(
            (
                min(content_counts),
                statistics.median(content_counts),
                nearest_rank_percentile(content_counts, 0.95),
                max(content_counts),
            ),
            (3, 81.0, 220.0, 220),
        )
        self.assertEqual(
            (
                min(embedding_counts),
                statistics.median(embedding_counts),
                nearest_rank_percentile(embedding_counts, 0.95),
                max(embedding_counts),
            ),
            (34, 112.0, 243.0, 267),
        )
        self.assertEqual(
            self.audit["content_tokens_per_chunk"],
            {"min": 3, "median": 81.0, "p95": 220.0, "max": 220, "p95_method": "nearest_rank"},
        )
        self.assertEqual(
            self.audit["embedding_tokens_per_chunk_including_prefix"],
            {"min": 34, "median": 112.0, "p95": 243.0, "max": 267, "p95_method": "nearest_rank"},
        )

    def test_registry_reconciles_sections_prefixes_ids_metadata_and_overlap(self) -> None:
        groups: dict[tuple[str, int], list[HeadingChunkRecord]] = defaultdict(list)
        for record in self.records:
            metadata = record.metadata
            self.assertEqual(tuple(metadata), HEADING_LOCAL_METADATA_KEYS)
            self.assertEqual(metadata["chunk_id"], record.chunk_id)
            self.assertEqual(metadata["chunk_strategy"], HEADING_CHUNK_STRATEGY)
            self.assertEqual(metadata["heading"], metadata["heading_path"][-1])
            self.assertTrue(metadata["heading_path"])
            self.assertEqual(
                record.text,
                structural_prefix(
                    str(metadata["title"]),
                    str(metadata["notice_code"]),
                    list(metadata["heading_path"]),
                )
                + record.content_text,
            )
            self.assertEqual(hashlib.sha256(record.text.encode("utf-8")).hexdigest(), record.text_sha256)
            self.assertEqual(
                hashlib.sha256(record.content_text.encode("utf-8")).hexdigest(),
                record.content_sha256,
            )
            self.assertEqual(
                record.chunk_id,
                (
                    f"{metadata['doc_id']}__{HEADING_CHUNK_STRATEGY}__"
                    f"s{int(metadata['section_index']):04d}__"
                    f"c{int(metadata['subchunk_index']):04d}__{record.text_sha256}"
                ),
            )
            self.assertLessEqual(record.content_token_count, CHUNK_SIZE)
            self.assertLessEqual(record.content_token_end, record.section_content_token_count)
            groups[(str(metadata["doc_id"]), int(metadata["section_index"]))].append(record)

        self.assertEqual(len(groups), 545)
        oversized_groups = [values for values in groups.values() if len(values) > 1]
        self.assertEqual(len(oversized_groups), 33)
        self.assertEqual(sum(len(values) for values in oversized_groups), 68)
        for key, values in groups.items():
            ordered = sorted(values, key=lambda record: int(record.metadata["subchunk_index"]))
            self.assertEqual(
                [int(record.metadata["subchunk_index"]) for record in ordered],
                list(range(len(ordered))),
            )
            self.assertTrue(all(record.section_subchunk_count == len(ordered) for record in ordered))
            self.assertTrue(all(record.content_token_start >= 0 for record in ordered))
            self.assertEqual(ordered[0].content_token_start, 0)
            self.assertEqual(ordered[-1].content_token_end, ordered[-1].section_content_token_count)
            self.assertEqual(
                {tuple(record.metadata["heading_path"]) for record in ordered},
                {tuple(ordered[0].metadata["heading_path"])},
                key,
            )
            for left, right in zip(ordered, ordered[1:]):
                self.assertEqual(right.content_token_start, left.content_token_end - CHUNK_OVERLAP)
                self.assertEqual(left.metadata["section_index"], right.metadata["section_index"])

    def test_exact_audit_quality_and_documented_structure_anomalies(self) -> None:
        self.assertEqual(self.audit["documents_processed"], 50)
        self.assertEqual(self.audit["total_heading_aware_chunks"], 580)
        self.assertEqual(
            self.audit["logical_sections"],
            {
                "h2_h6_heading_count": 595,
                "nonempty_section_count": 545,
                "empty_body_heading_count_skipped": 50,
                "unassigned_useful_pre_h2_content_count": 1,
                "unassigned_useful_pre_h2_document_ids": ["irs_cp2000_series"],
                "terminal_heading_level_counts_for_nonempty_sections": {
                    "H2": 206,
                    "H3": 230,
                    "H4": 109,
                },
                "oversized_sections_requiring_subchunking": 33,
                "chunks_emitted_from_oversized_sections": 68,
            },
        )
        self.assertEqual(self.audit["chunker"]["markdown_h1_manifest_title_mismatch_count"], 6)
        self.assertEqual(
            self.audit["chunker"]["markdown_h1_manifest_title_mismatch_doc_ids"],
            ["irs_cp12", "irs_cp24", "irs_cp101", "irs_cp106", "irs_cp120", "irs_cp297"],
        )
        integrity = self.audit["integrity"]
        for key in (
            "chunks_with_no_heading_path",
            "duplicate_chunk_ids",
            "empty_chunks",
            "content_chunks_over_220_tokens",
            "chunks_crossing_heading_boundaries",
        ):
            self.assertEqual(integrity[key], {"count": 0, "ids": []})
        self.assertEqual(integrity["source_hash_failures"], [])
        self.assertEqual(integrity["missing_document_ids"], [])
        self.assertEqual(
            integrity["boundary_proof"],
            "every chunk window is produced from one LogicalSection content token sequence",
        )
        self.assertTrue(self.audit["quality_gate_passed"])


class HeadingHierarchyAndChunkingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tokenizer = VocabularyTokenizer()
        self.markdown = (
            "# Expanded Markdown page title\n"
            "Source: https://example.test/page\n"
            "useful introduction before H2\n"
            "## Parent section\n"
            "parent body\n"
            "#### [Deep label](https://example.test/deep)\n"
            "deep body\n"
            "```md\n"
            "### Not a real heading\n"
            "```\n"
            "### Sibling section\n"
            "sibling body\n"
            "## Empty container\n"
            "### Child section\n"
            "child body\n"
        )

    def test_literal_hierarchy_visible_paths_fences_empty_parent_and_pre_h2(self) -> None:
        title, sections, inventory = parse_logical_sections(self.markdown)
        self.assertEqual(title, "Expanded Markdown page title")
        self.assertEqual(
            [section.heading_path for section in sections],
            [
                ("Parent section",),
                ("Parent section", "Deep label"),
                ("Parent section", "Sibling section"),
                ("Empty container", "Child section"),
            ],
        )
        self.assertEqual(
            [section.heading_levels for section in sections],
            [(2,), (2, 4), (2, 3), (2, 3)],
        )
        self.assertEqual([section.section_index for section in sections], [0, 1, 2, 4])
        self.assertEqual(
            sections[1].raw_heading_path,
            ("Parent section", "[Deep label](https://example.test/deep)"),
        )
        self.assertIn("### Not a real heading", sections[1].content)
        self.assertNotIn("Not a real heading", {part for section in sections for part in section.heading_path})
        self.assertEqual(sections[0].content, "parent body")
        self.assertEqual(sections[-1].content, "child body")
        self.assertEqual(
            inventory,
            {
                "h1_count": 1,
                "h2_h6_heading_count": 5,
                "nonempty_logical_section_count": 4,
                "empty_body_heading_count": 1,
                "unassigned_useful_pre_h2_content": 1,
            },
        )

    def test_manifest_title_prefix_metadata_and_id_are_exact_and_deterministic(self) -> None:
        _title, sections, _inventory = parse_logical_sections(self.markdown)
        row = manifest_row()
        first = chunk_section(
            tokenizer=self.tokenizer,
            manifest_row=row,
            document_title=row["title"],
            section=sections[0],
        )
        repeated = chunk_section(
            tokenizer=self.tokenizer,
            manifest_row=row,
            document_title=row["title"],
            section=sections[0],
        )
        self.assertEqual(first, repeated)
        self.assertEqual(len(first), 1)
        chunk = first[0]
        prefix = (
            "Understanding your CP14 notice\n\n"
            "Notice: CP14\n\n"
            "Section:\n"
            "Parent section\n\n"
        )
        self.assertEqual(structural_prefix(row["title"], "CP14", ["Parent section"]), prefix)
        self.assertEqual(chunk.text, prefix + "parent body")
        self.assertNotIn("Expanded Markdown page title", chunk.text)
        self.assertEqual(tuple(chunk.metadata), HEADING_LOCAL_METADATA_KEYS)
        self.assertEqual(
            chunk.metadata,
            {
                **row,
                "chunk_id": chunk.chunk_id,
                "chunk_strategy": HEADING_CHUNK_STRATEGY,
                "heading": "Parent section",
                "heading_path": ["Parent section"],
                "section_index": 0,
                "subchunk_index": 0,
            },
        )
        self.assertEqual(
            chunk.chunk_id,
            f"irs_cp14__{HEADING_CHUNK_STRATEGY}__s0000__c0000__{chunk.text_sha256}",
        )

        changed_title = chunk_section(
            tokenizer=self.tokenizer,
            manifest_row=row,
            document_title="Different structural title",
            section=sections[0],
        )[0]
        self.assertNotEqual(changed_title.text_sha256, chunk.text_sha256)
        self.assertNotEqual(changed_title.chunk_id, chunk.chunk_id)

    def test_oversized_sections_use_220_40_windows_without_crossing_or_merging(self) -> None:
        first_tokens = [f"alpha-{number}" for number in range(500)]
        second_tokens = [f"beta-{number}" for number in range(3)]
        markdown = (
            "# Page title\n## First section\n"
            + " ".join(first_tokens)
            + "\n## Second section\n"
            + " ".join(second_tokens)
            + "\n"
        )
        _title, sections, inventory = parse_logical_sections(markdown)
        self.assertEqual(inventory["nonempty_logical_section_count"], 2)
        row = manifest_row()
        first = chunk_section(
            tokenizer=self.tokenizer,
            manifest_row=row,
            document_title=row["title"],
            section=sections[0],
        )
        second = chunk_section(
            tokenizer=self.tokenizer,
            manifest_row=row,
            document_title=row["title"],
            section=sections[1],
        )
        self.assertEqual(
            [(item.content_token_start, item.content_token_end, item.content_token_count) for item in first],
            [(0, 220, 220), (180, 400, 220), (360, 500, 140)],
        )
        self.assertEqual(
            first[0].content_text.split()[-40:],
            first[1].content_text.split()[:40],
        )
        self.assertEqual(
            first[1].content_text.split()[-40:],
            first[2].content_text.split()[:40],
        )
        self.assertEqual(len(second), 1)
        self.assertEqual(second[0].content_text.split(), second_tokens)
        self.assertEqual(second[0].content_token_count, 3)
        self.assertTrue(all(item.section_subchunk_count == 3 for item in first))
        self.assertTrue(all(item.metadata["heading_path"] == ["First section"] for item in first))
        self.assertTrue(all(item.metadata["section_index"] == 0 for item in first))
        self.assertEqual(second[0].metadata["heading_path"], ["Second section"])
        self.assertEqual(second[0].metadata["section_index"], 1)
        self.assertFalse(any("beta-" in item.content_text for item in first))
        self.assertFalse(any("alpha-" in item.content_text for item in second))
        self.assertTrue(all(0 < item.content_token_count <= CHUNK_SIZE for item in first + second))
        self.assertGreater(first[0].embedding_token_count, CHUNK_SIZE)

    def test_parser_rejects_unclosed_fences(self) -> None:
        with self.assertRaises(HeadingChunkingGateError):
            parse_logical_sections("# Page\n## Section\n```md\n### fake\n")


class HeadingPineconeIsolationTests(unittest.TestCase):
    @staticmethod
    def compatible_client(*, target_ids: tuple[str, ...] = ()) -> tuple[MagicMock, MagicMock]:
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
        namespace_ids = {
            BASELINE_NAMESPACE: ("fixed-1", "fixed-2"),
            HEADING_NAMESPACE: target_ids,
        }
        index.describe_index_stats.return_value = {
            "namespaces": {
                namespace: {"vector_count": len(ids)}
                for namespace, ids in namespace_ids.items()
                if ids
            }
        }
        index.list.side_effect = lambda *, namespace: [
            [{"id": vector_id} for vector_id in namespace_ids[namespace]]
        ]
        index.query.return_value = {
            "matches": [
                {"id": f"heading-{number}", "score": 0.95 - number / 100}
                for number in range(5)
            ]
        }
        index.upsert.return_value = {"upserted_count": 1}
        return client, index

    @staticmethod
    def assert_no_mutations(test: unittest.TestCase, client: MagicMock, index: MagicMock) -> None:
        client.indexes.create.assert_not_called()
        client.indexes.configure.assert_not_called()
        client.indexes.delete.assert_not_called()
        index.upsert.assert_not_called()
        index.update.assert_not_called()
        index.delete.assert_not_called()

    def test_empty_target_preflight_is_strict_and_performs_no_mutation(self) -> None:
        secret = "PHASE4B_PINECONE_SENTINEL"
        client, index = self.compatible_client()
        store = HeadingAwarePineconeStore(api_key=secret, client=client)
        state = store.require_existing_index()
        result = store.preflight_target({"new-heading-id"})

        self.assertEqual((state.dimension, state.metric, state.vector_type), (4096, "cosine", "dense"))
        self.assertEqual(result, {"preexisting_vector_count": 0, "preexisting_expected_ids": 0})
        index.list.assert_called_once_with(namespace=HEADING_NAMESPACE)
        self.assert_no_mutations(self, client, index)
        self.assertNotIn(secret, repr(store))

        with self.assertRaises(ProviderGateError):
            store.preflight_target(set())
        with self.assertRaises(ProviderGateError):
            store.namespace_snapshot("unauthorized-namespace")
        self.assert_no_mutations(self, client, index)

    def test_nonempty_target_stops_before_any_write_or_delete(self) -> None:
        client, index = self.compatible_client(target_ids=("stale-heading-id",))
        store = HeadingAwarePineconeStore(api_key="offline-sentinel", client=client)
        store.require_existing_index()
        with self.assertRaises(ProviderGateError) as raised:
            store.preflight_target({"expected-heading-id"})
        self.assertIn("not empty", str(raised.exception))
        self.assert_no_mutations(self, client, index)

    def test_query_uses_only_heading_namespace_and_exact_notice_filter(self) -> None:
        client, index = self.compatible_client()
        store = HeadingAwarePineconeStore(api_key="offline-sentinel", client=client)
        store.require_existing_index()
        vector = [0.0] * EMBEDDING_DIMENSION
        matches = store.query_known_notice(vector, notice_code="CP14", eligible_chunk_count=5)

        self.assertEqual(len(matches), 5)
        kwargs = index.query.call_args.kwargs
        self.assertEqual(kwargs["namespace"], HEADING_NAMESPACE)
        self.assertNotEqual(kwargs["namespace"], BASELINE_NAMESPACE)
        self.assertEqual(kwargs["top_k"], 5)
        self.assertEqual(kwargs["filter"], {"notice_code": {"$eq": "CP14"}})
        self.assertEqual(set(kwargs["filter"]), {"notice_code"})
        self.assertTrue(kwargs["include_metadata"])
        self.assertFalse(kwargs["include_values"])
        self.assertEqual(len(kwargs["vector"]), EMBEDDING_DIMENSION)
        self.assert_no_mutations(self, client, index)

    def test_query_results_are_deterministically_score_sorted_without_semantic_reranking(self) -> None:
        client, index = self.compatible_client()
        store = HeadingAwarePineconeStore(api_key="offline-sentinel", client=client)
        store.require_existing_index()
        # Metadata deliberately suggests a conflicting semantic order. Recovery
        # may normalize only by provider score and ID; it must not rerank.
        raw_matches = [
            {"id": "z-low", "score": 0.80, "metadata": {"semantic_priority": 1}},
            {"id": "b-tie", "score": 0.90, "metadata": {"semantic_priority": 5}},
            {"id": "a-tie", "score": 0.90, "metadata": {"semantic_priority": 4}},
            {"id": "c-low", "score": 0.70, "metadata": {"semantic_priority": 2}},
            {"id": "d-low", "score": 0.60, "metadata": {"semantic_priority": 3}},
        ]
        index.query.return_value = {"matches": raw_matches}

        matches = store.query_known_notice(
            [0.0] * EMBEDDING_DIMENSION,
            notice_code="CP14",
            eligible_chunk_count=5,
        )
        self.assertEqual(
            [match["id"] for match in matches],
            ["a-tie", "b-tie", "z-low", "c-low", "d-low"],
        )
        self.assertEqual(
            [(match["score"], match["id"]) for match in matches],
            sorted(
                [(match["score"], match["id"]) for match in raw_matches],
                key=lambda item: (-item[0], item[1]),
            ),
        )
        self.assertEqual(store.provider_response_reordered_query_count, 1)

        already_sorted = list(matches)
        index.query.return_value = {"matches": already_sorted}
        repeated = store.query_known_notice(
            [0.0] * EMBEDDING_DIMENSION,
            notice_code="CP14",
            eligible_chunk_count=5,
        )
        self.assertEqual([match["id"] for match in repeated], [match["id"] for match in matches])
        self.assertEqual(store.provider_response_reordered_query_count, 1)
        self.assert_no_mutations(self, client, index)

    def test_mocked_upsert_targets_only_heading_namespace_with_exact_metadata(self) -> None:
        client, index = self.compatible_client()
        store = HeadingAwarePineconeStore(api_key="offline-sentinel", client=client)
        store.require_existing_index()
        chunk = one_heading_chunk()
        count = store.upsert_heading_batch([chunk], [[0.0] * EMBEDDING_DIMENSION])

        self.assertEqual(count, 1)
        kwargs = index.upsert.call_args.kwargs
        self.assertEqual(kwargs["namespace"], HEADING_NAMESPACE)
        self.assertNotEqual(kwargs["namespace"], BASELINE_NAMESPACE)
        self.assertEqual(len(kwargs["vectors"]), 1)
        payload = kwargs["vectors"][0]
        self.assertEqual(payload["id"], chunk.chunk_id)
        self.assertEqual(tuple(payload["metadata"]), HEADING_LOCAL_METADATA_KEYS)
        self.assertEqual(payload["metadata"], chunk.metadata)
        self.assertEqual(len(payload["values"]), EMBEDDING_DIMENSION)
        client.indexes.create.assert_not_called()
        client.indexes.configure.assert_not_called()
        client.indexes.delete.assert_not_called()
        index.update.assert_not_called()
        index.delete.assert_not_called()

    def test_missing_index_stops_without_creating_or_opening_it(self) -> None:
        client = MagicMock()
        client.indexes.exists.return_value = False
        store = HeadingAwarePineconeStore(api_key="offline-sentinel", client=client)
        with self.assertRaises(ProviderGateError):
            store.require_existing_index()
        client.indexes.describe.assert_not_called()
        client.indexes.create.assert_not_called()
        client.index.assert_not_called()

    def test_failed_target_preflight_happens_before_embedding_or_any_mutation(self) -> None:
        chunks = load_heading_registry(ROOT / HEADING_REGISTRY_PATH)
        fixed_chunks = load_chunk_registry(ROOT / FIXED_REGISTRY_PATH)
        fixed_ids = {chunk.chunk_id for chunk in fixed_chunks}
        questions = load_section_questions(ROOT / SECTION_QUESTIONS_PATH)
        self.assertEqual((len(chunks), len(fixed_ids)), (580, EXPECTED_FIXED_VECTOR_COUNT))

        store = MagicMock()
        store.require_existing_index.return_value = SimpleNamespace()
        store.namespace_snapshot.return_value = (EXPECTED_FIXED_VECTOR_COUNT, fixed_ids)
        store.preflight_target.side_effect = ProviderGateError("target is not empty")
        with (
            patch("noticelens.phase4b.HeadingAwarePineconeStore", return_value=store) as constructor,
            patch("noticelens.phase4b.NebiusEmbeddings") as embedder,
        ):
            with self.assertRaises(ProviderGateError):
                evaluate_and_index(
                    project_root=ROOT,
                    config=SimpleNamespace(pinecone_api_key="offline-sentinel"),
                    frozen_before={},
                    chunks=chunks,
                    chunk_audit={},
                    questions=questions,
                    fixed_results={},
                    fixed_by_id={},
                )

        constructor.assert_called_once_with(api_key="offline-sentinel")
        store.require_existing_index.assert_called_once_with()
        store.namespace_snapshot.assert_called_once_with(BASELINE_NAMESPACE)
        expected_ids = store.preflight_target.call_args.args[0]
        self.assertEqual(expected_ids, {chunk.chunk_id for chunk in chunks})
        embedder.assert_not_called()
        store.upsert_heading_batch.assert_not_called()
        store.query_known_notice.assert_not_called()
        store.wait_for_target_parity.assert_not_called()


class QueryOnlyRecoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.chunks = load_heading_registry(ROOT / HEADING_REGISTRY_PATH)
        cls.heading_ids = {chunk.chunk_id for chunk in cls.chunks}
        cls.fixed_ids = {
            chunk.chunk_id for chunk in load_chunk_registry(ROOT / FIXED_REGISTRY_PATH)
        }
        cls.questions = load_section_questions(ROOT / SECTION_QUESTIONS_PATH)
        cls.fixed_results, cls.fixed_by_id = load_and_validate_fixed_results(
            ROOT, cls.questions
        )
        cls.chunk_audit = json.loads((ROOT / CHUNK_AUDIT_PATH).read_text(encoding="utf-8"))
        cls.frozen = verify_frozen_artifacts(ROOT)
        if len(cls.heading_ids) != EXPECTED_HEADING_CHUNK_COUNT:
            raise AssertionError("Frozen heading registry fixture is not exactly 580 IDs")
        if len(cls.fixed_ids) != EXPECTED_FIXED_VECTOR_COUNT:
            raise AssertionError("Frozen baseline registry fixture is not exactly 350 IDs")

    def recovery_store(self, *, heading_ids: set[str] | None = None) -> MagicMock:
        remote_heading_ids = self.heading_ids if heading_ids is None else heading_ids
        store = MagicMock()
        store.require_existing_index.return_value = SimpleNamespace(
            dimension=4096,
            metric="cosine",
            cloud="aws",
            region="us-east-1",
        )

        def snapshot(namespace: str) -> tuple[int, set[str]]:
            if namespace == BASELINE_NAMESPACE:
                return len(self.fixed_ids), set(self.fixed_ids)
            if namespace == HEADING_NAMESPACE:
                return len(remote_heading_ids), set(remote_heading_ids)
            raise AssertionError(f"Unexpected namespace: {namespace}")

        store.namespace_snapshot.side_effect = snapshot
        store.provider_response_reordered_query_count = 0
        return store

    @staticmethod
    def successful_trace(**kwargs: Any) -> tuple[dict[str, Any], float]:
        question = kwargs["question"]
        manifest_row = kwargs["manifest_row"]
        path = list(question["expected_heading_path"])
        trace = {
            **question,
            "notice_family": manifest_row["notice_family"],
            "ranks": [
                {
                    "rank": 1,
                    "text_preview": "offline recovery preview",
                    "heading_path": path,
                    "attributed_heading_paths": [path],
                }
            ],
            "section_score": {
                "precision_at_1": 1,
                "reciprocal_rank": 1.0,
                "hit_at_5": 1,
                "first_correct_rank": 1,
            },
            "query_latency_seconds": 0.001,
        }
        return trace, 0.001

    def test_completed_580_id_target_recovers_by_querying_only_with_zero_upserts(self) -> None:
        store = self.recovery_store()
        embedder = MagicMock()
        embedder.request_latencies = []
        question_vectors = [[0.0] * EMBEDDING_DIMENSION for _ in self.questions]
        embedder.embed_documents.return_value = question_vectors
        config = SimpleNamespace(
            pinecone_api_key="offline-pinecone",
            nebius_api_key="offline-nebius",
            nebius_base_url="https://offline.invalid/v1",
        )

        with (
            patch("noticelens.phase4b.HeadingAwarePineconeStore", return_value=store),
            patch("noticelens.phase4b.NebiusEmbeddings", return_value=embedder),
            patch("noticelens.phase4b._trace_heading_query", side_effect=self.successful_trace) as trace,
        ):
            results = evaluate_and_index(
                project_root=ROOT,
                config=config,
                frozen_before=self.frozen,
                chunks=self.chunks,
                chunk_audit=self.chunk_audit,
                questions=self.questions,
                fixed_results=self.fixed_results,
                fixed_by_id=self.fixed_by_id,
                resume_query_only=True,
            )

        embedder.embed_documents.assert_called_once_with(
            [question["question"] for question in self.questions]
        )
        self.assertEqual(trace.call_count, 15)
        store.preflight_target.assert_not_called()
        store.upsert_heading_batch.assert_not_called()
        store.wait_for_target_parity.assert_not_called()
        self.assertEqual(results["execution"]["mode"], "query_only_recovery_after_completed_namespace_population")
        self.assertTrue(results["execution"]["query_only_recovery"])
        self.assertEqual(
            results["experiment_contract"]["execution_mode"],
            "query_only_recovery_after_completed_namespace_population",
        )
        self.assertEqual(results["embedding"]["document_chunk_count"], 580)
        self.assertEqual(results["embedding"]["document_chunks_embedded"], 0)
        self.assertEqual(results["embedding"]["question_count_embedded"], 15)
        self.assertEqual(results["pinecone"]["heading_embedded_count"], 0)
        self.assertEqual(results["pinecone"]["heading_upserted_count"], 0)
        self.assertEqual(results["pinecone"]["heading_namespace_count"], 580)
        self.assertTrue(results["pinecone"]["heading_exact_id_parity"])
        self.assertEqual(results["pinecone"]["index_create_calls"], 0)
        self.assertEqual(results["pinecone"]["baseline_upsert_calls"], 0)
        self.assertEqual(results["pinecone"]["delete_clear_update_calls"], 0)
        self.assertEqual(results["quality_gate_passed"], True)

        called_namespaces = [call.args[0] for call in store.namespace_snapshot.call_args_list]
        self.assertIn(BASELINE_NAMESPACE, called_namespaces)
        self.assertIn(HEADING_NAMESPACE, called_namespaces)
        self.assertEqual(sha256_file(ROOT / HEADING_REGISTRY_PATH), EXPECTED_REGISTRY_SHA256)
        self.assertEqual(sha256_file(ROOT / CHUNK_AUDIT_PATH), EXPECTED_AUDIT_SHA256)

    def test_recovery_rejects_any_target_id_mismatch_before_embedding_or_mutation(self) -> None:
        missing_one = set(self.heading_ids)
        missing_one.remove(next(iter(missing_one)))
        store = self.recovery_store(heading_ids=missing_one)
        config = SimpleNamespace(
            pinecone_api_key="offline-pinecone",
            nebius_api_key="offline-nebius",
            nebius_base_url="https://offline.invalid/v1",
        )
        with (
            patch("noticelens.phase4b.HeadingAwarePineconeStore", return_value=store),
            patch("noticelens.phase4b.NebiusEmbeddings") as embedder,
        ):
            with self.assertRaises(Phase4BGateError):
                evaluate_and_index(
                    project_root=ROOT,
                    config=config,
                    frozen_before=self.frozen,
                    chunks=self.chunks,
                    chunk_audit=self.chunk_audit,
                    questions=self.questions,
                    fixed_results=self.fixed_results,
                    fixed_by_id=self.fixed_by_id,
                    resume_query_only=True,
                )

        embedder.assert_not_called()
        store.preflight_target.assert_not_called()
        store.upsert_heading_batch.assert_not_called()
        store.wait_for_target_parity.assert_not_called()
        store.query_known_notice.assert_not_called()

    def test_cli_exposes_explicit_resume_query_only_flag(self) -> None:
        default = build_argument_parser().parse_args([])
        recovery = build_argument_parser().parse_args(["--resume-query-only"])
        self.assertFalse(default.resume_query_only)
        self.assertTrue(recovery.resume_query_only)


class HeadingScoringAndComparisonTests(unittest.TestCase):
    @staticmethod
    def score(*, precision: int, rank: int | None) -> dict[str, int | float | None]:
        return {
            "precision_at_1": precision,
            "reciprocal_rank": 0.0 if rank is None else 1.0 / rank,
            "hit_at_5": int(rank is not None),
            "first_correct_rank": rank,
        }

    def test_heading_aggregates_have_exact_overall_style_and_family_denominators(self) -> None:
        traces: list[dict[str, Any]] = []
        for number in range(15):
            if number < 9:
                score = self.score(precision=1, rank=1)
            elif number < 12:
                score = self.score(precision=0, rank=2)
            else:
                score = self.score(precision=0, rank=None)
            traces.append(
                {
                    "id": f"S{number + 1:02d}",
                    "language_style": "naive" if number < 10 else "expert",
                    "notice_family": "family_a" if number < 8 else "family_b",
                    "section_score": score,
                }
            )

        result = aggregate_heading_traces(traces)
        overall = result["overall"]
        self.assertEqual((overall["n"], overall["correct_at_1"], overall["hit_at_5_count"]), (15, 9, 12))
        self.assertAlmostEqual(float(overall["section_precision_at_1"]), 0.6)
        self.assertAlmostEqual(float(overall["section_mrr"]), 0.7)
        self.assertAlmostEqual(float(overall["section_hit_at_5"]), 0.8)
        self.assertEqual(result["by_language_style"]["naive"]["n"], 10)
        self.assertAlmostEqual(float(result["by_language_style"]["naive"]["section_precision_at_1"]), 0.9)
        self.assertAlmostEqual(float(result["by_language_style"]["naive"]["section_mrr"]), 0.95)
        self.assertEqual(result["by_language_style"]["expert"]["n"], 5)
        self.assertAlmostEqual(float(result["by_language_style"]["expert"]["section_mrr"]), 0.2)
        self.assertEqual(result["by_notice_family"]["family_a"]["n"], 8)
        self.assertEqual(result["by_notice_family"]["family_a"]["section_precision_at_1"], 1.0)
        self.assertEqual(result["by_notice_family"]["family_b"]["n"], 7)

        with self.assertRaises(Phase4BGateError):
            aggregate_heading_traces(traces[:-1])

    @staticmethod
    def comparison_trace(*, precision: int, rank: int | None, heading: bool) -> dict[str, Any]:
        top = {
            "text_preview": "offline top-one preview",
            "heading_path": ["Expected section"],
            "attributed_heading_paths": [["Expected section"]],
        }
        return {
            "section_score": {
                "precision_at_1": precision,
                "first_correct_rank": rank,
            },
            "ranks": [top if heading else dict(top)],
        }

    def test_paired_classification_is_based_only_on_p_at_1_not_rank_movement(self) -> None:
        questions = [
            {
                "id": f"S{number:02d}",
                "question": f"Question {number}",
                "expected_notice_code": f"CP{number}",
                "expected_heading": "Expected section",
                "expected_heading_path": ["Expected section"],
            }
            for number in range(1, 5)
        ]
        fixed = {
            "S01": self.comparison_trace(precision=0, rank=2, heading=False),
            "S02": self.comparison_trace(precision=1, rank=1, heading=False),
            "S03": self.comparison_trace(precision=0, rank=5, heading=False),
            "S04": self.comparison_trace(precision=0, rank=None, heading=False),
        }
        heading = {
            "S01": self.comparison_trace(precision=1, rank=1, heading=True),
            "S02": self.comparison_trace(precision=0, rank=2, heading=True),
            "S03": self.comparison_trace(precision=0, rank=2, heading=True),
            "S04": self.comparison_trace(precision=0, rank=None, heading=True),
        }

        rows, counts = build_paired_comparison(questions, fixed, heading)
        by_id = {row["question_id"]: row for row in rows}
        self.assertEqual(counts, {"improved": 1, "unchanged": 2, "regressed": 1})
        self.assertEqual(by_id["S01"]["classification"], "IMPROVED")
        self.assertEqual(by_id["S02"]["classification"], "REGRESSED")
        self.assertEqual(by_id["S03"]["classification"], "UNCHANGED")
        self.assertEqual(by_id["S04"]["classification"], "UNCHANGED")
        self.assertEqual(by_id["S03"]["rank_change"], 3)
        self.assertEqual(by_id["S04"]["rank_change"], 0)
        self.assertEqual(
            counts["improved"] - counts["regressed"],
            sum(row["heading_correct_at_1"] - row["fixed_correct_at_1"] for row in rows),
        )


class FrozenPhaseOneThroughFourATests(unittest.TestCase):
    def test_exact_phase1_through_phase4a_file_tree_and_composite_hashes(self) -> None:
        self.assertEqual(FROZEN_COMPOSITE_SHA256, EXPECTED_FROZEN_COMPOSITE_SHA256)
        self.assertEqual(len(FROZEN_FILE_HASHES), 18)
        self.assertEqual(len(FROZEN_TREE_HASHES), 3)
        for relative, expected_hash in FROZEN_FILE_HASHES.items():
            with self.subTest(relative=relative):
                self.assertEqual(sha256_file(ROOT / relative), expected_hash)

        observed = verify_frozen_artifacts(ROOT)
        self.assertEqual(observed["approved_composite_sha256"], EXPECTED_FROZEN_COMPOSITE_SHA256)
        self.assertEqual(len(observed), 22)
        for relative, (expected_count, expected_hash) in FROZEN_TREE_HASHES.items():
            with self.subTest(relative=relative):
                self.assertEqual(
                    observed[relative],
                    {"file_count": expected_count, "tree_sha256": expected_hash},
                )

    def test_frozen_phase4a_results_recompute_to_the_predeclared_metrics_and_ranks(self) -> None:
        questions = load_section_questions(ROOT / SECTION_QUESTIONS_PATH)
        fixed, by_id = load_and_validate_fixed_results(ROOT, questions)
        self.assertEqual(len(fixed["queries"]), 15)
        self.assertEqual(len(by_id), 15)
        self.assertEqual(
            FIXED_METRICS,
            {
                "section_precision_at_1": 0.8,
                "section_mrr": 0.8688888888888889,
                "section_hit_at_5": 1.0,
                "correct_at_1": 12,
                "n": 15,
            },
        )
        self.assertEqual(SPECIAL_FIXED_RANKS, {"S03": 2, "S07": 5, "S11": 3})
        self.assertEqual(
            {
                question_id: by_id[question_id]["section_score"]["first_correct_rank"]
                for question_id in SPECIAL_FIXED_RANKS
            },
            SPECIAL_FIXED_RANKS,
        )


if __name__ == "__main__":
    unittest.main()
