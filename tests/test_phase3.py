from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from noticelens.chunking import (  # noqa: E402
    CHUNK_STRATEGY,
    LOCAL_METADATA_KEYS,
    ChunkRecord,
    FixedTokenTextSplitter,
    _build_chunk_id,
    sha256_bytes,
)
from noticelens.config import (  # noqa: E402
    ConfigurationError,
    load_phase3_config,
)
from noticelens.evaluation import (  # noqa: E402
    EvaluationGateError,
    aggregate_scores,
    build_notice_alias_registry,
    normalize_notice_code,
    notice_tokens,
    load_golden_questions,
    questions_for_retrieval,
    resolve_notice_alias,
    score_ranking,
    validate_embedding_dimension,
)
from noticelens.phase3 import frozen_input_snapshot  # noqa: E402
from noticelens.providers import (  # noqa: E402
    EMBEDDING_DIMENSION,
    EMBEDDING_MODEL,
    NAMESPACE,
    PINECONE_METADATA_KEYS,
    NebiusEmbeddings,
    PineconeBaselineStore,
    ProviderGateError,
)


class VocabularyTokenizer:
    """Small reversible tokenizer used to keep chunk tests fully offline."""

    def __init__(self) -> None:
        self._token_to_id: dict[str, int] = {}
        self._id_to_token: dict[int, str] = {}

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        if add_special_tokens:
            raise AssertionError("Phase 3 must not add tokenizer special tokens")
        encoded: list[int] = []
        for token in text.split():
            if token not in self._token_to_id:
                token_id = len(self._token_to_id)
                self._token_to_id[token] = token_id
                self._id_to_token[token_id] = token
            encoded.append(self._token_to_id[token])
        return encoded

    def decode(
        self,
        token_ids: list[int],
        *,
        skip_special_tokens: bool,
        clean_up_tokenization_spaces: bool,
    ) -> str:
        if not skip_special_tokens or clean_up_tokenization_spaces:
            raise AssertionError("Unexpected tokenizer decode configuration")
        return " ".join(self._id_to_token[token_id] for token_id in token_ids)


def make_chunk(doc_id: str = "irs_cp14", ordinal: int = 0) -> ChunkRecord:
    text = "IRS CP14 balance due guidance"
    text_hash = sha256_bytes(text.encode("utf-8"))
    chunk_id = _build_chunk_id(doc_id, ordinal, text_hash)
    metadata: dict[str, str | int] = {
        "doc_id": doc_id,
        "notice_code": "CP14",
        "notice_family": "balance_collection",
        "title": "Understanding your CP14 notice",
        "source_url": "https://www.irs.gov/individuals/understanding-your-cp14-notice",
        "source_origin": "IRS",
        "chunk_id": chunk_id,
        "chunk_strategy": CHUNK_STRATEGY,
    }
    if tuple(metadata) != LOCAL_METADATA_KEYS:
        raise AssertionError("Test fixture metadata no longer matches the Phase 3 contract")
    return ChunkRecord(
        chunk_id=chunk_id,
        text=text,
        token_start=0,
        token_end=5,
        token_count=5,
        source_token_count=5,
        text_sha256=text_hash,
        metadata=metadata,
    )


class ConfigurationTests(unittest.TestCase):
    def test_environment_configuration_never_exposes_secret_values(self) -> None:
        nebius_secret = "NEBIUS_SENTINEL_DO_NOT_PRINT"
        pinecone_secret = "PINECONE_SENTINEL_DO_NOT_PRINT"
        with tempfile.TemporaryDirectory() as project_directory, tempfile.TemporaryDirectory() as external:
            missing_file = Path(external) / "does-not-exist.env"
            captured = io.StringIO()
            with redirect_stdout(captured), redirect_stderr(captured):
                config = load_phase3_config(
                    environ={
                        "NEBIUS_API_KEY": f"  {nebius_secret}  ",
                        "PINECONE_API_KEY": f"  {pinecone_secret}  ",
                    },
                    secret_path=missing_file,
                    project_root=Path(project_directory),
                )

        rendered = "\n".join(
            (
                repr(config),
                str(config),
                json.dumps(config.public_summary(), sort_keys=True),
                captured.getvalue(),
            )
        )
        self.assertEqual(config.nebius_api_key, nebius_secret)
        self.assertEqual(config.pinecone_api_key, pinecone_secret)
        self.assertNotIn(nebius_secret, rendered)
        self.assertNotIn(pinecone_secret, rendered)
        self.assertNotIn("nebius_api_key", repr(config))
        self.assertNotIn("pinecone_api_key", repr(config))
        self.assertEqual(
            set(config.public_summary()),
            {"nebius_base_url", "secrets_source"},
        )

    def test_missing_key_error_and_external_file_loading_are_secret_safe(self) -> None:
        environment_secret = "NEBIUS_ENV_SENTINEL"
        file_secret = "PINECONE_FILE_SENTINEL"
        with tempfile.TemporaryDirectory() as project_directory, tempfile.TemporaryDirectory() as external:
            missing_file = Path(external) / "missing.env"
            with self.assertRaises(ConfigurationError) as raised:
                load_phase3_config(
                    environ={"NEBIUS_API_KEY": environment_secret},
                    secret_path=missing_file,
                    project_root=Path(project_directory),
                )
            self.assertIn("PINECONE_API_KEY", str(raised.exception))
            self.assertNotIn(environment_secret, str(raised.exception))

            secrets_file = Path(external) / "noticelens.env"
            secrets_file.touch()
            with patch(
                "noticelens.config.dotenv_values",
                return_value={
                    "NEBIUS_API_KEY": "FILE_VALUE_MUST_NOT_OVERRIDE_ENV",
                    "PINECONE_API_KEY": file_secret,
                },
            ) as dotenv_loader:
                config = load_phase3_config(
                    environ={"NEBIUS_API_KEY": environment_secret},
                    secret_path=secrets_file,
                    project_root=Path(project_directory),
                )
            dotenv_loader.assert_called_once_with(secrets_file.resolve())
            self.assertEqual(config.nebius_api_key, environment_secret)
            self.assertEqual(config.pinecone_api_key, file_secret)
            self.assertEqual(config.public_summary()["secrets_source"], "external_local_file")

    def test_project_local_secret_file_is_rejected_without_being_read(self) -> None:
        with tempfile.TemporaryDirectory() as project_directory:
            project_root = Path(project_directory)
            unsafe_path = project_root / "unsafe.env"
            with patch("noticelens.config.dotenv_values") as dotenv_loader:
                with self.assertRaises(ConfigurationError):
                    load_phase3_config(
                        environ={},
                        secret_path=unsafe_path,
                        project_root=project_root,
                    )
            dotenv_loader.assert_not_called()


class FixedChunkTests(unittest.TestCase):
    def test_fixed_windows_use_220_tokens_with_40_token_overlap(self) -> None:
        tokens = [f"token-{index}" for index in range(500)]
        tokens[199:202] = ["##", "Heading", "CP501"]
        splitter = FixedTokenTextSplitter(VocabularyTokenizer())
        windows = splitter.split_windows(" ".join(tokens))

        self.assertEqual(
            [(item.token_start, item.token_end, item.token_count) for item in windows],
            [(0, 220, 220), (180, 400, 220), (360, 500, 140)],
        )
        self.assertEqual(windows[0].text.split()[-40:], windows[1].text.split()[:40])
        self.assertEqual(windows[1].text.split()[-40:], windows[2].text.split()[:40])
        self.assertIn("## Heading CP501", windows[0].text)
        self.assertIn("## Heading CP501", windows[1].text)

    def test_fixed_window_edge_cases_do_not_emit_overlap_only_chunk(self) -> None:
        expected = {
            0: [],
            1: [(0, 1)],
            220: [(0, 220)],
            221: [(0, 220), (180, 221)],
            400: [(0, 220), (180, 400)],
            401: [(0, 220), (180, 400), (360, 401)],
        }
        for token_count, spans in expected.items():
            with self.subTest(token_count=token_count):
                text = " ".join(f"t{index}" for index in range(token_count))
                windows = FixedTokenTextSplitter(VocabularyTokenizer()).split_windows(text)
                self.assertEqual([(item.token_start, item.token_end) for item in windows], spans)

    def test_chunk_id_is_deterministic_and_identity_sensitive(self) -> None:
        first_hash = sha256_bytes(b"same chunk text")
        second_hash = sha256_bytes(b"changed chunk text")
        first = _build_chunk_id("irs_cp14", 0, first_hash)
        self.assertEqual(first, _build_chunk_id("irs_cp14", 0, first_hash))
        self.assertNotEqual(first, _build_chunk_id("irs_cp501", 0, first_hash))
        self.assertNotEqual(first, _build_chunk_id("irs_cp14", 1, first_hash))
        self.assertNotEqual(first, _build_chunk_id("irs_cp14", 0, second_hash))
        self.assertEqual(
            first,
            f"irs_cp14__fixed_220_40__000000__{first_hash}",
        )


class EvaluationTests(unittest.TestCase):
    def test_notice_normalization_and_composite_aliases(self) -> None:
        self.assertEqual(normalize_notice_code(" cp-14 "), "CP14")
        self.assertEqual(normalize_notice_code("cp 05a"), "CP05A")
        self.assertEqual(normalize_notice_code("Letter 1058"), "LETTER1058")
        self.assertIsNone(normalize_notice_code("CP06/CP06A"))
        self.assertEqual(notice_tokens("CP06/CP06A"), ["CP06", "CP06A"])
        self.assertEqual(
            notice_tokens("LT11 / Letter 1058"),
            ["LT11", "LETTER1058"],
        )

        registry = build_notice_alias_registry(
            [
                {"doc_id": "irs_cp06_cp06a", "notice_code": "CP06/CP06A"},
                {"doc_id": "irs_lt11_1058", "notice_code": "LT11 / Letter 1058"},
            ]
        )
        self.assertEqual(resolve_notice_alias("CP06A", registry), "irs_cp06_cp06a")
        self.assertEqual(resolve_notice_alias("Letter 1058", registry), "irs_lt11_1058")
        self.assertIsNone(resolve_notice_alias("CP9999", registry))

    def test_notice_alias_collision_stops_instead_of_guessing(self) -> None:
        with self.assertRaises(EvaluationGateError):
            build_notice_alias_registry(
                [
                    {"doc_id": "first", "notice_code": "CP14"},
                    {"doc_id": "second", "notice_code": "CP14"},
                ]
            )

    def test_precision_mrr_and_hit_at_5_use_first_expected_document_rank(self) -> None:
        rankings = [
            ["expected", "wrong", "wrong", "wrong", "wrong"],
            ["wrong", "expected", "expected", "wrong", "wrong"],
            ["wrong", "wrong", "wrong", "wrong", "expected"],
            ["wrong", "wrong", "wrong", "wrong", "wrong", "expected"],
        ]
        scores = [score_ranking("expected", ranking) for ranking in rankings]
        self.assertEqual(
            [
                (
                    score.precision_at_1,
                    score.reciprocal_rank,
                    score.hit_at_5,
                    score.first_expected_rank,
                )
                for score in scores
            ],
            [
                (1, 1.0, 1, 1),
                (0, 0.5, 1, 2),
                (0, 0.2, 1, 5),
                (0, 0.0, 0, None),
            ],
        )
        aggregate = aggregate_scores(scores)
        self.assertEqual(aggregate["n"], 4)
        self.assertAlmostEqual(float(aggregate["precision_at_1"]), 0.25)
        self.assertAlmostEqual(float(aggregate["mrr"]), 0.425)
        self.assertAlmostEqual(float(aggregate["hit_at_5"]), 0.75)

    def test_embedding_dimension_requires_4096_finite_numbers(self) -> None:
        vector = [0.0] * EMBEDDING_DIMENSION
        self.assertEqual(validate_embedding_dimension(vector), EMBEDDING_DIMENSION)
        with self.assertRaises(EvaluationGateError):
            validate_embedding_dimension(vector[:-1])
        for invalid in (float("nan"), float("inf"), True, "0.0"):
            with self.subTest(invalid=invalid):
                malformed = vector.copy()
                malformed[0] = invalid  # type: ignore[list-item]
                with self.assertRaises(EvaluationGateError):
                    validate_embedding_dimension(malformed)


class FrozenContractTests(unittest.TestCase):
    def test_frozen_scope_hashes_and_retrieval_question_selection(self) -> None:
        snapshot = frozen_input_snapshot(ROOT)
        self.assertEqual(
            snapshot["sample_notice_manifest"]["sha256"],
            "0a92688fe008ee7fbc10fb5ec4b41733ecad250a47110335cc65e3d72dc99769",
        )
        self.assertEqual(snapshot["processed_guidance_inventory"]["file_count"], 50)
        self.assertIn("phase2_evaluation_manifest", snapshot)

        questions = load_golden_questions(
            ROOT / "eval" / "golden_questions.json",
            ROOT / "data" / "corpus_manifest.csv",
        )
        selected = questions_for_retrieval(questions)
        self.assertEqual(len(selected), 26)
        self.assertEqual(
            {prefix: sum(item["id"].startswith(prefix) for item in selected) for prefix in "ABCDE"},
            {"A": 6, "B": 8, "C": 8, "D": 4, "E": 0},
        )


class ProviderTests(unittest.TestCase):
    def test_nebius_mock_reorders_indices_and_never_leaks_key(self) -> None:
        secret = "NEBIUS_PROVIDER_SENTINEL"
        client = MagicMock()
        client.embeddings.create.return_value = SimpleNamespace(
            data=[
                SimpleNamespace(index=1, embedding=[2.0] * EMBEDDING_DIMENSION),
                SimpleNamespace(index=0, embedding=[1.0] * EMBEDDING_DIMENSION),
            ]
        )
        with patch("noticelens.providers.OpenAI", return_value=client) as constructor:
            embedder = NebiusEmbeddings(api_key=secret, batch_size=2)
            vectors = embedder.embed_documents(["first question", "second question"])

        constructor.assert_called_once()
        constructor_kwargs = constructor.call_args.kwargs
        self.assertEqual(constructor_kwargs["api_key"], secret)
        client.embeddings.create.assert_called_once_with(
            model=EMBEDDING_MODEL,
            input=["first question", "second question"],
            encoding_format="float",
        )
        self.assertEqual([vector[0] for vector in vectors], [1.0, 2.0])
        self.assertNotIn(secret, repr(embedder))

        client.embeddings.create.side_effect = RuntimeError(f"provider leaked {secret}")
        with self.assertRaises(ProviderGateError) as raised:
            embedder.embed_query("raw frozen question")
        self.assertNotIn(secret, str(raised.exception))
        self.assertIn("no credentials were logged", str(raised.exception))

    def test_nebius_mock_rejects_duplicate_response_indices(self) -> None:
        client = MagicMock()
        client.embeddings.create.return_value = SimpleNamespace(
            data=[
                SimpleNamespace(index=0, embedding=[1.0] * EMBEDDING_DIMENSION),
                SimpleNamespace(index=0, embedding=[2.0] * EMBEDDING_DIMENSION),
            ]
        )
        with patch("noticelens.providers.OpenAI", return_value=client):
            embedder = NebiusEmbeddings(api_key="offline-sentinel", batch_size=2)
            with self.assertRaises(ProviderGateError):
                embedder.embed_documents(["first", "second"])

    @staticmethod
    def _compatible_pinecone() -> tuple[MagicMock, MagicMock]:
        management = MagicMock()
        data_index = MagicMock()
        management.indexes.exists.return_value = True
        management.indexes.describe.return_value = {
            "dimension": EMBEDDING_DIMENSION,
            "metric": "cosine",
            "vector_type": "dense",
            "host": "offline-index-host",
            "status": {"ready": True},
            # The frozen reuse gate checks dimension and metric; location is reported.
            "spec": {"serverless": {"cloud": "gcp", "region": "europe-west4"}},
        }
        management.index.return_value = data_index
        return management, data_index

    def test_pinecone_mock_reuses_compatible_index_and_sends_whitelisted_metadata(self) -> None:
        secret = "PINECONE_PROVIDER_SENTINEL"
        management, data_index = self._compatible_pinecone()
        data_index.upsert.return_value = {"upserted_count": 1}
        data_index.query.return_value = {"matches": [{"id": f"id-{rank}"} for rank in range(5)]}

        with patch("noticelens.providers.Pinecone", return_value=management) as constructor:
            store = PineconeBaselineStore(api_key=secret)
            state = store.ensure_compatible_index()
            chunk = make_chunk()
            vector = [0.0] * EMBEDDING_DIMENSION
            self.assertEqual(store.upsert_batch([chunk], [vector]), 1)
            self.assertEqual(len(store.query(vector)), 5)

        constructor.assert_called_once_with(api_key=secret)
        management.indexes.create.assert_not_called()
        management.index.assert_called_once_with(host="offline-index-host")
        self.assertEqual(state.existed_or_created, "reused")
        self.assertEqual((state.cloud, state.region), ("gcp", "europe-west4"))
        self.assertNotIn(secret, repr(store))

        upsert_kwargs = data_index.upsert.call_args.kwargs
        self.assertEqual(upsert_kwargs["namespace"], NAMESPACE)
        payload = upsert_kwargs["vectors"][0]
        self.assertEqual(payload["id"], chunk.chunk_id)
        self.assertEqual(tuple(payload["metadata"]), PINECONE_METADATA_KEYS)
        self.assertNotIn("source_origin", payload["metadata"])
        self.assertNotIn("text", payload["metadata"])

        query_kwargs = data_index.query.call_args.kwargs
        self.assertEqual(query_kwargs["namespace"], NAMESPACE)
        self.assertEqual(query_kwargs["top_k"], 5)
        self.assertTrue(query_kwargs["include_metadata"])
        self.assertFalse(query_kwargs["include_values"])
        with self.assertRaises(ProviderGateError):
            store.query(vector, top_k=4)

    def test_pinecone_mock_stops_on_incompatible_index_without_modifying_it(self) -> None:
        management, _ = self._compatible_pinecone()
        management.indexes.describe.return_value["dimension"] = 3072
        with patch("noticelens.providers.Pinecone", return_value=management):
            store = PineconeBaselineStore(api_key="offline-sentinel")
            with self.assertRaises(ProviderGateError):
                store.ensure_compatible_index()
        management.indexes.create.assert_not_called()
        management.index.assert_not_called()



if __name__ == "__main__":
    unittest.main()
