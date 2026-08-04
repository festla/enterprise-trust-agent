from pathlib import Path
from types import SimpleNamespace

import pytest

import app.services.complex_oracle_retrieval_runtime as runtime_module
from app.schemas.bm25 import (
    BM25TokenizerSpec,
)
from app.schemas.hybrid_retrieval import (
    RRFConfig,
)
from app.schemas.reranker import (
    RerankerRuntimeConfig,
)
from app.services.complex_oracle_retrieval_runtime import (
    ComplexOracleRetrievalRuntimeError,
    ComplexReportRetrievalRoute,
    build_complex_oracle_hit_provider,
)


class FakeTokenizer:
    def __init__(self, *, spec):
        self.spec = spec


class FakeHybridRetriever:
    def __init__(
        self,
        *,
        dense_index,
        bm25_index,
        provider,
        tokenizer,
        config,
    ):
        self.dense_index = dense_index
        self.bm25_index = bm25_index
        self.provider = provider
        self.tokenizer = tokenizer
        self.config = config


def make_manifest(
    *,
    report_id,
    document_id=None,
):
    return SimpleNamespace(
        company_id=(
            report_id.removesuffix(
                "_2024"
            )
        ),
        report_id=report_id,
        fiscal_year=2024,
        report_type="annual_report",
        document_id=(
            document_id
            or f"doc_{report_id}_test"
        ),
        chunk_dataset_id=(
            f"chunk_dataset_{report_id}_test"
        ),
        chunk_strategy="fixed_length",
        tokenizer_spec=(
            BM25TokenizerSpec()
        ),
    )


def prepare_runtime(
    *,
    tmp_path,
    monkeypatch,
    report_ids=(
        "midea_group_2024",
    ),
    bm25_document_override=None,
):
    vector_root = (
        tmp_path / "vector"
    )

    vector_root.mkdir()

    routes = []

    for report_id in report_ids:
        chunk_dir = (
            tmp_path
            / "chunks"
            / report_id
        )

        bm25_dir = (
            tmp_path
            / "bm25"
            / report_id
        )

        chunk_dir.mkdir(
            parents=True
        )

        bm25_dir.mkdir(
            parents=True
        )

        routes.append(
            ComplexReportRetrievalRoute(
                report_id=report_id,
                chunk_dataset_directory=(
                    chunk_dir
                ),
                bm25_index_directory=(
                    bm25_dir
                ),
            )
        )

    def fake_build_vector_index(
        *,
        chunk_dataset_directory,
        output_root,
        provider,
    ):
        report_id = (
            chunk_dataset_directory.name
        )

        return SimpleNamespace(
            manifest=make_manifest(
                report_id=report_id
            ),
            index=(
                f"dense_index_{report_id}"
            ),
            created=False,
        )

    def fake_load_bm25_index(
        index_directory,
    ):
        report_id = (
            index_directory.name
        )

        document_id = None

        if (
            bm25_document_override
            is not None
        ):
            document_id = (
                bm25_document_override
            )

        return SimpleNamespace(
            manifest=make_manifest(
                report_id=report_id,
                document_id=document_id,
            ),
            index=(
                f"bm25_index_{report_id}"
            ),
        )

    monkeypatch.setattr(
        runtime_module,
        "build_vector_index",
        fake_build_vector_index,
    )

    monkeypatch.setattr(
        runtime_module,
        "load_bm25_index",
        fake_load_bm25_index,
    )

    monkeypatch.setattr(
        runtime_module,
        "DeterministicChineseBigramTokenizer",
        FakeTokenizer,
    )

    monkeypatch.setattr(
        runtime_module,
        "HybridRetriever",
        FakeHybridRetriever,
    )

    return routes, vector_root


def build_runtime(
    *,
    routes,
    vector_root,
):
    return build_complex_oracle_hit_provider(
        routes=routes,
        vector_index_output_root=(
            vector_root
        ),
        embedding_provider=(
            SimpleNamespace(
                spec="embedding_spec"
            )
        ),
        reranker_provider=(
            SimpleNamespace(
                spec="reranker_spec"
            )
        ),
        rrf_config=RRFConfig(
            rank_constant=60,
            dense_candidate_count=50,
            bm25_candidate_count=50,
        ),
        reranker_runtime_config=(
            RerankerRuntimeConfig(
                rerank_candidate_count=50,
                return_count=5,
            )
        ),
        provider_id=(
            "complex_runtime_test_v1"
        ),
    )


def test_builds_report_routes(
    tmp_path: Path,
    monkeypatch,
) -> None:
    report_ids = (
        "midea_group_2024",
        "hisense_home_2024",
    )

    routes, vector_root = (
        prepare_runtime(
            tmp_path=tmp_path,
            monkeypatch=monkeypatch,
            report_ids=report_ids,
        )
    )

    provider = build_runtime(
        routes=routes,
        vector_root=vector_root,
    )

    assert set(
        provider
        .hybrid_retrievers_by_report_id
    ) == set(report_ids)

    midea_retriever = (
        provider
        .hybrid_retrievers_by_report_id[
            "midea_group_2024"
        ]
    )

    assert midea_retriever.dense_index == (
        "dense_index_midea_group_2024"
    )

    assert midea_retriever.bm25_index == (
        "bm25_index_midea_group_2024"
    )

    assert midea_retriever.tokenizer.spec == (
        BM25TokenizerSpec()
    )


def test_rejects_duplicate_report_routes(
    tmp_path: Path,
    monkeypatch,
) -> None:
    routes, vector_root = (
        prepare_runtime(
            tmp_path=tmp_path,
            monkeypatch=monkeypatch,
        )
    )

    duplicate_routes = (
        routes[0],
        routes[0],
    )

    with pytest.raises(
        ComplexOracleRetrievalRuntimeError,
        match="重复 report_id",
    ):
        build_runtime(
            routes=duplicate_routes,
            vector_root=vector_root,
        )


def test_rejects_missing_chunk_directory(
    tmp_path: Path,
    monkeypatch,
) -> None:
    routes, vector_root = (
        prepare_runtime(
            tmp_path=tmp_path,
            monkeypatch=monkeypatch,
        )
    )

    invalid_route = (
        ComplexReportRetrievalRoute(
            report_id=(
                routes[0].report_id
            ),
            chunk_dataset_directory=(
                tmp_path / "missing_chunks"
            ),
            bm25_index_directory=(
                routes[0]
                .bm25_index_directory
            ),
        )
    )

    with pytest.raises(
        ComplexOracleRetrievalRuntimeError,
        match="ChunkDataset 目录不存在",
    ):
        build_runtime(
            routes=(invalid_route,),
            vector_root=vector_root,
        )


def test_rejects_missing_bm25_directory(
    tmp_path: Path,
    monkeypatch,
) -> None:
    routes, vector_root = (
        prepare_runtime(
            tmp_path=tmp_path,
            monkeypatch=monkeypatch,
        )
    )

    invalid_route = (
        ComplexReportRetrievalRoute(
            report_id=(
                routes[0].report_id
            ),
            chunk_dataset_directory=(
                routes[0]
                .chunk_dataset_directory
            ),
            bm25_index_directory=(
                tmp_path / "missing_bm25"
            ),
        )
    )

    with pytest.raises(
        ComplexOracleRetrievalRuntimeError,
        match="BM25 Index 目录不存在",
    ):
        build_runtime(
            routes=(invalid_route,),
            vector_root=vector_root,
        )


def test_rejects_route_report_mismatch(
    tmp_path: Path,
    monkeypatch,
) -> None:
    routes, vector_root = (
        prepare_runtime(
            tmp_path=tmp_path,
            monkeypatch=monkeypatch,
        )
    )

    route = routes[0]

    invalid_route = (
        ComplexReportRetrievalRoute(
            report_id="wrong_report_2024",
            chunk_dataset_directory=(
                route.chunk_dataset_directory
            ),
            bm25_index_directory=(
                route.bm25_index_directory
            ),
        )
    )

    with pytest.raises(
        ComplexOracleRetrievalRuntimeError,
        match=(
            "Dense Index 的 report_id "
            "与 Route 不一致"
        ),
    ):
        build_runtime(
            routes=(invalid_route,),
            vector_root=vector_root,
        )


def test_rejects_dense_bm25_identity_mismatch(
    tmp_path: Path,
    monkeypatch,
) -> None:
    routes, vector_root = (
        prepare_runtime(
            tmp_path=tmp_path,
            monkeypatch=monkeypatch,
            bm25_document_override=(
                "doc_wrong_document"
            ),
        )
    )

    with pytest.raises(
        ComplexOracleRetrievalRuntimeError,
        match="来源身份不一致",
    ):
        build_runtime(
            routes=routes,
            vector_root=vector_root,
        )