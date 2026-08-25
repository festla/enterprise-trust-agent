from __future__ import annotations

from dataclasses import (
    dataclass,
    field,
)
from pathlib import Path

from app.rag.competition_bm25 import (
    CompetitionExactBM25Index,
)
from app.rag.tokenization import (
    DeterministicChineseBigramTokenizer,
)
from app.schemas.competition_chunk import (
    CompetitionTextChunk,
)
from app.schemas.competition_corpus import (
    CompetitionCorpusSourceRecord,
)
from app.services.competition_bm25_index import (
    load_competition_bm25_index,
)
from app.services.competition_corpus import (
    load_competition_chunk_corpus,
)
from app.services.competition_evidence_assembler import (
    CompetitionEvidenceAssemblyConfig,
)
from app.services.competition_source_resolver import (
    CompetitionSourceResolver,
    build_competition_source_manifest,
)
from app.services.competition_text_retrieval import (
    CompetitionTextRetrievalConfig,
)
from app.rag.competition_answer_generation import (
    CompetitionAnswerProvider,
)
from app.rag.competition_bailian_answer_provider import (
    CompetitionBailianOpenAIAnswerProvider,
)
from app.rag.competition_bailian_sufficiency_provider import (
    CompetitionBailianSufficiencyProvider,
)
from app.rag.competition_sufficiency import (
    CompetitionEvidenceSufficiencyProvider,
)

def _build_frozen_evidence_config(
) -> CompetitionEvidenceAssemblyConfig:
    """
    Week8 7A Frozen Evidence Budget。
    """

    return (
        CompetitionEvidenceAssemblyConfig(
            max_evidences=48,
            max_chars=4000,
        )
    )


@dataclass(
    frozen=True,
    slots=True,
)
class CompetitionAgentRuntimeResources:
    """
    Competition Agent 的进程级运行资源。

    这些对象不属于单个 Question State，
    可以被多个 Graph Invocation 复用。
    """

    source_resolver: (
        CompetitionSourceResolver
    )

    bm25_index: (
        CompetitionExactBM25Index
    )

    tokenizer: (
        DeterministicChineseBigramTokenizer
    )

    corpus_chunks: tuple[
        CompetitionTextChunk,
        ...,
    ]

    source_records: tuple[
        CompetitionCorpusSourceRecord,
        ...,
    ]

    retrieval_config: (
        CompetitionTextRetrievalConfig
    ) = field(
        default_factory=(
            CompetitionTextRetrievalConfig
        )
    )

    evidence_config: (
        CompetitionEvidenceAssemblyConfig
    ) = field(
        default_factory=(
            _build_frozen_evidence_config
        )
    )


def load_competition_agent_runtime_resources(
    *,
    attachments_root: Path,
    corpus_directory: Path,
    index_directory: Path,
) -> CompetitionAgentRuntimeResources:
    """
    从 Frozen Artifacts 加载一次 Agent Runtime。

    加载阶段完成：
        attachments manifest
        source resolver
        frozen corpus
        frozen BM25 index
        tokenizer

    Graph Invocation 时不再重复加载。
    """

    source_manifest = (
        build_competition_source_manifest(
            attachments_root
        )
    )

    source_resolver = (
        CompetitionSourceResolver(
            source_manifest
        )
    )

    corpus = (
        load_competition_chunk_corpus(
            corpus_directory
        )
    )

    index_result = (
        load_competition_bm25_index(
            index_directory,
            corpus_directory=(
                corpus_directory
            ),
        )
    )

    tokenizer = (
        DeterministicChineseBigramTokenizer(
            spec=(
                index_result
                .manifest
                .tokenizer_spec
            )
        )
    )

    #
    # Runtime invariant 1:
    # BM25 与 Corpus 加载绑定已经由
    # load_competition_bm25_index(...)
    # 完成严格校验。
    #

    #
    # Runtime invariant 2:
    # Corpus 中的 source 必须能在
    # attachments manifest 中定位。
    #
    manifest_source_ids = {
        record.source_id
        for record
        in source_manifest
    }

    corpus_source_ids = {
        record.source_id
        for record
        in corpus.manifest.source_records
    }

    missing_sources = (
        corpus_source_ids
        - manifest_source_ids
    )

    if missing_sources:
        raise RuntimeError(
            "Frozen Corpus 包含无法在 "
            "attachments manifest 中定位的来源："
            f"{tuple(sorted(missing_sources))}"
        )

    return (
        CompetitionAgentRuntimeResources(
            source_resolver=(
                source_resolver
            ),
            bm25_index=(
                index_result.index
            ),
            tokenizer=tokenizer,
            corpus_chunks=(
                corpus.chunks
            ),
            source_records=(
                corpus
                .manifest
                .source_records
            ),
        )
    )

@dataclass(
    frozen=True,
    slots=True,
)
class CompetitionAgentModelResources:
    """
    Agent 的模型级进程资源。

    Provider 属于 Process Resource，
    不应该写入 Graph State。
    """

    sufficiency_provider: (
        CompetitionEvidenceSufficiencyProvider
    )

    answer_provider: (
        CompetitionAnswerProvider
    )


def load_competition_agent_model_resources_from_environment(
) -> CompetitionAgentModelResources:
    """
    从环境变量加载真实百炼 Provider。

    当前 Frozen Model：
        qwen3.8-max
    """

    return CompetitionAgentModelResources(
        sufficiency_provider=(
            CompetitionBailianSufficiencyProvider
            .from_environment()
        ),
        answer_provider=(
            CompetitionBailianOpenAIAnswerProvider
            .from_environment()
        ),
    )