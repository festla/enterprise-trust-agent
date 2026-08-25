from __future__ import annotations

from collections.abc import (
    Sequence,
)
from dataclasses import (
    dataclass,
    field,
)

from app.rag.competition_bm25 import (
    CompetitionExactBM25Index,
)
from app.rag.tokenization import (
    BM25Tokenizer,
)
from app.schemas.competition import (
    CompetitionQuestion,
)
from app.schemas.competition_chunk import (
    CompetitionTextChunk,
)
from app.schemas.competition_context_expansion import (
    CompetitionContextExpandedHit,
    CompetitionContextExpansionConfig,
)
from app.schemas.competition_option_anchored_merge import (
    CompetitionBM25OptionAnchoredMergeConfig,
    CompetitionBM25OptionAnchoredMergeHit,
)
from app.schemas.competition_retrieval import (
    CompetitionBM25Hit,
    CompetitionRetrievalFilter,
)
from app.services.competition_bm25_option_anchored_merge import (
    merge_competition_bm25_option_anchored,
)
from app.services.competition_context_expansion import (
    expand_competition_retrieval_context,
)


def _build_frozen_context_config(
) -> CompetitionContextExpansionConfig:
    """
    Week8 Frozen Retrieval Config。

    来自 6B Retrieval Acceptance：
        Top10 seeds
        ±1 neighbor
        child only
        depth <= 2
        seed_and_adjacent
    """

    return CompetitionContextExpansionConfig(
        seed_candidate_count=10,
        previous_window=1,
        next_window=1,
        enable_item_hierarchy=True,
        enable_item_parent=False,
        enable_item_child=True,
        max_item_depth_delta=2,
        item_hierarchy_trigger_mode=(
            "seed_and_adjacent"
        ),
    )


@dataclass(
    frozen=True,
    slots=True,
)
class CompetitionTextRetrievalConfig:
    """
    Production Text Retrieval 配置。

    默认值就是已经冻结的 Week8 策略，
    而不是实验期默认参数。
    """

    raw_top_k: int = 10
    merged_top_k: int = 10

    option_merge_config: (
        CompetitionBM25OptionAnchoredMergeConfig
    ) = field(
        default_factory=(
            CompetitionBM25OptionAnchoredMergeConfig
        )
    )

    context_expansion_config: (
        CompetitionContextExpansionConfig
    ) = field(
        default_factory=(
            _build_frozen_context_config
        )
    )

    def __post_init__(
        self,
    ) -> None:
        if self.raw_top_k < 1:
            raise ValueError(
                "raw_top_k 必须 >= 1"
            )

        if self.merged_top_k < 1:
            raise ValueError(
                "merged_top_k 必须 >= 1"
            )


@dataclass(
    frozen=True,
    slots=True,
)
class CompetitionTextRetrievalResult:
    """
    一次 Text Retrieval 的完整可审计结果。

    注意：
    不只保留最终 expanded hits，
    同时保留两路查询和 Seed，
    方便后续 Agent Trace。
    """

    question_only_query: str
    question_with_options_query: str

    question_only_hits: tuple[
        CompetitionBM25Hit,
        ...,
    ]

    question_with_options_hits: tuple[
        CompetitionBM25Hit,
        ...,
    ]

    seed_hits: tuple[
        CompetitionBM25OptionAnchoredMergeHit,
        ...,
    ]

    expanded_hits: tuple[
        CompetitionContextExpandedHit,
        ...,
    ]

    @property
    def retrieval_hit_count(
        self,
    ) -> int:
        return len(
            self.expanded_hits
        )


def build_competition_question_only_query(
    question: CompetitionQuestion,
) -> str:
    return question.question


def build_competition_question_with_options_query(
    question: CompetitionQuestion,
) -> str:
    option_lines = tuple(
        f"{option}. {text}"
        for option, text
        in question.options.items()
    )

    return "\n".join(
        (
            question.question,
            *option_lines,
        )
    )


def retrieve_competition_text(
    *,
    question: CompetitionQuestion,
    resolved_source_id: str,
    index: CompetitionExactBM25Index,
    tokenizer: BM25Tokenizer,
    corpus_chunks: Sequence[
        CompetitionTextChunk
    ],
    config: (
        CompetitionTextRetrievalConfig
        | None
    ) = None,
) -> CompetitionTextRetrievalResult:
    """
    Competition Text Runtime Retrieval V1。

    Pipeline：

        CompetitionQuestion
              ↓
        source scoped BM25
           ↙       ↘
        question  question+options
           ↘       ↙
        Option-Anchored Merge
              ↓
          Top10 Seeds
              ↓
        Structured Expansion
              ↓
          Expanded Hits

    这里完全不接受 Gold 数据。
    """

    if (
        question.source_type
        not in {
            "word",
            "pdf",
        }
    ):
        raise ValueError(
            "Text Retrieval 只支持 "
            "word / pdf Question"
        )

    if not resolved_source_id.strip():
        raise ValueError(
            "resolved_source_id 不能为空"
        )

    active_config = (
        config
        if config is not None
        else CompetitionTextRetrievalConfig()
    )

    question_only_query = (
        build_competition_question_only_query(
            question
        )
    )

    question_with_options_query = (
        build_competition_question_with_options_query(
            question
        )
    )

    source_filter = (
        CompetitionRetrievalFilter(
            source_ids=(
                resolved_source_id,
            )
        )
    )

    question_only_hits = (
        index.search(
            query=question_only_query,
            tokenizer=tokenizer,
            top_k=(
                active_config
                .raw_top_k
            ),
            filters=source_filter,
        )
    )

    question_with_options_hits = (
        index.search(
            query=(
                question_with_options_query
            ),
            tokenizer=tokenizer,
            top_k=(
                active_config
                .raw_top_k
            ),
            filters=source_filter,
        )
    )

    # 没有任何候选属于正常业务结果，
    # 后续 Agent Readiness Gate 会决定拒答。
    if (
        not question_only_hits
        and not question_with_options_hits
    ):
        return CompetitionTextRetrievalResult(
            question_only_query=(
                question_only_query
            ),
            question_with_options_query=(
                question_with_options_query
            ),
            question_only_hits=(),
            question_with_options_hits=(),
            seed_hits=(),
            expanded_hits=(),
        )

    seed_hits = (
        merge_competition_bm25_option_anchored(
            question_only_hits=(
                question_only_hits
            ),
            question_with_options_hits=(
                question_with_options_hits
            ),
            config=(
                active_config
                .option_merge_config
            ),
            top_k=(
                active_config
                .merged_top_k
            ),
        )
    )

    expanded_hits = (
        expand_competition_retrieval_context(
            seed_hits=seed_hits,
            corpus_chunks=(
                corpus_chunks
            ),
            config=(
                active_config
                .context_expansion_config
            ),
        )
    )

    # Runtime invariant：
    # Source-scoped Retrieval 不允许泄漏其他附件。
    foreign_sources = {
        hit.source_id
        for hit in expanded_hits
        if (
            hit.source_id
            != resolved_source_id
        )
    }

    if foreign_sources:
        raise RuntimeError(
            "Source-scoped Retrieval "
            "出现跨来源 Evidence："
            f"{tuple(sorted(foreign_sources))}"
        )

    return CompetitionTextRetrievalResult(
        question_only_query=(
            question_only_query
        ),
        question_with_options_query=(
            question_with_options_query
        ),
        question_only_hits=(
            question_only_hits
        ),
        question_with_options_hits=(
            question_with_options_hits
        ),
        seed_hits=seed_hits,
        expanded_hits=(
            expanded_hits
        ),
    )