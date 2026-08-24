from __future__ import annotations

import re
from typing import Protocol

from app.schemas.competition import (
    CompetitionQuestion,
)
from app.schemas.competition_answer import (
    CompetitionFinalAnswer,
    CompetitionGeneratedAnswer,
)
from app.schemas.competition_evidence import (
    CompetitionEvidence,
    CompetitionEvidenceBundle,
    CompetitionExcelEvidenceLocation,
    CompetitionTextEvidenceLocation,
)


class CompetitionAnswerGenerationError(
    ValueError
):
    """Competition 回答生成基础异常。"""


class UnauthorizedCompetitionCitationError(
    CompetitionAnswerGenerationError
):
    """模型引用了未授权 Evidence。"""


class MissingCompetitionInlineCitationError(
    CompetitionAnswerGenerationError
):
    """回答正文缺少合法 [E1] 引用。"""


class InvalidCompetitionAnswerProviderError(
    CompetitionAnswerGenerationError
):
    """Answer Provider 配置无效。"""


class CompetitionAnswerProvider(
    Protocol
):
    @property
    def provider_id(
        self,
    ) -> str:
        """返回 Provider / Model 可审计标识。"""

    def generate(
        self,
        *,
        question: str,
        options: dict[str, str],
        generation_context: str,
        allowed_citation_ids: tuple[
            str,
            ...
        ],
    ) -> CompetitionGeneratedAnswer:
        """
        只允许根据给定 Question /
        Options / Evidence Context 作答。
        """


_INLINE_CITATION_PATTERN = re.compile(
    r"\[(E[1-9][0-9]*)\]"
)


def _format_location(
    evidence: CompetitionEvidence,
) -> str:
    location = evidence.location

    if isinstance(
        location,
        CompetitionTextEvidenceLocation,
    ):
        parts = []

        if location.page is not None:
            parts.append(
                f"page={location.page}"
            )

        if location.section:
            parts.append(
                f"section={location.section}"
            )

        if location.article:
            parts.append(
                f"article={location.article}"
            )

        if (
            location.paragraph_index
            is not None
        ):
            parts.append(
                "paragraph="
                f"{location.paragraph_index}"
            )

        return (
            "; ".join(parts)
            if parts
            else "text"
        )

    if isinstance(
        location,
        CompetitionExcelEvidenceLocation,
    ):
        parts = [
            f"sheet={location.sheet_name}"
        ]

        if location.cell:
            parts.append(
                f"cell={location.cell}"
            )

        if location.cell_range:
            parts.append(
                "range="
                f"{location.cell_range}"
            )

        return "; ".join(
            parts
        )

    raise TypeError(
        "未知 Evidence Location"
    )


def build_competition_generation_context(
    bundle: CompetitionEvidenceBundle,
) -> tuple[
    str,
    dict[
        str,
        CompetitionEvidence,
    ],
]:
    """
    把 EvidenceBundle 转换成模型上下文。

    内部稳定 evidence_id 不直接要求模型记忆，
    对模型统一暴露 E1 / E2 / ...。
    """

    citation_map: dict[
        str,
        CompetitionEvidence,
    ] = {}

    blocks = []

    for index, evidence in enumerate(
        bundle.evidences,
        start=1,
    ):
        citation_id = (
            f"E{index}"
        )

        citation_map[
            citation_id
        ] = evidence

        blocks.append(
            "\n".join(
                (
                    f"[{citation_id}]",
                    (
                        "source: "
                        f"{evidence.source.title}"
                    ),
                    (
                        "location: "
                        f"{_format_location(evidence)}"
                    ),
                    "content:",
                    evidence.raw_content,
                )
            )
        )

    if not blocks:
        raise (
            CompetitionAnswerGenerationError(
                "EvidenceBundle 不能为空"
            )
        )

    context = "\n\n".join(
        blocks
    )

    return (
        context,
        citation_map,
    )


def _validate_generated_citations(
    *,
    generated: CompetitionGeneratedAnswer,
    allowed_citation_ids: tuple[
        str,
        ...
    ],
) -> tuple[str, ...]:
    """
    校验模型生成引用，并返回最终 canonical citation IDs。

    规则：
    1. generated.citation_ids 不能引用白名单外 Evidence；
    2. answer_text 必须至少包含一个 [E#]；
    3. inline citation 也必须全部属于白名单；
    4. 最终实际引用以 answer_text 中出现的 inline citation
       为准，并按首次出现顺序去重。

    原因：
    模型可能在结构化 citation_ids 中多列或漏列一个
    已授权引用，但真正面向用户的事实归因由正文中的
    inline citations 决定。
    """

    allowed_set = set(
        allowed_citation_ids
    )

    generated_set = set(
        generated.citation_ids
    )

    unauthorized_generated = (
        generated_set
        - allowed_set
    )

    if unauthorized_generated:
        raise (
            UnauthorizedCompetitionCitationError(
                "生成答案 citation_ids "
                "包含未授权 Evidence："
                + ", ".join(
                    sorted(
                        unauthorized_generated
                    )
                )
            )
        )

    inline_citation_ids = tuple(
        _INLINE_CITATION_PATTERN.findall(
            generated.answer_text
        )
    )

    if not inline_citation_ids:
        raise (
            MissingCompetitionInlineCitationError(
                "answer_text 必须包含 "
                "[E1] 形式引用"
            )
        )

    inline_set = set(
        inline_citation_ids
    )

    unauthorized_inline = (
        inline_set
        - allowed_set
    )

    if unauthorized_inline:
        raise (
            UnauthorizedCompetitionCitationError(
                "answer_text 引用了"
                "未授权 Evidence："
                + ", ".join(
                    sorted(
                        unauthorized_inline
                    )
                )
            )
        )

    canonical_ids: list[str] = []

    seen: set[str] = set()

    for citation_id in inline_citation_ids:
        if citation_id in seen:
            continue

        seen.add(
            citation_id
        )

        canonical_ids.append(
            citation_id
        )

    return tuple(
        canonical_ids
    )


def generate_competition_answer(
    *,
    question: CompetitionQuestion,
    evidence_bundle: (
        CompetitionEvidenceBundle
    ),
    provider: CompetitionAnswerProvider,
) -> CompetitionFinalAnswer:
    """
    Competition 文本 QA Answer Generation V1。

    关键约束：
    - 不接受 CompetitionQaCase，
      防止 answer/evidence Gold 泄漏；
    - Provider 只能看到 Evidence Pack；
    - 模型只能引用白名单 E1/E2/...；
    - 最终保留 citation -> evidence 映射。
    """

    provider_id = (
        provider.provider_id.strip()
    )

    if not provider_id:
        raise (
            InvalidCompetitionAnswerProviderError(
                "provider_id 不能为空"
            )
        )

    (
        generation_context,
        citation_map,
    ) = (
        build_competition_generation_context(
            evidence_bundle
        )
    )

    allowed_citation_ids = tuple(
        citation_map
    )

    generated = provider.generate(
        question=question.question,
        options=question.options,
        generation_context=(
            generation_context
        ),
        allowed_citation_ids=(
            allowed_citation_ids
        ),
    )

    canonical_citation_ids = (
        _validate_generated_citations(
            generated=generated,
            allowed_citation_ids=(
                allowed_citation_ids
            ),
        )
    )

    evidence_ids = tuple(
        citation_map[
            citation_id
        ].evidence_id
        for citation_id
        in canonical_citation_ids
    )

    return CompetitionFinalAnswer(
        case_id=question.case_id,
        answer=generated.answer,
        answer_text=(
            generated.answer_text
        ),
        citation_ids=(
            canonical_citation_ids
        ),
        evidence_ids=(
            evidence_ids
        ),
        generator_id=provider_id,
    )