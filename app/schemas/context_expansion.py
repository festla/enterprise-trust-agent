from __future__ import annotations

from typing import Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)

from app.schemas.enums import ReportType


ContextItemOrigin = Literal[
    "retrieved",
    "adjacent_page",
    "same_page_sibling",
]


class ContextExpansionItem(BaseModel):
    """
    A chunk used by the context expansion experiment.

    retrieved:
        A chunk returned directly by the base retriever.

    adjacent_page:
        A chunk added because its PDF page is adjacent to a
        retrieved anchor chunk.

    same_page_sibling:
        A non-retrieved chunk from the same PDF page as a
        retrieved anchor chunk.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=False,
    )

    context_order: int = Field(
        ge=1,
    )

    origin: ContextItemOrigin

    company_id: str = Field(
        pattern=r"^[a-z0-9_]+$",
    )

    report_id: str = Field(
        pattern=r"^[a-z0-9_]+$",
    )

    fiscal_year: int = Field(
        ge=2000,
        le=2100,
    )

    report_type: ReportType

    document_id: str = Field(
        pattern=(
            r"^doc_[a-z0-9_]+_"
            r"[0-9a-f]{24}$"
        ),
    )

    page_id: str = Field(
        pattern=(
            r"^doc_[a-z0-9_]+_"
            r"[0-9a-f]{24}_page_"
            r"[0-9]{4,}$"
        ),
    )

    pdf_page: int = Field(
        ge=1,
    )

    printed_page: int | None = Field(
        default=None,
        ge=1,
    )

    chunk_id: str = Field(
        pattern=(
            r"^chunk_[a-z0-9_]+_"
            r"[0-9a-f]{24}$"
        ),
    )

    chunk_index: int = Field(
        ge=0,
    )

    text: str = Field(
        min_length=1,
    )

    text_char_count: int = Field(
        ge=1,
    )

    # Only a directly retrieved item has its own retrieval
    # rank and retrieval score.
    retrieval_rank: int | None = Field(
        default=None,
        ge=1,
    )

    retrieval_score: float | None = Field(
        default=None,
        allow_inf_nan=False,
    )

    # Both base and expanded chunks point to the base chunk
    # responsible for their inclusion.
    anchor_chunk_id: str = Field(
        pattern=(
            r"^chunk_[a-z0-9_]+_"
            r"[0-9a-f]{24}$"
        ),
    )

    anchor_retrieval_rank: int = Field(
        ge=1,
    )

    # 0 for a base hit or same-page sibling, -1 or +1 for
    # an adjacent-page item.
    page_distance: int = Field(
        ge=-1,
        le=1,
    )

    @model_validator(mode="after")
    def validate_item_contract(
        self,
    ) -> Self:
        expected_page_id = (
            f"{self.document_id}_page_"
            f"{self.pdf_page:04d}"
        )

        if self.page_id != expected_page_id:
            raise ValueError(
                "page_id must match document_id "
                "and pdf_page"
            )

        if not self.document_id.startswith(
            f"doc_{self.report_id}_"
        ):
            raise ValueError(
                "document_id must belong to "
                "report_id"
            )

        if self.text_char_count != len(
            self.text
        ):
            raise ValueError(
                "text_char_count must equal "
                "the actual text length"
            )

        if self.origin == "retrieved":
            if self.retrieval_rank is None:
                raise ValueError(
                    "retrieved item must contain "
                    "retrieval_rank"
                )

            if self.retrieval_score is None:
                raise ValueError(
                    "retrieved item must contain "
                    "retrieval_score"
                )

            if (
                self.anchor_chunk_id
                != self.chunk_id
            ):
                raise ValueError(
                    "retrieved item must anchor "
                    "to itself"
                )

            if (
                self.anchor_retrieval_rank
                != self.retrieval_rank
            ):
                raise ValueError(
                    "retrieved item anchor rank "
                    "must equal retrieval_rank"
                )

            if self.page_distance != 0:
                raise ValueError(
                    "retrieved item page_distance "
                    "must be 0"
                )

            return self

        if self.retrieval_rank is not None:
            raise ValueError(
                "adjacent item cannot contain "
                "retrieval_rank"
            )

        if self.retrieval_score is not None:
            raise ValueError(
                "adjacent item cannot contain "
                "retrieval_score"
            )

        if (
            self.anchor_chunk_id
            == self.chunk_id
        ):
            raise ValueError(
                "adjacent item cannot anchor "
                "to itself"
            )

        if (
            self.origin == "adjacent_page"
            and self.page_distance == 0
        ):
            raise ValueError(
                "adjacent item page_distance "
                "cannot be 0"
            )

        if (
            self.origin == "same_page_sibling"
            and self.page_distance != 0
        ):
            raise ValueError(
                "same-page sibling page_distance "
                "must be 0"
            )

        return self


class AdjacentPageContextExpansion(
    BaseModel
):
    """
    Auditable output of one adjacent-page context expansion.

    The first experiment is deliberately frozen to a one-page
    window so that the result is reproducible and comparable.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=False,
    )

    schema_version: Literal[1] = 1

    strategy_id: Literal[
        "adjacent_page_context_v1",
        "same_and_adjacent_page_context_v2",
    ] = "adjacent_page_context_v1"

    query_id: str = Field(
        pattern=r"^q[1-9][0-9]*$",
    )

    original_query: str = Field(
        min_length=1,
    )

    semantic_query: str = Field(
        min_length=1,
    )

    company_id: str = Field(
        pattern=r"^[a-z0-9_]+$",
    )

    report_id: str = Field(
        pattern=r"^[a-z0-9_]+$",
    )

    fiscal_year: int = Field(
        ge=2000,
        le=2100,
    )

    report_type: ReportType

    document_id: str = Field(
        pattern=(
            r"^doc_[a-z0-9_]+_"
            r"[0-9a-f]{24}$"
        ),
    )

    base_top_k: int = Field(
        ge=1,
    )

    # Version 1 only tests immediately adjacent pages.
    page_window: Literal[1] = 1

    items: tuple[
        ContextExpansionItem,
        ...,
    ] = Field(
        min_length=1,
    )

    base_chunk_ids: tuple[
        str,
        ...,
    ] = Field(
        min_length=1,
    )

    expanded_chunk_ids: tuple[
        str,
        ...,
    ] = ()

    used_chunk_ids: tuple[
        str,
        ...,
    ] = Field(
        min_length=1,
    )

    base_item_count: int = Field(
        ge=1,
    )

    expanded_item_count: int = Field(
        ge=0,
    )

    total_item_count: int = Field(
        ge=1,
    )

    duplicate_chunk_count: int = Field(
        ge=0,
    )

    base_char_count: int = Field(
        ge=1,
    )

    expanded_char_count: int = Field(
        ge=0,
    )

    total_char_count: int = Field(
        ge=1,
    )

    @model_validator(mode="after")
    def validate_expansion(
        self,
    ) -> Self:
        expected_orders = tuple(
            range(
                1,
                len(self.items) + 1,
            )
        )

        actual_orders = tuple(
            item.context_order
            for item in self.items
        )

        if actual_orders != expected_orders:
            raise ValueError(
                "context_order must start at 1 "
                "and increase continuously"
            )

        item_chunk_ids = tuple(
            item.chunk_id
            for item in self.items
        )

        if len(item_chunk_ids) != len(
            set(item_chunk_ids)
        ):
            raise ValueError(
                "context items must not contain "
                "duplicate chunk_id values"
            )

        if (
            self.used_chunk_ids
            != item_chunk_ids
        ):
            raise ValueError(
                "used_chunk_ids must match "
                "the item order"
            )

        expected_base_chunk_ids = tuple(
            item.chunk_id
            for item in self.items
            if item.origin == "retrieved"
        )

        expected_expanded_chunk_ids = tuple(
            item.chunk_id
            for item in self.items
            if item.origin != "retrieved"
        )

        if (
            self.base_chunk_ids
            != expected_base_chunk_ids
        ):
            raise ValueError(
                "base_chunk_ids must match "
                "retrieved items"
            )

        if (
            self.expanded_chunk_ids
            != expected_expanded_chunk_ids
        ):
            raise ValueError(
                "expanded_chunk_ids must match "
                "adjacent-page items"
            )

        if self.base_item_count != len(
            expected_base_chunk_ids
        ):
            raise ValueError(
                "base_item_count is incorrect"
            )

        if self.expanded_item_count != len(
            expected_expanded_chunk_ids
        ):
            raise ValueError(
                "expanded_item_count is "
                "incorrect"
            )

        if self.total_item_count != len(
            self.items
        ):
            raise ValueError(
                "total_item_count is incorrect"
            )

        if (
            self.total_item_count
            != self.base_item_count
            + self.expanded_item_count
        ):
            raise ValueError(
                "total item counts are "
                "inconsistent"
            )

        if (
            self.base_item_count
            > self.base_top_k
        ):
            raise ValueError(
                "base_item_count cannot exceed "
                "base_top_k"
            )

        base_items = {
            item.chunk_id: item
            for item in self.items
            if item.origin == "retrieved"
        }

        base_ranks = tuple(
            item.retrieval_rank
            for item in self.items
            if item.origin == "retrieved"
        )

        if base_ranks != tuple(
            sorted(base_ranks)
        ):
            raise ValueError(
                "retrieved items must preserve "
                "retrieval rank order"
            )

        for item in self.items:
            if (
                item.company_id
                != self.company_id
            ):
                raise ValueError(
                    "item company_id does not "
                    "match expansion identity"
                )

            if (
                item.report_id
                != self.report_id
            ):
                raise ValueError(
                    "item report_id does not "
                    "match expansion identity"
                )

            if (
                item.fiscal_year
                != self.fiscal_year
            ):
                raise ValueError(
                    "item fiscal_year does not "
                    "match expansion identity"
                )

            if (
                item.report_type
                is not self.report_type
            ):
                raise ValueError(
                    "item report_type does not "
                    "match expansion identity"
                )

            if (
                item.document_id
                != self.document_id
            ):
                raise ValueError(
                    "item document_id does not "
                    "match expansion identity"
                )

            if item.origin == "retrieved":
                continue

            anchor = base_items.get(
                item.anchor_chunk_id
            )

            if anchor is None:
                raise ValueError(
                    "adjacent item must reference "
                    "a retrieved anchor chunk"
                )

            if (
                item.anchor_retrieval_rank
                != anchor.retrieval_rank
            ):
                raise ValueError(
                    "adjacent item anchor rank "
                    "does not match its anchor"
                )

            expected_page_distance = (
                item.pdf_page
                - anchor.pdf_page
            )

            if (
                item.page_distance
                != expected_page_distance
            ):
                raise ValueError(
                    "page_distance must match "
                    "the anchor and item pages"
                )

            if (
                abs(item.page_distance)
                > self.page_window
            ):
                raise ValueError(
                    "adjacent item exceeds "
                    "page_window"
                )

        expected_base_char_count = sum(
            item.text_char_count
            for item in self.items
            if item.origin == "retrieved"
        )

        expected_expanded_char_count = sum(
            item.text_char_count
            for item in self.items
            if item.origin != "retrieved"
        )

        if (
            self.base_char_count
            != expected_base_char_count
        ):
            raise ValueError(
                "base_char_count is incorrect"
            )

        if (
            self.expanded_char_count
            != expected_expanded_char_count
        ):
            raise ValueError(
                "expanded_char_count is "
                "incorrect"
            )

        if (
            self.total_char_count
            != self.base_char_count
            + self.expanded_char_count
        ):
            raise ValueError(
                "total_char_count is incorrect"
            )

        return self
