from __future__ import annotations

import re
from typing import (
    Literal,
    Self,
)

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)

from app.schemas.context_expansion import (
    AdjacentPageContextExpansion,
)


ContextExpansionGate = Literal[
    "base_resolved",
    "expansion_required",
]

ContextCandidateDecision = Literal[
    "selected",
    "item_budget",
    "char_budget",
]


class ContextBudgetPolicy(BaseModel):
    """Frozen policy for budgeted adjacent context."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    schema_version: Literal[1] = 1

    policy_id: Literal[
        "gated_lexical_adjacent_budget_v1",
        "gated_metric_aware_context_budget_v2",
    ] = (
        "gated_lexical_adjacent_budget_v1"
    )

    expand_only_when_base_unresolved: (
        Literal[True]
    ) = True

    lexical_score_version: Literal[
        "query_token_bigram_v1",
        "metric_name_query_bigram_v1",
    ] = "query_token_bigram_v1"

    max_expanded_items: int = Field(
        default=2,
        ge=1,
    )

    max_expanded_chars: int = Field(
        default=1600,
        ge=1,
    )


class ContextCandidateAudit(BaseModel):
    """Scoring and selection record for one candidate."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=False,
        allow_inf_nan=False,
    )

    selection_rank: int = Field(
        ge=1,
    )

    chunk_id: str = Field(
        pattern=(
            r"^chunk_[a-z0-9_]+_"
            r"[0-9a-f]{24}$"
        ),
    )

    pdf_page: int = Field(
        ge=1,
    )

    anchor_chunk_id: str = Field(
        pattern=(
            r"^chunk_[a-z0-9_]+_"
            r"[0-9a-f]{24}$"
        ),
    )

    anchor_retrieval_rank: int = Field(
        ge=1,
    )

    text_char_count: int = Field(
        ge=1,
    )

    token_hit_count: int = Field(
        ge=0,
    )

    bigram_overlap_count: int = Field(
        ge=0,
    )

    query_bigram_recall: float = Field(
        ge=0,
        le=1,
    )

    metric_exact_match: int = Field(
        default=0,
        ge=0,
        le=1,
    )

    metric_bigram_overlap_count: int = Field(
        default=0,
        ge=0,
    )

    metric_bigram_recall: float = Field(
        default=0.0,
        ge=0,
        le=1,
    )

    decision: ContextCandidateDecision


class BudgetedContextSelection(BaseModel):
    """
    Auditable output of gated and budgeted context
    selection.

    selected_context contains the actual context passed
    to the Fact Resolver.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=False,
        allow_inf_nan=False,
    )

    schema_version: Literal[1] = 1

    policy: ContextBudgetPolicy

    query_id: str = Field(
        pattern=r"^q[1-9][0-9]*$",
    )

    gate_decision: (
        ContextExpansionGate
    )

    # Facts already resolved from the original Top-k.
    # This must be non-empty when expansion is skipped.
    base_fact_ids: tuple[
        str,
        ...,
    ] = ()

    # Candidates are empty when the base context already
    # resolves the query because lexical selection is not
    # executed in that branch.
    candidate_item_count: int = Field(
        ge=0,
    )

    candidate_char_count: int = Field(
        ge=0,
    )

    candidates: tuple[
        ContextCandidateAudit,
        ...,
    ] = ()

    selected_expanded_chunk_ids: tuple[
        str,
        ...,
    ] = ()

    dropped_expanded_chunk_ids: tuple[
        str,
        ...,
    ] = ()

    selected_expanded_item_count: int = (
        Field(
            ge=0,
        )
    )

    selected_expanded_char_count: int = (
        Field(
            ge=0,
        )
    )

    # The final, budgeted context used by the resolver.
    selected_context: (
        AdjacentPageContextExpansion
    )

    @model_validator(mode="after")
    def validate_selection(
        self,
    ) -> Self:
        if (
            self.query_id
            != self.selected_context.query_id
        ):
            raise ValueError(
                "query_id must match "
                "selected_context"
            )

        self._validate_fact_ids()
        self._validate_candidate_order()
        self._validate_candidate_counts()
        self._validate_decision_ids()
        self._validate_selected_context()
        self._validate_gate_contract()

        return self

    def _validate_fact_ids(
        self,
    ) -> None:
        if len(self.base_fact_ids) != len(
            set(self.base_fact_ids)
        ):
            raise ValueError(
                "base_fact_ids cannot contain "
                "duplicate values"
            )

        for fact_id in self.base_fact_ids:
            if (
                re.fullmatch(
                    r"^fact_[a-z0-9_]+$",
                    fact_id,
                )
                is None
            ):
                raise ValueError(
                    "base_fact_ids contains an "
                    f"invalid ID: {fact_id}"
                )

    def _validate_candidate_order(
        self,
    ) -> None:
        expected_ranks = tuple(
            range(
                1,
                len(self.candidates) + 1,
            )
        )

        actual_ranks = tuple(
            candidate.selection_rank
            for candidate
            in self.candidates
        )

        if actual_ranks != expected_ranks:
            raise ValueError(
                "candidate selection_rank must "
                "start at 1 and increase "
                "continuously"
            )

        candidate_chunk_ids = tuple(
            candidate.chunk_id
            for candidate
            in self.candidates
        )

        if len(candidate_chunk_ids) != len(
            set(candidate_chunk_ids)
        ):
            raise ValueError(
                "candidates cannot contain "
                "duplicate chunk_id values"
            )

    def _validate_candidate_counts(
        self,
    ) -> None:
        if (
            self.candidate_item_count
            != len(self.candidates)
        ):
            raise ValueError(
                "candidate_item_count is "
                "incorrect"
            )

        expected_candidate_chars = sum(
            candidate.text_char_count
            for candidate
            in self.candidates
        )

        if (
            self.candidate_char_count
            != expected_candidate_chars
        ):
            raise ValueError(
                "candidate_char_count is "
                "incorrect"
            )

    def _validate_decision_ids(
        self,
    ) -> None:
        expected_selected_ids = tuple(
            candidate.chunk_id
            for candidate
            in self.candidates
            if candidate.decision
            == "selected"
        )

        expected_dropped_ids = tuple(
            candidate.chunk_id
            for candidate
            in self.candidates
            if candidate.decision
            != "selected"
        )

        if (
            self.selected_expanded_chunk_ids
            != expected_selected_ids
        ):
            raise ValueError(
                "selected_expanded_chunk_ids "
                "does not match candidate "
                "decisions"
            )

        if (
            self.dropped_expanded_chunk_ids
            != expected_dropped_ids
        ):
            raise ValueError(
                "dropped_expanded_chunk_ids "
                "does not match candidate "
                "decisions"
            )

        if set(
            self.selected_expanded_chunk_ids
        ).intersection(
            self.dropped_expanded_chunk_ids
        ):
            raise ValueError(
                "selected and dropped chunk IDs "
                "cannot overlap"
            )

    def _validate_selected_context(
        self,
    ) -> None:
        if (
            self.selected_context
            .expanded_chunk_ids
            != self.selected_expanded_chunk_ids
        ):
            raise ValueError(
                "selected_context expanded IDs "
                "must match selected candidate IDs"
            )

        if (
            self.selected_expanded_item_count
            != len(
                self
                .selected_expanded_chunk_ids
            )
        ):
            raise ValueError(
                "selected_expanded_item_count "
                "is incorrect"
            )

        selected_items = tuple(
            item
            for item
            in self.selected_context.items
            if (
                item.origin != "retrieved"
            )
        )

        expected_selected_chars = sum(
            item.text_char_count
            for item in selected_items
        )

        if (
            self.selected_expanded_char_count
            != expected_selected_chars
        ):
            raise ValueError(
                "selected_expanded_char_count "
                "is incorrect"
            )

        if (
            self.selected_expanded_item_count
            > self.policy.max_expanded_items
        ):
            raise ValueError(
                "selected item count exceeds "
                "policy budget"
            )

        if (
            self.selected_expanded_char_count
            > self.policy.max_expanded_chars
        ):
            raise ValueError(
                "selected char count exceeds "
                "policy budget"
            )

    def _validate_gate_contract(
        self,
    ) -> None:
        if (
            self.gate_decision
            == "base_resolved"
        ):
            if not self.base_fact_ids:
                raise ValueError(
                    "base_resolved requires "
                    "base_fact_ids"
                )

            if self.candidates:
                raise ValueError(
                    "base_resolved cannot contain "
                    "scored candidates"
                )

            if (
                self.candidate_item_count != 0
                or self.candidate_char_count != 0
            ):
                raise ValueError(
                    "base_resolved candidate "
                    "counts must be zero"
                )

            if (
                self.selected_expanded_chunk_ids
                or self.dropped_expanded_chunk_ids
            ):
                raise ValueError(
                    "base_resolved cannot select "
                    "or drop expanded chunks"
                )

            if (
                self.selected_context
                .expanded_item_count
                != 0
            ):
                raise ValueError(
                    "base_resolved context cannot "
                    "contain expanded items"
                )

            return

        if self.base_fact_ids:
            raise ValueError(
                "expansion_required cannot "
                "contain base_fact_ids"
            )
