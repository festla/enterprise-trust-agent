from __future__ import annotations

from collections.abc import (
    Sequence,
)
from pathlib import Path
from typing import (
    Annotated,
    TypedDict,
)

import pytest
from langgraph.checkpoint.memory import (
    InMemorySaver,
)
from langgraph.graph import (
    END,
    START,
    StateGraph,
)

from app.schemas.competition import (
    CompetitionQuestion,
)
from app.workflow.competition_checkpoint import (
    build_competition_checkpoint_serializer,
    open_competition_sqlite_checkpointer,
)
from app.workflow.competition_state import (
    merge_competition_trace,
)


# ============================================================
# Helpers
# ============================================================


def _merge_steps(
    left: (
        Sequence[str]
        | None
    ),
    right: (
        Sequence[str]
        | None
    ),
) -> tuple[str, ...]:
    """
    测试专用 reducer。

    故意使用与 Competition Trace
    相同的 persistence-safe 模式。
    """

    return (
        tuple(left or ())
        + tuple(right or ())
    )


class ResumeState(
    TypedDict,
    total=False,
):
    steps: Annotated[
        tuple[str, ...],
        _merge_steps,
    ]


class ThreadState(
    TypedDict,
    total=False,
):
    case_id: str
    result: str


class QuestionState(
    TypedDict,
    total=False,
):
    question: (
        CompetitionQuestion
    )


class InjectedResumeFailure(
    RuntimeError
):
    pass


# ============================================================
# 1. Trace Reducer
# ============================================================


@pytest.mark.parametrize(
    (
        "left",
        "right",
        "expected",
    ),
    [
        (
            (),
            (),
            (),
        ),
        (
            [],
            (),
            (),
        ),
        (
            ("a",),
            ("b",),
            (
                "a",
                "b",
            ),
        ),
        (
            ["a"],
            ("b",),
            (
                "a",
                "b",
            ),
        ),
        (
            ("a",),
            ["b"],
            (
                "a",
                "b",
            ),
        ),
        (
            ["a"],
            ["b"],
            (
                "a",
                "b",
            ),
        ),
    ],
)
def test_merge_competition_trace_accepts_checkpoint_sequences(
    left,
    right,
    expected,
) -> None:
    """
    Checkpoint serializer 可能把 tuple
    恢复成 list。

    reducer 必须保证：
        list + tuple
        tuple + list
        list + list
        tuple + tuple

    全部稳定工作。
    """

    actual = (
        merge_competition_trace(
            left,
            right,
        )
    )

    assert actual == expected

    assert isinstance(
        actual,
        tuple,
    )


# ============================================================
# 2. Strict serializer
# ============================================================


def test_strict_checkpoint_serializer_restores_competition_question(
    monkeypatch,
) -> None:
    """
    验证 Competition 自定义类型可以在
    strict msgpack 模式下经过 checkpoint
    serialize -> deserialize。

    不依赖 LangGraph 对未知 Python 类型的
    permissive fallback。
    """

    monkeypatch.setenv(
        "LANGGRAPH_STRICT_MSGPACK",
        "true",
    )

    serializer = (
        build_competition_checkpoint_serializer()
    )

    checkpointer = InMemorySaver(
        serde=serializer,
    )

    builder = StateGraph(
        QuestionState
    )

    def noop_node(
        state: QuestionState,
    ) -> dict:
        return {}

    builder.add_node(
        "noop",
        noop_node,
    )

    builder.add_edge(
        START,
        "noop",
    )

    builder.add_edge(
        "noop",
        END,
    )

    graph = builder.compile(
        checkpointer=checkpointer,
    )

    question = (
        CompetitionQuestion(
            case_id="Q999",
            source_type="pdf",
            qa_type="单事实检索",
            question="测试问题",
            option_a="选项 A",
            option_b="选项 B",
            option_c="选项 C",
            option_d="选项 D",
            source_title="测试文档",
            file_label="test.pdf",
        )
    )

    config = {
        "configurable": {
            "thread_id": (
                "strict-question-test"
            ),
        }
    }

    graph.invoke(
        {
            "question": question,
        },
        config=config,
    )

    snapshot = graph.get_state(
        config
    )

    restored = (
        snapshot
        .values[
            "question"
        ]
    )

    assert isinstance(
        restored,
        CompetitionQuestion,
    )

    assert (
        restored.case_id
        == "Q999"
    )

    assert (
        restored.question
        == "测试问题"
    )
    assert restored.options == {
        "A": "选项 A",
        "B": "选项 B",
        "C": "选项 C",
        "D": "选项 D",
    }


# ============================================================
# 3. SQLite Resume after DB reopen
# ============================================================


def _build_resume_graph(
    *,
    checkpointer,
    counters: dict[
        str,
        int,
    ],
    fail_second: bool,
):
    builder = StateGraph(
        ResumeState
    )

    def first_node(
        state: ResumeState,
    ):
        counters[
            "first"
        ] += 1

        return {
            "steps": (
                "first",
            ),
        }

    def second_node(
        state: ResumeState,
    ):
        counters[
            "second"
        ] += 1

        if fail_second:
            raise (
                InjectedResumeFailure(
                    "injected failure"
                )
            )

        return {
            "steps": (
                "second",
            ),
        }

    builder.add_node(
        "first",
        first_node,
    )

    builder.add_node(
        "second",
        second_node,
    )

    builder.add_edge(
        START,
        "first",
    )

    builder.add_edge(
        "first",
        "second",
    )

    builder.add_edge(
        "second",
        END,
    )

    return builder.compile(
        checkpointer=checkpointer,
    )


def test_sqlite_resume_after_reopen_does_not_repeat_completed_node(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """
    模拟：

        Process A
        first ✅
        second 💥
        SQLite close

        Process B
        reopen SQLite
        resume
        second ✅

    Process B 不能重新执行 first。
    """

    monkeypatch.setenv(
        "LANGGRAPH_STRICT_MSGPACK",
        "true",
    )

    database_path = (
        tmp_path
        / "resume.sqlite3"
    )

    thread_id = (
        "pytest-resume-thread"
    )

    config = {
        "configurable": {
            "thread_id": thread_id,
        }
    }

    # ========================================================
    # Process A equivalent
    # ========================================================

    counters_a = {
        "first": 0,
        "second": 0,
    }

    with (
        open_competition_sqlite_checkpointer(
            database_path
        )
        as checkpointer
    ):
        graph_a = (
            _build_resume_graph(
                checkpointer=(
                    checkpointer
                ),
                counters=(
                    counters_a
                ),
                fail_second=True,
            )
        )

        with pytest.raises(
            InjectedResumeFailure
        ):
            graph_a.invoke(
                {},
                config=config,
            )

        failed = (
            graph_a.get_state(
                config
            )
        )

        assert (
            failed.next
            == ("second",)
        )

        assert (
            tuple(
                failed
                .values[
                    "steps"
                ]
            )
            == ("first",)
        )

    assert (
        counters_a[
            "first"
        ]
        == 1
    )

    assert (
        counters_a[
            "second"
        ]
        == 1
    )

    # ========================================================
    # Process B equivalent:
    #
    # 新 connection
    # 新 graph object
    # 新 counters
    # ========================================================

    counters_b = {
        "first": 0,
        "second": 0,
    }

    with (
        open_competition_sqlite_checkpointer(
            database_path
        )
        as checkpointer
    ):
        graph_b = (
            _build_resume_graph(
                checkpointer=(
                    checkpointer
                ),
                counters=(
                    counters_b
                ),
                fail_second=False,
            )
        )

        before = (
            graph_b.get_state(
                config
            )
        )

        assert (
            before.next
            == ("second",)
        )

        final_state = (
            graph_b.invoke(
                None,
                config=config,
            )
        )

        latest = (
            graph_b.get_state(
                config
            )
        )

    assert tuple(
        final_state[
            "steps"
        ]
    ) == (
        "first",
        "second",
    )

    assert latest.next == ()

    # 核心：
    # 新 Graph 没有重新执行 first。
    assert (
        counters_b[
            "first"
        ]
        == 0
    )

    assert (
        counters_b[
            "second"
        ]
        == 1
    )


# ============================================================
# 4. SQLite thread isolation
# ============================================================


def _build_thread_graph(
    *,
    checkpointer,
):
    builder = StateGraph(
        ThreadState
    )

    def execute_node(
        state: ThreadState,
    ):
        case_id = state[
            "case_id"
        ]

        return {
            "result": (
                f"done:{case_id}"
            ),
        }

    builder.add_node(
        "execute",
        execute_node,
    )

    builder.add_edge(
        START,
        "execute",
    )

    builder.add_edge(
        "execute",
        END,
    )

    return builder.compile(
        checkpointer=checkpointer,
    )


def _history_case_ids(
    history,
) -> tuple[str, ...]:
    case_ids: list[
        str
    ] = []

    for snapshot in history:
        case_id = (
            snapshot
            .values
            .get(
                "case_id"
            )
        )

        if (
            case_id is not None
            and case_id
            not in case_ids
        ):
            case_ids.append(
                case_id
            )

    return tuple(
        case_ids
    )


def test_sqlite_threads_are_isolated(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """
    同一个 SQLite DB：
        thread A 只能看到 QA
        thread B 只能看到 QB
    """

    monkeypatch.setenv(
        "LANGGRAPH_STRICT_MSGPACK",
        "true",
    )

    database_path = (
        tmp_path
        / "threads.sqlite3"
    )

    config_a = {
        "configurable": {
            "thread_id": "thread-A",
        }
    }

    config_b = {
        "configurable": {
            "thread_id": "thread-B",
        }
    }

    with (
        open_competition_sqlite_checkpointer(
            database_path
        )
        as checkpointer
    ):
        graph = (
            _build_thread_graph(
                checkpointer=(
                    checkpointer
                ),
            )
        )

        graph.invoke(
            {
                "case_id": "QA",
            },
            config=config_a,
        )

        graph.invoke(
            {
                "case_id": "QB",
            },
            config=config_b,
        )

        state_a = (
            graph.get_state(
                config_a
            )
        )

        state_b = (
            graph.get_state(
                config_b
            )
        )

        history_a = list(
            graph.get_state_history(
                config_a
            )
        )

        history_b = list(
            graph.get_state_history(
                config_b
            )
        )

    assert (
        state_a
        .values[
            "case_id"
        ]
        == "QA"
    )

    assert (
        state_a
        .values[
            "result"
        ]
        == "done:QA"
    )

    assert (
        state_b
        .values[
            "case_id"
        ]
        == "QB"
    )

    assert (
        state_b
        .values[
            "result"
        ]
        == "done:QB"
    )

    assert (
        _history_case_ids(
            history_a
        )
        == ("QA",)
    )

    assert (
        _history_case_ids(
            history_b
        )
        == ("QB",)
    )

    assert (
        "QB"
        not in (
            _history_case_ids(
                history_a
            )
        )
    )

    assert (
        "QA"
        not in (
            _history_case_ids(
                history_b
            )
        )
    )