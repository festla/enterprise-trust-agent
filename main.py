from __future__ import annotations

from collections.abc import (
    AsyncIterator,
)
from contextlib import (
    asynccontextmanager,
)

from fastapi import (
    FastAPI,
)

from app.api.routes import (
    router,
)
from app.api.runtime import (
    CompetitionApiSettings,
    open_competition_api_graph,
)


@asynccontextmanager
async def lifespan(
    app: FastAPI,
) -> AsyncIterator[
    None
]:
    """
    FastAPI 生命周期。

    Startup:
        加载 Competition Runtime
        打开 SQLite Checkpointer
        编译 Unified Agent Graph

    Shutdown:
        关闭 SQLite connection
    """

    settings = (
        CompetitionApiSettings
        .from_environment()
    )

    with (
        open_competition_api_graph(
            settings
        )
        as graph
    ):
        app.state.competition_graph = (
            graph
        )

        app.state.competition_settings = (
            settings
        )

        yield


app = FastAPI(
    title=(
        "Enterprise Trust Agent"
    ),
    description=(
        "A trustworthy enterprise "
        "document analysis agent."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(
    router
)