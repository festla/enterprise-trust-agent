from fastapi import FastAPI

from app.api.routes import router


app = FastAPI(
    title="Enterprise Trust Agent",
    description="A trustworthy enterprise document analysis agent. ",
    version="0.1.0",
)

app.include_router(router)