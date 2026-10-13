"""FastAPI application entrypoint with lifespan startup hooks and CORS configuration."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from api.config import settings
from api.models.database import Base, engine
from api.routes.api_v1 import router as api_v1_router
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # Initialize SQLite/Postgres tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Campus Mental-Health Screening Aid — Privacy-Hardened Backend",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_v1_router)


@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "anjaney-api"}
