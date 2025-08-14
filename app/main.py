import asyncio
import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi_cache import FastAPICache
from fastapi_cache.backends.inmemory import InMemoryBackend
from opentelemetry import trace
from opentelemetry.exporter.cloud_trace import CloudTraceSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.grpc import GrpcInstrumentorClient
from opentelemetry.instrumentation.requests import RequestsInstrumentor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.trace.sampling import ParentBased, TraceIdRatioBased
from starlette.status import HTTP_504_GATEWAY_TIMEOUT

from api.v1.api import api_router
from schemas.healthcheck import Checks, HealthCheck, ReadyCheck
from utils.globals import PROJECT_ID, SERVICE

REQUEST_TIMEOUT_SECONDS = int(os.getenv("REQUEST_TIMEOUT_SECONDS", "600"))
SERVICE_VERSION = os.getenv("SERVICE_VERSION", "1.0.0")
SERVICE_ENV = os.getenv("SERVICE_ENV", "dev")
CORS_ALLOW_ORIGINS = os.getenv("CORS_ALLOW_ORIGINS", "*").split(",")


# ---------- OpenTelemetry ----------
def _configure_tracing():
    resource = Resource.create(
        {
            "service.name": SERVICE,
            "service.version": SERVICE_VERSION,
            "deployment.environment": SERVICE_ENV,
            "gcp.project_id": PROJECT_ID,
        }
    )
    provider = TracerProvider(
        sampler=ParentBased(
            TraceIdRatioBased(float(os.getenv("OTEL_SAMPLE_RATIO", "1.0")))
        ),
        resource=resource,
    )
    processor = BatchSpanProcessor(
        CloudTraceSpanExporter(project_id=PROJECT_ID),
        max_queue_size=2048,
        schedule_delay_millis=500,
        max_export_batch_size=512,
    )
    provider.add_span_processor(processor)
    trace.set_tracer_provider(provider)


# ---------- Lifespan ----------
@asynccontextmanager
async def lifespan(app: FastAPI):
    _configure_tracing()
    FastAPIInstrumentor.instrument_app(app)
    RequestsInstrumentor().instrument()
    GrpcInstrumentorClient().instrument()
    FastAPICache.init(InMemoryBackend(), prefix=f"{SERVICE}:{SERVICE_ENV}:cache")
    yield


app = FastAPI(
    title="GenAI RAG Assistant",
    version=SERVICE_VERSION,
    summary=(
        "A Python application on Cloud Run that combines document ingestion, embeddings, "
        "and vector search in Vertex AI with the Gemini LLM, offering contextual and "
        "referenced answers based on user-uploaded documents."
    ),
    openapi_url="/api/v1/openapi.json",
    docs_url="/docs",
    redoc_url=None,
    contact={"name": "Marcos Ximenes Junior", "email": "m.ximenes.junior@gmail.com"},
    openapi_tags=[
        {
            "name": "healthcheck",
            "description": "Service and dependency liveness/readiness.",
        },
        {
            "name": "document",
            "description": "Document CRUD: upload, ingest, embeddings, and vector index.",
        },
        {
            "name": "chat",
            "description": "Q&A sessions with RAG: create conversation, ask questions, cite sources.",
        },
    ],
    lifespan=lifespan,
)


# ---------- Timeout middleware ----------
@app.middleware("http")
async def timeout_middleware(request: Request, call_next):
    start = time.monotonic()
    try:
        return await asyncio.wait_for(
            call_next(request), timeout=REQUEST_TIMEOUT_SECONDS
        )
    except asyncio.TimeoutError:
        elapsed = time.monotonic() - start
        return JSONResponse(
            {"detail": f"Request timed out after {elapsed:.2f}s"},
            status_code=HTTP_504_GATEWAY_TIMEOUT,
        )


# ---------- CORS ----------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if CORS_ALLOW_ORIGINS == ["*"] else CORS_ALLOW_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=False,
)


# ---------- Health ----------
@app.get("/", response_model=HealthCheck, tags=["healthcheck"])
async def liveness():
    return HealthCheck(status="ok", service=SERVICE, version=SERVICE_VERSION)


@app.get("/readyz", response_model=ReadyCheck, tags=["healthcheck"])
async def readiness(request: Request):
    checks = Checks(
        vertex_ai=True,  # TODO: await check_vertex_ai()
        vector_search=True,  # TODO: await check_vector_index()
        firestore=True,  # TODO: await check_firestore()
    )
    return ReadyCheck(
        status="READY" if all(checks.model_dump().values()) else "DEGRADED",
        checks=checks,
    )


# ---------- Routers ----------
app.include_router(api_router, prefix="/api/v1")
