import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import PlainTextResponse

from routes.auth_routes import router as auth_router
from routes.ml_routes import router as ml_router
from utils import metrics, ml_utils
from utils.middleware import SecurityMiddleware

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # runs once when the server starts: load the saved model into memory
    ml_utils.load_model_on_startup()
    yield


app = FastAPI(title="ML API with FastAPI", version="5.0", lifespan=lifespan)
app.add_middleware(SecurityMiddleware)
app.include_router(auth_router)
app.include_router(ml_router)


@app.get("/")
def home():
    return {"message": "ML API with FastAPI is running"}


@app.get("/health")
def health():
    return {"status": "ok", "model_ready": ml_utils.model_ready()}


@app.get("/metrics", response_class=PlainTextResponse)
def prometheus_metrics():
    """Metrics in Prometheus text format (for monitoring tools)."""
    return PlainTextResponse(metrics.render(), media_type="text/plain; version=0.0.4")
