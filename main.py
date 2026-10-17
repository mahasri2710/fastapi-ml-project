import logging

from fastapi import FastAPI
from fastapi.responses import PlainTextResponse

from routes.auth_routes import router as auth_router
from routes.ml_routes import router as ml_router
from utils import metrics
from utils.middleware import SecurityMiddleware

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

app = FastAPI(title="ML API with FastAPI", version="4.1")
app.add_middleware(SecurityMiddleware)
app.include_router(auth_router)
app.include_router(ml_router)


@app.get("/")
def home():
    return {"message": "ML API with FastAPI is running"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/metrics", response_class=PlainTextResponse)
def prometheus_metrics():
    """Metrics in Prometheus text format (for monitoring tools)."""
    return PlainTextResponse(metrics.render(), media_type="text/plain; version=0.0.4")
