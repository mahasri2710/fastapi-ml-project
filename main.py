import logging

from fastapi import FastAPI

from routes.ml_routes import router as ml_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

app = FastAPI(title="ML API with FastAPI", version="4.0")
app.include_router(ml_router)


@app.get("/")
def home():
    return {"message": "ML API with FastAPI is running"}


@app.get("/health")
def health():
    return {"status": "ok"}
