from fastapi import FastAPI
from pydantic import BaseModel
app = FastAPI()
class PredictRequest(BaseModel):
    text:str
@app.get("/")
def home():
    return {"message": "My ML API is running!"}
@app.get("/health")
def health():
    return {"status": "ok"}
@app.post("/predict")
def predict(req:PredictRequest):
    return {"input": req.text, "prediction": "placeholder"}