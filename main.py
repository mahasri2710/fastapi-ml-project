from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def home():
    return {"message": "My ML API is running!"}