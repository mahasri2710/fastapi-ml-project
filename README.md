# FastAPI ML Project

A simple Machine Learning API built with FastAPI and scikit-learn.
It trains a model on a sample dataset (hours studied + attendance -> pass/fail)
and returns predictions in JSON.

## Setup
```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload
```
Open http://127.0.0.1:8000/docs for Swagger UI.

## Project Structure
- `main.py` : FastAPI app and endpoints
- `models.py` : ML model code (train, save, load, predict)
- `data/sample.csv` : sample dataset
- `models/` : saved trained model (`model.pkl`)
- `test_api.ps1` : quick PowerShell test script
- `requirements.txt` : dependencies

## API Endpoints

### GET /
Response: `{"message": "My ML API is running!"}`

### GET /health
Response: `{"status": "ok"}`

### POST /train
Trains the model on `data/sample.csv` and saves it.
Response: `{"message": "Model trained successfully", "rows_used": 20, "train_accuracy": 1.0}`

### POST /predict
Request: `{"hours_studied": 8, "attendance": 90}`
Response: `{"input": {...}, "passed": true, "pass_probability": 1.0}`

Input validation (pydantic): hours_studied 0-24, attendance 0-100.
If the model is not trained yet, `/predict` returns a 400 error asking to call `/train` first.

## Logging
Training progress, prediction requests and responses are logged in the terminal.

## Week 1 Progress
- Set up Python, VS Code, Git and GitHub
- Created virtual environment, installed FastAPI, uvicorn, pydantic
- Created project structure and basic GET/POST endpoints

## Week 2 Progress
- Added scikit-learn model (LogisticRegression) in `models.py`
- Added `/train` endpoint using pandas to load `data/sample.csv`
- Added `/predict` endpoint with pydantic validation, numpy computation
- Added error handling and logging for training and prediction
- Tested all endpoints using Swagger UI and PowerShell (`test_api.ps1`)

## Week 3 Progress
- Restructured project into routes/ and utils/ folders
- Added pydantic request and response validation
- POST /train, POST /predict, GET /evaluate endpoints
- Added error handling and logging

## Week 4 Progress (Day 22-23)
- Added POST /upload-dataset to upload a CSV dataset
- Added GET /dataset-info to show which dataset is used
- Added hyperparameter tuning with GridSearchCV (POST /train?tune=true)
- Added python-multipart to requirements.txt
