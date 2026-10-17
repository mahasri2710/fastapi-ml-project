# FastAPI ML Project
## Problem
Colleges want to know early which students may fail, so they can help them in time. Doing this by hand for many students is slow.

## Solution
A REST API built with FastAPI and scikit-learn. It trains a model on student data (hours studied, attendance) and predicts whether a student will pass. Admins train the model, and any logged in user can get predictions.

## Features
- Train a model (with optional hyperparameter tuning) and upload your own CSV dataset
- Predict for one student or a batch of up to 100 students
- Evaluate the model: accuracy, precision, recall, F1, confusion matrix
- JWT login with roles: admin (train, upload) and user (predict, evaluate)
- Rate limiting, IP blocking, request logging, Prometheus style /metrics
- Automated tests with pytest

## Setup Steps
1. Clone the repo: `git clone https://github.com/mahasri2710/fastapi-ml-project.git`
2. Go into the folder: `cd fastapi-ml-project`
3. Create a virtual environment: `python -m venv .venv`
4. Activate it (Windows): `.venv\Scripts\activate`
5. Install packages: `pip install -r requirements.txt`
6. Run the server: `uvicorn main:app --reload`
7. Open `http://127.0.0.1:8000/docs`
8. Log in with the Authorize button. Demo logins: admin / admin123 and student / student123
## Problem
Colleges want to know early which students may fail, so they can help them in time. Doing this by hand for many students is slow.

## Solution
A REST API built with FastAPI and scikit-learn. It trains a model on student data (hours studied, attendance) and predicts whether a student will pass. Admins train the model, and any logged in user can get predictions.

## Features
- Train a model (with optional hyperparameter tuning) and upload your own CSV dataset
- Predict for one student or a batch of up to 100 students
- Evaluate the model: accuracy, precision, recall, F1, confusion matrix
- JWT login with roles: admin (train, upload) and user (predict, evaluate)
- Rate limiting, IP blocking, request logging, Prometheus style /metrics
- Automated tests with pytest

## Setup Steps
1. Clone the repo: `git clone https://github.com/mahasri2710/fastapi-ml-project.git`
2. Go into the folder: `cd fastapi-ml-project`
3. Create a virtual environment: `python -m venv .venv`
4. Activate it (Windows): `.venv\Scripts\activate`
5. Install packages: `pip install -r requirements.txt`
6. Run the server: `uvicorn main:app --reload`
7. Open `http://127.0.0.1:8000/docs`
8. Log in with the Authorize button. Demo logins: admin / admin123 and student / student123

## API Endpoints
| Method | Path | Who can use it |
|--------|------|----------------|
| POST | /login | everyone |
| POST | /upload-dataset | admin |
| POST | /train | admin |
| POST | /predict | logged in user |
| POST | /predict/batch | logged in user |
| GET | /evaluate | logged in user |
| GET | /model-info | logged in user |
| GET | /health | everyone |

## Run Tests
`pytest`
## API Endpoints
| Method | Path | Who can use it |
|--------|------|----------------|
| POST | /login | everyone |
| POST | /upload-dataset | admin |
| POST | /train | admin |
| POST | /predict | logged in user |
| POST | /predict/batch | logged in user |
| GET | /evaluate | logged in user |
| GET | /model-info | logged in user |
| GET | /health | everyone |

## Run Tests
`pytest`
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

## Week 4 Progress (Day 24-26)
- Added JWT login (POST /login) and protected /train and /upload-dataset
- Added rate limiting (60 requests per minute) and IP blocking
- Added request logging and /metrics endpoint in Prometheus format
- Added automated tests with pytest (run: pytest)
- Added Dockerfile for containerization
- Default login for learning: admin / admin123 (set API_USERNAME and API_PASSWORD env variables for real use)

## Week 5 Progress (Day 29-33)
- Added role-based access control: admin (train, upload) and user (predict, evaluate)
- Model is loaded into memory when the server starts and reloaded after training
- Added GET /model-info and POST /predict/batch (up to 100 students)
- JWT tokens now include the user role
- Added more automated tests (roles, batch prediction, model info)
- Demo logins for learning: admin / admin123 and student / student123
## Week 6 Progress (Day 36-40)
- Added unit tests, edge case tests and integration tests (pytest, requests, JSON schema)
- Tested rate limiting, IP blocking, bad and expired tokens, bad uploads
- Secrets come from environment variables; production mode refuses default passwords
- Added Dockerfile, docker-compose.yml, Prometheus config and alert rules (not run locally, Docker is not installed on my laptop)
- Run tests: pytest
- Run integration tests with the server running: pytest tests/test_integration.py -v