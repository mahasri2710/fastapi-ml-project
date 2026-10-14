## API Endpoints

### GET /
Response: `{"message": "My ML API is running!"}`

### GET /health
Checks if the API is running.
Response: `{"status": "ok"}`

### POST /predict
Request: `{"text": "hello"}`
Response: `{"input": "hello", "prediction": "placeholder"}`

## How to Run
```
pip install -r requirements.txt
uvicorn main:app --reload
```
Open http://127.0.0.1:8000/docs to test.
## Week 1 Progress
- Set up Python, VS Code, Git and GitHub
- Created virtual environment and installed FastAPI, uvicorn, pydantic
- Created project structure (models/, data/, requirements.txt)
- Built GET /, GET /health and POST /predict endpoints
- Tested endpoints using Swagger UI and PowerShell
- Documented the API in README