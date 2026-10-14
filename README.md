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