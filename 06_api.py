# 06_api.py
# FastAPI server for the German Teacher AI.

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from router import generate_answer
import uvicorn

app = FastAPI(title="German Teacher AI (7B)", version="2.0.0")

class QueryRequest(BaseModel):
    query: str

@app.post("/chat")
async def chat_endpoint(request: QueryRequest):
    try:
        response = generate_answer(request.query)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)   # Bind to all interfaces