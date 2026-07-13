# 06_api.py
# Purpose: FastAPI server that exposes the chat endpoint.

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from router import generate_answer  # <--- BURASI DÜZELTİLDİ (src yok)
import uvicorn

# ===============================================
# 1. INITIALIZE APP
# ===============================================
app = FastAPI(
    title="German Teacher AI Assistant",
    description="Pruned + LoRA + RAG powered assistant for German learners.",
    version="1.0.0",
)

# ===============================================
# 2. REQUEST MODEL
# ===============================================
class QueryRequest(BaseModel):
    query: str

# ===============================================
# 3. ENDPOINT
# ===============================================
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

# ===============================================
# 4. RUN SERVER
# ===============================================
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)