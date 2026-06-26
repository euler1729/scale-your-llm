import os
import httpx
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from src.models.message import QueryRequest

# Load environment variables
load_dotenv()

app = FastAPI(title="FastAPI vLLM Proxy")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://chat.cognistorm.ai", "http://localhost:3000"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/api/v1/query")
async def query_vllm(request: QueryRequest):
    vllm_url = os.getenv("VLLM_API_URL", "http://127.0.0.1:8000/v1/chat/completions")
    vllm_key = os.getenv("VLLM_API_KEY")
    
    payload = request.model_dump(exclude_none=True)
    headers = {"Content-Type": "application/json"}

    if vllm_key:
        headers["Authorization"] = f"Bearer {vllm_key}"
        
    if payload.get("stream"):
        async def stream_generator():
            async with httpx.AsyncClient() as client:
                async with client.stream("POST", vllm_url, headers=headers, json=payload, timeout=120.0) as response:
                    if response.status_code != 200:
                        error_text = await response.aread()
                        yield f"data: {{\"error\": \"Backend Error: {response.status_code}\"}}\n\n"
                        return
                    
                    async for chunk in response.aiter_bytes():
                        yield chunk

        return StreamingResponse(stream_generator(), media_type="text/event-stream")
    else:
        async with httpx.AsyncClient() as client:
            response = await client.post(vllm_url, headers=headers, json=payload, timeout=120.0)
            response.raise_for_status()
            return response.json()
            
@app.get("/health")
async def health_check():
    return {"status": "ok", "service": "fastapi-vllm-proxy"}

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    reload = os.getenv("ENV", "development") == "development"
    uvicorn.run("src.main:app", host=host, port=port, reload=reload)