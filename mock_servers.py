from fastapi import FastAPI
import uvicorn
import threading

app_vllm = FastAPI()
@app_vllm.post("/v1/chat/completions")
def chat():
    return {
        "choices": [{"message": {"content": "This is a mock answer from vLLM platform engineering."}}],
        "model": "Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4"
    }

app_embed = FastAPI()
@app_embed.post("/embed")
def embed(data: dict):
    texts = data.get("texts", [])
    embeddings = [[0.1] * 384 for _ in texts]
    return {"embeddings": embeddings}

def run_vllm():
    uvicorn.run(app_vllm, host="0.0.0.0", port=8001)

def run_embed():
    uvicorn.run(app_embed, host="0.0.0.0", port=8002)

if __name__ == "__main__":
    threading.Thread(target=run_vllm, daemon=True).start()
    run_embed()
