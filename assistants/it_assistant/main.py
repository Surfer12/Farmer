from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, Tuple
import logging
import json
import os
import time
from datetime import datetime

OPENAI_AVAILABLE = False
try:
    import openai
    OPENAI_AVAILABLE = True
except Exception:
    openai = None

# --- Logger Setup ---
LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(LOG_DIR, exist_ok=True)
log_file = os.path.join(LOG_DIR, "queries.jsonl")

logger = logging.getLogger("it_assistant")
logger.setLevel(logging.INFO)
logger.propagate = False
if not logger.handlers:
    handler = logging.FileHandler(log_file)
    logger.addHandler(handler)


def build_prompt_it(question: str, context: Optional[str] = None) -> str:
    base = (
        "You are an AI IT support assistant. Help with password resets, access provisioning, device issues, network/VPN, and general IT questions. "
        "Provide clear, actionable steps and troubleshooting guidance."
    )
    if context:
        return f"{base}\nContext: {context}\nQuestion: {question}"
    else:
        return f"{base}\nQuestion: {question}"


async def call_llm(prompt: str) -> Tuple[str, str]:
    model_used = "fallback"
    if OPENAI_AVAILABLE and openai:
        try:
            resp = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful IT support assistant."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=300,
                temperature=0.3,
            )
            model_used = "llm"
            return resp.choices[0].message.get("content", "").strip(), model_used
        except Exception:
            pass
    lower = prompt.lower()
    if "password" in lower:
        return "To reset a password, visit the account portal and follow the password reset flow. If you are locked out, contact IT with your username.", model_used
    if "vpn" in lower or "network" in lower:
        return "For VPN issues, ensure your client is installed, check network connectivity, and verify the VPN service is running.", model_used
    if "software" in lower or "install" in lower:
        return "For software installation, provide the exact software name and version, then confirm OS compatibility.", model_used
    return "Please provide more details or contact IT support for assistance.", model_used


app = FastAPI(title="IT Support Assistant")

class QueryRequest(BaseModel):
    user: str
    question: str
    context: Optional[str] = None


@app.post("/query")
async def query(req: QueryRequest):
    start_time = time.time()
    prompt = build_prompt_it(req.question, req.context)
    answer, model_used = await call_llm(prompt)
    latency_ms = (time.time() - start_time) * 1000
    
    confidence = 0.75
    if any(word in req.question.lower() for word in ["urgent", "asap", "immediately"]):
        confidence = 0.85
        
    log_record = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "service": "it-assistant",
        "user": req.user,
        "question": req.question,
        "answer": answer,
        "confidence": confidence,
        "model_used": model_used,
        "latency_ms": round(latency_ms, 2),
    }
    logger.info(json.dumps(log_record))
    
    return {"answer": answer, "confidence": confidence}


@app.get("/health")
async def health():
    return {"status": "ok", "service": "it-assistant"}