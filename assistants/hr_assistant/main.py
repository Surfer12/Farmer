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

logger = logging.getLogger("hr_assistant")
logger.setLevel(logging.INFO)
logger.propagate = False
if not logger.handlers:
    handler = logging.FileHandler(log_file)
    logger.addHandler(handler)


def build_prompt_hr(question: str, context: Optional[str] = None) -> str:
    base = (
        "You are an AI HR assistant. You help employees with onboarding, benefits, policy lookup, leave requests, and general HR questions. "
        "Be concise, practical, and empathetic."
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
                    {"role": "system", "content": "You are a helpful HR assistant."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=300,
                temperature=0.3,
            )
            model_used = "llm"
            return resp.choices[0].message.get("content", "").strip(), model_used
        except Exception:
            pass
    # Fallback heuristic
    lower = prompt.lower()
    if "benefits" in lower:
        return "Our benefits include health, dental, vision, and retirement plans. Please specify your region for tailored options.", model_used
    if "onboarding" in lower or "new hire" in lower:
        return "For onboarding, access the onboarding portal and review the welcome packet.", model_used
    if "leave" in lower:
        return "Leave requests are submitted via the HR portal; you can check your balance there.", model_used
    return "Please contact HR for assistance or provide more details.", model_used


app = FastAPI(title="HR Assistant")

class QueryRequest(BaseModel):
    user: str
    question: str
    context: Optional[str] = None


@app.post("/query")
async def query(req: QueryRequest):
    start_time = time.time()
    prompt = build_prompt_hr(req.question, req.context)
    answer, model_used = await call_llm(prompt)
    latency_ms = (time.time() - start_time) * 1000

    confidence = 0.75
    if any(word in req.question.lower() for word in ["urgent", "asap", "immediately"]):
        confidence = 0.85

    log_record = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "service": "hr-assistant",
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
    return {"status": "ok", "service": "hr-assistant"}