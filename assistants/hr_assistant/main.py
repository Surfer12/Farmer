from fastapi import FastAPI, Request, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional, Tuple
import logging
import json
import os
import time
from datetime import datetime
from typing import Optional
import uuid

OPENAI_AVAILABLE = False
try:
    import openai
    OPENAI_AVAILABLE = True
except Exception:
    openai = None

# --- Env-driven Config & Logger Setup ---
REQUIRE_AUTH = str(os.environ.get("REQUIRE_AUTH", "0")).lower() in {"1", "true", "yes"}
AUTH_TOKEN = os.environ.get("AUTH_TOKEN")

# Determine project root and default unified log directory under data/logs/
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
DEFAULT_LOG_DIR = os.path.join(PROJECT_ROOT, "data", "logs")
LOG_DIR = os.environ.get("LOG_DIR", DEFAULT_LOG_DIR)
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "assistant-hr.jsonl")

logger = logging.getLogger("hr_assistant")
logger.setLevel(logging.INFO)
logger.propagate = False
if not logger.handlers:
    handler = logging.FileHandler(LOG_FILE)
    logger.addHandler(handler)

if REQUIRE_AUTH and not AUTH_TOKEN:
    raise RuntimeError("REQUIRE_AUTH is enabled but AUTH_TOKEN is not set")


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


def _auth_dep():
    return None

async def verify_auth(request: Request):
    if not REQUIRE_AUTH:
        return None
    auth_header = request.headers.get("authorization")
    if not auth_header or not auth_header.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Unauthorized")
    token = auth_header.split(" ", 1)[1]
    if token != AUTH_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return None


@app.middleware("http")
async def unified_logging_middleware(request: Request, call_next):
    request_id = str(uuid.uuid4())
    start_time = time.time()
    try:
        response = await call_next(request)
        status_code = response.status_code
    except Exception as exc:
        status_code = 500
        raise
    finally:
        duration_ms = int((time.time() - start_time) * 1000)
        client_ip = request.headers.get("x-forwarded-for") or (request.client.host if request.client else "?")
        log_record = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "service": "hr-assistant",
            "event": "request",
            "method": request.method,
            "path": request.url.path,
            "status": status_code,
            "duration_ms": duration_ms,
            "client_ip": client_ip,
            "request_id": request_id,
            "auth": "required" if REQUIRE_AUTH else "disabled",
        }
        logger.info(json.dumps(log_record))
    # Add request-id header for traceability
    try:
        response.headers["x-request-id"] = request_id
    except Exception:
        pass
    return response


@app.post("/query")
async def query(req: QueryRequest, _=Depends(verify_auth)):
    start_time = time.time()
    prompt = build_prompt_hr(req.question, req.context)
    answer, model_used = await call_llm(prompt)
    latency_ms = (time.time() - start_time) * 1000

    confidence = 0.75
    if any(word in req.question.lower() for word in ["urgent", "asap", "immediately"]):
        confidence = 0.85

    # Minimal content log (no PII where possible)
    log_record = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "service": "hr-assistant",
        "event": "query_complete",
        "latency_ms": round(latency_ms, 2),
        "confidence": confidence,
        "model_used": model_used,
    }
    logger.info(json.dumps(log_record))

    return {"answer": answer, "confidence": confidence}


@app.get("/health")
async def health():
    return {"status": "ok", "service": "hr-assistant"}
