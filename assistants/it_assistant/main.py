from fastapi import FastAPI, Request, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional, Tuple
import logging
import json
import os
import time
from datetime import datetime
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

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
DEFAULT_LOG_DIR = os.path.join(PROJECT_ROOT, "data", "logs")
LOG_DIR = os.environ.get("LOG_DIR", DEFAULT_LOG_DIR)
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "assistant-it.jsonl")

logger = logging.getLogger("it_assistant")
logger.setLevel(logging.INFO)
logger.propagate = False
if not logger.handlers:
    handler = logging.FileHandler(LOG_FILE)
    logger.addHandler(handler)

if REQUIRE_AUTH and not AUTH_TOKEN:
    raise RuntimeError("REQUIRE_AUTH is enabled but AUTH_TOKEN is not set")


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
    except Exception:
        status_code = 500
        raise
    finally:
        duration_ms = int((time.time() - start_time) * 1000)
        client_ip = request.headers.get("x-forwarded-for") or (request.client.host if request.client else "?")
        log_record = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "service": "it-assistant",
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
    try:
        response.headers["x-request-id"] = request_id
    except Exception:
        pass
    return response


@app.post("/query")
async def query(req: QueryRequest, _=Depends(verify_auth)):
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
        "event": "query_complete",
        "latency_ms": round(latency_ms, 2),
        "confidence": confidence,
        "model_used": model_used,
    }
    logger.info(json.dumps(log_record))
    
    return {"answer": answer, "confidence": confidence}


@app.get("/health")
async def health():
    return {"status": "ok", "service": "it-assistant"}
