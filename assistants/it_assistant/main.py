from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional

OPENAI_AVAILABLE = False
try:
    import openai
    OPENAI_AVAILABLE = True
except Exception:
    openai = None


def build_prompt_it(question: str, context: Optional[str] = None) -> str:
    base = (
        "You are an AI IT support assistant. Help with password resets, access provisioning, device issues, network/VPN, and general IT questions. "
        "Provide clear, actionable steps and troubleshooting guidance."
    )
    if context:
        return f"{base}\nContext: {context}\nQuestion: {question}"
    else:
        return f"{base}\nQuestion: {question}"


async def call_llm(prompt: str) -> str:
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
            return resp.choices[0].message.get("content", "").strip()
        except Exception:
            pass
    lower = prompt.lower()
    if "password" in lower:
        return "To reset a password, visit the account portal and follow the password reset flow. If you are locked out, contact IT with your username."
    if "vpn" in lower or "network" in lower:
        return "For VPN issues, ensure your client is installed, check network connectivity, and verify the VPN service is running."
    if "software" in lower or "install" in lower:
        return "For software installation, provide the exact software name and version, then confirm OS compatibility."
    return "Please provide more details or contact IT support for assistance."


app = FastAPI(title="IT Support Assistant")

class QueryRequest(BaseModel):
    user: str
    question: str
    context: Optional[str] = None


@app.post("/query")
async def query(req: QueryRequest):
    prompt = build_prompt_it(req.question, req.context)
    answer = await call_llm(prompt)
    confidence = 0.75
    if any(word in req.question.lower() for word in ["urgent", "asap", "immediately"]):
        confidence = 0.85
    return {"answer": answer, "confidence": confidence}


@app.get("/health")
async def health():
    return {"status": "ok", "service": "it-assistant"}