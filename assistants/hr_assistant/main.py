from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional

OPENAI_AVAILABLE = False
try:
    import openai
    OPENAI_AVAILABLE = True
except Exception:
    openai = None


def build_prompt_hr(question: str, context: Optional[str] = None) -> str:
    base = (
        "You are an AI HR assistant. You help employees with onboarding, benefits, policy lookup, leave requests, and general HR questions. "
        "Be concise, practical, and empathetic."
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
                    {"role": "system", "content": "You are a helpful HR assistant."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=300,
                temperature=0.3,
            )
            return resp.choices[0].message.get("content", "").strip()
        except Exception:
            pass
    # Fallback heuristic
    lower = prompt.lower()
    if "benefits" in lower:
        return "Our benefits include health, dental, vision, and retirement plans. Please specify your region for tailored options."
    if "onboarding" in lower or "new hire" in lower:
        return "For onboarding, access the onboarding portal and review the welcome packet."
    if "leave" in lower:
        return "Leave requests are submitted via the HR portal; you can check your balance there."
    return "Please contact HR for assistance or provide more details."


app = FastAPI(title="HR Assistant")

class QueryRequest(BaseModel):
    user: str
    question: str
    context: Optional[str] = None


@app.post("/query")
async def query(req: QueryRequest):
    prompt = build_prompt_hr(req.question, req.context)
    answer = await call_llm(prompt)
    confidence = 0.75
    if any(word in req.question.lower() for word in ["urgent", "asap", "immediately"]):
        confidence = 0.85
    return {"answer": answer, "confidence": confidence}


@app.get("/health")
async def health():
    return {"status": "ok", "service": "hr-assistant"}