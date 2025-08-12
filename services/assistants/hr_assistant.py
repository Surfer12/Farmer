"""
HR Assistant FastAPI Service
Handles HR-related queries with OpenAI integration and fallback logic
"""

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime
import os
import logging
from enum import Enum
import openai
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="HR Assistant Service",
    description="AI-powered HR assistant for employee queries",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
USE_OPENAI = bool(OPENAI_API_KEY)

if USE_OPENAI:
    openai.api_key = OPENAI_API_KEY
    logger.info("OpenAI integration enabled")
else:
    logger.warning("OpenAI API key not found, using fallback responses")

# Request/Response Models
class QueryType(str, Enum):
    BENEFITS = "benefits"
    LEAVE = "leave"
    PAYROLL = "payroll"
    POLICIES = "policies"
    ONBOARDING = "onboarding"
    TRAINING = "training"
    GENERAL = "general"

class HRQuery(BaseModel):
    query: str = Field(..., description="The HR-related question")
    employee_id: Optional[str] = Field(None, description="Employee ID for personalized responses")
    query_type: Optional[QueryType] = Field(None, description="Type of HR query")
    context: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional context")

class HRResponse(BaseModel):
    response: str = Field(..., description="The assistant's response")
    query_type: QueryType = Field(..., description="Detected query type")
    confidence: float = Field(..., description="Confidence score (0-1)")
    sources: List[str] = Field(default_factory=list, description="Reference sources")
    timestamp: datetime = Field(default_factory=datetime.now)
    used_ai: bool = Field(..., description="Whether AI was used for response")

# Fallback knowledge base
FALLBACK_RESPONSES = {
    QueryType.BENEFITS: {
        "default": "Our company offers comprehensive benefits including health insurance, dental, vision, 401(k) matching, and flexible spending accounts. For specific details, please contact HR directly.",
        "keywords": {
            "health": "We offer multiple health insurance plans including PPO and HMO options. Open enrollment typically occurs in November.",
            "401k": "We offer 401(k) with company matching up to 6% of your salary. Vesting occurs over 3 years.",
            "vacation": "Standard PTO policy includes 15 days for new employees, increasing with tenure."
        }
    },
    QueryType.LEAVE: {
        "default": "Our leave policies include vacation, sick leave, personal days, and FMLA. Please submit leave requests through the HR portal.",
        "keywords": {
            "maternity": "Maternity leave is 12 weeks paid at 100% salary. Additional unpaid leave available under FMLA.",
            "sick": "Sick leave accrues at 1 day per month, up to 12 days annually.",
            "fmla": "FMLA provides up to 12 weeks of unpaid, job-protected leave for qualifying events."
        }
    },
    QueryType.PAYROLL: {
        "default": "Payroll is processed bi-weekly. Direct deposit is available and pay stubs can be accessed through the employee portal.",
        "keywords": {
            "payday": "Paydays are every other Friday. If payday falls on a holiday, payment is made the preceding business day.",
            "w2": "W-2 forms are distributed by January 31st each year, available both electronically and by mail.",
            "deductions": "Standard deductions include federal/state taxes, Social Security, Medicare, and any elected benefits."
        }
    },
    QueryType.POLICIES: {
        "default": "Company policies are available in the employee handbook on the HR portal. Key policies cover code of conduct, anti-discrimination, and workplace safety.",
        "keywords": {
            "remote": "Remote work policies allow eligible employees to work from home up to 3 days per week with manager approval.",
            "dress": "Business casual dress code applies Monday-Thursday, casual Fridays are permitted.",
            "harassment": "We maintain a zero-tolerance policy for harassment. Report incidents immediately to HR or use the anonymous hotline."
        }
    },
    QueryType.ONBOARDING: {
        "default": "New employee onboarding includes orientation, benefits enrollment, IT setup, and department introductions over the first week.",
        "keywords": {
            "first day": "On your first day, report to reception at 9 AM. Bring two forms of ID and completed I-9 documentation.",
            "orientation": "Orientation covers company culture, policies, benefits, and safety procedures. Typically scheduled for Day 1-2.",
            "training": "Initial training includes required compliance modules and role-specific training with your manager."
        }
    },
    QueryType.TRAINING: {
        "default": "We offer various training programs including professional development, leadership training, and technical skills courses.",
        "keywords": {
            "leadership": "Leadership development programs are available for high-potential employees. Nominations occur annually.",
            "tuition": "Tuition reimbursement up to $5,250 annually for job-related courses with manager approval.",
            "certification": "Professional certification costs are reimbursable for job-relevant certifications."
        }
    }
}

def classify_query(query: str) -> QueryType:
    """Classify the query into a category based on keywords"""
    query_lower = query.lower()
    
    keyword_map = {
        QueryType.BENEFITS: ["benefit", "insurance", "health", "dental", "vision", "401k", "retirement"],
        QueryType.LEAVE: ["leave", "vacation", "pto", "sick", "maternity", "paternity", "fmla", "time off"],
        QueryType.PAYROLL: ["pay", "salary", "payroll", "paycheck", "w2", "tax", "deduction", "wage"],
        QueryType.POLICIES: ["policy", "rule", "dress code", "remote", "harassment", "conduct", "guideline"],
        QueryType.ONBOARDING: ["onboard", "new employee", "first day", "orientation", "start date", "new hire"],
        QueryType.TRAINING: ["training", "development", "course", "certification", "learning", "tuition", "skill"]
    }
    
    scores = {}
    for qtype, keywords in keyword_map.items():
        score = sum(1 for keyword in keywords if keyword in query_lower)
        if score > 0:
            scores[qtype] = score
    
    if scores:
        return max(scores, key=scores.get)
    return QueryType.GENERAL

def get_fallback_response(query: str, query_type: QueryType) -> tuple[str, float]:
    """Get fallback response when OpenAI is not available"""
    query_lower = query.lower()
    
    if query_type in FALLBACK_RESPONSES:
        category = FALLBACK_RESPONSES[query_type]
        
        # Check for keyword matches
        for keyword, response in category.get("keywords", {}).items():
            if keyword in query_lower:
                return response, 0.7
        
        # Return default response for category
        return category["default"], 0.5
    
    # Generic fallback
    return "I can help you with HR-related questions about benefits, leave policies, payroll, company policies, onboarding, and training. Please provide more specific details about your query.", 0.3

async def get_openai_response(query: str, query_type: QueryType, context: Dict[str, Any]) -> tuple[str, float]:
    """Get response from OpenAI API"""
    try:
        system_prompt = f"""You are an HR Assistant specializing in {query_type.value} queries. 
        Provide helpful, accurate, and professional responses to employee questions.
        Be concise but thorough. If you need more information, ask clarifying questions.
        Always maintain confidentiality and professionalism."""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ]
        
        if context:
            messages.append({"role": "system", "content": f"Additional context: {json.dumps(context)}"})
        
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.7,
            max_tokens=500
        )
        
        return response.choices[0].message.content, 0.95
    
    except Exception as e:
        logger.error(f"OpenAI API error: {e}")
        # Fall back to local response
        return get_fallback_response(query, query_type)

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "service": "HR Assistant",
        "status": "operational",
        "openai_enabled": USE_OPENAI,
        "version": "1.0.0"
    }

@app.post("/query", response_model=HRResponse)
async def process_query(hr_query: HRQuery):
    """Process an HR query and return a response"""
    try:
        # Classify query if not provided
        query_type = hr_query.query_type or classify_query(hr_query.query)
        
        # Get response
        if USE_OPENAI:
            response_text, confidence = await get_openai_response(
                hr_query.query, 
                query_type, 
                hr_query.context or {}
            )
            used_ai = True
        else:
            response_text, confidence = get_fallback_response(hr_query.query, query_type)
            used_ai = False
        
        # Prepare sources based on query type
        sources = []
        if query_type == QueryType.BENEFITS:
            sources.append("Employee Benefits Guide 2024")
        elif query_type == QueryType.POLICIES:
            sources.append("Employee Handbook v3.2")
        elif query_type == QueryType.LEAVE:
            sources.append("Leave Policy Documentation")
        
        return HRResponse(
            response=response_text,
            query_type=query_type,
            confidence=confidence,
            sources=sources,
            used_ai=used_ai
        )
    
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/categories")
async def get_categories():
    """Get available query categories"""
    return {
        "categories": [
            {
                "type": qtype.value,
                "description": FALLBACK_RESPONSES.get(qtype, {}).get("default", "")[:100] + "..."
            }
            for qtype in QueryType
        ]
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "openai_status": "connected" if USE_OPENAI else "disabled",
        "fallback_available": True,
        "categories_loaded": len(FALLBACK_RESPONSES)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)