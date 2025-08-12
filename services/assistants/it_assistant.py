"""
IT Assistant FastAPI Service
Handles IT support queries with OpenAI integration and fallback logic
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime
import os
import logging
from enum import Enum
import openai
import json
import random
import string

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="IT Assistant Service",
    description="AI-powered IT support assistant for technical queries",
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
class IssueType(str, Enum):
    HARDWARE = "hardware"
    SOFTWARE = "software"
    NETWORK = "network"
    ACCOUNT = "account"
    SECURITY = "security"
    EMAIL = "email"
    PRINTER = "printer"
    GENERAL = "general"

class Priority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ITQuery(BaseModel):
    query: str = Field(..., description="The IT-related question or issue")
    user_id: Optional[str] = Field(None, description="User ID for personalized support")
    issue_type: Optional[IssueType] = Field(None, description="Type of IT issue")
    priority: Optional[Priority] = Field(None, description="Issue priority")
    system_info: Optional[Dict[str, Any]] = Field(default_factory=dict, description="System information")

class ITResponse(BaseModel):
    response: str = Field(..., description="The assistant's response")
    issue_type: IssueType = Field(..., description="Detected issue type")
    priority: Priority = Field(..., description="Assigned priority")
    ticket_id: Optional[str] = Field(None, description="Support ticket ID if created")
    steps: List[str] = Field(default_factory=list, description="Troubleshooting steps")
    confidence: float = Field(..., description="Confidence score (0-1)")
    timestamp: datetime = Field(default_factory=datetime.now)
    used_ai: bool = Field(..., description="Whether AI was used for response")
    estimated_resolution_time: str = Field(..., description="Estimated time to resolution")

# Fallback knowledge base
FALLBACK_RESPONSES = {
    IssueType.HARDWARE: {
        "default": "For hardware issues, please try: 1) Restart the device, 2) Check all cable connections, 3) Run hardware diagnostics. If the issue persists, submit a ticket for hardware support.",
        "keywords": {
            "keyboard": "For keyboard issues: 1) Check USB/wireless connection, 2) Replace batteries if wireless, 3) Try a different USB port, 4) Test with on-screen keyboard.",
            "monitor": "For monitor issues: 1) Check cable connections, 2) Try a different cable, 3) Adjust display settings, 4) Test with another monitor if available.",
            "laptop": "For laptop issues: 1) Perform a hard reset (hold power 10 seconds), 2) Check battery and charger, 3) Boot in safe mode, 4) Run diagnostics (F12 at startup).",
            "mouse": "For mouse issues: 1) Check USB/wireless connection, 2) Clean the sensor, 3) Replace batteries if wireless, 4) Update mouse drivers."
        }
    },
    IssueType.SOFTWARE: {
        "default": "For software issues: 1) Restart the application, 2) Check for updates, 3) Clear cache/temporary files, 4) Reinstall if necessary.",
        "keywords": {
            "install": "To install software: 1) Check system requirements, 2) Download from approved sources, 3) Run as administrator if needed, 4) Contact IT for license keys.",
            "crash": "For application crashes: 1) Check Event Viewer for errors, 2) Update the application, 3) Run in compatibility mode, 4) Reinstall the application.",
            "update": "For update issues: 1) Check internet connection, 2) Clear Windows Update cache, 3) Run Windows Update troubleshooter, 4) Manually download updates.",
            "license": "For license issues: 1) Verify license key accuracy, 2) Check expiration date, 3) Ensure proper activation, 4) Contact IT for license renewal."
        }
    },
    IssueType.NETWORK: {
        "default": "For network issues: 1) Check cable/WiFi connection, 2) Restart router/modem, 3) Run network troubleshooter, 4) Check with IT for outages.",
        "keywords": {
            "wifi": "For WiFi issues: 1) Toggle WiFi off/on, 2) Forget and reconnect to network, 3) Update network drivers, 4) Check router distance/interference.",
            "vpn": "For VPN issues: 1) Check credentials, 2) Try different VPN server, 3) Update VPN client, 4) Verify firewall settings.",
            "slow": "For slow connection: 1) Run speed test, 2) Check for background downloads, 3) Reset network adapter, 4) Contact IT for bandwidth issues.",
            "dns": "For DNS issues: 1) Flush DNS cache (ipconfig /flushdns), 2) Change DNS servers, 3) Reset TCP/IP stack, 4) Check hosts file."
        }
    },
    IssueType.ACCOUNT: {
        "default": "For account issues: 1) Verify username spelling, 2) Use password reset if needed, 3) Check account status, 4) Contact IT for account unlock.",
        "keywords": {
            "password": "For password issues: 1) Use self-service password reset, 2) Ensure caps lock is off, 3) Check password requirements, 4) Contact IT if locked out.",
            "login": "For login issues: 1) Verify credentials, 2) Clear browser cache, 3) Try incognito mode, 4) Check system time/date settings.",
            "permission": "For permission issues: 1) Verify group membership, 2) Check with manager for approval, 3) Submit access request form, 4) Allow 24-48 hours for processing.",
            "mfa": "For MFA issues: 1) Sync authenticator app time, 2) Use backup codes, 3) Try SMS option, 4) Contact IT to reset MFA."
        }
    },
    IssueType.SECURITY: {
        "default": "For security concerns: 1) Run antivirus scan, 2) Change passwords, 3) Enable MFA, 4) Report suspicious activity to IT immediately.",
        "keywords": {
            "virus": "Suspected virus: 1) Disconnect from network, 2) Run full antivirus scan, 3) Boot in safe mode, 4) Contact IT Security immediately.",
            "phishing": "For phishing: 1) Don't click links, 2) Forward to IT Security, 3) Delete the email, 4) Change password if compromised.",
            "breach": "For data breach: 1) Document the incident, 2) Contact IT Security immediately, 3) Preserve evidence, 4) Don't attempt to fix yourself.",
            "suspicious": "For suspicious activity: 1) Document what you observed, 2) Take screenshots, 3) Report to IT Security, 4) Change passwords as precaution."
        }
    },
    IssueType.EMAIL: {
        "default": "For email issues: 1) Check internet connection, 2) Verify email settings, 3) Clear cache, 4) Try webmail as alternative.",
        "keywords": {
            "outlook": "For Outlook issues: 1) Run Outlook in safe mode, 2) Repair Office installation, 3) Create new profile, 4) Check .pst file size.",
            "spam": "For spam issues: 1) Mark as spam/junk, 2) Create rules to filter, 3) Never reply to spam, 4) Report persistent spam to IT.",
            "attachment": "For attachment issues: 1) Check file size limits (usually 25MB), 2) Compress large files, 3) Use cloud storage links, 4) Verify file type restrictions.",
            "calendar": "For calendar issues: 1) Check sharing permissions, 2) Refresh calendar view, 3) Clear calendar cache, 4) Verify time zone settings."
        }
    },
    IssueType.PRINTER: {
        "default": "For printer issues: 1) Check printer power/connection, 2) Clear print queue, 3) Update printer drivers, 4) Run printer troubleshooter.",
        "keywords": {
            "jam": "For paper jam: 1) Turn off printer, 2) Open all doors/trays, 3) Gently remove jammed paper, 4) Check for torn pieces, 5) Restart printer.",
            "quality": "For print quality: 1) Clean print heads, 2) Align printer, 3) Replace ink/toner, 4) Use appropriate paper type.",
            "driver": "For driver issues: 1) Uninstall current driver, 2) Download latest from manufacturer, 3) Install as administrator, 4) Restart print spooler.",
            "network": "For network printer: 1) Ping printer IP, 2) Check printer queue status, 3) Reinstall network printer, 4) Verify printer sharing settings."
        }
    }
}

def generate_ticket_id() -> str:
    """Generate a unique ticket ID"""
    return f"IT-{datetime.now().strftime('%Y%m%d')}-{''.join(random.choices(string.ascii_uppercase + string.digits, k=6))}"

def classify_issue(query: str) -> IssueType:
    """Classify the issue into a category based on keywords"""
    query_lower = query.lower()
    
    keyword_map = {
        IssueType.HARDWARE: ["hardware", "keyboard", "mouse", "monitor", "laptop", "computer", "device", "usb", "hdmi"],
        IssueType.SOFTWARE: ["software", "application", "app", "install", "crash", "error", "update", "program", "license"],
        IssueType.NETWORK: ["network", "internet", "wifi", "connection", "vpn", "slow", "bandwidth", "ethernet", "dns"],
        IssueType.ACCOUNT: ["account", "password", "login", "access", "permission", "username", "credential", "mfa", "authentication"],
        IssueType.SECURITY: ["security", "virus", "malware", "phishing", "breach", "suspicious", "hack", "threat", "firewall"],
        IssueType.EMAIL: ["email", "outlook", "mail", "inbox", "spam", "attachment", "calendar", "meeting", "invite"],
        IssueType.PRINTER: ["printer", "print", "scan", "copy", "toner", "ink", "paper", "jam", "queue"]
    }
    
    scores = {}
    for itype, keywords in keyword_map.items():
        score = sum(2 if keyword in query_lower else 0 for keyword in keywords)
        if score > 0:
            scores[itype] = score
    
    if scores:
        return max(scores, key=scores.get)
    return IssueType.GENERAL

def determine_priority(query: str, issue_type: IssueType) -> Priority:
    """Determine priority based on keywords and issue type"""
    query_lower = query.lower()
    
    critical_keywords = ["down", "outage", "breach", "hacked", "emergency", "urgent", "asap", "critical"]
    high_keywords = ["cannot work", "blocked", "error", "failed", "broken", "important"]
    medium_keywords = ["slow", "issue", "problem", "help", "need"]
    
    if any(keyword in query_lower for keyword in critical_keywords):
        return Priority.CRITICAL
    elif issue_type == IssueType.SECURITY:
        return Priority.HIGH
    elif any(keyword in query_lower for keyword in high_keywords):
        return Priority.HIGH
    elif any(keyword in query_lower for keyword in medium_keywords):
        return Priority.MEDIUM
    else:
        return Priority.LOW

def get_resolution_time(priority: Priority) -> str:
    """Get estimated resolution time based on priority"""
    times = {
        Priority.CRITICAL: "1-2 hours",
        Priority.HIGH: "4-6 hours",
        Priority.MEDIUM: "1-2 business days",
        Priority.LOW: "3-5 business days"
    }
    return times.get(priority, "To be determined")

def get_fallback_response(query: str, issue_type: IssueType) -> tuple[str, List[str], float]:
    """Get fallback response when OpenAI is not available"""
    query_lower = query.lower()
    steps = []
    
    if issue_type in FALLBACK_RESPONSES:
        category = FALLBACK_RESPONSES[issue_type]
        
        # Check for keyword matches
        for keyword, response in category.get("keywords", {}).items():
            if keyword in query_lower:
                # Parse steps from response
                if ":" in response:
                    parts = response.split(":", 1)
                    main_response = parts[0]
                    step_text = parts[1] if len(parts) > 1 else ""
                    steps = [s.strip() for s in step_text.split(",") if s.strip()]
                else:
                    main_response = response
                
                return response, steps, 0.7
        
        # Return default response for category
        default = category["default"]
        if ":" in default:
            parts = default.split(":", 1)
            step_text = parts[1] if len(parts) > 1 else ""
            steps = [s.strip() for s in step_text.split(",") if s.strip()]
        
        return default, steps, 0.5
    
    # Generic fallback
    return "I can help you with IT issues including hardware, software, network, account, security, email, and printer problems. Please provide more details about your specific issue.", [], 0.3

async def get_openai_response(query: str, issue_type: IssueType, system_info: Dict[str, Any]) -> tuple[str, List[str], float]:
    """Get response from OpenAI API"""
    try:
        system_prompt = f"""You are an IT Support Assistant specializing in {issue_type.value} issues. 
        Provide clear, step-by-step troubleshooting guidance.
        Be technical but accessible. Include specific commands or settings when relevant.
        Always prioritize security and data safety in your recommendations."""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ]
        
        if system_info:
            messages.append({"role": "system", "content": f"System info: {json.dumps(system_info)}"})
        
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.7,
            max_tokens=500
        )
        
        response_text = response.choices[0].message.content
        
        # Extract steps if numbered list is present
        steps = []
        lines = response_text.split('\n')
        for line in lines:
            if line.strip() and (line.strip()[0].isdigit() or line.strip().startswith('-')):
                steps.append(line.strip())
        
        return response_text, steps, 0.95
    
    except Exception as e:
        logger.error(f"OpenAI API error: {e}")
        # Fall back to local response
        return get_fallback_response(query, issue_type)

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "service": "IT Assistant",
        "status": "operational",
        "openai_enabled": USE_OPENAI,
        "version": "1.0.0"
    }

@app.post("/query", response_model=ITResponse)
async def process_query(it_query: ITQuery, background_tasks: BackgroundTasks):
    """Process an IT support query and return a response"""
    try:
        # Classify issue if not provided
        issue_type = it_query.issue_type or classify_issue(it_query.query)
        
        # Determine priority
        priority = it_query.priority or determine_priority(it_query.query, issue_type)
        
        # Generate ticket ID for high priority issues
        ticket_id = None
        if priority in [Priority.HIGH, Priority.CRITICAL]:
            ticket_id = generate_ticket_id()
            logger.info(f"Created ticket {ticket_id} for {priority.value} priority {issue_type.value} issue")
        
        # Get response
        if USE_OPENAI:
            response_text, steps, confidence = await get_openai_response(
                it_query.query, 
                issue_type, 
                it_query.system_info or {}
            )
            used_ai = True
        else:
            response_text, steps, confidence = get_fallback_response(it_query.query, issue_type)
            used_ai = False
        
        # Get estimated resolution time
        resolution_time = get_resolution_time(priority)
        
        return ITResponse(
            response=response_text,
            issue_type=issue_type,
            priority=priority,
            ticket_id=ticket_id,
            steps=steps,
            confidence=confidence,
            used_ai=used_ai,
            estimated_resolution_time=resolution_time
        )
    
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/categories")
async def get_categories():
    """Get available issue categories"""
    return {
        "categories": [
            {
                "type": itype.value,
                "description": FALLBACK_RESPONSES.get(itype, {}).get("default", "")[:100] + "...",
                "common_issues": list(FALLBACK_RESPONSES.get(itype, {}).get("keywords", {}).keys())
            }
            for itype in IssueType
        ]
    }

@app.get("/priorities")
async def get_priorities():
    """Get priority levels and resolution times"""
    return {
        "priorities": [
            {
                "level": priority.value,
                "resolution_time": get_resolution_time(priority)
            }
            for priority in Priority
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
        "categories_loaded": len(FALLBACK_RESPONSES),
        "ticket_system": "active"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)