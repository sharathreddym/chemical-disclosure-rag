"""
LLM Proxy Server for Chemical Disclosure RAG.

Forwards LLM requests to Anthropic Claude using a server-side API key.
Deployed on Hugging Face Spaces so the user never needs an API key.

Endpoints:
    POST /llm     - Send a prompt, get a Claude response
    GET  /health  - Health check
    GET  /        - Info page
"""

import os
import time
import logging
from collections import defaultdict, deque
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from anthropic import Anthropic, APIError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("llm-proxy")

# --- Config from environment ---
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
DEFAULT_MODEL = os.getenv("LLM_MODEL", "claude-sonnet-4-20250514")
RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "30"))
RATE_LIMIT_PER_DAY = int(os.getenv("RATE_LIMIT_PER_DAY", "500"))

if not ANTHROPIC_API_KEY:
    logger.warning("ANTHROPIC_API_KEY not set. Proxy will return errors.")

client = Anthropic(api_key=ANTHROPIC_API_KEY) if ANTHROPIC_API_KEY else None

# --- Simple in-memory rate limiter (per IP) ---
_minute_buckets: dict[str, deque] = defaultdict(deque)
_day_buckets: dict[str, deque] = defaultdict(deque)


def check_rate_limit(client_ip: str) -> Optional[str]:
    """Returns error message if rate limit exceeded, None otherwise."""
    now = time.time()
    minute_ago = now - 60
    day_ago = now - 86400

    # Clean old entries
    mb = _minute_buckets[client_ip]
    while mb and mb[0] < minute_ago:
        mb.popleft()
    db = _day_buckets[client_ip]
    while db and db[0] < day_ago:
        db.popleft()

    if len(mb) >= RATE_LIMIT_PER_MINUTE:
        return f"Rate limit exceeded: {RATE_LIMIT_PER_MINUTE} requests per minute"
    if len(db) >= RATE_LIMIT_PER_DAY:
        return f"Daily quota exceeded: {RATE_LIMIT_PER_DAY} requests per day"

    mb.append(now)
    db.append(now)
    return None


# --- Request / Response models ---
class LLMRequest(BaseModel):
    prompt: str = Field(..., max_length=20000)
    system_prompt: str = Field("", max_length=10000)
    model: Optional[str] = None
    temperature: float = Field(0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(4096, ge=1, le=8192)
    json_mode: bool = False


class LLMResponse(BaseModel):
    text: str
    model: str
    input_tokens: int
    output_tokens: int


# --- FastAPI app ---
app = FastAPI(
    title="Chemical Disclosure RAG - LLM Proxy",
    description="LLM proxy that forwards prompts to Claude. Used by the multi-agent orchestrator.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {
        "service": "Chemical Disclosure RAG - LLM Proxy",
        "status": "running" if client else "error: ANTHROPIC_API_KEY not configured",
        "model": DEFAULT_MODEL,
        "endpoints": {
            "POST /llm": "Send a prompt, receive Claude response",
            "GET /health": "Health check",
        },
        "rate_limits": {
            "per_minute": RATE_LIMIT_PER_MINUTE,
            "per_day": RATE_LIMIT_PER_DAY,
        },
    }


@app.get("/health")
def health():
    return {
        "status": "ok" if client else "error",
        "anthropic_configured": bool(client),
        "model": DEFAULT_MODEL,
    }


@app.post("/llm", response_model=LLMResponse)
def call_llm(payload: LLMRequest, request: Request):
    if client is None:
        raise HTTPException(status_code=503, detail="LLM service not configured")

    # Rate limiting
    client_ip = request.headers.get("x-forwarded-for", request.client.host).split(",")[0].strip()
    rate_error = check_rate_limit(client_ip)
    if rate_error:
        raise HTTPException(status_code=429, detail=rate_error)

    # Build system prompt
    system_prompt = payload.system_prompt
    if payload.json_mode and system_prompt:
        system_prompt += "\n\nIMPORTANT: You MUST respond with valid JSON only. No markdown, no explanation, just the JSON object."

    try:
        kwargs = {
            "model": payload.model or DEFAULT_MODEL,
            "max_tokens": payload.max_tokens,
            "temperature": payload.temperature,
            "messages": [{"role": "user", "content": payload.prompt}],
        }
        if system_prompt:
            kwargs["system"] = system_prompt

        response = client.messages.create(**kwargs)
        text = response.content[0].text

        return LLMResponse(
            text=text,
            model=response.model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )
    except APIError as e:
        logger.error(f"Anthropic API error: {e}")
        raise HTTPException(status_code=502, detail=f"Upstream LLM error: {str(e)[:200]}")
    except Exception as e:
        logger.exception("Unexpected error in /llm")
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)[:200]}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "7860")))
