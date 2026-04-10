"""LLM client - calls a hosted proxy so the user needs no API key.

The proxy is deployed on Hugging Face Spaces and holds the Anthropic API key
as a server-side secret. See `proxy/` for the proxy source code.

For local development, you can run the proxy locally:
    cd proxy && uvicorn app:app --port 7860
And set LLM_PROXY_URL=http://localhost:7860 in your .env
"""

import json
import re
import httpx
from app.config import LLM_PROXY_URL, LLM_MODEL


# Reusable HTTP client with sensible timeouts
_http_client = httpx.Client(timeout=httpx.Timeout(60.0, connect=10.0))


def call_llm(prompt: str, system_prompt: str = "", model: str = None,
             temperature: float = 0.0, json_mode: bool = False) -> str:
    """Call the LLM proxy and return the response text."""
    payload = {
        "prompt": prompt,
        "system_prompt": system_prompt,
        "temperature": temperature,
        "json_mode": json_mode,
    }
    if model:
        payload["model"] = model

    url = f"{LLM_PROXY_URL.rstrip('/')}/llm"
    try:
        response = _http_client.post(url, json=payload)
    except httpx.RequestError as e:
        raise RuntimeError(
            f"Failed to reach LLM proxy at {url}. "
            f"Check LLM_PROXY_URL in your .env. Error: {e}"
        )

    if response.status_code == 429:
        raise RuntimeError(f"LLM proxy rate limit exceeded: {response.json().get('detail', '')}")
    if response.status_code != 200:
        try:
            detail = response.json().get("detail", response.text)
        except Exception:
            detail = response.text
        raise RuntimeError(f"LLM proxy returned {response.status_code}: {detail}")

    data = response.json()
    return data["text"]


def call_llm_json(prompt: str, system_prompt: str = "", model: str = None) -> dict:
    """Call the LLM and parse the response as JSON.

    Handles cases where the model wraps JSON in markdown code blocks.
    """
    raw = call_llm(prompt, system_prompt, model, temperature=0.0, json_mode=True)
    cleaned = raw.strip()

    # Remove markdown code fences if present
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # Last-ditch attempt: extract first JSON object from the text
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise
