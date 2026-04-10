---
title: Chemical Disclosure RAG LLM Proxy
emoji: 🧪
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# Chemical Disclosure RAG - LLM Proxy

This is a thin FastAPI proxy that forwards LLM requests to Anthropic Claude.
It is used by the [Chemical Disclosure RAG Multi-Agent Orchestrator](https://github.com/sharathreddym/chemical-disclosure-rag) so that the public client code does not need an Anthropic API key.

## Why this exists

The main project is a multi-agent orchestrator over a cosmetics chemicals dataset.
It uses Claude as the reasoning engine for several agents (planner, entity extractor,
SQL generator, synthesizer, validator).

To let evaluators run the system **without signing up anywhere or providing an API key**,
the Anthropic API key lives here as a Hugging Face Spaces secret. The client code
calls this proxy instead of Anthropic directly.

## Endpoints

- `POST /llm` - Send a prompt, receive a Claude response
- `GET /health` - Health check
- `GET /` - Service info

## Configuration (Hugging Face Spaces secrets)

Set these as repository secrets:
- `ANTHROPIC_API_KEY` (required) - your Anthropic API key
- `LLM_MODEL` (optional) - defaults to `claude-sonnet-4-20250514`
- `RATE_LIMIT_PER_MINUTE` (optional) - defaults to 30
- `RATE_LIMIT_PER_DAY` (optional) - defaults to 500

## Rate limiting

Per-IP in-memory rate limiting protects against runaway costs:
- 30 requests per minute
- 500 requests per day
