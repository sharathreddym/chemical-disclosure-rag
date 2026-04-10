# Chemical Disclosure RAG - Multi-Agent Orchestrator

A multi-agent orchestration system that answers natural-language questions over the
[California Safe Cosmetics Program (CSCP) chemicals dataset](https://data.ca.gov/dataset/chemicals-in-cosmetics).

The system handles **hybrid questions** that combine structured filtering, entity resolution,
fuzzy semantic matching, and grounded narrative summarization with citations back to record IDs.

## Architecture: 8-Agent Pipeline

```
User Question
     ↓
[1] Input Guardrail + Triage      - Scope check, malformed input detection
     ↓
[2] Planner / Router              - Intent classification, retrieval strategy
     ↓
[3] Entity & Constraint Resolver  - Extract chemicals, CAS, brands, dates, filters
     ↓
[4] Structured Query (SQL)  ──┐   - Generates SQL, executes against SQLite
     ↓                         │
[5] Semantic Retrieval (Vec)  ←┘   - ChromaDB fuzzy match for partial names
     ↓
[6] Evidence Merger               - Dedup, conflict resolution, strength scoring
     ↓
[7] Answer Synthesizer            - Grounded NL response with citations
     ↓
[8] Output Validator              - Maker-checker grounding verification
     ↓
Final Answer + Evidence + Query Plan + Confidence
```

| Agent | LLM-based? | Purpose |
|---|---|---|
| 1. Guardrail | Lightweight LLM | Filter out-of-scope or malformed queries |
| 2. Planner / Router | LLM | Classify intent (lookup/list/compare/summarize/trend) and pick strategy (sql/vector/hybrid) |
| 3. Entity Resolver | LLM | Extract structured entities and constraints |
| 4. SQL Agent | LLM (generates) + Deterministic (executes) | Convert query → SQL → execute against SQLite |
| 5. Semantic Agent | Deterministic (ChromaDB) | Fuzzy match against entity vector index |
| 6. Evidence Merger | Deterministic | Merge SQL + vector results, detect conflicts |
| 7. Synthesizer | LLM | Build the final cited natural-language answer |
| 8. Validator | LLM | Verify the answer is fully grounded; bounded retry |

The pipeline uses **LangGraph** for stateful orchestration with conditional routing
and a maker-checker retry loop on the validator.

## Tech Stack

- **Python 3.11+** with **LangGraph** for orchestration
- **SQLite** for the structured store (114,635 rows from the CSCP dataset)
- **ChromaDB** with local ONNX embeddings (`all-MiniLM-L6-v2`) for semantic retrieval
- **Anthropic Claude** (Sonnet 4) as the LLM, accessed through a hosted proxy
- **Streamlit** for the optional web UI; **CLI** for the primary interface

## Quick Start (for evaluators)

The interviewer can run this with **zero signup and no API key**.
The LLM is accessed through a hosted proxy on Hugging Face Spaces; the proxy holds the
Anthropic API key as a server-side secret, so you don't need one.

```bash
# 1. Clone
git clone https://github.com/sharathreddym/chemical-disclosure-rag
cd chemical-disclosure-rag

# 2. Install (lightweight - no LLM model downloads)
pip install -r requirements.txt

# 3. One-time data ingestion
#    - Loads CSV into SQLite (~30 sec)
#    - Builds ChromaDB vector index (~3-5 min, ONNX runs locally)
python setup_data.py

# 4. Run a query
python -m app.main -q "Which products contain Titanium dioxide?" -v

# Or interactive mode
python -m app.main

# Or web UI
streamlit run app/streamlit_app.py
```

### Sample queries

```bash
python -m app.main -q "Which products contain Titanium dioxide?"
python -m app.main -q "What chemicals are reported for AVON in Lip Color products?"
python -m app.main -q "Show products discontinued in 2020 that had Formaldehyde"
python -m app.main -q "Summarize reporting trends for Hair Care Products"
python -m app.main -q "Find products with CAS number 75-07-0"
python -m app.main -q "Which companies have the most reported chemicals?"
```

### Output format

Every query returns:
- **Final answer** (natural language, with inline citations)
- **Evidence list** (CDPHId, ChemicalId, ProductName, CompanyName, etc.)
- **Query plan** (high-level execution trace through all 8 agents)
- **Confidence + warnings**
- **SQL query used** (if any)

Use `-v` for verbose output, `--json` for machine-readable output.

## Project Structure

```
.
├── app/
│   ├── agents/                    # The 8 agents
│   │   ├── base.py                # PipelineState, BaseAgent
│   │   ├── guardrail.py           # Agent 1
│   │   ├── planner.py             # Agent 2
│   │   ├── entity_resolver.py     # Agent 3
│   │   ├── sql_agent.py           # Agent 4
│   │   ├── semantic_agent.py      # Agent 5
│   │   ├── evidence_merger.py     # Agent 6
│   │   ├── synthesizer.py         # Agent 7
│   │   └── validator.py           # Agent 8
│   ├── orchestrator/
│   │   └── pipeline.py            # LangGraph workflow with conditional routing
│   ├── data/
│   │   ├── schema.py              # SQLite schema + column metadata
│   │   └── loader.py              # CSV → SQLite + ChromaDB ingestion
│   ├── utils/
│   │   └── llm.py                 # HTTP client to the LLM proxy
│   ├── config.py                  # Env-driven config
│   ├── main.py                    # CLI entry point
│   └── streamlit_app.py           # Streamlit web UI
├── proxy/                         # Deployed separately on Hugging Face Spaces
│   ├── app.py                     # FastAPI proxy that holds the Anthropic key
│   ├── Dockerfile                 # HF Spaces deployment image
│   ├── requirements.txt
│   └── README.md
├── interviewtestdataset.csv       # The CSCP chemicals dataset
├── setup_data.py                  # One-time ingestion script
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

## Design Decisions

| Decision | Why |
|---|---|
| **LangGraph for orchestration** | Native conditional routing (planner picks SQL/vector/hybrid) and bounded retry loops (validator's maker-checker) |
| **Only 3-4 agents are LLM-based** | Cost / latency control. SQL execution, vector search, and evidence merging are deterministic |
| **Local ONNX embeddings** | Zero cost, zero API key, offline-capable, fast enough for ~8K entity index |
| **Smart entity index in ChromaDB** | Indexes ~8.4K unique entities (chemicals, companies, brands, products, categories) instead of all 114K rows. Same recall, much faster ingestion |
| **Hosted LLM proxy** | Lets evaluators run the system without an API key. Anthropic key lives as a HF Spaces secret |
| **Maker-checker validation** | Catches ungrounded claims before they reach the user. Bounded to 1 retry to avoid loops |
| **Shared `PipelineState` dataclass** | Clean state passing between agents; full traceability for the query plan |

## How the LLM proxy works

The main app's `app/utils/llm.py` sends prompts via HTTPS to a FastAPI proxy
deployed on Hugging Face Spaces:

```
┌────────────────────────┐    HTTPS    ┌──────────────────────────┐
│ Local CLI / Streamlit  │ ──────────> │ HF Space (proxy/app.py)  │
│ (no API key needed)    │             │                          │
│                        │             │ Holds ANTHROPIC_API_KEY  │
│                        │ <────────── │ as a secret              │
│                        │  Claude     │ Calls Anthropic API      │
└────────────────────────┘  response   └──────────────────────────┘
```

The proxy enforces per-IP rate limiting (30 req/min, 500 req/day) so the demo can't
be abused. To run locally instead, see "Local proxy" below.

## Deploying the proxy yourself

If you want to run your own proxy (e.g., to use a different model or avoid rate limits):

1. Create a new Hugging Face Space (Docker SDK)
2. Upload the contents of `proxy/` to the space
3. Add `ANTHROPIC_API_KEY` as a Space secret in Settings
4. Wait for the build to complete (~3 min)
5. Update `LLM_PROXY_URL` in your `.env` to point at your space:
   `LLM_PROXY_URL=https://sharath88-chemical-rag-proxy.hf.space`

## Local proxy (for development)

```bash
cd proxy
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...
uvicorn app:app --port 7860
```

Then in the main project's `.env`:
```
LLM_PROXY_URL=http://localhost:7860
```

## License

Built as an interview assessment for the AI Engineer role.
