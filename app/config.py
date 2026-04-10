import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

# LLM proxy URL - defaults to a public Hugging Face Space.
# For local development, set LLM_PROXY_URL=http://localhost:7860 in .env
# after running the proxy locally with `cd proxy && uvicorn app:app --port 7860`
LLM_PROXY_URL = os.getenv(
    "LLM_PROXY_URL",
    "https://sharath88-chemical-rag-proxy.hf.space",
)
LLM_MODEL = os.getenv("LLM_MODEL", "claude-sonnet-4-20250514")

# Data paths
DATABASE_PATH = os.getenv("DATABASE_PATH", str(BASE_DIR / "data" / "cosmetics.db"))
CHROMA_PATH = os.getenv("CHROMA_PATH", str(BASE_DIR / "data" / "chroma_store_v2"))
CSV_PATH = os.getenv("CSV_PATH", str(BASE_DIR / "interviewtestdataset.csv"))

# Agent settings
MAX_SQL_RESULTS = 50
MAX_VECTOR_RESULTS = 20
VALIDATION_MAX_RETRIES = 1
