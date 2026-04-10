"""One-time data ingestion script: CSV → SQLite + ChromaDB."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.data.loader import ingest_all

if __name__ == "__main__":
    ingest_all()
