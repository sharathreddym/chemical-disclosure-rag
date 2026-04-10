"""CLI entry point for the Multi-Agent Chemical Disclosure RAG system."""

import sys
import json
import argparse

# Force UTF-8 output on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from app.orchestrator.pipeline import run_query
from app.data.loader import ingest_all


def print_result(result: dict, verbose: bool = False):
    """Pretty-print the pipeline result."""
    print("\n" + "=" * 80)
    print("QUESTION:", result["question"])
    print("=" * 80)

    if not result["is_valid"]:
        print(f"\n[REJECTED] {result['answer']}")
        return

    mode = "exact-lookup" if result.get("is_exact_lookup") else "hybrid"
    print(f"\nIntent: {result['intent']} | Strategy: {result['retrieval_strategy']} | Mode: {mode}")
    print(f"Confidence: {result['confidence']} | "
          f"Validated Evidence: {result.get('validated_evidence_count', 0)} rows | "
          f"Vector candidates retrieved: {result.get('candidates_retrieved', 0)}")

    print(f"\n{'-' * 80}")
    print("ANSWER:")
    print(f"{'-' * 80}")
    print(result["answer"])

    if result["warnings"]:
        print(f"\n{'-' * 80}")
        print("WARNINGS:")
        for w in result["warnings"]:
            print(f"  ! {w}")

    if result["sql_query"]:
        print(f"\n{'-' * 80}")
        print("SQL QUERY:")
        print(f"  {result['sql_query']}")

    if verbose and result["evidence"]:
        print(f"\n{'-' * 80}")
        print(f"EVIDENCE ({len(result['evidence'])} records):")
        for i, e in enumerate(result["evidence"][:10], 1):
            print(f"  [{i}] CDPHId={e.get('CDPHId', 'N/A')}, "
                  f"Product={e.get('ProductName', 'N/A')}, "
                  f"Chemical={e.get('ChemicalName', 'N/A')}, "
                  f"CAS={e.get('CasNumber', 'N/A')}, "
                  f"Company={e.get('CompanyName', 'N/A')}")

    if verbose and result["query_plan"]:
        print(f"\n{'-' * 80}")
        print("QUERY PLAN (execution trace):")
        for i, step in enumerate(result["query_plan"], 1):
            print(f"  {i}. [{step['agent']}] {step['action']}")
            if step["result"]:
                print(f"     -> {step['result']}")

    if result.get("suggested_followups"):
        print(f"\n{'-' * 80}")
        print("FOLLOW-UP SUGGESTIONS:")
        for i, f in enumerate(result["suggested_followups"], 1):
            print(f"  {i}. {f}")

    print(f"\n{'-' * 80}")
    print(f"Grounded: {result['is_grounded']} | Evidence Strength: {result['evidence_strength']} | "
          f"Truncated: {result.get('sql_truncated', False)} | Broad: {result.get('sql_is_broad', False)}")
    print("=" * 80)


def interactive_mode(verbose: bool = False):
    """Run in interactive chat mode."""
    print("\n" + "=" * 64)
    print("  Chemical Disclosure RAG - Multi-Agent Orchestrator")
    print("  Type your question or 'quit' to exit.")
    print("=" * 64 + "\n")

    while True:
        try:
            question = input("\n>> Your question: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if question.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        if not question:
            continue

        print("\nProcessing through 8-agent pipeline...")
        try:
            result = run_query(question)
            print_result(result, verbose=verbose)
        except Exception as e:
            print(f"\nError: {e}")
            import traceback
            traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(
        description="Multi-Agent Chemical Disclosure RAG System"
    )
    parser.add_argument("--ingest", action="store_true",
                        help="Run data ingestion (CSV → SQLite + ChromaDB)")
    parser.add_argument("--query", "-q", type=str,
                        help="Run a single query and exit")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show detailed evidence and query plan")
    parser.add_argument("--json", action="store_true",
                        help="Output result as JSON (for --query mode)")

    args = parser.parse_args()

    if args.ingest:
        ingest_all()
        return

    if args.query:
        result = run_query(args.query)
        if args.json:
            print(json.dumps(result, indent=2, default=str))
        else:
            print_result(result, verbose=args.verbose)
        return

    # Default: interactive mode
    interactive_mode(verbose=args.verbose)


if __name__ == "__main__":
    main()
