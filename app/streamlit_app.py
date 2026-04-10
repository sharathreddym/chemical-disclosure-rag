"""Streamlit UI for the Multi-Agent Chemical Disclosure RAG system."""

import streamlit as st
import json
import os
import sys
import pandas as pd
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.orchestrator.pipeline import run_query
from app.data.loader import ingest_all
from app.config import DATABASE_PATH, CHROMA_PATH


def check_data_ready() -> bool:
    """Check if SQLite and ChromaDB stores exist."""
    return Path(DATABASE_PATH).exists() and Path(CHROMA_PATH).exists()


def main():
    st.set_page_config(
        page_title="Chemical Disclosure RAG",
        page_icon="🧪",
        layout="wide",
    )

    st.title("🧪 Chemical Disclosure RAG")
    st.caption("Multi-Agent Orchestrator for California Safe Cosmetics Dataset")

    # Sidebar
    with st.sidebar:
        st.header("System Info")
        st.markdown("""
        **8-Agent Pipeline:**
        1. Input Guardrail + Triage
        2. Planner / Router
        3. Entity & Constraint Resolver
        4. Structured Query (SQL)
        5. Semantic Retrieval (Vector)
        6. Evidence Merger
        7. Answer Synthesizer
        8. Output Validator
        """)

        data_ready = check_data_ready()
        if data_ready:
            st.success("Data stores ready")
        else:
            st.warning("Data not ingested yet")
            if st.button("Run Data Ingestion"):
                with st.spinner("Ingesting data into SQLite + ChromaDB..."):
                    ingest_all()
                st.success("Data ingestion complete!")
                st.rerun()

        st.divider()
        show_details = st.checkbox("Show detailed output", value=True)
        show_json = st.checkbox("Show raw JSON", value=False)

        st.divider()
        st.markdown("**Sample Questions:**")
        sample_questions = [
            "Which products contain Titanium dioxide?",
            "What chemicals are reported for AVON in Lip Color?",
            "Show products discontinued in 2020 that had Formaldehyde",
            "Summarize reporting trends for Hair Care Products",
            "Which companies have the most reported chemicals?",
            "Find products with CAS 75-07-0",
        ]
        for q in sample_questions:
            if st.button(q, key=f"sample_{q[:20]}"):
                st.session_state["question"] = q

    # Main area
    if not data_ready:
        st.info("Please run data ingestion from the sidebar before querying.")
        return

    # Question input
    question = st.text_input(
        "Ask a question about chemicals in cosmetics:",
        value=st.session_state.get("question", ""),
        placeholder="e.g., Which products contain Titanium dioxide?",
    )

    if st.button("Submit", type="primary") or (question and question != st.session_state.get("last_question", "")):
        if not question:
            st.warning("Please enter a question.")
            return

        st.session_state["last_question"] = question

        with st.spinner("Processing through 8-agent pipeline..."):
            try:
                result = run_query(question)
            except Exception as e:
                st.error(f"Error: {e}")
                return

        # Display results
        if not result["is_valid"]:
            st.error(f"Query rejected: {result['answer']}")
            return

        # Metrics row
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Intent", result["intent"])
        col2.metric("Confidence", result["confidence"])
        col3.metric("Validated Evidence", result.get("validated_evidence_count", 0),
                    help="Rows that actually support the answer (not low-similarity neighbors)")
        col4.metric("Vector Candidates", result.get("candidates_retrieved", 0),
                    help="Total semantic candidates retrieved before filtering")
        col5.metric("Mode", "exact" if result.get("is_exact_lookup") else "hybrid",
                    help="exact = SQL only (specific entity); hybrid = SQL + filtered vector")

        # Truncation / broad-query banners (top-of-results, hard to miss)
        if result.get("sql_truncated"):
            st.info(
                "This question matched more rows than the result limit. "
                f"Showing only the first {result.get('validated_evidence_count', 0)} rows. "
                "Try narrowing by brand, category, year, or CAS number."
            )
        if result.get("sql_is_broad"):
            st.warning(
                "This query is very broad (no filter, no aggregation). "
                "Add filters (brand, category, date, CAS number) for a more targeted answer."
            )

        # Answer
        st.markdown("---")
        st.markdown("### Answer")
        st.markdown(result["answer"])

        # Follow-up suggestions (clickable)
        if result.get("suggested_followups"):
            st.markdown("**Try a follow-up question:**")
            cols = st.columns(min(3, len(result["suggested_followups"])))
            for i, f in enumerate(result["suggested_followups"][:3]):
                if cols[i].button(f, key=f"followup_{i}"):
                    st.session_state["question"] = f
                    st.rerun()

        # Warnings
        if result["warnings"]:
            with st.expander("Warnings", expanded=False):
                for w in result["warnings"]:
                    st.warning(w)

        if show_details:
            # SQL Query
            if result["sql_query"]:
                with st.expander("🔍 SQL Query", expanded=False):
                    st.code(result["sql_query"], language="sql")

            # Evidence - shown as a sortable, filterable dataframe
            if result["evidence"]:
                with st.expander(f"📋 Evidence ({len(result['evidence'])} records)", expanded=True):
                    df = pd.DataFrame(result["evidence"])

                    # Preferred column order - most informative first
                    preferred = [
                        "CDPHId", "CSFId", "ChemicalId",
                        "ProductName", "CompanyName", "BrandName",
                        "ChemicalName", "CasNumber",
                        "PrimaryCategory", "SubCategory",
                        "InitialDateReported", "MostRecentDateReported",
                        "DiscontinuedDate", "ChemicalCount",
                        "ChemicalCreatedAt", "ChemicalUpdatedAt", "ChemicalDateRemoved",
                        "source", "relevance_score",
                    ]
                    cols_in_df = [c for c in preferred if c in df.columns]
                    other_cols = [c for c in df.columns if c not in preferred]
                    df = df[cols_in_df + other_cols]

                    st.caption(
                        f"Showing all {len(df)} validated evidence rows. "
                        f"Click any column header to sort. Use the search icon (top-right of the table) to filter."
                    )
                    st.dataframe(df, use_container_width=True, hide_index=False, height=400)

                    # Optional CSV download
                    csv = df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="Download evidence as CSV",
                        data=csv,
                        file_name="evidence.csv",
                        mime="text/csv",
                    )

            # Query Plan
            if result["query_plan"]:
                with st.expander("🗺️ Query Plan (execution trace)", expanded=False):
                    for i, step in enumerate(result["query_plan"], 1):
                        st.markdown(f"**Step {i} [{step['agent']}]**: {step['action']}")
                        if step["result"]:
                            st.caption(f"→ {step['result']}")

        if show_json:
            with st.expander("Raw JSON Output", expanded=False):
                st.json(result)

        # Grounding status
        st.markdown("---")
        if result["is_grounded"]:
            st.success(f"✅ Answer grounded | Evidence: {result['evidence_strength']}")
        else:
            st.warning(f"⚠️ Grounding check: {result['evidence_strength']}")


if __name__ == "__main__":
    main()
