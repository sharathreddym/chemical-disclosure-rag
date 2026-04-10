"""Agent 7: Answer Synthesizer Agent.

Converts retrieved evidence into a final natural language response
with citations, query plan summary, and confidence assessment.

CRITICAL: The synthesizer is given the FULL evidence record (all fields).
It MUST only reference fields that exist in the provided records.
"""

from app.agents.base import BaseAgent, PipelineState
from app.utils.llm import call_llm, call_llm_json


SYSTEM_PROMPT = """You are an answer synthesizer for a cosmetics chemicals database assistant.
Your job is to produce a clear, accurate, well-cited answer based ONLY on the provided evidence records.

ABSOLUTE RULES (violating these is a critical failure):
1. ONLY make claims supported by the evidence records below. Do NOT invent, infer, or hallucinate any data.
2. Only reference fields that are PRESENT in the evidence records. If a field is empty or missing, do not mention it.
3. NEVER mention companies, products, chemicals, dates, IDs, or categories that don't appear verbatim in the evidence.
4. Cite each claim with the record identifier in this format: [CDPHId: X, ChemicalId: Y]
5. Use the EXACT spelling from the evidence (do not "correct" or normalize names).
6. If the evidence count seems too small to fully answer, say so explicitly. Do not guess what other records might contain.
7. For grouping/aggregation queries, only group what's in the evidence.
8. If the evidence was TRUNCATED (more matches exist beyond what's shown), explicitly tell the user
   that the answer reflects only the first N rows and suggest a narrower filter.
9. If the SQL was flagged as BROAD, mention that the answer is high-level and recommend the user
   add filters (brand, category, year, CAS number).

OUTPUT STRUCTURE (return as JSON):
{
    "answer": "**Short Answer**: 1-2 sentence direct answer\\n\\n**Details**: ...with inline citations\\n\\n**Note**: caveats",
    "followups": ["2 to 3 short follow-up questions the user might naturally ask next"]
}

REMEMBER: If you're tempted to mention something not in the evidence, STOP. The validator will catch it and you'll be retried."""


class SynthesizerAgent(BaseAgent):
    name = "AnswerSynthesizer"

    def run(self, state: PipelineState) -> PipelineState:
        state.add_plan_step(
            agent=self.name,
            action="Synthesizing final answer",
            details=f"Evidence: {len(state.merged_evidence)} validated rows, exact_lookup={state.is_exact_lookup}"
        )

        # Handle no-evidence case
        if not state.merged_evidence and state.evidence_strength == "none":
            state.final_answer = self._no_evidence_answer(state)
            state.confidence = "low"
            state.warnings.append("No matching records found in the database.")
            state.query_plan[-1].result_summary = "No evidence found; generated advisory response."
            return state

        # Format evidence with ALL fields - so the synthesizer can reference them safely
        evidence_text = self._format_full_evidence(state.merged_evidence[:30])

        # Tell the synthesizer about the lookup mode and validator behavior
        mode_note = (
            "MODE: Exact entity lookup. The user requested a specific entity, and SQL "
            "returned exact matches. Vector candidates were intentionally excluded. "
            "Answer ONLY about the rows below."
            if state.is_exact_lookup else
            "MODE: Hybrid retrieval. Evidence was merged from SQL + filtered vector results."
        )

        # Guardrail flags from upstream
        truncation_note = ""
        if state.sql_truncated:
            truncation_note = (
                f"\n\n*** IMPORTANT: SQL was TRUNCATED. The query returned more rows than the LIMIT "
                f"({len(state.sql_results)} shown). Tell the user the answer is partial and suggest "
                f"a narrower filter (brand, category, year, CAS number). ***"
            )
        broad_note = ""
        if state.sql_is_broad:
            broad_note = (
                "\n\n*** IMPORTANT: The SQL query was very broad (no WHERE filter, no aggregation). "
                "Recommend the user add filters to get a more targeted answer. ***"
            )

        prompt = f"""User question: "{state.sanitized_query or state.user_query}"

Intent: {state.intent.value}
Evidence strength: {state.evidence_strength}
{mode_note}{truncation_note}{broad_note}

SQL query used: {state.sql_query or 'N/A'}
SQL error: {state.sql_error or 'None'}
Conflicts detected: {state.conflicts if state.conflicts else 'None'}

VALIDATED EVIDENCE RECORDS ({len(state.merged_evidence)} total, showing up to 30):
Each record below is a complete row from the database. Only reference fields that appear here.

{evidence_text}

Based ONLY on the records above, answer the user's question. Do not mention any field
that is not present (or is empty) in these records. Do not mention any product, company,
chemical, or date that is not above.

Return JSON with keys: "answer" (the structured response) and "followups" (2-3 short follow-up questions).
"""

        try:
            result = call_llm_json(prompt=prompt, system_prompt=SYSTEM_PROMPT)
            state.final_answer = result.get("answer", "")
            followups = result.get("followups", [])
            if isinstance(followups, list):
                state.suggested_followups = [str(f) for f in followups[:3]]
        except Exception:
            # Fall back to plain-text answer if JSON parsing fails
            state.final_answer = call_llm(prompt=prompt, system_prompt=SYSTEM_PROMPT.replace(
                'OUTPUT STRUCTURE (return as JSON):', 'OUTPUT STRUCTURE:'
            ))
            state.suggested_followups = []

        # Return ALL merged evidence (no cap) so the UI can show every row that
        # supports the answer. The LLM prompt only sees the first 30 for context
        # budget, but the UI/CLI gets the full set for transparency.
        state.evidence_list = [r.to_dict() for r in state.merged_evidence]

        # Set confidence
        if state.evidence_strength == "strong" and not state.conflicts:
            state.confidence = "high"
        elif state.evidence_strength in ("strong", "moderate"):
            state.confidence = "medium"
        else:
            state.confidence = "low"
            state.warnings.append(f"Evidence strength is {state.evidence_strength}.")

        if state.sql_error:
            state.warnings.append(f"SQL query encountered an error: {state.sql_error}")
        if state.conflicts:
            state.warnings.extend(state.conflicts)
        if state.sql_truncated:
            state.warnings.append(
                f"Result set was truncated. Showing the first {len(state.sql_results)} rows; "
                f"more matches exist in the database. Try narrowing by brand, category, year, or CAS number."
            )
        if state.sql_is_broad:
            state.warnings.append(
                "Query is very broad (no filter, no aggregation). "
                "Add filters (brand, category, date, CAS number) for a more targeted answer."
            )

        state.query_plan[-1].result_summary = f"Answer generated. Confidence: {state.confidence}."
        return state

    def _format_full_evidence(self, records) -> str:
        """Format evidence with ALL fields, one record per block, so the synthesizer
        can reference any field without hallucinating."""
        blocks = []
        for i, r in enumerate(records, 1):
            d = r.to_full_dict()
            # Only include non-empty fields in the display, but show them all
            lines = [f"--- Record [{i}] ---"]
            for key in [
                "CDPHId", "CSFId", "CSF", "ProductName",
                "CompanyId", "CompanyName", "BrandName",
                "ChemicalId", "ChemicalName", "CasId", "CasNumber",
                "PrimaryCategoryId", "PrimaryCategory",
                "SubCategoryId", "SubCategory",
                "InitialDateReported", "MostRecentDateReported",
                "DiscontinuedDate", "ChemicalDateRemoved",
                "ChemicalCreatedAt", "ChemicalUpdatedAt", "ChemicalCount",
            ]:
                value = d.get(key, "")
                if value not in (None, "", "None"):
                    lines.append(f"  {key}: {value}")
            lines.append(f"  (source: {d.get('source', 'sql')}, score: {d.get('relevance_score', 1.0)})")
            blocks.append("\n".join(lines))
        return "\n\n".join(blocks)

    def _no_evidence_answer(self, state: PipelineState) -> str:
        query = state.sanitized_query or state.user_query
        parts = ["**Short Answer**: No matching records were found in the database for your query.\n"]
        parts.append("\n**Details**: ")
        if state.sql_error:
            parts.append(f"The SQL query encountered an error: {state.sql_error}. ")
        if state.entities.chemical_name:
            parts.append(f"No records found for chemical '{state.entities.chemical_name}'. ")
        if state.entities.cas_number:
            parts.append(f"No records found for CAS number '{state.entities.cas_number}'. ")
        if state.entities.brand_name:
            parts.append(f"No records found for brand '{state.entities.brand_name}'. ")
        if state.entities.product_name:
            parts.append(f"No records found for product '{state.entities.product_name}'. ")
        parts.append(
            "\n\n**Note**: The entity names may not exactly match the database. "
            "Try using partial names, alternative spellings, or check the CAS number format."
        )
        return "".join(parts)
