"""Agent 8: Output Validator / Citation Checker Agent.

Verifies that the final answer is grounded in the FULL evidence,
that cited IDs exist, and that no unsupported claims were made.
Implements a maker-checker pattern with max 1 retry.
"""

from app.agents.base import BaseAgent, PipelineState
from app.utils.llm import call_llm_json
from app.config import VALIDATION_MAX_RETRIES


SYSTEM_PROMPT = """You are an output validator for a cosmetics chemicals database assistant.
Your job is to verify that the generated answer is fully grounded in the FULL evidence records provided.

VALIDATION CHECKS:
1. Every factual claim in the answer is supported by at least one evidence record.
2. Any cited IDs (CDPHId, ChemicalId, CSFId, CAS numbers) actually appear in the evidence.
3. Counts and dates mentioned in the answer match the evidence.
4. No information was fabricated or hallucinated beyond what the evidence supports.
5. The answer does not invent companies, products, chemicals, or categories not in evidence.

IMPORTANT NUANCES:
- The evidence records below contain ALL fields from the database. If the answer
  references a field that exists in any evidence record (even with the same value),
  consider it grounded.
- Empty strings ("") in the evidence mean the field is missing for that record.
  If the answer says "no discontinuation date" and DiscontinuedDate is empty in
  the evidence, that is supported - count it as grounded.
- The answer may use different formatting (bold, bullets) - that is fine.
- Minor rephrasing of values (e.g., "13463-67-7" vs "CAS 13463-67-7") is fine.

Respond with JSON:
{
    "is_grounded": true/false,
    "issues": ["list of specific grounding issues - cite which claim is unsupported, empty if none"],
    "severity": "none, minor, or major",
    "suggested_fix": "brief suggestion for fixing issues, empty if none"
}

Severity guide:
- "none": Answer is fully grounded
- "minor": Small formatting/wording issues but facts are correct
- "major": Answer claims facts not in evidence (companies, dates, IDs that don't exist)

Be precise. Quote the exact phrase from the answer that is unsupported."""


class ValidatorAgent(BaseAgent):
    name = "OutputValidator"

    def run(self, state: PipelineState) -> PipelineState:
        state.add_plan_step(
            agent=self.name,
            action="Validating answer grounding and citations",
            details=f"Retry count: {state.retry_count}/{VALIDATION_MAX_RETRIES}"
        )

        if not state.final_answer:
            state.is_grounded = False
            state.validation_issues = ["No answer was generated."]
            state.query_plan[-1].result_summary = "Failed: no answer to validate"
            return state

        # Build FULL evidence dump (all fields per record)
        evidence_text = self._build_full_evidence(state)

        prompt = f"""Validate this answer against the FULL evidence records:

ANSWER:
{state.final_answer}

EVIDENCE RECORDS ({len(state.merged_evidence)} total):
Each record below contains ALL fields from the database. Empty fields are shown as "".

{evidence_text}

CONTEXT:
- SQL query used: {state.sql_query or 'N/A'}
- Evidence strength: {state.evidence_strength}
- Exact lookup mode: {state.is_exact_lookup}
- Known conflicts: {state.conflicts if state.conflicts else 'None'}

Check if every factual claim in the answer is supported by at least one record above.
Be strict about unsupported claims (companies, dates, IDs, categories that don't appear).
Be lenient about formatting and minor rephrasing.
"""

        result = call_llm_json(prompt=prompt, system_prompt=SYSTEM_PROMPT)

        state.is_grounded = result.get("is_grounded", False)
        issues = result.get("issues", [])
        severity = result.get("severity", "none")
        suggested_fix = result.get("suggested_fix", "")

        state.validation_issues = issues

        if not state.is_grounded and severity == "major" and state.retry_count < VALIDATION_MAX_RETRIES:
            # Trigger a retry by re-routing to synthesizer
            state.retry_count += 1
            state.warnings.append(
                f"Answer re-generated due to grounding issues: {'; '.join(issues[:3])}"
            )
            state.query_plan[-1].result_summary = (
                f"Validation failed (severity: {severity}). Retry {state.retry_count}. "
                f"Issues: {'; '.join(issues[:2])}"
            )
        else:
            if issues and severity == "minor":
                state.is_grounded = True  # Accept with minor issues
                state.warnings.extend([f"Minor validation note: {i}" for i in issues])

            if not state.is_grounded and state.retry_count >= VALIDATION_MAX_RETRIES:
                # Out of retries - downgrade confidence and accept
                state.warnings.append(
                    f"Validator could not fully ground the answer after {state.retry_count} retries. "
                    f"Treat the answer as 'best-effort'. Issues: {'; '.join(issues[:3])}"
                )
                state.confidence = "low"

            status = "Passed" if state.is_grounded else f"Failed: {'; '.join(issues[:2])}"
            state.query_plan[-1].result_summary = f"Validation: {status}"

        return state

    def _build_full_evidence(self, state: PipelineState) -> str:
        """Build a complete dump of every evidence record with all fields."""
        blocks = []
        for i, r in enumerate(state.merged_evidence[:25], 1):
            d = r.to_full_dict()
            lines = [f"--- Evidence Record [{i}] ---"]
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
                lines.append(f"  {key}: {value if value not in (None, '') else '(empty)'}")
            blocks.append("\n".join(lines))
        return "\n\n".join(blocks) if blocks else "No evidence records."
