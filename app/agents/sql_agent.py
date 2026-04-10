"""Agent 4: Structured Query Agent (SQL).

Generates and executes SQL queries against the SQLite database.
Returns results with record IDs for citation.

Includes guardrails:
- Truncation detection: fetches LIMIT+1 rows to detect when results are capped
- Broad query warning: heuristic check for queries with no WHERE / no aggregation
- Synonym hints: known cosmetics term -> field-value mappings
- Group-first preference: nudges the LLM to summarize before listing
"""

import re
import sqlite3
from app.agents.base import BaseAgent, PipelineState
from app.utils.llm import call_llm_json
from app.config import DATABASE_PATH, MAX_SQL_RESULTS
from app.data.schema import SCHEMA_SUMMARY, COLUMN_DESCRIPTIONS


# Common cosmetics-domain synonyms the user might say.
# These hints are passed to the SQL agent prompt so the LLM picks the right
# canonical column / value when generating SQL.
SYNONYM_HINTS = """
COMMON SYNONYMS (use these to match user terms to canonical fields):
- "makeup" / "make-up" / "cosmetics" -> PrimaryCategory LIKE '%Makeup%'
- "lipstick" / "lip color" / "lip stick" -> SubCategory LIKE '%Lipstick%' OR SubCategory LIKE '%Lip Color%'
- "lip gloss" / "lip glosses" -> SubCategory LIKE '%Lip Gloss%'
- "nail polish" / "nail color" / "nail varnish" -> PrimaryCategory LIKE '%Nail%'
- "eyeshadow" / "eye shadow" -> SubCategory LIKE '%Eye Shadow%'
- "mascara" -> SubCategory LIKE '%Mascara%'
- "foundation" -> SubCategory LIKE '%Foundation%'
- "shampoo" / "shampoos" -> SubCategory LIKE '%Shampoo%'
- "conditioner" / "conditioners" -> SubCategory LIKE '%Conditioner%'
- "perfume" / "fragrance" / "cologne" -> PrimaryCategory LIKE '%Fragrance%'
- "skincare" / "skin care" / "moisturizer" -> PrimaryCategory LIKE '%Skin Care%'
- "sunscreen" / "sun screen" / "sunblock" -> SubCategory LIKE '%Sun%' OR ProductName LIKE '%sunscreen%'
- "white pigment" / "TiO2" -> ChemicalName LIKE '%Titanium dioxide%'
- "lead" / "Pb" -> ChemicalName LIKE '%Lead%'
- "carcinogen" / "carcinogenic" -> all rows (the entire DB is hazardous chemicals)
- "discontinued" / "removed" / "no longer sold" -> DiscontinuedDate != ''
- "active" / "current" / "still sold" -> DiscontinuedDate = ''
"""


SYSTEM_PROMPT = f"""You are a SQL query generator for a cosmetics chemicals SQLite database.
Generate a SQL query to answer the user's question based on the extracted entities.

{SCHEMA_SUMMARY}

COLUMN DETAILS:
{chr(10).join(f'- {k}: {v}' for k, v in COLUMN_DESCRIPTIONS.items())}

{SYNONYM_HINTS}

IMPORTANT RULES:
1. Table name: chemicals_in_cosmetics
2. Dates are stored as TEXT in MM/DD/YYYY format. For date comparisons use:
   - substr(DiscontinuedDate, 7, 4) to extract year
   - Use string comparison for full dates, but be aware of the MM/DD/YYYY format
3. ALWAYS use `SELECT *` so the answer can reference any column. Do NOT select a subset.
4. Choosing equality vs LIKE - this is critical for accuracy:
   - **Use `= 'X' COLLATE NOCASE`** ONLY when the user gave a value that is clearly a complete identifier
     (e.g., a full CAS number `13463-67-7`, or a unique-looking product name like `BlueFX`).
   - **Use `LIKE '%X%' COLLATE NOCASE`** when:
     a) The value looks partial or could be a substring (`titanium`, `lip`, `avon`)
     b) The value might be a misspelling
     c) The user used phrases like "containing", "with", "like", "similar to"
     d) You're searching ProductName/CompanyName/BrandName/ChemicalName and the term looks like a fragment
   - When in doubt, prefer LIKE - false positives are caught by the validator,
     but false negatives (missing the right row) cannot be recovered.
5. Use COLLATE NOCASE for case-insensitive comparisons.
6. LIMIT results to {MAX_SQL_RESULTS} rows. The runtime will fetch LIMIT+1 to detect truncation.
7. **GROUP-FIRST RULE**: For broad questions like "list all products", "how many",
   "which brands", or "summarize trends", prefer GROUP BY queries that return summary
   rows. Only return raw rows when the user has applied a specific filter.
   - "How many products contain X" -> SELECT COUNT(*), ChemicalName ... GROUP BY ChemicalName
   - "Which brands report Formaldehyde" -> SELECT BrandName, COUNT(*) ... GROUP BY BrandName ORDER BY 2 DESC
   - "Show me all products" -> WARN, ask for narrowing filter, but still produce a grouped view
8. For aggregation queries, when possible also expose the underlying columns
   (SELECT *, COUNT(*) OVER (PARTITION BY ...) AS cnt) so evidence can still be returned.
9. Empty string '' means NULL/missing in this dataset.
10. For chemical names: prefer LIKE because chemicals often have parenthetical suffixes
    (e.g., `Titanium dioxide` vs `Titanium dioxide (airborne, unbound particles)`).
11. For brand/company names: prefer LIKE because organizations have variants
    (e.g., `Avon` vs `AVON` vs `New Avon LLC`).
12. **DISTINCT for unique-entity questions**: Use `SELECT DISTINCT` (on specific
    columns, not `SELECT DISTINCT *`) when the user asks for unique entities:
    - "How many distinct products contain X" -> `SELECT DISTINCT ProductName ...`
    - "Which unique brands report Y" -> `SELECT DISTINCT BrandName ...`
    - "List the chemicals reported for Z" -> `SELECT DISTINCT ChemicalName ...`
    The dataset legitimately has multiple rows per (CDPHId, ChemicalId) when
    things like reporting dates or status differ - those are NOT duplicates,
    they are distinct reporting events. Do not collapse them with DISTINCT
    unless the user explicitly asks for unique entities.

Respond with JSON:
{{
    "sql_query": "the SQL query to execute",
    "explanation": "brief explanation of what the query does"
}}
"""


class SQLAgent(BaseAgent):
    name = "StructuredQuery"

    def run(self, state: PipelineState) -> PipelineState:
        query = state.sanitized_query or state.user_query
        entities = state.entities

        state.add_plan_step(
            agent=self.name,
            action="Generating and executing SQL query",
            details=f"Entities: {self._summarize_entities(entities)}"
        )

        # Build context for SQL generation
        entity_context = self._build_entity_context(entities)

        result = call_llm_json(
            prompt=(
                f"User question: \"{query}\"\n\n"
                f"Extracted entities and constraints:\n{entity_context}\n\n"
                f"Intent: {state.intent.value}\n\n"
                f"Generate a SQL query to answer this question. "
                f"Remember: use SELECT *, prefer LIKE for partial names, "
                f"and use GROUP BY for broad questions."
            ),
            system_prompt=SYSTEM_PROMPT,
        )

        sql_query = result.get("sql_query", "")
        explanation = result.get("explanation", "")
        state.sql_query = sql_query

        if not sql_query:
            state.sql_error = "No SQL query generated."
            state.query_plan[-1].result_summary = "Failed: no SQL generated"
            return state

        # Apply broad-query heuristic
        state.sql_is_broad = self._looks_too_broad(sql_query)

        # Truncation detection: rewrite the LIMIT to fetch one extra row
        executed_sql = self._inject_truncation_probe(sql_query, MAX_SQL_RESULTS)

        # Execute the query
        try:
            conn = sqlite3.connect(DATABASE_PATH)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(executed_sql)
            rows = cursor.fetchall()
            conn.close()

            all_rows = [dict(row) for row in rows]
            state.sql_total_seen = len(all_rows)

            if len(all_rows) > MAX_SQL_RESULTS:
                # Truncation detected - keep only MAX_SQL_RESULTS but flag it
                state.sql_results = all_rows[:MAX_SQL_RESULTS]
                state.sql_truncated = True
            else:
                state.sql_results = all_rows
                state.sql_truncated = False

            summary = (
                f"SQL returned {len(state.sql_results)} rows"
                + (f" (truncated; more matches exist)" if state.sql_truncated else "")
                + (f" [BROAD QUERY]" if state.sql_is_broad else "")
                + f". Query: {sql_query[:100]}..."
            )
            state.query_plan[-1].result_summary = summary
        except Exception as e:
            state.sql_error = str(e)
            state.query_plan[-1].result_summary = f"SQL error: {str(e)[:100]}"

        return state

    def _inject_truncation_probe(self, sql: str, limit: int) -> str:
        """Replace LIMIT N with LIMIT N+1 so we can detect truncation.

        If the SQL has no LIMIT clause, append one.
        """
        cleaned = sql.rstrip(";").strip()
        # Detect existing LIMIT
        limit_match = re.search(r"\bLIMIT\s+\d+\b", cleaned, flags=re.IGNORECASE)
        if limit_match:
            return re.sub(r"\bLIMIT\s+\d+\b", f"LIMIT {limit + 1}",
                          cleaned, flags=re.IGNORECASE)
        return f"{cleaned} LIMIT {limit + 1}"

    def _looks_too_broad(self, sql: str) -> bool:
        """Heuristic: a SQL query is 'too broad' if it has no WHERE clause
        AND no aggregation. Such queries usually return huge dumps."""
        s = sql.lower()
        has_where = " where " in s
        has_aggregate = any(t in s for t in [
            " count(", " sum(", " avg(", " min(", " max(", " group by "
        ])
        return (not has_where) and (not has_aggregate)

    def _build_entity_context(self, entities) -> str:
        parts = []
        if entities.company_name:
            parts.append(f"CompanyName: {entities.company_name}")
        if entities.brand_name:
            parts.append(f"BrandName: {entities.brand_name}")
        if entities.product_name:
            parts.append(f"ProductName: {entities.product_name}")
        if entities.chemical_name:
            parts.append(f"ChemicalName: {entities.chemical_name}")
        if entities.cas_number:
            parts.append(f"CasNumber: {entities.cas_number}")
        if entities.primary_category:
            parts.append(f"PrimaryCategory: {entities.primary_category}")
        if entities.sub_category:
            parts.append(f"SubCategory: {entities.sub_category}")
        if entities.date_start:
            parts.append(f"Date start: {entities.date_start}")
        if entities.date_end:
            parts.append(f"Date end: {entities.date_end}")
        if entities.discontinued is True:
            parts.append("Filter: discontinued products only (DiscontinuedDate != '')")
        elif entities.discontinued is False:
            parts.append("Filter: active products only (DiscontinuedDate = '')")
        if entities.chemical_removed is True:
            parts.append("Filter: removed chemicals only (ChemicalDateRemoved != '')")
        if entities.raw_filters:
            for k, v in entities.raw_filters.items():
                parts.append(f"{k}: {v}")
        return "\n".join(parts) if parts else "No specific entities extracted."

    def _summarize_entities(self, entities) -> str:
        parts = []
        for field in ["company_name", "brand_name", "product_name", "chemical_name", "cas_number"]:
            val = getattr(entities, field, "")
            if val:
                parts.append(f"{field}={val}")
        return ", ".join(parts) if parts else "none"
