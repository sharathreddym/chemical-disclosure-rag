"""Agent 3: Entity & Constraint Extraction Agent.

Extracts and normalizes entities (chemical names, CAS numbers, companies, brands)
and constraints (date ranges, categories, discontinued status) from the user query.
"""

from app.agents.base import BaseAgent, PipelineState, ExtractedEntities
from app.utils.llm import call_llm_json
from app.data.schema import COLUMN_DESCRIPTIONS

SYSTEM_PROMPT = """You are an entity extraction specialist for a cosmetics chemicals database.
Extract all relevant entities and constraints from the user query.

AVAILABLE FIELDS:
- CompanyName: manufacturer/company name (e.g., "New Avon LLC", "L'Oreal")
- BrandName: product brand (e.g., "AVON", "Glover's", "Revlon")
- ProductName: specific product label name
- ChemicalName: hazardous chemical name (e.g., "Titanium dioxide", "Formaldehyde")
- CasNumber: Chemical Abstracts Service number (e.g., "13463-67-7", "75-07-0")
- PrimaryCategory: product category (e.g., "Makeup Products (non-permanent)", "Hair Care Products (non-coloring)")
- SubCategory: product subcategory (e.g., "Lip Color", "Hair Shampoos")
- Date constraints: reporting dates, discontinuation dates
- Status: discontinued, reformulated, removed

Respond with JSON:
{
    "company_name": "extracted company name or empty string",
    "brand_name": "extracted brand name or empty string",
    "product_name": "extracted product name or empty string",
    "chemical_name": "extracted chemical name or empty string",
    "cas_number": "extracted CAS number or empty string",
    "primary_category": "extracted category or empty string",
    "sub_category": "extracted subcategory or empty string",
    "date_start": "start date if mentioned (convert to MM/DD/YYYY) or empty string",
    "date_end": "end date if mentioned (convert to MM/DD/YYYY) or empty string",
    "discontinued": null or true or false,
    "chemical_removed": null or true or false,
    "raw_filters": {"any additional filters as key-value pairs"}
}

RULES:
- Extract entities EXACTLY as the user wrote them. Do NOT correct spelling, expand abbreviations, or add words.
  - "asetone" stays as "asetone" (the system handles fuzzy matching downstream)
  - "titanium" stays as "titanium" (do not expand to "titanium dioxide")
  - "Avon" stays as "Avon" (do not expand to "New Avon LLC")
- If the user says "discontinued in 2024", set date_start="01/01/2024", date_end="12/31/2024", discontinued=true.
- If the user mentions a year range like "2020-2023", convert to MM/DD/YYYY format.
- CAS numbers have a specific format: digits-digits-digit (e.g., 75-07-0). Extract them precisely.
- If no entity of a type is mentioned, leave it as empty string.
- discontinued=null means the user didn't specify; true means they want discontinued products; false means active only.
- If a name COULD be a chemical OR a product (ambiguous), prefer chemical_name only when the user uses words
  like "chemical", "ingredient", "contains", "containing", "with". Otherwise leave both empty and let the
  semantic agent disambiguate.
"""


class EntityResolverAgent(BaseAgent):
    name = "EntityResolver"

    def run(self, state: PipelineState) -> PipelineState:
        query = state.sanitized_query or state.user_query

        state.add_plan_step(
            agent=self.name,
            action="Extracting entities and constraints",
            details=f"Query: {query[:100]}"
        )

        result = call_llm_json(
            prompt=f"Extract entities and constraints from this query:\n\n\"{query}\"",
            system_prompt=SYSTEM_PROMPT,
        )

        state.entities = ExtractedEntities(
            company_name=result.get("company_name", ""),
            brand_name=result.get("brand_name", ""),
            product_name=result.get("product_name", ""),
            chemical_name=result.get("chemical_name", ""),
            cas_number=result.get("cas_number", ""),
            primary_category=result.get("primary_category", ""),
            sub_category=result.get("sub_category", ""),
            date_start=result.get("date_start", ""),
            date_end=result.get("date_end", ""),
            discontinued=result.get("discontinued", None),
            chemical_removed=result.get("chemical_removed", None),
            raw_filters=result.get("raw_filters", {}),
        )

        # Build summary of what was extracted
        extracted = []
        if state.entities.company_name:
            extracted.append(f"Company: {state.entities.company_name}")
        if state.entities.brand_name:
            extracted.append(f"Brand: {state.entities.brand_name}")
        if state.entities.product_name:
            extracted.append(f"Product: {state.entities.product_name}")
        if state.entities.chemical_name:
            extracted.append(f"Chemical: {state.entities.chemical_name}")
        if state.entities.cas_number:
            extracted.append(f"CAS: {state.entities.cas_number}")
        if state.entities.primary_category:
            extracted.append(f"Category: {state.entities.primary_category}")
        if state.entities.sub_category:
            extracted.append(f"SubCategory: {state.entities.sub_category}")
        if state.entities.discontinued is not None:
            extracted.append(f"Discontinued: {state.entities.discontinued}")
        if state.entities.date_start or state.entities.date_end:
            extracted.append(f"Date range: {state.entities.date_start} - {state.entities.date_end}")

        state.query_plan[-1].result_summary = f"Extracted: {', '.join(extracted) if extracted else 'No specific entities found'}"
        return state
