"""Base agent class and shared state definition."""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any
from enum import Enum


class IntentType(str, Enum):
    LOOKUP = "lookup"           # Find specific product/chemical
    LIST = "list"               # List products/chemicals matching criteria
    COMPARE = "compare"         # Compare across brands/companies/categories
    SUMMARIZE = "summarize"     # Summarize information
    TREND = "trend"             # Time-based trends
    DATA_QUALITY = "data_quality"  # Questions about data completeness/quality
    OUT_OF_SCOPE = "out_of_scope"  # Not related to the dataset


class RetrievalStrategy(str, Enum):
    SQL_ONLY = "sql_only"
    VECTOR_ONLY = "vector_only"
    HYBRID = "hybrid"


@dataclass
class ExtractedEntities:
    """Entities and constraints extracted from the user query."""
    company_name: str = ""
    brand_name: str = ""
    product_name: str = ""
    chemical_name: str = ""
    cas_number: str = ""
    primary_category: str = ""
    sub_category: str = ""
    date_start: str = ""
    date_end: str = ""
    discontinued: bool | None = None  # None=not specified, True=discontinued only, False=active only
    chemical_removed: bool | None = None
    raw_filters: dict = field(default_factory=dict)

    def has_specific_entity(self) -> bool:
        """True if the user provided a specific identifier (not just a category filter)."""
        return bool(
            self.product_name
            or self.cas_number
            or self.chemical_name
            or self.brand_name
            or self.company_name
        )

    def has_unique_identifier(self) -> bool:
        """True ONLY for identifiers unique enough to trust SQL exact-match.

        Brand and company names are excluded because they have many variants
        (e.g., 'AVON' vs 'New Avon LLC') and need fuzzy matching to find them all.
        """
        return bool(self.product_name or self.cas_number)


@dataclass
class QueryPlanStep:
    """A single step in the query execution plan."""
    agent: str
    action: str
    details: str = ""
    result_summary: str = ""


@dataclass
class EvidenceRecord:
    """A single piece of evidence from the dataset.

    Includes ALL columns from the source table so the synthesizer and validator
    have complete information without needing to look anything up elsewhere.
    """
    # Identification
    CDPHId: str = ""
    CSFId: str = ""
    CSF: str = ""
    ChemicalId: str = ""
    CasId: str = ""

    # Names
    ProductName: str = ""
    CompanyName: str = ""
    CompanyId: str = ""
    BrandName: str = ""
    ChemicalName: str = ""
    CasNumber: str = ""

    # Categorization
    PrimaryCategory: str = ""
    PrimaryCategoryId: str = ""
    SubCategory: str = ""
    SubCategoryId: str = ""

    # Dates and lifecycle
    InitialDateReported: str = ""
    MostRecentDateReported: str = ""
    DiscontinuedDate: str = ""
    ChemicalCreatedAt: str = ""
    ChemicalUpdatedAt: str = ""
    ChemicalDateRemoved: str = ""
    ChemicalCount: str = ""

    # Provenance
    source: str = ""  # "sql", "vector", or "merged"
    relevance_score: float = 1.0

    def to_dict(self) -> dict:
        """Return all non-empty fields as a dict."""
        return {k: v for k, v in self.__dict__.items() if v != "" and v is not None}

    def to_full_dict(self) -> dict:
        """Return ALL fields including empty ones (for validator)."""
        return dict(self.__dict__)


@dataclass
class PipelineState:
    """Shared state passed through the agent pipeline."""
    # Input
    user_query: str = ""

    # After Guardrail
    is_valid: bool = True
    rejection_reason: str = ""
    sanitized_query: str = ""

    # After Planner
    intent: IntentType = IntentType.LOOKUP
    sub_questions: list[str] = field(default_factory=list)
    retrieval_strategy: RetrievalStrategy = RetrievalStrategy.HYBRID

    # After Entity Resolver
    entities: ExtractedEntities = field(default_factory=ExtractedEntities)

    # After SQL Agent
    sql_query: str = ""
    sql_results: list[dict] = field(default_factory=list)
    sql_error: str = ""
    sql_truncated: bool = False  # True if SQL hit the row LIMIT (more results exist)
    sql_total_seen: int = 0      # Total rows fetched (may be LIMIT+1)
    sql_is_broad: bool = False   # True if query had no WHERE / no aggregation

    # After Semantic Agent
    vector_results: list[dict] = field(default_factory=list)
    vector_query_text: str = ""

    # After Evidence Merger
    merged_evidence: list[EvidenceRecord] = field(default_factory=list)
    vector_candidates: list[dict] = field(default_factory=list)  # Retrieved but NOT used as evidence
    is_exact_lookup: bool = False  # True if SQL found a specific entity match
    evidence_strength: str = "unknown"  # strong, moderate, weak, none
    conflicts: list[str] = field(default_factory=list)
    candidates_retrieved: int = 0  # Total semantic candidates retrieved (transparency)
    validated_evidence_count: int = 0  # Final supporting rows used for the answer

    # After Synthesizer
    final_answer: str = ""
    evidence_list: list[dict] = field(default_factory=list)
    query_plan: list[QueryPlanStep] = field(default_factory=list)
    confidence: str = ""
    warnings: list[str] = field(default_factory=list)
    suggested_followups: list[str] = field(default_factory=list)

    # After Validator
    is_grounded: bool = False
    validation_issues: list[str] = field(default_factory=list)
    retry_count: int = 0

    def add_plan_step(self, agent: str, action: str, details: str = "", result_summary: str = ""):
        self.query_plan.append(QueryPlanStep(agent=agent, action=action, details=details, result_summary=result_summary))


class BaseAgent(ABC):
    """Base class for all agents in the pipeline."""

    name: str = "BaseAgent"

    @abstractmethod
    def run(self, state: PipelineState) -> PipelineState:
        """Process the pipeline state and return updated state."""
        pass
