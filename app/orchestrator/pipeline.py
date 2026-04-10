"""LangGraph orchestration pipeline wiring all 8 agents together."""

from langgraph.graph import StateGraph, END
from typing import TypedDict, Any

from app.agents.base import PipelineState, RetrievalStrategy
from app.agents.guardrail import GuardrailAgent
from app.agents.planner import PlannerAgent
from app.agents.entity_resolver import EntityResolverAgent
from app.agents.sql_agent import SQLAgent
from app.agents.semantic_agent import SemanticAgent
from app.agents.evidence_merger import EvidenceMergerAgent
from app.agents.synthesizer import SynthesizerAgent
from app.agents.validator import ValidatorAgent
from app.config import VALIDATION_MAX_RETRIES


# Instantiate agents
guardrail = GuardrailAgent()
planner = PlannerAgent()
entity_resolver = EntityResolverAgent()
sql_agent = SQLAgent()
semantic_agent = SemanticAgent()
evidence_merger = EvidenceMergerAgent()
synthesizer = SynthesizerAgent()
validator = ValidatorAgent()


# --- LangGraph state schema ---
class GraphState(TypedDict):
    pipeline: PipelineState


# --- Node functions ---
def guardrail_node(state: GraphState) -> GraphState:
    state["pipeline"] = guardrail.run(state["pipeline"])
    return state


def planner_node(state: GraphState) -> GraphState:
    state["pipeline"] = planner.run(state["pipeline"])
    return state


def entity_resolver_node(state: GraphState) -> GraphState:
    state["pipeline"] = entity_resolver.run(state["pipeline"])
    return state


def sql_node(state: GraphState) -> GraphState:
    state["pipeline"] = sql_agent.run(state["pipeline"])
    return state


def semantic_node(state: GraphState) -> GraphState:
    state["pipeline"] = semantic_agent.run(state["pipeline"])
    return state


def evidence_merger_node(state: GraphState) -> GraphState:
    state["pipeline"] = evidence_merger.run(state["pipeline"])
    return state


def synthesizer_node(state: GraphState) -> GraphState:
    state["pipeline"] = synthesizer.run(state["pipeline"])
    return state


def validator_node(state: GraphState) -> GraphState:
    state["pipeline"] = validator.run(state["pipeline"])
    return state


# --- Conditional routing functions ---
def route_after_guardrail(state: GraphState) -> str:
    """Route based on guardrail validation."""
    if state["pipeline"].is_valid:
        return "planner"
    return END


def route_after_planner(state: GraphState) -> str:
    """Always go to entity resolver next."""
    return "entity_resolver"


def route_after_entities(state: GraphState) -> str:
    """Route to retrieval agents based on strategy."""
    strategy = state["pipeline"].retrieval_strategy
    if strategy == RetrievalStrategy.SQL_ONLY:
        return "sql_agent"
    elif strategy == RetrievalStrategy.VECTOR_ONLY:
        return "semantic_agent"
    else:  # hybrid
        return "sql_agent"  # SQL first, then vector


def route_after_sql(state: GraphState) -> str:
    """After SQL, decide whether to also run vector search."""
    strategy = state["pipeline"].retrieval_strategy
    if strategy == RetrievalStrategy.HYBRID:
        return "semantic_agent"
    return "evidence_merger"


def route_after_semantic(state: GraphState) -> str:
    """After semantic search, always go to evidence merger."""
    return "evidence_merger"


def route_after_validator(state: GraphState) -> str:
    """After validation, either finish or retry synthesis."""
    ps = state["pipeline"]
    if ps.is_grounded or ps.retry_count >= VALIDATION_MAX_RETRIES:
        return END
    # Retry: go back to synthesizer
    return "synthesizer"


# --- Build the graph ---
def build_pipeline() -> StateGraph:
    """Construct and compile the LangGraph pipeline."""
    workflow = StateGraph(GraphState)

    # Add nodes
    workflow.add_node("guardrail", guardrail_node)
    workflow.add_node("planner", planner_node)
    workflow.add_node("entity_resolver", entity_resolver_node)
    workflow.add_node("sql_agent", sql_node)
    workflow.add_node("semantic_agent", semantic_node)
    workflow.add_node("evidence_merger", evidence_merger_node)
    workflow.add_node("synthesizer", synthesizer_node)
    workflow.add_node("validator", validator_node)

    # Set entry point
    workflow.set_entry_point("guardrail")

    # Add edges with conditional routing
    workflow.add_conditional_edges("guardrail", route_after_guardrail, {
        "planner": "planner",
        END: END,
    })
    workflow.add_edge("planner", "entity_resolver")

    workflow.add_conditional_edges("entity_resolver", route_after_entities, {
        "sql_agent": "sql_agent",
        "semantic_agent": "semantic_agent",
    })

    workflow.add_conditional_edges("sql_agent", route_after_sql, {
        "semantic_agent": "semantic_agent",
        "evidence_merger": "evidence_merger",
    })

    workflow.add_edge("semantic_agent", "evidence_merger")
    workflow.add_edge("evidence_merger", "synthesizer")
    workflow.add_edge("synthesizer", "validator")

    workflow.add_conditional_edges("validator", route_after_validator, {
        "synthesizer": "synthesizer",
        END: END,
    })

    return workflow.compile()


# Compiled pipeline singleton
_pipeline = None


def get_pipeline():
    global _pipeline
    if _pipeline is None:
        _pipeline = build_pipeline()
    return _pipeline


def run_query(question: str) -> dict:
    """Run a question through the full multi-agent pipeline.

    Returns a dict with: answer, evidence, query_plan, confidence, warnings
    """
    pipeline = get_pipeline()

    initial_state: GraphState = {
        "pipeline": PipelineState(user_query=question)
    }

    final_state = pipeline.invoke(initial_state)
    ps = final_state["pipeline"]

    # Format query plan for output
    query_plan_formatted = []
    for step in ps.query_plan:
        query_plan_formatted.append({
            "agent": step.agent,
            "action": step.action,
            "details": step.details,
            "result": step.result_summary,
        })

    return {
        "question": question,
        "answer": ps.final_answer or ps.rejection_reason or "Unable to generate an answer.",
        "evidence": ps.evidence_list,
        "query_plan": query_plan_formatted,
        "confidence": ps.confidence,
        "warnings": ps.warnings,
        "sql_query": ps.sql_query,
        "is_valid": ps.is_valid,
        "is_grounded": ps.is_grounded,
        "intent": ps.intent.value,
        "retrieval_strategy": ps.retrieval_strategy.value,
        "evidence_strength": ps.evidence_strength,
        # Distinguish validated evidence from retrieved candidates
        "validated_evidence_count": ps.validated_evidence_count,
        "candidates_retrieved": ps.candidates_retrieved,
        "is_exact_lookup": ps.is_exact_lookup,
        # Guardrail flags
        "sql_truncated": ps.sql_truncated,
        "sql_is_broad": ps.sql_is_broad,
        "suggested_followups": ps.suggested_followups,
        # Backward compat
        "total_evidence_records": ps.validated_evidence_count,
    }
