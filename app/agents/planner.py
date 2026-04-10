"""Agent 2: Planner / Router Agent.

Classifies the intent of the user query, decomposes multi-part questions,
and decides which retrieval strategy and agents to use.
"""

from app.agents.base import BaseAgent, PipelineState, IntentType, RetrievalStrategy
from app.utils.llm import call_llm_json
from app.data.schema import SCHEMA_SUMMARY

SYSTEM_PROMPT = f"""You are a query planner for a cosmetics chemicals database.
Your job is to classify the user's intent and plan the retrieval strategy.

DATABASE SCHEMA:
{SCHEMA_SUMMARY}

Analyze the query and respond with JSON:
{{
    "intent": "one of: lookup, list, compare, summarize, trend, data_quality",
    "sub_questions": ["list of sub-questions if the query is multi-part, otherwise empty list"],
    "retrieval_strategy": "one of: sql_only, vector_only, hybrid",
    "reasoning": "brief explanation of your classification"
}}

INTENT DEFINITIONS:
- lookup: Find a specific product, chemical, or entity (e.g., "Which products contain Titanium dioxide?")
- list: List multiple items matching criteria (e.g., "List all chemicals in Brand X")
- compare: Compare across entities (e.g., "Compare chemicals in Brand A vs Brand B")
- summarize: Summarize or aggregate information (e.g., "How many products have Chemical X?")
- trend: Time-based analysis (e.g., "Show reporting trends over time for Company Y")
- data_quality: Questions about data completeness (e.g., "How many products have missing CAS numbers?")

RETRIEVAL STRATEGY:
- sql_only: Use when the query has exact names, IDs, CAS numbers, dates, or aggregation needs.
  Structured filtering, counting, grouping, date ranges → SQL.
- vector_only: Use when the query has vague/partial names, misspellings, or needs fuzzy matching.
- hybrid: Use when the query might benefit from both exact matching AND fuzzy retrieval.
  Default to hybrid when unsure.

DECOMPOSITION:
- If the query asks multiple things, break it into sub-questions.
- Each sub-question should be answerable independently.
- For simple single-intent queries, return an empty sub_questions list.
"""


class PlannerAgent(BaseAgent):
    name = "PlannerRouter"

    def run(self, state: PipelineState) -> PipelineState:
        query = state.sanitized_query or state.user_query

        state.add_plan_step(
            agent=self.name,
            action="Classifying intent and planning retrieval strategy",
            details=f"Query: {query[:100]}"
        )

        result = call_llm_json(
            prompt=f"Plan the retrieval for this query:\n\n\"{query}\"",
            system_prompt=SYSTEM_PROMPT,
        )

        # Map intent
        intent_str = result.get("intent", "lookup")
        try:
            state.intent = IntentType(intent_str)
        except ValueError:
            state.intent = IntentType.LOOKUP

        # Map retrieval strategy
        strategy_str = result.get("retrieval_strategy", "hybrid")
        try:
            state.retrieval_strategy = RetrievalStrategy(strategy_str)
        except ValueError:
            state.retrieval_strategy = RetrievalStrategy.HYBRID

        state.sub_questions = result.get("sub_questions", [])

        reasoning = result.get("reasoning", "")
        state.query_plan[-1].result_summary = (
            f"Intent: {state.intent.value}, Strategy: {state.retrieval_strategy.value}, "
            f"Sub-questions: {len(state.sub_questions)}. {reasoning}"
        )
        return state
