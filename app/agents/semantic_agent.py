"""Agent 5: Semantic Retrieval Agent (Vector).

Handles fuzzy matching using ChromaDB vector search for partial names,
misspellings, synonyms, and ambiguous chemical/product references.

IMPORTANT: This agent returns CANDIDATES, not validated evidence.
The Evidence Merger decides which candidates to promote to evidence.
"""

import chromadb
from app.agents.base import BaseAgent, PipelineState
from app.config import CHROMA_PATH, MAX_VECTOR_RESULTS

# Cosine similarity threshold below which results are discarded as irrelevant.
# ChromaDB returns cosine distance; we convert to similarity = 1 - distance.
# 0.35 keeps misspellings and partial-name matches that might score in the 0.4-0.5 range,
# while still dropping clearly unrelated neighbors. The Evidence Merger applies an
# additional, stricter threshold for promotion to evidence.
MIN_SIMILARITY_THRESHOLD = 0.35


class SemanticAgent(BaseAgent):
    name = "SemanticRetrieval"

    def run(self, state: PipelineState) -> PipelineState:
        query = state.sanitized_query or state.user_query
        entities = state.entities

        state.add_plan_step(
            agent=self.name,
            action="Performing semantic/vector search",
            details=f"Strategy: {state.retrieval_strategy.value}"
        )

        # Build a search query from the user query and extracted entities
        search_text = self._build_search_text(query, entities)
        state.vector_query_text = search_text

        try:
            client = chromadb.PersistentClient(path=CHROMA_PATH)
            collection = client.get_collection("cosmetics_chemicals")

            # Build where filters for metadata if we have specific entities
            where_filter = self._build_where_filter(entities)

            if where_filter:
                results = collection.query(
                    query_texts=[search_text],
                    n_results=MAX_VECTOR_RESULTS,
                    where=where_filter,
                )
            else:
                results = collection.query(
                    query_texts=[search_text],
                    n_results=MAX_VECTOR_RESULTS,
                )

            # Parse results and apply similarity threshold
            vector_results = []
            total_retrieved = 0
            if results and results["metadatas"]:
                for i, metadata in enumerate(results["metadatas"][0]):
                    distance = results["distances"][0][i] if results["distances"] else 0
                    similarity = round(1 - distance, 4)
                    total_retrieved += 1

                    # Apply threshold filter
                    if similarity < MIN_SIMILARITY_THRESHOLD:
                        continue

                    record = {
                        **metadata,
                        "relevance_score": similarity,
                        "source": "vector",
                        "document": results["documents"][0][i] if results["documents"] else "",
                    }
                    vector_results.append(record)

            state.vector_results = vector_results
            filtered_out = total_retrieved - len(vector_results)
            state.query_plan[-1].result_summary = (
                f"Vector search: {total_retrieved} retrieved, {len(vector_results)} above similarity threshold "
                f"{MIN_SIMILARITY_THRESHOLD} ({filtered_out} filtered out as low-confidence). "
                f"Search text: {search_text[:80]}..."
            )

        except Exception as e:
            state.vector_results = []
            state.query_plan[-1].result_summary = f"Vector search error: {str(e)[:100]}"

        return state

    def _build_search_text(self, query: str, entities) -> str:
        """Build an optimized search text combining the query and key entities."""
        parts = [query]
        if entities.chemical_name:
            parts.append(f"Chemical: {entities.chemical_name}")
        if entities.product_name:
            parts.append(f"Product: {entities.product_name}")
        if entities.brand_name:
            parts.append(f"Brand: {entities.brand_name}")
        if entities.company_name:
            parts.append(f"Company: {entities.company_name}")
        if entities.cas_number:
            parts.append(f"CAS: {entities.cas_number}")
        return " | ".join(parts)

    def _build_where_filter(self, entities) -> dict | None:
        """Build ChromaDB metadata filters from extracted entities."""
        conditions = []

        if entities.cas_number:
            conditions.append({"CasNumber": {"$eq": entities.cas_number}})
        if entities.brand_name:
            conditions.append({"BrandName": {"$eq": entities.brand_name}})
        if entities.company_name:
            conditions.append({"CompanyName": {"$eq": entities.company_name}})

        if not conditions:
            return None
        if len(conditions) == 1:
            return conditions[0]
        return {"$and": conditions}
