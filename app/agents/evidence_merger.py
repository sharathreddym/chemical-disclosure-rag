"""Agent 6: Evidence Merger + Conflict Resolver.

CRITICAL DESIGN PRINCIPLE:
- For EXACT entity lookups (specific product, CAS, chemical), use SQL results ONLY.
  Vector results are kept as `vector_candidates` for transparency but are NOT
  promoted to evidence. This prevents semantic neighbors from polluting answers.
- For FUZZY/EXPLORATORY queries (no specific entity), merge SQL + filtered vector
  results, with strict similarity thresholds applied during merge.
"""

from app.agents.base import BaseAgent, PipelineState, EvidenceRecord

# Tiered similarity thresholds for promoting vector candidates to evidence.
# - HIGH: high enough that we trust the result even without entity context match
#         (handles misspellings, synonyms, partial names where context check would fail)
# - MEDIUM: trustworthy only if it ALSO matches the user's entity context
# - Below MEDIUM: dropped from evidence (kept as candidates for transparency)
HIGH_SIMILARITY = 0.70
MEDIUM_SIMILARITY = 0.50


class EvidenceMergerAgent(BaseAgent):
    name = "EvidenceMerger"

    def run(self, state: PipelineState) -> PipelineState:
        state.add_plan_step(
            agent=self.name,
            action="Merging and validating evidence",
            details=f"SQL results: {len(state.sql_results)}, Vector results: {len(state.vector_results)}"
        )

        # Detect exact-lookup mode:
        # - User provided a UNIQUE identifier (specific product or full CAS number).
        #   We exclude brand/company because they have many variants.
        # - SQL returned at least one result.
        # In exact-lookup mode we suppress vector results from the evidence to
        # prevent semantic neighbors from polluting answers about a specific entity.
        has_specific_entity = state.entities.has_specific_entity()
        has_unique_id = state.entities.has_unique_identifier()
        sql_has_results = len(state.sql_results) > 0
        state.is_exact_lookup = has_unique_id and sql_has_results

        merged: list[EvidenceRecord] = []
        seen_keys: set[str] = set()
        candidates_count = len(state.vector_results)

        # Step 1: Add SQL results (always trusted, always promoted to evidence)
        for row in state.sql_results:
            record = self._row_to_evidence(row, source="sql", score=1.0)
            key = self._make_key_from_record(record)
            if key not in seen_keys:
                seen_keys.add(key)
                merged.append(record)

        # Step 2: Decide what to do with vector results
        if state.is_exact_lookup:
            # EXACT MODE: Vector results are NOT merged into evidence.
            # They are kept as candidates only, for transparency in the query plan.
            state.vector_candidates = list(state.vector_results)
            state.candidates_retrieved = candidates_count
            mode = "exact-lookup (vector results held as candidates only)"
        else:
            # FUZZY/HYBRID MODE: Tiered promotion of vector candidates.
            # This is the path for misspellings, partial names, synonyms, and
            # exploratory queries where the user did not provide an exact identifier.
            promoted = 0
            sql_failed = (len(state.sql_results) == 0)

            for v in state.vector_results:
                similarity = float(v.get("relevance_score", 0))

                # Decide whether this candidate becomes evidence
                should_promote = False
                if similarity >= HIGH_SIMILARITY:
                    # Tier 1: high similarity - trust it even without entity context match.
                    # This is what handles misspellings: "asetone" -> "Acetone" might
                    # not pass the substring context check but will have high similarity.
                    should_promote = True
                elif similarity >= MEDIUM_SIMILARITY:
                    # Tier 2: medium similarity - require entity context match,
                    # OR trust if SQL completely failed (vector is the only signal).
                    if not has_specific_entity:
                        should_promote = True  # Exploratory query, trust filtered results
                    elif self._matches_entity_context(v, state):
                        should_promote = True
                    elif sql_failed:
                        # SQL came up empty - vector is our only hope.
                        # Be lenient: promote medium-similarity even without context match.
                        should_promote = True
                # Below MEDIUM_SIMILARITY: never promoted

                if not should_promote:
                    continue

                record = self._vector_to_evidence(v)
                key = self._make_key_from_record(record)
                if key not in seen_keys:
                    seen_keys.add(key)
                    merged.append(record)
                    promoted += 1
                else:
                    # Already in SQL results - boost confidence
                    for existing in merged:
                        if self._make_key_from_record(existing) == key:
                            existing.source = "merged"
                            existing.relevance_score = min(1.0, existing.relevance_score + 0.1)
                            break

            # Candidates not promoted are kept for transparency
            promoted_ids = {self._make_key_from_record(self._vector_to_evidence(v))
                            for v in state.vector_results
                            if float(v.get("relevance_score", 0)) >= MEDIUM_SIMILARITY}
            state.vector_candidates = [
                v for v in state.vector_results
                if self._make_key_from_record(self._vector_to_evidence(v)) not in promoted_ids
            ]
            state.candidates_retrieved = candidates_count
            mode = f"hybrid (promoted {promoted}/{candidates_count} vector candidates" + (
                ", SQL fallback active" if sql_failed else ""
            ) + ")"

        # Step 3: Detect conflicts
        conflicts = self._detect_conflicts(merged)

        # Step 4: Assess strength
        evidence_strength = self._assess_strength(state, merged)

        # Step 5: Sort by relevance score
        merged.sort(key=lambda r: r.relevance_score, reverse=True)

        state.merged_evidence = merged
        state.validated_evidence_count = len(merged)
        state.evidence_strength = evidence_strength
        state.conflicts = conflicts

        state.query_plan[-1].result_summary = (
            f"Mode: {mode}. Validated evidence: {len(merged)} rows. "
            f"Candidates retrieved: {candidates_count}. "
            f"Strength: {evidence_strength}. Conflicts: {len(conflicts)}."
        )
        return state

    def _row_to_evidence(self, row: dict, source: str, score: float) -> EvidenceRecord:
        """Convert a SQL row dict into an EvidenceRecord with all fields populated."""
        return EvidenceRecord(
            CDPHId=str(row.get("CDPHId", "") or ""),
            CSFId=str(row.get("CSFId", "") or ""),
            CSF=str(row.get("CSF", "") or ""),
            ChemicalId=str(row.get("ChemicalId", "") or ""),
            CasId=str(row.get("CasId", "") or ""),
            ProductName=str(row.get("ProductName", "") or ""),
            CompanyName=str(row.get("CompanyName", "") or ""),
            CompanyId=str(row.get("CompanyId", "") or ""),
            BrandName=str(row.get("BrandName", "") or ""),
            ChemicalName=str(row.get("ChemicalName", "") or ""),
            CasNumber=str(row.get("CasNumber", "") or ""),
            PrimaryCategory=str(row.get("PrimaryCategory", "") or ""),
            PrimaryCategoryId=str(row.get("PrimaryCategoryId", "") or ""),
            SubCategory=str(row.get("SubCategory", "") or ""),
            SubCategoryId=str(row.get("SubCategoryId", "") or ""),
            InitialDateReported=str(row.get("InitialDateReported", "") or ""),
            MostRecentDateReported=str(row.get("MostRecentDateReported", "") or ""),
            DiscontinuedDate=str(row.get("DiscontinuedDate", "") or ""),
            ChemicalCreatedAt=str(row.get("ChemicalCreatedAt", "") or ""),
            ChemicalUpdatedAt=str(row.get("ChemicalUpdatedAt", "") or ""),
            ChemicalDateRemoved=str(row.get("ChemicalDateRemoved", "") or ""),
            ChemicalCount=str(row.get("ChemicalCount", "") or ""),
            source=source,
            relevance_score=score,
        )

    def _vector_to_evidence(self, v: dict) -> EvidenceRecord:
        """Convert a vector candidate (limited metadata) into an EvidenceRecord."""
        return EvidenceRecord(
            CDPHId=str(v.get("CDPHId", "") or ""),
            ProductName=str(v.get("ProductName", "") or ""),
            CompanyName=str(v.get("CompanyName", "") or ""),
            BrandName=str(v.get("BrandName", "") or ""),
            ChemicalName=str(v.get("ChemicalName", "") or ""),
            CasNumber=str(v.get("CasNumber", "") or ""),
            ChemicalId=str(v.get("ChemicalId", "") or ""),
            PrimaryCategory=str(v.get("PrimaryCategory", "") or ""),
            SubCategory=str(v.get("SubCategory", "") or ""),
            DiscontinuedDate=str(v.get("DiscontinuedDate", "") or ""),
            source="vector",
            relevance_score=float(v.get("relevance_score", 0.5)),
        )

    def _matches_entity_context(self, vector_row: dict, state: PipelineState) -> bool:
        """Check whether a vector candidate plausibly matches the user's entities.

        Uses BIDIRECTIONAL substring match so partial names work both ways:
        - User says "titanium" matching record "Titanium dioxide" (needle in haystack)
        - User says "titanium dioxide variant" matching record "Titanium dioxide" (haystack in needle)
        """
        e = state.entities

        def _fuzzy_match(record_value, user_value) -> bool:
            if not record_value or not user_value:
                return False
            r = str(record_value).strip().lower()
            u = str(user_value).strip().lower()
            if not r or not u:
                return False
            # Bidirectional containment + token overlap
            if r == u:
                return True
            if u in r or r in u:
                return True
            # Token overlap: at least one word in common (handles "AVON" vs "New Avon LLC")
            r_tokens = set(r.replace(",", " ").split())
            u_tokens = set(u.replace(",", " ").split())
            common = r_tokens & u_tokens
            # Filter out trivial common words
            common -= {"the", "a", "an", "and", "or", "of", "in", "for", "llc", "inc", "co"}
            return len(common) > 0

        if e.cas_number and _fuzzy_match(vector_row.get("CasNumber"), e.cas_number):
            return True
        if e.product_name and _fuzzy_match(vector_row.get("ProductName"), e.product_name):
            return True
        if e.chemical_name and _fuzzy_match(vector_row.get("ChemicalName"), e.chemical_name):
            return True
        if e.brand_name and _fuzzy_match(vector_row.get("BrandName"), e.brand_name):
            return True
        if e.company_name and _fuzzy_match(vector_row.get("CompanyName"), e.company_name):
            return True
        return False

    def _make_key_from_record(self, record: EvidenceRecord) -> str:
        """Create a stable dedup key. Includes CSFId so different CSF rows are kept distinct."""
        return f"{record.CDPHId}:{record.CSFId}:{record.ChemicalId or record.ChemicalName}"

    def _detect_conflicts(self, records: list[EvidenceRecord]) -> list[str]:
        """Detect potential conflicts in the evidence."""
        conflicts = []
        products = {}
        for r in records:
            if r.ProductName:
                products.setdefault(r.ProductName, []).append(r)

        for product, recs in products.items():
            discontinued_vals = set(r.DiscontinuedDate for r in recs if r.DiscontinuedDate)
            if len(discontinued_vals) > 1:
                conflicts.append(
                    f"Product '{product}' has conflicting discontinued dates: {discontinued_vals}"
                )
        return conflicts

    def _assess_strength(self, state: PipelineState, merged: list[EvidenceRecord]) -> str:
        """Assess the overall evidence strength."""
        if not merged:
            return "none"

        if state.is_exact_lookup:
            return "strong"

        has_sql = any(r.source in ("sql", "merged") for r in merged)
        has_vector = any(r.source in ("vector", "merged") for r in merged)
        has_merged = any(r.source == "merged" for r in merged)
        high_relevance = any(r.relevance_score > 0.8 for r in merged)

        if has_sql and len(merged) > 0 and not state.sql_error:
            return "strong"
        if has_merged or (has_sql and has_vector):
            return "strong"
        if has_sql or (has_vector and high_relevance):
            return "moderate"
        if has_vector:
            return "weak"
        return "none"
