"""Deterministic hybrid multi-stream retrieval orchestrator."""

from __future__ import annotations

import math
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from time import perf_counter
from typing import Sequence
from uuid import uuid4

from .enums import (
    AuthorityTier,
    HumanFeedback,
    MemoryType,
    Modality,
    OutputStream,
    ProtocolType,
    RetrievalMode,
    SourceTable,
    TaskOutcome,
)
from .models import (
    AugmentedQuery,
    BudgetReport,
    EmbeddedSubQuery,
    EvidenceResponse,
    FragmentLocator,
    FusedCandidate,
    HistoryEntry,
    MemoryEntry,
    MemorySummary,
    MIMOStreamResults,
    ModelCapabilityProfile,
    ProvenanceOrigin,
    ProvenanceRecord,
    ProvenanceScoring,
    ProvenanceTaggedFragment,
    ProvenanceTrust,
    QualityAssessment,
    QualityConfig,
    RankedFragment,
    RankingConfig,
    RetrievalConfig,
    SelectedModelPlan,
    ScoredFragment,
    SessionContext,
    SessionTurn,
    SubQuery,
    TimeWindow,
    UserScope,
)
from .providers import (
    CrossEncoderScorer,
    EmbeddingProvider,
    HashingEmbeddingProvider,
    HeuristicCrossEncoder,
    HeuristicQueryPlanner,
    QueryPlanner,
)
from .repository import RetrievalRepository
from .routing import ModelRouter, default_profiles
from .scoring import (
    authority_multiplier,
    bm25_score,
    build_bm25_corpus,
    classify_query,
    count_tokens,
    cosine_similarity,
    custody_hash,
    feedback_bonus,
    freshness_decay,
    lexical_overlap_score,
    mmr_select,
    normalize_scores,
    outcome_weight,
    quality_assessment,
    tokenize,
    trigram_similarity,
    extract_topic_trajectory,
)
from .types import EmbeddingVector, TokenCounter

SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _coerce_authority(value: object) -> AuthorityTier:
    if isinstance(value, AuthorityTier):
        return value
    if isinstance(value, str):
        try:
            return AuthorityTier(value)
        except ValueError:
            return AuthorityTier.DERIVED
    return AuthorityTier.DERIVED


def _metadata_modality(metadata: dict[str, object]) -> str | None:
    modality = metadata.get("modality")
    if isinstance(modality, str):
        return modality
    modalities = metadata.get("modalities")
    if isinstance(modalities, (list, tuple)) and modalities:
        first = modalities[0]
        if isinstance(first, str):
            return first
    return None


@dataclass(slots=True)
class _PipelineArtifacts:
    augmented_query: AugmentedQuery
    stream_results: MIMOStreamResults
    fused_candidates: list[FusedCandidate]
    reranked: list[RankedFragment]
    diverse_set: list[RankedFragment]


@dataclass(slots=True)
class HybridRetrievalEngine:
    repository: RetrievalRepository
    embedding_provider: EmbeddingProvider = field(default_factory=HashingEmbeddingProvider)
    planner: QueryPlanner = field(default_factory=HeuristicQueryPlanner)
    cross_encoder: CrossEncoderScorer = field(default_factory=HeuristicCrossEncoder)
    config: RetrievalConfig = field(default_factory=RetrievalConfig)
    default_ranking: RankingConfig = field(default_factory=RankingConfig)
    default_quality: QualityConfig = field(default_factory=QualityConfig)
    token_counter: TokenCounter = count_tokens
    model_router: ModelRouter = field(default_factory=ModelRouter)
    model_profiles: tuple[ModelCapabilityProfile, ...] = field(default_factory=default_profiles)
    _embedding_cache: dict[str, EmbeddingVector] = field(default_factory=dict)

    def plan_models(
        self,
        raw_query: str,
        *,
        session_topic: str = "",
        turn_count: int = 0,
    ) -> SelectedModelPlan:
        query_class = classify_query(raw_query, session_topic, turn_count=turn_count)
        return self.model_router.select(query_class, self.model_profiles)

    def retrieve(
        self,
        raw_query: str,
        *,
        uu_id,
        session_id,
        token_budget: int,
        latency_deadline_ms: int | None = None,
        ranking_config: RankingConfig | None = None,
        quality_config: QualityConfig | None = None,
    ) -> EvidenceResponse:
        started_at = perf_counter()
        deadline_ms = latency_deadline_ms or self.config.latency_deadline_ms
        ranking = ranking_config or self.default_ranking
        quality = quality_config or self.default_quality

        scope = self._user_scope_lock(uu_id)
        session_ctx, memory_summary = self._parallel_context_load(scope, session_id)
        primary = self._run_pipeline(
            raw_query=raw_query,
            scope=scope,
            session_ctx=session_ctx,
            memory_summary=memory_summary,
            ranking=ranking,
        )

        elapsed_ms = (perf_counter() - started_at) * 1000.0
        quality_snapshot = quality_assessment(
            primary.diverse_set,
            now=_utc_now(),
            staleness_limit_hours=quality.staleness_limit_hours,
        )
        if (
            quality.max_iterations > 0
            and quality_snapshot.issue is not None
            and elapsed_ms + self.config.min_refinement_budget_ms < deadline_ms
        ):
            expanded_query = self.planner.expand_query(primary.augmented_query.resolved).strip()
            if expanded_query and expanded_query != primary.augmented_query.resolved:
                refined = self._run_pipeline(
                    raw_query=expanded_query,
                    scope=scope,
                    session_ctx=session_ctx,
                    memory_summary=memory_summary,
                    ranking=ranking,
                )
                primary = self._merge_pipeline_results(primary, refined, ranking=ranking)

        provenance_tagged = self._provenance_assembly(primary.diverse_set, scope)
        fitted, budget_report = self._token_budget_fitting(
            provenance_tagged,
            token_budget=token_budget,
            quality=quality,
        )

        final_quality = quality_assessment(
            primary.diverse_set,
            now=_utc_now(),
            staleness_limit_hours=quality.staleness_limit_hours,
        )
        latency_ms = (perf_counter() - started_at) * 1000.0
        coverage = {"session": 0, "history": 0, "memory": 0}
        for fragment in fitted:
            source = fragment.provenance.origin.source_table
            if source is SourceTable.SESSION:
                coverage["session"] += 1
            elif source in {SourceTable.HISTORY, SourceTable.HISTORY_DERIVED}:
                coverage["history"] += 1
            elif source is SourceTable.MEMORY:
                coverage["memory"] += 1

        return EvidenceResponse(
            fragments=tuple(fitted),
            total_candidates_considered=len(primary.fused_candidates),
            latency_ms=latency_ms,
            source_coverage=coverage,
            budget_report=budget_report,
            quality_assessment=final_quality,
            streams_completed=primary.stream_results.streams_completed,
            streams_timed_out=primary.stream_results.streams_timed_out,
            cache_hit_ratio=0.0,
            retrieval_trace_id=uuid4().hex,
        )

    def _user_scope_lock(self, uu_id) -> UserScope:
        if not self.repository.user_exists(uu_id):
            raise LookupError(f"user {uu_id} is not active")
        return UserScope(
            uu_id=uu_id,
            partition_key=hash(uu_id.int) % self.config.partition_modulus,
            acl_scope=self.repository.acl_scope(uu_id),
        )

    def _parallel_context_load(self, scope: UserScope, session_id) -> tuple[SessionContext, MemorySummary]:
        with ThreadPoolExecutor(max_workers=2) as executor:
            session_future = executor.submit(self._load_session_context, scope, session_id)
            memory_future = executor.submit(self._load_memory_summary, scope)
            return session_future.result(), memory_future.result()

    def _load_session_context(self, scope: UserScope, session_id) -> SessionContext:
        turns = tuple(
            self.repository.list_session_turns(
                scope.uu_id,
                session_id,
                limit=self.config.max_session_turns,
            )
        )
        texts = [turn.content_text for turn in turns]
        topic_trajectory = extract_topic_trajectory(texts)
        last_user_query = next((turn.content_text for turn in reversed(turns) if turn.role.value == "user"), None)
        active_tool_context = tuple(
            turn.content_text
            for turn in turns
            if turn.tool_calls or turn.metadata.get("protocol_type") or turn.metadata.get("tool_name")
        )[-4:]
        if turns:
            session_start = turns[0].created_at
            session_duration = _utc_now() - session_start
        else:
            session_start = None
            session_duration = timedelta(0)
        return SessionContext(
            turns=turns,
            turn_count=len(turns),
            last_user_query=last_user_query,
            topic_trajectory=topic_trajectory,
            active_tool_context=active_tool_context,
            session_start=session_start,
            session_duration=session_duration,
        )

    def _load_memory_summary(self, scope: UserScope) -> MemorySummary:
        rows = [
            item
            for item in self.repository.list_memory_entries(
                scope.uu_id,
                limit=self.config.max_memory_items,
            )
            if item.authority_tier is not AuthorityTier.EPHEMERAL
        ]
        authority_rank = {
            AuthorityTier.CANONICAL: 4,
            AuthorityTier.CURATED: 3,
            AuthorityTier.DERIVED: 2,
            AuthorityTier.EPHEMERAL: 1,
        }
        rows.sort(
            key=lambda item: (
                authority_rank[item.authority_tier],
                item.confidence,
                item.access_count,
                item.updated_at,
            ),
            reverse=True,
        )
        compressed_parts: list[str] = []
        running_tokens = 0
        for item in rows:
            rendered = f"[{item.memory_type}] {item.content_text}"
            cost = self.token_counter(rendered)
            if running_tokens + cost > self.config.memory_summary_token_budget:
                break
            compressed_parts.append(rendered)
            running_tokens += cost
        return MemorySummary(
            facts=tuple(item for item in rows if item.memory_type is MemoryType.FACT),
            preferences=tuple(item for item in rows if item.memory_type is MemoryType.PREFERENCE),
            corrections=tuple(item for item in rows if item.memory_type is MemoryType.CORRECTION),
            constraints=tuple(item for item in rows if item.memory_type is MemoryType.CONSTRAINT),
            procedures=tuple(item for item in rows if item.memory_type is MemoryType.PROCEDURE),
            total_items=len(rows),
            compressed_text="\n".join(compressed_parts),
        )

    def _run_pipeline(
        self,
        *,
        raw_query: str,
        scope: UserScope,
        session_ctx: SessionContext,
        memory_summary: MemorySummary,
        ranking: RankingConfig,
    ) -> _PipelineArtifacts:
        query_class = classify_query(raw_query, session_ctx.topic_trajectory, turn_count=session_ctx.turn_count)
        augmented_query = self._augment_query(raw_query, query_class, session_ctx, memory_summary)
        subqueries = self._decompose_query(augmented_query)
        embedded_subqueries = self._embed_subqueries(subqueries, session_ctx, memory_summary)
        stream_results = self._mimo_retrieval_dispatch(
            embedded_subqueries,
            scope=scope,
            session_ctx=session_ctx,
            ranking=ranking,
        )
        fused_candidates = self._adaptive_multi_stream_fusion(
            stream_results=stream_results,
            augmented_query=augmented_query,
            ranking=ranking,
        )
        reranked = self._cross_encoder_reranking(
            fused_candidates,
            query=augmented_query.resolved,
            ranking=ranking,
        )
        diverse_set = self._mmr_diversity_selection(reranked, ranking=ranking)
        return _PipelineArtifacts(
            augmented_query=augmented_query,
            stream_results=stream_results,
            fused_candidates=fused_candidates,
            reranked=reranked,
            diverse_set=diverse_set,
        )

    def _augment_query(
        self,
        raw_query: str,
        query_class,
        session_ctx: SessionContext,
        memory_summary: MemorySummary,
    ) -> AugmentedQuery:
        resolved = raw_query.strip()
        if query_class.requires_coreference_resolution:
            resolved = self.planner.resolve_coreferences(
                resolved,
                [turn.content_text for turn in session_ctx.turns[-5:]],
            ).strip()

        temporal_scope = self._parse_temporal_scope(resolved)
        constraints = tuple(
            item
            for item in memory_summary.constraints
            if lexical_overlap_score(item.content_text, resolved) > 0.15
        )
        preferences = tuple(
            item
            for item in memory_summary.preferences
            if lexical_overlap_score(item.content_text, resolved) > 0.10
        )
        return AugmentedQuery(
            original=raw_query,
            resolved=resolved,
            constraints=constraints,
            preferences=preferences,
            temporal_scope=temporal_scope,
            session_topic=session_ctx.topic_trajectory,
            query_class=query_class,
        )

    def _parse_temporal_scope(self, query_text: str) -> TimeWindow | None:
        now = _utc_now()
        lowered = query_text.lower()
        if "today" in lowered:
            start = now.replace(hour=0, minute=0, second=0, microsecond=0)
            return TimeWindow(start=start, end=now, label="today")
        if "yesterday" in lowered:
            end = now.replace(hour=0, minute=0, second=0, microsecond=0)
            start = end - timedelta(days=1)
            return TimeWindow(start=start, end=end, label="yesterday")
        if "last week" in lowered or "recently" in lowered:
            return TimeWindow(start=now - timedelta(days=7), end=now, label="last_week")
        return None

    def _decompose_query(self, augmented_query: AugmentedQuery) -> list[SubQuery]:
        query_class = augmented_query.query_class
        target_tables = (SourceTable.SESSION, SourceTable.HISTORY, SourceTable.MEMORY)
        retrieval_modes = (RetrievalMode.EXACT, RetrievalMode.SEMANTIC)
        if not query_class.requires_decomposition:
            return [
                SubQuery(
                    text=augmented_query.resolved,
                    target_tables=target_tables,
                    retrieval_modes=retrieval_modes,
                    priority=1.0,
                    temporal_scope=augmented_query.temporal_scope,
                    output_stream=query_class.output_stream,
                    preferred_modalities=query_class.preferred_modalities,
                    protocol_filters=query_class.protocol_hints,
                    is_followup=query_class.requires_coreference_resolution,
                )
            ]

        decomposed = list(
            self.planner.decompose_query(
                augmented_query.resolved,
                max_subqueries=self.config.max_subqueries,
            )
        )[: self.config.max_subqueries]
        subqueries = [
            SubQuery(
                text=text,
                target_tables=target_tables,
                retrieval_modes=retrieval_modes,
                priority=max(1.0 - 0.1 * index, 0.5),
                temporal_scope=augmented_query.temporal_scope,
                output_stream=query_class.output_stream,
                preferred_modalities=query_class.preferred_modalities,
                protocol_filters=query_class.protocol_hints,
                is_followup=query_class.requires_coreference_resolution,
            )
            for index, text in enumerate(decomposed)
            if text.strip()
        ]
        if all(subquery.text != augmented_query.resolved for subquery in subqueries):
            subqueries.append(
                SubQuery(
                    text=augmented_query.resolved,
                    target_tables=target_tables,
                    retrieval_modes=retrieval_modes,
                    priority=0.8,
                    temporal_scope=augmented_query.temporal_scope,
                    output_stream=query_class.output_stream,
                    preferred_modalities=query_class.preferred_modalities,
                    protocol_filters=query_class.protocol_hints,
                    is_original=True,
                    is_followup=query_class.requires_coreference_resolution,
                )
            )
        return subqueries

    def _embed_subqueries(
        self,
        subqueries: Sequence[SubQuery],
        session_ctx: SessionContext,
        memory_summary: MemorySummary,
    ) -> list[EmbeddedSubQuery]:
        context_prefix_parts = []
        if session_ctx.topic_trajectory:
            context_prefix_parts.append(f"Context: {self._truncate_to_tokens(session_ctx.topic_trajectory, 50)}.")
        if memory_summary.compressed_text:
            context_prefix_parts.append(
                f"User knowledge: {self._truncate_to_tokens(memory_summary.compressed_text, 30)}."
            )
        context_prefix = " ".join(context_prefix_parts)
        texts_to_embed: list[str] = []
        for subquery in subqueries:
            texts_to_embed.append(subquery.text)
            texts_to_embed.append(f"{context_prefix} {subquery.text}".strip())
        embeddings = self.embedding_provider.embed_batch(texts_to_embed)

        embedded: list[EmbeddedSubQuery] = []
        for index, subquery in enumerate(subqueries):
            plain = embeddings[2 * index]
            augmented = embeddings[2 * index + 1]
            plain_weight = 0.3 if subquery.is_followup else 0.6
            augmented_weight = 1.0 - plain_weight
            combined = self._normalize_vector(
                [
                    plain_weight * left + augmented_weight * right
                    for left, right in zip(plain, augmented, strict=True)
                ]
            )
            embedded.append(
                EmbeddedSubQuery(
                    subquery=subquery,
                    plain_embedding=plain,
                    augmented_embedding=augmented,
                    combined_embedding=combined,
                )
            )
        return embedded

    def _mimo_retrieval_dispatch(
        self,
        embedded_subqueries: Sequence[EmbeddedSubQuery],
        *,
        scope: UserScope,
        session_ctx: SessionContext,
        ranking: RankingConfig,
    ) -> MIMOStreamResults:
        with ThreadPoolExecutor(max_workers=5) as executor:
            exact_future = executor.submit(self._exact_retrieval_stream, embedded_subqueries, scope, session_ctx, ranking)
            semantic_future = executor.submit(self._semantic_retrieval_stream, embedded_subqueries, scope, session_ctx, ranking)
            history_future = executor.submit(self._historical_utility_stream, embedded_subqueries, scope, ranking)
            memory_future = executor.submit(self._memory_augmented_stream, embedded_subqueries, scope, ranking)
            negative_future = executor.submit(self._negative_evidence_stream, embedded_subqueries, scope, ranking)
            exact = tuple(exact_future.result())
            semantic = tuple(semantic_future.result())
            history = tuple(history_future.result())
            memory = tuple(memory_future.result())
            negative = frozenset(negative_future.result())
        return MIMOStreamResults(
            exact=exact,
            semantic=semantic,
            history=history,
            memory=memory,
            negative=negative,
            streams_completed=5,
            streams_timed_out=0,
        )

    def _exact_retrieval_stream(
        self,
        embedded_subqueries: Sequence[EmbeddedSubQuery],
        scope: UserScope,
        session_ctx: SessionContext,
        ranking: RankingConfig,
    ) -> list[ScoredFragment]:
        history_rows = tuple(
            row for row in self.repository.list_history_entries(scope.uu_id) if not row.negative_flag
        )
        memory_rows = tuple(self.repository.list_memory_entries(scope.uu_id))
        docs_by_table = {
            SourceTable.SESSION: tuple(session_ctx.turns),
            SourceTable.HISTORY: history_rows,
            SourceTable.MEMORY: memory_rows,
        }
        corpora = {
            SourceTable.SESSION: build_bm25_corpus([turn.content_text for turn in session_ctx.turns]),
            SourceTable.HISTORY: build_bm25_corpus([self._history_content(row) for row in history_rows]),
            SourceTable.MEMORY: build_bm25_corpus([row.content_text for row in memory_rows]),
        }

        best_by_locator: dict[str, ScoredFragment] = {}
        for entry in embedded_subqueries:
            if RetrievalMode.EXACT not in entry.subquery.retrieval_modes:
                continue
            for table in entry.subquery.target_tables:
                if table not in docs_by_table:
                    continue
                scored_rows: list[ScoredFragment] = []
                for row in docs_by_table[table]:
                    if not self._within_temporal_scope(row, entry.subquery.temporal_scope):
                        continue
                    content = self._row_content(row, table)
                    bm25 = bm25_score(
                        entry.subquery.text,
                        content,
                        corpora[table],
                        k1=self.config.bm25_k1,
                        b=self.config.bm25_b,
                    )
                    trigram = trigram_similarity(entry.subquery.text, content)
                    if bm25 <= 0.0 and trigram <= 0.0:
                        continue
                    score = (
                        ranking.bm25_exact_weight * bm25
                        + ranking.trigram_exact_weight * trigram
                    ) * self._stream_match_boost(entry.subquery, self._row_metadata(row, table))
                    fragment = ScoredFragment(
                        locator=self._row_locator(row, table),
                        content=content,
                        score=score,
                        retrieval_mode=RetrievalMode.EXACT,
                        created_at=self._row_timestamp(row, table),
                        metadata=self._row_metadata(row, table),
                        subquery_origin=entry.subquery.text,
                    )
                    scored_rows.append(fragment)
                scored_rows.sort(key=lambda item: item.score, reverse=True)
                for fragment in scored_rows[: ranking.k_exact_per_table]:
                    key = fragment.locator.as_cache_key()
                    previous = best_by_locator.get(key)
                    if previous is None or fragment.score > previous.score:
                        best_by_locator[key] = fragment
        return list(best_by_locator.values())

    def _semantic_retrieval_stream(
        self,
        embedded_subqueries: Sequence[EmbeddedSubQuery],
        scope: UserScope,
        session_ctx: SessionContext,
        ranking: RankingConfig,
    ) -> list[ScoredFragment]:
        history_rows = tuple(
            row for row in self.repository.list_history_entries(scope.uu_id) if not row.negative_flag
        )
        memory_rows = tuple(self.repository.list_memory_entries(scope.uu_id))
        rows_by_table = {
            SourceTable.SESSION: tuple(session_ctx.turns),
            SourceTable.HISTORY: history_rows,
            SourceTable.MEMORY: memory_rows,
        }
        best_by_locator: dict[str, ScoredFragment] = {}

        for entry in embedded_subqueries:
            if RetrievalMode.SEMANTIC not in entry.subquery.retrieval_modes:
                continue
            for table in entry.subquery.target_tables:
                scored_rows: list[ScoredFragment] = []
                for row in rows_by_table.get(table, ()):
                    if not self._within_temporal_scope(row, entry.subquery.temporal_scope):
                        continue
                    vector = self._embedding_for_row(row, table)
                    similarity = max(cosine_similarity(entry.combined_embedding, vector), 0.0)
                    if similarity <= 0.0:
                        continue
                    metadata = self._row_metadata(row, table)
                    scored_rows.append(
                        ScoredFragment(
                            locator=self._row_locator(row, table),
                            content=self._row_content(row, table),
                            score=similarity * self._stream_match_boost(entry.subquery, metadata),
                            retrieval_mode=RetrievalMode.SEMANTIC,
                            created_at=self._row_timestamp(row, table),
                            metadata=metadata,
                            subquery_origin=entry.subquery.text,
                        )
                    )
                scored_rows.sort(key=lambda item: item.score, reverse=True)
                for fragment in scored_rows[: ranking.k_semantic_per_table]:
                    key = fragment.locator.as_cache_key()
                    previous = best_by_locator.get(key)
                    if previous is None or fragment.score > previous.score:
                        best_by_locator[key] = fragment
        return list(best_by_locator.values())

    def _historical_utility_stream(
        self,
        embedded_subqueries: Sequence[EmbeddedSubQuery],
        scope: UserScope,
        ranking: RankingConfig,
    ) -> list[ScoredFragment]:
        now = _utc_now()
        aggregated: dict[str, ScoredFragment] = {}
        history_rows = tuple(self.repository.list_history_entries(scope.uu_id))
        for entry in embedded_subqueries:
            for past in history_rows:
                if past.task_outcome not in {TaskOutcome.SUCCESS, TaskOutcome.PARTIAL}:
                    continue
                if past.utility_score <= ranking.utility_threshold:
                    continue
                similarity = max(
                    cosine_similarity(entry.combined_embedding, self._history_embedding(past)),
                    0.0,
                )
                if similarity <= ranking.similarity_threshold:
                    continue
                recency = freshness_decay(
                    past.created_at,
                    now=now,
                    half_life_hours=self.config.history_decay_days * 24.0,
                )
                for evidence in past.evidence_used:
                    if evidence.utility_score <= 0.3:
                        continue
                    resolved = self.repository.resolve_fragment(scope.uu_id, evidence.locator)
                    if resolved is None:
                        continue
                    content, created_at, metadata = resolved
                    hist_score = (
                        similarity
                        * evidence.utility_score
                        * outcome_weight(past.task_outcome)
                        * recency
                        + feedback_bonus(past.human_feedback)
                    )
                    key = evidence.locator.as_cache_key()
                    if key not in aggregated:
                        merged_metadata = dict(metadata)
                        merged_metadata.update(
                            {
                                "original_query": past.query_text,
                                "original_outcome": str(past.task_outcome),
                                "times_useful": evidence.use_count,
                                "created_at": created_at,
                            }
                        )
                        aggregated[key] = ScoredFragment(
                            locator=evidence.locator,
                            content=content,
                            score=hist_score,
                            retrieval_mode=RetrievalMode.HISTORICAL_UTILITY,
                            created_at=created_at,
                            metadata=merged_metadata,
                            subquery_origin=entry.subquery.text,
                        )
                    else:
                        aggregated[key].score += hist_score
                        aggregated[key].metadata["times_useful"] = int(
                            aggregated[key].metadata.get("times_useful", 1)
                        ) + evidence.use_count
        return list(aggregated.values())

    def _memory_augmented_stream(
        self,
        embedded_subqueries: Sequence[EmbeddedSubQuery],
        scope: UserScope,
        ranking: RankingConfig,
    ) -> list[ScoredFragment]:
        now = _utc_now()
        rows = tuple(self.repository.list_memory_entries(scope.uu_id))
        max_access_count = max((row.access_count for row in rows), default=1)
        best_by_locator: dict[str, ScoredFragment] = {}

        for entry in embedded_subqueries:
            for row in rows:
                if not self._within_temporal_scope(row, entry.subquery.temporal_scope):
                    continue
                semantic = max(cosine_similarity(entry.combined_embedding, self._memory_embedding(row)), 0.0)
                lexical = lexical_overlap_score(entry.subquery.text, row.content_text)
                base_score = max(semantic, lexical)
                if base_score <= 0.0:
                    continue
                confidence_factor = 0.5 + 0.5 * row.confidence
                freshness = freshness_decay(
                    row.updated_at,
                    now=now,
                    half_life_hours=self.config.memory_decay_days * 24.0,
                )
                usage_boost = math.log1p(row.access_count) / math.log1p(max_access_count) if max_access_count > 0 else 0.0
                type_boost = {
                    MemoryType.CORRECTION: 1.3,
                    MemoryType.CONSTRAINT: 1.2,
                    MemoryType.FACT: 1.0,
                    MemoryType.PREFERENCE: 0.9,
                    MemoryType.PROCEDURE: 1.1,
                    MemoryType.SCHEMA: 1.0,
                }[row.memory_type]
                metadata = self._row_metadata(row, SourceTable.MEMORY)
                final_score = (
                    base_score
                    * authority_multiplier(row.authority_tier)
                    * confidence_factor
                    * freshness
                    * (1.0 + 0.1 * usage_boost)
                    * type_boost
                    * self._stream_match_boost(entry.subquery, metadata)
                )
                locator = self._row_locator(row, SourceTable.MEMORY)
                key = locator.as_cache_key()
                fragment = ScoredFragment(
                    locator=locator,
                    content=row.content_text,
                    score=final_score,
                    retrieval_mode=RetrievalMode.MEMORY_AUGMENTED,
                    created_at=row.updated_at,
                    metadata=metadata,
                    subquery_origin=entry.subquery.text,
                )
                previous = best_by_locator.get(key)
                if previous is None or fragment.score > previous.score:
                    best_by_locator[key] = fragment

        selected_memory_ids = [fragment.locator.chunk_id for fragment in best_by_locator.values()]
        self.repository.touch_memories(scope.uu_id, selected_memory_ids, accessed_at=now)
        return list(best_by_locator.values())

    def _negative_evidence_stream(
        self,
        embedded_subqueries: Sequence[EmbeddedSubQuery],
        scope: UserScope,
        ranking: RankingConfig,
    ) -> set[FragmentLocator]:
        negative: set[FragmentLocator] = set()
        history_rows = tuple(self.repository.list_history_entries(scope.uu_id))
        for entry in embedded_subqueries:
            for past in history_rows:
                if not (past.negative_flag or past.human_feedback is HumanFeedback.REJECTED):
                    continue
                similarity = max(
                    cosine_similarity(entry.combined_embedding, self._history_embedding(past)),
                    0.0,
                )
                if similarity <= ranking.negative_similarity_threshold:
                    continue
                for evidence in past.evidence_used:
                    if evidence.utility_score < ranking.negative_utility_threshold:
                        negative.add(evidence.locator)
        return negative

    def _adaptive_multi_stream_fusion(
        self,
        *,
        stream_results: MIMOStreamResults,
        augmented_query: AugmentedQuery,
        ranking: RankingConfig,
    ) -> list[FusedCandidate]:
        bias = augmented_query.query_class.retrieval_bias.normalized()
        ranked_lists = {
            "exact": sorted(stream_results.exact, key=lambda item: item.score, reverse=True),
            "semantic": sorted(stream_results.semantic, key=lambda item: item.score, reverse=True),
            "history": sorted(stream_results.history, key=lambda item: item.score, reverse=True),
            "memory": sorted(stream_results.memory, key=lambda item: item.score, reverse=True),
        }
        weights = {
            "exact": bias.bm25_weight,
            "semantic": bias.dense_weight,
            "history": bias.history_weight,
            "memory": bias.memory_weight,
        }

        accumulators: dict[str, dict[str, object]] = {}
        for stream_name, ranked_list in ranked_lists.items():
            for rank, fragment in enumerate(ranked_list, start=1):
                key = fragment.locator.as_cache_key()
                accumulator = accumulators.setdefault(
                    key,
                    {
                        "locator": fragment.locator,
                        "content": fragment.content,
                        "rrf_score": 0.0,
                        "source_streams": set(),
                        "per_stream_ranks": {},
                        "per_stream_scores": {},
                        "metadata": dict(fragment.metadata),
                    },
                )
                accumulator["rrf_score"] = float(accumulator["rrf_score"]) + (
                    weights[stream_name] / (ranking.rrf_kappa + rank)
                )
                accumulator["source_streams"].add(stream_name)
                accumulator["per_stream_ranks"][stream_name] = rank
                accumulator["per_stream_scores"][stream_name] = fragment.score
                accumulator["metadata"].update(fragment.metadata)

        fused: list[FusedCandidate] = []
        for key, accumulator in accumulators.items():
            stream_count = len(accumulator["source_streams"])
            agreement_bonus = (stream_count / max(len(ranked_lists), 1)) ** 0.5
            rrf_score = float(accumulator["rrf_score"]) * (1.0 + ranking.agreement_multiplier * agreement_bonus)
            if stream_count == 1 and "history" in accumulator["source_streams"]:
                rrf_score *= 0.8
            if accumulator["locator"] in stream_results.negative:
                rrf_score *= ranking.negative_penalty
                accumulator["metadata"]["negative_flag"] = True

            authority = accumulator["metadata"].get("authority_tier")
            if authority is not None:
                rrf_score *= {
                    AuthorityTier.CANONICAL.value: 1.3,
                    AuthorityTier.CURATED.value: 1.15,
                    AuthorityTier.DERIVED.value: 1.0,
                    AuthorityTier.EPHEMERAL.value: 0.9,
                }.get(str(authority), 1.0)
            if (
                accumulator["metadata"].get("source_table") == SourceTable.SESSION.value
                and augmented_query.query_class.type.value == "conversational"
            ):
                rrf_score *= 1.2

            fused.append(
                FusedCandidate(
                    locator=accumulator["locator"],
                    content=str(accumulator["content"]),
                    fused_score=rrf_score,
                    source_streams=frozenset(accumulator["source_streams"]),
                    per_stream_ranks=dict(accumulator["per_stream_ranks"]),
                    per_stream_scores=dict(accumulator["per_stream_scores"]),
                    metadata=dict(accumulator["metadata"]),
                    stream_agreement=stream_count,
                )
            )
        fused.sort(key=lambda item: item.fused_score, reverse=True)
        return fused

    def _cross_encoder_reranking(
        self,
        fused_candidates: Sequence[FusedCandidate],
        *,
        query: str,
        ranking: RankingConfig,
    ) -> list[RankedFragment]:
        candidates = list(fused_candidates[: ranking.k_rerank])
        if not candidates:
            return []
        cross_scores = self.cross_encoder.score_pairs(query, [candidate.content for candidate in candidates])
        if len(cross_scores) != len(candidates):
            cross_scores = [candidate.fused_score for candidate in candidates]

        reranked: list[RankedFragment] = []
        for candidate, cross_score in zip(candidates, cross_scores, strict=True):
            combined = ranking.beta_cross * cross_score + ranking.beta_fused * candidate.fused_score
            combined *= 1.0 + 0.05 * candidate.stream_agreement
            reranked.append(
                RankedFragment(
                    locator=candidate.locator,
                    content=candidate.content,
                    final_score=combined,
                    cross_encoder_score=cross_score,
                    fused_score=candidate.fused_score,
                    source_streams=candidate.source_streams,
                    metadata=dict(candidate.metadata),
                )
            )
        reranked.sort(key=lambda item: item.final_score, reverse=True)
        return reranked

    def _mmr_diversity_selection(
        self,
        reranked: Sequence[RankedFragment],
        *,
        ranking: RankingConfig,
    ) -> list[RankedFragment]:
        if not reranked:
            return []
        missing = [item for item in reranked if item.locator.as_cache_key() not in self._embedding_cache]
        if missing:
            embeddings = self.embedding_provider.embed_batch([item.content for item in missing])
            for item, embedding in zip(missing, embeddings, strict=True):
                self._embedding_cache[item.locator.as_cache_key()] = embedding
        return mmr_select(
            reranked,
            k_final=ranking.k_final,
            diversity_lambda=ranking.diversity_lambda,
            source_diversity_bonus=ranking.source_diversity_bonus,
            embedding_lookup=self._embedding_cache,
        )

    def _provenance_assembly(
        self,
        diverse_set: Sequence[RankedFragment],
        scope: UserScope,
    ) -> list[ProvenanceTaggedFragment]:
        now = _utc_now()
        tagged: list[ProvenanceTaggedFragment] = []
        for fragment in diverse_set:
            created_at = fragment.metadata.get("created_at")
            timestamp = created_at if isinstance(created_at, datetime) else now
            source_table = fragment.metadata.get("source_table", fragment.locator.source_table.value)
            authority = _coerce_authority(fragment.metadata.get("authority_tier"))
            trust = ProvenanceTrust(
                authority_tier=authority,
                confidence=float(fragment.metadata.get("confidence", 0.5)),
                is_memory_validated=str(source_table) == SourceTable.MEMORY.value,
                is_historically_proven="history" in fragment.source_streams,
                negative_flag=bool(fragment.metadata.get("negative_flag", False)),
                freshness=now - timestamp,
            )
            scoring = ProvenanceScoring(
                cross_encoder_score=fragment.cross_encoder_score,
                fused_score=fragment.fused_score,
                final_score=fragment.final_score,
                source_streams=tuple(sorted(fragment.source_streams)),
                stream_agreement=len(fragment.source_streams),
            )
            origin = ProvenanceOrigin(
                source_table=SourceTable(str(source_table)),
                source_id=f"{scope.uu_id}:{source_table}",
                chunk_id=fragment.locator.chunk_id,
                extraction_timestamp=timestamp,
                retrieval_method="+".join(sorted(fragment.source_streams)),
            )
            provenance = ProvenanceRecord(
                origin=origin,
                scoring=scoring,
                trust=trust,
                custody_hash=custody_hash(scope.uu_id, fragment.locator.as_cache_key(), fragment.final_score),
            )
            tagged.append(
                ProvenanceTaggedFragment(
                    locator=fragment.locator,
                    content=fragment.content,
                    final_score=fragment.final_score,
                    cross_encoder_score=fragment.cross_encoder_score,
                    fused_score=fragment.fused_score,
                    provenance=provenance,
                    token_count=self.token_counter(fragment.content),
                )
            )
        return tagged

    def _token_budget_fitting(
        self,
        fragments: Sequence[ProvenanceTaggedFragment],
        *,
        token_budget: int,
        quality: QualityConfig,
    ) -> tuple[list[ProvenanceTaggedFragment], BudgetReport]:
        selected: list[ProvenanceTaggedFragment] = []
        tokens_used = 0
        truncation_applied = False
        for fragment in fragments:
            fragment_cost = fragment.token_count + quality.provenance_overhead_per_fragment
            if tokens_used + fragment_cost <= token_budget:
                selected.append(fragment)
                tokens_used += fragment_cost
                continue

            remaining = token_budget - tokens_used
            if remaining <= quality.provenance_overhead_per_fragment + quality.min_useful_tokens:
                break
            truncated_content = self._truncate_at_sentence_boundary(
                fragment.content,
                max_tokens=remaining - quality.provenance_overhead_per_fragment,
            )
            truncated_tokens = self.token_counter(truncated_content)
            if truncated_tokens <= quality.min_useful_tokens:
                break
            selected.append(
                replace(
                    fragment,
                    content=truncated_content,
                    token_count=truncated_tokens,
                    provenance=replace(
                        fragment.provenance,
                        scoring=replace(fragment.provenance.scoring, truncated=True),
                    ),
                )
            )
            tokens_used += truncated_tokens + quality.provenance_overhead_per_fragment
            truncation_applied = True
            break

        report = BudgetReport(
            budget_total=token_budget,
            budget_used=tokens_used,
            budget_remaining=max(token_budget - tokens_used, 0),
            utilization=(tokens_used / token_budget) if token_budget > 0 else 0.0,
            fragments_included=len(selected),
            fragments_excluded=max(len(fragments) - len(selected), 0),
            truncation_applied=truncation_applied,
        )
        return selected, report

    def _merge_pipeline_results(
        self,
        left: _PipelineArtifacts,
        right: _PipelineArtifacts,
        *,
        ranking: RankingConfig,
    ) -> _PipelineArtifacts:
        merged_map: dict[str, RankedFragment] = {}
        for fragment in [*left.reranked, *right.reranked]:
            key = fragment.locator.as_cache_key()
            previous = merged_map.get(key)
            if previous is None or fragment.final_score > previous.final_score:
                merged_map[key] = fragment
        merged_reranked = sorted(merged_map.values(), key=lambda item: item.final_score, reverse=True)
        merged_diverse = self._mmr_diversity_selection(merged_reranked, ranking=ranking)
        return _PipelineArtifacts(
            augmented_query=left.augmented_query,
            stream_results=left.stream_results,
            fused_candidates=left.fused_candidates,
            reranked=merged_reranked,
            diverse_set=merged_diverse,
        )

    def _stream_match_boost(self, subquery: SubQuery, metadata: dict[str, object]) -> float:
        boost = 1.0
        modality = _metadata_modality(metadata)
        if modality is not None and modality in {item.value for item in subquery.preferred_modalities}:
            boost *= 1.12
        protocol_type = metadata.get("protocol_type")
        if isinstance(protocol_type, str) and protocol_type in {item.value for item in subquery.protocol_filters}:
            boost *= 1.10
        if subquery.output_stream is OutputStream.VISION_EVIDENCE and modality in {
            Modality.IMAGE.value,
            Modality.DOCUMENT.value,
        }:
            boost *= 1.08
        if subquery.output_stream is OutputStream.TOOL_TRACE and (
            protocol_type or metadata.get("tool_name") or metadata.get("tool_call_count")
        ):
            boost *= 1.08
        return boost

    def _row_locator(self, row: object, table: SourceTable) -> FragmentLocator:
        if table is SourceTable.SESSION:
            turn = row
            return FragmentLocator(source_table=table, chunk_id=turn.turn_id())
        if table is SourceTable.HISTORY:
            return FragmentLocator(source_table=table, chunk_id=row.history_id)
        if table is SourceTable.MEMORY:
            return FragmentLocator(source_table=table, chunk_id=row.memory_id)
        raise ValueError(f"unsupported table {table}")

    def _row_content(self, row: object, table: SourceTable) -> str:
        if table is SourceTable.SESSION:
            return row.content_text
        if table is SourceTable.HISTORY:
            return self._history_content(row)
        if table is SourceTable.MEMORY:
            return row.content_text
        raise ValueError(f"unsupported table {table}")

    def _history_content(self, row: HistoryEntry) -> str:
        if row.response_summary:
            return f"{row.query_text}\n{row.response_summary}"
        return row.query_text

    def _row_timestamp(self, row: object, table: SourceTable) -> datetime:
        if table is SourceTable.MEMORY:
            return row.updated_at
        return row.created_at

    def _row_metadata(self, row: object, table: SourceTable) -> dict[str, object]:
        if table is SourceTable.SESSION:
            metadata = dict(row.metadata)
            metadata.update(
                {
                    "source_table": table.value,
                    "role": row.role.value,
                    "turn_index": row.turn_index,
                    "created_at": row.created_at,
                    "tool_call_count": len(row.tool_calls or ()),
                }
            )
            if "modality" not in metadata:
                metadata["modality"] = Modality.RUNTIME.value if row.tool_calls else Modality.TEXT.value
            return metadata
        if table is SourceTable.HISTORY:
            metadata = dict(row.metadata)
            metadata.update(
                {
                    "source_table": table.value,
                    "task_outcome": row.task_outcome.value,
                    "utility_score": row.utility_score,
                    "created_at": row.created_at,
                }
            )
            if "modality" not in metadata:
                metadata["modality"] = Modality.TEXT.value
            return metadata
        if table is SourceTable.MEMORY:
            metadata = dict(row.metadata)
            metadata.update(
                {
                    "source_table": table.value,
                    "authority_tier": row.authority_tier.value,
                    "confidence": row.confidence,
                    "memory_type": row.memory_type.value,
                    "access_count": row.access_count,
                    "created_at": row.updated_at,
                }
            )
            if "modality" not in metadata:
                metadata["modality"] = Modality.TEXT.value
            return metadata
        raise ValueError(f"unsupported table {table}")

    def _within_temporal_scope(self, row: object, temporal_scope: TimeWindow | None) -> bool:
        if temporal_scope is None:
            return True
        if hasattr(row, "updated_at"):
            timestamp = row.updated_at
        else:
            timestamp = row.created_at
        return temporal_scope.start <= timestamp <= temporal_scope.end

    def _embedding_for_row(self, row: object, table: SourceTable) -> EmbeddingVector:
        if table is SourceTable.SESSION:
            cache_key = f"session:{row.turn_id()}"
            return self._cached_embedding(cache_key, row.content_text)
        if table is SourceTable.HISTORY:
            return self._history_embedding(row)
        if table is SourceTable.MEMORY:
            return self._memory_embedding(row)
        raise ValueError(f"unsupported table {table}")

    def _history_embedding(self, row: HistoryEntry) -> EmbeddingVector:
        if row.query_embedding is not None:
            return row.query_embedding
        return self._cached_embedding(f"history:{row.history_id}", row.query_text)

    def _memory_embedding(self, row: MemoryEntry) -> EmbeddingVector:
        if row.content_embedding is not None:
            return row.content_embedding
        return self._cached_embedding(f"memory:{row.memory_id}", row.content_text)

    def _cached_embedding(self, cache_key: str, text: str) -> EmbeddingVector:
        if cache_key not in self._embedding_cache:
            self._embedding_cache[cache_key] = self.embedding_provider.embed_batch([text])[0]
        return self._embedding_cache[cache_key]

    def _truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        tokens = tokenize(text)
        if len(tokens) <= max_tokens:
            return text
        return " ".join(tokens[:max_tokens])

    def _truncate_at_sentence_boundary(self, text: str, *, max_tokens: int) -> str:
        sentences = SENTENCE_BOUNDARY.split(text)
        selected: list[str] = []
        running = 0
        for sentence in sentences:
            cost = self.token_counter(sentence)
            if running + cost > max_tokens:
                break
            selected.append(sentence)
            running += cost
        if selected:
            return " ".join(selected).strip()
        return self._truncate_to_tokens(text, max_tokens)

    def _normalize_vector(self, values: Sequence[float]) -> EmbeddingVector:
        norm = math.sqrt(sum(value * value for value in values))
        if norm == 0.0:
            return tuple(0.0 for _ in values)
        return tuple(value / norm for value in values)
