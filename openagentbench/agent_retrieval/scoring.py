"""Low-level scoring utilities for lexical, semantic, and diversity-aware ranking."""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from statistics import fmean
from typing import Iterable, Sequence

from .enums import (
    AuthorityTier,
    HumanFeedback,
    LoopStrategy,
    ModelExecutionMode,
    ModelRole,
    Modality,
    OutputStream,
    ProtocolType,
    QualityIssue,
    QueryType,
    ReasoningEffort,
    SignalTopology,
    TaskOutcome,
)
from .models import (
    FusedCandidate,
    QualityAssessment,
    QueryClassification,
    RankedFragment,
    RetrievalBias,
)
from .types import EmbeddingVector

TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_\.]+")
CODE_PATTERN = re.compile(r"```|def |class |import |SELECT |FROM |JOIN |[A-Za-z_]+\(")
IDENTIFIER_PATTERN = re.compile(r"[A-Z][a-zA-Z]+\.[a-z]+|[a-z_]+\(|0x[0-9a-fA-F]+")
QUESTION_PREFIXES = ("what", "how", "why", "when", "where", "which", "is", "can", "does")
TEMPORAL_PATTERN = re.compile(r"\b(yesterday|last week|recently|today|ago|since)\b", re.IGNORECASE)
THINKING_HINTS = (
    "think step by step",
    "reason step by step",
    "reason carefully",
    "deliberate",
    "deeply analyze",
    "deep analysis",
    "show your reasoning",
)
REFLECTIVE_HINTS = ("self-correct", "reflect", "re-evaluate", "critique", "repair")
TOOL_LOOP_HINTS = ("tool", "function", "api", "json-rpc", "grpc", "mcp", "rpc")
AGENTIC_LOOP_HINTS = (
    "agentic",
    "loop until",
    "iterate until",
    "retry until",
    "plan and execute",
    "replan",
    "keep trying",
)
DUAL_MODEL_HINTS = ("rerank", "cross-encoder", "embedding model", "dual model")
MULTI_MODEL_HINTS = (
    "ensemble",
    "committee",
    "planner executor critic",
    "planner and executor",
    "multi model",
    "judge and solver",
)
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "i",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "was",
    "what",
    "when",
    "where",
    "which",
    "with",
    "why",
    "you",
}


def tokenize(text: str) -> tuple[str, ...]:
    return tuple(match.group(0).lower() for match in TOKEN_PATTERN.finditer(text))


def count_tokens(text: str) -> int:
    if not text.strip():
        return 0
    return max(1, math.ceil(len(tokenize(text)) * 1.25))


def cosine_similarity(lhs: Sequence[float], rhs: Sequence[float]) -> float:
    if len(lhs) != len(rhs):
        raise ValueError("embedding dimensions must match")
    numerator = 0.0
    lhs_norm = 0.0
    rhs_norm = 0.0
    for left, right in zip(lhs, rhs, strict=True):
        numerator += left * right
        lhs_norm += left * left
        rhs_norm += right * right
    if lhs_norm == 0.0 or rhs_norm == 0.0:
        return 0.0
    return numerator / math.sqrt(lhs_norm * rhs_norm)


def lexical_overlap_score(query_text: str, content_text: str) -> float:
    query_terms = set(tokenize(query_text))
    content_terms = set(tokenize(content_text))
    if not query_terms or not content_terms:
        return 0.0
    return len(query_terms & content_terms) / len(query_terms | content_terms)


def jaccard_similarity(lhs: Iterable[str], rhs: Iterable[str]) -> float:
    left = set(lhs)
    right = set(rhs)
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def trigram_similarity(lhs: str, rhs: str) -> float:
    def grams(text: str) -> set[str]:
        normalized = f"  {text.lower()}  "
        return {normalized[index : index + 3] for index in range(max(len(normalized) - 2, 0))}

    return jaccard_similarity(grams(lhs), grams(rhs))


@dataclass(slots=True, frozen=True)
class BM25Corpus:
    doc_freq: dict[str, int]
    avg_doc_len: float
    document_count: int


def build_bm25_corpus(documents: Sequence[str]) -> BM25Corpus:
    document_count = len(documents)
    if document_count == 0:
        return BM25Corpus({}, 0.0, 0)
    doc_freq: Counter[str] = Counter()
    total_length = 0
    for document in documents:
        tokens = tokenize(document)
        total_length += len(tokens)
        doc_freq.update(set(tokens))
    return BM25Corpus(dict(doc_freq), total_length / document_count, document_count)


def bm25_score(
    query_text: str,
    document_text: str,
    corpus: BM25Corpus,
    *,
    k1: float = 1.5,
    b: float = 0.75,
) -> float:
    query_terms = tokenize(query_text)
    doc_terms = tokenize(document_text)
    if not query_terms or not doc_terms or corpus.document_count <= 0 or corpus.avg_doc_len <= 0.0:
        return 0.0

    term_counts = Counter(doc_terms)
    doc_len = len(doc_terms)
    score = 0.0
    for term in query_terms:
        df = corpus.doc_freq.get(term, 0)
        if df <= 0:
            continue
        idf = math.log(1.0 + (corpus.document_count - df + 0.5) / (df + 0.5))
        tf = term_counts.get(term, 0)
        if tf <= 0:
            continue
        numerator = tf * (k1 + 1.0)
        denominator = tf + k1 * (1.0 - b + b * (doc_len / corpus.avg_doc_len))
        score += idf * (numerator / denominator)
    return score


def normalize_scores(scores: Sequence[float]) -> list[float]:
    if not scores:
        return []
    minimum = min(scores)
    maximum = max(scores)
    if math.isclose(minimum, maximum):
        if math.isclose(maximum, 0.0):
            return [0.0 for _ in scores]
        return [1.0 for _ in scores]
    scale = maximum - minimum
    return [(score - minimum) / scale for score in scores]


def _topic_terms(text: str, *, max_terms: int = 16) -> tuple[str, ...]:
    tokens = [token for token in tokenize(text) if token not in STOPWORDS and len(token) > 2]
    if not tokens:
        return ()
    counts = Counter(tokens)
    return tuple(token for token, _ in counts.most_common(max_terms))


def extract_topic_trajectory(texts: Sequence[str], *, max_terms: int = 24) -> str:
    counts: Counter[str] = Counter()
    for text in texts:
        counts.update(_topic_terms(text, max_terms=max_terms))
    if not counts:
        return ""
    return " ".join(token for token, _ in counts.most_common(max_terms))


def classify_query(query_text: str, session_topic: str, *, turn_count: int) -> QueryClassification:
    lowered = query_text.strip().lower()
    query_tokens = tokenize(query_text)
    topic_overlap = jaccard_similarity(query_tokens, tokenize(session_topic))
    is_followup = turn_count > 0 and (
        lowered.startswith(("it", "this", "that", "the same", "also", "and"))
        or topic_overlap > 0.3
    )

    has_code_markers = bool(CODE_PATTERN.search(query_text))
    has_identifiers = bool(IDENTIFIER_PATTERN.search(query_text))
    has_temporal_ref = bool(TEMPORAL_PATTERN.search(query_text))
    is_question = lowered.endswith("?") or lowered.startswith(QUESTION_PREFIXES)
    requires_decomposition = len(query_tokens) > 30 or any(marker in lowered for marker in (" and ", " then ", ","))
    preferred_modalities = infer_modalities(query_text)
    protocol_hints = infer_protocol_hints(query_text)
    output_streams = infer_output_streams(
        query_text,
        preferred_modalities=preferred_modalities,
        protocol_hints=protocol_hints,
    )
    reasoning_effort = infer_reasoning_effort(query_text)
    loop_strategy = infer_loop_strategy(query_text, protocol_hints=protocol_hints, requires_decomposition=requires_decomposition)
    model_execution_mode = infer_model_execution_mode(query_text, loop_strategy=loop_strategy)
    signal_topology = infer_signal_topology(preferred_modalities, output_streams)
    model_roles = infer_model_roles(
        query_text,
        preferred_modalities=preferred_modalities,
        output_streams=output_streams,
        reasoning_effort=reasoning_effort,
        loop_strategy=loop_strategy,
        model_execution_mode=model_execution_mode,
    )

    if any(modality in preferred_modalities for modality in (Modality.IMAGE, Modality.AUDIO, Modality.VIDEO, Modality.DOCUMENT)):
        query_type = QueryType.MULTIMODAL
    elif loop_strategy in {LoopStrategy.TOOL_LOOP, LoopStrategy.AGENTIC_LOOP, LoopStrategy.CRITIQUE_REPAIR}:
        query_type = QueryType.AGENTIC
    elif has_code_markers or has_identifiers:
        query_type = QueryType.CODE
    elif has_temporal_ref and any(term in lowered for term in ("error", "fail", "bug", "issue", "exception")):
        query_type = QueryType.DIAGNOSTIC
    elif reasoning_effort is not ReasoningEffort.DIRECT:
        query_type = QueryType.REASONING
    elif lowered.startswith("how "):
        query_type = QueryType.PROCEDURAL
    elif lowered.startswith("why "):
        query_type = QueryType.CONCEPTUAL
    elif is_question:
        query_type = QueryType.FACTUAL
    else:
        query_type = QueryType.CONVERSATIONAL

    retrieval_bias_map = {
        QueryType.CODE: RetrievalBias(0.50, 0.25, 0.10, 0.15),
        QueryType.DIAGNOSTIC: RetrievalBias(0.30, 0.30, 0.10, 0.30),
        QueryType.FACTUAL: RetrievalBias(0.35, 0.40, 0.15, 0.10),
        QueryType.CONCEPTUAL: RetrievalBias(0.20, 0.50, 0.15, 0.15),
        QueryType.PROCEDURAL: RetrievalBias(0.25, 0.35, 0.20, 0.20),
        QueryType.CONVERSATIONAL: RetrievalBias(0.30, 0.35, 0.15, 0.20),
        QueryType.MULTIMODAL: RetrievalBias(0.20, 0.30, 0.15, 0.35),
        QueryType.REASONING: RetrievalBias(0.15, 0.45, 0.20, 0.20),
        QueryType.AGENTIC: RetrievalBias(0.20, 0.25, 0.20, 0.35),
    }
    return QueryClassification(
        type=query_type,
        retrieval_bias=retrieval_bias_map[query_type],
        requires_decomposition=requires_decomposition,
        requires_coreference_resolution=is_followup,
        requires_temporal_scoping=has_temporal_ref,
        session_topic_overlap=topic_overlap,
        output_stream=output_streams[0],
        output_streams=output_streams,
        preferred_modalities=preferred_modalities,
        protocol_hints=protocol_hints,
        model_execution_mode=model_execution_mode,
        signal_topology=signal_topology,
        reasoning_effort=reasoning_effort,
        loop_strategy=loop_strategy,
        model_roles=model_roles,
    )


def infer_modalities(query_text: str) -> tuple[Modality, ...]:
    lowered = query_text.lower()
    modalities: list[Modality] = []
    if bool(CODE_PATTERN.search(query_text)) or any(token in lowered for token in ("code", "function", "class", "sql", "python")):
        modalities.append(Modality.CODE)
    if any(token in lowered for token in ("image", "diagram", "screenshot", "vision", "figure", "png", "jpg")):
        modalities.append(Modality.IMAGE)
    if any(token in lowered for token in ("audio", "speech", "voice", "transcript", "wav", "mp3")):
        modalities.append(Modality.AUDIO)
    if any(token in lowered for token in ("video", "frame", "clip", "movie", "mp4")):
        modalities.append(Modality.VIDEO)
    if any(token in lowered for token in ("document", "pdf", "doc", "report", "page", "spreadsheet")):
        modalities.append(Modality.DOCUMENT)
    if any(token in lowered for token in ("json", "schema", "table", "csv", "yaml", "xml")):
        modalities.append(Modality.STRUCTURED)
    if any(token in lowered for token in ("realtime", "live", "stream", "latency", "tool", "rpc", "grpc", "json-rpc", "mcp")):
        modalities.append(Modality.RUNTIME)
    if any(token in lowered for token in ("trace", "span", "log", "event")):
        modalities.append(Modality.TRACE)
    if not modalities:
        modalities.append(Modality.TEXT)
    return tuple(dict.fromkeys(modalities))


def infer_output_streams(
    query_text: str,
    *,
    preferred_modalities: Sequence[Modality],
    protocol_hints: Sequence[ProtocolType],
) -> tuple[OutputStream, ...]:
    lowered = query_text.lower()
    streams: list[OutputStream] = []
    if any(modality in preferred_modalities for modality in (Modality.IMAGE, Modality.VIDEO)):
        streams.append(OutputStream.VISION_EVIDENCE)
    if any(modality in preferred_modalities for modality in (Modality.RUNTIME, Modality.TRACE)) or protocol_hints:
        streams.append(OutputStream.TOOL_TRACE)
    if Modality.STRUCTURED in preferred_modalities:
        streams.append(OutputStream.STRUCTURED_DATA)
    if Modality.CODE in preferred_modalities:
        streams.append(OutputStream.CODE_EVIDENCE)
    if any(token in lowered for token in ("plan", "execute", "verify", "state")):
        streams.append(OutputStream.RUNTIME_STATE)
    if not streams or any(modality in preferred_modalities for modality in (Modality.TEXT, Modality.DOCUMENT, Modality.AUDIO)):
        streams.append(OutputStream.TEXT_EVIDENCE)
    return tuple(dict.fromkeys(streams))


def infer_protocol_hints(query_text: str) -> tuple[ProtocolType, ...]:
    lowered = query_text.lower()
    hints: list[ProtocolType] = []
    if "json-rpc" in lowered or "json rpc" in lowered:
        hints.append(ProtocolType.JSON_RPC)
    if "grpc" in lowered:
        hints.append(ProtocolType.GRPC)
    if "mcp" in lowered:
        hints.append(ProtocolType.MCP)
    if "function" in lowered or "tool call" in lowered:
        hints.append(ProtocolType.FUNCTION_CALL)
    if "http" in lowered or "rest" in lowered:
        hints.append(ProtocolType.HTTP)
    return tuple(dict.fromkeys(hints))


def infer_reasoning_effort(query_text: str) -> ReasoningEffort:
    lowered = query_text.lower()
    if any(hint in lowered for hint in REFLECTIVE_HINTS):
        return ReasoningEffort.SELF_REFLECTIVE
    if any(hint in lowered for hint in THINKING_HINTS):
        return ReasoningEffort.THINKING
    if any(token in lowered for token in ("deliberate", "prove", "compare tradeoffs", "justify")):
        return ReasoningEffort.DELIBERATE
    return ReasoningEffort.DIRECT


def infer_loop_strategy(
    query_text: str,
    *,
    protocol_hints: Sequence[ProtocolType],
    requires_decomposition: bool,
) -> LoopStrategy:
    lowered = query_text.lower()
    if any(hint in lowered for hint in REFLECTIVE_HINTS):
        return LoopStrategy.CRITIQUE_REPAIR
    if any(hint in lowered for hint in AGENTIC_LOOP_HINTS):
        return LoopStrategy.AGENTIC_LOOP
    if protocol_hints or any(hint in lowered for hint in TOOL_LOOP_HINTS):
        return LoopStrategy.TOOL_LOOP
    if requires_decomposition:
        return LoopStrategy.RETRIEVAL_REFINEMENT
    return LoopStrategy.SINGLE_PASS


def infer_model_execution_mode(query_text: str, *, loop_strategy: LoopStrategy) -> ModelExecutionMode:
    lowered = query_text.lower()
    if any(hint in lowered for hint in MULTI_MODEL_HINTS):
        return ModelExecutionMode.MULTI_MODEL
    if any(hint in lowered for hint in DUAL_MODEL_HINTS):
        return ModelExecutionMode.DUAL_MODEL
    if loop_strategy in {LoopStrategy.AGENTIC_LOOP, LoopStrategy.CRITIQUE_REPAIR}:
        return ModelExecutionMode.MULTI_MODEL
    return ModelExecutionMode.SINGLE_MODEL


def infer_signal_topology(
    preferred_modalities: Sequence[Modality],
    output_streams: Sequence[OutputStream],
) -> SignalTopology:
    input_count = len(preferred_modalities)
    output_count = len(output_streams)
    if input_count <= 1 and output_count <= 1:
        return SignalTopology.SISO
    if input_count <= 1:
        return SignalTopology.SIMO
    if output_count <= 1:
        return SignalTopology.MISO
    return SignalTopology.MIMO


def infer_model_roles(
    query_text: str,
    *,
    preferred_modalities: Sequence[Modality],
    output_streams: Sequence[OutputStream],
    reasoning_effort: ReasoningEffort,
    loop_strategy: LoopStrategy,
    model_execution_mode: ModelExecutionMode,
) -> tuple[ModelRole, ...]:
    lowered = query_text.lower()
    roles: list[ModelRole] = [ModelRole.EMBEDDING, ModelRole.GENERATION]
    if model_execution_mode in {ModelExecutionMode.DUAL_MODEL, ModelExecutionMode.MULTI_MODEL} or any(
        hint in lowered for hint in DUAL_MODEL_HINTS
    ):
        roles.append(ModelRole.RERANKING)
    if any(modality in preferred_modalities for modality in (Modality.IMAGE, Modality.AUDIO, Modality.VIDEO, Modality.DOCUMENT)):
        roles.append(ModelRole.MULTIMODAL)
    if reasoning_effort is not ReasoningEffort.DIRECT:
        roles.append(ModelRole.THINKING)
    if loop_strategy in {LoopStrategy.TOOL_LOOP, LoopStrategy.AGENTIC_LOOP, LoopStrategy.CRITIQUE_REPAIR}:
        roles.extend((ModelRole.PLANNER, ModelRole.EXECUTOR))
    if loop_strategy in {LoopStrategy.AGENTIC_LOOP, LoopStrategy.CRITIQUE_REPAIR} or "verify" in lowered:
        roles.append(ModelRole.CRITIC)
    if loop_strategy is LoopStrategy.AGENTIC_LOOP:
        roles.append(ModelRole.AGENTIC_LOOP)
    if OutputStream.TOOL_TRACE in output_streams and ModelRole.EXECUTOR not in roles:
        roles.append(ModelRole.EXECUTOR)
    return tuple(dict.fromkeys(roles))


def feedback_bonus(feedback: HumanFeedback) -> float:
    if feedback is HumanFeedback.APPROVED:
        return 0.2
    if feedback is HumanFeedback.CORRECTED:
        return 0.1
    return 0.0


def outcome_weight(outcome: TaskOutcome) -> float:
    if outcome is TaskOutcome.SUCCESS:
        return 1.0
    if outcome is TaskOutcome.PARTIAL:
        return 0.6
    return 0.0


def freshness_decay(reference_time: datetime, *, now: datetime, half_life_hours: float) -> float:
    if reference_time.tzinfo is None:
        reference_time = reference_time.replace(tzinfo=timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    delta_hours = max((now - reference_time).total_seconds() / 3600.0, 0.0)
    if half_life_hours <= 0.0:
        return 1.0
    return math.exp(-delta_hours / half_life_hours)


def authority_multiplier(authority_tier: AuthorityTier) -> float:
    if authority_tier is AuthorityTier.CANONICAL:
        return 1.5
    if authority_tier is AuthorityTier.CURATED:
        return 1.2
    if authority_tier is AuthorityTier.DERIVED:
        return 1.0
    return 0.7


def quality_assessment(
    fragments: Sequence[RankedFragment],
    *,
    now: datetime,
    staleness_limit_hours: float,
) -> QualityAssessment:
    if not fragments:
        return QualityAssessment(score=0.0, issue=QualityIssue.LOW_RELEVANCE)

    avg_score = fmean(fragment.final_score for fragment in fragments)
    source_diversity = len(
        {str(fragment.metadata.get("source_table", "")) for fragment in fragments if fragment.metadata.get("source_table")}
    ) / 3.0
    freshness_hours = []
    confidences = []
    for fragment in fragments:
        created_at = fragment.metadata.get("created_at")
        if isinstance(created_at, datetime):
            freshness_hours.append(max((now - created_at).total_seconds() / 3600.0, 0.0))
        confidences.append(float(fragment.metadata.get("confidence", 0.5)))

    max_staleness = max(freshness_hours) if freshness_hours else 0.0
    min_confidence = min(confidences) if confidences else 0.5
    quality_score = (
        0.4 * avg_score
        + 0.2 * source_diversity
        + 0.2 * (1.0 - min(max_staleness / max(staleness_limit_hours, 1e-6), 1.0))
        + 0.2 * min_confidence
    )

    issue = None
    if avg_score < 0.3:
        issue = QualityIssue.LOW_RELEVANCE
    elif source_diversity < 0.33:
        issue = QualityIssue.LOW_DIVERSITY
    elif max_staleness > staleness_limit_hours:
        issue = QualityIssue.STALE_EVIDENCE
    return QualityAssessment(score=quality_score, issue=issue)


def custody_hash(*parts: object) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(str(part).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def mmr_select(
    reranked: Sequence[RankedFragment],
    *,
    k_final: int,
    diversity_lambda: float,
    source_diversity_bonus: float,
    embedding_lookup: dict[str, EmbeddingVector],
) -> list[RankedFragment]:
    if not reranked or k_final <= 0:
        return []

    selected: list[RankedFragment] = []
    remaining = list(reranked)
    normalized_rel = normalize_scores([item.final_score for item in reranked])
    rel_map = {item.locator.as_cache_key(): rel for item, rel in zip(reranked, normalized_rel, strict=True)}

    while remaining and len(selected) < k_final:
        best: RankedFragment | None = None
        best_score = float("-inf")
        source_tables_in_selected = {
            str(item.metadata.get("source_table", "")) for item in selected if item.metadata.get("source_table")
        }

        for candidate in remaining:
            rel = rel_map[candidate.locator.as_cache_key()]
            if not selected:
                max_similarity = 0.0
            else:
                current_embedding = embedding_lookup[candidate.locator.as_cache_key()]
                max_similarity = max(
                    cosine_similarity(current_embedding, embedding_lookup[item.locator.as_cache_key()])
                    for item in selected
                )
            mmr_score = diversity_lambda * rel - (1.0 - diversity_lambda) * max_similarity
            candidate_source = str(candidate.metadata.get("source_table", ""))
            if candidate_source and candidate_source not in source_tables_in_selected:
                mmr_score += source_diversity_bonus
            if mmr_score > best_score:
                best_score = mmr_score
                best = candidate

        if best is None:
            break
        selected.append(best)
        remaining = [candidate for candidate in remaining if candidate.locator != best.locator]
    return selected
