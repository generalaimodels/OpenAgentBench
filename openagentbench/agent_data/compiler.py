"""Context compiler that emits OpenAI-compatible message arrays."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Mapping, Protocol, Sequence

from .enums import MemoryScope, TaskType
from .models import CompileRequest, CompiledContext, ContextBudget, HistoryRecord, MemoryRecord
from .packing import pack_memories, select_contiguous_history_suffix
from .scoring import RetrievalWeights, score_memories
from .types import EmbeddingVector, JSONValue


class EmbeddingProvider(Protocol):
    def embed(self, text: str) -> EmbeddingVector:
        """Return an embedding for the supplied text."""


TASK_BUDGET_RATIOS: dict[TaskType, tuple[float, float]] = {
    TaskType.CONTINUATION: (0.15, 0.85),
    TaskType.KNOWLEDGE_INTENSIVE: (0.40, 0.60),
    TaskType.NEW_SESSION_WITH_HISTORY: (0.50, 0.50),
    TaskType.TOOL_HEAVY: (0.10, 0.70),
}


@dataclass(slots=True)
class ContextCompiler:
    embedding_provider: EmbeddingProvider | None = None
    retrieval_weights: RetrievalWeights = RetrievalWeights()

    def compile_context(
        self,
        request: CompileRequest,
        *,
        history: Sequence[HistoryRecord],
        memories: Sequence[MemoryRecord],
        now: datetime | None = None,
    ) -> CompiledContext:
        current_time = now or datetime.now(timezone.utc)
        task_type = request.task_type or classify_task_type(
            request.query_text,
            session_metadata=request.session.metadata,
            request_metadata=request.metadata,
            has_existing_history=bool(history),
        )
        budget = allocate_budget(
            context_window_size=request.session.context_window_size,
            system_prompt_tokens=request.session.system_prompt_tokens,
            response_reserve=request.session.max_response_tokens,
            tool_budget=request.tool_token_budget,
            task_type=task_type,
            memory_override=request.memory_budget_override,
            history_override=request.history_budget_override,
        )

        history_selection = select_contiguous_history_suffix(history, budget.history_budget)
        remaining_budget = max(budget.total_budget - history_selection.token_count, 0)
        memory_budget = min(budget.memory_budget, remaining_budget)

        query_embedding = request.query_embedding or self._embed_query(request.query_text)
        scored_memories = score_memories(
            query_text=request.query_text,
            query_embedding=query_embedding,
            memories=memories,
            now=current_time,
            weights=self.retrieval_weights,
        )
        memory_selection = pack_memories(scored_memories, memory_budget)

        messages: list[dict[str, object]] = []
        system_prompt = (
            request.system_prompt_text
            or request.session.system_prompt_text
            or request.session.metadata.get("system_prompt_text")
        )
        if isinstance(system_prompt, str) and system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        if memory_selection.records:
            messages.append(
                {
                    "role": "system",
                    "content": format_memory_block(memory_selection.records),
                }
            )

        if history_selection.truncated and request.session.summary_text:
            messages.append(
                {
                    "role": "system",
                    "content": f"[Session Summary]\n{request.session.summary_text}",
                }
            )

        messages.extend(record.to_chat_message().as_openai_dict() for record in history_selection.records)

        tokens_used = history_selection.token_count + memory_selection.token_count + request.session.system_prompt_tokens
        return CompiledContext(
            messages=messages,
            selected_memories=memory_selection.records,
            selected_history=history_selection.records,
            budget=budget,
            task_type=task_type,
            tokens_used=tokens_used,
        )

    def _embed_query(self, query_text: str) -> EmbeddingVector | None:
        if self.embedding_provider is None or not query_text.strip():
            return None
        return self.embedding_provider.embed(query_text)


def classify_task_type(
    query_text: str,
    *,
    session_metadata: Mapping[str, JSONValue],
    request_metadata: Mapping[str, JSONValue],
    has_existing_history: bool,
) -> TaskType:
    explicit = request_metadata.get("task_type") or session_metadata.get("task_type")
    if isinstance(explicit, str):
        try:
            return TaskType(explicit)
        except ValueError:
            pass

    lowered = query_text.lower()
    if any(token in lowered for token in ("tool", "function", "api", "endpoint", "call")):
        return TaskType.TOOL_HEAVY
    if any(token in lowered for token in ("remember", "recall", "history", "what do you know", "preference")):
        return TaskType.KNOWLEDGE_INTENSIVE
    if not has_existing_history or bool(request_metadata.get("new_session")):
        return TaskType.NEW_SESSION_WITH_HISTORY
    return TaskType.CONTINUATION


def allocate_budget(
    *,
    context_window_size: int,
    system_prompt_tokens: int,
    response_reserve: int,
    tool_budget: int,
    task_type: TaskType,
    memory_override: int | None = None,
    history_override: int | None = None,
) -> ContextBudget:
    available_budget = max(context_window_size - system_prompt_tokens - response_reserve - tool_budget, 0)
    memory_ratio, history_ratio = TASK_BUDGET_RATIOS[task_type]

    memory_budget = int(available_budget * memory_ratio)
    history_budget = available_budget - memory_budget

    if memory_override is not None or history_override is not None:
        if memory_override is not None:
            memory_budget = max(memory_override, 0)
        if history_override is not None:
            history_budget = max(history_override, 0)
        total_requested = memory_budget + history_budget
        if total_requested > available_budget and total_requested > 0:
            scale = available_budget / total_requested
            memory_budget = int(memory_budget * scale)
            history_budget = available_budget - memory_budget

    return ContextBudget(
        total_budget=available_budget,
        memory_budget=memory_budget,
        history_budget=history_budget,
        response_reserve=response_reserve,
        tool_budget=tool_budget,
    )


def format_memory_block(records: Sequence) -> str:
    lines = ["[Memory Context]", "Use the following user-specific facts only when relevant."]
    for scored in records:
        memory = scored.memory
        scope = "global" if memory.memory_scope is MemoryScope.GLOBAL else "local"
        lines.append(
            f"- L{int(memory.memory_tier)}/{scope}/conf={memory.confidence:.2f}: {memory.content_text}"
        )
    return "\n".join(lines)
