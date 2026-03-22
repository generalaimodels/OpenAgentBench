"""Prefill compiler for the memory module."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Sequence

from openagentbench.agent_data import MemoryRecord, MemoryScope
from openagentbench.agent_data.enums import MemoryTier
from openagentbench.agent_retrieval import QueryType, classify_query
from openagentbench.agent_retrieval.scoring import tokenize

from .budgeting import allocate_layer_budgets
from .models import BudgetAllocation, CompiledMemoryContext, MemoryCompileRequest, MemoryFragment, WorkingMemoryItem
from .scoring import episodic_recall_score, memory_record_priority, working_memory_utility


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _format_fragment_line(fragment: MemoryFragment) -> str:
    suffix = ""
    ref = fragment.metadata.get("ref")
    if isinstance(ref, str) and ref:
        suffix = f" ref={ref}"
    return f"- score={fragment.score:.3f} {fragment.content}{suffix}"


def _format_memory_section(title: str, fragments: Sequence[MemoryFragment]) -> str:
    lines = [title]
    lines.extend(_format_fragment_line(fragment) for fragment in fragments)
    return "\n".join(lines)


def _pack_fragments(
    *,
    fragments: Sequence[MemoryFragment],
    token_budget: int,
) -> list[MemoryFragment]:
    selected: list[MemoryFragment] = []
    running_total = 0
    for fragment in sorted(fragments, key=lambda item: (-item.score / max(item.token_count, 1), -item.score, item.token_count)):
        if running_total + fragment.token_count > token_budget:
            continue
        selected.append(fragment)
        running_total += fragment.token_count
    return selected


def filter_scoped_memories(
    *,
    memories: Sequence[MemoryRecord],
    session_id,
) -> list[MemoryRecord]:
    scoped: list[MemoryRecord] = []
    for memory in memories:
        if memory.memory_scope is MemoryScope.GLOBAL:
            scoped.append(memory)
            continue
        if memory.session_id == session_id:
            scoped.append(memory)
    return scoped


def _scope_adjusted_priority(memory: MemoryRecord, *, query_text: str, session_id, now: datetime) -> float:
    base_score = episodic_recall_score(memory=memory, query_text=query_text, now=now)
    if memory.memory_tier is not MemoryTier.EPISODIC:
        base_score = memory_record_priority(memory=memory, query_text=query_text, now=now)
    if memory.memory_scope is MemoryScope.LOCAL:
        if memory.session_id != session_id:
            return -1.0
        base_score *= 1.20
    else:
        if memory.memory_tier in {MemoryTier.SEMANTIC, MemoryTier.PROCEDURAL}:
            base_score *= 1.05
    if memory.memory_tier is MemoryTier.SESSION and memory.session_id == session_id:
        base_score *= 1.25
    return base_score


@dataclass(slots=True)
class MemoryContextCompiler:
    def compile_context(
        self,
        request: MemoryCompileRequest,
        *,
        memories: Sequence[MemoryRecord],
        working_items: Sequence[WorkingMemoryItem] = (),
    ) -> CompiledMemoryContext:
        classification = request.classification or classify_query(
            request.query_text,
            request.session.summary_text or "",
            turn_count=request.session.turn_count,
        )
        budget = allocate_layer_budgets(total_budget=request.total_budget, classification=classification)
        current_time = _utc_now()
        scoped_memories = filter_scoped_memories(memories=memories, session_id=request.session.session_id)

        working_fragments = [
            MemoryFragment(
                layer=MemoryTier.WORKING,
                content=item.content_text,
                token_count=item.token_count,
                score=working_memory_utility(item=item, query_text=request.query_text, now=current_time),
                source_id=item.item_id,
                modality=item.modality,
                metadata={"ref": item.binary_ref} if item.binary_ref else {},
            )
            for item in working_items
        ]
        selected_working_fragments = _pack_fragments(fragments=working_fragments, token_budget=budget.working_budget)
        selected_working_ids = {fragment.source_id for fragment in selected_working_fragments}
        selected_working = [item for item in working_items if item.item_id in selected_working_ids]

        memory_fragments: list[MemoryFragment] = []
        for memory in scoped_memories:
            score = _scope_adjusted_priority(
                memory,
                query_text=request.query_text,
                session_id=request.session.session_id,
                now=current_time,
            )
            if score < 0.0:
                continue
            memory_fragments.append(
                MemoryFragment(
                    layer=memory.memory_tier,
                    content=memory.content_text,
                    token_count=memory.token_count,
                    score=score,
                    source_id=memory.memory_id,
                    metadata={
                        "scope": "global" if memory.memory_scope is MemoryScope.GLOBAL else "local",
                        "tags": list(memory.tags),
                        **memory.metadata,
                    },
                )
            )

        fragments_by_tier = {
            tier: [fragment for fragment in memory_fragments if fragment.layer is tier]
            for tier in (MemoryTier.SESSION, MemoryTier.EPISODIC, MemoryTier.SEMANTIC, MemoryTier.PROCEDURAL)
        }
        selected_fragments = [
            *_pack_fragments(fragments=fragments_by_tier[MemoryTier.SESSION], token_budget=budget.session_budget),
            *_pack_fragments(fragments=fragments_by_tier[MemoryTier.EPISODIC], token_budget=budget.episodic_budget),
            *_pack_fragments(fragments=fragments_by_tier[MemoryTier.SEMANTIC], token_budget=budget.semantic_budget),
            *_pack_fragments(fragments=fragments_by_tier[MemoryTier.PROCEDURAL], token_budget=budget.procedural_budget),
        ]

        messages: list[dict[str, object]] = []
        system_prompt = request.system_prompt_text or request.session.system_prompt_text
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if request.session.summary_text and budget.session_budget > 0:
            summary_tokens = len(tokenize(request.session.summary_text))
            if summary_tokens <= budget.session_budget:
                messages.append({"role": "system", "content": f"[Session Summary]\n{request.session.summary_text}"})

        if selected_working_fragments:
            messages.append(
                {"role": "system", "content": _format_memory_section("[Working Memory]", selected_working_fragments)}
            )
        for title, tier, scope in (
            ("[Session Memory]", MemoryTier.SESSION, None),
            ("[Local Episodic Memory]", MemoryTier.EPISODIC, "local"),
            ("[Global Semantic Memory]", MemoryTier.SEMANTIC, "global"),
            ("[Global Procedures]", MemoryTier.PROCEDURAL, "global"),
        ):
            tier_fragments = [fragment for fragment in selected_fragments if fragment.layer is tier]
            if scope is not None:
                tier_fragments = [fragment for fragment in tier_fragments if fragment.metadata.get("scope") == scope]
            if not tier_fragments:
                continue
            messages.append({"role": "system", "content": _format_memory_section(title, tier_fragments)})

        if classification.type is QueryType.MULTIMODAL and budget.multimodal_budget > 0:
            multimodal_refs = []
            for fragment in [*selected_working_fragments, *selected_fragments]:
                ref = fragment.metadata.get("ref") or fragment.metadata.get("modality_ref")
                if isinstance(ref, str) and ref:
                    multimodal_refs.append(ref)
            if multimodal_refs:
                messages.append(
                    {
                        "role": "system",
                        "content": "[Multimodal References]\n" + "\n".join(f"- {ref}" for ref in multimodal_refs[:8]),
                    }
                )

        total_tokens = 0
        for message in messages:
            content = message.get("content")
            if isinstance(content, str):
                total_tokens += len(tokenize(content))

        return CompiledMemoryContext(
            messages=messages,
            budget=budget,
            selected_working=selected_working,
            selected_fragments=selected_fragments,
            classification=classification,
            total_tokens=total_tokens,
        )

__all__ = ["MemoryContextCompiler", "filter_scoped_memories"]
