"""Session summarization, turn marker detection, and checkpoint helpers."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Iterable

from openagentbench.agent_data import HistoryRecord, SessionRecord
from openagentbench.agent_retrieval import Modality
from openagentbench.agent_retrieval.scoring import tokenize

from .models import SessionCheckpointRecord, SessionTurnMarkers, WorkingMemoryItem, new_checkpoint
from .providers import MemoryProviderSuite


def detect_turn_markers(text: str) -> SessionTurnMarkers:
    lowered = f" {text.lower()} "
    correction_flag = any(
        token in lowered
        for token in (
            " actually ",
            " correction ",
            " correction:",
            " instead ",
            " not that ",
            " i meant ",
            " corrected ",
        )
    )
    decision_flag = any(
        token in lowered
        for token in (
            " decide ",
            " decision ",
            " decision:",
            " choose ",
            " selected ",
            " final approach ",
        )
    )
    segment_boundary = any(token in lowered for token in (" next task ", " new topic ", " separate issue ", " moving on "))
    return SessionTurnMarkers(
        correction_flag=correction_flag,
        decision_flag=decision_flag,
        segment_boundary=segment_boundary,
    )


def project_turn_text(record: HistoryRecord, providers: MemoryProviderSuite | None = None) -> str:
    if isinstance(record.content, str) and record.content.strip():
        return record.content.strip()
    parts = []
    for part in record.content_parts or ():
        part_type = str(part.get("type"))
        if part_type.endswith("text"):
            text = part.get("text")
            if isinstance(text, str) and text.strip():
                parts.append(text.strip())
            continue
        if "image" in part_type:
            reference = part.get("image_url")
            if isinstance(reference, str):
                if providers is None:
                    parts.append(f"[Image reference: {reference}]")
                else:
                    parts.append(f"[Image] {providers.describe_visual_reference(reference)}")
            continue
        if "audio" in part_type:
            reference = part.get("audio_url") or part.get("file_id")
            if isinstance(reference, str):
                if providers is None:
                    parts.append(f"[Audio reference: {reference}]")
                else:
                    parts.append(f"[Audio] {providers.transcribe_reference(reference)}")
    return " ".join(part for part in parts if part).strip()


def _truncate_summary_lines(lines: Iterable[str], max_tokens: int) -> str:
    if max_tokens <= 0:
        return ""
    output_lines: list[str] = []
    running = 0
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        tokens = len(tokenize(stripped))
        if running + tokens > max_tokens:
            break
        output_lines.append(stripped)
        running += tokens
    return "\n".join(output_lines)


def update_session_summary(
    *,
    existing_summary: str,
    new_turns: Sequence[HistoryRecord],
    max_tokens: int,
    providers: MemoryProviderSuite | None = None,
) -> str:
    high_priority_lines: list[str] = []
    normal_lines: list[str] = []

    for turn in new_turns:
        projected = project_turn_text(turn, providers)
        if not projected:
            continue
        markers = detect_turn_markers(projected)
        prefix = f"[T{turn.turn_index} {turn.role.as_openai_role()}]"
        if markers.correction_flag:
            high_priority_lines.append(f"{prefix} [Correction] {projected}")
        elif markers.decision_flag:
            high_priority_lines.append(f"{prefix} [Decision] {projected}")
        else:
            normal_lines.append(f"{prefix} {projected}")

    if providers is None:
        combined = [line for line in (existing_summary.strip(), *high_priority_lines, *normal_lines) if line]
        return _truncate_summary_lines(combined, max_tokens)

    summary = providers.summarize(
        existing_summary=existing_summary,
        additions=tuple([*high_priority_lines, *normal_lines]),
        max_tokens=max_tokens,
    )
    if not summary.strip():
        combined = [line for line in (existing_summary.strip(), *high_priority_lines, *normal_lines) if line]
        return _truncate_summary_lines(combined, max_tokens)
    if high_priority_lines:
        summary = _truncate_summary_lines([*high_priority_lines, summary], max_tokens)
    return summary


def build_session_checkpoint(
    *,
    session: SessionRecord,
    checkpoint_seq: int,
    summary_text: str,
    summary_version: int,
    turn_count: int,
    working_items: Sequence[WorkingMemoryItem],
    metadata: dict[str, object] | None = None,
) -> SessionCheckpointRecord:
    return new_checkpoint(
        user_id=session.user_id,
        session_id=session.session_id,
        checkpoint_seq=checkpoint_seq,
        summary_text=summary_text,
        summary_version=summary_version,
        turn_count=turn_count,
        working_items=working_items,
        metadata=metadata,
    )


__all__ = [
    "build_session_checkpoint",
    "detect_turn_markers",
    "project_turn_text",
    "update_session_summary",
]
