"""Integrated OpenAgentBench tool catalog and reference handlers."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlparse
from uuid import uuid4

from openagentbench.agent_data import (
    OPENAI_ENDPOINTS,
    CompileRequest,
    ContextCompiler,
    MemoryRecord,
    MemoryScope,
    MemoryTier,
    ProvenanceType,
    SessionRecord,
    hash_normalized_text,
)
from openagentbench.agent_memory import WorkingMemoryItem, build_memory_inspect_tool_definition, build_memory_read_tool_definition
from openagentbench.agent_memory.endpoint_compat import build_memory_write_tool_definition
from openagentbench.agent_memory.compiler import filter_scoped_memories
from openagentbench.agent_retrieval import ModelRouter, classify_query, default_profiles
from openagentbench.agent_retrieval.scoring import lexical_overlap_score

from .enums import MutationClass, TimeoutClass, ToolSourceType, TypeClass
from .models import (
    AuthContract,
    IdempotencySpec,
    ObservabilityContract,
    SideEffectManifest,
    ToolDescriptor,
)
from .router import ToolExecutionEngine


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _function_tool_definition(*, name: str, description: str, parameters: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": parameters,
        },
    }


def build_data_compile_context_tool_definition() -> dict[str, Any]:
    return _function_tool_definition(
        name="data_compile_context",
        description="Compile user-scoped history and memories into model-ready messages.",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "session_id": {"type": "string"},
                "tool_budget": {"type": "integer", "minimum": 0, "maximum": 4096},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    )


def build_data_endpoint_catalog_tool_definition() -> dict[str, Any]:
    return _function_tool_definition(
        name="data_endpoint_catalog",
        description="List OpenAI endpoint coverage tracked by the agent_data catalog.",
        parameters={
            "type": "object",
            "properties": {
                "modality": {"type": "string"},
            },
            "additionalProperties": False,
        },
    )


def build_retrieval_plan_tool_definition() -> dict[str, Any]:
    return _function_tool_definition(
        name="retrieval_plan",
        description="Classify a query and return the retrieval/model-routing plan.",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "session_summary": {"type": "string"},
                "turn_count": {"type": "integer", "minimum": 0, "maximum": 1_000_000},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    )


def build_tool_registry_list_definition() -> dict[str, Any]:
    return _function_tool_definition(
        name="tool_registry_list",
        description="List active tool contracts from the unified tool registry.",
        parameters={
            "type": "object",
            "properties": {
                "task_hint": {"type": "string"},
                "token_budget": {"type": "integer", "minimum": 1, "maximum": 8192},
            },
            "additionalProperties": False,
        },
    )


def build_tool_endpoint_matrix_definition() -> dict[str, Any]:
    return _function_tool_definition(
        name="tool_endpoint_matrix",
        description="Return the model-compatibility matrix and protocol payload shapes for tools.",
        parameters={
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
    )


def build_browser_navigate_tool_definition() -> dict[str, Any]:
    return _function_tool_definition(
        name="browser_navigate",
        description="Navigate a browser session and return lightweight page state.",
        parameters={
            "type": "object",
            "properties": {
                "url": {"type": "string"},
                "capture_screenshot": {"type": "boolean"},
            },
            "required": ["url"],
            "additionalProperties": False,
        },
    )


def build_vision_describe_tool_definition() -> dict[str, Any]:
    return _function_tool_definition(
        name="vision_describe",
        description="Describe an image or screenshot for multimodal tool workflows.",
        parameters={
            "type": "object",
            "properties": {
                "image_ref": {"type": "string"},
                "prompt": {"type": "string"},
                "max_tokens": {"type": "integer", "minimum": 1, "maximum": 4096},
            },
            "required": ["image_ref"],
            "additionalProperties": False,
        },
    )


def build_a2a_delegate_tool_definition() -> dict[str, Any]:
    return _function_tool_definition(
        name="a2a_delegate",
        description="Delegate a task to another agent using an A2A-style task card.",
        parameters={
            "type": "object",
            "properties": {
                "agent_name": {"type": "string"},
                "task": {"type": "string"},
                "artifacts": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "uri": {"type": "string"},
                        },
                        "required": ["name", "uri"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["agent_name", "task"],
            "additionalProperties": False,
        },
    )


def build_default_tool_definitions() -> tuple[dict[str, Any], ...]:
    return (
        build_data_compile_context_tool_definition(),
        build_data_endpoint_catalog_tool_definition(),
        build_retrieval_plan_tool_definition(),
        build_memory_read_tool_definition(),
        build_memory_write_tool_definition(),
        build_memory_inspect_tool_definition(),
        build_tool_registry_list_definition(),
        build_tool_endpoint_matrix_definition(),
        build_browser_navigate_tool_definition(),
        build_vision_describe_tool_definition(),
        build_a2a_delegate_tool_definition(),
    )


def _success_output_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": True,
    }


def _error_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "code": {"type": "string"},
            "message": {"type": "string"},
            "retryable": {"type": "boolean"},
        },
        "required": ["code", "message", "retryable"],
        "additionalProperties": True,
    }


def _descriptor_from_tool_definition(
    tool_definition: dict[str, Any],
    *,
    type_class: TypeClass,
    mutation_class: MutationClass,
    handler,
    required_scopes: tuple[str, ...],
    timeout_class: TimeoutClass = TimeoutClass.STANDARD,
    cache_ttl_seconds: int = 0,
    side_effect_manifest: SideEffectManifest | None = None,
) -> ToolDescriptor:
    function = tool_definition["function"]
    source_type = {
        TypeClass.FUNCTION: ToolSourceType.FUNCTION,
        TypeClass.METHOD: ToolSourceType.FUNCTION,
        TypeClass.MCP_TOOL: ToolSourceType.MCP,
        TypeClass.MCP_RESOURCE: ToolSourceType.MCP,
        TypeClass.MCP_PROMPT: ToolSourceType.MCP,
        TypeClass.JSON_RPC: ToolSourceType.JSONRPC,
        TypeClass.GRPC: ToolSourceType.GRPC,
        TypeClass.A2A_AGENT: ToolSourceType.A2A,
        TypeClass.SDK_WRAPPED: ToolSourceType.SDK,
        TypeClass.BROWSER: ToolSourceType.BROWSER,
        TypeClass.DESKTOP: ToolSourceType.DESKTOP,
        TypeClass.VISION: ToolSourceType.VISION,
        TypeClass.COMPOSITE: ToolSourceType.COMPOSITE,
    }[type_class]
    return ToolDescriptor(
        tool_id=function["name"],
        version="1.0.0",
        type_class=type_class,
        input_schema=function["parameters"],
        output_schema=_success_output_schema(),
        error_schema=_error_schema(),
        auth_contract=AuthContract(required_scopes=required_scopes),
        timeout_class=timeout_class,
        idempotency_spec=IdempotencySpec(enabled=mutation_class is not MutationClass.READ_ONLY),
        mutation_class=mutation_class,
        observability_contract=ObservabilityContract(metric_prefix=f"agent_tools.{function['name']}"),
        source_endpoint="openagentbench://agent_tools/catalog",
        source_type=source_type,
        compressed_description=function["description"],
        handler=handler,
        side_effect_manifest=side_effect_manifest,
        cache_ttl_seconds=cache_ttl_seconds,
    )


def _memory_type_to_provenance(memory_type: str) -> ProvenanceType:
    return {
        "fact": ProvenanceType.FACT,
        "preference": ProvenanceType.PREFERENCE,
        "correction": ProvenanceType.CORRECTION,
        "constraint": ProvenanceType.INSTRUCTION,
        "procedure": ProvenanceType.INSTRUCTION,
    }.get(memory_type, ProvenanceType.TOOL_OUTPUT)


def _layer_from_name(name: str) -> MemoryTier:
    return {
        "session": MemoryTier.SESSION,
        "episodic": MemoryTier.EPISODIC,
        "semantic": MemoryTier.SEMANTIC,
        "procedural": MemoryTier.PROCEDURAL,
        "auto": MemoryTier.SEMANTIC,
    }[name]


def _scope_from_name(name: str) -> MemoryScope:
    return MemoryScope.GLOBAL if name == "global" else MemoryScope.LOCAL


@dataclass(slots=True)
class OpenAgentBenchToolSuite:
    sessions: dict[Any, SessionRecord] = field(default_factory=dict)
    history_by_session: dict[Any, list[Any]] = field(default_factory=dict)
    memories_by_user: dict[Any, list[MemoryRecord]] = field(default_factory=dict)
    working_by_session: dict[tuple[Any, Any], list[WorkingMemoryItem]] = field(default_factory=dict)

    def register_into(self, engine: ToolExecutionEngine) -> tuple[ToolDescriptor, ...]:
        descriptors = (
            self._data_compile_context_descriptor(),
            self._data_endpoint_catalog_descriptor(),
            self._retrieval_plan_descriptor(),
            self._memory_read_descriptor(),
            self._memory_write_descriptor(),
            self._memory_inspect_descriptor(),
            self._registry_list_descriptor(engine),
            self._endpoint_matrix_descriptor(),
            self._browser_navigate_descriptor(),
            self._vision_describe_descriptor(),
            self._a2a_delegate_descriptor(),
        )
        for descriptor in descriptors:
            engine.register(descriptor)
        return descriptors

    def _resolve_session(self, session_id, context) -> SessionRecord:
        resolved_id = session_id or context.session_id
        if resolved_id is None or resolved_id not in self.sessions:
            raise ValueError("session_id is required and must reference a known session")
        return self.sessions[resolved_id]

    def _user_memories(self, user_id) -> list[MemoryRecord]:
        return list(self.memories_by_user.get(user_id, ()))

    def _data_compile_context_descriptor(self) -> ToolDescriptor:
        definition = build_data_compile_context_tool_definition()

        def handler(params: dict[str, Any], context) -> dict[str, Any]:
            session = self._resolve_session(params.get("session_id"), context)
            compiler = ContextCompiler()
            memories = filter_scoped_memories(memories=self._user_memories(context.user_id), session_id=session.session_id)
            history = list(self.history_by_session.get(session.session_id, ()))
            compiled = compiler.compile_context(
                CompileRequest(
                    user_id=context.user_id,
                    session=session,
                    query_text=str(params["query"]),
                    tool_token_budget=int(params.get("tool_budget", 0) or 0),
                ),
                history=history,
                memories=memories,
            )
            return {
                "messages": compiled.messages,
                "tokens_used": compiled.tokens_used,
                "task_type": compiled.task_type.value,
                "selected_memory_ids": [str(item.memory.memory_id) for item in compiled.selected_memories],
            }

        return _descriptor_from_tool_definition(
            definition,
            type_class=TypeClass.FUNCTION,
            mutation_class=MutationClass.READ_ONLY,
            handler=handler,
            required_scopes=("tools.read",),
            timeout_class=TimeoutClass.STANDARD,
            cache_ttl_seconds=60,
        )

    def _data_endpoint_catalog_descriptor(self) -> ToolDescriptor:
        definition = build_data_endpoint_catalog_tool_definition()

        def handler(params: dict[str, Any], _) -> dict[str, Any]:
            modality = str(params["modality"]).strip().lower() if params.get("modality") else None
            endpoints = [
                {
                    "name": endpoint.name,
                    "path": endpoint.path,
                    "input_modalities": list(endpoint.input_modalities),
                    "output_modalities": list(endpoint.output_modalities),
                    "supports_mixed_content": endpoint.supports_mixed_content,
                }
                for endpoint in OPENAI_ENDPOINTS
                if modality is None
                or modality in endpoint.input_modalities
                or modality in endpoint.output_modalities
            ]
            return {"endpoints": endpoints, "count": len(endpoints)}

        return _descriptor_from_tool_definition(
            definition,
            type_class=TypeClass.FUNCTION,
            mutation_class=MutationClass.READ_ONLY,
            handler=handler,
            required_scopes=("tools.read",),
            timeout_class=TimeoutClass.INTERACTIVE,
            cache_ttl_seconds=300,
        )

    def _retrieval_plan_descriptor(self) -> ToolDescriptor:
        definition = build_retrieval_plan_tool_definition()

        def handler(params: dict[str, Any], context) -> dict[str, Any]:
            query = str(params["query"])
            session_summary = str(params.get("session_summary") or "")
            turn_count = int(params.get("turn_count") or 0)
            if not session_summary and context.session_id in self.sessions:
                session_summary = self.sessions[context.session_id].summary_text or ""
                turn_count = self.sessions[context.session_id].turn_count
            classification = classify_query(query, session_summary, turn_count=turn_count)
            plan = ModelRouter().select(classification, default_profiles())
            return {
                "query_type": classification.type.value,
                "loop_strategy": classification.loop_strategy.value,
                "reasoning_effort": classification.reasoning_effort.value,
                "preferred_modalities": [modality.value for modality in classification.preferred_modalities],
                "output_streams": [stream.value for stream in classification.output_streams],
                "role_bindings": {
                    role.value: profile.model_name
                    for role, profile in plan.role_bindings.items()
                },
                "primary_model": None if plan.primary_model is None else plan.primary_model.model_name,
            }

        return _descriptor_from_tool_definition(
            definition,
            type_class=TypeClass.FUNCTION,
            mutation_class=MutationClass.READ_ONLY,
            handler=handler,
            required_scopes=("tools.read",),
            timeout_class=TimeoutClass.INTERACTIVE,
            cache_ttl_seconds=120,
        )

    def _memory_read_descriptor(self) -> ToolDescriptor:
        definition = build_memory_read_tool_definition()
        parameters = definition["function"]["parameters"]
        parameters["properties"]["session_id"] = {"type": "string"}

        def handler(params: dict[str, Any], context) -> dict[str, Any]:
            session_id = params.get("session_id") or context.session_id
            query = str(params["query"])
            layer = str(params.get("layer") or "auto")
            scope = str(params.get("scope") or "auto")
            top_k = int(params.get("top_k") or 5)

            scoped = filter_scoped_memories(memories=self._user_memories(context.user_id), session_id=session_id)
            selected: list[dict[str, Any]] = []
            for memory in scoped:
                if layer != "auto" and memory.memory_tier is not _layer_from_name(layer):
                    continue
                if scope == "global" and memory.memory_scope is not MemoryScope.GLOBAL:
                    continue
                if scope in {"local", "session"} and memory.memory_scope is not MemoryScope.LOCAL:
                    continue
                score = 0.75 * lexical_overlap_score(query, memory.content_text) + 0.25 * memory.confidence
                selected.append(
                    {
                        "memory_id": str(memory.memory_id),
                        "layer": memory.memory_tier.name.lower(),
                        "scope": "global" if memory.memory_scope is MemoryScope.GLOBAL else "local",
                        "content": memory.content_text,
                        "score": round(score, 4),
                        "updated_at": memory.updated_at.isoformat(),
                    }
                )
            selected.sort(key=lambda item: (-item["score"], item["updated_at"]), reverse=False)
            return {"items": selected[:top_k], "count": min(len(selected), top_k)}

        return _descriptor_from_tool_definition(
            definition,
            type_class=TypeClass.FUNCTION,
            mutation_class=MutationClass.READ_ONLY,
            handler=handler,
            required_scopes=("tools.read",),
            timeout_class=TimeoutClass.STANDARD,
            cache_ttl_seconds=45,
        )

    def _memory_write_descriptor(self) -> ToolDescriptor:
        definition = build_memory_write_tool_definition()

        def handler(params: dict[str, Any], context) -> dict[str, Any]:
            target_layer = str(params["target_layer"])
            target_scope = str(params["target_scope"])
            content = str(params["content"])
            now = _utc_now()
            session_id = context.session_id if target_scope != "global" else None
            record = MemoryRecord(
                memory_id=uuid4(),
                user_id=context.user_id,
                session_id=session_id,
                memory_tier=_layer_from_name(target_layer),
                memory_scope=_scope_from_name(target_scope),
                content_text=content,
                content_embedding=None,
                content_hash=hash_normalized_text(content),
                provenance_type=_memory_type_to_provenance(str(params["memory_type"])),
                provenance_turn_id=None,
                confidence=float(params.get("confidence", 0.8)),
                relevance_accumulator=1.0,
                access_count=0,
                last_accessed_at=None,
                created_at=now,
                updated_at=now,
                expires_at=None,
                is_active=True,
                is_validated=True,
                token_count=max(1, len(content.split())),
                tags=("agent_tools", target_layer, target_scope),
                metadata={"written_by": "memory_write"},
            )
            self.memories_by_user.setdefault(context.user_id, []).append(record)
            return {
                "memory_id": str(record.memory_id),
                "layer": record.memory_tier.name.lower(),
                "scope": "global" if record.memory_scope is MemoryScope.GLOBAL else "local",
                "content": record.content_text,
                "created_at": record.created_at.isoformat(),
            }

        return _descriptor_from_tool_definition(
            definition,
            type_class=TypeClass.FUNCTION,
            mutation_class=MutationClass.WRITE_REVERSIBLE,
            handler=handler,
            required_scopes=("tools.write",),
            timeout_class=TimeoutClass.STANDARD,
            side_effect_manifest=SideEffectManifest(
                resources=("agent_data.memory_store",),
                operations=("insert",),
                reversible=True,
            ),
        )

    def _memory_inspect_descriptor(self) -> ToolDescriptor:
        definition = build_memory_inspect_tool_definition()
        parameters = definition["function"]["parameters"]
        parameters["properties"]["session_id"] = {"type": "string"}

        def handler(params: dict[str, Any], context) -> dict[str, Any]:
            session_id = params.get("session_id") or context.session_id
            memories = filter_scoped_memories(memories=self._user_memories(context.user_id), session_id=session_id)
            counts_by_tier = {tier.name.lower(): 0 for tier in MemoryTier if tier is not MemoryTier.WORKING}
            counts_by_scope = {"local": 0, "global": 0}
            for memory in memories:
                counts_by_tier[memory.memory_tier.name.lower()] += 1
                counts_by_scope["global" if memory.memory_scope is MemoryScope.GLOBAL else "local"] += 1
            working_items = self.working_by_session.get((context.user_id, session_id), ())
            return {
                "counts_by_tier": counts_by_tier,
                "counts_by_scope": counts_by_scope,
                "working_items": len(working_items),
                "include_audit": bool(params.get("include_audit", False)),
            }

        return _descriptor_from_tool_definition(
            definition,
            type_class=TypeClass.FUNCTION,
            mutation_class=MutationClass.READ_ONLY,
            handler=handler,
            required_scopes=("tools.read",),
            timeout_class=TimeoutClass.INTERACTIVE,
            cache_ttl_seconds=30,
        )

    def _registry_list_descriptor(self, engine: ToolExecutionEngine) -> ToolDescriptor:
        definition = build_tool_registry_list_definition()

        def handler(params: dict[str, Any], _) -> dict[str, Any]:
            task_hint = str(params["task_hint"]) if params.get("task_hint") else None
            if task_hint:
                selected = engine.registry.select_tools_for_task(
                    task_objective=task_hint,
                    token_budget=int(params.get("token_budget", 512) or 512),
                )
                summaries = [asdict(tool.as_summary()) for tool in selected]
            else:
                summaries = [asdict(summary) for summary in engine.registry.compressed_index()]
            return {"tools": summaries, "count": len(summaries)}

        return _descriptor_from_tool_definition(
            definition,
            type_class=TypeClass.FUNCTION,
            mutation_class=MutationClass.READ_ONLY,
            handler=handler,
            required_scopes=("tools.admin",),
            timeout_class=TimeoutClass.INTERACTIVE,
            cache_ttl_seconds=15,
        )

    def _endpoint_matrix_descriptor(self) -> ToolDescriptor:
        definition = build_tool_endpoint_matrix_definition()

        def handler(_: dict[str, Any], __) -> dict[str, Any]:
            from .endpoint_compat import build_agent_tools_endpoint_compatibility_report

            report = build_agent_tools_endpoint_compatibility_report()
            return {
                "tool_count": len(report.tool_definitions),
                "openai_tool_count": len(report.openai_tools_format["tools"]),
                "anthropic_tool_count": len(report.anthropic_tools_format["tools"]),
                "google_tool_count": len(report.google_tools_format["function_declarations"]),
                "jsonrpc_method": report.jsonrpc_invoke_request["method"],
                "a2a_task_id": report.a2a_task_request["id"],
            }

        return _descriptor_from_tool_definition(
            definition,
            type_class=TypeClass.FUNCTION,
            mutation_class=MutationClass.READ_ONLY,
            handler=handler,
            required_scopes=("tools.read",),
            timeout_class=TimeoutClass.INTERACTIVE,
            cache_ttl_seconds=120,
        )

    def _browser_navigate_descriptor(self) -> ToolDescriptor:
        definition = build_browser_navigate_tool_definition()

        def handler(params: dict[str, Any], context) -> dict[str, Any]:
            url = str(params["url"])
            parsed = urlparse(url)
            title = parsed.netloc or parsed.path or "untitled"
            result = {
                "url": url,
                "title": title,
                "session_id": "" if context.session_id is None else str(context.session_id),
            }
            if params.get("capture_screenshot"):
                result["screenshot_ref"] = f"browser://{context.session_id or 'no-session'}/{hashlib.sha1(url.encode('utf-8')).hexdigest()}"
            return result

        import hashlib

        return _descriptor_from_tool_definition(
            definition,
            type_class=TypeClass.BROWSER,
            mutation_class=MutationClass.READ_ONLY,
            handler=handler,
            required_scopes=("tools.browser",),
            timeout_class=TimeoutClass.STANDARD,
        )

    def _vision_describe_descriptor(self) -> ToolDescriptor:
        definition = build_vision_describe_tool_definition()

        def handler(params: dict[str, Any], _) -> dict[str, Any]:
            prompt = str(params.get("prompt") or "Describe the referenced image.")
            return {
                "image_ref": str(params["image_ref"]),
                "description": f"{prompt} Reference={params['image_ref']}",
                "max_tokens": int(params.get("max_tokens") or 256),
            }

        return _descriptor_from_tool_definition(
            definition,
            type_class=TypeClass.VISION,
            mutation_class=MutationClass.READ_ONLY,
            handler=handler,
            required_scopes=("tools.vision",),
            timeout_class=TimeoutClass.STANDARD,
        )

    def _a2a_delegate_descriptor(self) -> ToolDescriptor:
        definition = build_a2a_delegate_tool_definition()

        def handler(params: dict[str, Any], context) -> dict[str, Any]:
            task_id = f"a2a-{uuid4()}"
            return {
                "task_id": task_id,
                "state": "working",
                "agent_name": str(params["agent_name"]),
                "task": str(params["task"]),
                "artifact_count": len(params.get("artifacts") or []),
                "submitted_by": str(context.agent_id),
            }

        return _descriptor_from_tool_definition(
            definition,
            type_class=TypeClass.A2A_AGENT,
            mutation_class=MutationClass.WRITE_REVERSIBLE,
            handler=handler,
            required_scopes=("tools.delegate",),
            timeout_class=TimeoutClass.LONG_RUNNING,
            side_effect_manifest=SideEffectManifest(
                resources=("a2a.task",),
                operations=("create",),
                reversible=True,
            ),
        )


__all__ = [
    "OpenAgentBenchToolSuite",
    "build_a2a_delegate_tool_definition",
    "build_browser_navigate_tool_definition",
    "build_data_compile_context_tool_definition",
    "build_data_endpoint_catalog_tool_definition",
    "build_default_tool_definitions",
    "build_retrieval_plan_tool_definition",
    "build_tool_endpoint_matrix_definition",
    "build_tool_registry_list_definition",
    "build_vision_describe_tool_definition",
]
