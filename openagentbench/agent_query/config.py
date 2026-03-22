"""Configuration and policy surfaces for the query-understanding module."""

from __future__ import annotations

from dataclasses import dataclass, field

DEFAULT_QUERY_BUDGET_RATIO = 0.15
DEFAULT_TOOL_TOKEN_BUDGET = 1024
DEFAULT_MAX_SUBQUERIES = 6
DEFAULT_QUERY_CACHE_TTL_SECONDS = 300
DEFAULT_QUERY_TOOL_BUDGET_MAXIMUM = 8192


@dataclass(slots=True, frozen=True)
class QueryBudgetPolicy:
    query_budget_ratio: float = DEFAULT_QUERY_BUDGET_RATIO
    context_weight: float = 0.40
    intent_weight: float = 0.10
    pragmatic_weight: float = 0.08
    cognitive_weight: float = 0.08
    rewrite_weight: float = 0.10
    decomposition_weight: float = 0.10
    routing_weight: float = 0.08
    clarification_weight: float = 0.03
    reserve_weight: float = 0.03

    def stage_weights(self) -> dict[str, float]:
        return {
            "context_budget": self.context_weight,
            "intent_budget": self.intent_weight,
            "pragmatic_budget": self.pragmatic_weight,
            "cognitive_budget": self.cognitive_weight,
            "rewrite_budget": self.rewrite_weight,
            "decomposition_budget": self.decomposition_weight,
            "routing_budget": self.routing_weight,
            "clarification_budget": self.clarification_weight,
            "reserve_budget": self.reserve_weight,
        }


@dataclass(slots=True, frozen=True)
class ToolAffordancePolicy:
    minimum_token_floor: int = 12
    per_property_token_cost: int = 6
    default_relevance_score: float = 0.25
    prior_relevance_weight: float = 0.75
    lexical_relevance_weight: float = 0.25
    minimum_selected_relevance: float = 0.05
    browser_prefixes: tuple[str, ...] = ("browser_",)
    vision_prefixes: tuple[str, ...] = ("vision_",)
    delegation_prefixes: tuple[str, ...] = ("a2a_",)
    memory_prefixes: tuple[str, ...] = ("memory_",)
    browser_protocols: tuple[str, ...] = ("browser", "vision")
    browser_tool_ids: tuple[str, ...] = ("browser_navigate", "vision_describe")
    browser_keywords: tuple[str, ...] = ("browser", "open", "navigate", "screenshot", "dashboard", "page", "site")
    terminal_keywords: tuple[str, ...] = (
        "terminal",
        "shell",
        "bash",
        "powershell",
        "command",
        "cmd",
        "console",
        "pytest",
        "unittest",
        "unit test",
        "test suite",
        "run tests",
        "execute tests",
    )
    terminal_tool_markers: tuple[str, ...] = ("terminal", "shell", "command")
    memory_keywords: tuple[str, ...] = ("remember", "recall", "history", "session", "preference", "constraint")
    navigation_keywords: tuple[str, ...] = ("browser", "screenshot", "vision")
    transactional_protocols: tuple[str, ...] = ("function", "json-rpc", "grpc", "mcp")


@dataclass(slots=True, frozen=True)
class PragmaticHeuristicPolicy:
    ambiguous_referents: tuple[str, ...] = ("it", "this", "that", "those", "them", "same", "again")
    urgent_hints: tuple[str, ...] = ("urgent", "asap", "immediately", "now", "deadline")
    frustrated_hints: tuple[str, ...] = ("broken", "frustrating", "annoying", "hate", "why is this failing")
    confused_hints: tuple[str, ...] = ("confused", "stuck", "not sure", "unclear", "what does this mean")
    expert_hints: tuple[str, ...] = ("json-rpc", "grpc", "mcp", "vector", "cache locality", "idempotency", "oauth", "jwt")
    novice_hints: tuple[str, ...] = ("beginner", "new to", "explain simply", "what is")
    control_environment_hints: tuple[str, ...] = ("control", "controlling", "micromanage", "pressure", "hostile", "toxic")
    stress_hints: tuple[str, ...] = ("stress", "stressed", "overwhelmed", "burnout", "panic", "fear", "anxiety")
    protection_goal_hints: tuple[str, ...] = ("protect", "preserve", "avoid harm", "stay safe", "reduce risk", "risk", "safest")
    autonomy_goal_hints: tuple[str, ...] = ("independent", "autonomy", "boundaries", "take control", "self-directed")
    resolution_goal_hints: tuple[str, ...] = ("solve", "achieve", "fix", "resolve", "handle")
    followup_action_prefixes: tuple[str, ...] = ("fix", "do", "use", "open")
    selection_criteria_markers: tuple[str, ...] = ("for", "under", "using")
    high_coreference_risk: float = 0.85
    medium_coreference_risk: float = 0.45
    required_clarification_risk_threshold: float = 0.80
    analytical_hints: tuple[str, ...] = ("compare", "tradeoff", "analyze", "diagnose", "why")
    transactional_hints: tuple[str, ...] = (
        "create",
        "update",
        "delete",
        "send",
        "book",
        "purchase",
        "write to",
        "run",
        "execute",
        "terminal",
        "shell",
        "bash",
        "powershell",
        "command",
    )
    generative_hints: tuple[str, ...] = ("write", "draft", "compose", "generate")
    navigational_hints: tuple[str, ...] = ("navigate", "open", "visit", "browser", "screenshot")
    verification_hints: tuple[str, ...] = ("verify", "validate", "check", "test", "confirm")
    memory_hints: tuple[str, ...] = ("remember", "recall", "preference", "history", "session")


@dataclass(slots=True, frozen=True)
class CognitiveHeuristicPolicy:
    token_normalizer: float = 80.0
    decomposition_bonus: float = 0.12
    tool_bonus: float = 0.04
    tool_count_cap: int = 4
    memory_bonus: float = 0.03
    memory_count_cap: int = 4
    advanced_query_bonus: float = 0.15
    structured_output_bonus: float = 0.05
    decomposition_threshold: float = 0.45
    max_estimated_steps: int = 6
    priority_floor: float = 0.10
    priority_decay: float = 0.10
    minimum_subquery_length: int = 6


@dataclass(slots=True, frozen=True)
class QueryProviderPolicy:
    default_completion_tokens: int = 256
    default_expand_tokens: int = 128
    default_hypothetical_tokens: int = 128
    default_clarification_tokens: int = 64
    deterministic_temperature: float = 0.0
    expand_prompt: str = (
        "Expand and rewrite the user's task for retrieval. Preserve requested actions, tool usage, entities, "
        "and constraints. Do not answer the task."
    )
    hypothetical_prompt: str = (
        "Produce a compact hypothetical result snippet for retrieval expansion. Stay inside the task context "
        "and do not discuss missing access or limitations."
    )
    clarification_prompt: str = "Produce one concise clarifying question for the ambiguous query. Do not answer the task."
    hypothetical_prefix: str = "Hypothesis:"
    clarification_template: str = "What specific target should I resolve for: {query}?"
    expansion_suffix: str = "relevant constraints evidence tools and history"


@dataclass(slots=True, frozen=True)
class QueryResolverPolicy:
    minimum_expand_tokens: int = 32
    minimum_hypothetical_tokens: int = 48
    minimum_intent_confidence: float = 0.55
    cache_ttl_seconds: int = DEFAULT_QUERY_CACHE_TTL_SECONDS
    continuation_question: str = "Which prior task or artifact should this query continue from?"


@dataclass(slots=True, frozen=True)
class QueryEndpointCompatibilityConfig:
    resolve_tool_name: str = "query_resolve"
    clarify_tool_name: str = "query_clarify"
    tool_budget_maximum: int = DEFAULT_QUERY_TOOL_BUDGET_MAXIMUM
    openai_responses_model: str = "gpt-5-mini"
    openai_chat_model: str = "gpt-4o-mini"
    openai_realtime_model: str = "gpt-realtime-mini"
    vllm_responses_model: str = "Qwen/Qwen2.5-72B-Instruct"
    vllm_chat_model: str = "meta-llama/Llama-3.1-70B-Instruct"
    openai_system_prompt: str = "You are a query resolution engine."
    openai_user_example: str = "Plan and verify my browser-based debugging workflow."
    openai_chat_example: str = "Use memory and tool context to answer this follow-up request."
    vllm_user_example: str = "Resolve, decompose, and route this multi-step request."
    vllm_context_example: str = "Context: the user previously asked for browser automation and memory inspection."
    vllm_chat_example: str = "Resolve this follow-up task and emit the route plan."
    vllm_chat_context_example: str = "Context: remember the PostgreSQL durability rule."
    gemini_system_instruction: str = "You are a query resolution checker."
    gemini_user_text: str = "Inspect ambiguity and route the request."
    reasoning_effort: str = "medium"


@dataclass(slots=True, frozen=True)
class QueryModuleConfig:
    budget_policy: QueryBudgetPolicy = field(default_factory=QueryBudgetPolicy)
    tool_policy: ToolAffordancePolicy = field(default_factory=ToolAffordancePolicy)
    pragmatic_policy: PragmaticHeuristicPolicy = field(default_factory=PragmaticHeuristicPolicy)
    cognitive_policy: CognitiveHeuristicPolicy = field(default_factory=CognitiveHeuristicPolicy)
    provider_policy: QueryProviderPolicy = field(default_factory=QueryProviderPolicy)
    resolver_policy: QueryResolverPolicy = field(default_factory=QueryResolverPolicy)
    endpoint_compatibility: QueryEndpointCompatibilityConfig = field(default_factory=QueryEndpointCompatibilityConfig)


__all__ = [
    "CognitiveHeuristicPolicy",
    "DEFAULT_MAX_SUBQUERIES",
    "DEFAULT_QUERY_BUDGET_RATIO",
    "DEFAULT_QUERY_CACHE_TTL_SECONDS",
    "DEFAULT_QUERY_TOOL_BUDGET_MAXIMUM",
    "DEFAULT_TOOL_TOKEN_BUDGET",
    "PragmaticHeuristicPolicy",
    "QueryBudgetPolicy",
    "QueryEndpointCompatibilityConfig",
    "QueryModuleConfig",
    "QueryProviderPolicy",
    "QueryResolverPolicy",
    "ToolAffordancePolicy",
]
