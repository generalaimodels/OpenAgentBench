# Agent Query Skills

## Objective

This file captures the operating skills for resolving hard user queries inside `agent_query`.

The module must automatically apply these skills during query understanding:

- ambiguity reduction
- constraint preservation
- tool-aware planning
- memory-governed enrichment
- decomposition under token budget
- protocol-aware routing
- clarification when blocking context is missing

## Skill 1: Ambiguity Elimination

When the query contains unstable referents such as `it`, `this`, `that`, `same`, or `again`, the system should:

1. attempt deterministic coreference repair from recent history
2. preserve the repaired referent in the rewritten query
3. emit a blocking clarification question if the referent remains underspecified

## Skill 2: Constraint Extraction

Every explicit or implicit constraint must be preserved through rewrite and decomposition:

- `without`
- `using`
- `must`
- `only`
- `before`
- `after`
- environment constraints
- model/provider constraints

## Skill 3: Tool-Aware Decomposition

The decomposition phase must not split only by syntax. It must also consider:

- available tool affordances
- protocol compatibility
- likely mutation risk
- latency sensitivity
- whether a sub-step should route to memory, retrieval, tools, or delegated analysis

## Skill 4: Memory-Governed Enrichment

The module should enrich a query from memory only when the added context is relevant and scoped correctly:

- session memory stays session-scoped
- local episodic memory must not leak across sessions
- semantic and procedural memory may guide rewrite and routing
- every enrichment must remain attributable to a provenance source

## Skill 5: Budget Discipline

The engine must operate under a bounded query-understanding budget:

- reserve a fixed fraction of the full context window for query understanding
- allocate stage budgets deterministically
- prefer high-utility tool schemas over bulk inclusion
- degrade gracefully by pruning low-value enrichments first

## Skill 6: Clarification Strategy

Clarification is required only when missing information blocks safe or correct routing. Good clarification questions are:

- short
- specific
- tied to a missing slot
- actionable for the user

## Skill 7: Verification Checklist

Before a query plan is returned, the module should validate:

- semantic intent is preserved
- constraints survive rewrite
- decomposition is bounded
- route targets are executable
- provider payloads remain OpenAI/vLLM compatible
- clarification is emitted when ambiguity is still blocking

## Skill 8: Psychological-State and Pressure Analysis

For complex real-world problems, the module should estimate non-clinical behavioral signals from the query and surrounding context:

- stress load
- urgency and time pressure
- frustration and emotional escalation
- uncertainty and confusion
- signs of a controlling or hostile environment

This is not for diagnosis. It is for better planning, routing, and clarification under difficult human conditions.

## Skill 9: Hidden Goal Inference

Complex users often ask for one thing while optimizing for another. The module should infer latent goals such as:

- risk reduction
- autonomy preservation
- conflict avoidance
- task completion under pressure
- reputation or relationship protection

These latent goals should influence rewrite, decomposition order, and the tone of clarification.

## Skill 10: High-Pressure Complex Problem Handling

When the problem is psychologically heavy or strategically constrained, the module should automatically:

1. preserve the user’s stated objective
2. surface hidden blockers
3. reduce ambiguity before action
4. favor safe reversible steps early
5. keep explanations structured and decision-oriented
6. avoid collapsing emotionally loaded requests into generic advice
