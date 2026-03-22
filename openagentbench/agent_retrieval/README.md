# Agent Retrieval

This folder is the executable review surface for the retrieval design in [plan.md](/mnt/c/Users/heman/Desktop/code/Agentic_frame_work/OpenAgentBench/openagentbench/agent_retrieval/plan.md).

It is meant to be easy for an expert reviewer to audit:

- the retrieval plan
- the current code shape
- the public API
- the SQL schema
- the compatibility layer
- the test coverage

## Plan Summary

The plan in [plan.md](/mnt/c/Users/heman/Desktop/code/Agentic_frame_work/OpenAgentBench/openagentbench/agent_retrieval/plan.md) targets a hybrid retrieval stack with:

- per-user isolation
- session, history, and memory retrieval tiers
- exact plus semantic retrieval
- multimodal and MIMO-aware routing
- selective model-purpose assignment
- OpenAI-compatible and Gemini-compatible boundaries
- provenance, budget fitting, and quality gates

This folder contains the executable reference implementation of that plan, plus compatibility and smoke utilities around it.

## Review Map

Start here if you want the shortest high-signal review path:

1. [plan.md](/mnt/c/Users/heman/Desktop/code/Agentic_frame_work/OpenAgentBench/openagentbench/agent_retrieval/plan.md)
   Read the intended architecture, invariants, and scoring model.
2. [orchestrator.py](/mnt/c/Users/heman/Desktop/code/Agentic_frame_work/OpenAgentBench/openagentbench/agent_retrieval/orchestrator.py)
   Review the actual retrieve pipeline.
3. [scoring.py](/mnt/c/Users/heman/Desktop/code/Agentic_frame_work/OpenAgentBench/openagentbench/agent_retrieval/scoring.py)
   Review query classification, modality inference, MIMO topology, loop strategy, and scoring helpers.
4. [routing.py](/mnt/c/Users/heman/Desktop/code/Agentic_frame_work/OpenAgentBench/openagentbench/agent_retrieval/routing.py)
   Review selective-purpose model routing for embedding, generation, multimodal, reranking, planner, executor, and critic roles.
5. [providers.py](/mnt/c/Users/heman/Desktop/code/Agentic_frame_work/OpenAgentBench/openagentbench/agent_retrieval/providers.py) and [endpoint_compat.py](/mnt/c/Users/heman/Desktop/code/Agentic_frame_work/OpenAgentBench/openagentbench/agent_retrieval/endpoint_compat.py)
   Review the OpenAI/vLLM/Gemini boundary assumptions and payload builders.
6. [queries.py](/mnt/c/Users/heman/Desktop/code/Agentic_frame_work/OpenAgentBench/openagentbench/agent_retrieval/queries.py) and [sql/001_agent_retrieval_schema.sql](/mnt/c/Users/heman/Desktop/code/Agentic_frame_work/OpenAgentBench/openagentbench/agent_retrieval/sql/001_agent_retrieval_schema.sql)
   Review the storage and SQL APIs.

## File Guide

- [__init__.py](/mnt/c/Users/heman/Desktop/code/Agentic_frame_work/OpenAgentBench/openagentbench/agent_retrieval/__init__.py)
  Public export surface for the submodule.
- [enums.py](/mnt/c/Users/heman/Desktop/code/Agentic_frame_work/OpenAgentBench/openagentbench/agent_retrieval/enums.py)
  Retrieval, modality, topology, reasoning, and loop enums.
- [models.py](/mnt/c/Users/heman/Desktop/code/Agentic_frame_work/OpenAgentBench/openagentbench/agent_retrieval/models.py)
  Typed records for session/history/memory rows, fused candidates, provenance, budgets, and selected-model plans.
- [repository.py](/mnt/c/Users/heman/Desktop/code/Agentic_frame_work/OpenAgentBench/openagentbench/agent_retrieval/repository.py)
  Repository protocol plus the in-memory reference repository used by tests.
- [runtime.py](/mnt/c/Users/heman/Desktop/code/Agentic_frame_work/OpenAgentBench/openagentbench/agent_retrieval/runtime.py)
  Runtime helpers to locate plan and schema assets.
- [gemini_smoke.py](/mnt/c/Users/heman/Desktop/code/Agentic_frame_work/OpenAgentBench/openagentbench/agent_retrieval/gemini_smoke.py)
  Live compatibility smoke path for Gemini plus optional PostgreSQL persistence.

## Public API Areas

The submodule currently exposes reviewable APIs in five groups:

- Retrieval engine APIs
  `HybridRetrievalEngine`, `plan_models`, repository contracts, ranking config, quality config
- Classification and routing APIs
  `classify_query`, modality inference, loop strategy inference, signal topology inference, `ModelRouter`
- Provider and endpoint APIs
  OpenAI-compatible embedding/text providers, Gemini text model, endpoint payload builders, compatibility report builders
- SQL template APIs
  insert, load, exact retrieval, semantic retrieval, and memory-touch builders
- Runtime and smoke APIs
  asset readers plus `run_gemini_smoke`

## Tests

The retrieval module is covered by:

- [test_agent_retrieval.py](/mnt/c/Users/heman/Desktop/code/Agentic_frame_work/OpenAgentBench/tests/test_agent_retrieval.py)
  In-memory orchestration, isolation, suppression, routing, provider adapters, and token-budget behavior.
- [test_agent_retrieval_compat.py](/mnt/c/Users/heman/Desktop/code/Agentic_frame_work/OpenAgentBench/tests/test_agent_retrieval_compat.py)
  Compatibility and query-shape classification checks.
- [test_agent_retrieval_api.py](/mnt/c/Users/heman/Desktop/code/Agentic_frame_work/OpenAgentBench/tests/test_agent_retrieval_api.py)
  API-surface tests for payload builders, SQL builders, runtime asset readers, model-plan API, success experiments, and Gemini smoke failure contract.

## Endpoint Matrix Coverage

The retrieval compatibility layer now includes executable payload-shape builders and tests for:

- OpenAI `responses`
- OpenAI `chat/completions`
- OpenAI `embeddings`
- OpenAI realtime session update payloads
- OpenAI audio speech
- OpenAI audio transcriptions
- OpenAI audio translations
- OpenAI image generations
- OpenAI image edits
- OpenAI video generation
- Gemini `generateContent`
- Gemini `countTokens`

The retrieval module still treats these as compatibility payload builders and validation scaffolding. It does not attempt to fully execute every upstream modality endpoint inside the retrieval engine itself.

## Success Experiments

The API test suite includes a scenario matrix that makes pass conditions easy to audit:

- `normal-single-model`: classifier remains single-model and the router selects a text-generation primary model
- `multimodal-mimo`: classifier expands to multimodal MIMO handling and the router binds a multimodal-capable primary model
- `thinking-reasoning`: classifier raises reasoning effort and the router binds a thinking-capable model
- `dual-model-reranking`: classifier enters dual-model execution and the router binds embedding plus reranking roles
- `agentic-loop`: classifier enters multi-model tool-aware loop mode and the router binds planner, executor, and critic roles

## Run

Run retrieval tests with `unittest`:

```bash
PYTHONPATH=OpenAgentBench python3 -m unittest \
  discover -s OpenAgentBench/tests -p 'test_agent_retrieval*.py'
```

Run the live Gemini smoke:

```bash
PYTHONPATH=OpenAgentBench python3 -m openagentbench.agent_retrieval.gemini_smoke
```

## Environment

- `GEMINI_API_KEY` is required for the live Gemini smoke.
- `AGENT_RETRIEVAL_DATABASE_URL`, `TEST_DATABASE_URL`, or `DATABASE_URL` enables PostgreSQL persistence in the live smoke path.
- The smoke path defaults to `gemini-2.5-flash-lite` unless another model is provided.

## Review Notes

- The OpenAI-compatible provider layer is intentionally client-injected so the same request shape can target OpenAI-hosted endpoints or vLLM OpenAI-compatible serving.
- The current module is a reference implementation and validation scaffold, not a proof of production p99 claims by itself.
- The highest-value review feedback will usually come from challenging:
  query classification heuristics,
  fusion and reranking choices,
  negative-evidence handling,
  and selective model-routing rules.
