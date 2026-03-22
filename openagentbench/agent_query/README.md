# Agent Query

`agent_query` is the deterministic query-understanding layer for OpenAgentBench.

It sits between ingress parsing and retrieval/tool execution, and it is responsible for:

- context assembly across session history, memory, and tool manifests
- intent classification and ambiguity analysis
- cognitive-complexity estimation and decomposition
- tool-aware routing compatible with the existing `agent_data`, `agent_memory`, `agent_retrieval`, and `agent_tools` modules
- OpenAI-compatible and vLLM-compatible payload generation for structured query planning

The implementation is designed to stay deterministic under test while remaining provider-compatible for production wiring.
