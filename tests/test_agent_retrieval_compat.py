from __future__ import annotations

from pathlib import Path
import unittest

from openagentbench.agent_retrieval.endpoint_compat import (
    assert_endpoint_payload_compatibility,
    build_endpoint_compatibility_report,
)
from openagentbench.agent_retrieval.enums import (
    LoopStrategy,
    ModelExecutionMode,
    ModelRole,
    OutputStream,
    QueryType,
    ReasoningEffort,
    SignalTopology,
)
from openagentbench.agent_retrieval.scoring import classify_query


class EndpointCompatibilityTests(unittest.TestCase):
    def test_endpoint_payloads_match_expected_shapes(self) -> None:
        report = build_endpoint_compatibility_report()
        assert_endpoint_payload_compatibility(report)

        self.assertEqual(report.openai_responses_request["input"][0]["role"], "system")
        self.assertEqual(report.openai_chat_request["messages"][0]["role"], "system")
        self.assertIn("prompt", report.openai_completions_request)
        self.assertEqual(report.openai_embeddings_request["model"], "text-embedding-3-small")
        self.assertIn("input", report.openai_moderations_request)
        self.assertEqual(report.vllm_responses_request["input"][0]["role"], "system")
        self.assertEqual(report.vllm_chat_request["messages"][0]["role"], "system")
        self.assertIn("prompt", report.vllm_completions_request)
        self.assertEqual(report.vllm_realtime_request["type"], "session.update")
        self.assertIn("text_1", report.vllm_score_request)
        self.assertIn("contents", report.gemini_generate_content_request)


class QueryClassificationCoverageTests(unittest.TestCase):
    def test_single_model_normal_query(self) -> None:
        classification = classify_query("Answer this normally in one short paragraph.", "", turn_count=0)
        self.assertEqual(classification.type, QueryType.CONVERSATIONAL)
        self.assertEqual(classification.model_execution_mode, ModelExecutionMode.SINGLE_MODEL)
        self.assertEqual(classification.signal_topology, SignalTopology.SISO)
        self.assertEqual(classification.reasoning_effort, ReasoningEffort.DIRECT)
        self.assertEqual(classification.loop_strategy, LoopStrategy.SINGLE_PASS)

    def test_multimodal_mimo_query(self) -> None:
        classification = classify_query(
            "Use the screenshot and audio clip to produce JSON and a summary.",
            "",
            turn_count=0,
        )
        self.assertEqual(classification.type, QueryType.MULTIMODAL)
        self.assertEqual(classification.signal_topology, SignalTopology.MIMO)
        self.assertIn(OutputStream.VISION_EVIDENCE, classification.output_streams)
        self.assertIn(OutputStream.STRUCTURED_DATA, classification.output_streams)
        self.assertIn(ModelRole.MULTIMODAL, classification.model_roles)

    def test_reasoning_query(self) -> None:
        classification = classify_query("Think step by step and reason carefully about the tradeoffs.", "", turn_count=0)
        self.assertEqual(classification.type, QueryType.REASONING)
        self.assertEqual(classification.reasoning_effort, ReasoningEffort.THINKING)
        self.assertIn(ModelRole.THINKING, classification.model_roles)

    def test_agentic_loop_query(self) -> None:
        classification = classify_query(
            "Plan and execute with tools, retry until deployment is verified.",
            "",
            turn_count=0,
        )
        self.assertEqual(classification.type, QueryType.AGENTIC)
        self.assertEqual(classification.loop_strategy, LoopStrategy.AGENTIC_LOOP)
        self.assertEqual(classification.model_execution_mode, ModelExecutionMode.MULTI_MODEL)
        self.assertIn(OutputStream.TOOL_TRACE, classification.output_streams)
        self.assertIn(ModelRole.AGENTIC_LOOP, classification.model_roles)


class TestLayoutConventionTests(unittest.TestCase):
    def test_agent_retrieval_has_no_package_local_test_modules(self) -> None:
        package_root = Path(__file__).resolve().parents[1] / "openagentbench" / "agent_retrieval"
        disallowed = sorted(
            path.relative_to(package_root).as_posix()
            for path in package_root.rglob("*.py")
            if path.name.startswith("test_")
        )
        self.assertEqual(
            disallowed,
            [],
            "agent_retrieval tests must live under tests/, not inside the package",
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
