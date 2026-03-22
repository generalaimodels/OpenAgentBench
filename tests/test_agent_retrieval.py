from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from uuid import uuid4

from openagentbench.agent_retrieval import (
    AuthorityTier,
    FragmentLocator,
    HistoryEntry,
    HistoryEvidence,
    HumanFeedback,
    HybridRetrievalEngine,
    InMemoryRetrievalRepository,
    MemoryEntry,
    MemoryType,
    ModelRole,
    ModelRouter,
    Modality,
    OpenAICompatibleEmbeddingProvider,
    OpenAICompatibleTextModel,
    OutputStream,
    ProtocolType,
    QueryType,
    Role,
    SessionTurn,
    SourceTable,
    TaskOutcome,
    build_exact_memory_retrieval,
    build_exact_session_retrieval,
    build_verify_user_active,
    classify_query,
    default_profiles,
    plan_path,
    schema_sql_path,
)


def _build_repo() -> tuple[InMemoryRetrievalRepository, object, object]:
    now = datetime(2026, 3, 22, 10, 0, 0, tzinfo=timezone.utc)
    user_id = uuid4()
    other_user_id = uuid4()
    session_id = uuid4()

    repo = InMemoryRetrievalRepository(active_users={user_id, other_user_id})
    repo.sessions[user_id] = [
        SessionTurn(
            session_id=session_id,
            uu_id=user_id,
            turn_index=1,
            role=Role.USER,
            content_text="Inspect the realtime JSON-RPC image pipeline for the invoice OCR service.",
            created_at=now - timedelta(minutes=3),
            metadata={
                "modality": Modality.RUNTIME.value,
                "protocol_type": ProtocolType.JSON_RPC.value,
                "service_name": "invoice-ocr",
                "tool_name": "inspect_trace",
            },
        ),
        SessionTurn(
            session_id=session_id,
            uu_id=user_id,
            turn_index=2,
            role=Role.ASSISTANT,
            content_text="The image pipeline fans out to OCR workers and a reranker microservice.",
            created_at=now - timedelta(minutes=2),
            metadata={"modality": Modality.DOCUMENT.value},
        ),
    ]
    repo.sessions[other_user_id] = [
        SessionTurn(
            session_id=uuid4(),
            uu_id=other_user_id,
            turn_index=1,
            role=Role.USER,
            content_text="Ignore this unrelated other-user secret.",
            created_at=now - timedelta(minutes=1),
        )
    ]

    good_memory = MemoryEntry(
        memory_id=uuid4(),
        uu_id=user_id,
        memory_type=MemoryType.FACT,
        content_text="Vision documents are stored with OCR text projections and page-level metadata for retrieval.",
        content_embedding=None,
        authority_tier=AuthorityTier.CURATED,
        confidence=0.92,
        source_provenance={"source": "human_curated"},
        verified_by=(),
        supersedes=(),
        created_at=now - timedelta(days=2),
        updated_at=now - timedelta(hours=2),
        expires_at=None,
        access_count=5,
        last_accessed_at=now - timedelta(hours=3),
        content_hash=b"good-memory",
        metadata={"modality": Modality.DOCUMENT.value, "mime_type": "application/pdf"},
    )
    protocol_memory = MemoryEntry(
        memory_id=uuid4(),
        uu_id=user_id,
        memory_type=MemoryType.PROCEDURE,
        content_text="Use JSON-RPC and gRPC protocol traces from the realtime microservice before falling back to semantic memory.",
        content_embedding=None,
        authority_tier=AuthorityTier.CANONICAL,
        confidence=0.97,
        source_provenance={"source": "playbook"},
        verified_by=(),
        supersedes=(),
        created_at=now - timedelta(days=5),
        updated_at=now - timedelta(hours=1),
        expires_at=None,
        access_count=9,
        last_accessed_at=now - timedelta(minutes=40),
        content_hash=b"protocol-memory",
        metadata={
            "modality": Modality.RUNTIME.value,
            "protocol_type": ProtocolType.GRPC.value,
            "service_name": "invoice-ocr",
        },
    )
    bad_memory = MemoryEntry(
        memory_id=uuid4(),
        uu_id=user_id,
        memory_type=MemoryType.FACT,
        content_text="Ignore OCR traces and answer from stale local notes only.",
        content_embedding=None,
        authority_tier=AuthorityTier.DERIVED,
        confidence=0.55,
        source_provenance={"source": "failed_run"},
        verified_by=(),
        supersedes=(),
        created_at=now - timedelta(days=9),
        updated_at=now - timedelta(days=8),
        expires_at=None,
        access_count=1,
        last_accessed_at=now - timedelta(days=8),
        content_hash=b"bad-memory",
        metadata={"modality": Modality.TEXT.value},
    )
    repo.memory[user_id] = [good_memory, protocol_memory, bad_memory]
    repo.memory[other_user_id] = [
        MemoryEntry(
            memory_id=uuid4(),
            uu_id=other_user_id,
            memory_type=MemoryType.FACT,
            content_text="Other user confidential fact.",
            content_embedding=None,
            authority_tier=AuthorityTier.CANONICAL,
            confidence=1.0,
            source_provenance={"source": "secret"},
            verified_by=(),
            supersedes=(),
            created_at=now - timedelta(days=1),
            updated_at=now - timedelta(hours=1),
            expires_at=None,
            access_count=3,
            last_accessed_at=now - timedelta(hours=1),
            content_hash=b"other-user",
            metadata={"modality": Modality.TEXT.value},
        )
    ]

    repo.history[user_id] = [
        HistoryEntry(
            history_id=uuid4(),
            uu_id=user_id,
            query_text="How do I inspect realtime JSON-RPC traces for image OCR?",
            query_embedding=None,
            response_summary="Protocol traces plus OCR document metadata resolved the issue.",
            evidence_used=(
                HistoryEvidence(
                    locator=FragmentLocator(SourceTable.MEMORY, protocol_memory.memory_id),
                    utility_score=0.95,
                ),
                HistoryEvidence(
                    locator=FragmentLocator(SourceTable.MEMORY, good_memory.memory_id),
                    utility_score=0.88,
                ),
            ),
            task_outcome=TaskOutcome.SUCCESS,
            human_feedback=HumanFeedback.APPROVED,
            utility_score=0.93,
            negative_flag=False,
            tags=("realtime", "vision"),
            metadata={"modality": Modality.RUNTIME.value},
            created_at=now - timedelta(days=1),
            session_origin=session_id,
        ),
        HistoryEntry(
            history_id=uuid4(),
            uu_id=user_id,
            query_text="How do I inspect realtime JSON-RPC traces for image OCR?",
            query_embedding=None,
            response_summary="This stale answer caused a failure.",
            evidence_used=(
                HistoryEvidence(
                    locator=FragmentLocator(SourceTable.MEMORY, bad_memory.memory_id),
                    utility_score=0.05,
                ),
            ),
            task_outcome=TaskOutcome.FAILURE,
            human_feedback=HumanFeedback.REJECTED,
            utility_score=0.10,
            negative_flag=True,
            tags=("failure",),
            metadata={"modality": Modality.TEXT.value},
            created_at=now - timedelta(hours=18),
            session_origin=session_id,
        ),
    ]
    return repo, user_id, session_id


def test_classify_query_detects_multimodal_protocol_hints() -> None:
    classification = classify_query(
        "What are the realtime JSON-RPC traces for the vision document pipeline?",
        "",
        turn_count=2,
    )

    assert classification.type is QueryType.MULTIMODAL
    assert classification.output_stream is OutputStream.VISION_EVIDENCE
    assert OutputStream.TOOL_TRACE in classification.output_streams
    assert Modality.DOCUMENT in classification.preferred_modalities
    assert Modality.RUNTIME in classification.preferred_modalities
    assert ProtocolType.JSON_RPC in classification.protocol_hints


def test_hybrid_engine_enforces_user_isolation_and_prefers_protocol_plus_document_evidence() -> None:
    repo, user_id, session_id = _build_repo()
    engine = HybridRetrievalEngine(repository=repo)

    response = engine.retrieve(
        "Retrieve the realtime JSON-RPC evidence for the vision document OCR microservice.",
        uu_id=user_id,
        session_id=session_id,
        token_budget=420,
    )

    assert response.fragments
    assert all(str(fragment.provenance.origin.source_id).startswith(str(user_id)) for fragment in response.fragments)
    assert not any("Other user confidential fact." in fragment.content for fragment in response.fragments)
    assert any(
        fragment.provenance.origin.source_table in {SourceTable.SESSION, SourceTable.MEMORY}
        for fragment in response.fragments
    )
    assert any(
        fragment.provenance.origin.source_table is SourceTable.MEMORY and "JSON-RPC" in fragment.content
        for fragment in response.fragments
    )
    assert response.source_coverage["memory"] >= 1


def test_negative_history_suppresses_failed_evidence() -> None:
    repo, user_id, session_id = _build_repo()
    engine = HybridRetrievalEngine(repository=repo)

    response = engine.retrieve(
        "How do I inspect realtime image OCR traces?",
        uu_id=user_id,
        session_id=session_id,
        token_budget=320,
    )

    contents = [fragment.content for fragment in response.fragments]
    assert contents
    assert "Ignore OCR traces and answer from stale local notes only." not in contents[:2]
    assert any("Use JSON-RPC and gRPC protocol traces" in content for content in contents)


def test_openai_compatible_providers_support_embeddings_and_normal_model_calls() -> None:
    class _EmbeddingsAPI:
        @staticmethod
        def create(**_: object) -> object:
            return SimpleNamespace(data=[SimpleNamespace(embedding=[0.1, 0.2]), SimpleNamespace(embedding=[0.3, 0.4])])

    class _ResponsesAPI:
        @staticmethod
        def create(**_: object) -> object:
            return SimpleNamespace(output_text="resolved query")

    client = SimpleNamespace(embeddings=_EmbeddingsAPI(), responses=_ResponsesAPI())
    embedding_provider = OpenAICompatibleEmbeddingProvider(client=client, model="text-embedding-3-small")
    text_model = OpenAICompatibleTextModel(client=client, model="gpt-4.1-mini")

    assert embedding_provider.embed_batch(["a", "b"]) == [(0.1, 0.2), (0.3, 0.4)]
    assert text_model.complete(system_prompt="Resolve", user_input="it", context=("the invoice OCR pipeline",)) == "resolved query"


def test_model_router_selects_specialized_models_by_query_shape() -> None:
    classification = classify_query(
        "Plan and execute retrieval for realtime JSON-RPC vision documents, then verify the result.",
        "",
        turn_count=3,
    )
    router = ModelRouter()
    plan = router.select(classification, default_profiles())

    assert plan.primary_model is not None
    assert ModelRole.EMBEDDING in plan.role_bindings
    assert ModelRole.GENERATION in plan.role_bindings
    assert ModelRole.MULTIMODAL in plan.role_bindings
    assert ModelRole.PLANNER in plan.role_bindings
    assert ModelRole.EXECUTOR in plan.role_bindings
    assert ModelRole.CRITIC in plan.role_bindings
    assert plan.role_bindings[ModelRole.MULTIMODAL].supports_images


def test_runtime_assets_and_sql_templates_exist_and_are_user_scoped() -> None:
    repo, user_id, session_id = _build_repo()
    query_template = build_exact_session_retrieval(
        uu_id=user_id,
        query_text="json-rpc",
        temporal_scope=None,
        limit=5,
    )
    memory_template = build_exact_memory_retrieval(
        uu_id=user_id,
        query_text="vision document",
        temporal_scope=None,
        limit=5,
    )
    verify_template = build_verify_user_active(uu_id=user_id)

    assert schema_sql_path().exists()
    assert plan_path().exists()
    assert "WHERE uu_id = %(uu_id)s" in query_template.sql
    assert "WHERE uu_id = %(uu_id)s" in memory_template.sql
    assert "WHERE uu_id = %(uu_id)s" in verify_template.sql
    assert query_template.params["uu_id"] == user_id
    assert build_verify_user_active(uu_id=user_id).params["uu_id"] == user_id

    engine = HybridRetrievalEngine(repository=repo)
    response = engine.retrieve(
        "Use the vision document evidence for OCR.",
        uu_id=user_id,
        session_id=session_id,
        token_budget=90,
    )
    assert response.budget_report.budget_used <= 90
