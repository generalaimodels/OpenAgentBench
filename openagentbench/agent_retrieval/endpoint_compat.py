"""Endpoint compatibility helpers for OpenAI-style and Gemini-style retrieval clients."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Sequence

from openagentbench.agent_data import OPENAI_HTTP_ENDPOINTS


def _filtered_context(context: Sequence[str]) -> list[str]:
    return [item.strip() for item in context if item.strip()]


def build_openai_responses_request(
    *,
    model: str,
    system_prompt: str,
    user_input: str,
    context: Sequence[str] = (),
    max_output_tokens: int = 256,
    temperature: float = 0.0,
    reasoning_effort: str | None = None,
) -> dict[str, Any]:
    user_content: list[dict[str, str]] = []
    filtered_context = _filtered_context(context)
    if filtered_context:
        user_content.append({"type": "input_text", "text": "Context:\n" + "\n".join(filtered_context)})
    user_content.append({"type": "input_text", "text": user_input})

    payload: dict[str, Any] = {
        "model": model,
        "input": [
            {"role": "system", "content": [{"type": "input_text", "text": system_prompt}]},
            {"role": "user", "content": user_content},
        ],
        "max_output_tokens": max_output_tokens,
    }
    if reasoning_effort is not None and reasoning_effort != "none":
        payload["reasoning"] = {"effort": reasoning_effort}
    return payload


def build_openai_chat_completions_request(
    *,
    model: str,
    system_prompt: str,
    user_input: str,
    context: Sequence[str] = (),
    max_tokens: int = 256,
    temperature: float = 0.0,
) -> dict[str, Any]:
    messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
    filtered_context = _filtered_context(context)
    if filtered_context:
        messages.append({"role": "system", "content": "Context:\n" + "\n".join(filtered_context)})
    messages.append({"role": "user", "content": user_input})
    return {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }


def build_openai_completions_request(
    *,
    model: str,
    prompt: str,
    max_tokens: int = 256,
    temperature: float = 0.0,
) -> dict[str, Any]:
    return {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }


def build_openai_embeddings_request(
    *,
    model: str,
    inputs: Sequence[str],
    dimensions: int | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {"model": model, "input": list(inputs)}
    if dimensions is not None:
        payload["dimensions"] = dimensions
    return payload


def build_openai_moderations_request(
    *,
    model: str,
    inputs: Sequence[str] | str,
) -> dict[str, Any]:
    payload_input: str | list[str]
    if isinstance(inputs, str):
        payload_input = inputs
    else:
        payload_input = list(inputs)
    return {
        "model": model,
        "input": payload_input,
    }


def build_openai_realtime_session_request(
    *,
    model: str,
    modalities: Sequence[str] = ("text",),
    instructions: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "type": "session.update",
        "session": {
            "model": model,
            "modalities": list(modalities),
        },
    }
    if instructions:
        payload["session"]["instructions"] = instructions
    return payload


def build_openai_audio_speech_request(
    *,
    model: str,
    input_text: str,
    voice: str = "alloy",
    audio_format: str = "mp3",
) -> dict[str, Any]:
    return {
        "model": model,
        "input": input_text,
        "voice": voice,
        "format": audio_format,
    }


def build_openai_audio_transcription_request(
    *,
    model: str,
    file_name: str,
    response_format: str = "json",
) -> dict[str, Any]:
    return {
        "model": model,
        "file": file_name,
        "response_format": response_format,
    }


def build_openai_audio_translation_request(
    *,
    model: str,
    file_name: str,
    response_format: str = "json",
) -> dict[str, Any]:
    return {
        "model": model,
        "file": file_name,
        "response_format": response_format,
    }


def build_openai_image_generation_request(
    *,
    model: str,
    prompt: str,
    size: str = "1024x1024",
) -> dict[str, Any]:
    return {
        "model": model,
        "prompt": prompt,
        "size": size,
    }


def build_openai_image_edit_request(
    *,
    model: str,
    prompt: str,
    image_name: str,
    mask_name: str | None = None,
    size: str = "1024x1024",
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "image": image_name,
        "size": size,
    }
    if mask_name is not None:
        payload["mask"] = mask_name
    return payload


def build_openai_video_generation_request(
    *,
    model: str,
    prompt: str,
    image_name: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "prompt": prompt,
    }
    if image_name is not None:
        payload["image"] = image_name
    return payload


_PATH_PARAM_EXAMPLES = {
    "assistant_id": "asst_123",
    "batch_id": "batch_123",
    "call_id": "call_123",
    "completion_id": "chatcmpl_123",
    "consent_id": "consent_123",
    "container_id": "container_123",
    "conversation_id": "conv_123",
    "eval_id": "eval_123",
    "file_id": "file_123",
    "fine_tuned_model_checkpoint": "ftckpt_123",
    "fine_tuning_job_id": "ftjob_123",
    "item_id": "item_123",
    "message_id": "msg_123",
    "model": "gpt-4.1-mini",
    "permission_id": "perm_123",
    "response_id": "resp_123",
    "run_id": "run_123",
    "step_id": "step_123",
    "thread_id": "thread_123",
    "upload_id": "upload_123",
    "vector_store_id": "vs_123",
    "video_id": "video_123",
    "character_id": "char_123",
}


def _path_params_for_template(path: str) -> dict[str, str]:
    params: dict[str, str] = {}
    for field_name in re.findall(r"{([^}]+)}", path):
        params[field_name] = _PATH_PARAM_EXAMPLES.get(field_name, f"example_{field_name}")
    return params


def _rest_request_example(
    method: str,
    path: str,
    *,
    json_body: dict[str, Any] | None = None,
    multipart: dict[str, Any] | None = None,
    query: dict[str, Any] | None = None,
) -> dict[str, Any]:
    example: dict[str, Any] = {
        "method": method,
        "path": path,
    }
    path_params = _path_params_for_template(path)
    if path_params:
        example["path_params"] = path_params
    if json_body is not None:
        example["json"] = json_body
    if multipart is not None:
        example["multipart"] = multipart
    if query is not None:
        example["query"] = query
    return example


def build_openai_http_endpoint_examples() -> dict[str, dict[str, Any]]:
    responses_create = build_openai_responses_request(
        model="gpt-5-mini",
        system_prompt="You are a compatibility probe.",
        user_input="Return a concise status update.",
        context=("Trace endpoint coverage.",),
        reasoning_effort="medium",
    )
    chat_create = build_openai_chat_completions_request(
        model="gpt-4.1-mini",
        system_prompt="You are a compatibility probe.",
        user_input="Return a concise status update.",
        context=("Trace endpoint coverage.",),
    )
    realtime_session = build_openai_realtime_session_request(
        model="gpt-realtime-mini",
        modalities=("text", "audio"),
        instructions="Stay concise.",
    )
    image_generation = build_openai_image_generation_request(
        model="gpt-image-1",
        prompt="Generate an endpoint matrix thumbnail.",
    )
    image_edit = build_openai_image_edit_request(
        model="gpt-image-1",
        prompt="Highlight the primary path.",
        image_name="input.png",
        mask_name="mask.png",
    )
    video_generation = build_openai_video_generation_request(
        model="sora-2",
        prompt="Animate endpoint coverage.",
        image_name="input.png",
    )

    examples: dict[str, dict[str, Any]] = {
        "POST /v1/responses": _rest_request_example("POST", "/v1/responses", json_body=responses_create),
        "POST /v1/responses/input_tokens": _rest_request_example(
            "POST",
            "/v1/responses/input_tokens",
            json_body={
                "model": "gpt-5-mini",
                "input": responses_create["input"],
                "tools": [],
            },
        ),
        "POST /v1/responses/compact": _rest_request_example(
            "POST",
            "/v1/responses/compact",
            json_body={"response_id": "resp_123", "summary": "Condense the prior exchange."},
        ),
        "POST /v1/conversations": _rest_request_example(
            "POST",
            "/v1/conversations",
            json_body={"metadata": {"topic": "compatibility-audit"}},
        ),
        "POST /v1/conversations/{conversation_id}": _rest_request_example(
            "POST",
            "/v1/conversations/{conversation_id}",
            json_body={"metadata": {"status": "verified"}},
        ),
        "POST /v1/conversations/{conversation_id}/items": _rest_request_example(
            "POST",
            "/v1/conversations/{conversation_id}/items",
            json_body={
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "What changed in the endpoint matrix?"}],
            },
        ),
        "POST /v1/chat/completions": _rest_request_example("POST", "/v1/chat/completions", json_body=chat_create),
        "POST /v1/chat/completions/{completion_id}": _rest_request_example(
            "POST",
            "/v1/chat/completions/{completion_id}",
            json_body={"metadata": {"label": "compatibility-follow-up"}},
        ),
        "POST /v1/realtime/client_secrets": _rest_request_example(
            "POST",
            "/v1/realtime/client_secrets",
            json_body={"session": realtime_session["session"]},
        ),
        "POST /v1/realtime/calls/{call_id}/accept": _rest_request_example(
            "POST",
            "/v1/realtime/calls/{call_id}/accept",
            json_body={"instructions": "Continue the verified support call."},
        ),
        "POST /v1/realtime/calls/{call_id}/hangup": _rest_request_example(
            "POST",
            "/v1/realtime/calls/{call_id}/hangup",
            json_body={"reason": "verification-complete"},
        ),
        "POST /v1/realtime/calls/{call_id}/refer": _rest_request_example(
            "POST",
            "/v1/realtime/calls/{call_id}/refer",
            json_body={"destination": "sip:triage@example.com"},
        ),
        "POST /v1/realtime/calls/{call_id}/reject": _rest_request_example(
            "POST",
            "/v1/realtime/calls/{call_id}/reject",
            json_body={"status_code": 603},
        ),
        "POST /v1/realtime/sessions": _rest_request_example(
            "POST",
            "/v1/realtime/sessions",
            json_body=realtime_session["session"],
        ),
        "POST /v1/realtime/transcription_sessions": _rest_request_example(
            "POST",
            "/v1/realtime/transcription_sessions",
            json_body={
                "input_audio_format": "pcm16",
                "input_audio_transcription": {"model": "gpt-4o-mini-transcribe"},
            },
        ),
        "POST /v1/audio/transcriptions": _rest_request_example(
            "POST",
            "/v1/audio/transcriptions",
            multipart=build_openai_audio_transcription_request(
                model="gpt-4o-mini-transcribe",
                file_name="sample.wav",
            ),
        ),
        "POST /v1/audio/translations": _rest_request_example(
            "POST",
            "/v1/audio/translations",
            multipart=build_openai_audio_translation_request(
                model="gpt-4o-mini-transcribe",
                file_name="sample.wav",
            ),
        ),
        "POST /v1/audio/speech": _rest_request_example(
            "POST",
            "/v1/audio/speech",
            json_body=build_openai_audio_speech_request(
                model="gpt-4o-mini-tts",
                input_text="Read the compatibility result aloud.",
            ),
        ),
        "POST /v1/audio/voices": _rest_request_example(
            "POST",
            "/v1/audio/voices",
            multipart={
                "name": "SupportVoice",
                "language": "en-US",
                "recording": "voice_sample.wav",
            },
        ),
        "POST /v1/audio/voice_consents": _rest_request_example(
            "POST",
            "/v1/audio/voice_consents",
            multipart={
                "name": "John Doe",
                "language": "en-US",
                "recording": "consent.wav",
            },
        ),
        "POST /v1/audio/voice_consents/{consent_id}": _rest_request_example(
            "POST",
            "/v1/audio/voice_consents/{consent_id}",
            json_body={"status": "approved"},
        ),
        "POST /v1/videos": _rest_request_example("POST", "/v1/videos", json_body=video_generation),
        "POST /v1/videos/edits": _rest_request_example(
            "POST",
            "/v1/videos/edits",
            json_body={"video_id": "video_123", "prompt": "Tighten the pacing."},
        ),
        "POST /v1/videos/extensions": _rest_request_example(
            "POST",
            "/v1/videos/extensions",
            json_body={"video_id": "video_123", "prompt": "Add a five second outro."},
        ),
        "POST /v1/videos/{video_id}/remix": _rest_request_example(
            "POST",
            "/v1/videos/{video_id}/remix",
            json_body={"prompt": "Remix with a blueprint visual style."},
        ),
        "POST /v1/videos/characters": _rest_request_example(
            "POST",
            "/v1/videos/characters",
            json_body={"name": "Navigator", "image": "character.png"},
        ),
        "POST /v1/images/generations": _rest_request_example("POST", "/v1/images/generations", json_body=image_generation),
        "POST /v1/images/edits": _rest_request_example("POST", "/v1/images/edits", json_body=image_edit),
        "POST /v1/images/variations": _rest_request_example(
            "POST",
            "/v1/images/variations",
            multipart={"model": "gpt-image-1", "image": "input.png"},
        ),
        "POST /v1/embeddings": _rest_request_example(
            "POST",
            "/v1/embeddings",
            json_body=build_openai_embeddings_request(
                model="text-embedding-3-small",
                inputs=("endpoint coverage", "compatibility matrix"),
                dimensions=256,
            ),
        ),
        "POST /v1/moderations": _rest_request_example(
            "POST",
            "/v1/moderations",
            json_body=build_openai_moderations_request(
                model="omni-moderation-latest",
                inputs=("Validate this output.",),
            ),
        ),
        "POST /v1/files": _rest_request_example(
            "POST",
            "/v1/files",
            multipart={"purpose": "assistants", "file": "dataset.jsonl"},
        ),
        "POST /v1/uploads": _rest_request_example(
            "POST",
            "/v1/uploads",
            json_body={"filename": "dataset.jsonl", "bytes": 1024, "purpose": "fine-tune"},
        ),
        "POST /v1/uploads/{upload_id}/complete": _rest_request_example(
            "POST",
            "/v1/uploads/{upload_id}/complete",
            json_body={"part_ids": ["part_1", "part_2"]},
        ),
        "POST /v1/uploads/{upload_id}/parts": _rest_request_example(
            "POST",
            "/v1/uploads/{upload_id}/parts",
            multipart={"data": "chunk-0001.bin"},
        ),
        "POST /v1/batches": _rest_request_example(
            "POST",
            "/v1/batches",
            json_body={
                "input_file_id": "file_123",
                "endpoint": "/v1/responses",
                "completion_window": "24h",
            },
        ),
        "POST /v1/fine_tuning/jobs": _rest_request_example(
            "POST",
            "/v1/fine_tuning/jobs",
            json_body={"model": "gpt-4.1-mini", "training_file": "file_123"},
        ),
        "POST /v1/fine_tuning/checkpoints/{fine_tuned_model_checkpoint}/permissions": _rest_request_example(
            "POST",
            "/v1/fine_tuning/checkpoints/{fine_tuned_model_checkpoint}/permissions",
            json_body={"project_id": "proj_123"},
        ),
        "POST /v1/fine_tuning/alpha/graders/run": _rest_request_example(
            "POST",
            "/v1/fine_tuning/alpha/graders/run",
            json_body={"grader": "safety", "sample": {"input": "Hello"}},
        ),
        "POST /v1/fine_tuning/alpha/graders/validate": _rest_request_example(
            "POST",
            "/v1/fine_tuning/alpha/graders/validate",
            json_body={"grader": "safety", "sample": {"input": "Hello"}},
        ),
        "POST /v1/vector_stores": _rest_request_example(
            "POST",
            "/v1/vector_stores",
            json_body={"name": "compatibility-index"},
        ),
        "POST /v1/vector_stores/{vector_store_id}": _rest_request_example(
            "POST",
            "/v1/vector_stores/{vector_store_id}",
            json_body={"name": "compatibility-index-v2"},
        ),
        "POST /v1/vector_stores/{vector_store_id}/search": _rest_request_example(
            "POST",
            "/v1/vector_stores/{vector_store_id}/search",
            json_body={"query": "responses api", "max_num_results": 5},
        ),
        "POST /v1/vector_stores/{vector_store_id}/files": _rest_request_example(
            "POST",
            "/v1/vector_stores/{vector_store_id}/files",
            json_body={"file_id": "file_123"},
        ),
        "POST /v1/vector_stores/{vector_store_id}/files/{file_id}": _rest_request_example(
            "POST",
            "/v1/vector_stores/{vector_store_id}/files/{file_id}",
            json_body={"attributes": {"label": "verified"}},
        ),
        "POST /v1/vector_stores/{vector_store_id}/file_batches": _rest_request_example(
            "POST",
            "/v1/vector_stores/{vector_store_id}/file_batches",
            json_body={"file_ids": ["file_123", "file_456"]},
        ),
        "POST /v1/vector_stores/{vector_store_id}/file_batches/{batch_id}/cancel": _rest_request_example(
            "POST",
            "/v1/vector_stores/{vector_store_id}/file_batches/{batch_id}/cancel",
        ),
        "POST /v1/evals": _rest_request_example(
            "POST",
            "/v1/evals",
            json_body={"name": "compatibility-eval", "metadata": {"suite": "endpoint-audit"}},
        ),
        "POST /v1/evals/{eval_id}": _rest_request_example(
            "POST",
            "/v1/evals/{eval_id}",
            json_body={"metadata": {"status": "active"}},
        ),
        "POST /v1/evals/{eval_id}/runs": _rest_request_example(
            "POST",
            "/v1/evals/{eval_id}/runs",
            json_body={"model": "gpt-5-mini", "data_source": {"type": "file", "id": "file_123"}},
        ),
        "POST /v1/evals/{eval_id}/runs/{run_id}": _rest_request_example(
            "POST",
            "/v1/evals/{eval_id}/runs/{run_id}",
            json_body={"metadata": {"label": "rerun"}},
        ),
        "POST /v1/containers": _rest_request_example(
            "POST",
            "/v1/containers",
            json_body={"name": "compatibility-sandbox"},
        ),
        "POST /v1/containers/{container_id}/files": _rest_request_example(
            "POST",
            "/v1/containers/{container_id}/files",
            multipart={"file": "artifact.txt"},
        ),
        "POST /v1/assistants": _rest_request_example(
            "POST",
            "/v1/assistants",
            json_body={"model": "gpt-4.1-mini", "name": "Compatibility Assistant"},
        ),
        "POST /v1/assistants/{assistant_id}": _rest_request_example(
            "POST",
            "/v1/assistants/{assistant_id}",
            json_body={"name": "Compatibility Assistant v2"},
        ),
        "POST /v1/threads": _rest_request_example(
            "POST",
            "/v1/threads",
            json_body={"messages": [{"role": "user", "content": "Audit the endpoint matrix."}]},
        ),
        "POST /v1/threads/runs": _rest_request_example(
            "POST",
            "/v1/threads/runs",
            json_body={"assistant_id": "asst_123", "thread": {"messages": [{"role": "user", "content": "Audit the endpoint matrix."}]}},
        ),
        "POST /v1/threads/{thread_id}": _rest_request_example(
            "POST",
            "/v1/threads/{thread_id}",
            json_body={"metadata": {"status": "active"}},
        ),
        "POST /v1/threads/{thread_id}/messages": _rest_request_example(
            "POST",
            "/v1/threads/{thread_id}/messages",
            json_body={"role": "user", "content": "Continue the audit."},
        ),
        "POST /v1/threads/{thread_id}/messages/{message_id}": _rest_request_example(
            "POST",
            "/v1/threads/{thread_id}/messages/{message_id}",
            json_body={"metadata": {"edited": True}},
        ),
        "POST /v1/threads/{thread_id}/runs": _rest_request_example(
            "POST",
            "/v1/threads/{thread_id}/runs",
            json_body={"assistant_id": "asst_123"},
        ),
        "POST /v1/threads/{thread_id}/runs/{run_id}": _rest_request_example(
            "POST",
            "/v1/threads/{thread_id}/runs/{run_id}",
            json_body={"metadata": {"priority": "high"}},
        ),
        "POST /v1/threads/{thread_id}/runs/{run_id}/submit_tool_outputs": _rest_request_example(
            "POST",
            "/v1/threads/{thread_id}/runs/{run_id}/submit_tool_outputs",
            json_body={"tool_outputs": [{"tool_call_id": "call_1", "output": "verified"}]},
        ),
    }

    for method_path in (
        "GET /v1/responses/{response_id}",
        "DELETE /v1/responses/{response_id}",
        "GET /v1/responses/{response_id}/input_items",
        "POST /v1/responses/{response_id}/cancel",
        "GET /v1/conversations/{conversation_id}",
        "DELETE /v1/conversations/{conversation_id}",
        "GET /v1/conversations/{conversation_id}/items",
        "GET /v1/conversations/{conversation_id}/items/{item_id}",
        "DELETE /v1/conversations/{conversation_id}/items/{item_id}",
        "GET /v1/chat/completions",
        "GET /v1/chat/completions/{completion_id}",
        "DELETE /v1/chat/completions/{completion_id}",
        "GET /v1/chat/completions/{completion_id}/messages",
        "GET /v1/audio/voice_consents",
        "GET /v1/audio/voice_consents/{consent_id}",
        "DELETE /v1/audio/voice_consents/{consent_id}",
        "GET /v1/videos",
        "GET /v1/videos/{video_id}",
        "DELETE /v1/videos/{video_id}",
        "GET /v1/videos/{video_id}/content",
        "GET /v1/videos/characters/{character_id}",
        "GET /v1/files",
        "GET /v1/files/{file_id}",
        "DELETE /v1/files/{file_id}",
        "GET /v1/files/{file_id}/content",
        "POST /v1/uploads/{upload_id}/cancel",
        "GET /v1/batches",
        "GET /v1/batches/{batch_id}",
        "POST /v1/batches/{batch_id}/cancel",
        "GET /v1/models",
        "GET /v1/models/{model}",
        "DELETE /v1/models/{model}",
        "GET /v1/fine_tuning/jobs",
        "GET /v1/fine_tuning/jobs/{fine_tuning_job_id}",
        "GET /v1/fine_tuning/jobs/{fine_tuning_job_id}/events",
        "POST /v1/fine_tuning/jobs/{fine_tuning_job_id}/cancel",
        "POST /v1/fine_tuning/jobs/{fine_tuning_job_id}/pause",
        "POST /v1/fine_tuning/jobs/{fine_tuning_job_id}/resume",
        "GET /v1/fine_tuning/checkpoints/{fine_tuned_model_checkpoint}/permissions",
        "DELETE /v1/fine_tuning/checkpoints/{fine_tuned_model_checkpoint}/permissions/{permission_id}",
        "GET /v1/vector_stores",
        "GET /v1/vector_stores/{vector_store_id}",
        "DELETE /v1/vector_stores/{vector_store_id}",
        "GET /v1/vector_stores/{vector_store_id}/files",
        "GET /v1/vector_stores/{vector_store_id}/files/{file_id}",
        "DELETE /v1/vector_stores/{vector_store_id}/files/{file_id}",
        "GET /v1/vector_stores/{vector_store_id}/file_batches/{batch_id}",
        "GET /v1/vector_stores/{vector_store_id}/file_batches/{batch_id}/files",
        "GET /v1/evals",
        "GET /v1/evals/{eval_id}",
        "DELETE /v1/evals/{eval_id}",
        "GET /v1/evals/{eval_id}/runs",
        "GET /v1/evals/{eval_id}/runs/{run_id}",
        "DELETE /v1/evals/{eval_id}/runs/{run_id}",
        "GET /v1/containers",
        "GET /v1/containers/{container_id}",
        "DELETE /v1/containers/{container_id}",
        "GET /v1/containers/{container_id}/files",
        "GET /v1/containers/{container_id}/files/{file_id}",
        "DELETE /v1/containers/{container_id}/files/{file_id}",
        "GET /v1/containers/{container_id}/files/{file_id}/content",
        "GET /v1/assistants",
        "GET /v1/assistants/{assistant_id}",
        "DELETE /v1/assistants/{assistant_id}",
        "GET /v1/threads/{thread_id}",
        "DELETE /v1/threads/{thread_id}",
        "GET /v1/threads/{thread_id}/messages",
        "GET /v1/threads/{thread_id}/messages/{message_id}",
        "DELETE /v1/threads/{thread_id}/messages/{message_id}",
        "GET /v1/threads/{thread_id}/runs",
        "GET /v1/threads/{thread_id}/runs/{run_id}",
        "POST /v1/threads/{thread_id}/runs/{run_id}/cancel",
        "GET /v1/threads/{thread_id}/runs/{run_id}/steps",
        "GET /v1/threads/{thread_id}/runs/{run_id}/steps/{step_id}",
    ):
        method, path = method_path.split(" ", 1)
        examples[method_path] = _rest_request_example(method, path)

    missing = [endpoint for endpoint in OPENAI_HTTP_ENDPOINTS if endpoint not in examples]
    if missing:
        raise AssertionError(f"missing OpenAI HTTP endpoint examples: {missing}")
    return examples


def build_vllm_responses_request(
    *,
    model: str,
    system_prompt: str,
    user_input: str,
    context: Sequence[str] = (),
    max_output_tokens: int = 256,
    temperature: float = 0.0,
    reasoning_effort: str | None = None,
) -> dict[str, Any]:
    return build_openai_responses_request(
        model=model,
        system_prompt=system_prompt,
        user_input=user_input,
        context=context,
        max_output_tokens=max_output_tokens,
        temperature=temperature,
        reasoning_effort=reasoning_effort,
    )


def build_vllm_chat_completions_request(
    *,
    model: str,
    system_prompt: str,
    user_input: str,
    context: Sequence[str] = (),
    max_tokens: int = 256,
    temperature: float = 0.0,
) -> dict[str, Any]:
    return build_openai_chat_completions_request(
        model=model,
        system_prompt=system_prompt,
        user_input=user_input,
        context=context,
        max_tokens=max_tokens,
        temperature=temperature,
    )


def build_vllm_completions_request(
    *,
    model: str,
    prompt: str,
    max_tokens: int = 256,
    temperature: float = 0.0,
) -> dict[str, Any]:
    return build_openai_completions_request(
        model=model,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
    )


def build_vllm_embeddings_request(
    *,
    model: str,
    inputs: Sequence[str],
    dimensions: int | None = None,
) -> dict[str, Any]:
    return build_openai_embeddings_request(model=model, inputs=inputs, dimensions=dimensions)


def build_vllm_audio_transcription_request(
    *,
    model: str,
    file_name: str,
    response_format: str = "json",
) -> dict[str, Any]:
    return build_openai_audio_transcription_request(
        model=model,
        file_name=file_name,
        response_format=response_format,
    )


def build_vllm_audio_translation_request(
    *,
    model: str,
    file_name: str,
    response_format: str = "json",
) -> dict[str, Any]:
    return build_openai_audio_translation_request(
        model=model,
        file_name=file_name,
        response_format=response_format,
    )


def build_vllm_realtime_session_request(
    *,
    model: str,
    modalities: Sequence[str] = ("audio",),
    instructions: str | None = None,
) -> dict[str, Any]:
    return build_openai_realtime_session_request(
        model=model,
        modalities=modalities,
        instructions=instructions,
    )


def build_vllm_score_request(
    *,
    model: str,
    text_1: str,
    text_2: str | Sequence[str],
    normalize: bool = True,
) -> dict[str, Any]:
    payload_text_2: str | list[str]
    if isinstance(text_2, str):
        payload_text_2 = text_2
    else:
        payload_text_2 = list(text_2)
    return {
        "model": model,
        "text_1": text_1,
        "text_2": payload_text_2,
        "normalize": normalize,
    }


def build_gemini_generate_content_request(
    *,
    system_instruction: str,
    user_text: str,
    temperature: float = 0.0,
    max_output_tokens: int = 128,
) -> dict[str, Any]:
    combined_text = f"System instruction:\n{system_instruction}\n\nUser request:\n{user_text}"
    return {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": combined_text}],
            }
        ],
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_output_tokens,
        },
    }


def build_gemini_count_tokens_request(*, system_instruction: str, user_text: str) -> dict[str, Any]:
    combined_text = f"System instruction:\n{system_instruction}\n\nUser request:\n{user_text}"
    return {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": combined_text}],
            }
        ],
    }


def extract_gemini_text(response_payload: dict[str, Any]) -> str:
    candidates = response_payload.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("Gemini response did not include candidates")
    first_candidate = candidates[0]
    if not isinstance(first_candidate, dict):
        raise ValueError("Gemini candidate payload is malformed")
    content = first_candidate.get("content")
    if not isinstance(content, dict):
        raise ValueError("Gemini candidate content is missing")
    parts = content.get("parts")
    if not isinstance(parts, list):
        raise ValueError("Gemini content parts are missing")
    text_parts = [part.get("text") for part in parts if isinstance(part, dict) and isinstance(part.get("text"), str)]
    if not text_parts:
        raise ValueError("Gemini response did not include text parts")
    return "".join(text_parts).strip()


@dataclass(slots=True, frozen=True)
class EndpointCompatibilityReport:
    openai_http_endpoints: tuple[str, ...]
    openai_http_endpoint_examples: dict[str, dict[str, Any]]
    openai_responses_request: dict[str, Any]
    openai_chat_request: dict[str, Any]
    openai_completions_request: dict[str, Any]
    openai_embeddings_request: dict[str, Any]
    openai_moderations_request: dict[str, Any]
    openai_realtime_request: dict[str, Any]
    openai_audio_speech_request: dict[str, Any]
    openai_audio_transcription_request: dict[str, Any]
    openai_audio_translation_request: dict[str, Any]
    openai_image_generation_request: dict[str, Any]
    openai_image_edit_request: dict[str, Any]
    openai_video_generation_request: dict[str, Any]
    vllm_responses_request: dict[str, Any]
    vllm_chat_request: dict[str, Any]
    vllm_completions_request: dict[str, Any]
    vllm_embeddings_request: dict[str, Any]
    vllm_audio_transcription_request: dict[str, Any]
    vllm_audio_translation_request: dict[str, Any]
    vllm_realtime_request: dict[str, Any]
    vllm_score_request: dict[str, Any]
    gemini_generate_content_request: dict[str, Any]
    gemini_count_tokens_request: dict[str, Any]


def build_endpoint_compatibility_report() -> EndpointCompatibilityReport:
    system_prompt = "You are a retrieval planner."
    user_input = "Summarize the retrieval state in one line."
    context = ("Previous memory: user prefers PostgreSQL.", "Session topic: endpoint compatibility.")
    return EndpointCompatibilityReport(
        openai_http_endpoints=OPENAI_HTTP_ENDPOINTS,
        openai_http_endpoint_examples=build_openai_http_endpoint_examples(),
        openai_responses_request=build_openai_responses_request(
            model="gpt-5-mini",
            system_prompt=system_prompt,
            user_input=user_input,
            context=context,
            reasoning_effort="medium",
        ),
        openai_chat_request=build_openai_chat_completions_request(
            model="gpt-4o-mini",
            system_prompt=system_prompt,
            user_input=user_input,
            context=context,
        ),
        openai_completions_request=build_openai_completions_request(
            model="gpt-3.5-turbo-instruct",
            prompt=user_input,
            max_tokens=64,
        ),
        openai_embeddings_request=build_openai_embeddings_request(
            model="text-embedding-3-small",
            inputs=("endpoint compatibility", "retrieval memory"),
            dimensions=256,
        ),
        openai_moderations_request=build_openai_moderations_request(
            model="omni-moderation-latest",
            inputs=(user_input,),
        ),
        openai_realtime_request=build_openai_realtime_session_request(
            model="gpt-realtime-mini",
            modalities=("text", "audio"),
            instructions=system_prompt,
        ),
        openai_audio_speech_request=build_openai_audio_speech_request(
            model="gpt-4o-mini-tts",
            input_text=user_input,
        ),
        openai_audio_transcription_request=build_openai_audio_transcription_request(
            model="gpt-4o-mini-transcribe",
            file_name="sample.wav",
        ),
        openai_audio_translation_request=build_openai_audio_translation_request(
            model="gpt-4o-mini-transcribe",
            file_name="sample.wav",
        ),
        openai_image_generation_request=build_openai_image_generation_request(
            model="gpt-image-1",
            prompt=user_input,
        ),
        openai_image_edit_request=build_openai_image_edit_request(
            model="gpt-image-1",
            prompt=user_input,
            image_name="sample.png",
            mask_name="mask.png",
        ),
        openai_video_generation_request=build_openai_video_generation_request(
            model="sora-2",
            prompt=user_input,
            image_name="sample.png",
        ),
        vllm_responses_request=build_vllm_responses_request(
            model="meta-llama/Meta-Llama-3.1-8B-Instruct",
            system_prompt=system_prompt,
            user_input=user_input,
            context=context,
            reasoning_effort="medium",
        ),
        vllm_chat_request=build_vllm_chat_completions_request(
            model="meta-llama/Meta-Llama-3.1-8B-Instruct",
            system_prompt=system_prompt,
            user_input=user_input,
            context=context,
        ),
        vllm_completions_request=build_vllm_completions_request(
            model="meta-llama/Meta-Llama-3.1-8B-Instruct",
            prompt=user_input,
            max_tokens=64,
        ),
        vllm_embeddings_request=build_vllm_embeddings_request(
            model="intfloat/e5-small-v2",
            inputs=("endpoint compatibility", "retrieval memory"),
        ),
        vllm_audio_transcription_request=build_vllm_audio_transcription_request(
            model="openai/whisper-small",
            file_name="sample.wav",
        ),
        vllm_audio_translation_request=build_vllm_audio_translation_request(
            model="openai/whisper-small",
            file_name="sample.wav",
        ),
        vllm_realtime_request=build_vllm_realtime_session_request(
            model="openai/whisper-small",
            modalities=("audio",),
            instructions=system_prompt,
        ),
        vllm_score_request=build_vllm_score_request(
            model="BAAI/bge-reranker-base",
            text_1="postgres retrieval",
            text_2=("postgres retrieval memory", "image OCR reranking"),
        ),
        gemini_generate_content_request=build_gemini_generate_content_request(
            system_instruction=system_prompt,
            user_text=user_input,
        ),
        gemini_count_tokens_request=build_gemini_count_tokens_request(
            system_instruction=system_prompt,
            user_text=user_input,
        ),
    )


def assert_endpoint_payload_compatibility(report: EndpointCompatibilityReport | None = None) -> None:
    current = report or build_endpoint_compatibility_report()

    if len(current.openai_http_endpoints) != len(OPENAI_HTTP_ENDPOINTS):
        raise AssertionError("OpenAI HTTP endpoint inventory must preserve the exact documented coverage set")
    if set(current.openai_http_endpoint_examples) != set(current.openai_http_endpoints):
        raise AssertionError("OpenAI HTTP endpoint examples must cover every documented endpoint exactly once")
    responses_input = current.openai_responses_request["input"]
    if not isinstance(responses_input, list) or len(responses_input) != 2:
        raise AssertionError("OpenAI Responses payload must contain system and user items")
    if current.openai_chat_request["messages"][0]["role"] != "system":
        raise AssertionError("OpenAI Chat payload must start with a system message")
    if "prompt" not in current.openai_completions_request:
        raise AssertionError("OpenAI completions payload must include a prompt")
    if "input" not in current.openai_embeddings_request or not current.openai_embeddings_request["input"]:
        raise AssertionError("OpenAI embeddings payload must include at least one input string")
    if "input" not in current.openai_moderations_request:
        raise AssertionError("OpenAI moderations payload must include input")
    if current.openai_realtime_request.get("type") != "session.update":
        raise AssertionError("OpenAI realtime payload must be a session.update event")
    if "input" not in current.openai_audio_speech_request:
        raise AssertionError("OpenAI audio speech payload must include input text")
    if "file" not in current.openai_audio_transcription_request:
        raise AssertionError("OpenAI audio transcription payload must include a file")
    if "file" not in current.openai_audio_translation_request:
        raise AssertionError("OpenAI audio translation payload must include a file")
    if "prompt" not in current.openai_image_generation_request:
        raise AssertionError("OpenAI image generation payload must include a prompt")
    if "image" not in current.openai_image_edit_request:
        raise AssertionError("OpenAI image edit payload must include an image")
    if "prompt" not in current.openai_video_generation_request:
        raise AssertionError("OpenAI video generation payload must include a prompt")
    if current.vllm_responses_request["input"][0]["role"] != "system":
        raise AssertionError("vLLM Responses payload must contain a system role")
    if current.vllm_chat_request["messages"][0]["role"] != "system":
        raise AssertionError("vLLM Chat payload must start with a system message")
    if "prompt" not in current.vllm_completions_request:
        raise AssertionError("vLLM completions payload must include a prompt")
    if "input" not in current.vllm_embeddings_request or not current.vllm_embeddings_request["input"]:
        raise AssertionError("vLLM embeddings payload must include at least one input string")
    if "file" not in current.vllm_audio_transcription_request:
        raise AssertionError("vLLM audio transcription payload must include a file")
    if "file" not in current.vllm_audio_translation_request:
        raise AssertionError("vLLM audio translation payload must include a file")
    if current.vllm_realtime_request.get("type") != "session.update":
        raise AssertionError("vLLM realtime payload must be a session.update event")
    if "text_1" not in current.vllm_score_request or "text_2" not in current.vllm_score_request:
        raise AssertionError("vLLM score payload must include text_1 and text_2")
    if "contents" not in current.gemini_generate_content_request:
        raise AssertionError("Gemini generateContent payload must include contents")
    if "contents" not in current.gemini_count_tokens_request:
        raise AssertionError("Gemini countTokens payload must include contents")


__all__ = [
    "EndpointCompatibilityReport",
    "assert_endpoint_payload_compatibility",
    "build_endpoint_compatibility_report",
    "build_openai_http_endpoint_examples",
    "build_gemini_count_tokens_request",
    "build_gemini_generate_content_request",
    "build_openai_audio_speech_request",
    "build_openai_audio_transcription_request",
    "build_openai_audio_translation_request",
    "build_openai_chat_completions_request",
    "build_openai_completions_request",
    "build_openai_embeddings_request",
    "build_openai_image_edit_request",
    "build_openai_image_generation_request",
    "build_openai_moderations_request",
    "build_openai_realtime_session_request",
    "build_openai_responses_request",
    "build_openai_video_generation_request",
    "build_vllm_audio_transcription_request",
    "build_vllm_audio_translation_request",
    "build_vllm_chat_completions_request",
    "build_vllm_completions_request",
    "build_vllm_embeddings_request",
    "build_vllm_realtime_session_request",
    "build_vllm_responses_request",
    "build_vllm_score_request",
    "extract_gemini_text",
]
