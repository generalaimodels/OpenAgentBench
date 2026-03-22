# OpenAI Python Endpoint Matrix

This module is designed to preserve and replay the OpenAI Python endpoint families that matter for agent evaluation:

| Endpoint | Python Call Shape | Input Modalities | Output Modalities | Source |
|---|---|---|---|---|
| `POST /v1/responses` | `client.responses.create(...)` | text, image, audio, file, mixed content | text, tool calls, structured outputs | https://developers.openai.com/api/docs/models |
| `POST /v1/chat/completions` | `client.chat.completions.create(...)` | text, image, audio, mixed content | text, audio, tool calls, structured outputs | https://developers.openai.com/api/docs/models/gpt-5 |
| `POST /v1/realtime` | realtime session | text, image, audio, mixed content | text, audio, event streams | https://developers.openai.com/api/docs/models/all |
| `POST /v1/audio/speech` | `client.audio.speech.create(...)` | text | audio | https://developers.openai.com/api/docs/models/all |
| `POST /v1/audio/transcriptions` | `client.audio.transcriptions.create(...)` | audio | text | https://developers.openai.com/api/docs/models/all |
| `POST /v1/audio/translations` | `client.audio.translations.create(...)` | audio | text | https://developers.openai.com/api/docs/models/gpt-5 |
| `POST /v1/images/generations` | `client.images.generate(...)` | text, image | image | https://platform.openai.com/docs/guides/images-vision?api-mode=responses&format=file |
| `POST /v1/images/edits` | `client.images.edit(...)` | text, image, mask | image | https://platform.openai.com/docs/guides/images-vision?api-mode=responses&format=file |
| `POST /v1/videos` | `client.videos.generate(...)` | text, image, mixed content | video, audio | https://developers.openai.com/api/docs/models/sora-2 |
| `POST /v1/embeddings` | `client.embeddings.create(...)` | text | embedding vectors | https://developers.openai.com/api/docs/models/text-embedding-3-small |

## Storage Rule

To avoid losing fidelity:

- multimodal message parts are stored in `conversation_history.content_parts`
- textual projections stay in `conversation_history.content` for ranking, compaction, and retrieval
- raw upstream requests and responses are stored in `model_api_calls`
- ordered chunk and token events are stored in `model_stream_events`

## Why This Matters

OpenAgentBench is a testing system. If the platform only stores the final merged assistant message, you cannot:

- replay a streaming failure
- verify tool-call ordering
- inspect partial refusal behavior
- measure token-level latency
- audit multimodal input loss

That is why the database and API contract now treat raw API activity as first-class data, not disposable transport noise.
