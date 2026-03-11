# Making Qwen3.5-35B-A3B-4bit Work with omlx + Cline

## Model Overview

**Qwen3.5-35B-A3B-4bit** is a Mixture-of-Experts (MoE) reasoning model:
- 35B total parameters, ~3B active per token
- 4-bit quantized, ~20 GB on disk
- Built-in chain-of-thought thinking via `<think>...</think>` blocks
- Uses XML-style tool calling (`<function=name>...</function>`)

The model's thinking capability is great for agent tasks, but it creates several edge cases in streaming APIs that caused [Cline](https://github.com/cline/cline) to fail with **"Invalid API Response"**.

---

## Problems Found and Fixed

### 1. Missing `GET /v1/models/{model_id}` endpoint

**Problem:** Cline verifies model existence by calling `GET /v1/models/<model-id>` before sending any chat request. omlx only implemented `GET /v1/models` (list). Every Cline request failed with HTTP 404 before even reaching the model.

**Fix:** Added a single-model lookup endpoint in `server.py`:

```python
@app.get("/v1/models/{model_id}")
async def get_model(model_id: str, _: bool = Depends(verify_api_key)) -> ModelInfo:
    if _server_state.engine_pool is not None:
        status = _server_state.engine_pool.get_status()
        settings_manager = _server_state.settings_manager
        for m in status["models"]:
            mid = m["id"]
            display_id = mid
            if settings_manager:
                ms = settings_manager.get_settings(mid)
                if ms.model_alias:
                    display_id = ms.model_alias
            if display_id == model_id or mid == model_id:
                return ModelInfo(id=display_id, owned_by="omlx")
    raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")
```

---

### 2. Whitespace content leaking before tool_calls

**Problem:** When the model outputs `\n\n<function=execute_command>...`, the `ToolCallStreamFilter` correctly suppressed the `<function=...>` XML but not the preceding `\n\n`. Cline received a delta with `content="\n\n"` alongside `tool_calls`, which broke its parser.

**Fix:** Skip content deltas that are whitespace-only when the request contains tools:

```python
# Before:
if content_delta:
    yield content chunk

# After:
if content_delta and (not has_tools or content_delta.strip()):
    yield content chunk
```

---

### 3. Content and tool_calls interleaving in streaming mode

**Problem:** In streaming mode, partial content tokens could be emitted before the tool call XML was fully assembled. Cline received mixed content+tool_calls in the same response, which it couldn't parse.

**Fix:** When the request contains tools, always use buffered mode — accumulate everything, then emit once at the end:

```python
stream_content = not has_tools   # stream live only when no tools in request

if has_tools:
    _f = ToolCallStreamFilter(engine.tokenizer)
    if _f.active:
        tool_filter = _f
```

In the buffered emit path, never emit content when tool_calls exist:

```python
if not stream_content:
    if thinking_content:
        yield reasoning_content chunk
    if not tool_calls:           # only emit text content when no tool calls
        emit_text = cleaned_text.strip()
        if emit_text:
            yield content chunk
```

---

### 4. Thinking-only responses (the root cause of most failures)

**Problem:** This was the hardest bug. The omlx scheduler prepends `<think>\n` to the model output when it detects the chat template ends with a `<think>` token. The MoE model then generates its entire response inside the `<think>` block and sometimes never produces content after `</think>`. The `ThinkingParser` routed all tokens to `reasoning_content`, leaving `content` completely empty.

Cline only reads `content` (ignores `reasoning_content`) and received a blank message → **"Invalid API Response"**.

Captured via proxy logging:
```
content(0 chunks), reasoning(470 tokens), finish_reason=stop
```

**Fix:** Track whether any content was emitted during streaming. After the stream ends, if nothing was emitted, extract the thinking text and emit it as regular `content` so clients always get a non-empty response:

```python
# Track flag during stream
content_emitted = False

# ... (set to True whenever a content delta is yielded)

# After stream completes — fallback for thinking-only responses
if not content_emitted and not has_tools and accumulated_text:
    import re as _re
    thinking_text, regular_text = extract_thinking(accumulated_text)
    # Prefer regular content; fall back to thinking text
    raw_fallback = regular_text.strip() or thinking_text.strip() or accumulated_text
    # Strip any residual <think> tags (unclosed tag edge case)
    fallback = _re.sub(r'</?think>\n?', '', raw_fallback).strip()
    if fallback:
        chunk = ChatCompletionChunk(...)
        chunk.choices[0].delta.content = fallback
        yield f"data: {chunk.model_dump_json(exclude_none=True)}\n\n"
```

**Edge case:** When the model never emits `</think>`, `extract_thinking` returns the raw text including the opening `<think>` tag as `regular_content`. The `re.sub` strips those residual tags before emitting to the client.

---

## Benchmark Results (Mac Mini M4, 64 GB)

Running both models simultaneously loaded in omlx's engine pool:

| Test | Qwen3-32B-4bit | Qwen3.5-35B-A3B-4bit |
|------|---------------|----------------------|
| Short prompt | 10.1 tok/s | 33.7 tok/s |
| Long prompt (500 tok prefill) | 11.7 tok/s | 37.5 tok/s |
| Coding task | 9.4 tok/s | 37.0 tok/s |
| Tool calling | 9.7 tok/s | 35.1 tok/s |
| Long generation (400 tok) | 9.1 tok/s | 32.9 tok/s |
| **Average** | **10.0 tok/s** | **35.2 tok/s** |

**Qwen3.5-35B-A3B-4bit is ~3.5× faster** in generation speed despite having nominally more parameters, because MoE only activates ~3B parameters per token vs the dense 32B.

---

## Files Changed

- `omlx/server.py` — all four fixes above
- `benchmark_model.py` — benchmark script for Qwen3.5-35B-A3B-4bit
- `benchmark_compare.py` — side-by-side comparison of both models
