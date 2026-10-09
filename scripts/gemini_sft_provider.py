"""Lossless native Gemini rendering for canonical two-tool teacher inputs."""

import base64
import copy
from pathlib import Path
from urllib.parse import quote, urlsplit


def endpoint_for_model(endpoint: str, model: str) -> str:
    parsed = urlsplit(endpoint)
    if parsed.query or parsed.fragment or parsed.username or parsed.password:
        raise ValueError("Gemini endpoint must not contain credentials, query or fragment")
    return endpoint.format(model=quote(model, safe=""))


def render_request(payload: dict) -> dict:
    wire = {"contents": [], "generationConfig": {"maxOutputTokens": payload["max_tokens"]}}
    if payload.get("thinking_level"):
        wire["generationConfig"]["thinkingConfig"] = {"thinkingLevel": payload["thinking_level"]}
    system_parts = []
    for message in payload["messages"]:
        content = message["content"]
        if isinstance(content, str):
            parts = [{"text": content}]
        else:
            parts = []
            for item in content:
                if item["type"] == "text":
                    parts.append({"text": item["text"]})
                elif item["type"] == "image_url":
                    data = Path(item["image_url"]["url"]).read_bytes()
                    parts.append({"inlineData": {"mimeType": "image/png",
                                                 "data": base64.b64encode(data).decode("ascii")}})
                else:
                    raise ValueError("Unsupported Gemini input content type")
        if message["role"] == "system":
            if any("text" not in part for part in parts):
                raise ValueError("Gemini system instruction must contain only text")
            system_parts.extend(parts)
        elif message["role"] in {"user", "assistant"}:
            wire["contents"].append({"role": "model" if message["role"] == "assistant" else "user",
                                     "parts": parts})
        else:
            raise ValueError("Unsupported Gemini input role")
    if system_parts:
        wire["systemInstruction"] = {"parts": system_parts}
    return wire


def normalize_response(raw: dict) -> dict:
    candidates = raw.get("candidates") or []
    candidate = candidates[0] if candidates else {}
    parts = candidate.get("content", {}).get("parts", [])
    message = {"role": "assistant", "content": "".join(
        part.get("text", "") for part in parts if not part.get("thought"))}
    thoughts = "".join(part.get("text", "") for part in parts if part.get("thought"))
    if thoughts:
        message["reasoning_content"] = thoughts
    calls = [copy.deepcopy(part["functionCall"]) for part in parts if "functionCall" in part]
    if calls:
        message["tool_calls"] = calls
    reason = candidate.get("finishReason", "MISSING_CANDIDATE")
    finish = {"STOP": "stop", "MAX_TOKENS": "length"}.get(reason, reason.lower())
    if any(set(part) - {"text", "thought", "thoughtSignature", "functionCall"} for part in parts):
        finish = "unsupported_response_part"
    metadata = copy.deepcopy(raw.get("usageMetadata", {}))
    return {"choices": [{"message": message, "finish_reason": finish}],
            "usage": {"prompt_tokens": metadata.get("promptTokenCount"),
                      "completion_tokens": metadata.get("candidatesTokenCount"),
                      "reasoning_tokens": metadata.get("thoughtsTokenCount"),
                      "total_tokens": metadata.get("totalTokenCount"), "native": metadata},
            "model": raw.get("modelVersion"), "provider_response": copy.deepcopy(raw)}
