"""Minimal OpenAI-compatible chat client used by Agent4Rec.

The original repository pins openai==0.27.8 and hard-codes GPT-3.5.  Keeping
the transport here makes the simulation model configurable without changing
the prompt or response parsing logic.
"""

import json
import os
from urllib.error import HTTPError
from urllib.request import Request, urlopen


def chat_completion(messages, model, base_url, temperature, timeout, max_tokens,
                    reasoning_effort=None):
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")

    body = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_completion_tokens": max_tokens,
    }
    if reasoning_effort:
        body["reasoning_effort"] = reasoning_effort

    request = Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": "Bearer " + api_key,
            "Content-Type": "application/json",
        },
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            result = json.load(response)
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:1000]
        raise RuntimeError(
            f"OpenAI-compatible API returned HTTP {exc.code}: {detail}"
        ) from None

    try:
        content = result["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"Unexpected chat completion response: {result}") from exc
    return content, result.get("usage", {})
