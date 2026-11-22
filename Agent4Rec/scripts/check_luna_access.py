"""Fail-fast connectivity check for the configured Agent4Rec chat model."""

from simulation.llm_client import chat_completion


content, usage = chat_completion(
    messages=[{"role": "user", "content": "Reply with exactly: READY"}],
    model="gpt-6-luna",
    base_url="https://api.openai.com/v1",
    temperature=0.0,
    timeout=60,
    max_tokens=16,
    reasoning_effort="none",
)
if "READY" not in content.upper():
    raise RuntimeError(f"Unexpected gpt-6-luna smoke-test response: {content!r}")
print(f"gpt-6-luna smoke test passed; total_tokens={usage.get('total_tokens', 0)}")
