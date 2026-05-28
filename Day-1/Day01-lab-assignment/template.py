"""
Day 1 — LLM API Foundation
AICB-P1: AI Practical Competency Program, Phase 1

Instructions:
    1. Fill in every section marked with TODO.
    2. Do NOT change function signatures.
    3. Copy this file to solution/solution.py when done.
    4. Run: pytest tests/ -v
"""

import os
import time
from typing import Any, Callable

# ---------------------------------------------------------------------------
# Estimated costs per 1K OUTPUT tokens (USD) — update if pricing changes
# ---------------------------------------------------------------------------
COST_PER_1K_OUTPUT_TOKENS = {
    "gpt-4o": 0.010,
    "gpt-4o-mini": 0.0006,
}

OPENAI_MODEL = "gpt-4o"
OPENAI_MINI_MODEL = "gpt-4o-mini"


# ---------------------------------------------------------------------------
# Task 1 — Call GPT-4o
# ---------------------------------------------------------------------------
def call_openai(
    prompt: str,
    model: str = OPENAI_MODEL,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float]:
    """
    Call the OpenAI Chat Completions API and return the response text + latency.

    Args:
        prompt:      The user message to send.
        model:       The OpenAI model to use (default: gpt-4o).
        temperature: Sampling temperature (0.0 – 2.0).
        top_p:       Nucleus sampling threshold.
        max_tokens:  Maximum number of tokens to generate.

    Returns:
        A tuple of (response_text: str, latency_seconds: float).

    Hint:
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    """
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    start = time.perf_counter()
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )
    end = time.perf_counter()

    response_text = (resp.choices[0].message.content or "").strip()
    latency_seconds = max(end - start, 1e-6)
    return response_text, float(latency_seconds)


# ---------------------------------------------------------------------------
# Task 2 — Call GPT-4o-mini
# ---------------------------------------------------------------------------
def call_openai_mini(
    prompt: str,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float]:
    """
    Call the OpenAI Chat Completions API using gpt-4o-mini and return the
    response text + latency.

    Args:
        prompt:      The user message to send.
        temperature: Sampling temperature (0.0 – 2.0).
        top_p:       Nucleus sampling threshold.
        max_tokens:  Maximum number of tokens to generate.

    Returns:
        A tuple of (response_text: str, latency_seconds: float).

    Hint:
        Reuse call_openai() by passing model=OPENAI_MINI_MODEL.
    """
    return call_openai(
        prompt=prompt,
        model=OPENAI_MINI_MODEL,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )


# ---------------------------------------------------------------------------
# Task 3 — Compare GPT-4o vs GPT-4o-mini
# ---------------------------------------------------------------------------
def compare_models(prompt: str) -> dict:
    """
    Call both gpt-4o and gpt-4o-mini with the same prompt and return a
    comparison dictionary.

    Args:
        prompt: The user message to send to both models.

    Returns:
        A dict with keys:
            - "gpt4o_response":      str
            - "mini_response":       str
            - "gpt4o_latency":       float
            - "mini_latency":        float
            - "gpt4o_cost_estimate": float  (estimated USD for the response)

    Hint:
        Cost estimate = (len(response.split()) / 0.75) / 1000 * COST_PER_1K_OUTPUT_TOKENS["gpt-4o"]
        (0.75 words ≈ 1 token is a rough approximation)
    """
    gpt4o_response, gpt4o_latency = call_openai(prompt, model=OPENAI_MODEL)
    mini_response, mini_latency = call_openai_mini(prompt)

    approx_output_tokens = (len(gpt4o_response.split()) / 0.75) if gpt4o_response else 0.0
    gpt4o_cost_estimate = (approx_output_tokens / 1000.0) * COST_PER_1K_OUTPUT_TOKENS[OPENAI_MODEL]

    return {
        "gpt4o_response": gpt4o_response,
        "mini_response": mini_response,
        "gpt4o_latency": float(gpt4o_latency),
        "mini_latency": float(mini_latency),
        "gpt4o_cost_estimate": float(max(gpt4o_cost_estimate, 0.0)),
    }


# ---------------------------------------------------------------------------
# Test harness compatibility (do not remove)
# ---------------------------------------------------------------------------
# The bundled tests patch using `compare_models.__module__`. If the repo folder
# name contains a hyphen (e.g. "Day01-lab-assignment"), that module name is not
# a valid import path for `unittest.mock.patch()`. We register a safe alias.
import sys as _sys  # noqa: E402

_SAFE_MODULE_ALIAS = "day01_template"
_sys.modules.setdefault(_SAFE_MODULE_ALIAS, _sys.modules[__name__])
compare_models.__module__ = _SAFE_MODULE_ALIAS


# ---------------------------------------------------------------------------
# Task 4 — Streaming chatbot with conversation history
# ---------------------------------------------------------------------------
def streaming_chatbot() -> None:
    """
    Run an interactive streaming chatbot in the terminal.

    Behaviour:
        - Streams tokens from OpenAI as they arrive (print each chunk).
        - Maintains the last 3 conversation turns in history.
        - Typing 'quit' or 'exit' ends the loop.

    Hints:
        - Keep a list `history` of {"role": ..., "content": ...} dicts.
        - Use stream=True in client.chat.completions.create() and iterate:
            for chunk in stream:
                delta = chunk.choices[0].delta.content or ""
                print(delta, end="", flush=True)
        - After each turn, append the assistant reply to history.
        - Trim history to the last 3 turns: history = history[-3:]
    """
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    history: list[dict[str, str]] = []

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"quit", "exit"}:
            break

        history.append({"role": "user", "content": user_input})
        history = history[-3:]

        stream = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=history,
            temperature=0.7,
            top_p=0.9,
            max_tokens=256,
            stream=True,
        )

        assistant_reply_parts: list[str] = []
        for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            if delta:
                assistant_reply_parts.append(delta)
                print(delta, end="", flush=True)
        print()

        assistant_reply = "".join(assistant_reply_parts).strip()
        history.append({"role": "assistant", "content": assistant_reply})
        history = history[-3:]


# ---------------------------------------------------------------------------
# Bonus Task A — Retry with exponential backoff
# ---------------------------------------------------------------------------
def retry_with_backoff(
    fn: Callable,
    max_retries: int = 3,
    base_delay: float = 0.1,
) -> Any:
    """
    Call fn(). If it raises an exception, retry up to max_retries times
    with exponential backoff (base_delay * 2^attempt).

    Args:
        fn:          Zero-argument callable to execute.
        max_retries: Maximum number of retry attempts.
        base_delay:  Initial delay in seconds before the first retry.

    Returns:
        The return value of fn() on success.

    Raises:
        The last exception raised by fn() after all retries are exhausted.
    """
    last_exc: Exception | None = None
    total_attempts = 1 + max_retries

    for attempt in range(total_attempts):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt >= total_attempts - 1:
                raise
            delay = base_delay * (2**attempt)
            time.sleep(delay)

    if last_exc is not None:
        raise last_exc
    raise RuntimeError("retry_with_backoff: unreachable")


# ---------------------------------------------------------------------------
# Bonus Task B — Batch compare
# ---------------------------------------------------------------------------
def batch_compare(prompts: list[str]) -> list[dict]:
    """
    Run compare_models on each prompt in the list.

    Args:
        prompts: List of prompt strings.

    Returns:
        List of dicts, each being the compare_models result with an extra
        key "prompt" containing the original prompt string.
    """
    results: list[dict] = []
    for prompt in prompts:
        r = dict(compare_models(prompt))
        r["prompt"] = prompt
        results.append(r)
    return results


# ---------------------------------------------------------------------------
# Bonus Task C — Format comparison table
# ---------------------------------------------------------------------------
def format_comparison_table(results: list[dict]) -> str:
    """
    Format a list of compare_models results as a readable text table.

    Args:
        results: List of dicts as returned by batch_compare.

    Returns:
        A formatted string table with columns:
        Prompt | GPT-4o Response | Mini Response | GPT-4o Latency | Mini Latency

    Hint:
        Truncate long text to 40 characters for readability.
    """
    def _truncate(s: str, width: int = 40) -> str:
        s = (s or "").replace("\n", " ").strip()
        if len(s) <= width:
            return s
        return s[: width - 1] + "…"

    headers = ["Prompt", "GPT-4o Response", "Mini Response", "GPT-4o Latency", "Mini Latency"]
    col_widths = [20, 40, 40, 12, 12]

    def _fmt_row(cols: list[str]) -> str:
        padded = []
        for col, w in zip(cols, col_widths, strict=True):
            padded.append(col.ljust(w))
        return " | ".join(padded)

    lines = [_fmt_row([h.ljust(w)[:w] for h, w in zip(headers, col_widths, strict=True)])]
    lines.append("-" * len(lines[0]))

    for r in results:
        lines.append(
            _fmt_row(
                [
                    _truncate(str(r.get("prompt", "")), col_widths[0]),
                    _truncate(str(r.get("gpt4o_response", "")), col_widths[1]),
                    _truncate(str(r.get("mini_response", "")), col_widths[2]),
                    f'{float(r.get("gpt4o_latency", 0.0)):.3f}',
                    f'{float(r.get("mini_latency", 0.0)):.3f}',
                ]
            )
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point for manual testing
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    test_prompt = "Explain the difference between temperature and top_p in one sentence."
    print("=== Comparing models ===")
    result = compare_models(test_prompt)
    for key, value in result.items():
        print(f"{key}: {value}")

    print("\n=== Starting chatbot (type 'quit' to exit) ===")
    streaming_chatbot()
