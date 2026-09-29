#!/usr/bin/env python3

"""Real-model regression: --power must not change a decoded token sequence.

The duty-cycle throttle sleeps between the same prefill commands power 100
dispatches; it never reslices a prompt into shorter commands. The prefill
path's floating-point kernels are not bit-invariant to a command's row budget,
so the committed fixtures once diverged between --power 100 and a throttled
run. Each power level gets a fresh server process, and the fixture prompt is
that process's first, cold request: identical greedy text proves the throttle
left the computation, including the KV cache, untouched.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import smoke_real as smoke  # noqa: E402

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "power-identity"
DEFAULT_PROMPTS = (FIXTURES / "screen3.txt", FIXTURES / "k25.txt")


class PowerFailure(smoke.SmokeFailure):
    pass


def complete(port: int, model: str, prompt: str, max_tokens: int) -> str:
    """One cold greedy completion; the caller starts a fresh server first."""
    status, document = smoke.request(
        port,
        "POST",
        "/v1/chat/completions",
        smoke.chat_body(
            model,
            prompt,
            max_completion_tokens=max_tokens,
            seed=1,
            temperature=0,
        ),
        timeout=1800,
    )
    if status != 200:
        raise PowerFailure(f"chat request failed: {status} {document}")
    prompt_tokens = document.get("usage", {}).get("prompt_tokens")
    cached = (
        document.get("usage", {}).get("prompt_tokens_details", {}).get("cached_tokens")
    )
    if cached not in (0, None):
        raise PowerFailure(f"prompt was not cold: cached_tokens={cached}")
    try:
        text = document["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError, TypeError) as error:
        raise PowerFailure(f"chat response has no completion: {document}") from error
    if not prompt_tokens:
        raise PowerFailure("chat response reported no prompt tokens")
    return text


def run_prompt(arguments, prompt_path: Path) -> None:
    prompt = prompt_path.read_text()
    results: dict[int, str] = {}
    for power in (100, 50):
        server = smoke.RealServer(arguments, power=power)
        try:
            smoke.validate_status(server.wait_ready(arguments.startup_timeout))
            results[power] = complete(
                server.port, arguments.model, prompt, arguments.max_tokens
            )
        finally:
            server.close()
    if not results[100]:
        raise PowerFailure(f"{prompt_path.name}: greedy completion was empty")
    if results[100] != results[50]:
        raise PowerFailure(
            f"{prompt_path.name}: --power 50 changed greedy output\n"
            f"  power 100: {results[100]!r}\n"
            f"  power  50: {results[50]!r}"
        )
    print(f"{prompt_path.name}: power 100 == power 50 ({results[100]!r})", flush=True)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    smoke.add_server_arguments(parser)
    parser.add_argument(
        "--prompt",
        type=Path,
        action="append",
        help="prompt file to check (default: the two committed repro fixtures)",
    )
    parser.add_argument("--max-tokens", type=int, default=24)
    return smoke.resolve_server_arguments(parser.parse_args(argv))


def main(argv=None) -> int:
    arguments = parse_args(argv)
    smoke.hold_package(arguments)
    for prompt in arguments.prompt or DEFAULT_PROMPTS:
        run_prompt(arguments, prompt)
    print("power identity: PASS", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
