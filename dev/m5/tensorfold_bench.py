#!/usr/bin/env python3
"""Run TensorFold's single-stream client (ashhart/TensorFold tools/bench_openai.py, unmodified)
against a Splash server, for a like-for-like comparison with TensorFold's published M5 Max numbers.

Splash differs in three ways, bridged here by rewriting each request before it is sent; the
timing (decode tok/s = (tokens - 1) / (last - first streamed token)) stays TensorFold's:
- Splash requires an API key: the Authorization header is added.
- Splash has no raw /v1/completions: the raw "fibonacci-raw" prompt goes to /v1/chat/completions as
  a user message, so that "code" row is a chat-templated prompt, not a raw one.
- Thinking is off through `reasoning_effort: none`, not `chat_template_kwargs`.
`ignore_eos` and `seed` are passed through; Splash ignores them (64-token replies rarely stop early).

  python3 dev/m5/tensorfold_bench.py BENCH_OPENAI_PY http://127.0.0.1:8042 MODEL --reps 5 --label fork
"""
import importlib.util, json, os, sys, urllib.request

KEY = open(os.path.expanduser("~/.splash/api-key")).read().strip()
_Request = urllib.request.Request


def adapted(url, data=None, headers=None, **kwargs):
    headers = dict(headers or {}, Authorization=f"Bearer {KEY}")
    if data is not None:
        body = json.loads(data)
        if url.endswith("/v1/completions"):
            url = url[: -len("/v1/completions")] + "/v1/chat/completions"
            body["messages"] = [{"role": "user", "content": body.pop("prompt")}]
        if body.pop("chat_template_kwargs", None) is not None or "messages" in body:
            body["reasoning_effort"] = "none"
        data = json.dumps(body).encode()
    return _Request(url, data=data, headers=headers, **kwargs)


urllib.request.Request = adapted
spec = importlib.util.spec_from_file_location("bench_openai", sys.argv[1])
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)
sys.argv = [sys.argv[1]] + sys.argv[2:]
bench.main()
