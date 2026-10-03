#!/usr/bin/env python3
"""splash-m5: greedy requests for the copy-rule study (run with SPLASH_M5_TOKEN_LOG set on the
server): three whole-file code edits, the copy-heavy work of coding agents, and three controls
(two prose answers and one reasoning answer).

  dev/m5/copy_workload.py PORT MODEL
"""

import hashlib
import json
import os
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
KEY = open(os.path.expanduser("~/.splash/api-key")).read().strip()
port, model = sys.argv[1], sys.argv[2]
sampled = "--sampled" in sys.argv
code = {
    name: open(os.path.join(HERE, name)).read()
    for name in ("serve_bench.py", "accept_hist.py", "charts.py")
}


def parses(text):
    """Whether the reply's first python code block parses (truncated replies excluded)."""
    import ast
    import re

    m = re.search(r"```python\n(.*?)```", text, re.S)
    if not m:
        return "no-complete-block"
    try:
        ast.parse(m.group(1))
        return "ok"
    except SyntaxError as e:
        return f"SyntaxError line {e.lineno}"


TASKS = [
    (
        "edit-typehints",
        f"Add Python type hints to every function in this file. Return the complete file, unchanged otherwise, in one code block.\n\n```python\n{code['serve_bench.py']}```",
        "none",
    ),
    (
        "edit-rename",
        f"Rename the variable `steps` to `request_steps` everywhere in this file. Return the complete file in one code block.\n\n```python\n{code['accept_hist.py']}```",
        "none",
    ),
    (
        "edit-docstrings",
        f"Add a one-line docstring to every function in this file that lacks one. Return the complete file in one code block.\n\n```python\n{code['charts.py']}```",
        "none",
    ),
    (
        "prose-fridge",
        "Explain in detail how a refrigerator works, for a curious teenager.",
        "none",
    ),
    (
        "prose-history",
        "Write a detailed account of the causes and consequences of the printing press in Europe.",
        "none",
    ),
    (
        "reason-math",
        "How many integers n between 1 and 1000 make n^2 + 3n + 5 divisible by 121? Show your reasoning.",
        "medium",
    ),
]
for name, prompt, effort in TASKS:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 2048,
        "temperature": 0,
        "reasoning_effort": effort,
    }
    if sampled:
        body.update(temperature=1.0, top_p=0.95, top_k=20, max_tokens=4096)
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        json.dumps(body).encode(),
        {"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"},
    )
    start = time.perf_counter()
    d = json.load(urllib.request.urlopen(req, timeout=1800))
    wall = time.perf_counter() - start
    text = d["choices"][0]["message"].get("content") or ""
    text += d["choices"][0]["message"].get("reasoning_content") or ""
    n = d["usage"]["completion_tokens"]
    print(
        f"{name:16} prompt {d['usage']['prompt_tokens']:5} out {n:5} {d['choices'][0]['finish_reason']:6} "
        f"{n / wall:6.1f} tok/s (wall, incl. prefill)  sha {hashlib.sha256(text.encode()).hexdigest()[:12]}"
        + (f"  python {parses(text)}" if name.startswith("edit") else ""),
        flush=True,
    )
