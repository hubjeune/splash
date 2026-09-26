#!/usr/bin/env python3
"""splash-m5 steady-state serving bench: C concurrent long generations against one
running Splash server, measured only while all C decode (ninfer-ext's method).

Each of the C requests streams a long-reasoning prompt (distinct per lane) with the
recommended sampling (temperature 1.0, top_p 0.95, top_k 20) up to --max-tokens. The
steady window runs from the last stream's first chunk to the first stream's last chunk;
each stream's tokens inside it are its completion tokens scaled by the share of its
streamed characters that arrived inside it. Reported per run: steady aggregate tok/s,
the naive aggregate (all tokens / wall), and, from macmon over the window, mean package
power, joules per token and GPU temperature (mean and max).

  dev/m5/serve_bench.py --label fork --port 8042 --model incoai/Qwen3.8-27B-Splash \\
      --concurrency 1,2,3,4 --out results.jsonl
"""
import argparse, json, os, subprocess, threading, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

PROBLEMS = [
    "Find the number of ordered pairs of positive integers (m, n) with m*n = 2^10 * 3^6 such that "
    "gcd(m, n) is a perfect square. Reason carefully step by step.",
    "A regular 12-gon is inscribed in a unit circle. Find the sum of the squares of the lengths of "
    "all its sides and diagonals. Reason carefully step by step.",
    "How many integers n between 1 and 1000 inclusive make n^2 + 3n + 5 divisible by 121? "
    "Reason carefully step by step.",
    "Three fair dice are rolled until the sum is at least 15. Find the expected number of rolls, "
    "as a reduced fraction. Reason carefully step by step.",
]
KEY = open(os.path.expanduser("~/.splash/api-key")).read().strip()


def stream(port, model, lane, max_tokens, reasoning, prefix=""):
    body = {"model": model, "stream": True, "stream_options": {"include_usage": True},
            "max_tokens": max_tokens, "temperature": 1.0, "top_p": 0.95, "top_k": 20,
            "reasoning_effort": reasoning,
            "messages": [{"role": "user", "content": (f"Here is a document:\n\n{prefix}\n\nIgnore the document. " if prefix else "") + PROBLEMS[lane % len(PROBLEMS)]}]}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", json.dumps(body).encode(),
                                 {"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"})
    chunks, usage, finish = [], None, None  # (time, characters)
    with urllib.request.urlopen(req, timeout=3600) as response:
        for raw in response:
            line = raw.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            event = json.loads(line[5:])
            if event.get("usage"):
                usage = event["usage"]
            for choice in event.get("choices", []):
                delta = choice.get("delta", {})
                text = (delta.get("content") or "") + (delta.get("reasoning_content") or "") + \
                       (delta.get("reasoning") or "")
                if text:
                    chunks.append((time.perf_counter(), len(text)))
                finish = choice.get("finish_reason") or finish
    return {"chunks": chunks, "tokens": usage["completion_tokens"] if usage else None, "finish": finish,
            "prompt_tokens": usage["prompt_tokens"] if usage else None}


class Power:
    """macmon samples (every 250 ms) on a thread for the life of the run."""
    def __init__(self):
        self.samples, self.process = [], None
        try:
            self.process = subprocess.Popen(["macmon", "pipe", "-i", "250"], stdout=subprocess.PIPE,
                                            stderr=subprocess.DEVNULL, text=True)
            threading.Thread(target=self._read, daemon=True).start()
        except FileNotFoundError:
            pass

    def _read(self):
        for line in self.process.stdout:
            try:
                d = json.loads(line)
                self.samples.append((time.perf_counter(), d["all_power"], d["sys_power"],
                                     d["temp"]["gpu_temp_avg"]))
            except (ValueError, KeyError):
                pass

    def stop(self):
        if self.process:
            self.process.terminate()

    def window(self, start, end):
        s = [x for x in self.samples if start <= x[0] <= end]
        if not s:
            return {}
        return {"package_w": sum(x[1] for x in s) / len(s), "system_w": sum(x[2] for x in s) / len(s),
                "gpu_temp_mean": sum(x[3] for x in s) / len(s), "gpu_temp_max": max(x[3] for x in s)}


def run(port, model, c, max_tokens, reasoning, context_tokens=0, corpus=None):
    power = Power()
    time.sleep(1.0)
    start = time.perf_counter()
    with ThreadPoolExecutor(c) as pool:
        # A distinct passage per lane (about 4.2 characters per token), so lanes share no prefix.
        prefixes = [corpus[lane * int(context_tokens * 4.4):][:int(context_tokens * 4.2)] if context_tokens else ""
                    for lane in range(c)]
        results = list(pool.map(lambda lane: stream(port, model, lane, max_tokens, reasoning, prefixes[lane]), range(c)))
    wall = time.perf_counter() - start
    time.sleep(0.5)
    power.stop()
    first = max(r["chunks"][0][0] for r in results)
    last = min(r["chunks"][-1][0] for r in results)
    steady_tokens = 0.0
    for r in results:
        total = sum(n for _, n in r["chunks"]) or 1
        inside = sum(n for t, n in r["chunks"] if first <= t <= last)
        steady_tokens += (r["tokens"] or 0) * inside / total
    window = max(last - first, 1e-9)
    tokens = sum(r["tokens"] or 0 for r in results)
    row = {"concurrency": c, "steady_tok_s": steady_tokens / window, "naive_tok_s": tokens / wall,
           "window_s": window, "wall_s": wall, "tokens": tokens,
           "finish": [r["finish"] for r in results], "prompt_tokens": [r["prompt_tokens"] for r in results],
           **power.window(first, last)}
    if "package_w" in row:
        row["joules_per_token"] = row["package_w"] * window / max(steady_tokens, 1)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--concurrency", default="1,2,3,4")
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--reasoning", default="medium")
    ap.add_argument("--round", type=int, default=1)
    ap.add_argument("--context-tokens", type=int, default=0, help="long document before each prompt")
    ap.add_argument("--corpus", default=os.path.expanduser("~/Models/splash/swift-splash-project/evaluation/corpora/wiki.test.raw"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    corpus = open(a.corpus).read() if a.context_tokens else None
    for c in [int(x) for x in a.concurrency.split(",")]:
        row = {"label": a.label, "model": a.model, "round": a.round, "context_tokens": a.context_tokens,
               **run(a.port, a.model, c, a.max_tokens, a.reasoning, a.context_tokens, corpus)}
        with open(a.out, "a") as f:
            f.write(json.dumps(row) + "\n")
        print(f"{a.label:6} r{a.round} C={c}: steady {row['steady_tok_s']:6.1f} tok/s (naive {row['naive_tok_s']:6.1f})"
              f"  {row.get('joules_per_token', float('nan')):.2f} J/tok  gpu {row.get('gpu_temp_max', float('nan')):.0f}C max",
              flush=True)


if __name__ == "__main__":
    main()
