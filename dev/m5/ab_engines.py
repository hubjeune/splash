"""Interleaved greedy A/B between Splash servers: tok/s, acceptance, text identity.
usage: ab_engines.py OUT.json LABEL=PORT [LABEL=PORT ...]"""
import json, os, sys, time, urllib.request
KEY = open(os.path.expanduser("~/.splash/api-key")).read().strip()
MODEL = os.environ.get("MODEL", "local/Swift-1.5-4bit-MLX-Splash")
PROMPTS = [
 "Write a Python function that parses an ISO 8601 duration like P3DT4H12M into seconds, with tests.",
 "Explain how a refrigerator works to a curious ten-year-old.",
 "Write a short story about a lighthouse keeper who finds a message in a bottle.",
 "A train leaves at 09:40 and travels 212 km at 84 km/h, then 95 km at 61 km/h after a 12-minute stop. When does it arrive? Show your working.",
 "List twelve practical tips for reducing energy use in a small flat, with one sentence each.",
 "Write a SQL schema for a library lending system and three example queries.",
 "Summarise the causes of the French Revolution in about 300 words.",
 "Write a polite but firm email to a landlord about a boiler that has been broken for a week."]
out, engines = sys.argv[1], [a.split("=") for a in sys.argv[2:]]
def call(port, path, body=None):
    r = urllib.request.Request(f"http://127.0.0.1:{port}{path}", json.dumps(body).encode() if body else None,
                               {"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(r, timeout=900))
def metrics(port):
    m = call(port, "/status")["metrics"]; return m["drafted_tokens"], m["accepted_draft_tokens"]
res = {l: [] for l, _ in engines}
for rnd in range(2):
    for i, p in enumerate(PROMPTS):
        order = engines if (i + rnd) % 2 == 0 else engines[::-1]
        for label, port in order:
            a0 = metrics(port); t = time.perf_counter()
            d = call(port, "/v1/chat/completions", {"model": MODEL, "messages": [{"role": "user", "content": p}],
                     "max_tokens": 512, "temperature": 0, "reasoning_effort": "none"})
            w = time.perf_counter() - t; a1 = metrics(port); n = d["usage"]["completion_tokens"]
            res[label].append({"round": rnd, "i": i, "tokens": n, "tps": n / w,
                               "drafted": a1[0] - a0[0], "accepted": a1[1] - a0[1],
                               "text": d["choices"][0]["message"]["content"]})
json.dump(res, open(out, "w"))
labels = [l for l, _ in engines]; base = labels[0]
for l in labels:
    r = res[l]; tps = [x["tps"] for x in r]
    acc = sum(x["accepted"] for x in r) / max(1, sum(x["drafted"] for x in r))
    same = sum(x["text"] == y["text"] for x, y in zip(r, res[base]))
    det = sum(r[k]["text"] == r[k + len(PROMPTS)]["text"] for k in range(len(PROMPTS)))
    print(f"{l:8} mean {sum(tps)/len(tps):6.1f} tok/s  acceptance {acc:.3f}  identical-to-{base} {same}/{len(r)}  run-to-run {det}/{len(PROMPTS)}")
