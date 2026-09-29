import json, os, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
KEY = open(os.path.expanduser("~/.splash/api-key")).read().strip()
TOPICS = ["how a refrigerator works", "the causes of the French Revolution", "TCP versus UDP", "the water cycle",
          "how vaccines train the immune system", "the history of the printing press", "how B-trees work",
          "photosynthesis", "the rules of cricket", "how GPS finds your position", "plate tectonics", "compound interest"]
def one(i, samp):
    body = {"model": os.environ["MODEL"], "max_tokens": 512, "reasoning_effort": "none", **samp,
            "messages": [{"role": "user", "content": f"Explain {TOPICS[i % len(TOPICS)]} in detail."}]}
    req = urllib.request.Request(os.environ["URL"], json.dumps(body).encode(),
                                 {"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"})
    t = time.perf_counter(); d = json.load(urllib.request.urlopen(req, timeout=900))
    return d["usage"]["completion_tokens"], time.perf_counter() - t
for mode, samp in (("greedy", {"temperature": 0}), ("sampled", {"temperature": 1.0, "top_p": 0.95, "top_k": 20})):
    for n in [int(x) for x in os.environ.get("NS","1 2 4 8").split()]:
        t = time.perf_counter()
        with ThreadPoolExecutor(n) as ex: res = list(ex.map(lambda i: one(i, samp), range(n)))
        wall = time.perf_counter() - t; tok = sum(r[0] for r in res)
        per = sorted(r[0] / r[1] for r in res)
        print(f"{mode:7} n={n}: aggregate {tok / wall:6.1f} tok/s | per-request {per[0]:.0f}-{per[-1]:.0f} tok/s | {tok} tok in {wall:.1f}s", flush=True)
