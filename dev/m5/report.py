#!/usr/bin/env python3
"""Splish performance report: measure a running Splish server (and optionally a stock Splash
server) on this Mac and print a Markdown report to paste into a GitHub issue.

Start Splish as in the README (`./splash serve ...`), then:

  python3 dev/m5/report.py --port 8000 --model MODEL_ID
  python3 dev/m5/report.py --port 8000 --model MODEL_ID --baseline-port 8001   # vs stock Splash

For the comparison, run stock Splash (`brew install incoai/tap/splash`, then `splash serve
--model MODEL_ID --port 8001`) at the same time, with the same model; each engine is measured
alternately. The load is dev/m5/serve_bench.py's: 1-4 concurrent long-reasoning requests,
recommended sampling, counted only while all requests decode. Needs the API key the servers
use: ~/.splash/api-key, or SPLASH_API_KEY.
"""
import argparse, json, os, platform, statistics, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def run(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=60).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def device():
    chip = run(["sysctl", "-n", "machdep.cpu.brand_string"])
    memory = run(["sysctl", "-n", "hw.memsize"])
    cores = ""
    for line in run(["system_profiler", "SPDisplaysDataType"]).splitlines():
        if "Total Number of Cores" in line:
            cores = line.split(":")[-1].strip()
    root = os.path.dirname(os.path.dirname(HERE))
    commit = run(["git", "-C", root, "describe", "--tags", "--always", "--dirty"])
    return {"chip": chip, "gpu_cores": cores,
            "memory_gb": round(int(memory) / 2**30) if memory.isdigit() else None,
            "macos": platform.mac_ver()[0], "splish": commit}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, required=True, help="the Splish server")
    ap.add_argument("--baseline-port", type=int, help="a stock Splash server, for the comparison")
    ap.add_argument("--model", required=True, help="the model id both servers serve")
    ap.add_argument("--concurrency", default="1,2,3,4")
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--json", help="also write the raw measurements here")
    a = ap.parse_args()
    if not os.environ.get("SPLASH_API_KEY") and not os.path.exists(os.path.expanduser("~/.splash/api-key")):
        sys.exit("report: no API key (~/.splash/api-key or SPLASH_API_KEY)")
    import serve_bench  # its key: SPLASH_API_KEY, else ~/.splash/api-key

    engines = [("Splish", a.port)] + ([("stock", a.baseline_port)] if a.baseline_port else [])
    levels = [int(x) for x in a.concurrency.split(",")]
    results = {label: {c: [] for c in levels} for label, _ in engines}
    for r in range(a.rounds):
        order = engines if r % 2 == 0 else engines[::-1]
        for c in levels:
            for label, port in order:
                row = serve_bench.run(port, a.model, c, a.max_tokens, "medium")
                results[label][c].append(row)
                print(f"round {r + 1} {label:6} C={c}: {row['steady_tok_s']:.1f} tok/s", file=sys.stderr, flush=True)

    info = device()
    print(f"### Splish performance report\n")
    print(f"- **Chip:** {info['chip']}, {info['gpu_cores']} GPU cores, {info['memory_gb']} GB")
    print(f"- **macOS:** {info['macos']}  **Splish:** {info['splish'] or 'unknown'}")
    print(f"- **Model:** `{a.model}`")
    print(f"- **Splish settings:** SPLASH_KERNEL_CHOICES=`<fill in>`, SPLASH_M5_COPY_MIN_MATCH=`<fill in>`")
    print(f"- **Load:** long reasoning, sampled, up to {a.max_tokens} tokens, steady state, "
          f"mean of {a.rounds} rounds\n")
    header = "| Requests | Splish tok/s |" + (" Stock tok/s | Gain |" if a.baseline_port else "")
    print(header)
    print("|---:|---:|" + ("---:|---:|" if a.baseline_port else ""))
    for c in levels:
        splish = statistics.mean(x["steady_tok_s"] for x in results["Splish"][c])
        line = f"| {c} | {splish:.1f} |"
        if a.baseline_port:
            stock = statistics.mean(x["steady_tok_s"] for x in results["stock"][c])
            line += f" {stock:.1f} | {100 * (splish / stock - 1):+.0f}% |"
        print(line)
    temps = [x.get("gpu_temp_max") for rows in results["Splish"].values() for x in rows if x.get("gpu_temp_max")]
    if temps:
        print(f"\nGPU temperature peak during Splish runs: {max(temps):.0f} °C (macmon).")
    if a.json:
        with open(a.json, "w") as f:
            json.dump({"device": info, "model": a.model, "results": results}, f, indent=1)


if __name__ == "__main__":
    main()
