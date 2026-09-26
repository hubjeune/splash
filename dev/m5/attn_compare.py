#!/usr/bin/env python3
"""splash-m5: time a variant metallib's verify attention against build/splash.metallib
(Swift 27B shape) with output agreement.  dev/m5/attn_compare.py VARIANT [histories] [lanes]"""
import json, os, subprocess, sys
variant = sys.argv[1]
histories = sys.argv[2] if len(sys.argv) > 2 else "2048,32768,131072"
lanes = sys.argv[3] if len(sys.argv) > 3 else "1,4"
out = subprocess.run(["./build/engine-tests/attention-sweep", "build/splash.metallib", "--compare-metallib", variant,
                      "--histories", histories, "--shapes", "27b", "--lanes", lanes, "--phases", "verify", "--repeat", "7"],
                     capture_output=True, text=True, env=dict(os.environ, SPLASH_M5_NO_BIT_CHECK="1"))
base = {}
for line in out.stderr.splitlines():
    if "output vs base" in line or "rror" in line:
        print(line.rstrip())
if out.returncode:
    sys.exit(out.stdout[-500:] + out.stderr[-500:])
for c in json.loads(out.stdout)["cases"]:
    key = (c["history"], c["lanes"])
    split = sum(v for k, v in c["pipelines"].items() if "_split" in k)
    if c["variant"] == 0:
        base[key] = (split, c["fused_ms"]); continue
    b = base[key]
    print(f"{key[0]:>7} L{key[1]}  split {b[0]:.3f} -> {split:.3f} ms ({100*(split/b[0]-1):+.1f}%)"
          f"   fused {b[1]:.3f} -> {c['fused_ms']:.3f} ms ({100*(c['fused_ms']/b[1]-1):+.1f}%)")
