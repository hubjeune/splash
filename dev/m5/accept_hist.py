#!/usr/bin/env python3
"""splash-m5: acceptance-length histogram from SPLASH_M5_ACCEPT_LOG ("lanes drafted accepted"
per request per decode step), and what shorter drafts would keep.

For each batch width it prints the share of steps that accepted k drafts, the mean tokens
per step (accepted + 1), and, for a draft cut to D tokens, the mean tokens per step that
would remain (min(accepted, D) + 1): an upper bound on what a shorter draft keeps, before
its cheaper verify is priced in.

  dev/m5/accept_hist.py accept.log
"""

import collections
import sys

steps = collections.defaultdict(list)
for line in open(sys.argv[1]):
    parts = line.split()
    if len(parts) == 3:
        width, drafted, accepted = map(int, parts)
        if drafted:
            steps[width].append((drafted, accepted))
for width in sorted(steps):
    rows = steps[width]
    drafted = max(d for d, _ in rows)
    counts = collections.Counter(a for _, a in rows)
    mean = sum(a + 1 for _, a in rows) / len(rows)
    print(
        f"width {width}: {len(rows)} request-steps, {drafted} drafted, mean {mean:.2f} tokens/step"
    )
    print(
        "  accepted k:  "
        + "  ".join(
            f"{k}:{100 * counts[k] / len(rows):4.1f}%" for k in range(drafted + 1)
        )
    )
    kept = []
    for d in range(1, drafted + 1):
        m = sum(min(a, d) + 1 for _, a in rows) / len(rows)
        kept.append(f"D={d}:{m:.2f} ({100 * m / mean:.0f}%)")
    print("  draft cut to D keeps: " + "  ".join(kept))
