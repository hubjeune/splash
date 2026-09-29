#!/usr/bin/env python3
"""splash-m5: what TensorFold's copy rule would give Splash, replayed from SPLASH_M5_TOKEN_LOG.

Greedy transcripts only: the continuation is then fixed whatever the drafter proposed.
For each request the replay walks its final token sequence from the first decode position:
- DFlash only (observed): every step advances by what DFlash actually got accepted, plus one.
- Copy rule at verify width W: when the last MIN_MATCH (8) tokens occur earlier in the
  sequence (prompt or output), the draft is the W - 1 tokens that followed the most recent
  earlier occurrence. The step advances by the matching prefix of that copy, plus one. With no
  match, DFlash's observed acceptance from this position is used (the remainder of the
  observed step that covers it). This is an estimate, not a re-run.
Speed weighs each step by its verify cost relative to 8 rows (--cost W=ratio). The defaults
come from Swift-1.5's decode_profile: 16 rows 1.11x, 32 rows 1.55x, the cost of widening a
verify window at one request.

  dev/m5/copy_rule.py tokens.log
"""
import argparse, collections

MIN_MATCH = 8


def load(path):
    prompts, steps = {}, collections.defaultdict(list)
    for line in open(path):
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "P":
            prompts[parts[1]] = [int(t) for t in parts[3:]]
        elif parts[0] == "S":
            accepted, n = int(parts[2]), int(parts[3])
            steps[parts[1]].append((accepted, [int(t) for t in parts[4:4 + n]]))
    return prompts, steps


def replay(prompt, steps, width, costs):
    tokens = list(prompt)
    observed = {}  # position -> tokens DFlash's step emitted from here (its remainder)
    for accepted, out in steps:
        start = len(tokens)
        for k in range(len(out)):
            observed[start + k] = len(out) - k
        tokens += out
    begin, end = len(prompt), len(tokens)
    # Most recent earlier occurrence of each MIN_MATCH-gram, built as the walk advances.
    index, indexed = {}, 0
    position, steps_taken, cost, copied = begin, 0, 0.0, 0
    while position < end:
        while indexed + MIN_MATCH < position:  # earlier occurrences only, never the current window
            index[tuple(tokens[indexed:indexed + MIN_MATCH])] = indexed + MIN_MATCH
            indexed += 1
        key = tuple(tokens[position - MIN_MATCH:position]) if position >= MIN_MATCH else None
        follow = index.get(key) if key else None
        draft = tokens[follow:min(follow + width - 1, position)] if follow is not None and width else []  # seen only
        if draft:
            actual = tokens[position:position + width - 1]
            match = 0
            while match < min(len(draft), len(actual)) and draft[match] == actual[match]:
                match += 1
            advance = min(match + 1, end - position)
            step_cost = costs[width]
            copied += 1
        else:
            advance = observed.get(position, 1)
            step_cost = 1.0
        position += advance
        steps_taken += 1
        cost += step_cost
    return end - begin, steps_taken, cost, copied


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--min-match", type=int, default=8, help="matching tokens a copy needs")
    ap.add_argument("--cost", action="append", default=[], help="W=ratio, verify cost relative to 8 rows")
    a = ap.parse_args()
    global MIN_MATCH
    MIN_MATCH = a.min_match
    costs = {8: 1.0, 16: 1.11, 32: 1.55}
    for item in a.cost:
        w, r = item.split("="); costs[int(w)] = float(r)
    prompts, steps = load(a.log)
    rows = []
    for rid, request_steps in steps.items():
        if rid not in prompts:
            continue
        base = replay(prompts[rid], request_steps, 0, costs)
        line = [rid, base[0], base[1]]
        for w in (8, 16, 32):
            line.append(replay(prompts[rid], request_steps, w, costs))
        rows.append(line)
    print(f"{'request':>12} {'tokens':>6}  DFlash tok/step | copy rule W=8, W=16, W=32: tok/step (speed vs DFlash, share of steps copying)")
    totals = collections.defaultdict(float)
    for rid, n, base_steps, *ws in rows:
        cells = []
        totals["n"] += n; totals["base"] += base_steps
        for w, (m, s, c, copied) in zip((8, 16, 32), ws):
            speed = (n / c) / (n / base_steps)
            cells.append(f"{n / s:5.2f} ({speed:5.2f}x, {100 * copied / s:3.0f}%)")
            totals[f"s{w}"] += s; totals[f"c{w}"] += c
        print(f"{rid[-12:]:>12} {n:6d}  {n / base_steps:5.2f} | " + "  ".join(cells))
    n = totals["n"]
    print(f"{'all':>12} {int(n):6d}  {n / totals['base']:5.2f} | " + "  ".join(
        f"{n / totals[f's{w}']:5.2f} ({totals['base'] / totals[f'c{w}']:5.2f}x)" for w in (8, 16, 32)))


if __name__ == "__main__":
    main()
