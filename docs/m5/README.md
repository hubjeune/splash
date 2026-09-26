<!-- splash-m5 public write-up. DRAFT: private until the fork goes public; {{...}} marks
     values that tonight's run (dev/m5/tonight.sh) or a later measurement fills in. At go-live,
     link this from the top of the repository README. -->

# splash-m5

splash-m5 is a fork of [Inco's Splash](https://github.com/incoai/splash) that retunes and
extends its Metal kernels for Apple **M5**-family GPUs. It was developed and measured on a
40-core **M5 Max** (128 GB).

What the fork adds on top of Splash 1.1.0:

- **Faster decode for Qwen3.8-27B-family models.** New split-K verify kernels
  (`SplitSums32`) and measured kernel choices for the 27B shapes. On Inco's own
  Qwen3.8-27B package it is 15–37% faster than stock at 1–4 concurrent requests.
- **Kernel choices from a file** (`SPLASH_KERNEL_CHOICES`), so a device can be retuned without
  a rebuild. The same mechanism covers GGUF (block-quantized) decode.
- **Tools that keep the numbers honest.** They include a kernel bench checked against fp64, a
  whole-step benchmark with confidence intervals, bit-for-bit build comparison and a
  steady-state serving benchmark with power readings.

**It is not faster everywhere.** Most of the single-request gain on a 27B comes from measured
kernel choices, which Splash's own `tune-kernels` tool can also produce for its built-in
kernels. The fork's own kernels pay off mainly at 2–4 concurrent requests. Long-context decode (past ~40K tokens) is limited by
attention, and nothing here moved it measurably. On GGUF models the gains are small (~2%).
[Where stock is ahead or even](#where-stock-is-ahead-or-even) lists these cases.

Everything else is Inco's: the engine, the package format, the models and draft models, and
the server. The fork tracks upstream releases deliberately (currently Splash 1.1.0; the
`upstream` remote).

- [Quick start](#quick-start)
- [Recommended settings](#recommended-settings)
- [Versus stock Splash](#versus-stock-splash)
- [What worked](#what-worked)
- [What did not](#what-did-not)
- [Ideas left to try](#ideas-left-to-try)
- [Reproduce](#reproduce)
- [Credits](#credits)
- [Support](#support)

## Quick start

{{QUICK_START: build (`make`), run `build/splash` through `server/server.py` with
`SPLASH_KERNEL_CHOICES=tuning/m5max-40c-swift15-v8.choices`; requirements (macOS 26.4+, an
M5-family Mac; other chips run but are untuned).}}

## Recommended settings

| Model | Choices file | Notes |
|---|---|---|
| Qwen3.8-27B family, Splash package (Inco's, Swift-1.5, other fine-tunes) | `tuning/m5max-40c-swift15-v8.choices` | Tuned on a 40-core M5 Max. Other M5 chips: run the tuner (`dev/tuning`). |
| Qwen3.8-27B GGUF Q8_0 | `tuning/m5max-40c-swift15-q80.choices` | ~2% |
| Qwen3.6-35B-A3B | none | The fork adds nothing measurable here. |

Draft length: {{DRAFT_LENGTH_FINDING (acceptance histograms, tonight's phase A)}}.

## Versus stock Splash

**Method.**
- **Builds.** Stock is vanilla Splash 1.1.0. The fork is this repository with the v8
  choices. Both ran on the same M5 Max with the same packages, side by side on separate
  ports.
- **Serving load.** C concurrent requests (C = 1–4), each generating up to {{4,096}} tokens
  from a long-reasoning prompt with the recommended sampling (temperature 1.0, top_p 0.95,
  top_k 20).
- **Metric.** Aggregate decode tok/s, counted only while all C requests decode
  (`dev/m5/serve_bench.py`). Package power and GPU temperature come from `macmon` over the
  same window.
- **Repeats.** Each value is the mean of two rounds, with stock and fork alternating. Runs
  vary by up to {{NOISE}}%, so **differences under about 5% are ties**.
- **Splash's own harness.** `dev/benchmarks/http_regression.py` (ABBA order, `abba.py`'s
  pass rule) measured decode ms per token and time to first token at 2K and 32K context.
- **Quality.** A 95-task set (maths, code, reasoning, exact-match scoring) on both builds.

**Stale measurements.** The G4a GGUF kernel and the A4 attention change landed after the
v8 server measurements. Both are bit-identical and affect GGUF models and long context only.

### Where splash-m5 leads

Inco's Qwen3.8-27B package (`incoai/Qwen3.8-27B-Splash`), aggregate tok/s, fork and change
against stock:

| Load | C=1 | C=2 | C=3 | C=4 |
|---|---:|---:|---:|---:|
| Greedy, 512 tokens | 98.8 **+26%** | 163.9 **+23%** | 170.8 **+24%** | 209.7 **+15%** |
| Sampled, 512 tokens | 90.5 **+23%** | 151.3 **+28%** | 171.2 **+37%** | 208.6 **+28%** |
| Steady state, sampled, 4,096 tokens | {{S1}} | {{S2}} | {{S3}} | {{S4}} |
| Energy per token (J) | {{E1}} | {{E2}} | {{E3}} | {{E4}} |

![Inco's Qwen3.8-27B, stock vs splash-m5, greedy](charts/official-27b-concurrency.svg)
![Inco's Qwen3.8-27B, stock vs splash-m5, sampled](charts/official-27b-concurrency-sampled.svg)

Stock for reference: greedy 78.2 / 133.7 / 137.8 / 182.6 tok/s, and sampled 73.8 / 118.2 /
124.9 / 163.4 tok/s.

Swift-1.5 (a Qwen3.8-27B fine-tune, same shapes), decode step time from Splash's
`decode_profile` with a 2,048-token prompt:

![Swift-1.5 decode step time](charts/swift-step-time.svg)

| Requests | Stock 1.0.2 | 1.0.2 + tuned choices | splash-m5 (v8) |
|---|---:|---:|---:|
| 1 | 49.0 ms | 41.5 ms | **40.5 ms** |
| 2 | 59.1 ms | 59.4 ms | **44.9 ms** |
| 3 | 87.2 ms | 88.2 ms | **59.3 ms** |
| 4 | 87.0 ms | 85.8 ms | **62.9 ms** |

At one request the fork adds ~2.5% over tuned choices alone. The gain from the fork's own
kernels is at 2–4 requests (1.3–1.5×).

**Quality.** Swift-1.5 scores 95/95 on both engines, and 380/380 with four copies running
concurrently. On Inco's 27B: stock {{Q_STOCK}}/95, fork {{Q_FORK}}/95. The fork's greedy text
differs from stock at near-ties because the split-K kernels add in a different order. Both
engines are deterministic run to run.

### Where stock is ahead or even

| Case | Result |
|---|---|
| Speculative acceptance, Inco's 27B | Fork 0.380 vs stock 0.398 (greedy, 16 prompts). Slightly lower; the speed gain covers it. |
| Long-context decode (40K+) | Tie. Attention sets the limit, and the one change kept (A4) is ~3% of attention and not visible per step. |
| GGUF Q8_0 decode, 2 requests | Tie (−0.1%). 1, 3 and 4 requests: ~2% faster. |
| Qwen3.6-35B-A3B | Not tuned; the fork's attention change is disabled for its shape (it was 3% slower there). |
| Harness, 2K / 32K context | {{HARNESS_RESULT: decode ms/token and TTFT, pass/fail by abba.py's rule}} |

## What worked

1. **Split-K verify kernels with row sums computed once (`SplitSums32`)** on the MPP tensor
   units. The residual, plain and gate/up projections gained at 16–32 rows, which cut the
   step 23–32% at 2–4 requests.
2. **Lighter barriers (H1).** A simdgroup barrier replaces a threadgroup one where a tile
   has one simdgroup.
3. **Measured choices per shape** (`v8`), loaded from a file.
4. **GGUF input prefetch (G4a).** The staged GGUF tile now prefetches its input rows a step
   ahead into space freed from its weight stage. It uses the same threadgroup memory,
   produces bit-identical output and is 2–8% faster at 32 rows.

   ![GGUF Q8_0 at 32 rows](charts/gguf-g4a.svg)
5. **Attention QK on 4 simdgroups for 6-head groups (A4).** Bit-identical; attention 2–3%
   faster.

## What did not

The full log, with numbers, is in [FORK.md](../../FORK.md).

| Idea | Result | Why |
|---|---|---|
| Epilogue redesigns (bias matmul, early loads, shuffles, cache) | No gain | The 32-row matmul is compute-bound at ~57 TFLOPS. |
| Kernel fusion | −0.1% | Launch cost is not the gap. |
| Concurrent encoder | 0.6–2.9% slower | Decode is a dependency chain. |
| Attention: operand swap (A1) | Wrong output | Discarded. |
| Attention: two pages per step (A2), next-page prefetch (A3) | 0%, −2% | Not latency-bound. |
| Attention: int8 × int8 QK (A6) | 3–7% slower, 3× the error | The M5 tensor unit is no faster at int8 for this shape. |
| GGUF: extra threadgroup memory (G1) or registers (G2) for prefetch | 17–35% slower | The tile's occupancy collapses above 8 KB. |

![Long-context attention probes](charts/attention-probes.svg)

**Findings that should transfer to other M5 work:**
- Verify attention is bound by the tensor unit's bf16 × int8 matmuls. The softmax between
  them is serialized.
- The GGUF staged tile's speed depends on its threadgroup-memory footprint. Doubling it
  cost 30–38%.
- MPP's `relaxed_precision` flag had no measurable effect on bf16 × int8 QK.

## Ideas left to try

1. **Attention rewrite.** Overlap one threadgroup's softmax with another's matmul (a
   producer/consumer layout). It is the only lever left for long context.
2. **Draft length.** Splash verifies 8 rows per request. {{DRAFT_LENGTH_FINDING}}
3. **DFlash draft fine-tuned on Swift.** Acceptance on a fine-tune is below the base
   model's. Cloud compute for this is what [Support](#support) is for.
4. **Batch width 8.** A plan exists (the converter project's `docs/PLAN_CONCURRENCY.md`).
5. **Q4_K and other GGUF formats.** G4a applies to them but is only measured on Q8_0.
6. **Token-agreement check.** Compare next-token choices position by position against
   upstream, as the M1 port does. It is a finer quality gate than a task set.
7. **Tuning other M5 chips.** The choices files are for a 40-core M5 Max.

## Reproduce

```sh
make && dev/m5/build.sh
dev/m5/tonight.sh                                 # every serving, quality and harness number above
./build/m5/kernel-bench build/splash.metallib --check              # affine Q4 kernels, fp64
./build/m5/kernel-bench build/splash.metallib --gguf q80 --check   # GGUF Q8_0
python3 dev/m5/step_bench.py base=- new=tuning/m5max-40c-swift15-v8.choices
python3 dev/m5/charts.py                          # the charts in docs/m5/charts
```

## Credits

- **[Inco](https://github.com/incoai/splash)** built Splash, its models and its DFlash draft
  models. This fork only changes kernels and tuning.
- **SnooPredictions515**, whose M1 port and write-up
  ([r/LocalLLaMA](https://www.reddit.com/r/LocalLLaMA/comments/1wmbbf9/splash_engine_qwen3827b_in_native_8bit_at_3755/))
  suggested the attention and quality-agreement experiments we ran.
- **[giveen/ninfer-ext](https://github.com/giveen/ninfer-ext)** for the reporting layout this
  write-up follows: method first, and losses next to wins.

## Support

If this is useful and you would like to help fund cloud compute for fine-tuning a DFlash
draft model for Swift: [ko-fi.com/severalviolins](https://ko-fi.com/severalviolins).
