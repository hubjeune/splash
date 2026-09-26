#pragma once
#include "metal/abi/Gguf.h"
#include "metal/kernels/common/gguf_staged.h"
#include "metal/kernels/common/gguf_tile.h"

#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#include <metal_stdlib>
using namespace metal;
using namespace mpp::tensor_ops;

// The staged decode tile of kernels/shared/gguf_linear.metal and kernels/shared/moe_gguf.metal: each simdgroup
// dequantizes its own columns' weights into a half stage and runs matmul2d on it alone. Includers set `#pragma clang
// fp reassociate(off)` first.

// splash-m5 diagnostic (timing only; wrong outputs when nonzero): 1 skips the decode
// tile's matmul2d, 2 reads its input rows from the stage (threadgroup) instead of device.
#ifndef SPLASH_M5_GGUF_DIAG
#define SPLASH_M5_GGUF_DIAG 0
#endif

// The two stages of a simdgroup, and of the threadgroup's GGUF_TILE_COLUMNS / GGUF_STAGED_COLUMNS simdgroups.
// splash-m5 G1: the input rows of each step are prefetched a step ahead into registers and
// staged per simdgroup (double-buffered, up to 32 rows), as the weights already are; the
// matmul then reads them from threadgroup memory instead of stalling on device loads.
#ifndef SPLASH_M5_GGUF_STAGE_PAD  // diagnostic: G1's stage size, device input
#define SPLASH_M5_GGUF_STAGE_PAD 0
#endif
// splash-m5 G2: the input rows of step + 1 are loaded into a left-input cooperative tensor
// (registers) while step is dequantized and multiplied: no threadgroup memory.
// splash-m5 G3: one weight stage per simdgroup instead of two (half the threadgroup
// memory, for occupancy); a simdgroup barrier orders each matmul before the next dequant.
#ifndef SPLASH_M5_GGUF_SINGLE_STAGE
#define SPLASH_M5_GGUF_SINGLE_STAGE 1  // G4a, adopted
#endif
#ifndef SPLASH_M5_GGUF_ACT
#define SPLASH_M5_GGUF_ACT 0
#endif
#ifndef SPLASH_M5_GGUF_APREFETCH
#define SPLASH_M5_GGUF_APREFETCH 1  // G4a, adopted
#endif
constant constexpr uint kStagedInputRows = 32;
constant constexpr uint kStagedBuffers = SPLASH_M5_GGUF_SINGLE_STAGE ? 1 : 2;
constant constexpr uint kStagedSimdgroupStage = kStagedBuffers * GGUF_STAGED_COLUMNS * GGUF_STAGED_STEP +
    (SPLASH_M5_GGUF_APREFETCH || SPLASH_M5_GGUF_STAGE_PAD ? kStagedBuffers * kStagedInputRows * GGUF_STAGED_STEP : 0);
constant constexpr uint kStagedStages = GGUF_TILE_COLUMNS / GGUF_STAGED_COLUMNS * kStagedSimdgroupStage;

// The destination of a Rows-row tile, zeroed by the caller: returning an initialized cooperative tensor loses its
// initial values on Apple9 in runtime-format kernels (also with shader validation).
template <ushort Rows, ushort Cols, ushort KS>
inline auto staged_accumulator(device bfloat *input, uint input_size, threadgroup half *stage) {
  auto a = tensor(input, dextents<int, 2>{int(input_size), Rows}, array<int, 2>{1, int(input_size)});
  constexpr auto descriptor = matmul2d_descriptor(Rows, Cols, KS, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<descriptor, execution_simdgroups<1>> operation;
  auto a0 = a.template slice<KS, Rows>(0, 0);
  tensor<threadgroup half, dextents<int, 2>, tensor_inline> bt0(stage, dextents<int, 2>{KS, Cols}, array<int, 2>{1, KS});
  auto b0 = bt0.slice<KS, Cols>(0, 0);
  return operation.template get_destination_cooperative_tensor<decltype(a0), decltype(b0), float>();
}
template <class Acc> inline void gguf_zero(thread Acc &acc) {
#pragma unroll
  for (ushort i = 0; i < acc.get_capacity(); ++i) acc[i] = 0.0f;
}
// fn(row, column, value) for every element of a tile's destination.
template <class Acc, class Fn> inline void gguf_elements(thread Acc &acc, Fn fn) {
#pragma unroll
  for (ushort i = 0; i < acc.get_capacity(); ++i) {
    if (!acc.is_valid_element(i)) continue;
    const auto index = acc.get_multidimensional_index(i);
    fn(uint(index[1]), uint(index[0]), float(acc[i]));
  }
}

// The decode tile loop without the store: dequantize one KS-input step of the Cols columns into the stage, then run
// the tile's matmul2d on it, over steps [step_begin, step_end) of K.
template <class F, ushort Rows, ushort Cols, ushort KS, class Acc>
inline void staged_accumulate(device bfloat *input, device uchar *w0, device uchar *w1, device uchar *meta, uint input_size, uint output_origin,
                     threadgroup half *stage, threadgroup half2 *tl, uint simd_lane, uint step_begin, uint step_end, thread Acc &acc) {
  // Prefetch: the steps whose weights are loaded ahead of the one being staged.
  constexpr ushort Prefetch = 1, GPS = KS / 32, Items = Cols * GPS, IPT = (Items + 31) / 32;
  auto a = tensor(input, dextents<int, 2>{int(input_size), Rows}, array<int, 2>{1, int(input_size)});
  constexpr auto descriptor = matmul2d_descriptor(Rows, Cols, KS, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
  [[maybe_unused]] matmul2d<descriptor, execution_simdgroups<1>> operation;
  const uint groups = input_size / 32, units = groups / F::MetaGroups;
  const uint plane_tile = output_origin / QUANT_TILE_ROWS, plane_row = output_origin % QUANT_TILE_ROWS;
  device uchar *tw0 = w0 + (ulong(plane_tile) * groups * QUANT_TILE_ROWS + plane_row) * F::P0;
  device uchar *tw1 = w1 + (ulong(plane_tile) * groups * QUANT_TILE_ROWS + plane_row) * F::P1;
  device uchar *tmeta = meta + (ulong(plane_tile) * units * QUANT_TILE_ROWS + plane_row) * F::MetaBytes;
  tensor<threadgroup half, dextents<int, 2>, tensor_inline> bt0(stage, dextents<int, 2>{KS, Cols}, array<int, 2>{1, KS});
  tensor<threadgroup half, dextents<int, 2>, tensor_inline> bt1(stage + (SPLASH_M5_GGUF_SINGLE_STAGE ? 0 : KS * Cols), dextents<int, 2>{KS, Cols}, array<int, 2>{1, KS});
  auto b0 = bt0.slice<KS, Cols>(0, 0), b1 = bt1.slice<KS, Cols>(0, 0);
  typename F::Payload packed[Prefetch][IPT]; typename F::Meta hdr[IPT]; uint hdr_unit[IPT];
  const uint unit0 = (step_begin * GPS) / F::MetaGroups;
#pragma unroll
  for (ushort it = 0; it < IPT; ++it) {
    const uint item = simd_lane + it * 32; const bool live = item < Items;
    const uint col = live ? item % Cols : 0, gi = live ? item / Cols : 0;
#pragma unroll
    for (ushort pf = 0; pf < Prefetch; ++pf) {
      const ulong g = ulong(step_begin + pf) * GPS + gi;
      if (live && step_begin + pf < step_end) packed[pf][it] = F::load(tw0 + (g * QUANT_TILE_ROWS + col) * F::P0, tw1 + (g * QUANT_TILE_ROWS + col) * F::P1);
    }
    hdr[it] = F::loadMeta(tmeta + (ulong(unit0) * QUANT_TILE_ROWS + col) * F::MetaBytes); hdr_unit[it] = unit0;
  }
#if SPLASH_M5_GGUF_ACT
  auto input_now = operation.template get_left_input_cooperative_tensor<bfloat, half, float>();
  auto input_next = operation.template get_left_input_cooperative_tensor<bfloat, half, float>();
  input_next.load(a.template slice<KS, Rows>(step_begin * KS, 0));
#endif
#if SPLASH_M5_GGUF_APREFETCH
  // Rows x KS bf16 per step: Rows * KS / 8 uint4, Rows / 8 per lane (Rows is 8, 16 or 32).
  static_assert(Rows <= kStagedInputRows && Rows % 8 == 0 && KS == 32, "");
  constexpr ushort AV = Rows * KS / 8 / 32;
  threadgroup bfloat *astage = reinterpret_cast<threadgroup bfloat *>(stage + kStagedBuffers * KS * Cols);
  uint4 next_input[AV];
  const auto load_input = [&](uint step) {
#pragma unroll
    for (ushort v = 0; v < AV; ++v) {
      const uint u = simd_lane + v * 32, row = u / (KS / 8), quad = u % (KS / 8);
      next_input[v] = *reinterpret_cast<device const uint4 *>(input + ulong(row) * input_size + step * KS + quad * 8);
    }
  };
  if (SPLASH_M5_GGUF_APREFETCH == 1) load_input(step_begin);
#endif
  for (uint step = step_begin; step < step_end; ++step) {
    threadgroup half *buf = stage + (SPLASH_M5_GGUF_SINGLE_STAGE ? 0 : (step & 1)) * (KS * Cols);
#if SPLASH_M5_GGUF_APREFETCH
    threadgroup bfloat *abuf = astage + (SPLASH_M5_GGUF_SINGLE_STAGE ? 0 : (step & 1)) * (Rows * KS);
    if (SPLASH_M5_GGUF_APREFETCH == 2) load_input(step);  // G4b: no prefetch
#pragma unroll
    for (ushort v = 0; v < AV; ++v) {
      const uint u = simd_lane + v * 32, row = u / (KS / 8), quad = u % (KS / 8);
      *reinterpret_cast<threadgroup uint4 *>(abuf + row * KS + quad * 8) = next_input[v];
    }
#endif
#pragma unroll
    for (ushort it = 0; it < IPT; ++it) {
      const uint item = simd_lane + it * 32; if (item >= Items) break;
      const uint col = item % Cols, gi = item / Cols, g = step * GPS + gi, unit = g / F::MetaGroups; const ushort j = g % F::MetaGroups;
      if (unit != hdr_unit[it]) { hdr[it] = F::loadMeta(tmeta + (ulong(unit) * QUANT_TILE_ROWS + col) * F::MetaBytes); hdr_unit[it] = unit; }
      dequant32<F>(packed[0][it], hdr[it], j, tl, buf + col * KS + gi * 32);
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
#if SPLASH_M5_GGUF_APREFETCH
    if (SPLASH_M5_GGUF_APREFETCH == 1 && step + 1 < step_end) load_input(step + 1);
#endif
#if SPLASH_M5_GGUF_ACT
    input_now = input_next;
    if (step + 1 < step_end) input_next.load(a.template slice<KS, Rows>((step + 1) * KS, 0));
#endif
    if (step + Prefetch < step_end) {
#pragma unroll
      for (ushort it = 0; it < IPT; ++it) {
        const uint item = simd_lane + it * 32; if (item >= Items) break;
        const uint col = item % Cols, gi = item / Cols; const ulong g = ulong(step + Prefetch) * GPS + gi;
        packed[Prefetch - 1][it] = F::load(tw0 + (g * QUANT_TILE_ROWS + col) * F::P0, tw1 + (g * QUANT_TILE_ROWS + col) * F::P1);
      }
    }
#if SPLASH_M5_GGUF_DIAG == 2
    tensor<threadgroup bfloat, dextents<int, 2>, tensor_inline> at(reinterpret_cast<threadgroup bfloat *>(buf), dextents<int, 2>{KS, Rows}, array<int, 2>{1, KS});
    auto a_slice = at.template slice<KS, Rows>(0, 0);
    if (step & 1) operation.run(a_slice, b1, acc); else operation.run(a_slice, b0, acc);
#elif SPLASH_M5_GGUF_ACT
    if (step & 1) operation.run(input_now, b1, acc); else operation.run(input_now, b0, acc);
#elif SPLASH_M5_GGUF_APREFETCH
    tensor<threadgroup bfloat, dextents<int, 2>, tensor_inline> at(abuf, dextents<int, 2>{KS, Rows}, array<int, 2>{1, KS});
    auto a_slice = at.template slice<KS, Rows>(0, 0);
    if (step & 1) operation.run(a_slice, b1, acc); else operation.run(a_slice, b0, acc);
    if (SPLASH_M5_GGUF_SINGLE_STAGE) simdgroup_barrier(mem_flags::mem_threadgroup);
#else
    auto a_slice = a.template slice<KS, Rows>(step * KS, 0);
    if (SPLASH_M5_GGUF_DIAG != 1) { if (step & 1) operation.run(a_slice, b1, acc); else operation.run(a_slice, b0, acc); }
#endif
    if (SPLASH_M5_GGUF_SINGLE_STAGE) simdgroup_barrier(mem_flags::mem_threadgroup);
  }
  simdgroup_barrier(mem_flags::mem_threadgroup);   // the stage may be reused by a following accumulate
}

// runtime dequantizer selection (uniform per threadgroup): the format's pair table, which every thread of the
// threadgroup fills, then its tile loop
template <ushort Rows, ushort Cols, ushort KS, class Acc>
inline void staged_accumulate_any(uint fmt, device bfloat *input, device uchar *w0, device uchar *w1, device uchar *meta, uint input_size, uint origin,
                           threadgroup half *stage, threadgroup half2 *tl, uint thread_index, uint simd_lane, uint sb, uint se, thread Acc &acc) {
  quant_format_switch(fmt, [&](auto format) {
    typedef decltype(format) F;
    quant_pair_table<F>(tl, thread_index, GGUF_STAGED_THREADS);
    staged_accumulate<F, Rows, Cols, KS>(input, w0, w1, meta, input_size, origin, stage, tl, simd_lane, sb, se, acc);
  });
}
