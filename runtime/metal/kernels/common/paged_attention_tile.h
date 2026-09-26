#pragma once

#include "metal/abi/PagedAttention.h"
#include "metal/kernels/common/q8_paging.h"
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#include <metal_stdlib>

using namespace metal;
using namespace mpp::tensor_ops;

// Paged attention over INT8 or BF16 KV. INT8 has one fp32 scale per
// (token, KV head); BF16 reads the stored values directly. The preceding
// store writes current rows to their final slots; committed_tokens controls
// visibility, and the next command overwrites rejected verify rows.
//
// A threadgroup owns one KV head, query tile and history split. It processes
// the tile's GQA group as M = rows x query heads per KV head. Each split writes
// fp32 partials and statistics for a fixed-order reduce.
//
// Prefill and verify share this device-operand page loop.
// splash-m5 diagnostic (timing only; outputs are wrong when nonzero):
// 1 skips softmax, 2 skips PV, 3 skips QK, 4 skips softmax and its two barriers.
// splash-m5 A4: simdgroups that run QK (its 48 x 32 output is six 16 x 16 tiles).
#ifndef SPLASH_M5_ATTN_QK_SG
#define SPLASH_M5_ATTN_QK_SG 8
#endif
// splash-m5 A6: QK as int8 x int8 -> int32. The tile's queries are quantized once
// per threadgroup, one symmetric scale per fused row; scores = int32 * row scale
// (the key scales stay where the placement puts them). 0 keeps bf16 x int8.
#ifndef SPLASH_M5_ATTN_QK_INT8
#define SPLASH_M5_ATTN_QK_INT8 0
#endif
// splash-m5 A5: QK with relaxed precision (PV already has it).
#ifndef SPLASH_M5_ATTN_QK_RELAXED
#define SPLASH_M5_ATTN_QK_RELAXED false
#endif
#ifndef SPLASH_M5_ATTN_DIAG
#define SPLASH_M5_ATTN_DIAG 0
#endif
constant uint SplashVerifyMaximumSplits =
    SPLASH_VERIFY_ATTENTION_MAXIMUM_SPLITS;
constant uint SplashPrefillTileRows = SPLASH_PREFILL_ATTENTION_TILE_ROWS;
constant uint SplashPrefillMaximumSplits =
    SPLASH_PREFILL_ATTENTION_MAXIMUM_SPLITS;

inline bool splash_q8_prefill_attention_contract_valid(
    constant SplashQ8PrefillAttentionParams &params) {
  return params.rows > 0 && params.rows <= SPLASH_PREFILL_TOKEN_BUDGET &&
         params.chunk_stride >= params.rows &&
         params.chunk_stride <= SPLASH_PREFILL_TOKEN_BUDGET &&
         params.chunk_stride % SPLASH_TARGET_KV_BLOCK_TOKENS == 0 &&
         params.page_table_entries >=
             (params.committed_tokens + params.rows +
              SplashQ8PageTokens - 1) /
                 SplashQ8PageTokens &&
         params.physical_page_count > 0 && params.split_count > 0 &&
         params.split_count <= SplashPrefillMaximumSplits &&
         ulong(params.committed_tokens) + params.rows <=
             ulong(SPLASH_MAXIMUM_PHYSICAL_KV_TOKENS) &&
         params.reserved0 == 0 && params.reserved1 == 0;
}

inline bool splash_q8_verify_attention_contract_valid(
    constant SplashQ8VerifyAttentionParams &params) {
  return params.active_rows > 0 &&
         params.active_rows <= SPLASH_TARGET_VERIFY_ROWS &&
         params.chunk_stride % SPLASH_TARGET_KV_BLOCK_TOKENS == 0 &&
         params.chunk_stride >= params.active_rows &&
         params.page_table_entries >=
             (params.committed_tokens + params.active_rows +
              SplashQ8PageTokens - 1) /
                 SplashQ8PageTokens &&
         params.physical_page_count > 0 &&
         ulong(params.committed_tokens) + params.active_rows <=
             ulong(SPLASH_MAXIMUM_PHYSICAL_KV_TOKENS) &&
         params.split_count > 0 &&
         params.split_count <= SplashVerifyMaximumSplits &&
         params.slot_splits >= params.split_count &&
         params.slot_splits <= SplashVerifyMaximumSplits &&
         params.reserved2 == 0;
}

// Split and reduce derive the same balanced partition of each query tile's
// visible pages. Causal masking remains per query row inside each split.
inline uint splash_attention_pages(uint visible_tokens) {
  return (visible_tokens + SplashQ8PageTokens - 1) / SplashQ8PageTokens;
}

inline uint splash_attention_pages_per_split(uint pages, uint splits) {
  return (pages + splits - 1) / splits;
}

// One Page32 block of one tile: key-scaled scores become value-scaled bf16
// probabilities and the row statistics advance. Four lanes own one fused row,
// eight consecutive tokens each, so the row maximum and sum are two xor
// shuffles instead of a 32-lane reduction per row and the eight bf16
// probabilities leave as one 16-byte store. Threads past 4 x FusedRows
// (kv4_g6: simdgroups 6 and 7) only take part in the caller's barriers.
// The float4 loads need 16-byte aligned slabs (alignas in the entries) and
// the page's 32 key and value scales as float4 vectors, in device memory
// (splash_q8_scale_index is a multiple of 32 floats and the bound scale
// buffers start at their placement-aligned base).
// Rows whose running maximum grew atomically set the shared boolean rescale
// flag. Existing threadgroup barriers separate reset, concurrent set, and
// read; relaxed atomics make the same-value writes safe without changing
// arithmetic.
template <uint QueryHeadsPerKVHead, uint RowsPerTile, bool ScaleInSoftmax,
          bool Quantized>
inline void splash_attention_page_softmax(
    threadgroup const float *scores, threadgroup bfloat *probabilities,
    threadgroup float *row_max, threadgroup float *row_sum,
    threadgroup float *previous_scale, threadgroup atomic_uint *rescale,
    device const float4 *key_scales, device const float4 *value_scales,
    uint token_start,
    uint visible_tokens, uint committed_tokens, uint active_rows,
    uint thread_index) {
  constexpr uint N = SplashQ8PageTokens;
  constexpr uint FusedRows = RowsPerTile * QueryHeadsPerKVHead;
  constexpr uint TokensPerLane = 8;
  constexpr uint LanesPerRow = N / TokensPerLane;
  static_assert(N == 32, "four lanes of eight tokens span one page");
  static_assert(LanesPerRow * FusedRows <= 256,
                "one softmax lane per thread of the 256-thread tile");
  if (thread_index >= LanesPerRow * FusedRows)
    return;
  const uint fused_row = thread_index / LanesPerRow;
  const uint column = thread_index % LanesPerRow * TokensPerLane;
  const uint query_row = fused_row / QueryHeadsPerKVHead;
  const uint causal_end =
      committed_tokens + min(query_row, active_rows - 1) + 1;
  const uint limit = min(visible_tokens, causal_end);
  const uint token = token_start + column;
  threadgroup const float4 *scores4 =
      reinterpret_cast<threadgroup const float4 *>(scores + fused_row * N +
                                                   column);
  const uint vector = column / 4;
  float score[TokensPerLane];
  {
    float4 low = scores4[0], high = scores4[1];
    if constexpr (Quantized && ScaleInSoftmax) {
      low *= key_scales[vector];
      high *= key_scales[vector + 1];
    }
    low *= 0.0625f;
    high *= 0.0625f;
    score[0] = low.x, score[1] = low.y, score[2] = low.z, score[3] = low.w;
    score[4] = high.x, score[5] = high.y, score[6] = high.z, score[7] = high.w;
  }
  float local_max = -INFINITY;
#pragma unroll
  for (uint j = 0; j < TokensPerLane; ++j) {
    score[j] = token + j < limit ? score[j] : -INFINITY;
    local_max = max(local_max, score[j]);
  }
  local_max = max(local_max, simd_shuffle_xor(local_max, 1));
  local_max = max(local_max, simd_shuffle_xor(local_max, 2));
  const float previous_max = row_max[fused_row];
  const float next_max = max(previous_max, local_max);
  float probability[TokensPerLane];
  float local_sum = 0.0f;
#pragma unroll
  for (uint j = 0; j < TokensPerLane; ++j) {
    probability[j] = token + j < limit ? fast::exp(score[j] - next_max) : 0.0f;
    local_sum += probability[j];
  }
  local_sum += simd_shuffle_xor(local_sum, 1);
  local_sum += simd_shuffle_xor(local_sum, 2);
  if (column == 0) {
    const float scale = next_max == -INFINITY || next_max == previous_max
                            ? 1.0f
                            : fast::exp(previous_max - next_max);
    previous_scale[fused_row] = scale;
    row_sum[fused_row] = row_sum[fused_row] * scale + local_sum;
    row_max[fused_row] = next_max;
    if (scale != 1.0f)
      atomic_store_explicit(rescale, 1u, memory_order_relaxed);
  }
  // Masked tokens stay exactly zero whatever their stored value scale holds.
  float4 low_scales(1.0f), high_scales(1.0f);
  if constexpr (Quantized) {
    low_scales = value_scales[vector];
    high_scales = value_scales[vector + 1];
  }
  const float4 low(token + 0 < limit ? probability[0] * low_scales.x : 0.0f,
                   token + 1 < limit ? probability[1] * low_scales.y : 0.0f,
                   token + 2 < limit ? probability[2] * low_scales.z : 0.0f,
                   token + 3 < limit ? probability[3] * low_scales.w : 0.0f);
  const float4 high(token + 4 < limit ? probability[4] * high_scales.x : 0.0f,
                    token + 5 < limit ? probability[5] * high_scales.y : 0.0f,
                    token + 6 < limit ? probability[6] * high_scales.z : 0.0f,
                    token + 7 < limit ? probability[7] * high_scales.w : 0.0f);
  threadgroup bfloat4 *probabilities4 = reinterpret_cast<threadgroup bfloat4 *>(
      probabilities + fused_row * N + column);
  probabilities4[0] = bfloat4(low);
  probabilities4[1] = bfloat4(high);
}

// One tile over its split's pages with the INT8 K/V and the scales consumed
// in place as device tensor operands: the prefill tile, and the verify tile
// in its direct placement, which differs only in RowsPerTile. Queries are
// KV-head-major [kv head][row][query head in group][dimension], so the tile's
// fused rows form one contiguous M x D tensor. Three barriers per page order
// the score store, the softmax and the probability reads of PV.
template <uint KVHeads, uint QueryHeadsPerKVHead, uint RowsPerTile,
          bool ScaleInSoftmax, typename CacheElement>
inline void splash_paged_attention_tile(
    device bfloat *tile_queries, device CacheElement *cache_keys,
    device const float *key_scales_buffer, device CacheElement *cache_values,
    device const float *value_scales_buffer, device const uint *page_table,
    uint kv_head, uint committed_tokens, uint active_rows, uint splits,
    uint split, device float *partials, device float *statistics, ulong slot,
    threadgroup float *scores, threadgroup bfloat *probabilities,
    threadgroup float *row_max, threadgroup float *row_sum,
    threadgroup float *previous_scale, threadgroup atomic_uint *rescale,
    uint thread_index, threadgroup int8_t *query_stage = nullptr,
    threadgroup float *query_scale = nullptr) {
  constexpr ushort M = RowsPerTile * QueryHeadsPerKVHead;
  constexpr bool Quantized = is_same<CacheElement, int8_t>::value;
  constexpr ushort N = SplashQ8PageTokens;
  constexpr ushort D = SplashQ8HeadDimension;
  uint visible_tokens = committed_tokens + active_rows;
  uint pages = splash_attention_pages(visible_tokens);
  uint per_split = splash_attention_pages_per_split(pages, splits);
  uint page_begin = split * per_split;
  if (page_begin >= pages)
    return;
  uint page_end = min(pages, page_begin + per_split);
  if (thread_index < M) {
    row_max[thread_index] = -INFINITY;
    row_sum[thread_index] = 0.0f;
  }
#if SPLASH_M5_ATTN_QK_INT8
  if constexpr (Quantized) {
    // Simdgroup g quantizes rows g, g + 8, ...; lane l owns 8 of the row's 256.
    static_assert(D == 256, "");
    const uint lane = thread_index % 32, group = thread_index / 32;
    for (uint row = group; row < M; row += 8) {
      float values[8];
      float largest = 0.0f;
#pragma unroll
      for (uint j = 0; j < 8; ++j) {
        values[j] = float(tile_queries[row * D + lane * 8 + j]);
        largest = max(largest, abs(values[j]));
      }
      largest = simd_max(largest);
      const float scale = largest > 0.0f ? largest / 127.0f : 1.0f;
      const float inverse = 1.0f / scale;
#pragma unroll
      for (uint j = 0; j < 8; ++j)
        query_stage[row * D + lane * 8 + j] =
            int8_t(clamp(rint(values[j] * inverse), -127.0f, 127.0f));
      if (lane == 0)
        query_scale[row] = scale;
    }
  }
#endif
  threadgroup_barrier(mem_flags::mem_threadgroup);

  auto qt = tensor(tile_queries, dextents<int, 2>{D, M}, array<int, 2>{1, D});
  auto st = tensor(scores, dextents<int, 2>{N, M}, array<int, 2>{1, N});
  auto pt = tensor(probabilities, dextents<int, 2>{N, M}, array<int, 2>{1, N});
  auto p0 = pt.slice<N, M>(0, 0);
  auto key_type = tensor(cache_keys, dextents<int, 2>{D, N}, array<int, 2>{1, D});
  auto value_type =
      tensor(cache_values, dextents<int, 2>{N, D}, array<int, 2>{1, N});
  auto q0 = qt.slice<D, M>(0, 0);
  auto k0 = key_type.template slice<D, N>(0, 0);
  auto v0 = value_type.template slice<N, D>(0, 0);
  // QK writes a complete page score tile; PV accumulates the running output.
  // Key scaling is a precompiled placement choice; both use the same QK/PV.
  constexpr auto qk_descriptor =
      matmul2d_descriptor(M, N, D, false, true, SPLASH_M5_ATTN_QK_RELAXED,
                          matmul2d_descriptor::mode::multiply);
  constexpr auto pv_descriptor =
      matmul2d_descriptor(M, D, N, false, true, true,
                          matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<qk_descriptor, execution_simdgroups<SPLASH_M5_ATTN_QK_SG>> qk;
  matmul2d<pv_descriptor, execution_simdgroups<8>> pv;
  auto running = pv.template get_destination_cooperative_tensor<
      decltype(p0), decltype(v0), float>();
  // Uniform capacity partitions the logical tile across eight 32-lane groups.
  // Equality proves no padding; otherwise only valid entries may be accessed.
  const bool running_full =
      uint(running.get_capacity()) * (8u * 32u) == uint(M) * D;
#pragma unroll
  for (ushort index = 0; index < running.get_capacity(); ++index) {
    if (running_full || running.is_valid_element(index))
      running[index] = 0.0f;
  }

  for (uint page = page_begin; page < page_end; ++page) {
    uint physical = page_table[page];
    uint token_start = page * N;
    auto kt = tensor(
        cache_keys + splash_q8_key_index<KVHeads>(physical, kv_head, 0, 0),
        dextents<int, 2>{D, N}, array<int, 2>{1, D});
    auto vt = tensor(
        cache_values + splash_q8_value_index<KVHeads>(physical, kv_head, 0, 0),
        dextents<int, 2>{N, D}, array<int, 2>{1, N});
    device const float *key_scales = nullptr;
    device const float *value_scales = nullptr;
    if constexpr (Quantized) {
      ulong scale_index = splash_q8_scale_index<KVHeads>(physical, kv_head, 0);
      key_scales = key_scales_buffer + scale_index;
      value_scales = value_scales_buffer + scale_index;
    }

    auto ks = kt.template slice<D, N>(0, 0);
    const bool qk_group = SPLASH_M5_ATTN_QK_SG == 8 || thread_index / 32 < SPLASH_M5_ATTN_QK_SG;
    if constexpr (SPLASH_M5_ATTN_QK_INT8 && Quantized) {
      auto q8t = tensor(query_stage, dextents<int, 2>{D, M}, array<int, 2>{1, D});
      auto q8 = q8t.template slice<D, M>(0, 0);
      auto integer_scores = qk.template get_destination_cooperative_tensor<
          decltype(q8), decltype(k0), int32_t>();
      if (qk_group) {
        qk.run(q8, ks, integer_scores);
        const bool integers_full = uint(integer_scores.get_capacity()) *
            (SPLASH_M5_ATTN_QK_SG * 32u) == uint(M) * N;
#pragma unroll
        for (ushort index = 0; index < integer_scores.get_capacity(); ++index) {
          if (!integers_full && !integer_scores.is_valid_element(index))
            continue;
          const auto coordinates = integer_scores.get_multidimensional_index(index);
          float value = float(integer_scores[index]) * query_scale[coordinates[1]];
          if constexpr (!ScaleInSoftmax)
            value *= key_scales[coordinates[0]];
          scores[coordinates[1] * N + coordinates[0]] = value;
        }
      }
    } else {
      auto page_scores = qk.template get_destination_cooperative_tensor<
          decltype(q0), decltype(k0), float>();
      // One full-dimension product initializes the score CT through MPP.
      if (SPLASH_M5_ATTN_DIAG != 3 && qk_group) qk.run(q0, ks, page_scores);
      if constexpr (Quantized && !ScaleInSoftmax) {
        const bool scores_full =
            uint(page_scores.get_capacity()) * (8u * 32u) == uint(M) * N;
#pragma unroll
        for (ushort index = 0; index < page_scores.get_capacity(); ++index) {
          if (!scores_full && !page_scores.is_valid_element(index))
            continue;
          const auto coordinates = page_scores.get_multidimensional_index(index);
          page_scores[index] *= key_scales[coordinates[0]];
        }
      }
      if (qk_group)
        page_scores.store(st.slice<N, M>(0, 0));
    }
    if (thread_index == 0)
      atomic_store_explicit(rescale, 0u, memory_order_relaxed);
    if (SPLASH_M5_ATTN_DIAG != 4) threadgroup_barrier(mem_flags::mem_threadgroup);
    if (SPLASH_M5_ATTN_DIAG != 1 && SPLASH_M5_ATTN_DIAG != 4)
    splash_attention_page_softmax<QueryHeadsPerKVHead, RowsPerTile,
                             ScaleInSoftmax, Quantized>(
        scores, probabilities, row_max, row_sum, previous_scale, rescale,
        reinterpret_cast<device const float4 *>(key_scales),
        reinterpret_cast<device const float4 *>(value_scales), token_start,
        visible_tokens, committed_tokens, active_rows, thread_index);
    if (SPLASH_M5_ATTN_DIAG != 4) threadgroup_barrier(mem_flags::mem_threadgroup);
    if (atomic_load_explicit(rescale, memory_order_relaxed)) {
#pragma unroll
      for (ushort index = 0; index < running.get_capacity(); ++index) {
        if (!running_full && !running.is_valid_element(index))
          continue;
        auto coordinates = running.get_multidimensional_index(index);
        running[index] *= previous_scale[coordinates[1]];
      }
    }
    auto vs = vt.template slice<N, D>(0, 0);
    if (SPLASH_M5_ATTN_DIAG != 2) pv.run(p0, vs, running);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  auto target = tensor(partials + slot * M * D, dextents<int, 2>{D, M},
                       array<int, 2>{1, D});
  running.store(target.slice<D, M>(0, 0));
  if (thread_index < M) {
    statistics[(slot * M + thread_index) * 2] = row_max[thread_index];
    statistics[(slot * M + thread_index) * 2 + 1] = row_sum[thread_index];
  }
}

// splash-m5 A2: two pages per iteration. Both pages' QK products are issued
// before the score barrier (twice the K bytes in flight per threadgroup) and a
// 64-token online softmax halves the barriers per token. Each softmax lane owns
// 16 tokens inside one page; tokens past the split's own pages are masked.
template <uint QueryHeadsPerKVHead, uint RowsPerTile, bool ScaleInSoftmax>
inline void splash_attention_page_softmax2(
    threadgroup const float *scores, threadgroup bfloat *probabilities,
    threadgroup float *row_max, threadgroup float *row_sum,
    threadgroup float *previous_scale, threadgroup atomic_uint *rescale,
    device const float4 *key_scales_a, device const float4 *value_scales_a,
    device const float4 *key_scales_b, device const float4 *value_scales_b,
    bool has_b, uint token_start, uint split_end_tokens,
    uint visible_tokens, uint committed_tokens, uint active_rows,
    uint thread_index) {
  constexpr uint N2 = 2 * SplashQ8PageTokens;
  constexpr uint FusedRows = RowsPerTile * QueryHeadsPerKVHead;
  constexpr uint TokensPerLane = 16;
  constexpr uint LanesPerRow = N2 / TokensPerLane;
  static_assert(LanesPerRow == 4 && LanesPerRow * FusedRows <= 256, "");
  if (thread_index >= LanesPerRow * FusedRows)
    return;
  const uint fused_row = thread_index / LanesPerRow;
  const uint column = thread_index % LanesPerRow * TokensPerLane;
  const bool second = column >= SplashQ8PageTokens;
  const uint query_row = fused_row / QueryHeadsPerKVHead;
  const uint causal_end =
      committed_tokens + min(query_row, active_rows - 1) + 1;
  const uint limit = min(min(visible_tokens, causal_end), split_end_tokens);
  const uint token = token_start + column;
  const bool page_valid = !second || has_b;
  device const float4 *key_scales = second ? key_scales_b : key_scales_a;
  device const float4 *value_scales = second ? value_scales_b : value_scales_a;
  const uint vector = (column % SplashQ8PageTokens) / 4;
  threadgroup const float4 *scores4 =
      reinterpret_cast<threadgroup const float4 *>(scores + fused_row * N2 + column);
  float score[TokensPerLane];
#pragma unroll
  for (uint v = 0; v < 4; ++v) {
    float4 value = scores4[v];
    if constexpr (ScaleInSoftmax)
      value *= page_valid ? key_scales[vector + v] : float4(1.0f);
    value *= 0.0625f;
    score[4 * v + 0] = value.x, score[4 * v + 1] = value.y;
    score[4 * v + 2] = value.z, score[4 * v + 3] = value.w;
  }
  float local_max = -INFINITY;
#pragma unroll
  for (uint j = 0; j < TokensPerLane; ++j) {
    score[j] = token + j < limit ? score[j] : -INFINITY;
    local_max = max(local_max, score[j]);
  }
  local_max = max(local_max, simd_shuffle_xor(local_max, 1));
  local_max = max(local_max, simd_shuffle_xor(local_max, 2));
  const float previous_max = row_max[fused_row];
  const float next_max = max(previous_max, local_max);
  float probability[TokensPerLane];
  float local_sum = 0.0f;
#pragma unroll
  for (uint j = 0; j < TokensPerLane; ++j) {
    probability[j] = token + j < limit ? fast::exp(score[j] - next_max) : 0.0f;
    local_sum += probability[j];
  }
  local_sum += simd_shuffle_xor(local_sum, 1);
  local_sum += simd_shuffle_xor(local_sum, 2);
  if (column == 0) {
    const float scale = next_max == -INFINITY || next_max == previous_max
                            ? 1.0f
                            : fast::exp(previous_max - next_max);
    previous_scale[fused_row] = scale;
    row_sum[fused_row] = row_sum[fused_row] * scale + local_sum;
    row_max[fused_row] = next_max;
    if (scale != 1.0f)
      atomic_store_explicit(rescale, 1u, memory_order_relaxed);
  }
  threadgroup bfloat4 *probabilities4 = reinterpret_cast<threadgroup bfloat4 *>(
      probabilities + fused_row * N2 + column);
#pragma unroll
  for (uint v = 0; v < 4; ++v) {
    const float4 scales = page_valid ? value_scales[vector + v] : float4(1.0f);
    const uint t = token + 4 * v;
    probabilities4[v] = bfloat4(float4(t + 0 < limit ? probability[4 * v + 0] * scales.x : 0.0f,
                                       t + 1 < limit ? probability[4 * v + 1] * scales.y : 0.0f,
                                       t + 2 < limit ? probability[4 * v + 2] * scales.z : 0.0f,
                                       t + 3 < limit ? probability[4 * v + 3] * scales.w : 0.0f));
  }
}

template <uint KVHeads, uint QueryHeadsPerKVHead, uint RowsPerTile,
          bool ScaleInSoftmax>
inline void splash_paged_attention_tile2(
    device bfloat *tile_queries, device int8_t *cache_keys,
    device const float *key_scales_buffer, device int8_t *cache_values,
    device const float *value_scales_buffer, device const uint *page_table,
    uint kv_head, uint committed_tokens, uint active_rows, uint splits,
    uint split, device float *partials, device float *statistics, ulong slot,
    threadgroup float *scores, threadgroup bfloat *probabilities,
    threadgroup float *row_max, threadgroup float *row_sum,
    threadgroup float *previous_scale, threadgroup atomic_uint *rescale,
    uint thread_index) {
  constexpr ushort M = RowsPerTile * QueryHeadsPerKVHead;
  constexpr ushort N = SplashQ8PageTokens;
  constexpr ushort N2 = 2 * N;
  constexpr ushort D = SplashQ8HeadDimension;
  uint visible_tokens = committed_tokens + active_rows;
  uint pages = splash_attention_pages(visible_tokens);
  uint per_split = splash_attention_pages_per_split(pages, splits);
  uint page_begin = split * per_split;
  if (page_begin >= pages)
    return;
  uint page_end = min(pages, page_begin + per_split);
  if (thread_index < M) {
    row_max[thread_index] = -INFINITY;
    row_sum[thread_index] = 0.0f;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  auto qt = tensor(tile_queries, dextents<int, 2>{D, M}, array<int, 2>{1, D});
  auto st = tensor(scores, dextents<int, 2>{N2, M}, array<int, 2>{1, N2});
  auto pt = tensor(probabilities, dextents<int, 2>{N2, M}, array<int, 2>{1, N2});
  auto p0 = pt.slice<N, M>(0, 0);
  auto key_type = tensor(cache_keys, dextents<int, 2>{D, N}, array<int, 2>{1, D});
  auto value_type =
      tensor(cache_values, dextents<int, 2>{N, D}, array<int, 2>{1, N});
  auto q0 = qt.slice<D, M>(0, 0);
  auto k0 = key_type.template slice<D, N>(0, 0);
  auto v0 = value_type.template slice<N, D>(0, 0);
  constexpr auto qk_descriptor =
      matmul2d_descriptor(M, N, D, false, true, false,
                          matmul2d_descriptor::mode::multiply);
  constexpr auto pv_descriptor =
      matmul2d_descriptor(M, D, N, false, true, true,
                          matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<qk_descriptor, execution_simdgroups<8>> qk;
  matmul2d<pv_descriptor, execution_simdgroups<8>> pv;
  auto running = pv.template get_destination_cooperative_tensor<
      decltype(p0), decltype(v0), float>();
  const bool running_full =
      uint(running.get_capacity()) * (8u * 32u) == uint(M) * D;
#pragma unroll
  for (ushort index = 0; index < running.get_capacity(); ++index) {
    if (running_full || running.is_valid_element(index))
      running[index] = 0.0f;
  }

  for (uint page = page_begin; page < page_end; page += 2) {
    const bool has_b = page + 1 < page_end;
    const uint physical_a = page_table[page];
    const uint physical_b = has_b ? page_table[page + 1] : physical_a;
    auto kt_a = tensor(cache_keys + splash_q8_key_index<KVHeads>(physical_a, kv_head, 0, 0),
                       dextents<int, 2>{D, N}, array<int, 2>{1, D});
    auto kt_b = tensor(cache_keys + splash_q8_key_index<KVHeads>(physical_b, kv_head, 0, 0),
                       dextents<int, 2>{D, N}, array<int, 2>{1, D});
    auto vt_a = tensor(cache_values + splash_q8_value_index<KVHeads>(physical_a, kv_head, 0, 0),
                       dextents<int, 2>{N, D}, array<int, 2>{1, N});
    auto vt_b = tensor(cache_values + splash_q8_value_index<KVHeads>(physical_b, kv_head, 0, 0),
                       dextents<int, 2>{N, D}, array<int, 2>{1, N});
    device const float *key_scales_a =
        key_scales_buffer + splash_q8_scale_index<KVHeads>(physical_a, kv_head, 0);
    device const float *key_scales_b =
        key_scales_buffer + splash_q8_scale_index<KVHeads>(physical_b, kv_head, 0);
    device const float *value_scales_a =
        value_scales_buffer + splash_q8_scale_index<KVHeads>(physical_a, kv_head, 0);
    device const float *value_scales_b =
        value_scales_buffer + splash_q8_scale_index<KVHeads>(physical_b, kv_head, 0);

    auto scores_a = qk.template get_destination_cooperative_tensor<
        decltype(q0), decltype(k0), float>();
    auto scores_b = qk.template get_destination_cooperative_tensor<
        decltype(q0), decltype(k0), float>();
    auto ks_a = kt_a.template slice<D, N>(0, 0);
    auto ks_b = kt_b.template slice<D, N>(0, 0);
    qk.run(q0, ks_a, scores_a);
    if (has_b)
      qk.run(q0, ks_b, scores_b);
    if constexpr (!ScaleInSoftmax) {
      const bool scores_full =
          uint(scores_a.get_capacity()) * (8u * 32u) == uint(M) * N;
#pragma unroll
      for (ushort index = 0; index < scores_a.get_capacity(); ++index) {
        if (!scores_full && !scores_a.is_valid_element(index))
          continue;
        const auto coordinates = scores_a.get_multidimensional_index(index);
        scores_a[index] *= key_scales_a[coordinates[0]];
        if (has_b) scores_b[index] *= key_scales_b[coordinates[0]];
      }
    }
    scores_a.store(st.slice<N, M>(0, 0));
    if (has_b)
      scores_b.store(st.slice<N, M>(N, 0));
    if (thread_index == 0)
      atomic_store_explicit(rescale, 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    splash_attention_page_softmax2<QueryHeadsPerKVHead, RowsPerTile, ScaleInSoftmax>(
        scores, probabilities, row_max, row_sum, previous_scale, rescale,
        reinterpret_cast<device const float4 *>(key_scales_a),
        reinterpret_cast<device const float4 *>(value_scales_a),
        reinterpret_cast<device const float4 *>(key_scales_b),
        reinterpret_cast<device const float4 *>(value_scales_b), has_b,
        page * N, page_end * N, visible_tokens, committed_tokens, active_rows,
        thread_index);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (atomic_load_explicit(rescale, memory_order_relaxed)) {
#pragma unroll
      for (ushort index = 0; index < running.get_capacity(); ++index) {
        if (!running_full && !running.is_valid_element(index))
          continue;
        auto coordinates = running.get_multidimensional_index(index);
        running[index] *= previous_scale[coordinates[1]];
      }
    }
    auto p_a = pt.slice<N, M>(0, 0);
    auto p_b = pt.slice<N, M>(N, 0);
    auto vs_a = vt_a.template slice<N, D>(0, 0);
    auto vs_b = vt_b.template slice<N, D>(0, 0);
    pv.run(p_a, vs_a, running);
    if (has_b)
      pv.run(p_b, vs_b, running);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  auto target = tensor(partials + slot * M * D, dextents<int, 2>{D, M},
                       array<int, 2>{1, D});
  running.store(target.slice<D, M>(0, 0));
  if (thread_index < M) {
    statistics[(slot * M + thread_index) * 2] = row_max[thread_index];
    statistics[(slot * M + thread_index) * 2 + 1] = row_sum[thread_index];
  }
}

// splash-m5 A3: register prefetch of the next page's INT8 K and V, with the
// current page's QK and PV reading threadgroup copies. Each thread issues its
// 2 x 16 bytes of K and V for page p+1 before page p's math and writes them
// behind barriers the loop already has: K after the score barrier (QK is done),
// V after the PV barrier. No barrier is added.
template <uint KVHeads, uint QueryHeadsPerKVHead, uint RowsPerTile,
          bool ScaleInSoftmax>
inline void splash_paged_attention_tile_staged(
    device bfloat *tile_queries, device int8_t *cache_keys,
    device const float *key_scales_buffer, device int8_t *cache_values,
    device const float *value_scales_buffer, device const uint *page_table,
    uint kv_head, uint committed_tokens, uint active_rows, uint splits,
    uint split, device float *partials, device float *statistics, ulong slot,
    threadgroup float *scores, threadgroup bfloat *probabilities,
    threadgroup float *row_max, threadgroup float *row_sum,
    threadgroup float *previous_scale, threadgroup atomic_uint *rescale,
    threadgroup int8_t *key_stage, threadgroup int8_t *value_stage,
    uint thread_index) {
  constexpr ushort M = RowsPerTile * QueryHeadsPerKVHead;
  constexpr ushort N = SplashQ8PageTokens;
  constexpr ushort D = SplashQ8HeadDimension;
  constexpr uint Vectors = N * D / 16 / 256;  // uint4 per thread per operand
  static_assert(Vectors * 16 * 256 == N * D, "");
  uint visible_tokens = committed_tokens + active_rows;
  uint pages = splash_attention_pages(visible_tokens);
  uint per_split = splash_attention_pages_per_split(pages, splits);
  uint page_begin = split * per_split;
  if (page_begin >= pages)
    return;
  uint page_end = min(pages, page_begin + per_split);
  if (thread_index < M) {
    row_max[thread_index] = -INFINITY;
    row_sum[thread_index] = 0.0f;
  }
  threadgroup uint4 *key_stage4 = reinterpret_cast<threadgroup uint4 *>(key_stage);
  threadgroup uint4 *value_stage4 = reinterpret_cast<threadgroup uint4 *>(value_stage);
  uint4 next_key[Vectors], next_value[Vectors];
  const auto fetch = [&](uint page) {
    const uint physical = page_table[page];
    device const uint4 *k4 = reinterpret_cast<device const uint4 *>(
        cache_keys + splash_q8_key_index<KVHeads>(physical, kv_head, 0, 0));
    device const uint4 *v4 = reinterpret_cast<device const uint4 *>(
        cache_values + splash_q8_value_index<KVHeads>(physical, kv_head, 0, 0));
#pragma unroll
    for (uint i = 0; i < Vectors; ++i) {
      next_key[i] = k4[i * 256 + thread_index];
      next_value[i] = v4[i * 256 + thread_index];
    }
  };
  fetch(page_begin);
#pragma unroll
  for (uint i = 0; i < Vectors; ++i) {
    key_stage4[i * 256 + thread_index] = next_key[i];
    value_stage4[i * 256 + thread_index] = next_value[i];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  auto qt = tensor(tile_queries, dextents<int, 2>{D, M}, array<int, 2>{1, D});
  auto st = tensor(scores, dextents<int, 2>{N, M}, array<int, 2>{1, N});
  auto pt = tensor(probabilities, dextents<int, 2>{N, M}, array<int, 2>{1, N});
  auto p0 = pt.slice<N, M>(0, 0);
  auto kt = tensor(key_stage, dextents<int, 2>{D, N}, array<int, 2>{1, D});
  auto vt = tensor(value_stage, dextents<int, 2>{N, D}, array<int, 2>{1, N});
  auto q0 = qt.slice<D, M>(0, 0);
  auto k0 = kt.template slice<D, N>(0, 0);
  auto v0 = vt.template slice<N, D>(0, 0);
  constexpr auto qk_descriptor =
      matmul2d_descriptor(M, N, D, false, true, false,
                          matmul2d_descriptor::mode::multiply);
  constexpr auto pv_descriptor =
      matmul2d_descriptor(M, D, N, false, true, true,
                          matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<qk_descriptor, execution_simdgroups<8>> qk;
  matmul2d<pv_descriptor, execution_simdgroups<8>> pv;
  auto running = pv.template get_destination_cooperative_tensor<
      decltype(p0), decltype(v0), float>();
  const bool running_full =
      uint(running.get_capacity()) * (8u * 32u) == uint(M) * D;
#pragma unroll
  for (ushort index = 0; index < running.get_capacity(); ++index) {
    if (running_full || running.is_valid_element(index))
      running[index] = 0.0f;
  }

  for (uint page = page_begin; page < page_end; ++page) {
    const bool has_next = page + 1 < page_end;
    if (has_next)
      fetch(page + 1);
    const uint physical = page_table[page];
    const ulong scale_index = splash_q8_scale_index<KVHeads>(physical, kv_head, 0);
    device const float *key_scales = key_scales_buffer + scale_index;
    device const float *value_scales = value_scales_buffer + scale_index;

    auto page_scores = qk.template get_destination_cooperative_tensor<
        decltype(q0), decltype(k0), float>();
    qk.run(q0, k0, page_scores);
    if constexpr (!ScaleInSoftmax) {
      const bool scores_full =
          uint(page_scores.get_capacity()) * (8u * 32u) == uint(M) * N;
#pragma unroll
      for (ushort index = 0; index < page_scores.get_capacity(); ++index) {
        if (!scores_full && !page_scores.is_valid_element(index))
          continue;
        const auto coordinates = page_scores.get_multidimensional_index(index);
        page_scores[index] *= key_scales[coordinates[0]];
      }
    }
    page_scores.store(st.slice<N, M>(0, 0));
    if (thread_index == 0)
      atomic_store_explicit(rescale, 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // Every simdgroup has finished QK: the key stage takes page + 1.
    if (has_next) {
#pragma unroll
      for (uint i = 0; i < Vectors; ++i)
        key_stage4[i * 256 + thread_index] = next_key[i];
    }
    splash_attention_page_softmax<QueryHeadsPerKVHead, RowsPerTile,
                                  ScaleInSoftmax, true>(
        scores, probabilities, row_max, row_sum, previous_scale, rescale,
        reinterpret_cast<device const float4 *>(key_scales),
        reinterpret_cast<device const float4 *>(value_scales), page * N,
        visible_tokens, committed_tokens, active_rows, thread_index);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (atomic_load_explicit(rescale, memory_order_relaxed)) {
#pragma unroll
      for (ushort index = 0; index < running.get_capacity(); ++index) {
        if (!running_full && !running.is_valid_element(index))
          continue;
        auto coordinates = running.get_multidimensional_index(index);
        running[index] *= previous_scale[coordinates[1]];
      }
    }
    pv.run(p0, v0, running);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // Every simdgroup has finished PV: the value stage takes page + 1, visible
    // to the next PV after the next iteration's two barriers.
    if (has_next) {
#pragma unroll
      for (uint i = 0; i < Vectors; ++i)
        value_stage4[i * 256 + thread_index] = next_value[i];
    }
  }
  auto target = tensor(partials + slot * M * D, dextents<int, 2>{D, M},
                       array<int, 2>{1, D});
  running.store(target.slice<D, M>(0, 0));
  if (thread_index < M) {
    statistics[(slot * M + thread_index) * 2] = row_max[thread_index];
    statistics[(slot * M + thread_index) * 2 + 1] = row_sum[thread_index];
  }
}

// Partials are [slot][fused row][dimension]; statistics are
// [slot][fused row]{max, sum}. The callers lay slots out as
// [tile][KV head][split] (prefill) and [lane][KV head][split] (verify).
// Combines the splits of one fused row in split order. Only splits that own
// at least one page were written; the partition is recomputed here.
template <uint QueryHeadsPerKVHead, uint RowsPerTile>
inline void splash_q8_attention_reduce_row(
    device const float *partials, device const float *statistics,
    device bfloat *tile_output, uint committed_tokens, uint active_rows,
    uint splits, ulong head_slot, uint fused_row, uint thread_index) {
  constexpr ushort M = RowsPerTile * QueryHeadsPerKVHead;
  constexpr ushort D = SplashQ8HeadDimension;
  uint query_row = fused_row / QueryHeadsPerKVHead;
  uint visible_tokens = committed_tokens + active_rows;
  uint pages = splash_attention_pages(visible_tokens);
  uint per_split = splash_attention_pages_per_split(pages, splits);
  uint written = (pages + per_split - 1) / per_split;
  float value = 0.0f;
  if (query_row < active_rows) {
    float maximum = -INFINITY;
    for (uint split = 0; split < written; ++split)
      maximum =
          max(maximum, statistics[((head_slot + split) * M + fused_row) * 2]);
    float numerator = 0.0f;
    float denominator = 0.0f;
    for (uint split = 0; split < written; ++split) {
      ulong stat = ((head_slot + split) * M + fused_row) * 2;
      float weight = fast::exp(statistics[stat] - maximum);
      numerator +=
          weight *
          partials[((head_slot + split) * M + fused_row) * D + thread_index];
      denominator += weight * statistics[stat + 1];
    }
    value = denominator > 0.0f ? numerator / denominator : 0.0f;
  }
  tile_output[fused_row * D + thread_index] = bfloat(value);
}

// Statistics are shared by all 256 output dimensions. Compute their weights
// once per row, then let each lane stream one dimension of the partials.
template <uint QueryHeadsPerKVHead, uint RowsPerTile>
inline void splash_q8_attention_reduce_row_shared(
    device const float *partials, device const float *statistics,
    device bfloat *tile_output, uint committed_tokens, uint active_rows,
    uint splits, ulong head_slot, uint fused_row, uint thread_index,
    threadgroup float *weights, threadgroup float *group_values) {
  constexpr uint M = RowsPerTile * QueryHeadsPerKVHead;
  constexpr uint D = SplashQ8HeadDimension;
  if (fused_row / QueryHeadsPerKVHead >= active_rows) {
    tile_output[fused_row * D + thread_index] = bfloat(0.0f);
    return;
  }
  const uint pages = splash_attention_pages(committed_tokens + active_rows);
  const uint per_split = splash_attention_pages_per_split(pages, splits);
  const uint written = (pages + per_split - 1) / per_split;
  const uint lane = thread_index % 32, sg = thread_index / 32;
  const ulong stat = ((head_slot + thread_index) * M + fused_row) * 2;
  const float maximum = thread_index < written ? statistics[stat] : -INFINITY;
  const float group_maximum = simd_max(maximum);
  if (lane == 0) group_values[sg] = group_maximum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float row_maximum = simd_max(lane < 8 ? group_values[lane] : -INFINITY);
  if (thread_index < written)
    weights[thread_index] = fast::exp(maximum - row_maximum);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float numerator = 0.0f, denominator = 0.0f;
  // Keep both accumulations in the original split order. A parallel sum of
  // the denominator changes speculative acceptance on real model prompts.
  for (uint split = 0; split < written; ++split) {
    const float weight = weights[split];
    const ulong stat = ((head_slot + split) * M + fused_row) * 2;
    numerator += weight *
        partials[((head_slot + split) * M + fused_row) * D + thread_index];
    denominator += weight * statistics[stat + 1];
  }
  tile_output[fused_row * D + thread_index] =
      bfloat(denominator > 0.0f ? numerator / denominator : 0.0f);
}
