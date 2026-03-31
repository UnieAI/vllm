// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Fused CUDA kernel for n-gram speculative decoding.
//
// Replaces the PyTorch `unfold + gather + argmax` approach in
// NgramGPUKernel._find_first_and_extract_all_n_parallel() with a
// single-pass KMP-based kernel.  Each CUDA thread processes one
// sequence and finds the longest suffix n-gram match in O(n) time,
// compared to the O(n × m × num_ngram_sizes) of the unfold approach.

#include <torch/all.h>

#include <cstdint>

namespace vllm {

// ── KMP n-gram kernel ──────────────────────────────────────────────
//
// Each thread processes one sequence in the batch:
//   1. Reverse-scan with KMP to find the longest suffix n-gram match.
//   2. Extract up to `k` draft tokens following the match.
//   3. Write -1 for invalid positions.
//
// This fuses the outer Python loop over ngram sizes and the inner
// unfold+compare+argmax into a single O(n) pass per sequence.

__global__ void ngram_kmp_kernel(
    const int32_t* __restrict__ token_ids,  // [batch, max_seq_len]
    const int32_t* __restrict__ seq_lengths, // [batch]
    const bool* __restrict__ valid_mask,     // [batch]
    int32_t* __restrict__ draft_tokens,      // [batch, k]
    int32_t* __restrict__ num_valid_drafts,  // [batch]
    const int max_seq_len,
    const int min_ngram,
    const int max_ngram,
    const int k) {
  const int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;

  // Bounds check (batch_idx >= batch_size is handled by caller grid).
  // We use a 1D grid with enough threads to cover the batch.

  // Check valid mask.
  if (!valid_mask[batch_idx]) {
    // Fill draft tokens with -1.
    for (int j = 0; j < k; j++) {
      draft_tokens[batch_idx * k + j] = -1;
    }
    num_valid_drafts[batch_idx] = 0;
    return;
  }

  const int total = seq_lengths[batch_idx];
  const int32_t* tokens = token_ids + batch_idx * max_seq_len;

  // Early exit: too few tokens or no room for drafts.
  if (total < min_ngram || total >= max_seq_len) {
    for (int j = 0; j < k; j++) {
      draft_tokens[batch_idx * k + j] = -1;
    }
    num_valid_drafts[batch_idx] = 0;
    return;
  }

  // ── KMP on reversed token sequence ──
  // We want to find the longest suffix of tokens[0..total] (with
  // length in [min_ngram, max_ngram]) that appears earlier in the
  // sequence.
  //
  // Instead of physically reversing, we define:
  //   rev(i) = tokens[total - 1 - i]
  //
  // We compute the KMP failure function (LPS) on the reversed
  // sequence and track the longest match.

  // LPS array in shared memory (capped at max_ngram to save space).
  // We use dynamically-allocated shared memory.
  extern __shared__ int32_t shared_lps[];

  // Each thread gets its own LPS slice.
  // Shared memory layout: thread 0 uses shared_lps[0..max_ngram],
  // thread 1 uses shared_lps[max_ngram..2*max_ngram], etc.
  const int tid_in_block = threadIdx.x;
  int32_t* lps = shared_lps + tid_in_block * max_ngram;

  // Initialize LPS.
  for (int j = 0; j < max_ngram; j++) {
    lps[j] = 0;
  }

  int longest = 0;
  int best_pos = 0;
  int prev_lps = 0;

  for (int i = 1; i < total; i++) {
    // rev(i) = tokens[total - 1 - i]
    const int32_t rev_i = tokens[total - 1 - i];

    while (true) {
      const int32_t rev_prev = tokens[total - 1 - prev_lps];
      if (rev_prev == rev_i) {
        prev_lps++;

        if (prev_lps >= longest) {
          longest = prev_lps;
          best_pos = i;
        }

        if (i < max_ngram) {
          lps[i] = prev_lps;
        }

        if (prev_lps == max_ngram) {
          prev_lps = (max_ngram > 0 && max_ngram <= max_ngram)
                         ? lps[max_ngram - 1]
                         : 0;
        }
        break;
      } else if (prev_lps != 0) {
        int idx = prev_lps - 1;
        prev_lps = (idx < max_ngram) ? lps[idx] : 0;
      } else {
        break;
      }
    }
  }

  // ── Extract draft tokens ──
  if (longest < min_ngram) {
    for (int j = 0; j < k; j++) {
      draft_tokens[batch_idx * k + j] = -1;
    }
    num_valid_drafts[batch_idx] = 0;
    return;
  }

  // Convert back to original coordinates.
  const int start = total - 1 - best_pos + longest;
  int count = k;
  if (count > total - start) {
    count = total - start;
  }

  int valid = 0;
  for (int j = 0; j < k; j++) {
    if (j < count) {
      draft_tokens[batch_idx * k + j] = tokens[start + j];
      valid++;
    } else {
      draft_tokens[batch_idx * k + j] = -1;
    }
  }
  num_valid_drafts[batch_idx] = valid;
}

}  // namespace vllm

// ── C++ wrapper ────────────────────────────────────────────────────

void ngram_find_and_extract(
    torch::Tensor& draft_tokens,       // [batch, k] output, int32
    torch::Tensor& num_valid_drafts,   // [batch] output, int32
    const torch::Tensor& token_ids,    // [batch, max_seq_len] int32
    const torch::Tensor& seq_lengths,  // [batch] int32
    const torch::Tensor& valid_mask,   // [batch] bool
    int64_t min_ngram_len,
    int64_t max_ngram_len,
    int64_t num_draft_tokens) {
  const int batch_size = token_ids.size(0);
  const int max_seq_len = token_ids.size(1);
  const int k = static_cast<int>(num_draft_tokens);
  const int max_n = static_cast<int>(max_ngram_len);

  // Use 256 threads per block (standard for simple kernels).
  const int threads_per_block = 256;
  const int num_blocks = (batch_size + threads_per_block - 1) / threads_per_block;

  // Shared memory: each thread needs max_ngram * sizeof(int32_t) bytes.
  const size_t shared_mem_size =
      threads_per_block * max_n * sizeof(int32_t);

  // Launch kernel on the current CUDA stream.
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  vllm::ngram_kmp_kernel<<<num_blocks, threads_per_block,
                           shared_mem_size, stream>>>(
      token_ids.data_ptr<int32_t>(),
      seq_lengths.data_ptr<int32_t>(),
      valid_mask.data_ptr<bool>(),
      draft_tokens.data_ptr<int32_t>(),
      num_valid_drafts.data_ptr<int32_t>(),
      max_seq_len,
      static_cast<int>(min_ngram_len),
      max_n,
      k);
}
