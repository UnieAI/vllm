// SPDX-License-Identifier: Apache-2.0
//
// Rust-accelerated n-gram speculative decoding proposer.
//
// Replaces Numba `batch_propose_numba` with a Rust implementation that:
//   1. Uses KMP without heap-allocating a reversed copy (indexes backwards).
//   2. Releases the GIL and uses std threads for parallelism.
//   3. Has zero JIT warmup (AOT compiled).

use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;


/// KMP-based n-gram matching on a token slice, without reversing.
///
/// Finds the longest suffix of `tokens[..total]` (with length in
/// `[min_n, max_n]`) that also appears earlier in the sequence. Returns
/// `(start, count)` where `start` is the index of the first draft token
/// and `count` is the number of draft tokens (0 = no match).
#[inline]
fn kmp_ngram_match(
    tokens: &[i32],
    total: usize,
    min_n: usize,
    max_n: usize,
    max_model_len: usize,
    k: usize,
) -> (usize, usize) {
    if total < min_n {
        return (0, 0);
    }
    let k = k.min(max_model_len.saturating_sub(total));
    if k == 0 {
        return (0, 0);
    }

    // We want to find the longest suffix of tokens[..total] that matches
    // an earlier occurrence. The original Python reverses tokens and uses
    // KMP on the reversed sequence. We do the same but index backwards
    // via a closure to avoid allocating a reversed copy.
    //
    // rev(i) = tokens[total - 1 - i]
    let rev = |i: usize| -> i32 { tokens[total - 1 - i] };

    // LPS array — we only need entries for indices < max_n (the prefix
    // of the reversed sequence). For indices >= max_n, we don't store
    // lps but still use the running `prev_lps` variable.
    let lps_cap = max_n.min(total);
    let mut lps = vec![0u32; lps_cap];

    let mut longest: u32 = 0;
    let mut best_pos: usize = 0;
    let mut prev_lps: u32 = 0;
    let mut i: usize = 1;

    while i < total {
        if rev(prev_lps as usize) == rev(i) {
            prev_lps += 1;

            if prev_lps >= longest {
                longest = prev_lps;
                best_pos = i;
            }

            if i < lps_cap {
                lps[i] = prev_lps;
            }

            if prev_lps == max_n as u32 {
                prev_lps = if max_n > 0 && max_n <= lps_cap {
                    lps[max_n - 1]
                } else {
                    0
                };
            }

            i += 1;
        } else if prev_lps != 0 {
            let idx = prev_lps as usize - 1;
            prev_lps = if idx < lps_cap { lps[idx] } else { 0 };
        } else {
            i += 1;
        }
    }

    if (longest as usize) < min_n {
        return (0, 0);
    }

    // Convert back to original coordinates.
    let start = total - 1 - best_pos + longest as usize;
    let count = k.min(total - start);
    (start, count)
}

/// Batch n-gram propose for multiple requests.
///
/// # Arguments
///
/// * `token_ids`     – 2-D i32 array `[batch_size, max_seq_len]`
/// * `num_tokens`    – 1-D i32 array `[batch_size]`
/// * `valid_indices` – request indices that need proposals
/// * `min_n`, `max_n` – n-gram length range
/// * `max_model_len` – maximum sequence length
/// * `k`             – number of draft tokens to propose
///
/// # Returns
///
/// `(draft_tokens [batch, k], num_draft_tokens [batch])` as numpy arrays.
#[pyfunction]
#[pyo3(signature = (
    token_ids,
    num_tokens,
    valid_indices,
    min_n,
    max_n,
    max_model_len,
    k,
))]
pub fn batch_ngram_propose<'py>(
    py: Python<'py>,
    token_ids: PyReadonlyArray2<'py, i32>,
    num_tokens: PyReadonlyArray1<'py, i32>,
    valid_indices: Vec<usize>,
    min_n: usize,
    max_n: usize,
    max_model_len: usize,
    k: usize,
) -> PyResult<(Bound<'py, numpy::PyArray2<i32>>, Bound<'py, PyArray1<i32>>)> {
    let arr = token_ids.as_array();
    let num_tok = num_tokens.as_slice()?;
    let batch_size = arr.nrows();

    // Copy token rows for valid requests into owned Vecs so we can
    // release the GIL.  For 128 × 1500 tokens this is ~768KB — negligible.
    let rows: Vec<(usize, Vec<i32>)> = valid_indices
        .iter()
        .filter_map(|&idx| {
            if idx >= batch_size {
                return None;
            }
            let n = num_tok[idx] as usize;
            if n == 0 {
                return None;
            }
            let row_slice = arr.row(idx);
            let row_data: Vec<i32> = row_slice.as_slice().unwrap()[..n].to_vec();
            Some((idx, row_data))
        })
        .collect();

    // Pre-allocate output.
    let mut draft_out = vec![0i32; batch_size * k];
    let mut ndrafts_out = vec![0i32; batch_size];

    // Release the GIL for the compute-heavy loop.
    py.allow_threads(|| {
        if rows.len() <= 4 {
            // Sequential for small batches.
            for (idx, tokens) in &rows {
                let n = tokens.len();
                let (start, count) = kmp_ngram_match(tokens, n, min_n, max_n, max_model_len, k);
                ndrafts_out[*idx] = count as i32;
                for j in 0..count {
                    draft_out[*idx * k + j] = tokens[start + j];
                }
            }
        } else {
            // Parallel via scoped threads.
            let num_threads = std::thread::available_parallelism()
                .map(|n| n.get().min(8))
                .unwrap_or(4);
            let chunk_size = (rows.len() + num_threads - 1) / num_threads;

            // Each thread returns (idx, draft_tokens) pairs.
            let results: Vec<Vec<(usize, Vec<i32>)>> = std::thread::scope(|s| {
                let handles: Vec<_> = rows
                    .chunks(chunk_size)
                    .map(|chunk| {
                        s.spawn(move || {
                            let mut local: Vec<(usize, Vec<i32>)> = Vec::new();
                            for (idx, tokens) in chunk {
                                let n = tokens.len();
                                let (start, count) =
                                    kmp_ngram_match(tokens, n, min_n, max_n, max_model_len, k);
                                if count > 0 {
                                    local.push((*idx, tokens[start..start + count].to_vec()));
                                }
                            }
                            local
                        })
                    })
                    .collect();
                handles.into_iter().map(|h| h.join().unwrap()).collect()
            });

            // Merge results.
            for thread_results in &results {
                for (idx, draft) in thread_results {
                    ndrafts_out[*idx] = draft.len() as i32;
                    for (j, &tok) in draft.iter().enumerate() {
                        draft_out[*idx * k + j] = tok;
                    }
                }
            }
        }
    });

    let draft_nd = Array2::from_shape_vec((batch_size, k), draft_out)
        .expect("shape mismatch");
    let draft_array = draft_nd.into_pyarray(py);
    let num_drafts_array = PyArray1::from_vec(py, ndrafts_out);

    Ok((draft_array, num_drafts_array))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_ngram_match() {
        let tokens = vec![1, 2, 3, 4, 5, 1, 2, 3];
        let (start, count) = kmp_ngram_match(&tokens, 8, 2, 5, 100, 3);
        assert_eq!(count, 3);
        assert_eq!(&tokens[start..start + count], &[4, 5, 1]);
    }

    #[test]
    fn test_no_match() {
        let tokens = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let (_, count) = kmp_ngram_match(&tokens, 8, 2, 5, 100, 3);
        assert_eq!(count, 0);
    }

    #[test]
    fn test_min_ngram_not_reached() {
        let tokens = vec![1, 2, 3, 1];
        let (_, count) = kmp_ngram_match(&tokens, 4, 2, 5, 100, 3);
        assert_eq!(count, 0);
    }

    #[test]
    fn test_repeated_pattern() {
        let tokens = vec![1, 2, 1, 2, 1, 2];
        let (start, count) = kmp_ngram_match(&tokens, 6, 2, 5, 100, 3);
        assert!(count > 0);
        assert_eq!(&tokens[start..start + count], &[1, 2, 1]);
    }

    #[test]
    fn test_max_model_len_limit() {
        let tokens = vec![1, 2, 3, 4, 5, 1, 2, 3];
        let (_, count) = kmp_ngram_match(&tokens, 8, 2, 5, 9, 3);
        assert_eq!(count, 1);
    }
}
