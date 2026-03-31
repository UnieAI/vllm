// SPDX-License-Identifier: Apache-2.0
//
// Fast block hash computation for prefix caching.
//
// Replaces the Python path:
//   hash_function((parent_hash, tuple(token_ids), extra_keys))
// where hash_function = cbor2.dumps() + sha256/xxhash.
//
// By hashing raw bytes directly (parent_hash || token_id_le_bytes),
// we eliminate the cbor2/pickle serialization overhead entirely.

use pyo3::prelude::*;
use pyo3::types::PyBytes;
use xxhash_rust::xxh3::xxh3_128;

/// Compute a block hash by directly hashing raw data.
///
/// hash = xxh3_128(parent_hash || token_id[0].to_le_bytes() || ... || token_id[n].to_le_bytes())
///
/// This is NOT compatible with the existing sha256_cbor / xxhash_cbor
/// hash algorithms because it skips serialization.  Use this as a new
/// hash algorithm ("builtin") via `get_hash_fn_by_name()`.
///
/// Returns 16 bytes (128-bit xxh3 digest).
#[pyfunction]
pub fn hash_block_tokens_rust<'py>(
    py: Python<'py>,
    parent_hash: &[u8],
    token_ids: &[u8], // raw bytes of token IDs (already as bytes)
) -> Bound<'py, PyBytes> {
    let cap = parent_hash.len() + token_ids.len();
    let mut buf = Vec::with_capacity(cap);
    buf.extend_from_slice(parent_hash);
    buf.extend_from_slice(token_ids);
    let digest = xxh3_128(&buf);
    PyBytes::new(py, &digest.to_le_bytes())
}

/// Batch-compute chained block hashes for consecutive full blocks.
///
/// Given an initial parent hash and a flat token ID array, computes
/// the chain of block hashes:
///
///   hash[0] = xxh3_128(parent_hash  || tokens[0..block_size])
///   hash[1] = xxh3_128(hash[0]      || tokens[block_size..2*block_size])
///   ...
///
/// Token IDs are serialized as little-endian i32 bytes (matching the
/// common `int` size used in vLLM token lists).
///
/// Returns a list of 16-byte `bytes` objects, one per block.
#[pyfunction]
pub fn batch_hash_blocks<'py>(
    py: Python<'py>,
    parent_hash: &[u8],
    token_ids: Vec<i32>,
    block_size: usize,
) -> PyResult<Vec<Bound<'py, PyBytes>>> {
    if block_size == 0 {
        return Ok(vec![]);
    }
    let num_blocks = token_ids.len() / block_size;
    let mut result = Vec::with_capacity(num_blocks);
    let mut prev_hash: [u8; 16] = [0u8; 16];

    // Initialize prev_hash from parent_hash (may be variable length).
    if parent_hash.len() >= 16 {
        prev_hash.copy_from_slice(&parent_hash[..16]);
    } else {
        prev_hash[..parent_hash.len()].copy_from_slice(parent_hash);
    }

    // Pre-allocate buffer: 16 (parent) + block_size * 4 (i32 le bytes)
    let buf_size = 16 + block_size * 4;
    let mut buf = vec![0u8; buf_size];

    for i in 0..num_blocks {
        buf[..16].copy_from_slice(&prev_hash);
        let start = i * block_size;
        for (j, &tok) in token_ids[start..start + block_size].iter().enumerate() {
            let offset = 16 + j * 4;
            buf[offset..offset + 4].copy_from_slice(&tok.to_le_bytes());
        }
        let digest = xxh3_128(&buf);
        prev_hash = digest.to_le_bytes();
        result.push(PyBytes::new(py, &prev_hash));
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_batch_hash_deterministic() {
        pyo3::prepare_freethreaded_python();
        pyo3::Python::with_gil(|py| {
            let parent = [0u8; 16];
            let tokens = vec![1i32, 2, 3, 4, 5, 6];
            let block_size = 3;

            let r1 = batch_hash_blocks(py, &parent, tokens.clone(), block_size).unwrap();
            let r2 = batch_hash_blocks(py, &parent, tokens, block_size).unwrap();

            assert_eq!(r1.len(), 2);
            assert_eq!(
                r1[0].as_bytes(),
                r2[0].as_bytes(),
                "Same input must produce same hash"
            );
            assert_ne!(
                r1[0].as_bytes(),
                r1[1].as_bytes(),
                "Different blocks should have different hashes"
            );
        });
    }
}
