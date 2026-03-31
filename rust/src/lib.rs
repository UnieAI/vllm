// SPDX-License-Identifier: Apache-2.0
// Rust-accelerated core for vLLM v1.
//
// This crate provides high-performance implementations of the CPU-bound
// hot paths in the vLLM scheduler and engine, exposed to Python via PyO3.

use pyo3::prelude::*;

mod block_hash;
mod block_pool;
mod ngram;
mod schedule;
mod serial_helpers;
mod stop_check;
mod stop_strings;
mod update_output;

/// The top-level Python module, importable as `vllm._rs`.
#[pymodule]
#[pyo3(name = "_rs")]
fn vllm_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Scheduler acceleration
    m.add_function(wrap_pyfunction!(schedule::compute_running_tokens, m)?)?;
    m.add_function(wrap_pyfunction!(schedule::compute_waiting_tokens, m)?)?;
    m.add_function(wrap_pyfunction!(stop_check::batch_check_stop, m)?)?;
    m.add_function(wrap_pyfunction!(
        update_output::batch_apply_generated_tokens,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(ngram::batch_ngram_propose, m)?)?;

    // Block hash acceleration (prefix caching)
    m.add_function(wrap_pyfunction!(block_hash::hash_block_tokens_rust, m)?)?;
    m.add_function(wrap_pyfunction!(block_hash::batch_hash_blocks, m)?)?;

    // Free block queue (KV cache management)
    m.add_class::<block_pool::RustFreeBlockQueue>()?;

    // Stop string matching (detokenizer)
    m.add_class::<stop_strings::StopStringMatcher>()?;

    // Serialization helpers
    m.add_function(wrap_pyfunction!(
        serial_helpers::batch_encode_int32_arrays,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        serial_helpers::encode_int_array_as_bytes,
        m
    )?)?;

    Ok(())
}
