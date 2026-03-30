// SPDX-License-Identifier: Apache-2.0
// Rust-accelerated scheduler core for vLLM v1.
//
// This crate provides high-performance implementations of the CPU-bound
// hot paths in the vLLM scheduler, exposed to Python via PyO3.

use pyo3::prelude::*;

mod ngram;
mod schedule;
mod stop_check;
mod update_output;

/// The top-level Python module `vllm_rs`.
///
/// NOTE: When the Rust build is integrated into vllm's pyproject.toml,
/// this should be renamed to `vllm._rs` (a vllm submodule) by changing
/// the `module-name` in `rust/pyproject.toml` and adding a
/// `#[pyo3(name = "_rs")]` attribute here.
#[pymodule]
fn vllm_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(schedule::compute_running_tokens, m)?)?;
    m.add_function(wrap_pyfunction!(schedule::compute_waiting_tokens, m)?)?;
    m.add_function(wrap_pyfunction!(stop_check::batch_check_stop, m)?)?;
    m.add_function(wrap_pyfunction!(
        update_output::batch_apply_generated_tokens,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(ngram::batch_ngram_propose, m)?)?;
    Ok(())
}
