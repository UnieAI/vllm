// SPDX-License-Identifier: Apache-2.0
//
// Vectorized token-budget computation for the scheduler's running and
// waiting request loops.

use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

/// Compute `num_new_tokens` for every running request in a single pass.
///
/// Replaces the inner `while req_index < len(self.running)` loop
/// (scheduler.py lines 424-557) for the token-count computation.
/// Block allocation and encoder scheduling remain in Python.
///
/// Returns a 1-D `i64` array where each element is the number of tokens
/// to schedule for that request.  Zero means "skip this request".
#[pyfunction]
#[pyo3(signature = (
    num_tokens_with_spec,
    num_output_placeholders,
    num_computed_tokens,
    num_prompt_tokens,
    max_tokens_per_req,
    token_budget,
    long_prefill_threshold,
    max_model_len,
))]
pub fn compute_running_tokens<'py>(
    py: Python<'py>,
    num_tokens_with_spec: PyReadonlyArray1<'py, i64>,
    num_output_placeholders: PyReadonlyArray1<'py, i64>,
    num_computed_tokens: PyReadonlyArray1<'py, i64>,
    num_prompt_tokens: PyReadonlyArray1<'py, i64>,
    max_tokens_per_req: PyReadonlyArray1<'py, i64>,
    token_budget: i64,
    long_prefill_threshold: i64,
    max_model_len: i64,
) -> PyResult<Bound<'py, PyArray1<i64>>> {
    let spec = num_tokens_with_spec.as_slice()?;
    let placeholders = num_output_placeholders.as_slice()?;
    let computed = num_computed_tokens.as_slice()?;
    let prompt = num_prompt_tokens.as_slice()?;
    let max_tok = max_tokens_per_req.as_slice()?;
    let n = spec.len();

    let mut result = vec![0i64; n];
    let mut budget = token_budget;

    for i in 0..n {
        if budget <= 0 {
            break;
        }

        // Async scheduling early-skip: avoid scheduling an extra step
        // when the request has already reached max_tokens.
        if placeholders[i] > 0
            && (computed[i] + 2 - placeholders[i]) >= (prompt[i] + max_tok[i])
        {
            continue;
        }

        let mut num_new = spec[i] + placeholders[i] - computed[i];

        // Long-prefill threshold.
        if long_prefill_threshold > 0 && num_new > long_prefill_threshold {
            num_new = long_prefill_threshold;
        }

        // Clamp to budget.
        num_new = num_new.min(budget);

        // Clamp to max model length.
        num_new = num_new.min(max_model_len - 1 - computed[i]);

        num_new = num_new.max(0);

        result[i] = num_new;
        budget -= num_new;
    }

    Ok(PyArray1::from_vec(py, result))
}

/// Compute `num_new_tokens` for waiting (new/preempted) requests.
///
/// Returns (tokens_array, remaining_budget).
#[pyfunction]
#[pyo3(signature = (
    num_tokens,
    num_computed_tokens,
    token_budget,
    long_prefill_threshold,
    enable_chunked_prefill,
))]
pub fn compute_waiting_tokens<'py>(
    py: Python<'py>,
    num_tokens: PyReadonlyArray1<'py, i64>,
    num_computed_tokens: PyReadonlyArray1<'py, i64>,
    token_budget: i64,
    long_prefill_threshold: i64,
    enable_chunked_prefill: bool,
) -> PyResult<(Bound<'py, PyArray1<i64>>, i64)> {
    let tokens = num_tokens.as_slice()?;
    let computed = num_computed_tokens.as_slice()?;
    let n = tokens.len();

    let mut result = vec![0i64; n];
    let mut budget = token_budget;

    for i in 0..n {
        if budget <= 0 {
            break;
        }

        let mut num_new = tokens[i] - computed[i];

        if long_prefill_threshold > 0 && num_new > long_prefill_threshold {
            num_new = long_prefill_threshold;
        }

        if !enable_chunked_prefill && num_new > budget {
            break;
        }

        num_new = num_new.min(budget);

        if num_new <= 0 {
            continue;
        }

        result[i] = num_new;
        budget -= num_new;
    }

    Ok((PyArray1::from_vec(py, result), budget))
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_compute_running_basic() {
        pyo3::prepare_freethreaded_python();
        pyo3::Python::with_gil(|py| {
            use numpy::{PyArray1, PyArrayMethods};

            let spec = PyArray1::from_vec(py, vec![100i64, 50, 200]);
            let placeholders = PyArray1::from_vec(py, vec![0i64, 0, 0]);
            let computed = PyArray1::from_vec(py, vec![90i64, 49, 100]);
            let prompt = PyArray1::from_vec(py, vec![100i64, 50, 200]);
            let max_tok = PyArray1::from_vec(py, vec![100i64, 100, 100]);

            let result = super::compute_running_tokens(
                py,
                spec.readonly(),
                placeholders.readonly(),
                computed.readonly(),
                prompt.readonly(),
                max_tok.readonly(),
                1000,
                0,
                4096,
            )
            .unwrap();

            let result_vec: Vec<i64> = result.to_vec().unwrap();
            assert_eq!(result_vec, vec![10, 1, 100]);
        });
    }
}
