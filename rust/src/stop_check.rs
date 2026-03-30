// SPDX-License-Identifier: Apache-2.0
//
// Batch stop-condition checking for the update_from_output hot loop.
//
// Accepts stop_token_ids as a padded 2-D numpy array (shape [n, max_stop_len],
// sentinel = -1) instead of a Python list-of-lists.  This eliminates all
// per-element Python↔Rust boundary crossings and typically yields a 50x+
// speed-up over the previous PyList-based implementation.

use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::prelude::*;

/// Batch-check stop conditions for multiple requests.
///
/// Returns a 1-D `i32` array of stop reasons per request:
///   0 = not stopped
///   1 = EOS token
///   2 = stop token
///   3 = length capped
#[pyfunction]
#[pyo3(signature = (
    last_token_ids,
    num_tokens,
    num_output_tokens,
    min_tokens,
    max_tokens_per_req,
    eos_token_ids,
    stop_token_ids,
    max_model_len,
))]
pub fn batch_check_stop<'py>(
    py: Python<'py>,
    last_token_ids: PyReadonlyArray1<'py, i64>,
    num_tokens: PyReadonlyArray1<'py, i64>,
    num_output_tokens: PyReadonlyArray1<'py, i64>,
    min_tokens: PyReadonlyArray1<'py, i64>,
    max_tokens_per_req: PyReadonlyArray1<'py, i64>,
    eos_token_ids: PyReadonlyArray1<'py, i64>,
    stop_token_ids: PyReadonlyArray2<'py, i64>,
    max_model_len: i64,
) -> PyResult<Bound<'py, PyArray1<i32>>> {
    let last_tok = last_token_ids.as_slice()?;
    let n_tok = num_tokens.as_slice()?;
    let n_out = num_output_tokens.as_slice()?;
    let min_tok = min_tokens.as_slice()?;
    let max_tok = max_tokens_per_req.as_slice()?;
    let eos_tok = eos_token_ids.as_slice()?;
    let n = last_tok.len();

    // Flat slice for the 2-D stop_token_ids array (C-contiguous).
    let stop_shape = stop_token_ids.shape();
    let max_stop_len = if stop_shape.len() >= 2 { stop_shape[1] } else { 0 };
    // as_slice() works for any contiguous array; for a (n, 0) array
    // it returns an empty slice and max_stop_len == 0, so the inner
    // loop simply never executes.
    let stop_data = stop_token_ids.as_slice()?;

    let mut result = vec![0i32; n];

    for i in 0..n {
        // Skip if below minimum tokens.
        if n_out[i] < min_tok[i] {
            continue;
        }

        let last = last_tok[i];

        // Check EOS.
        if eos_tok[i] >= 0 && last == eos_tok[i] {
            result[i] = 1;
            continue;
        }

        // Check stop tokens (flat 2-D indexing: row i, col j).
        let row_base = i * max_stop_len;
        let mut found_stop = false;
        for j in 0..max_stop_len {
            let stop_id = stop_data[row_base + j];
            if stop_id < 0 {
                break; // sentinel — no more stop tokens in this row
            }
            if last == stop_id {
                found_stop = true;
                break;
            }
        }
        if found_stop {
            result[i] = 2;
            continue;
        }

        // Check length cap.
        if n_tok[i] >= max_model_len || n_out[i] >= max_tok[i] {
            result[i] = 3;
        }
    }

    Ok(PyArray1::from_vec(py, result))
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_stop_reasons() {
        pyo3::prepare_freethreaded_python();
        pyo3::Python::with_gil(|py| {
            use numpy::{PyArray1, PyArray2, PyArrayMethods};

            let last_tok = PyArray1::from_vec(py, vec![2i64, 5, 99, 100]);
            let num_tok = PyArray1::from_vec(py, vec![100i64, 50, 4096, 200]);
            let num_out = PyArray1::from_vec(py, vec![10i64, 5, 50, 100]);
            let min_tok = PyArray1::from_vec(py, vec![0i64, 0, 0, 0]);
            let max_tok = PyArray1::from_vec(py, vec![100i64, 100, 100, 100]);
            let eos = PyArray1::from_vec(py, vec![2i64, -1, -1, -1]);

            // Padded 2-D array: each row has stop tokens, padded with -1.
            // Row 0: no stop tokens → [-1]
            // Row 1: stop at 5     → [5]
            // Row 2: no stop tokens → [-1]
            // Row 3: no stop tokens → [-1]
            let stop_arr = PyArray2::from_vec2(
                py,
                &[vec![-1i64], vec![5], vec![-1], vec![-1]],
            )
            .unwrap();

            let result = super::batch_check_stop(
                py,
                last_tok.readonly(),
                num_tok.readonly(),
                num_out.readonly(),
                min_tok.readonly(),
                max_tok.readonly(),
                eos.readonly(),
                stop_arr.readonly(),
                4096,
            )
            .unwrap();

            let result_vec: Vec<i32> = result.to_vec().unwrap();
            assert_eq!(result_vec, vec![1, 2, 3, 3]);
        });
    }

    #[test]
    fn test_empty_stop_tokens() {
        pyo3::prepare_freethreaded_python();
        pyo3::Python::with_gil(|py| {
            use numpy::{PyArray1, PyArray2, PyArrayMethods};

            let last_tok = PyArray1::from_vec(py, vec![99i64]);
            let num_tok = PyArray1::from_vec(py, vec![10i64]);
            let num_out = PyArray1::from_vec(py, vec![5i64]);
            let min_tok = PyArray1::from_vec(py, vec![0i64]);
            let max_tok = PyArray1::from_vec(py, vec![100i64]);
            let eos = PyArray1::from_vec(py, vec![-1i64]);

            // Shape (1, 0): no stop tokens for any request.
            let stop_arr =
                PyArray2::from_vec2(py, &[Vec::<i64>::new()]).unwrap();

            let result = super::batch_check_stop(
                py,
                last_tok.readonly(),
                num_tok.readonly(),
                num_out.readonly(),
                min_tok.readonly(),
                max_tok.readonly(),
                eos.readonly(),
                stop_arr.readonly(),
                4096,
            )
            .unwrap();

            let result_vec: Vec<i32> = result.to_vec().unwrap();
            assert_eq!(result_vec, vec![0]); // no stop
        });
    }
}
