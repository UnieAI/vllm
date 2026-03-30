// SPDX-License-Identifier: Apache-2.0
//
// Batch processing of speculative decoding acceptance/rejection.

use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

/// For each request that used speculative decoding, compute adjusted
/// num_computed_tokens and num_output_placeholders after rejection.
///
/// Returns (adjusted_computed, adjusted_placeholders, num_accepted, num_rejected)
/// as four 1-D i64 arrays.
#[pyfunction]
#[pyo3(signature = (
    num_computed_tokens,
    num_output_placeholders,
    num_generated,
    num_draft_tokens,
))]
pub fn batch_apply_generated_tokens<'py>(
    py: Python<'py>,
    num_computed_tokens: PyReadonlyArray1<'py, i64>,
    num_output_placeholders: PyReadonlyArray1<'py, i64>,
    num_generated: PyReadonlyArray1<'py, i64>,
    num_draft_tokens: PyReadonlyArray1<'py, i64>,
) -> PyResult<(
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<i64>>,
)> {
    let computed = num_computed_tokens.as_slice()?;
    let placeholders = num_output_placeholders.as_slice()?;
    let generated = num_generated.as_slice()?;
    let draft = num_draft_tokens.as_slice()?;
    let n = computed.len();

    let mut adj_computed = vec![0i64; n];
    let mut adj_placeholders = vec![0i64; n];
    let mut accepted = vec![0i64; n];
    let mut rejected = vec![0i64; n];

    for i in 0..n {
        if draft[i] > 0 && generated[i] > 0 {
            let num_accepted = generated[i] - 1;
            let num_rejected = draft[i] - num_accepted;

            adj_computed[i] = if computed[i] > 0 {
                computed[i] - num_rejected
            } else {
                computed[i]
            };
            adj_placeholders[i] = if placeholders[i] > 0 {
                placeholders[i] - num_rejected
            } else {
                placeholders[i]
            };
            accepted[i] = num_accepted;
            rejected[i] = num_rejected;
        } else {
            adj_computed[i] = computed[i];
            adj_placeholders[i] = placeholders[i];
            accepted[i] = generated[i];
            rejected[i] = 0;
        }
    }

    Ok((
        PyArray1::from_vec(py, adj_computed),
        PyArray1::from_vec(py, adj_placeholders),
        PyArray1::from_vec(py, accepted),
        PyArray1::from_vec(py, rejected),
    ))
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_no_spec_decode() {
        pyo3::prepare_freethreaded_python();
        pyo3::Python::with_gil(|py| {
            use numpy::{PyArray1, PyArrayMethods};

            let computed = PyArray1::from_vec(py, vec![100i64, 200]);
            let placeholders = PyArray1::from_vec(py, vec![0i64, 0]);
            let generated = PyArray1::from_vec(py, vec![1i64, 1]);
            let draft = PyArray1::from_vec(py, vec![0i64, 0]);

            let (ac, ap, aa, ar) = super::batch_apply_generated_tokens(
                py,
                computed.readonly(),
                placeholders.readonly(),
                generated.readonly(),
                draft.readonly(),
            )
            .unwrap();

            assert_eq!(ac.to_vec().unwrap(), vec![100, 200]);
            assert_eq!(ap.to_vec().unwrap(), vec![0, 0]);
            assert_eq!(aa.to_vec().unwrap(), vec![1, 1]);
            assert_eq!(ar.to_vec().unwrap(), vec![0, 0]);
        });
    }

    #[test]
    fn test_with_spec_decode() {
        pyo3::prepare_freethreaded_python();
        pyo3::Python::with_gil(|py| {
            use numpy::{PyArray1, PyArrayMethods};

            let computed = PyArray1::from_vec(py, vec![105i64]);
            let placeholders = PyArray1::from_vec(py, vec![5i64]);
            let generated = PyArray1::from_vec(py, vec![3i64]);
            let draft = PyArray1::from_vec(py, vec![5i64]);

            let (ac, ap, aa, ar) = super::batch_apply_generated_tokens(
                py,
                computed.readonly(),
                placeholders.readonly(),
                generated.readonly(),
                draft.readonly(),
            )
            .unwrap();

            assert_eq!(ac.to_vec().unwrap(), vec![102]);
            assert_eq!(ap.to_vec().unwrap(), vec![2]);
            assert_eq!(aa.to_vec().unwrap(), vec![2]);
            assert_eq!(ar.to_vec().unwrap(), vec![3]);
        });
    }
}
