// SPDX-License-Identifier: Apache-2.0
//
// Serialization helpers for the vLLM IPC path.
//
// Provides a fast batch tensor metadata encoder that avoids repeated
// Python isinstance() checks and attribute lookups in enc_hook().

use numpy::PyReadonlyArray1;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyList};

/// Batch-encode a list of 1-D numpy int32 arrays into raw byte buffers.
///
/// For each array, produces a ``bytes`` object containing the raw
/// little-endian data.  This is the hot path for encoding
/// ``sampled_token_ids`` in ``EngineCoreOutputs`` where each request
/// contributes a small int32 array.
///
/// Returns a list of ``bytes`` objects (one per input array).
#[pyfunction]
pub fn batch_encode_int32_arrays<'py>(
    py: Python<'py>,
    arrays: &Bound<'py, PyList>,
) -> PyResult<Vec<Bound<'py, PyBytes>>> {
    let n = arrays.len();
    let mut result = Vec::with_capacity(n);
    for item in arrays.iter() {
        let arr: PyReadonlyArray1<i32> = item.extract()?;
        let slice = arr.as_slice()?;
        // Each i32 → 4 bytes LE
        let mut buf = Vec::with_capacity(slice.len() * 4);
        for &v in slice {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        result.push(PyBytes::new(py, &buf));
    }
    Ok(result)
}

/// Batch-encode a flat int32 numpy array as raw bytes.
///
/// Avoids per-element Python overhead by doing a single memcpy-like
/// operation in Rust.
#[pyfunction]
pub fn encode_int_array_as_bytes<'py>(
    py: Python<'py>,
    arr: PyReadonlyArray1<'py, i32>,
) -> PyResult<Bound<'py, PyBytes>> {
    let slice = arr.as_slice()?;
    let byte_len = slice.len() * 4;
    let mut buf = Vec::with_capacity(byte_len);
    for &v in slice {
        buf.extend_from_slice(&v.to_le_bytes());
    }
    Ok(PyBytes::new(py, &buf))
}
