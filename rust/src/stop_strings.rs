// SPDX-License-Identifier: Apache-2.0
//
// Aho-Corasick multi-pattern stop string matching.
//
// Replaces the Python loop in detokenizer.py `check_stop_strings()`
// which calls `str.find()` for each stop string sequentially.
// Aho-Corasick matches ALL patterns in a single O(n) pass.

use aho_corasick::AhoCorasick;
use pyo3::prelude::*;

/// Pre-compiled multi-pattern matcher for stop strings.
///
/// Build once per request (when stop strings are set), reuse for every
/// token generation step.
#[pyclass]
pub struct StopStringMatcher {
    ac: AhoCorasick,
    patterns: Vec<String>,
}

#[pymethods]
impl StopStringMatcher {
    /// Build the Aho-Corasick automaton from a list of stop strings.
    #[new]
    fn new(stop_strings: Vec<String>) -> PyResult<Self> {
        let ac = AhoCorasick::new(&stop_strings).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!(
                "Failed to build Aho-Corasick automaton: {e}"
            ))
        })?;
        Ok(Self {
            ac,
            patterns: stop_strings,
        })
    }

    /// Find the first (leftmost) stop string match in `text[search_start..]`.
    ///
    /// Returns ``(pattern_string, match_start_in_text)`` or ``None``.
    ///
    /// ``search_start`` is a **byte** offset into ``text``.  For correct
    /// behaviour the caller must convert a character offset to bytes
    /// (Python ``str`` is UTF-8 in Rust via PyO3).
    fn find_first(
        &self,
        text: &str,
        search_start: usize,
    ) -> Option<(String, usize)> {
        if search_start > text.len() {
            return None;
        }
        let mat = self.ac.find(&text[search_start..])?;
        let pattern = &self.patterns[mat.pattern().as_usize()];
        let abs_start = search_start + mat.start();
        Some((pattern.clone(), abs_start))
    }

    /// Check stop strings, replicating the exact semantics of
    /// ``detokenizer.check_stop_strings()``.
    ///
    /// Returns ``(stop_string, truncate_offset)`` or ``None``.
    /// ``truncate_offset == -1`` means no truncation is needed.
    fn check(
        &self,
        output_text: &str,
        new_char_count: usize,
        include_in_output: bool,
    ) -> Option<(String, i64)> {
        if new_char_count == 0 || self.patterns.is_empty() {
            return None;
        }

        let text_len = output_text.len();
        // Mirror the Python start-position logic:
        //   start = max(0, len(text) - new_char_count - max_stop_len + 1)
        // But since Aho-Corasick is O(n) regardless of pattern count,
        // we just search from the overlap region.
        let max_stop_len = self.patterns.iter().map(|s| s.len()).max().unwrap_or(0);
        let search_start = if new_char_count + max_stop_len > text_len {
            0
        } else {
            text_len - new_char_count - max_stop_len + 1
        };

        let (stop_str, stop_index) = self.find_first(output_text, search_start)?;

        if include_in_output {
            let end = stop_index + stop_str.len();
            if end >= text_len {
                return Some((stop_str, -1));
            }
            return Some((stop_str, end as i64));
        }

        Some((stop_str, stop_index as i64))
    }

    fn __len__(&self) -> usize {
        self.patterns.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_match() {
        pyo3::prepare_freethreaded_python();
        pyo3::Python::with_gil(|_py| {
            let m = StopStringMatcher::new(vec!["</s>".into(), "STOP".into()]).unwrap();

            // Match at end
            let r = m.check("Hello world</s>", 15, false);
            assert!(r.is_some());
            let (s, off) = r.unwrap();
            assert_eq!(s, "</s>");
            assert_eq!(off, 11); // index of '<'

            // No match
            let r = m.check("Hello world", 11, false);
            assert!(r.is_none());

            // Match with include_in_output
            let r = m.check("Hello STOP world", 16, true);
            assert!(r.is_some());
            let (s, off) = r.unwrap();
            assert_eq!(s, "STOP");
            assert_eq!(off, 10); // end of "STOP"
        });
    }
}
