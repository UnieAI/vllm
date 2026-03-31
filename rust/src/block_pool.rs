// SPDX-License-Identifier: Apache-2.0
//
// Index-based doubly-linked free block queue.
//
// Replaces the Python FreeKVCacheBlockQueue which uses Python object
// attribute access (prev_free_block / next_free_block) for linked-list
// operations.  This Rust implementation uses flat Vec<i32> arrays for
// prev/next pointers, eliminating Python attribute lookup and reference
// counting overhead.

use pyo3::prelude::*;

const SENTINEL: i32 = -1;

/// A fast, index-based doubly-linked list for managing free KV cache blocks.
///
/// Block IDs range from 0 to capacity-1.  The queue maintains insertion
/// order (LRU at front, MRU at back) using prev/next index arrays.
#[pyclass]
pub struct RustFreeBlockQueue {
    prev: Vec<i32>,
    next: Vec<i32>,
    in_queue: Vec<bool>,
    head: i32,
    tail: i32,
    num_free: usize,
}

#[pymethods]
impl RustFreeBlockQueue {
    /// Create a new queue containing the given block IDs in order.
    #[new]
    fn new(block_ids: Vec<i32>, capacity: usize) -> Self {
        let mut prev = vec![SENTINEL; capacity];
        let mut next = vec![SENTINEL; capacity];
        let mut in_queue = vec![false; capacity];

        let n = block_ids.len();
        if n == 0 {
            return Self {
                prev,
                next,
                in_queue,
                head: SENTINEL,
                tail: SENTINEL,
                num_free: 0,
            };
        }

        // Build linked list from block_ids.
        for i in 0..n {
            let bid = block_ids[i] as usize;
            in_queue[bid] = true;
            prev[bid] = if i > 0 { block_ids[i - 1] } else { SENTINEL };
            next[bid] = if i + 1 < n {
                block_ids[i + 1]
            } else {
                SENTINEL
            };
        }

        Self {
            prev,
            next,
            in_queue,
            head: block_ids[0],
            tail: block_ids[n - 1],
            num_free: n,
        }
    }

    /// Pop the first (LRU) block from the queue.
    fn popleft(&mut self) -> PyResult<i32> {
        if self.head == SENTINEL {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "No free blocks available",
            ));
        }
        let bid = self.head;
        self._remove(bid as usize);
        Ok(bid)
    }

    /// Pop the first n blocks from the queue.
    fn popleft_n(&mut self, n: usize) -> PyResult<Vec<i32>> {
        if n > self.num_free {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Cannot pop {} blocks, only {} free",
                n, self.num_free
            )));
        }
        let mut result = Vec::with_capacity(n);
        for _ in 0..n {
            let bid = self.head;
            self._remove(bid as usize);
            result.push(bid);
        }
        Ok(result)
    }

    /// Remove a specific block from the queue (O(1)).
    fn remove(&mut self, block_id: i32) {
        let bid = block_id as usize;
        if bid < self.in_queue.len() && self.in_queue[bid] {
            self._remove(bid);
        }
    }

    /// Append a block to the tail of the queue.
    fn append(&mut self, block_id: i32) {
        let bid = block_id as usize;
        if bid >= self.in_queue.len() || self.in_queue[bid] {
            return;
        }
        self._append(bid);
    }

    /// Append multiple blocks to the tail in order.
    fn append_n(&mut self, block_ids: Vec<i32>) {
        for &bid in &block_ids {
            let b = bid as usize;
            if b < self.in_queue.len() && !self.in_queue[b] {
                self._append(b);
            }
        }
    }

    /// Return all free block IDs in order (head to tail).
    fn get_all(&self) -> Vec<i32> {
        let mut result = Vec::with_capacity(self.num_free);
        let mut cur = self.head;
        while cur != SENTINEL {
            result.push(cur);
            cur = self.next[cur as usize];
        }
        result
    }

    /// Number of free blocks.
    fn __len__(&self) -> usize {
        self.num_free
    }

    /// Check if a block is in the queue.
    fn contains(&self, block_id: i32) -> bool {
        let bid = block_id as usize;
        bid < self.in_queue.len() && self.in_queue[bid]
    }

    #[getter]
    fn num_free_blocks(&self) -> usize {
        self.num_free
    }
}

impl RustFreeBlockQueue {
    fn _remove(&mut self, bid: usize) {
        let p = self.prev[bid];
        let n = self.next[bid];

        if p == SENTINEL {
            self.head = n;
        } else {
            self.next[p as usize] = n;
        }
        if n == SENTINEL {
            self.tail = p;
        } else {
            self.prev[n as usize] = p;
        }

        self.prev[bid] = SENTINEL;
        self.next[bid] = SENTINEL;
        self.in_queue[bid] = false;
        self.num_free -= 1;
    }

    fn _append(&mut self, bid: usize) {
        self.in_queue[bid] = true;
        self.prev[bid] = self.tail;
        self.next[bid] = SENTINEL;

        if self.tail != SENTINEL {
            self.next[self.tail as usize] = bid as i32;
        } else {
            self.head = bid as i32;
        }
        self.tail = bid as i32;
        self.num_free += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_operations() {
        pyo3::prepare_freethreaded_python();
        pyo3::Python::with_gil(|_py| {
            let mut q = RustFreeBlockQueue::new(vec![0, 1, 2, 3, 4], 5);
            assert_eq!(q.num_free, 5);

            let b = q.popleft().unwrap();
            assert_eq!(b, 0);
            assert_eq!(q.num_free, 4);

            q.remove(2);
            assert_eq!(q.num_free, 3);
            assert_eq!(q.get_all(), vec![1, 3, 4]);

            q.append(2);
            assert_eq!(q.get_all(), vec![1, 3, 4, 2]);

            let popped = q.popleft_n(2).unwrap();
            assert_eq!(popped, vec![1, 3]);
            assert_eq!(q.get_all(), vec![4, 2]);
        });
    }
}
