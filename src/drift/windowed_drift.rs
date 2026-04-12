use super::{
    baseline::{BaselineCategoricalBins, BaselineContinuousBins},
    distribution::QuantileType,
};
use crate::errors::DriftError;
use std::env;
use std::fs::File;
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::num::NonZeroUsize;
use std::sync::OnceLock;
use tempfile::NamedTempFile;
/*
* The goal with this module is to allow for a smooth transition between drift windows.
*
* This will be more space and computationally intensive, but is able to have more elegant
* transition from one window to the next.
*
* Each window will be held in seperate buckets, determined by a configuration. The number of
* captured windows will be statically determined, then all windows will be preallocated.
*
* Every time the total count of examples in the container rolls passed WINDOW_SIZE, or in other
* words after an insert total_count % WINDOW_SIZE crosses rolls back to 0, then a window will be
* cleaned up in the stream.
*
* Stored global histogram. When a window is cleaned up, clean up the stale window, then resolve the
* global state.
*
* On each insert, if the current bin is maxed out, then clean up the previous flow state.
*
* 2 options here
* 1. check on each insert
* 2. determine before if there needs to be a reset, and add a break point on a batch insert
*
*
* I think that spilling to disk should happen internally, and dynamically. There should be some
* threshold where the cached windows should be spilled to disk. Could this be available process
* memory relative to dataset size, it num items * size >= some threshold.
*
*
* Implementation details
* - increment epoch counter
*   - allows for easy is saturated computation and current buffer offset
* - Keep live window count
*   - increment live window count every push
*   - when window is saturated, forward backend pointer
* - When forwarding epoch pointer
*   - call evict window every time
*   - within the evict window method, take in mut global window
*   - if buffer is not saturated, then it is a no op
*   - when it is saturated, clear the evicted window from backend buffer
*   - zero out live window
* - the backend owns the live window
* - outer type owns the global window
* - methods needed on backend
*   - push
*       - internally handles backend ops when windows roll
*       - when saturated
*           - calls into flush window, flush window from global
*           - evicts stale window
*
*   - epoch_offset
*       - compute current epoch offset
*       - window offset in buffer
*   - bin_offset
*       - only on in memory backend
*       - compute offset into buffer of bin based on epoch offset and bin offset
*   - is_saturated
*       - determine if entire backend buffer has been saturated
*   - forward_offset
*       - increment epoch offset pointer
*   - offset_tail
*       - get the tail window
*   - live_window
*       - get a ref to the live window
*       - for in memory backend this will be the slab for the current window
*       - a seperate live window will be stored for the disk backed buffer
* */

/*
* Statically sized container to hold the window buckets.
* Both the buckets container and the bin buckets themselves are allocated upfront.
*
* the current bucket is represented by head.
* */

static DISK_SPILL_THRESHOLD: OnceLock<usize> = OnceLock::new();
const DISK_SPILL_THRESHOLD_ENV_VAR: &'static str = "FAIR_PERF_ML_DRIFT_DISK_SPILL_THRESHOLD";
const DEFAULT_DISK_SPILL_THRESHOLD: usize = 64 * 1000 * 1000; // 64 MB

fn get_spill_heuristic() -> usize {
    *DISK_SPILL_THRESHOLD.get_or_init(|| {
        let Ok(raw_env) = env::var(DISK_SPILL_THRESHOLD_ENV_VAR) else {
            return DEFAULT_DISK_SPILL_THRESHOLD;
        };
        let Ok(parsed_thres) = raw_env.parse::<usize>() else {
            return DEFAULT_DISK_SPILL_THRESHOLD;
        };
        parsed_thres
    })
}

pub trait WindowCoreBackend {
    // Evict a stale window from the buffer. Update global state window.
    // If buffer is not saturated, this is a no-op.
    fn evict_window(&mut self, global_window: &mut [f64]);

    // Flush the live local window to the buffer.
    fn flush_window(&mut self, global_window: &mut [f64]);

    // Current window offset
    fn epoch_offset(&self) -> usize;

    // Flag for if the entire buffer has been saturated
    fn is_saturated(&self) -> bool;

    // Push the head pointer.
    fn forward_offset(&mut self);

    // Get a ref to the current live window.
    fn live_window(&self) -> &[f64];

    // Push into the live window and global window.
    fn push(&mut self, bin: usize, global_window: &mut [f64]);
}

struct DiskBackedBackend {
    mem_buf: Vec<u8>,
    disk_buf: NamedTempFile,
    window_size: usize,
    num_epochs: usize,
    live_window: Vec<f64>,
}

impl WindowCoreBackend for DiskBackedBackend {
    fn evict_window(&mut self, global_window: &mut [f64]) {
        if !self.is_saturated() {
            return;
        }
    }

    fn flush_window(&mut self, global_window: &mut [f64]) {
        todo!()
    }

    fn epoch_offset(&self) -> usize {
        todo!()
    }

    #[inline]
    fn is_saturated(&self) -> bool {
        todo!()
    }

    #[inline]
    fn forward_offset(&mut self) {
        todo!()
    }

    fn live_window(&self) -> &[f64] {
        &self.live_window
    }

    fn push(&mut self, bin: usize, global_window: &mut [f64]) {
        todo!()
    }
}

// after buffer saturation...
// when a window gets saturated, then
// read in the buffer to be evicted, resolve global state,
// then clear the live window

impl DiskBackedBackend {
    fn new(window_size: usize, num_epochs: usize) -> std::io::Result<DiskBackedBackend> {
        let disk_buf = NamedTempFile::new()?;
        let mem_buf_size = window_size * num_epochs * 8_usize;
        Ok(DiskBackedBackend {
            disk_buf,
            mem_buf: vec![0_u8; mem_buf_size],
            window_size: window_size.into(),
            num_epochs: num_epochs.into(),
            live_window: vec![0_f64; window_size],
        })
    }
}

struct InMemoryWindowBackend {
    window_buffer: Vec<f64>, // contiguous allocation of all the windows
    epoch_count: usize,      // the "index" of the current epoch
    epoch_inc: usize,        // the current saturation of the current epoch
    epoch_size: usize,       // size of a single window epoch
    live_ex_count: usize,
}

impl WindowCoreBackend for InMemoryWindowBackend {
    fn evict_window(&mut self, global_window: &mut [f64]) {
        // Remove the examples captured in the global buffer that are to be flushed.
        debug_assert!(self.is_saturated());
        let offset_start = self.window_base_offset();
        let offset_end = offset_start + self.epoch_size;
        for i in offset_start..offset_end {
            global_window[i - offset_start] -= self.window_buffer[i];
        }
    }

    fn flush_window(&mut self, global_window: &mut [f64]) {
        // Flush window in window buffer.
        // Evict the window from global buffer.
        if self.is_saturated() {
            self.evict_window(global_window);

            let offset_start = self.window_base_offset();
            let offset_end = offset_start + self.epoch_size;
            for i in offset_start..offset_end {
                self.window_buffer[i] = 0_f64;
            }
        }
        self.forward_offset();
    }

    fn epoch_offset(&self) -> usize {
        self.epoch_inc % self.epoch_count
    }

    fn is_saturated(&self) -> bool {
        self.epoch_inc >= self.epoch_count
    }

    fn forward_offset(&mut self) {
        self.epoch_inc += 1;
    }

    fn live_window(&self) -> &[f64] {
        let offset_start = self.window_base_offset();
        &self.window_buffer[offset_start..offset_start + self.epoch_size]
    }

    fn push(&mut self, bin: usize, global_window: &mut [f64]) {
        if self.live_ex_count == self.epoch_size {
            self.flush_window(global_window)
        }
        let bin_offset = self.window_base_offset() + bin;
        global_window[bin] += 1_f64;
        self.window_buffer[bin_offset] += 1_f64;
    }
}

struct ConsumedWindow(bool);

impl InMemoryWindowBackend {
    fn new(epoch_size: usize, epoch_count: usize) -> InMemoryWindowBackend {
        let window_buffer = vec![0_f64; epoch_size * epoch_count];

        InMemoryWindowBackend {
            window_buffer,
            epoch_count,
            epoch_inc: 0,
            epoch_size,
            live_ex_count: 0,
        }
    }

    #[inline]
    fn window_base_offset(&self) -> usize {
        self.epoch_size * self.epoch_offset()
    }
}

pub struct ContinuousWindowedDriftConfig {
    pub epoch_size: NonZeroUsize,
    pub epoch_count: NonZeroUsize,
    pub quantile_type: Option<QuantileType>,
}

pub struct CategoricalWindowedDriftConfig {
    pub epoch_size: NonZeroUsize,
    pub epoch_count: NonZeroUsize,
}

pub struct WindowedContinuousDrift<S: WindowCoreBackend> {
    global_bins: Vec<f64>,
    backend: S,
    baseline_bins: BaselineContinuousBins,
}

impl WindowedContinuousDrift<InMemoryWindowBackend> {
    pub fn new_in_memory_backend(
        config: ContinuousWindowedDriftConfig,
        baseline_dataset: &[f64],
    ) -> Result<WindowedContinuousDrift<InMemoryWindowBackend>, DriftError> {
        let ContinuousWindowedDriftConfig {
            epoch_size,
            epoch_count,
            quantile_type,
        } = config;
        let baseline_bins =
            BaselineContinuousBins::new(baseline_dataset, quantile_type.unwrap_or_default())?;
        let backend = InMemoryWindowBackend::new(epoch_size.into(), epoch_count.into());
        Ok(WindowedContinuousDrift {
            global_bins: vec![0_f64; epoch_size.into()],
            baseline_bins,
            backend,
        })
    }
}

impl<S: WindowCoreBackend> WindowedContinuousDrift<S> {
    #[inline]
    pub fn push(&mut self, example: f64) {
        let bin = self.baseline_bins.resolve_bin(example);
        self.backend.push(bin, &mut self.global_bins)
    }

    pub fn push_batch(&mut self, examples: &[f64]) {
        for ex in examples {
            let bin = self.baseline_bins.resolve_bin(*ex);
            self.backend.push(bin, &mut self.global_bins)
        }
    }
}
