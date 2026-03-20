use super::{
    baseline::{BaselineCategoricalBins, BaselineContinuousBins},
    distribution::QuantileType,
};
use crate::errors::DriftError;
use std::num::NonZeroUsize;
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
* */

/*
* Statically sized container to hold the window buckets.
* Both the buckets container and the bin buckets themselves are allocated upfront.
*
* the current bucket is represented by head.
* */
struct WindowBucketContainer {
    window_buffer: Vec<f64>, // contiguous allocation of all the windows
    epoch_offset: usize,     // the "index" of the current epoch
    curr_epoch_sat: usize,   // the current saturation of the current epoch
    epoch_size: usize,       // size of a single window epoch
    saturated: bool,         // flag for if the entire buffer has been used
}

struct ConsumedWindow(bool);

impl WindowBucketContainer {
    fn new(epoch_size: usize, num_bins: usize, num_windows: usize) -> WindowBucketContainer {
        let window_buffer = vec![0_f64; num_windows * num_bins];

        WindowBucketContainer {
            window_buffer,
            epoch_offset: 0,
            curr_epoch_sat: 0,
            epoch_size,
            saturated: false,
        }
    }

    fn num_epochs(&self) -> usize {
        debug_assert!(self.window_buffer.len() % self.epoch_size == 0);
        self.window_buffer.len() / self.epoch_size
    }

    fn compute_offset(&self, bin_idx: usize) -> usize {
        (self.epoch_offset * self.epoch_size) + bin_idx
    }

    #[inline]
    fn push(&mut self, bin_idx: usize) -> ConsumedWindow {
        let bin_offset = self.compute_offset(bin_idx);
        self.window_buffer[bin_offset] += 1_f64;
        ConsumedWindow(self.curr_epoch_sat == self.epoch_size)
    }

    /// Push the head epoch forward. When the buffer has been saturated, then clean the global
    /// window and clean up the epoch being evicted, as it is now the current epoch to be used.
    fn forward_head(&mut self) {
        let size = self.num_epochs();
        if !self.saturated && self.epoch_offset == size.saturating_sub(1) {
            self.saturated = true;
        }
        self.epoch_offset = (self.epoch_offset + 1) % size;
    }

    fn roll_window(&mut self, global_bins: &mut [f64]) {
        self.forward_head();
        if self.saturated {
            self.clean_window(global_bins);
        }
        self.curr_epoch_sat = 0;
    }

    fn tail(&self) -> usize {
        debug_assert!(!self.saturated);
        (self.curr_epoch_sat) + 1 % self.num_epochs()
    }

    fn clean_window(&mut self, global_bins: &mut [f64]) {
        let size = self.num_epochs();

        // start offset at the base of the window to be evicted.
        let base_window_offset = self.epoch_offset * size;
        for i in 0..self.epoch_size {
            let buffer_offset = base_window_offset + i;
            global_bins[i] -= self.window_buffer[buffer_offset];
            self.window_buffer[buffer_offset] = 0_f64;
        }
    }
}

pub struct WindowedDataConfig {
    pub epoch_size: NonZeroUsize,
    pub num_epochs: NonZeroUsize,
    pub quantile_type: Option<QuantileType>,
}

pub struct WindowedContinuousDirft {
    global_bins: Vec<f64>,
    windowed_buckets: WindowBucketContainer,
    baseline_bins: BaselineContinuousBins,
}

impl WindowedContinuousDirft {
    pub fn new(
        config: WindowedDataConfig,
        baseline_dataset: &[f64],
    ) -> Result<WindowedContinuousDirft, DriftError> {
        let WindowedDataConfig {
            epoch_size,
            num_epochs,
            quantile_type,
        } = config;
        let baseline_bins =
            BaselineContinuousBins::new(baseline_dataset, quantile_type.unwrap_or_default())?;
        todo!()
    }

    #[inline]
    pub fn push(&mut self, runtime_example: f64) {
        let bin_idx = self.baseline_bins.resolve_bin(runtime_example);
        self.global_bins[bin_idx] += 1_f64;
        let ConsumedWindow(window_consumed) = self.windowed_buckets.push(bin_idx);

        if window_consumed {
            self.windowed_buckets.roll_window(&mut self.global_bins)
        }
    }

    pub fn push_batch(&mut self, runtime_dataset: &[f64]) {
        for example in runtime_dataset {
            self.push(*example)
        }
    }
}
