use super::{
    baseline::{BaselineCategoricalBins, BaselineContinuousBins},
    distribution::QuantileType,
};
use crate::errors::DriftError;
use std::fs::File;
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::num::NonZeroUsize;
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
* */

/*
* Statically sized container to hold the window buckets.
* Both the buckets container and the bin buckets themselves are allocated upfront.
*
* the current bucket is represented by head.
* */

enum WindowBuffer {
    Memory(InMemoryWindowBuffer),
    Disk,
}

struct DiskBuffer {
    mem_buf: Vec<u8>,
    disk_buf: NamedTempFile,
    window_size: usize,
    num_epochs: usize,
}

// after buffer saturation...
// when a window gets saturated, then
// read in the buffer to be evicted, resolve global state,
// then clear the live window

impl DiskBuffer {
    fn new(window_size: usize, num_epochs: usize) -> std::io::Result<DiskBuffer> {
        let disk_buf = NamedTempFile::new()?;
        let mem_buf_size = window_size * num_epochs * 8_usize;
        Ok(DiskBuffer {
            disk_buf,
            mem_buf: vec![0_u8; mem_buf_size],
            window_size: window_size.into(),
            num_epochs: num_epochs.into(),
        })
    }
    fn flush_window(&mut self, window: &[f64]) -> std::io::Result<()> {
        debug_assert_eq!(window.len(), self.window_size);

        // When we reach the end "wrap around" by seeking back to the start.
        // This maintains a "static size" disk buffer.
        if self.disk_buf.stream_position()? == self.disk_buf_size() {
            self.disk_buf.seek(SeekFrom::Start(0))?;
        }
        let bytes_written = self.disk_buf.write(bytemuck::cast_slice(window))?;
        debug_assert_eq!(bytes_written, (std::mem::size_of::<f64>() * window.len()));
        Ok(())
    }

    fn read_window_to_evict(&mut self) -> std::io::Result<()> {
        self.disk_buf.read(&mut self.mem_buf)?;
        Ok(())
    }

    fn disk_buf_size(&self) -> u64 {
        (self.window_size * self.num_epochs) as u64 * 8_u64
    }

    fn get_evicted_window(&mut self) -> &[f64] {
        bytemuck::cast_slice(self.mem_buf.as_slice())
    }
}

struct DiskWindowBuffer {
    window_buffer: DiskBuffer,
    live_window: Vec<f64>,
}

struct InMemoryWindowBuffer {
    window_buffer: Vec<f64>, // contiguous allocation of all the windows
    epoch_count: usize,      // the "index" of the current epoch
    curr_epoch_sat: usize,   // the current saturation of the current epoch
    epoch_size: usize,       // size of a single window epoch
}

struct ConsumedWindow(bool);

impl InMemoryWindowBuffer {
    fn new(epoch_size: usize, num_bins: usize, num_windows: usize) -> InMemoryWindowBuffer {
        let window_buffer = vec![0_f64; num_windows * num_bins];

        InMemoryWindowBuffer {
            window_buffer,
            epoch_count: 0,
            curr_epoch_sat: 0,
            epoch_size,
        }
    }

    fn num_epochs(&self) -> usize {
        debug_assert!(self.window_buffer.len() % self.epoch_size == 0);
        self.window_buffer.len() / self.epoch_size
    }

    fn compute_offset(&self, bin_idx: usize) -> usize {
        (self.epoch_offset() * self.epoch_size) + bin_idx
    }

    #[inline]
    fn push(&mut self, bin_idx: usize) -> ConsumedWindow {
        let bin_offset = self.compute_offset(bin_idx);
        self.window_buffer[bin_offset] += 1_f64;
        ConsumedWindow(self.curr_epoch_sat == self.epoch_size)
    }

    fn is_saturated(&self) -> bool {
        self.epoch_count > self.num_epochs()
    }

    fn epoch_offset(&self) -> usize {
        self.epoch_count % self.num_epochs()
    }

    /// Push the head epoch forward. When the buffer has been saturated, then clean the global
    /// window and clean up the epoch being evicted, as it is now the current epoch to be used.
    fn forward_head(&mut self) {
        self.epoch_count += 1;
    }

    fn roll_window(&mut self, global_bins: &mut [f64]) {
        self.forward_head();
        if self.is_saturated() {
            self.clean_window(global_bins);
        }
        self.curr_epoch_sat = 0;
    }

    fn tail(&self) -> usize {
        debug_assert!(!self.is_saturated());
        (self.curr_epoch_sat) + 1 % self.num_epochs()
    }

    fn clean_window(&mut self, global_bins: &mut [f64]) {
        let size = self.num_epochs();

        // start offset at the base of the window to be evicted.
        let base_window_offset = self.epoch_offset() * size;
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
    windowed_buckets: InMemoryWindowBuffer,
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
