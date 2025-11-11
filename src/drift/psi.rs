use chrono::Utc;
use numpy::PyReadonlyArray1;
use pyo3::prelude::*;
use pyo3::{exceptions::PyValueError, pyclass, pymethods};
use std::cmp::Ordering;
use std::collections::HashMap;

const DEFAULT_STREAM_FLUSH: i64 = 3600 * 24;
const MAX_STREAM_SIZE: usize = 1_000_000;

#[inline]
fn process_hist(num_items: usize, hist: &[usize]) -> Result<Vec<f64>, String> {
    let total_n = num_items as f64;
    if total_n == 0_f64 {
        return Err("Empty hist".to_string());
    }
    let bl_hist = hist
        .iter()
        .map(|n| *n as f64 / total_n)
        .collect::<Vec<f64>>();
    Ok(bl_hist)
}

#[inline]
fn compute_psi(baseline_hist: &[f64], runtime_hist: &[f64]) -> f64 {
    debug_assert_eq!(runtime_hist.len(), baseline_hist.len());
    baseline_hist
        .iter()
        .zip(runtime_hist)
        .map(|(baseline, runtime)| {
            let b = (baseline + f64::EPSILON).max(f64::EPSILON);
            let r = (runtime + f64::EPSILON).max(f64::EPSILON);
            (b - r) * (b / r).ln()
        })
        .sum()
}

#[pyclass]
pub struct ContinuousPSI {
    pub bin_edges: Vec<f64>,
    pub baseline_hist: Vec<f64>,
    n_bins: usize,
}

impl ContinuousPSI {
    fn init_baseline_hist<'py>(
        &mut self,
        baseline_data: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<()> {
        let baseline_slice = baseline_data.as_slice()?;
        self.define_bins(baseline_slice)?;
        let (bl_count, bl_hist) = self.build_hist(baseline_slice);
        if let Ok(bl_hist) = process_hist(bl_count, &bl_hist) {
            self.baseline_hist = bl_hist
        } else {
            return Err(PyValueError::new_err("Baseline data must not be empty"));
        }
        Ok(())
    }

    fn build_hist<'py>(&self, data_slice: &[f64]) -> (usize, Vec<usize>) {
        let bl_count = data_slice.len();
        let mut hist = vec![0_usize; self.bin_edges.len() - 1];
        let n_bins = self.bin_edges.len() - 1;
        for item in data_slice {
            let mut idx: usize = n_bins - 1;
            for i in 0..n_bins {
                if *item < self.bin_edges[i + 1] {
                    idx = i;
                    break;
                }
            }
            hist[idx] += 1;
        }

        (bl_count, hist)
    }

    fn define_bins<'py>(&mut self, data: &[f64]) -> PyResult<()> {
        let mut sorted_baseline: Vec<f64> = data.to_vec();
        if sorted_baseline.len() <= 1 {
            return Err(PyValueError::new_err("Baseline array requires > 1 value"));
        }
        sorted_baseline.sort_by(|a, b| a.partial_cmp(&b).unwrap_or(Ordering::Equal));

        self.bin_edges.clear();

        // safe unwrap because we know there are items in the array
        if sorted_baseline.first().unwrap() == sorted_baseline.last().unwrap() {
            self.bin_edges
                .extend(vec![sorted_baseline[0], sorted_baseline[0]]);
            return Ok(());
        }

        let n_bl_samples = sorted_baseline.len();
        let n_bins = self.n_bins.min(n_bl_samples - 1).max(1);
        let bin_size = sorted_baseline.len() / n_bins;

        self.bin_edges.push(sorted_baseline[0]);
        for i in 1..(n_bins) {
            let idx = i * bin_size;
            if idx < n_bl_samples {
                self.bin_edges.push(sorted_baseline[idx - 1]);
            }
        }
        self.bin_edges.push(sorted_baseline[n_bl_samples - 1]);
        self.n_bins = n_bins;
        Ok(())
    }
}

#[pymethods]
impl ContinuousPSI {
    #[new]
    pub fn new<'py>(
        n_bins: usize,
        baseline_data: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<ContinuousPSI> {
        let bin_edges: Vec<f64> = Vec::with_capacity(n_bins + 1);

        let mut obj = ContinuousPSI {
            bin_edges,
            baseline_hist: Vec::new(),
            n_bins,
        };
        obj.init_baseline_hist(baseline_data)?;
        Ok(obj)
    }

    fn reset_baseline<'py>(&mut self, baseline_data: PyReadonlyArray1<'py, f64>) -> PyResult<()> {
        self.baseline_hist.clear();
        self.init_baseline_hist(baseline_data)?;
        Ok(())
    }

    fn compute_psi_drift<'py>(&self, runtime_data: PyReadonlyArray1<'py, f64>) -> PyResult<f64> {
        let runtime_data_slice = runtime_data.as_slice()?;
        let (n, base_runtime_hist) = self.build_hist(&runtime_data_slice);
        let Ok(runtime_hist) = process_hist(n, &base_runtime_hist) else {
            return Err(PyValueError::new_err(
                "Runtime data array must not be empty",
            ));
        };
        Ok(compute_psi(&self.baseline_hist, &runtime_hist))
    }
}

#[pyclass]
pub struct StreamingContinuousPSI {
    baseline: ContinuousPSI,
    stream_bins: Vec<usize>,
    total_stream_size: usize,
    last_flush_ts: i64,
    flush_rate: i64,
}

impl StreamingContinuousPSI {
    fn init_runtime_stream(&mut self, runtime_slice: &[f64]) {
        let (n_bins, stream_bins) = self.baseline.build_hist(&runtime_slice);
        self.total_stream_size = n_bins;
        self.stream_bins = stream_bins;
    }

    // zero out all bins
    fn flush_runtime_stream(&mut self) {
        for bin in &mut self.stream_bins {
            *bin = 0;
        }
        self.total_stream_size = 0;
    }

    fn update_stream_bins(&mut self, data_slice: &[f64]) {
        let data_size = data_slice.len();
        let n_bins = self.baseline.bin_edges.len() - 1;
        for item in data_slice {
            let mut idx: usize = n_bins - 1;
            for i in 0..n_bins {
                if *item < self.baseline.bin_edges[i + 1] {
                    idx = i;
                    break;
                }
            }
            self.stream_bins[idx] += 1;
        }
        self.total_stream_size += data_size;
    }

    #[inline]
    fn normalize(&self) -> Result<f64, String> {
        match process_hist(self.total_stream_size, &self.stream_bins) {
            Ok(snapshot) => Ok(compute_psi(&self.baseline.baseline_hist, &snapshot)),
            Err(e) => Err(e),
        }
    }
}

#[pymethods]
impl StreamingContinuousPSI {
    #[new]
    fn new<'py>(
        n_bins: usize,
        baseline_data: PyReadonlyArray1<'py, f64>,
        flush_cadence: Option<i64>,
    ) -> PyResult<StreamingContinuousPSI> {
        let flush_rate = flush_cadence.unwrap_or_else(|| DEFAULT_STREAM_FLUSH);
        let baseline = ContinuousPSI::new(n_bins, baseline_data)?;
        let total_stream_size = 0_usize;
        let last_flush_ts: i64 = Utc::now().timestamp().into();
        let stream_bins: Vec<usize> = Vec::new();

        Ok(StreamingContinuousPSI {
            stream_bins,
            baseline,
            total_stream_size,
            last_flush_ts,
            flush_rate,
        })
    }

    fn update_stream<'py>(&mut self, runtime_data: PyReadonlyArray1<'py, f64>) -> PyResult<f64> {
        let runtime_data_slice = runtime_data.as_slice()?;

        let curr_ts: i64 = Utc::now().timestamp();

        if self.stream_bins.is_empty() {
            // need to init stream
            self.init_runtime_stream(&runtime_data_slice);
            self.last_flush_ts = curr_ts;
        } else {
            if curr_ts > (self.last_flush_ts + self.flush_rate)
                || (self.total_stream_size + runtime_data_slice.len()) > MAX_STREAM_SIZE
            {
                // reset and flush
                self.flush_runtime_stream();
                self.last_flush_ts = curr_ts;
            }
            self.update_stream_bins(&runtime_data_slice);
        }

        match self.normalize() {
            Ok(drift) => Ok(drift),
            Err(e) => Err(PyValueError::new_err(e)),
        }
    }

    fn flush(&mut self) {
        self.flush_runtime_stream();
        self.last_flush_ts = Utc::now().timestamp();
    }

    #[getter]
    fn total_samples(&self) -> usize {
        self.total_stream_size
    }

    #[getter]
    fn last_flush(&self) -> i64 {
        self.last_flush_ts
    }
}

#[pyclass]
pub struct CategoricalPSI {}
