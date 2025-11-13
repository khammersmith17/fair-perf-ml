use chrono::{DateTime, Utc};
use numpy::PyReadonlyArray1;
use once_cell::sync::Lazy;
use pyo3::prelude::*;
use pyo3::{
    exceptions::{PySystemError, PyValueError},
    pyclass, pymethods,
};
use std::cmp::Ordering;
use std::collections::HashMap;
use thiserror::Error;
use uuid::Uuid;

const DEFAULT_STREAM_FLUSH: i64 = 3600 * 24;
const MAX_STREAM_SIZE: usize = 1_000_000;
const STABILITY_EPS: Lazy<f64> = Lazy::new(|| {
    let fallback = 1e-8;
    let Ok(var) = std::env::var("FAIR_PERF_STABILITY_EPS") else {
        return fallback;
    };

    var.parse::<f64>().unwrap_or_else(|_| fallback)
});

/*
* Streaming PSI API
* trait bounds are not compatible with #[pymethods]
* 1. new
* 2. reset baseline stream
* 3. update stream
* 4. flush
* 5. total stream size
* 6. last flush
* */
#[derive(Debug, Error)]
pub enum PSIError {
    #[error("Data used for runtime drift analysis must be non empty")]
    EmptyRuntimeData,
    #[error("Unable to convert internal timestamp into DateTime object")]
    DateTimeError,
    #[error("Internal runtime bins are malformed")]
    MalformedRuntimeData,
    #[error("Baseline data must be non empty")]
    EmptyBaselineData,
    #[error("NaN values are not supported")]
    NaNValueError,
}

impl Into<PyErr> for PSIError {
    fn into(self) -> PyErr {
        let err_message = self.to_string();
        match self {
            Self::EmptyRuntimeData | Self::EmptyBaselineData | Self::NaNValueError => {
                PyValueError::new_err(err_message)
            }
            Self::DateTimeError | Self::MalformedRuntimeData => PySystemError::new_err(err_message),
        }
    }
}

#[inline]
fn process_hist(num_items: usize, hist: &[usize]) -> Result<Vec<f64>, PSIError> {
    let total_n = num_items as f64;
    if total_n == 0_f64 {
        return Err(PSIError::EmptyRuntimeData);
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
            let b = (baseline + *STABILITY_EPS).max(*STABILITY_EPS);
            let r = (runtime + *STABILITY_EPS).max(*STABILITY_EPS);
            (b - r) * (b / r).ln()
        })
        .sum()
}

#[pyclass]
pub struct ContinuousPSI {
    bin_edges: Vec<f64>,
    baseline_hist: Vec<f64>,
    rt_bins: Vec<usize>,
    n_bins: usize,
}

// TODO:
// 1. preallocate runtime Vec for hist scores and zero out all entries every runtime run
// 2. Add doc comments
// 3. add python feature so this can also be used in rust context

impl ContinuousPSI {
    fn init_baseline_hist<'py>(
        &mut self,
        baseline_data: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<()> {
        let baseline_slice = baseline_data.as_slice()?;
        self.define_bins(baseline_slice)?;
        let (bl_count, bl_hist) = self.build_bl_hist(baseline_slice);
        match process_hist(bl_count, &bl_hist) {
            Ok(processed_bl_hist) => {
                self.baseline_hist = processed_bl_hist;
                Ok(())
            }
            Err(_) => Err(PSIError::EmptyBaselineData.into()),
        }
    }

    fn build_bl_hist<'py>(&self, data_slice: &[f64]) -> (usize, Vec<usize>) {
        let bl_count = data_slice.len();
        let mut hist = vec![0_usize; self.bin_edges.len() - 1];
        let n_bins = self.bin_edges.len() - 1;
        for item in data_slice {
            let i = self.bin_edges.partition_point(|edge| *item >= *edge);
            let idx = i.saturating_sub(1).min(n_bins - 1);
            hist[idx] += 1;
        }

        (bl_count, hist)
    }

    fn clear_rt_hist(&mut self) {
        for bin in self.rt_bins.iter_mut() {
            *bin = 0_usize;
        }
    }

    fn build_rt_hist(&mut self, data_slice: &[f64]) {
        let n_bins = self.bin_edges.len() - 1;
        for item in data_slice {
            let i = self.bin_edges.partition_point(|edge| *item >= *edge);
            let idx = i.saturating_sub(1).min(n_bins - 1);
            self.rt_bins[idx] += 1;
        }
    }

    fn init_runtime_container(&mut self) {
        let mut rt_bins: Vec<f64> = Vec::with_capacity(self.baseline_hist.len());
        for _ in 0..self.baseline_hist.len() {
            rt_bins.push(0_f64)
        }
    }

    fn define_bins<'py>(&mut self, data: &[f64]) -> PyResult<()> {
        let mut sorted_baseline: Vec<f64> = data.to_vec();
        if sorted_baseline.len() <= 1 {
            return Err(PyValueError::new_err("Baseline array requires > 1 value"));
        }

        if sorted_baseline.iter().any(|value| value.is_nan()) {
            return Err(PSIError::NaNValueError.into());
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
            rt_bins: Vec::new(),
            n_bins,
        };
        obj.init_baseline_hist(baseline_data)?;
        obj.init_runtime_container();
        Ok(obj)
    }

    fn reset_baseline<'py>(&mut self, baseline_data: PyReadonlyArray1<'py, f64>) -> PyResult<()> {
        self.baseline_hist.clear();
        self.init_baseline_hist(baseline_data)?;
        self.init_runtime_container();
        Ok(())
    }

    fn compute_psi_drift<'py>(
        &mut self,
        runtime_data: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<f64> {
        let runtime_data_slice = runtime_data.as_slice()?;
        let n = runtime_data_slice.len();
        self.build_rt_hist(&runtime_data_slice);
        let runtime_hist = match process_hist(n, &self.rt_bins) {
            Ok(hist) => hist,
            Err(e) => return Err(e.into()),
        };

        let psi = compute_psi(&self.baseline_hist, &runtime_hist);
        self.clear_rt_hist();
        Ok(psi)
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
        let (n_bins, stream_bins) = self.baseline.build_bl_hist(&runtime_slice);
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
    fn normalize(&self) -> Result<f64, PSIError> {
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

    fn reset_baseline<'py>(&mut self, baseline_data: PyReadonlyArray1<'py, f64>) -> PyResult<()> {
        self.baseline.reset_baseline(baseline_data)?;
        Ok(())
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
            Err(e) => Err(e.into()),
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
    fn last_flush(&self) -> PyResult<DateTime<Utc>> {
        let Some(ts) = DateTime::from_timestamp(self.last_flush_ts, 0_u32) else {
            return Err(PSIError::DateTimeError.into());
        };
        Ok(ts)
    }
}

// using a log scaling heuristic to preallocate
fn compute_expected_categorical_bins(n: f64) -> usize {
    n.powf(0.7).ceil() as usize
}

const OTHER_LABEL: &'static str = "__fairperf_othercat__";

#[inline]
fn process_categorical_hist(
    rt_hist: &HashMap<String, usize>,
    n: usize,
) -> Result<HashMap<String, f64>, PSIError> {
    if rt_hist.is_empty() {
        return Err(PSIError::EmptyRuntimeData);
    }
    let mut hist_quantiles: HashMap<String, f64> = HashMap::with_capacity(rt_hist.len());

    for (key, value) in rt_hist.iter() {
        hist_quantiles.insert(key.clone(), *value as f64 / n as f64);
    }
    Ok(hist_quantiles)
}

#[inline]
fn categorical_normalize(
    bl_hist: &HashMap<String, f64>,
    rt_hist: HashMap<String, f64>,
) -> Option<f64> {
    let mut psi = 0_f64;

    for (key, mut r) in rt_hist.into_iter() {
        let b_ref = bl_hist.get(key.as_str())?;
        let b = (*b_ref + *STABILITY_EPS).max(*STABILITY_EPS);
        r = (r + *STABILITY_EPS).max(*STABILITY_EPS);
        psi += (b - r) * (b / r).ln()
    }
    Some(psi)
}
//TODO:
//create an error type to distinguish between malformed bins and no runtime data in stream

#[pyclass]
pub struct CategoricalPSI {
    baseline_hist: HashMap<String, f64>,
    other_key: String,
}

impl CategoricalPSI {
    fn init_baseline_hist(&mut self, predicted_capacity: usize, bl_data: Vec<String>) {
        let n = bl_data.len();
        let mut baseline_bins: HashMap<String, usize> = HashMap::with_capacity(predicted_capacity);

        for cat in bl_data.into_iter() {
            if let Some(count) = baseline_bins.get_mut(&cat) {
                *count += 1;
            } else {
                baseline_bins.insert(cat.clone(), 1_usize);
                self.baseline_hist.insert(cat, 0_f64);
            }
        }

        self.baseline_hist.insert(self.other_key.clone(), 0_f64);

        for (key, value) in self.baseline_hist.iter_mut() {
            // safe unwrap, we know this key exists
            let count = baseline_bins.get(key).unwrap();
            *value = *count as f64 / n as f64;
        }
    }

    fn compute_rt(&self, runtime_data: Vec<String>) -> Option<f64> {
        let mut runtime_bins: HashMap<String, usize> =
            HashMap::with_capacity(self.baseline_hist.len());
        let mut runtime_hist: HashMap<String, f64> =
            HashMap::with_capacity(self.baseline_hist.len());
        let n = runtime_data.len();

        for key in self.baseline_hist.keys() {
            runtime_bins.insert(key.clone(), 0_usize);
            runtime_hist.insert(key.clone(), 0_f64);
        }

        for cat in runtime_data.into_iter() {
            if let Some(seen_bin) = runtime_bins.get_mut(&cat) {
                *seen_bin += 1;
            } else {
                let other_bin = runtime_bins.get_mut(&self.other_key)?;
                *other_bin += 1;
            }
        }

        for (key, value) in runtime_hist.iter_mut() {
            let count = runtime_bins.get(key)?;
            *value = *count as f64 / n as f64;
        }

        let mut psi = 0_f64;

        for (key, mut r) in runtime_hist.into_iter() {
            let b_ref = self.baseline_hist.get(&key)?;
            let b = (*b_ref + *STABILITY_EPS).max(*STABILITY_EPS);
            r = (r + *STABILITY_EPS).max(*STABILITY_EPS);
            psi += (b - r) * (b / r).ln()
        }

        Some(psi)
    }
}

#[pymethods]
impl CategoricalPSI {
    #[new]
    fn new(baseline_data: Vec<String>) -> PyResult<CategoricalPSI> {
        if baseline_data.is_empty() {
            return Err(PyValueError::new_err("Baseline data must be non empty"));
        }
        let n = baseline_data.len();
        let predicted_capacity = compute_expected_categorical_bins(n as f64);

        let baseline_hist: HashMap<String, f64> = HashMap::with_capacity(predicted_capacity);
        let uuid = Uuid::new_v4();
        let other_key = format!("{}_{}", OTHER_LABEL, uuid);
        let mut obj = CategoricalPSI {
            baseline_hist,
            other_key,
        };
        obj.init_baseline_hist(predicted_capacity, baseline_data);
        Ok(obj)
    }

    fn reset_baseline(&mut self, new_baseline: Vec<String>) {
        self.baseline_hist.clear();
        let n = new_baseline.len();
        let predicted_capacity = compute_expected_categorical_bins(n as f64);
        self.init_baseline_hist(predicted_capacity, new_baseline);
    }

    fn compute_psi_drift(&self, runtime_data: Vec<String>) -> PyResult<f64> {
        if let Some(psi) = self.compute_rt(runtime_data) {
            Ok(psi)
        } else {
            Err(PSIError::MalformedRuntimeData.into())
        }
    }

    #[getter]
    fn other_bucket_label(&self) -> String {
        self.other_key.clone()
    }
}

#[pyclass]
pub struct StreamingCategoricalPSI {
    baseline: CategoricalPSI,
    stream_bins: HashMap<String, usize>,
    total_stream_size: usize,
    last_flush_ts: i64,
    flush_rate: i64,
}

impl StreamingCategoricalPSI {
    fn init_stream_bins(&mut self) {
        for key in self.baseline.baseline_hist.keys() {
            self.stream_bins.insert(key.clone(), 0_usize);
        }
    }

    fn flush_runtime_stream(&mut self) {
        for value in self.stream_bins.values_mut() {
            *value = 0;
        }
        self.total_stream_size = 0;
    }

    fn update_stream_bins(&mut self, runtime_data: Vec<String>) {
        let n = runtime_data.len();
        for cat in runtime_data.into_iter() {
            if let Some(count) = self.stream_bins.get_mut(&cat) {
                *count += 1
            } else {
                let count = self.stream_bins.get_mut(&self.baseline.other_key).unwrap();
                *count += 1
            }
        }

        self.total_stream_size += n;
    }

    fn normalize(&self) -> Result<f64, PSIError> {
        let quantile_hist =
            match process_categorical_hist(&self.stream_bins, self.total_stream_size) {
                Ok(hist) => hist,
                Err(e) => return Err(e),
            };
        match categorical_normalize(&self.baseline.baseline_hist, quantile_hist) {
            Some(psi) => Ok(psi),
            None => Err(PSIError::MalformedRuntimeData),
        }
    }
}

#[pymethods]
impl StreamingCategoricalPSI {
    #[new]
    fn new(
        baseline_data: Vec<String>,
        user_flush_rate: Option<i64>,
    ) -> PyResult<StreamingCategoricalPSI> {
        let baseline = CategoricalPSI::new(baseline_data)?;
        let flush_rate = user_flush_rate.unwrap_or_else(|| DEFAULT_STREAM_FLUSH);
        let stream_bins: HashMap<String, usize> =
            HashMap::with_capacity(baseline.baseline_hist.len());

        let last_flush_ts = Utc::now().timestamp();
        let mut obj = StreamingCategoricalPSI {
            baseline,
            stream_bins,
            last_flush_ts,
            flush_rate,
            total_stream_size: 0_usize,
        };
        obj.init_stream_bins();
        Ok(obj)
    }

    fn reset_baseline(&mut self, new_baseline: Vec<String>) {
        self.baseline.reset_baseline(new_baseline);
        self.init_stream_bins();
    }

    fn update_stream(&mut self, runtime_data: Vec<String>) -> PyResult<f64> {
        let curr_ts: i64 = Utc::now().timestamp();

        if curr_ts > (self.last_flush_ts + self.flush_rate)
            || (self.total_stream_size + runtime_data.len()) > MAX_STREAM_SIZE
        {
            // reset and flush
            self.flush_runtime_stream();
            self.last_flush_ts = curr_ts;
        }
        self.update_stream_bins(runtime_data);

        match self.normalize() {
            Ok(drift) => Ok(drift),
            Err(e) => Err(e.into()),
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
    fn last_flush(&self) -> PyResult<DateTime<Utc>> {
        let Some(ts) = DateTime::from_timestamp(self.last_flush_ts, 0_u32) else {
            return Err(PSIError::DateTimeError.into());
        };
        Ok(ts)
    }

    #[getter]
    fn other_bucket_label(&self) -> String {
        self.baseline.other_key.clone()
    }
}
