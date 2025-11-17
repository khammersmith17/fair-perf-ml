use ahash::{HashMap, HashMapExt};
use chrono::{DateTime, Utc};
use numpy::PyReadonlyArray1;
use once_cell::sync::Lazy;
use pyo3::prelude::*;
use pyo3::{
    exceptions::{PySystemError, PyValueError},
    pyclass, pymethods,
};
use std::cmp::Ordering;
use thiserror::Error;
use uuid::Uuid;

const DEFAULT_STREAM_FLUSH: i64 = 3600 * 24;
const MAX_STREAM_SIZE: usize = 1_000_000;
// read in from user defined env var or set to default epsilon
const STABILITY_EPS: Lazy<f64> = Lazy::new(|| {
    let default = 1e-12;
    let Ok(var) = std::env::var("FAIR_PERF_STABILITY_EPS") else {
        return default;
    };

    var.parse::<f64>().unwrap_or_else(|_| default)
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
fn compute_new_hist_prob(num_items: usize, hist: &[f64]) -> Result<Vec<f64>, PSIError> {
    let total_n = num_items as f64;
    if total_n == 0_f64 {
        return Err(PSIError::EmptyRuntimeData);
    }
    let bl_hist = hist.iter().map(|n| *n / total_n).collect::<Vec<f64>>();
    Ok(bl_hist)
}

// lazily evaluate bin ratio
// cheap computation and saves space/memory lookup
#[inline]
fn compute_psi(baseline_hist: &[f64], runtime_bins: &[f64], n: f64) -> f64 {
    debug_assert_eq!(runtime_bins.len(), baseline_hist.len());
    let mut psi: f64 = 0_f64;
    let eps = *STABILITY_EPS;
    for i in 0..baseline_hist.len() {
        // cheap clone, unsafe to skip bounds checks
        let mut b = unsafe { *baseline_hist.get_unchecked(i) };
        b = (b + eps).max(eps);
        let mut r = unsafe { *runtime_bins.get_unchecked(i) };
        r = r / n;
        r = (r + eps).max(eps);
        psi += (b - r) * (b / r).ln();
    }

    psi
}

// break out baseline to have shared logic between the discrete and the streaming variants of PSI
// allows for more elegant composition of different usage
struct BaselineContinuousPSI {
    n_bins: usize,
    bin_edges: Vec<f64>,
    baseline_hist: Vec<f64>,
}

impl BaselineContinuousPSI {
    fn new(n_bins: usize, baseline_data: &[f64]) -> Result<BaselineContinuousPSI, PSIError> {
        let mut obj = BaselineContinuousPSI {
            n_bins,
            bin_edges: Vec::new(),
            baseline_hist: Vec::new(),
        };

        obj.init_baseline_hist(baseline_data)?;
        Ok(obj)
    }

    // init method moved out of constructor to be reusable across new initialization as well as
    // when reseting the baseline
    fn init_baseline_hist(&mut self, baseline_data: &[f64]) -> Result<(), PSIError> {
        self.define_bins(baseline_data)?;
        let (bl_count, bl_hist) = self.build_bl_hist(baseline_data);
        match compute_new_hist_prob(bl_count, &bl_hist) {
            Ok(processed_bl_hist) => {
                self.baseline_hist = processed_bl_hist;
                Ok(())
            }
            Err(_) => Err(PSIError::EmptyBaselineData.into()),
        }
    }

    fn build_bl_hist(&self, data_slice: &[f64]) -> (usize, Vec<f64>) {
        let bl_count = data_slice.len();
        let mut hist = vec![0_f64; self.bin_edges.len() - 1];
        let n_bins = self.bin_edges.len() - 1;
        for item in data_slice {
            let i = self.bin_edges.partition_point(|edge| *item >= *edge);
            let idx = i.saturating_sub(1).min(n_bins - 1);
            hist[idx] += 1_f64;
        }

        (bl_count, hist)
    }

    fn define_bins(&mut self, data: &[f64]) -> Result<(), PSIError> {
        // sort baseline data for more efficient bin edge computation
        let mut sorted_baseline: Vec<f64> = data.to_vec();

        // baselining requires > 1 baseline sample
        if sorted_baseline.len() <= 1 {
            return Err(PSIError::EmptyBaselineData);
        }

        // do not accept NaNs
        if sorted_baseline.iter().any(|value| value.is_nan()) {
            return Err(PSIError::NaNValueError);
        }

        sorted_baseline.sort_by(|a, b| a.partial_cmp(&b).unwrap_or(Ordering::Equal));

        self.bin_edges.clear();

        // if all sorted items in the baseline sample are equal, one logical bin
        // safe unwrap because we know there are items in the array
        if sorted_baseline.first().unwrap() == sorted_baseline.last().unwrap() {
            self.bin_edges
                .extend(vec![sorted_baseline[0], sorted_baseline[0]]);
            return Ok(());
        }

        let n_bl_samples = sorted_baseline.len();
        let n_bins = self.n_bins.min(n_bl_samples - 1).max(1);
        let bin_size = sorted_baseline.len() / n_bins;

        // set population left edge
        self.bin_edges.push(sorted_baseline[0]);

        // set bins using an O(number of bins) loop
        // index in bin_size * 1
        // creates bins of [sorted_baseline[(i - 1) * bin_size], sorted_baseline[i * bin_size]]
        for i in 1..(n_bins) {
            let idx = i * bin_size;
            if idx < n_bl_samples {
                self.bin_edges.push(sorted_baseline[idx - 1]);
            }
        }

        // set population right edge
        self.bin_edges.push(sorted_baseline[n_bl_samples - 1]);
        self.n_bins = n_bins;
        Ok(())
    }

    #[inline]
    fn resolve_bin(&self, sample: f64) -> usize {
        // find "pivot" point
        // ie the bin where value >= left and < right
        let i = self.bin_edges.partition_point(|edge| sample >= *edge);
        i.clamp(0, self.n_bins - 1)
    }

    // call into init method
    fn reset(&mut self, baseline_data: &[f64]) -> Result<(), PSIError> {
        self.init_baseline_hist(baseline_data)?;
        Ok(())
    }
}

#[pyclass]
pub struct ContinuousPSI {
    baseline: BaselineContinuousPSI,
    rt_bins: Vec<f64>,
}

// TODO:
// 1. Add doc comments
// 2. add python feature so this can also be used in rust context

impl ContinuousPSI {
    fn clear_rt(&mut self) {
        self.rt_bins.fill(0_f64);
    }

    #[inline]
    fn build_rt_hist(&mut self, data_slice: &[f64]) {
        let n_bins = self.baseline.bin_edges.len() - 1;
        for item in data_slice {
            let i = self.baseline.resolve_bin(*item);
            let idx = i.saturating_sub(1).min(n_bins - 1);
            self.rt_bins[idx] += 1_f64;
        }
    }

    fn init_runtime_containers(&mut self) {
        let len = self.baseline.baseline_hist.len();
        self.rt_bins = vec![0_f64; len];
    }
}

#[pymethods]
impl ContinuousPSI {
    #[new]
    pub fn new<'py>(
        n_bins: usize,
        baseline_data: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<ContinuousPSI> {
        let bl_slice = baseline_data.as_slice()?;
        let baseline = match BaselineContinuousPSI::new(n_bins, &bl_slice) {
            Ok(bl) => bl,
            Err(e) => return Err(e.into()),
        };
        let mut obj = ContinuousPSI {
            baseline,
            rt_bins: Vec::new(),
        };
        obj.init_runtime_containers();
        Ok(obj)
    }

    fn reset_baseline<'py>(&mut self, baseline_data: PyReadonlyArray1<'py, f64>) -> PyResult<()> {
        let baseline_slice = baseline_data.as_slice()?;
        if let Err(e) = self.baseline.reset(baseline_slice) {
            return Err(e.into());
        };
        self.init_runtime_containers();
        Ok(())
    }

    fn compute_psi_drift<'py>(
        &mut self,
        runtime_data: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<f64> {
        let runtime_data_slice = runtime_data.as_slice()?;
        let n = runtime_data_slice.len() as f64;
        if n == 0_f64 {
            return Err(PSIError::EmptyRuntimeData.into());
        }

        self.build_rt_hist(runtime_data_slice);

        let psi = compute_psi(&self.baseline.baseline_hist, &self.rt_bins, n);
        // reset runtime state after every discrete computation
        self.clear_rt();
        Ok(psi)
    }
}

#[pyclass]
pub struct StreamingContinuousPSI {
    baseline: BaselineContinuousPSI,
    stream_bins: Vec<f64>,
    total_stream_size: usize,
    last_flush_ts: i64,
    flush_rate: i64,
}

impl StreamingContinuousPSI {
    // zero out all bins
    fn flush_runtime_stream(&mut self) {
        self.stream_bins.fill(0_f64);
        self.total_stream_size = 0;
    }

    fn update_stream_bins(&mut self, data_slice: &[f64]) {
        // accumulate bin count based on sample bin
        let data_size = data_slice.len();
        for item in data_slice {
            let idx = self.baseline.resolve_bin(*item);
            self.stream_bins[idx] += 1_f64;
        }
        self.total_stream_size += data_size;
    }

    #[inline]
    fn normalize(&mut self) -> Result<f64, PSIError> {
        let psi = compute_psi(
            &self.baseline.baseline_hist,
            &self.stream_bins,
            self.total_stream_size as f64,
        );
        Ok(psi)
    }
}

#[pymethods]
impl StreamingContinuousPSI {
    #[new]
    #[pyo3(signature = (n_bins, baseline_data, flush_cadence))]
    fn new<'py>(
        n_bins: usize,
        baseline_data: PyReadonlyArray1<'py, f64>,
        flush_cadence: Option<i64>,
    ) -> PyResult<StreamingContinuousPSI> {
        let flush_rate = flush_cadence.unwrap_or_else(|| DEFAULT_STREAM_FLUSH);
        let baseline_slice = baseline_data.as_slice()?;
        let baseline = match BaselineContinuousPSI::new(n_bins, baseline_slice) {
            Ok(bl) => bl,
            Err(e) => return Err(e.into()),
        };
        let total_stream_size = 0_usize;
        let last_flush_ts: i64 = Utc::now().timestamp().into();

        let bl_hist_len = baseline.baseline_hist.len();
        let stream_bins: Vec<f64> = vec![0_f64; bl_hist_len];

        Ok(StreamingContinuousPSI {
            stream_bins,
            baseline,
            total_stream_size,
            last_flush_ts,
            flush_rate,
        })
    }

    #[pyo3(signature = (baseline_data))]
    fn reset_baseline<'py>(&mut self, baseline_data: PyReadonlyArray1<'py, f64>) -> PyResult<()> {
        if let Err(e) = self.baseline.reset(baseline_data.as_slice()?) {
            return Err(e.into());
        };
        self.stream_bins = vec![0_f64; self.baseline.baseline_hist.len()];
        self.total_stream_size = 0;
        Ok(())
    }

    #[pyo3(signature = (runtime_data))]
    fn update_stream<'py>(&mut self, runtime_data: PyReadonlyArray1<'py, f64>) -> PyResult<f64> {
        let runtime_data_slice = runtime_data.as_slice()?;
        if runtime_data_slice.len() == 0 {
            return Err(PSIError::EmptyRuntimeData.into());
        }

        let curr_ts: i64 = Utc::now().timestamp();

        if curr_ts > (self.last_flush_ts + self.flush_rate)
            || (self.total_stream_size + runtime_data_slice.len()) > MAX_STREAM_SIZE
        {
            // reset and flush
            self.flush_runtime_stream();
            self.last_flush_ts = curr_ts;
        }
        self.update_stream_bins(&runtime_data_slice);

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

// let user specify other label
// otherwise use preset label with uuid to be sure
const OTHER_LABEL: Lazy<String> = Lazy::new(|| {
    let uuid = Uuid::new_v4();
    let default = format!("__fairperf_othercat__{}", uuid);
    let Ok(var) = std::env::var("FAIR_PERF_OTHER_LABEL") else {
        return default;
    };
    var
});

//TODO:
//create an error type to distinguish between malformed bins and no runtime data in stream

// store bins as vec for better performance on psi computation and bin accumulation
// store cat label to index in map
struct BaselineCategoricalPSI {
    idx_map: HashMap<String, usize>,
    baseline_bins: Vec<f64>,
}

impl BaselineCategoricalPSI {
    fn new(baseline_data: Vec<String>) -> BaselineCategoricalPSI {
        let n = baseline_data.len() as f64;

        let predicted_capacity = compute_expected_categorical_bins(n);
        let mut initial_bins: HashMap<String, f64> = HashMap::with_capacity(predicted_capacity);

        for cat in baseline_data.into_iter() {
            if let Some(count) = initial_bins.get_mut(&cat) {
                *count += 1_f64;
            } else {
                initial_bins.insert(cat.clone(), 1_f64);
            }
        }
        initial_bins.insert(OTHER_LABEL.clone(), 0_f64);

        let mut idx_map: HashMap<String, usize> = HashMap::with_capacity(initial_bins.len());
        let mut baseline_bins: Vec<f64> = Vec::with_capacity(initial_bins.len());

        let mut i = 0_usize;
        for (key, count) in initial_bins.into_iter() {
            idx_map.insert(key, i);
            baseline_bins.push(count / n);
            i += 1;
        }

        BaselineCategoricalPSI {
            idx_map,
            baseline_bins,
        }
    }

    fn reset(&mut self, baseline_data: Vec<String>) {
        let n = baseline_data.len() as f64;
        let new_predicted_capacity = compute_expected_categorical_bins(n);

        self.idx_map.clear();

        let mut initial_bins: HashMap<String, f64> = HashMap::with_capacity(new_predicted_capacity);
        for cat in baseline_data.into_iter() {
            if let Some(count) = initial_bins.get_mut(&cat) {
                *count += 1_f64;
            } else {
                initial_bins.insert(cat.clone(), 1_f64);
            }
        }
        initial_bins.insert(OTHER_LABEL.clone(), 0_f64);

        self.baseline_bins.clear();
        let mut i = 0_usize;
        for (key, count) in initial_bins.into_iter() {
            self.idx_map.insert(key, i);
            self.baseline_bins.push(count / n);
            i += 1;
        }
    }

    #[inline]
    fn normalize(&self, rt_bins: &[f64], n: f64) -> f64 {
        compute_psi(&self.baseline_bins, rt_bins, n)
    }
}

#[pyclass]
pub struct CategoricalPSI {
    baseline: BaselineCategoricalPSI,
    rt_bins: Vec<f64>,
}

impl CategoricalPSI {
    fn compute_rt(&mut self, runtime_data: Vec<String>) -> Option<f64> {
        let n = runtime_data.len() as f64;

        let other_idx = self.baseline.idx_map[OTHER_LABEL.as_str()];
        for cat in runtime_data.into_iter() {
            let i = *self
                .baseline
                .idx_map
                .get(&cat)
                .unwrap_or_else(|| &other_idx);

            self.rt_bins[i] += 1_f64;
        }

        Some(self.baseline.normalize(&self.rt_bins, n))
    }

    fn clear_rt(&mut self) {
        self.rt_bins.fill(0_f64);
    }
}

#[pymethods]
impl CategoricalPSI {
    #[new]
    fn new(baseline_data: Vec<String>) -> PyResult<CategoricalPSI> {
        if baseline_data.is_empty() {
            return Err(PyValueError::new_err("Baseline data must be non empty"));
        }

        let baseline = BaselineCategoricalPSI::new(baseline_data);
        let num_bins = baseline.baseline_bins.len();
        let rt_bins: Vec<f64> = vec![0_f64; num_bins];

        Ok(CategoricalPSI { baseline, rt_bins })
    }

    fn reset_baseline(&mut self, new_baseline: Vec<String>) {
        self.baseline.reset(new_baseline);
        let num_bins = self.baseline.baseline_bins.len();

        // pay the cost to reallocate bins in order to have correct size
        self.rt_bins = vec![0_f64; num_bins];
    }

    fn compute_psi_drift(&mut self, runtime_data: Vec<String>) -> PyResult<f64> {
        if runtime_data.is_empty() {
            return Err(PSIError::EmptyRuntimeData.into());
        }
        let psi = match self.compute_rt(runtime_data) {
            Some(res) => Ok(res),
            None => Err(PSIError::MalformedRuntimeData.into()),
        };

        self.clear_rt();
        psi
    }

    #[getter]
    fn other_bucket_label(&self) -> String {
        OTHER_LABEL.clone()
    }
}

#[pyclass]
pub struct StreamingCategoricalPSI {
    baseline: BaselineCategoricalPSI,
    stream_bins: Vec<f64>,
    total_stream_size: usize,
    last_flush_ts: i64,
    flush_rate: i64,
}

impl StreamingCategoricalPSI {
    fn init_stream_bins(&mut self) {
        self.stream_bins = vec![0_f64; self.baseline.baseline_bins.len()]
    }

    fn flush_runtime_stream(&mut self) {
        self.stream_bins.fill(0_f64);
        self.total_stream_size = 0;
    }

    fn update_stream_bins(&mut self, runtime_data: Vec<String>) {
        let n = runtime_data.len();
        let other_idx = self.baseline.idx_map[OTHER_LABEL.as_str()];
        for cat in runtime_data.into_iter() {
            let idx = *self
                .baseline
                .idx_map
                .get(&cat)
                .unwrap_or_else(|| &other_idx);
            self.stream_bins[idx] += 1_f64;
        }

        self.total_stream_size += n;
    }

    fn normalize(&self) -> f64 {
        self.baseline
            .normalize(&self.stream_bins, self.total_stream_size as f64)
    }
}

#[pymethods]
impl StreamingCategoricalPSI {
    #[new]
    fn new(
        baseline_data: Vec<String>,
        user_flush_rate: Option<i64>,
    ) -> PyResult<StreamingCategoricalPSI> {
        let baseline = BaselineCategoricalPSI::new(baseline_data);
        let flush_rate = user_flush_rate.unwrap_or_else(|| DEFAULT_STREAM_FLUSH);
        let n_bins = baseline.baseline_bins.len();
        let stream_bins: Vec<f64> = vec![0_f64; n_bins];

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
        self.baseline.reset(new_baseline);
        self.init_stream_bins();
    }

    fn update_stream(&mut self, runtime_data: Vec<String>) -> f64 {
        let curr_ts: i64 = Utc::now().timestamp();

        if curr_ts > (self.last_flush_ts + self.flush_rate)
            || (self.total_stream_size + runtime_data.len()) > MAX_STREAM_SIZE
        {
            // reset and flush
            self.flush_runtime_stream();
            self.last_flush_ts = curr_ts;
        }
        self.update_stream_bins(runtime_data);

        self.normalize()
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
        OTHER_LABEL.clone()
    }
}
