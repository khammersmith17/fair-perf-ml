use super::{
    compute_psi, BaselineCategoricalBins, BaselineContinuousBins, StringLike, DEFAULT_STREAM_FLUSH,
    MAX_STREAM_SIZE,
};
use crate::errors::DriftError;
use ahash::{HashMap, HashMapExt};
use chrono::{DateTime, Utc};

/*
* All types in this module provide core logic implementation in rust and expose
* an api to Python contexts via a pyclass wrapper
* */

#[cfg(feature = "python")]
pub(crate) mod py_api {
    use ahash::HashMap;
    use chrono::{DateTime, Utc};
    use numpy::PyReadonlyArray1;
    use pyo3::{prelude::*, pyclass, pymethods};

    use super::{
        CategoricalPSI, ContinuousPSI, DriftError, StreamingCategoricalPSI, StreamingContinuousPSI,
    };

    #[pyclass]
    pub(crate) struct PyContinuousPSI {
        inner: ContinuousPSI,
    }

    // exposes python APIs to the python type
    // encapsulates all rust logic
    /// Python exposed api for discrete categorical PSI
    #[pymethods]
    impl PyContinuousPSI {
        #[new]
        pub fn new<'py>(
            n_bins: usize,
            baseline_data: PyReadonlyArray1<'py, f64>,
        ) -> PyResult<PyContinuousPSI> {
            let bl_slice = baseline_data.as_slice()?;
            let inner = match ContinuousPSI::new_from_baseline(n_bins, bl_slice) {
                Ok(psi) => psi,
                Err(e) => return Err(e.into()),
            };

            Ok(Self { inner })
        }

        fn reset_baseline<'py>(
            &mut self,
            baseline_data: PyReadonlyArray1<'py, f64>,
        ) -> PyResult<()> {
            let bl_slice = baseline_data.as_slice()?;
            if let Err(e) = self.inner.reset_baseline(bl_slice) {
                return Err(e.into());
            }
            Ok(())
        }

        fn compute_psi_drift<'py>(
            &mut self,
            runtime_data: PyReadonlyArray1<'py, f64>,
        ) -> PyResult<f64> {
            let runtime_data_slice = runtime_data.as_slice()?;
            let drift = match self.inner.compute_psi_drift(runtime_data_slice) {
                Ok(psi_drift) => psi_drift,
                Err(e) => return Err(e.into()),
            };
            Ok(drift)
        }

        #[getter]
        fn num_bins(&self) -> usize {
            self.inner.n_bins()
        }
    }

    /// Exposed Python APIs for streaming continuous PSI
    #[pyclass]
    pub(crate) struct PyStreamingContinuousPSI {
        inner: StreamingContinuousPSI,
    }

    #[pymethods]
    impl PyStreamingContinuousPSI {
        #[new]
        #[pyo3(signature = (n_bins, baseline_data, flush_cadence))]
        fn new<'py>(
            n_bins: usize,
            baseline_data: PyReadonlyArray1<'py, f64>,
            flush_cadence: Option<i64>,
        ) -> PyResult<PyStreamingContinuousPSI> {
            let baseline_slice = baseline_data.as_slice()?;
            let inner = match StreamingContinuousPSI::new(n_bins, baseline_slice, flush_cadence) {
                Ok(psi) => psi,
                Err(e) => return Err(e.into()),
            };

            Ok(Self { inner })
        }

        #[pyo3(signature = (baseline_data))]
        fn reset_baseline<'py>(
            &mut self,
            baseline_data: PyReadonlyArray1<'py, f64>,
        ) -> PyResult<()> {
            let baseline_slice = baseline_data.as_slice()?;
            if let Err(e) = self.inner.reset_baseline(baseline_slice) {
                return Err(e.into());
            };

            Ok(())
        }

        #[pyo3(signature = (runtime_data))]
        fn update_stream<'py>(
            &mut self,
            runtime_data: PyReadonlyArray1<'py, f64>,
        ) -> PyResult<f64> {
            let runtime_data_slice = runtime_data.as_slice()?;
            match self.inner.update_stream(runtime_data_slice) {
                Ok(drift) => Ok(drift),
                Err(e) => Err(e.into()),
            }
        }

        fn flush(&mut self) {
            self.inner.flush();
        }

        #[getter]
        fn total_samples(&self) -> usize {
            self.inner.total_samples()
        }

        #[getter]
        fn last_flush(&self) -> PyResult<DateTime<Utc>> {
            let Ok(ts) = self.inner.last_flush() else {
                return Err(DriftError::DateTimeError.into());
            };
            Ok(ts)
        }

        #[getter]
        fn n_bins(&self) -> usize {
            self.inner.n_bins()
        }

        fn export_snapshot(&self) -> HashMap<String, Vec<f64>> {
            // determine snapshot shape
            self.inner.export_snapshot()
        }
    }

    /// Python exposed api for discrete categorical PSI
    #[pyclass]
    pub(crate) struct PyCategoricalPSI {
        inner: CategoricalPSI,
    }

    #[pymethods]
    impl PyCategoricalPSI {
        #[new]
        #[pyo3(signature = (baseline_data))]
        fn new(baseline_data: Vec<String>) -> PyResult<PyCategoricalPSI> {
            let inner = match CategoricalPSI::new(&baseline_data) {
                Ok(psi) => psi,
                Err(e) => return Err(e.into()),
            };

            Ok(Self { inner })
        }

        #[pyo3(signature = (new_baseline))]
        fn reset_baseline(&mut self, new_baseline: Vec<String>) {
            self.inner.reset_baseline(&new_baseline);
        }

        #[pyo3(signature = (runtime_data))]
        fn compute_psi_drift(&mut self, runtime_data: Vec<String>) -> PyResult<f64> {
            match self.inner.compute_psi_drift(&runtime_data) {
                Ok(psi) => Ok(psi),
                Err(e) => Err(e.into()),
            }
        }

        #[getter]
        fn other_bucket_label(&self) -> String {
            self.inner.other_bucket_label().clone()
        }

        fn export_baseline(&self) -> HashMap<String, f64> {
            self.inner.baseline.export_baseline()
        }
    }

    /// Exposed Python APIs for streaming categorical PSI
    #[pyclass]
    pub(crate) struct PyStreamingCategoricalPSI {
        inner: StreamingCategoricalPSI,
    }

    #[pymethods]
    impl PyStreamingCategoricalPSI {
        #[new]
        fn new(
            baseline_data: Vec<String>,
            flush_rate: Option<i64>,
        ) -> PyResult<PyStreamingCategoricalPSI> {
            let inner = match StreamingCategoricalPSI::new(&baseline_data, flush_rate) {
                Ok(psi) => psi,
                Err(e) => return Err(e.into()),
            };

            Ok(Self { inner })
        }

        #[pyo3(signature = (new_baseline))]
        fn reset_baseline(&mut self, new_baseline: Vec<String>) {
            self.inner.reset_baseline(&new_baseline);
        }

        #[pyo3(signature = (runtime_data))]
        fn update_stream(&mut self, runtime_data: Vec<String>) -> f64 {
            self.inner.update_stream(&runtime_data)
        }

        fn flush(&mut self) {
            self.inner.flush()
        }

        #[getter]
        fn total_samples(&self) -> usize {
            self.inner.total_samples()
        }

        #[getter]
        fn last_flush(&self) -> PyResult<DateTime<Utc>> {
            let Ok(ts) = self.inner.last_flush() else {
                return Err(DriftError::DateTimeError.into());
            };
            Ok(ts)
        }

        #[getter]
        fn other_bucket_label(&self) -> String {
            self.inner.other_bucket_label().clone()
        }

        fn export_snapshot(&self) -> HashMap<String, f64> {
            self.inner.export_snapshot()
        }

        fn export_baseline(&self) -> HashMap<String, f64> {
            self.inner.export_baseline()
        }
    }
}

// lazily evaluate bin ratio
// cheap computation and saves space/memory lookup

pub struct ContinuousPSI {
    baseline: BaselineContinuousBins,
    rt_bins: Vec<f64>,
}

impl ContinuousPSI {
    pub fn new_from_baseline(n_bins: usize, bl_slice: &[f64]) -> Result<ContinuousPSI, DriftError> {
        let baseline = BaselineContinuousBins::new(n_bins, &bl_slice)?;
        let mut obj = ContinuousPSI {
            baseline,
            rt_bins: Vec::new(),
        };
        obj.init_runtime_containers();
        Ok(obj)
    }

    fn clear_rt(&mut self) {
        self.rt_bins.fill(0_f64);
    }

    #[inline]
    fn build_rt_hist(&mut self, data_slice: &[f64]) {
        for item in data_slice {
            let idx = self.baseline.resolve_bin(*item);
            self.rt_bins[idx] += 1_f64;
        }
    }

    fn init_runtime_containers(&mut self) {
        let len = self.baseline.baseline_hist.len();
        self.rt_bins = vec![0_f64; len];
    }

    pub fn reset_baseline(&mut self, baseline_slice: &[f64]) -> Result<(), DriftError> {
        if let Err(e) = self.baseline.reset(baseline_slice) {
            return Err(e.into());
        };
        self.init_runtime_containers();
        Ok(())
    }

    pub fn compute_psi_drift(&mut self, runtime_slice: &[f64]) -> Result<f64, DriftError> {
        let n = runtime_slice.len() as f64;
        if n == 0_f64 {
            return Err(DriftError::EmptyRuntimeData);
        }

        self.build_rt_hist(runtime_slice);

        let psi = compute_psi(&self.baseline.baseline_hist, &self.rt_bins, n);
        // reset runtime state after every discrete computation
        self.clear_rt();
        Ok(psi)
    }

    pub fn n_bins(&self) -> usize {
        self.baseline.n_bins
    }
}

pub struct StreamingContinuousPSI {
    baseline: BaselineContinuousBins,
    stream_bins: Vec<f64>,
    total_stream_size: usize,
    last_flush_ts: i64,
    flush_rate: i64,
}

impl StreamingContinuousPSI {
    pub fn new(
        n_bins: usize,
        baseline_slice: &[f64],
        flush_cadence: Option<i64>,
    ) -> Result<StreamingContinuousPSI, DriftError> {
        let flush_rate = flush_cadence.unwrap_or_else(|| DEFAULT_STREAM_FLUSH);
        let baseline = match BaselineContinuousBins::new(n_bins, baseline_slice) {
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

    pub fn reset_baseline(&mut self, baseline_slice: &[f64]) -> Result<(), DriftError> {
        if let Err(e) = self.baseline.reset(baseline_slice) {
            return Err(e.into());
        };
        self.stream_bins = vec![0_f64; self.baseline.baseline_hist.len()];
        self.total_stream_size = 0;
        self.last_flush_ts = Utc::now().timestamp();
        Ok(())
    }

    pub fn update_stream(&mut self, runtime_slice: &[f64]) -> Result<f64, DriftError> {
        if runtime_slice.len() == 0 {
            return Err(DriftError::EmptyRuntimeData);
        }

        let curr_ts: i64 = Utc::now().timestamp();

        if curr_ts > (self.last_flush_ts + self.flush_rate)
            || (self.total_stream_size + runtime_slice.len()) > MAX_STREAM_SIZE
        {
            // reset and flush
            self.flush_runtime_stream();
            self.last_flush_ts = curr_ts;
        }
        self.update_stream_bins(runtime_slice);

        Ok(self.normalize()?)
    }

    pub fn flush(&mut self) {
        self.flush_runtime_stream();
        self.last_flush_ts = Utc::now().timestamp();
    }

    pub fn total_samples(&self) -> usize {
        self.total_stream_size
    }

    pub fn last_flush(&self) -> Result<DateTime<Utc>, DriftError> {
        let Some(ts) = DateTime::from_timestamp(self.last_flush_ts, 0_u32) else {
            return Err(DriftError::DateTimeError);
        };
        Ok(ts)
    }

    pub fn n_bins(&self) -> usize {
        self.baseline.n_bins
    }

    pub fn export_snapshot(&self) -> HashMap<String, Vec<f64>> {
        // determine snapshot shape
        let mut table: HashMap<String, Vec<f64>> = HashMap::with_capacity(3);
        table.insert("binEdges".into(), self.baseline.bin_edges.clone());
        table.insert("baselineBins".into(), self.baseline.baseline_hist.clone());
        table.insert("streamBins".into(), self.stream_bins.clone());
        table
    }

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
    fn normalize(&mut self) -> Result<f64, DriftError> {
        let psi = compute_psi(
            &self.baseline.baseline_hist,
            &self.stream_bins,
            self.total_stream_size as f64,
        );
        Ok(psi)
    }
}

// using a log scaling heuristic to preallocate

// store bins as vec for better performance on psi computation and bin accumulation
// store cat label to index in map

pub struct CategoricalPSI {
    baseline: BaselineCategoricalBins,
    rt_bins: Vec<f64>,
}

impl CategoricalPSI {
    pub fn new<S: StringLike>(baseline_data: &[S]) -> Result<CategoricalPSI, DriftError> {
        if baseline_data.is_empty() {
            return Err(DriftError::EmptyBaselineData);
        }

        let baseline = BaselineCategoricalBins::new(baseline_data);
        let num_bins = baseline.baseline_bins.len();
        let rt_bins: Vec<f64> = vec![0_f64; num_bins];

        Ok(CategoricalPSI { baseline, rt_bins })
    }

    fn compute_rt<S: StringLike>(&mut self, runtime_data: &[S]) -> Option<f64> {
        let n = runtime_data.len() as f64;

        let other_idx = self.baseline.idx_map[self.baseline.other_label.as_str()];
        for cat in runtime_data.iter() {
            let i = *self
                .baseline
                .idx_map
                .get(cat.as_ref())
                .unwrap_or_else(|| &other_idx);

            self.rt_bins[i] += 1_f64;
        }

        Some(self.baseline.normalize(&self.rt_bins, n))
    }

    fn clear_rt(&mut self) {
        self.rt_bins.fill(0_f64);
    }

    pub fn reset_baseline<S: StringLike>(&mut self, new_baseline: &[S]) {
        self.baseline.reset(new_baseline);
        let num_bins = self.baseline.baseline_bins.len();

        // pay the cost to reallocate bins in order to have correct size
        // not common path
        self.rt_bins = vec![0_f64; num_bins];
    }

    pub fn compute_psi_drift<S: StringLike>(
        &mut self,
        runtime_data: &[S],
    ) -> Result<f64, DriftError> {
        // will not compute on empty data
        if runtime_data.is_empty() {
            return Err(DriftError::EmptyRuntimeData.into());
        }
        let psi = match self.compute_rt(runtime_data) {
            Some(res) => Ok(res),
            None => Err(DriftError::MalformedRuntimeData.into()),
        };

        self.clear_rt();
        psi
    }

    pub fn other_bucket_label(&self) -> String {
        self.baseline.other_label.clone()
    }

    pub fn export_baseline(&self) -> HashMap<String, f64> {
        self.baseline.export_baseline()
    }
}

pub struct StreamingCategoricalPSI {
    baseline: BaselineCategoricalBins,
    stream_bins: Vec<f64>,
    total_stream_size: usize,
    last_flush_ts: i64,
    flush_rate: i64,
}

impl StreamingCategoricalPSI {
    pub fn new<S: StringLike>(
        baseline_data: &[S],
        user_flush_rate: Option<i64>,
    ) -> Result<StreamingCategoricalPSI, DriftError> {
        let baseline = BaselineCategoricalBins::new(baseline_data);
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

    pub fn reset_baseline<S: StringLike>(&mut self, new_baseline: &[S]) {
        self.baseline.reset(new_baseline);
        self.init_stream_bins();
    }

    pub fn update_stream<S: StringLike>(&mut self, runtime_data: &[S]) -> f64 {
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

    pub fn flush(&mut self) {
        self.flush_runtime_stream();
        self.last_flush_ts = Utc::now().timestamp();
    }

    pub fn total_samples(&self) -> usize {
        self.total_stream_size
    }

    pub fn last_flush(&self) -> Result<DateTime<Utc>, DriftError> {
        let Some(ts) = DateTime::from_timestamp(self.last_flush_ts, 0_u32) else {
            return Err(DriftError::DateTimeError);
        };
        Ok(ts)
    }

    pub fn other_bucket_label(&self) -> String {
        self.baseline.other_label.clone()
    }

    pub fn export_snapshot(&self) -> HashMap<String, f64> {
        self.baseline
            .idx_map
            .iter()
            .map(|(feat_name, i)| (feat_name.clone(), self.stream_bins[*i]))
            .collect()
    }

    pub fn export_baseline(&self) -> HashMap<String, f64> {
        self.baseline.export_baseline()
    }

    fn init_stream_bins(&mut self) {
        self.stream_bins = vec![0_f64; self.baseline.baseline_bins.len()]
    }

    pub fn flush_runtime_stream(&mut self) {
        self.stream_bins.fill(0_f64);
        self.total_stream_size = 0;
    }

    fn update_stream_bins<S: StringLike>(&mut self, runtime_data: &[S]) {
        let n = runtime_data.len();
        let other_idx = self.baseline.idx_map[self.baseline.other_label.as_str()];
        for cat in runtime_data.into_iter() {
            let idx = *self
                .baseline
                .idx_map
                .get(cat.as_ref())
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

#[cfg(test)]
mod continuous_tests {
    use super::*;

    #[test]
    fn test_continuous_baseline_builds_expected_bins() {
        let baseline = [1.0, 2.0, 3.0, 4.0];
        let psi = ContinuousPSI::new_from_baseline(3, &baseline).unwrap();

        // 3 bins → 4 edges
        assert_eq!(psi.baseline.bin_edges.len(), 4);
        assert_eq!(psi.rt_bins.len(), 3);
    }

    #[test]
    fn test_continuous_psi_zero_when_no_drift() {
        let baseline = [1.0, 2.0, 3.0, 4.0];
        let mut psi = ContinuousPSI::new_from_baseline(3, &baseline).unwrap();
        let runtime = [1.0, 2.0, 3.0, 4.0];

        let drift = psi.compute_psi_drift(&runtime).unwrap();
        assert!(drift.abs() < 1e-9);
    }

    #[test]
    fn test_continuous_psi_detects_shift() {
        let baseline = [1.0, 2.0, 3.0, 4.0];
        let mut psi = ContinuousPSI::new_from_baseline(3, &baseline).unwrap();
        let runtime = [10.0, 11.0, 12.0, 13.0];

        let drift = psi.compute_psi_drift(&runtime).unwrap();
        assert!(drift > 0.5);
    }

    #[test]
    fn test_streaming_continuous_accumulation() {
        let baseline = [1_f64, 2_f64, 3_f64, 3_f64, 4_f64];
        let mut streaming = StreamingContinuousPSI::new(3, &baseline, None).unwrap();

        let d1 = streaming.update_stream(&[1.0, 2.0, 2.0, 3.0, 4.0]).unwrap();
        let d2 = streaming
            .update_stream(&[3.0, 4.0, 2.0, 2.0, 1.0, 3.0])
            .unwrap();

        assert!(d1.abs() < 1e-9);
        assert!(d2.abs() < 1e-2);
        assert_eq!(streaming.total_samples(), 11);
    }

    #[test]
    fn test_streaming_flush() {
        let baseline = [1.0, 2.0, 3.0, 4.0];
        let mut streaming = StreamingContinuousPSI::new(3, &baseline, None).unwrap();

        streaming.update_stream(&[1.0, 2.0, 3.0]).unwrap();
        streaming.flush();

        assert_eq!(streaming.total_samples(), 0);
    }
}

#[cfg(test)]
mod categorical_tests {
    use super::*;

    #[test]
    fn test_categorical_baseline_builds_expected_size() {
        let baseline = ["a", "b", "a", "c"];
        let psi = CategoricalPSI::new(&baseline).unwrap();

        // baseline has 3 real labels + OTHER bucket
        assert_eq!(psi.baseline.baseline_bins.len(), 4);
    }

    #[test]
    fn test_categorical_psi_zero_when_no_drift() {
        let baseline = ["a", "b", "a", "c"];
        let mut psi = CategoricalPSI::new(&baseline).unwrap();
        let runtime = ["a", "b", "a", "c"];

        let drift = psi.compute_psi_drift(&runtime).unwrap();
        assert!(drift.abs() < 1e-9);
    }

    #[test]
    fn test_categorical_psi_detects_shift() {
        let baseline = ["a", "b", "a", "c"];
        let mut psi = CategoricalPSI::new(&baseline).unwrap();
        let runtime = ["x", "x", "x", "x"]; // go to other bucket

        let drift = psi.compute_psi_drift(&runtime).unwrap();
        assert!(drift > 0.5);
    }

    #[test]
    fn test_other_bucket_label_exposed() {
        let baseline = ["a", "b"];
        let psi = CategoricalPSI::new(&baseline).unwrap();
        let other = psi.other_bucket_label();

        assert!(other.starts_with("__fairperf_othercat__"));
    }

    #[test]
    fn test_streaming_categorical_accumulation() {
        let baseline = ["a", "b"];
        let mut streaming = StreamingCategoricalPSI::new(&baseline, None).unwrap();

        let d1 = streaming.update_stream(&["a", "b"]);
        let mut stream = Vec::new();

        for _ in 0..500 {
            stream.push("a")
        }

        for _ in 0..490 {
            stream.push("b")
        }
        let d2 = streaming.update_stream(&stream);

        assert_eq!(streaming.total_samples(), 992);
        assert!(d1 < 1e-9);
        assert!(d2 < 1e-2);
    }
}
