use super::{
    baseline::{BaselineCategoricalBins, BaselineContinuousBins},
    drift_metrics::{
        compute_jensen_shannon_divergence_drift, compute_kl_divergence_drift, compute_psi,
        CategoricalJensenShannonDivergenceDrift, CategoricalKlDivergenceDrift, CategoricalPSIDrift,
        ContinuousJensenShannonDivergenceDrift, ContinuousKlDivergenceDrift, ContinuousPSIDrift,
        StreamingJensenShannonDivergenceDrift, StreamingKlDivergenceDrift,
        StreamingPopulationStabilityIndexDrift,
    },
    StringLike, DEFAULT_STREAM_FLUSH, MAX_STREAM_SIZE,
};
use crate::errors::DriftError;
use ahash::{HashMap, HashMapExt};
use chrono::{DateTime, Utc};

#[cfg(feature = "python")]
pub(crate) mod py_api {

    //TODO: implement an entry point for all drift methods here
    use ahash::HashMap;
    use chrono::{DateTime, Utc};
    use numpy::PyReadonlyArray1;
    use pyo3::{prelude::*, pyclass, pymethods};

    use super::{
        CategoricalDataDrift, CategoricalKlDivergenceDrift, CategoricalPSIDrift,
        ContinuousDataDrift, ContinuousKlDivergenceDrift, ContinuousPSIDrift, DriftError,
        StreamingCategoricalDataDrift, StreamingContinuousDataDrift, StreamingKlDivergenceDrift,
        StreamingPopulationStabilityIndexDrift,
    };

    #[pyclass]
    pub(crate) struct PyContinuousDataDrift {
        inner: ContinuousDataDrift,
    }

    // exposes python APIs to the python type
    // encapsulates all rust logic
    /// Python exposed api for discrete categorical PSI

    //  TODO: expose all drift methods here, thus need to bring in all trait implementations.
    #[pymethods]
    impl PyContinuousDataDrift {
        #[new]
        pub fn new<'py>(
            n_bins: usize,
            baseline_data: PyReadonlyArray1<'py, f64>,
        ) -> PyResult<PyContinuousDataDrift> {
            let bl_slice = baseline_data.as_slice()?;
            let inner = match ContinuousDataDrift::new_from_baseline(n_bins, bl_slice) {
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
            let psi_drift = self.inner.psi_drift(runtime_data_slice)?;

            Ok(psi_drift)
        }

        fn compute_kl_divergence_drift<'py>(
            &mut self,
            runtime_data: PyReadonlyArray1<'py, f64>,
        ) -> PyResult<f64> {
            let runtime_data_slice = runtime_data.as_slice()?;
            let kl_drift = self.inner.kl_divergence_drift(runtime_data_slice)?;

            Ok(kl_drift)
        }

        fn export_baseline(&self) -> Vec<f64> {
            self.inner.export_baseline()
        }

        #[getter]
        fn num_bins(&self) -> usize {
            self.inner.n_bins()
        }
    }

    /*
     * reset_baseline
     * update_stream
     * update_stream_batch
     *  compute_psi_drift
     * compute_kl_divergence_drift
     * flush
     * total_samples
     * last_flush
     * n_bins
     * export_snapshot
     * export_baseline
     * */

    /// Exposed Python APIs for streaming continuous PSI
    #[pyclass]
    pub(crate) struct PyStreamingContinuousDataDrift {
        inner: StreamingContinuousDataDrift,
    }

    #[pymethods]
    impl PyStreamingContinuousDataDrift {
        #[new]
        #[pyo3(signature = (n_bins, baseline_data, flush_cadence))]
        fn new<'py>(
            n_bins: usize,
            baseline_data: PyReadonlyArray1<'py, f64>,
            flush_cadence: Option<i64>,
        ) -> PyResult<PyStreamingContinuousDataDrift> {
            let baseline_slice = baseline_data.as_slice()?;
            let inner =
                match StreamingContinuousDataDrift::new(n_bins, baseline_slice, flush_cadence) {
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
        fn update_stream_batch<'py>(
            &mut self,
            runtime_data: PyReadonlyArray1<'py, f64>,
        ) -> PyResult<()> {
            let runtime_data_slice = runtime_data.as_slice()?;
            self.inner.update_stream_batch(runtime_data_slice)?;
            Ok(())
        }

        #[pyo3(signature = (runtime_example))]
        fn update_stream(&mut self, runtime_example: f64) {
            self.inner.update_stream(runtime_example);
        }

        fn compute_psi_drift(&self) -> PyResult<f64> {
            let psi_drift = self.inner.psi_drift()?;
            Ok(psi_drift)
        }

        fn compute_kl_divergence_drift(&self) -> PyResult<f64> {
            let kl_drift = self.inner.kl_divergence_drift()?;
            Ok(kl_drift)
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

        fn export_baseline(&self) -> Vec<f64> {
            self.inner.export_baseline()
        }
    }

    /// Python exposed api for discrete categorical PSI
    #[pyclass]
    pub(crate) struct PyCategoricalDataDrift {
        inner: CategoricalDataDrift,
    }

    #[pymethods]
    impl PyCategoricalDataDrift {
        #[new]
        #[pyo3(signature = (baseline_data))]
        fn new(baseline_data: Vec<String>) -> PyResult<PyCategoricalDataDrift> {
            let inner = match CategoricalDataDrift::new(&baseline_data) {
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
        fn compute_psi_drift<'py>(&mut self, runtime_data: Vec<String>) -> PyResult<f64> {
            let psi_drift = self.inner.psi_drift(&runtime_data)?;

            Ok(psi_drift)
        }

        #[pyo3(signature = (runtime_data))]
        fn compute_kl_divergence_drift<'py>(&mut self, runtime_data: Vec<String>) -> PyResult<f64> {
            let kl_drift = self.inner.kl_divergence_drift(&runtime_data)?;

            Ok(kl_drift)
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
    pub(crate) struct PyStreamingCategoricalDataDrift {
        inner: StreamingCategoricalDataDrift,
    }

    #[pymethods]
    impl PyStreamingCategoricalDataDrift {
        #[new]
        fn new(
            baseline_data: Vec<String>,
            flush_rate: Option<i64>,
        ) -> PyResult<PyStreamingCategoricalDataDrift> {
            let inner = match StreamingCategoricalDataDrift::new(&baseline_data, flush_rate) {
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
        fn update_stream_batch(&mut self, runtime_data: Vec<String>) -> PyResult<()> {
            self.inner.update_stream_batch(&runtime_data)?;
            Ok(())
        }

        #[pyo3(signature = (runtime_example))]
        fn update_stream(&mut self, runtime_example: String) {
            self.inner.update_stream(&runtime_example);
        }

        fn compute_psi_drift(&self) -> PyResult<f64> {
            let psi_drift = self.inner.psi_drift()?;
            Ok(psi_drift)
        }

        fn compute_kl_divergence_drift(&self) -> PyResult<f64> {
            let kl_drift = self.inner.kl_divergence_drift()?;
            Ok(kl_drift)
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

/// Data drift type for continuous data. This type keeps a baseline state, and will compute the
/// drift on a discrete dataset.
pub struct ContinuousDataDrift {
    baseline: BaselineContinuousBins,
    rt_bins: Vec<f64>,
}

// Implementation of different drift methods
impl ContinuousPSIDrift for ContinuousDataDrift {
    fn psi_drift(&mut self, runtime_slice: &[f64]) -> Result<f64, DriftError> {
        self.build_rt_hist(runtime_slice)?;

        let psi = compute_psi(
            &self.baseline.baseline_hist,
            &self.rt_bins,
            runtime_slice.len() as f64,
        );
        // reset runtime state after every discrete computation
        self.clear_rt();
        Ok(psi)
    }
}

impl ContinuousKlDivergenceDrift for ContinuousDataDrift {
    fn kl_divergence_drift(&mut self, runtime_slice: &[f64]) -> Result<f64, DriftError> {
        self.build_rt_hist(runtime_slice)?;

        let kl_drift = compute_kl_divergence_drift(
            &self.baseline.baseline_hist,
            &self.rt_bins,
            runtime_slice.len() as f64,
        );
        self.clear_rt();
        Ok(kl_drift)
    }
}

impl ContinuousJensenShannonDivergenceDrift for ContinuousDataDrift {
    fn js_drift(&mut self, runtime_slice: &[f64]) -> Result<f64, DriftError> {
        self.build_rt_hist(runtime_slice)?;

        let js_drift = compute_jensen_shannon_divergence_drift(
            &self.baseline.baseline_hist,
            &self.rt_bins,
            runtime_slice.len() as f64,
        );
        self.clear_rt();
        Ok(js_drift)
    }
}

impl ContinuousDataDrift {
    /// Construct a new instance with the provided baseline set. There will be best effort attempt
    /// to use the provided number of bins, but in the case where a bin may be empty, then the
    /// number of bins might be altered internally. Errors when the provided dataset is empty.
    pub fn new_from_baseline(
        n_bins: usize,
        bl_slice: &[f64],
    ) -> Result<ContinuousDataDrift, DriftError> {
        let baseline = BaselineContinuousBins::new(n_bins, &bl_slice)?;
        let mut obj = ContinuousDataDrift {
            baseline,
            rt_bins: Vec::new(),
        };
        obj.init_runtime_containers();
        Ok(obj)
    }

    fn clear_rt(&mut self) {
        self.rt_bins.fill(0.);
    }

    #[inline]
    fn build_rt_hist(&mut self, runtime_data: &[f64]) -> Result<(), DriftError> {
        if runtime_data.is_empty() {
            return Err(DriftError::EmptyRuntimeData);
        }
        for item in runtime_data {
            let idx = self.baseline.resolve_bin(*item);
            self.rt_bins[idx] += 1_f64;
        }
        Ok(())
    }

    fn init_runtime_containers(&mut self) {
        let len = self.baseline.baseline_hist.len();
        self.rt_bins = vec![0_f64; len];
    }

    /// Reset the baseline state with a new baseline dataset. Same rules apply to the bin count at
    /// construction, but in this instance, a best effort attempt will be made to use the current
    /// number of bins. Errors when the dataset passed in empty.
    pub fn reset_baseline(&mut self, baseline_slice: &[f64]) -> Result<(), DriftError> {
        if let Err(e) = self.baseline.reset(baseline_slice) {
            return Err(e.into());
        };
        self.init_runtime_containers();
        Ok(())
    }

    pub fn n_bins(&self) -> usize {
        self.baseline.n_bins
    }

    pub fn export_baseline(&self) -> Vec<f64> {
        self.baseline.export_baseline()
    }
}

/// A streaming variant of the [`ContinuousDataDrift`] type. This is a stateful "stream" for long
/// running drift monitoring. Every new example pushed into the stream will update state, and a
/// drift snapshot can be computed at any point in time, granted that there is data present in the
/// stream.
pub struct StreamingContinuousDataDrift {
    baseline: BaselineContinuousBins,
    stream_bins: Vec<f64>,
    total_stream_size: usize,
    last_flush_ts: i64,
    flush_rate: i64,
}

// Implementation of different drift methods
impl StreamingPopulationStabilityIndexDrift for StreamingContinuousDataDrift {
    fn psi_drift(&self) -> Result<f64, DriftError> {
        if self.total_stream_size == 0 {
            return Err(DriftError::EmptyRuntimeData);
        }

        Ok(compute_psi(
            &self.baseline.baseline_hist,
            &self.stream_bins,
            self.total_stream_size as f64,
        ))
    }
}

impl StreamingKlDivergenceDrift for StreamingContinuousDataDrift {
    fn kl_divergence_drift(&self) -> Result<f64, DriftError> {
        if self.total_stream_size == 0 {
            return Err(DriftError::EmptyRuntimeData);
        }
        Ok(compute_kl_divergence_drift(
            &self.baseline.baseline_hist,
            &self.stream_bins,
            self.total_stream_size as f64,
        ))
    }
}

impl StreamingJensenShannonDivergenceDrift for StreamingContinuousDataDrift {
    fn js_drift(&self) -> Result<f64, DriftError> {
        if self.total_stream_size == 0 {
            return Err(DriftError::EmptyRuntimeData);
        }
        Ok(compute_jensen_shannon_divergence_drift(
            &self.baseline.baseline_hist,
            &self.stream_bins,
            self.total_stream_size as f64,
        ))
    }
}

impl StreamingContinuousDataDrift {
    /// Construct a new stream. As with the more discrete type, a best effort attempt will be made
    /// to use the desired number of bins. Additionally, a flush cadence can optionally be
    /// provided. In the case is not, the 'DEFAULT_STREAM_FLUSH` constant will be used, which is
    /// once per day. The flush cadence is to prevent overflow in the bin counts, if a large number
    /// of examples are accumulated on a high traffic service.
    pub fn new(
        n_bins: usize,
        baseline_slice: &[f64],
        flush_cadence: Option<i64>,
    ) -> Result<StreamingContinuousDataDrift, DriftError> {
        let flush_rate = flush_cadence.unwrap_or_else(|| DEFAULT_STREAM_FLUSH);
        let baseline = match BaselineContinuousBins::new(n_bins, baseline_slice) {
            Ok(bl) => bl,
            Err(e) => return Err(e.into()),
        };
        let total_stream_size = 0_usize;
        let last_flush_ts: i64 = Utc::now().timestamp().into();

        let bl_hist_len = baseline.baseline_hist.len();
        let stream_bins: Vec<f64> = vec![0_f64; bl_hist_len];

        Ok(StreamingContinuousDataDrift {
            stream_bins,
            baseline,
            total_stream_size,
            last_flush_ts,
            flush_rate,
        })
    }

    /// Reset the baseline with a new baseline dataset. A best effort is made to maintain the same
    /// number of bins, but is subject to the same bin size restrictions as the initial baseline
    /// construction.
    pub fn reset_baseline(&mut self, baseline_slice: &[f64]) -> Result<(), DriftError> {
        if let Err(e) = self.baseline.reset(baseline_slice) {
            return Err(e.into());
        };
        self.stream_bins = vec![0_f64; self.baseline.baseline_hist.len()];
        self.total_stream_size = 0;
        self.last_flush_ts = Utc::now().timestamp();
        Ok(())
    }

    /// Push a single example into the stream.
    #[inline]
    pub fn update_stream(&mut self, runtime_example: f64) {
        let idx = self.baseline.resolve_bin(runtime_example);
        self.stream_bins[idx] += 1_f64;
        self.total_stream_size += 1;
    }

    /// Push a batch dataset to the stream.
    pub fn update_stream_batch(&mut self, runtime_slice: &[f64]) -> Result<(), DriftError> {
        if runtime_slice.is_empty() {
            return Err(DriftError::EmptyRuntimeData);
        }

        let curr_ts: i64 = Utc::now().timestamp();

        if curr_ts > (self.last_flush_ts + self.flush_rate)
            || (self.total_stream_size + runtime_slice.len()) > *MAX_STREAM_SIZE
        {
            // reset and flush
            self.flush_runtime_stream();
            self.last_flush_ts = curr_ts;
        }

        for item in runtime_slice {
            self.update_stream(*item)
        }

        Ok(())
    }

    /// FLush stream. This will clear all runtime bins, this is reccomended every so often to clear
    /// old state and get a more recent view of drift. Baseline state is not altered.
    pub fn flush(&mut self) {
        self.flush_runtime_stream();
        self.last_flush_ts = Utc::now().timestamp();
    }

    /// The number of total samples accumulated in the stream.
    pub fn total_samples(&self) -> usize {
        self.total_stream_size
    }

    /// 'chrono::DataTime<Utc>` timestamp of when the last stream flush occured.
    pub fn last_flush(&self) -> Result<DateTime<Utc>, DriftError> {
        let Some(ts) = DateTime::from_timestamp(self.last_flush_ts, 0_u32) else {
            return Err(DriftError::DateTimeError);
        };
        Ok(ts)
    }

    /// The number of histogram bins.
    pub fn n_bins(&self) -> usize {
        self.baseline.n_bins
    }

    /// Export a snapshot of the stream state. This includes, the baseline bins, the current bin
    /// distribution of the runtime data, and the bin edges that determine the internal histogram binning.
    pub fn export_snapshot(&self) -> HashMap<String, Vec<f64>> {
        // determine snapshot shape
        let mut table: HashMap<String, Vec<f64>> = HashMap::with_capacity(3);
        table.insert("binEdges".into(), self.baseline.bin_edges.clone());
        table.insert("baselineBins".into(), self.export_baseline());
        table.insert("streamBins".into(), self.stream_bins.clone());
        table
    }

    /// Export a the baseline bin proportions. Returns an owned `Vec<f64>`, which contains the
    /// proportional bin distribution present in the baseline set, and thus what all drift metrics
    /// are computed with respect to.
    pub fn export_baseline(&self) -> Vec<f64> {
        self.baseline.export_baseline()
    }

    // zero out all bins
    fn flush_runtime_stream(&mut self) {
        self.stream_bins.fill(0_f64);
        self.total_stream_size = 0;
    }
}

// store bins as vec for better performance on psi computation and bin accumulation
// store cat label to index in map
pub struct CategoricalDataDrift {
    baseline: BaselineCategoricalBins,
    rt_bins: Vec<f64>,
}

// Implementation of different drift methods from the assoicated traits
impl CategoricalPSIDrift for CategoricalDataDrift {
    fn psi_drift<S>(&mut self, runtime_slice: &[S]) -> Result<f64, DriftError>
    where
        S: StringLike,
    {
        self.build_rt_hist(runtime_slice)?;
        let psi = compute_psi(
            &self.baseline.baseline_bins,
            &self.rt_bins,
            runtime_slice.len() as f64,
        );

        self.clear_rt();
        Ok(psi)
    }
}

impl CategoricalKlDivergenceDrift for CategoricalDataDrift {
    fn kl_divergence_drift<S>(&mut self, runtime_slice: &[S]) -> Result<f64, DriftError>
    where
        S: StringLike,
    {
        self.build_rt_hist(runtime_slice)?;
        let kl_drift = compute_kl_divergence_drift(
            &self.baseline.baseline_bins,
            &self.rt_bins,
            runtime_slice.len() as f64,
        );

        self.clear_rt();
        Ok(kl_drift)
    }
}

impl CategoricalJensenShannonDivergenceDrift for CategoricalDataDrift {
    fn js_drift<S: StringLike>(&mut self, runtime_slice: &[S]) -> Result<f64, DriftError> {
        self.build_rt_hist(runtime_slice)?;
        let js_drift = compute_jensen_shannon_divergence_drift(
            &self.baseline.baseline_bins,
            &self.rt_bins,
            runtime_slice.len() as f64,
        );

        self.clear_rt();
        Ok(js_drift)
    }
}

impl CategoricalDataDrift {
    /// Construct a new instance with the provided baseline dataset. [`StringLike`] indicates
    /// something that can be used as a reference to key into a `HashMap<String, f64>`, these
    /// bounds are to allow some other type of label value, such as an enum. The number of bins
    /// will be equal to the number of unique values present in the baseline data set, with an
    /// additional bin for values that occur in the runtime dataset that do not occur in the
    /// baseline dataset. The value associated with this other bucket can be overwritten using the
    /// FAIR_PERF_OTHER_BUCKET environment variable, otherwise is defaults to a constant prefix,
    /// with a uuid suffix for uniqueness.
    pub fn new<S: StringLike>(baseline_data: &[S]) -> Result<CategoricalDataDrift, DriftError> {
        if baseline_data.is_empty() {
            return Err(DriftError::EmptyBaselineData);
        }

        let baseline = BaselineCategoricalBins::new(baseline_data);
        let num_bins = baseline.baseline_bins.len();
        let rt_bins: Vec<f64> = vec![0_f64; num_bins];

        Ok(CategoricalDataDrift { baseline, rt_bins })
    }

    fn build_rt_hist<S: StringLike>(&mut self, runtime_data: &[S]) -> Result<(), DriftError> {
        if runtime_data.is_empty() {
            return Err(DriftError::EmptyRuntimeData.into());
        }
        let other_idx = self.baseline.idx_map[self.baseline.other_label.as_str()];
        for cat in runtime_data.iter() {
            let i = *self
                .baseline
                .idx_map
                .get(cat.as_ref())
                .unwrap_or_else(|| &other_idx);

            self.rt_bins[i] += 1_f64;
        }
        Ok(())
    }

    fn clear_rt(&mut self) {
        self.rt_bins.fill(0_f64);
    }

    // Reset the baseline state with a new baseline dataset. This will adjust the number of bins to
    // n + 1 where n is the number of observed unique examples.
    pub fn reset_baseline<S: StringLike>(&mut self, new_baseline: &[S]) {
        self.baseline.reset(new_baseline);
        let num_bins = self.baseline.baseline_bins.len();

        // pay the cost to reallocate bins in order to have correct size
        // not common path
        self.rt_bins = vec![0_f64; num_bins];
    }

    pub fn other_bucket_label(&self) -> String {
        self.baseline.other_label.clone()
    }

    pub fn export_baseline(&self) -> HashMap<String, f64> {
        self.baseline.export_baseline()
    }
}

/// Streaming implementation of '[CategoricalDataDrift]' type. This is intended for long running
/// services to give an indication of the data drift over a longer contiguous window. A stateful
/// stream, where point in time snapshots can be generated.
pub struct StreamingCategoricalDataDrift {
    baseline: BaselineCategoricalBins,
    stream_bins: Vec<f64>,
    total_stream_size: usize,
    last_flush_ts: i64,
    flush_rate: i64,
}

// Implementation of different drift methods
impl StreamingPopulationStabilityIndexDrift for StreamingCategoricalDataDrift {
    fn psi_drift(&self) -> Result<f64, DriftError> {
        if self.total_stream_size == 0 {
            return Err(DriftError::EmptyRuntimeData);
        }

        Ok(compute_psi(
            &self.baseline.baseline_bins,
            &self.stream_bins,
            self.total_stream_size as f64,
        ))
    }
}

impl StreamingKlDivergenceDrift for StreamingCategoricalDataDrift {
    fn kl_divergence_drift(&self) -> Result<f64, DriftError> {
        if self.total_stream_size == 0 {
            return Err(DriftError::EmptyRuntimeData);
        }

        Ok(compute_kl_divergence_drift(
            &self.baseline.baseline_bins,
            &self.stream_bins,
            self.total_stream_size as f64,
        ))
    }
}

impl StreamingJensenShannonDivergenceDrift for StreamingCategoricalDataDrift {
    fn js_drift(&self) -> Result<f64, DriftError> {
        if self.total_stream_size == 0 {
            return Err(DriftError::EmptyRuntimeData);
        }
        Ok(compute_jensen_shannon_divergence_drift(
            &self.baseline.baseline_bins,
            &self.stream_bins,
            self.total_stream_size as f64,
        ))
    }
}

impl StreamingCategoricalDataDrift {
    pub fn new<S: StringLike>(
        baseline_data: &[S],
        user_flush_rate: Option<i64>,
    ) -> Result<StreamingCategoricalDataDrift, DriftError> {
        let baseline = BaselineCategoricalBins::new(baseline_data);
        let flush_rate = user_flush_rate.unwrap_or_else(|| DEFAULT_STREAM_FLUSH);
        let n_bins = baseline.baseline_bins.len();
        let stream_bins: Vec<f64> = vec![0_f64; n_bins];

        let last_flush_ts = Utc::now().timestamp();
        let mut obj = StreamingCategoricalDataDrift {
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

    #[inline]
    fn push_item<S: StringLike>(&mut self, runtime_example: &S, other_idx: usize) {
        let idx = *self
            .baseline
            .idx_map
            .get(runtime_example.as_ref())
            .unwrap_or_else(|| &other_idx);
        self.stream_bins[idx] += 1_f64;
        self.total_stream_size += 1;
    }

    pub fn update_stream<S: StringLike>(&mut self, item: &S) {
        self.push_item(
            item,
            self.baseline.idx_map[self.baseline.other_label.as_str()],
        )
    }

    pub fn update_stream_batch<S: StringLike>(
        &mut self,
        runtime_data: &[S],
    ) -> Result<(), DriftError> {
        if runtime_data.is_empty() {
            return Err(DriftError::EmptyRuntimeData);
        }
        let curr_ts: i64 = Utc::now().timestamp();

        if curr_ts > (self.last_flush_ts + self.flush_rate)
            || (self.total_stream_size + runtime_data.len()) > *MAX_STREAM_SIZE
        {
            // reset and flush
            self.flush_runtime_stream();
            self.last_flush_ts = curr_ts;
        }

        let other_idx = self.baseline.idx_map[self.baseline.other_label.as_str()];
        for cat in runtime_data.iter() {
            self.push_item(cat, other_idx)
        }

        Ok(())
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
}

#[cfg(test)]
mod continuous_tests {
    use super::*;

    #[test]
    fn test_continuous_baseline_builds_expected_bins() {
        let baseline = [1.0, 2.0, 3.0, 4.0];
        let psi = ContinuousDataDrift::new_from_baseline(3, &baseline).unwrap();

        // 3 bins → 4 edges
        assert_eq!(psi.baseline.bin_edges.len(), 4);
        assert_eq!(psi.rt_bins.len(), 3);
    }

    #[test]
    fn test_continuous_psi_zero_when_no_drift() {
        let baseline = [1.0, 2.0, 3.0, 4.0];
        let mut psi = ContinuousDataDrift::new_from_baseline(3, &baseline).unwrap();
        let runtime = [1.0, 2.0, 3.0, 4.0];

        let drift = psi.psi_drift(&runtime).unwrap();
        assert!(drift.abs() < 1e-9);
    }

    #[test]
    fn test_continuous_psi_detects_shift() {
        let baseline = [1.0, 2.0, 3.0, 4.0];
        let mut psi = ContinuousDataDrift::new_from_baseline(3, &baseline).unwrap();
        let runtime = [10.0, 11.0, 12.0, 13.0];

        let drift = psi.psi_drift(&runtime).unwrap();
        assert!(drift > 0.5);
    }

    #[test]
    fn test_streaming_continuous_accumulation() {
        let baseline = [1_f64, 2_f64, 3_f64, 3_f64, 4_f64];
        let mut streaming = StreamingContinuousDataDrift::new(3, &baseline, None).unwrap();

        streaming
            .update_stream_batch(&[1.0, 2.0, 2.0, 3.0, 4.0])
            .unwrap();

        let d1 = streaming.psi_drift().unwrap();
        streaming
            .update_stream_batch(&[3.0, 4.0, 2.0, 2.0, 1.0, 3.0])
            .unwrap();

        let d2 = streaming.psi_drift().unwrap();

        assert!(d1.abs() < 1e-9);
        assert!(d2.abs() < 1e-2);
        assert_eq!(streaming.total_samples(), 11);
    }

    #[test]
    fn test_streaming_flush() {
        let baseline = [1.0, 2.0, 3.0, 4.0];
        let mut streaming = StreamingContinuousDataDrift::new(3, &baseline, None).unwrap();

        streaming.update_stream_batch(&[1.0, 2.0, 3.0]).unwrap();
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
        let psi = CategoricalDataDrift::new(&baseline).unwrap();

        // baseline has 3 real labels + OTHER bucket
        assert_eq!(psi.baseline.baseline_bins.len(), 4);
    }

    #[test]
    fn test_categorical_psi_zero_when_no_drift() {
        let baseline = ["a", "b", "a", "c"];
        let mut psi = CategoricalDataDrift::new(&baseline).unwrap();
        let runtime = ["a", "b", "a", "c"];

        let drift = psi.psi_drift(&runtime).unwrap();
        assert!(drift.abs() < 1e-9);
    }

    #[test]
    fn test_categorical_psi_detects_shift() {
        let baseline = ["a", "b", "a", "c"];
        let mut psi = CategoricalDataDrift::new(&baseline).unwrap();
        let runtime = ["x", "x", "x", "x"]; // go to other bucket

        let drift = psi.psi_drift(&runtime).unwrap();
        assert!(drift > 0.5);
    }

    #[test]
    fn test_other_bucket_label_exposed() {
        let baseline = ["a", "b"];
        let psi = CategoricalDataDrift::new(&baseline).unwrap();
        let other = psi.other_bucket_label();

        assert!(other.starts_with("__fairperf_othercat__"));
    }

    #[test]
    fn test_streaming_categorical_accumulation() {
        let baseline = ["a", "b"];
        let mut streaming = StreamingCategoricalDataDrift::new(&baseline, None).unwrap();

        streaming.update_stream_batch(&["a", "b"]).unwrap();
        let d1 = streaming.psi_drift().unwrap();
        let mut stream = Vec::new();

        for _ in 0..500 {
            stream.push("a")
        }

        for _ in 0..490 {
            stream.push("b")
        }
        streaming.update_stream_batch(&stream).unwrap();
        let d2 = streaming.psi_drift().unwrap();

        assert_eq!(streaming.total_samples(), 992);
        assert!(d1 < 1e-9);
        assert!(d2 < 1e-2);
    }
}
