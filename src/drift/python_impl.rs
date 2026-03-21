#[cfg(feature = "python")]
pub(crate) mod py_api {

    use ahash::HashMap;
    use numpy::PyReadonlyArray1;
    use pyo3::{prelude::*, pyclass, pymethods};
    use std::num::NonZeroU64;
    use std::time::Duration;

    use crate::{
        drift::{
            data_drift::{
                CategoricalDataDrift, ContinuousDataDrift, DecayModeMark, FlushModeMark,
                StreamingCategoricalDataDrift, StreamingContinuousDataDrift,
            },
            distribution::QuantileType,
            drift_metrics::DataDriftType,
        },
        errors::DriftError,
    };

    /// Batch continuous data drift detector. Computes drift metrics between a fixed
    /// baseline histogram and a runtime data slice on each call.
    #[pyclass]
    pub(crate) struct PyContinuousDataDrift {
        inner: ContinuousDataDrift,
    }

    #[pymethods]
    impl PyContinuousDataDrift {
        /// Construct from a baseline data slice.
        ///
        /// `quantile_type_str` controls how many histogram bins are derived from the baseline.
        /// The bin count determines the resolution of the drift signal — more bins capture finer
        /// distributional shifts but require more runtime data per bin to be statistically
        /// meaningful. Options:
        ///
        /// - `"FreedmanDiaconis"` *(default)*: IQR-based bin width. Robust to outliers. Preferred
        ///   for most use cases.
        /// - `"Scott"`: standard-deviation-based bin width. Assumes approximately normal data;
        ///   sensitive to outliers in the tails.
        /// - `"Sturges"`: simple `floor(log2(n)) + 1` rule. Works best for small, roughly normal
        ///   datasets.
        #[new]
        pub fn new<'py>(
            baseline_data: PyReadonlyArray1<'py, f64>,
            quantile_type_str: Option<String>,
        ) -> PyResult<PyContinuousDataDrift> {
            let quantile_type = if let Some(q_type) = quantile_type_str {
                Some(QuantileType::try_from(q_type.as_str())?)
            } else {
                None
            };
            let bl_slice = baseline_data.as_slice()?;
            let inner = ContinuousDataDrift::new_from_baseline(quantile_type, bl_slice)?;
            Ok(Self { inner })
        }

        /// Compute a single drift metric against `runtime_data`.
        ///
        /// `drift_type` must be one of: `"JensenShannon"`, `"PopulationStabilityIndex"`,
        /// `"WassersteinDistance"`, `"KullbackLeibler"`.
        fn compute_drift<'py>(
            &mut self,
            runtime_data: PyReadonlyArray1<'py, f64>,
            drift_type: String,
        ) -> PyResult<f64> {
            let rt_slice = runtime_data.as_slice()?;
            let drift_t = DataDriftType::try_from(drift_type.as_str())?;
            Ok(self.inner.compute_drift(rt_slice, drift_t)?)
        }

        /// Replace the baseline with a new data slice, recomputing the histogram.
        fn reset_baseline<'py>(
            &mut self,
            baseline_data: PyReadonlyArray1<'py, f64>,
        ) -> PyResult<()> {
            let bl_slice = baseline_data.as_slice()?;
            self.inner.reset_baseline(bl_slice)?;
            Ok(())
        }

        /// Return the baseline as a normalized probability distribution (sums to 1.0).
        fn export_baseline(&self) -> Vec<f64> {
            self.inner.export_baseline()
        }

        /// Number of histogram bins used for this baseline.
        #[getter]
        fn num_bins(&self) -> usize {
            self.inner.n_bins()
        }
    }

    /// Streaming continuous data drift detector in flush mode. Accumulates runtime
    /// samples and resets accumulated bins on each flush.
    #[pyclass]
    pub(crate) struct PyStreamingContinuousDataDriftFlush {
        inner: StreamingContinuousDataDrift<FlushModeMark>,
    }

    #[pymethods]
    impl PyStreamingContinuousDataDriftFlush {
        /// Construct from a baseline data slice.
        ///
        /// `quantile_type_str` controls histogram bin count. See [`PyContinuousDataDrift::new`]
        /// for a full description of each option. Defaults to `"FreedmanDiaconis"`.
        ///
        /// `flush_rate_opt`: number of accumulated samples that triggers an automatic flush,
        /// resetting the stream window. A lower value means more frequent resets and a more
        /// responsive signal, but each window will contain fewer samples, making the drift
        /// estimate noisier. Defaults to 1,000,000.
        ///
        /// `flush_cadence_opt`: time in seconds after which an automatic flush is triggered,
        /// regardless of sample count. Combined with `flush_rate_opt`, whichever threshold is
        /// reached first triggers the flush. Defaults to 86,400 (24 hours).
        #[new]
        #[pyo3(signature = (baseline_data, quantile_type_str, flush_rate_opt, flush_cadence_opt))]
        fn new<'py>(
            baseline_data: PyReadonlyArray1<'py, f64>,
            quantile_type_str: Option<String>,
            flush_rate_opt: Option<u64>,
            flush_cadence_opt: Option<u64>,
        ) -> PyResult<PyStreamingContinuousDataDriftFlush> {
            let baseline_slice = baseline_data.as_slice()?;
            let qtype_opt: Option<QuantileType> = if let Some(ref qtype_str) = quantile_type_str {
                Some(QuantileType::try_from(qtype_str.as_str())?)
            } else {
                None
            };
            let cadence: Option<Duration> = if let Some(flush_cadence) = flush_cadence_opt {
                Some(Duration::new(flush_cadence, 0))
            } else {
                None
            };
            let inner = StreamingContinuousDataDrift::new_flush(
                baseline_slice,
                qtype_opt,
                flush_rate_opt,
                cadence,
            )?;
            Ok(Self { inner })
        }

        /// Compute a single drift metric against the current stream bins. Triggers a
        /// flush if the configured flush cadence or rate has been reached.
        ///
        /// `drift_type` must be one of: `"JensenShannon"`, `"PopulationStabilityIndex"`,
        /// `"WassersteinDistance"`, `"KullbackLeibler"`.
        #[pyo3(signature = (drift_type))]
        fn compute_drift(&mut self, drift_type: String) -> PyResult<f64> {
            let drift_t = DataDriftType::try_from(drift_type.as_str())?;
            Ok(self.inner.compute_drift(drift_t)?)
        }

        /// Compute multiple drift metrics in a single pass. Returns scores in the same
        /// order as `drift_types`.
        #[pyo3(signature = (drift_types))]
        fn compute_drift_multiple_criteria(
            &mut self,
            drift_types: Vec<String>,
        ) -> PyResult<Vec<f64>> {
            let mut drift_t = Vec::with_capacity(drift_types.len());
            for dt in drift_types.iter() {
                drift_t.push(DataDriftType::try_from(dt.as_str())?);
            }

            Ok(self.inner.compute_drift_multiple_criteria(&drift_t)?)
        }

        /// Replace the baseline distribution with a new data slice and clear the stream.
        #[pyo3(signature = (baseline_data))]
        fn reset_baseline<'py>(
            &mut self,
            baseline_data: PyReadonlyArray1<'py, f64>,
        ) -> PyResult<()> {
            let baseline_slice = baseline_data.as_slice()?;
            self.inner.reset_baseline(baseline_slice)?;
            Ok(())
        }

        /// Add a batch of runtime samples to the stream buffer.
        #[pyo3(signature = (runtime_data))]
        fn update_stream_batch<'py>(
            &mut self,
            runtime_data: PyReadonlyArray1<'py, f64>,
        ) -> PyResult<()> {
            let runtime_data_slice = runtime_data.as_slice()?;
            self.inner.update_stream_batch(runtime_data_slice)?;
            Ok(())
        }

        /// Add a single runtime sample to the stream buffer.
        #[pyo3(signature = (runtime_example))]
        fn update_stream(&mut self, runtime_example: f64) {
            self.inner.update_stream(runtime_example);
        }

        /// Flush the stream, resetting accumulated runtime bins.
        fn flush(&mut self) {
            self.inner.flush()
        }

        /// Total number of runtime samples accumulated since the last flush.
        #[getter]
        fn total_samples(&self) -> usize {
            self.inner.total_samples()
        }

        /// Unix timestamp (seconds) of the last flush.
        #[getter]
        fn last_flush(&self) -> u64 {
            self.inner.last_flush()
        }

        /// Number of histogram bins.
        #[getter]
        fn n_bins(&self) -> usize {
            self.inner.n_bins()
        }

        /// Export the current stream histogram as a map of bin label to bin values.
        fn export_snapshot(&self) -> PyResult<HashMap<String, Vec<f64>>> {
            Ok(self.inner.export_snapshot()?)
        }

        /// Return the baseline as a normalized probability distribution (sums to 1.0).
        fn export_baseline(&self) -> Vec<f64> {
            self.inner.export_baseline()
        }
    }

    /// Streaming continuous data drift detector in decay mode. Applies exponential
    /// decay to accumulated runtime bins on each drift computation, giving more weight
    /// to recent samples.
    #[pyclass]
    pub(crate) struct PyStreamingContinuousDataDriftDecay {
        inner: StreamingContinuousDataDrift<DecayModeMark>,
    }

    #[pymethods]
    impl PyStreamingContinuousDataDriftDecay {
        /// Construct from a baseline data slice.
        ///
        /// `quantile_type_str` controls histogram bin count. See [`PyContinuousDataDrift::new`]
        /// for a full description of each option. Defaults to `"FreedmanDiaconis"`.
        ///
        /// `decay_opt`: half-life in seconds for the exponential decay weight α = 0.5^(1/half_life),
        /// applied to all bin counts on each call to `compute_drift` or
        /// `compute_drift_multiple_criteria`. A shorter half-life causes older data to be
        /// down-weighted faster, making the signal more sensitive to recent distribution shifts
        /// at the cost of higher variance. A longer half-life produces a smoother, more stable
        /// signal that responds slowly to new patterns. Defaults to 86,400 (24 hours), meaning a
        /// sample's contribution is halved after 24 hours worth of `compute_drift` calls.
        #[new]
        #[pyo3(signature = (baseline_data, quantile_type_str, decay_opt))]
        fn new<'py>(
            baseline_data: PyReadonlyArray1<'py, f64>,
            quantile_type_str: Option<String>,
            decay_opt: Option<u64>,
        ) -> PyResult<PyStreamingContinuousDataDriftDecay> {
            let baseline_slice = baseline_data.as_slice()?;
            let qtype_opt: Option<QuantileType> = if let Some(ref qtype_str) = quantile_type_str {
                Some(QuantileType::try_from(qtype_str.as_str())?)
            } else {
                None
            };
            let decay: Option<NonZeroU64> = if let Some(decay_raw) = decay_opt {
                Some(NonZeroU64::new(decay_raw).ok_or(DriftError::UnsupportedConfig)?)
            } else {
                None
            };
            let inner = StreamingContinuousDataDrift::new_decay(baseline_slice, qtype_opt, decay)?;
            Ok(Self { inner })
        }

        /// Apply decay to stream bins and compute a single drift metric.
        ///
        /// `drift_type` must be one of: `"JensenShannon"`, `"PopulationStabilityIndex"`,
        /// `"WassersteinDistance"`, `"KullbackLeibler"`.
        #[pyo3(signature = (drift_type))]
        fn compute_drift(&mut self, drift_type: String) -> PyResult<f64> {
            let drift_t = DataDriftType::try_from(drift_type.as_str())?;
            Ok(self.inner.compute_drift(drift_t)?)
        }

        /// Apply decay and compute multiple drift metrics in a single pass. Returns
        /// scores in the same order as `drift_types`.
        #[pyo3(signature = (drift_types))]
        fn compute_drift_multiple_criteria(
            &mut self,
            drift_types: Vec<String>,
        ) -> PyResult<Vec<f64>> {
            let mut drift_t = Vec::with_capacity(drift_types.len());
            for dt in drift_types.iter() {
                drift_t.push(DataDriftType::try_from(dt.as_str())?);
            }

            Ok(self.inner.compute_drift_multiple_criteria(&drift_t)?)
        }

        /// Replace the baseline distribution with a new data slice and clear the stream.
        #[pyo3(signature = (baseline_data))]
        fn reset_baseline<'py>(
            &mut self,
            baseline_data: PyReadonlyArray1<'py, f64>,
        ) -> PyResult<()> {
            let baseline_slice = baseline_data.as_slice()?;
            self.inner.reset_baseline(baseline_slice)?;
            Ok(())
        }

        /// Add a batch of runtime samples to the stream buffer.
        #[pyo3(signature = (runtime_data))]
        fn update_stream_batch<'py>(
            &mut self,
            runtime_data: PyReadonlyArray1<'py, f64>,
        ) -> PyResult<()> {
            let runtime_data_slice = runtime_data.as_slice()?;
            self.inner.update_stream_batch(runtime_data_slice)?;
            Ok(())
        }

        /// Add a single runtime sample to the stream buffer.
        #[pyo3(signature = (runtime_example))]
        fn update_stream(&mut self, runtime_example: f64) {
            self.inner.update_stream(runtime_example);
        }

        /// Effective sample count after decay weighting.
        #[getter]
        fn total_samples(&self) -> usize {
            self.inner.total_samples()
        }

        /// Number of histogram bins.
        #[getter]
        fn n_bins(&self) -> usize {
            self.inner.n_bins()
        }

        /// Export the current stream histogram as a map of bin label to bin values.
        fn export_snapshot(&self) -> PyResult<HashMap<String, Vec<f64>>> {
            Ok(self.inner.export_snapshot()?)
        }

        /// Return the baseline as a normalized probability distribution (sums to 1.0).
        fn export_baseline(&self) -> Vec<f64> {
            self.inner.export_baseline()
        }
    }

    /// Batch categorical data drift detector. Computes drift metrics between a fixed
    /// baseline label distribution and a runtime label slice on each call.
    #[pyclass]
    pub(crate) struct PyCategoricalDataDrift {
        inner: CategoricalDataDrift<String>,
    }

    #[pymethods]
    impl PyCategoricalDataDrift {
        /// Construct from a baseline dataset of string labels.
        #[new]
        #[pyo3(signature = (baseline_data))]
        fn new(baseline_data: Vec<String>) -> PyResult<PyCategoricalDataDrift> {
            let inner = match CategoricalDataDrift::new(&baseline_data) {
                Ok(psi) => psi,
                Err(e) => return Err(e.into()),
            };
            Ok(Self { inner })
        }

        /// Compute a single drift metric against `runtime_data`.
        ///
        /// `drift_type` must be one of: `"JensenShannon"`, `"PopulationStabilityIndex"`,
        /// `"WassersteinDistance"`, `"KullbackLeibler"`.
        fn compute_drift(
            &mut self,
            runtime_data: Vec<String>,
            drift_type: String,
        ) -> PyResult<f64> {
            let drift_t = DataDriftType::try_from(drift_type.as_str())?;
            Ok(self.inner.compute_drift(&runtime_data, drift_t)?)
        }

        /// Replace the baseline with a new set of labels, recomputing the distribution.
        #[pyo3(signature = (new_baseline))]
        fn reset_baseline(&mut self, new_baseline: Vec<String>) -> PyResult<()> {
            self.inner.reset_baseline(&new_baseline)?;
            Ok(())
        }

        /// Return the baseline label frequencies as a normalized distribution.
        fn export_baseline(&self) -> HashMap<String, f64> {
            self.inner.export_baseline()
        }
    }

    /// Streaming categorical data drift detector in flush mode. Accumulates runtime
    /// label counts and resets accumulated bins on each flush.
    #[pyclass]
    pub(crate) struct PyStreamingCategoricalDataDriftFlush {
        inner: StreamingCategoricalDataDrift<String, FlushModeMark>,
    }

    #[pymethods]
    impl PyStreamingCategoricalDataDriftFlush {
        /// Construct from a baseline dataset of string labels.
        ///
        /// `flush_rate`: number of accumulated samples that triggers an automatic flush, resetting
        /// the stream window. A lower value means more frequent resets and a more responsive
        /// signal, but each window will contain fewer samples, making the drift estimate noisier.
        /// Defaults to 1,000,000.
        ///
        /// `flush_cadence`: time in seconds after which an automatic flush is triggered, regardless
        /// of sample count. Combined with `flush_rate`, whichever threshold is reached first
        /// triggers the flush. Defaults to 86,400 (24 hours).
        #[new]
        fn new(
            baseline_data: Vec<String>,
            flush_rate: Option<u64>,
            flush_cadence: Option<u64>,
        ) -> PyResult<PyStreamingCategoricalDataDriftFlush> {
            let cadence: Option<Duration> = if let Some(cadence_raw) = flush_cadence {
                Some(Duration::new(cadence_raw, 0))
            } else {
                None
            };
            let inner =
                StreamingCategoricalDataDrift::new_flush(&baseline_data, flush_rate, cadence)?;

            Ok(Self { inner })
        }

        /// Compute a single drift metric against the current stream bins. Triggers a
        /// flush if the configured flush cadence or rate has been reached.
        ///
        /// `drift_type` must be one of: `"JensenShannon"`, `"PopulationStabilityIndex"`,
        /// `"WassersteinDistance"`, `"KullbackLeibler"`.
        #[pyo3(signature = (drift_type))]
        fn compute_drift(&mut self, drift_type: String) -> PyResult<f64> {
            let drift_t = DataDriftType::try_from(drift_type.as_str())?;
            Ok(self.inner.compute_drift(drift_t)?)
        }

        /// Compute multiple drift metrics in a single pass. Returns scores in the same
        /// order as `drift_types`.
        #[pyo3(signature = (drift_types))]
        fn compute_drift_multiple_criteria(
            &mut self,
            drift_types: Vec<String>,
        ) -> PyResult<Vec<f64>> {
            let mut drift_t = Vec::with_capacity(drift_types.len());
            for dt in drift_types.iter() {
                drift_t.push(DataDriftType::try_from(dt.as_str())?);
            }

            Ok(self.inner.compute_drift_multiple_criteria(&drift_t)?)
        }

        /// Replace the baseline distribution with a new set of labels and clear the stream.
        #[pyo3(signature = (new_baseline))]
        fn reset_baseline(&mut self, new_baseline: Vec<String>) -> PyResult<()> {
            self.inner.reset_baseline(&new_baseline)?;
            Ok(())
        }

        /// Add a batch of runtime labels to the stream buffer.
        #[pyo3(signature = (runtime_data))]
        fn update_stream_batch(&mut self, runtime_data: Vec<String>) -> PyResult<()> {
            self.inner.update_stream_batch(&runtime_data)?;
            Ok(())
        }

        /// Add a single runtime label to the stream buffer.
        #[pyo3(signature = (runtime_example))]
        fn update_stream(&mut self, runtime_example: String) {
            self.inner.update_stream(&runtime_example);
        }

        /// Flush the stream, resetting accumulated runtime label counts.
        fn flush(&mut self) {
            self.inner.flush()
        }

        /// Total number of runtime samples accumulated since the last flush.
        #[getter]
        fn total_samples(&self) -> usize {
            self.inner.total_samples()
        }

        /// Unix timestamp (seconds) of the last flush.
        #[getter]
        fn last_flush(&self) -> u64 {
            self.inner.last_flush()
        }

        /// Number of distinct label bins.
        #[getter]
        fn n_bins(&self) -> usize {
            self.inner.num_bins()
        }

        /// Export the current stream label counts as a map of label to count.
        fn export_snapshot(&self) -> HashMap<String, f64> {
            self.inner.export_snapshot()
        }

        /// Return the baseline label frequencies as a normalized distribution.
        fn export_baseline(&self) -> HashMap<String, f64> {
            self.inner.export_baseline()
        }
    }

    /// Streaming categorical data drift detector in decay mode. Applies exponential
    /// decay to accumulated label counts on each drift computation, giving more weight
    /// to recent samples.
    #[pyclass]
    pub(crate) struct PyStreamingCategoricalDataDriftDecay {
        inner: StreamingCategoricalDataDrift<String, DecayModeMark>,
    }

    #[pymethods]
    impl PyStreamingCategoricalDataDriftDecay {
        /// Construct from a baseline dataset of string labels.
        ///
        /// `half_life_opt`: half-life in seconds for the exponential decay weight
        /// α = 0.5^(1/half_life), applied to all label counts on each call to `compute_drift` or
        /// `compute_drift_multiple_criteria`. A shorter half-life causes older data to be
        /// down-weighted faster, making the signal more sensitive to recent distribution shifts
        /// at the cost of higher variance. A longer half-life produces a smoother, more stable
        /// signal that responds slowly to new patterns. Defaults to 86,400 (24 hours), meaning a
        /// sample's contribution is halved after 24 hours worth of `compute_drift` calls.
        #[new]
        fn new(
            baseline_data: Vec<String>,
            half_life_opt: Option<u64>,
        ) -> PyResult<PyStreamingCategoricalDataDriftDecay> {
            let half_life: Option<NonZeroU64> = if let Some(half_life_raw) = half_life_opt {
                Some(NonZeroU64::new(half_life_raw).ok_or(DriftError::UnsupportedConfig)?)
            } else {
                None
            };
            let inner = StreamingCategoricalDataDrift::new_decay(&baseline_data, half_life)?;

            Ok(Self { inner })
        }

        /// Apply decay to stream bins and compute a single drift metric.
        ///
        /// `drift_type` must be one of: `"JensenShannon"`, `"PopulationStabilityIndex"`,
        /// `"WassersteinDistance"`, `"KullbackLeibler"`.
        #[pyo3(signature = (drift_type))]
        fn compute_drift(&mut self, drift_type: String) -> PyResult<f64> {
            let drift_t = DataDriftType::try_from(drift_type.as_str())?;
            Ok(self.inner.compute_drift(drift_t)?)
        }

        /// Apply decay and compute multiple drift metrics in a single pass. Returns
        /// scores in the same order as `drift_types`.
        #[pyo3(signature = (drift_types))]
        fn compute_drift_multiple_criteria(
            &mut self,
            drift_types: Vec<String>,
        ) -> PyResult<Vec<f64>> {
            let mut drift_t = Vec::with_capacity(drift_types.len());
            for dt in drift_types.iter() {
                drift_t.push(DataDriftType::try_from(dt.as_str())?);
            }

            Ok(self.inner.compute_drift_multiple_criteria(&drift_t)?)
        }

        /// Replace the baseline distribution with a new set of labels and clear the stream.
        #[pyo3(signature = (new_baseline))]
        fn reset_baseline(&mut self, new_baseline: Vec<String>) -> PyResult<()> {
            self.inner.reset_baseline(&new_baseline)?;
            Ok(())
        }

        /// Add a batch of runtime labels to the stream buffer.
        #[pyo3(signature = (runtime_data))]
        fn update_stream_batch(&mut self, runtime_data: Vec<String>) -> PyResult<()> {
            self.inner.update_stream_batch(&runtime_data)?;
            Ok(())
        }

        /// Add a single runtime label to the stream buffer.
        #[pyo3(signature = (runtime_example))]
        fn update_stream(&mut self, runtime_example: String) {
            self.inner.update_stream(&runtime_example);
        }

        /// Effective sample count after decay weighting.
        #[getter]
        fn total_samples(&self) -> usize {
            self.inner.total_samples()
        }

        /// Number of distinct label bins.
        #[getter]
        fn n_bins(&self) -> usize {
            self.inner.num_bins()
        }

        /// Export the current stream label counts as a map of label to count.
        fn export_snapshot(&self) -> HashMap<String, f64> {
            self.inner.export_snapshot()
        }

        /// Return the baseline label frequencies as a normalized distribution.
        fn export_baseline(&self) -> HashMap<String, f64> {
            self.inner.export_baseline()
        }
    }
}
