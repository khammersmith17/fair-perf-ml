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

    #[pyclass]
    pub(crate) struct PyContinuousDataDrift {
        inner: ContinuousDataDrift,
    }

    #[pymethods]
    impl PyContinuousDataDrift {
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

        fn compute_drift<'py>(
            &mut self,
            runtime_data: PyReadonlyArray1<'py, f64>,
            drift_type: String,
        ) -> PyResult<f64> {
            let rt_slice = runtime_data.as_slice()?;
            let drift_t = DataDriftType::try_from(drift_type.as_str())?;
            Ok(self.inner.compute_drift(rt_slice, drift_t)?)
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

        fn export_baseline(&self) -> Vec<f64> {
            self.inner.export_baseline()
        }

        #[getter]
        fn num_bins(&self) -> usize {
            self.inner.n_bins()
        }
    }

    enum StreamingInitMode {
        Flush,
        Decay,
    }

    impl StreamingInitMode {
        /// Based on the passed arguments, derive which mode is being configured.
        fn get_mode(
            decay_is_some: bool,
            flush_rate_is_some: bool,
            flush_cadence_is_some: bool,
        ) -> Result<Self, DriftError> {
            match (decay_is_some, flush_rate_is_some, flush_cadence_is_some) {
                // default to flush mode
                (false, false, false) => Ok(Self::Flush),
                (false, true, true) => Ok(Self::Flush),
                (true, false, false) => Ok(Self::Decay),
                _ => Err(DriftError::UnsupportedConfig),
            }
        }
    }

    enum PyContinuousStreamingDriftInner {
        FlushMode(StreamingContinuousDataDrift<FlushModeMark>),
        DecayMode(StreamingContinuousDataDrift<DecayModeMark>),
    }

    impl PyContinuousStreamingDriftInner {
        fn serialize_from_args(
            baseline_data: &[f64],
            quantile_type_str: Option<String>,
            decay_opt: Option<u64>,
            flush_rate_opt: Option<u64>,
            flush_cadence_opt: Option<u64>,
        ) -> PyResult<Self> {
            let mode = StreamingInitMode::get_mode(
                decay_opt.is_some(),
                flush_rate_opt.is_some(),
                flush_cadence_opt.is_some(),
            )?;
            let quantile_type = if let Some(q_type) = quantile_type_str {
                QuantileType::try_from(q_type.as_str())?
            } else {
                QuantileType::default()
            };

            match mode {
                StreamingInitMode::Flush => {
                    let inner = match (flush_rate_opt, flush_cadence_opt) {
                        (Some(flush_rate), Some(cadence)) => {
                            StreamingContinuousDataDrift::new_flush(
                                baseline_data,
                                quantile_type,
                                flush_rate,
                                Duration::from_secs(cadence),
                            )?
                        }
                        _ => StreamingContinuousDataDrift::default_flush(baseline_data)?,
                    };
                    Ok(Self::FlushMode(inner))
                }
                StreamingInitMode::Decay => {
                    let decay_raw = decay_opt.ok_or(DriftError::UnsupportedConfig)?;
                    let decay = NonZeroU64::new(decay_raw).ok_or(DriftError::UnsupportedConfig)?;
                    let inner = StreamingContinuousDataDrift::new_decay(
                        baseline_data,
                        quantile_type,
                        decay,
                    )?;
                    Ok(Self::DecayMode(inner))
                }
            }
        }

        fn compute_drift(&mut self, drift_type: DataDriftType) -> Result<f64, DriftError> {
            match self {
                Self::FlushMode(inner) => inner.compute_drift(drift_type),
                Self::DecayMode(inner) => inner.compute_drift(drift_type),
            }
        }

        fn compute_drift_multiple_criteria(
            &mut self,
            drift_types: &[DataDriftType],
        ) -> Result<Vec<f64>, DriftError> {
            match self {
                Self::FlushMode(inner) => inner.compute_drift_multiple_criteria(drift_types),
                Self::DecayMode(inner) => inner.compute_drift_multiple_criteria(drift_types),
            }
        }

        fn reset_baseline(&mut self, baseline_slice: &[f64]) -> Result<(), DriftError> {
            match self {
                Self::FlushMode(inner) => inner.reset_baseline(baseline_slice),
                Self::DecayMode(inner) => inner.reset_baseline(baseline_slice),
            }
        }

        fn update_stream(&mut self, runtime_example: f64) {
            match self {
                Self::FlushMode(inner) => inner.update_stream(runtime_example),
                Self::DecayMode(inner) => inner.update_stream(runtime_example),
            }
        }

        fn update_stream_batch(&mut self, runtime_slice: &[f64]) -> Result<(), DriftError> {
            match self {
                Self::FlushMode(inner) => inner.update_stream_batch(runtime_slice),
                Self::DecayMode(inner) => inner.update_stream_batch(runtime_slice),
            }
        }

        fn flush(&mut self) -> Result<(), DriftError> {
            match self {
                Self::FlushMode(inner) => {
                    inner.flush();
                    Ok(())
                }
                Self::DecayMode(_) => Err(DriftError::UnsupportedOperation),
            }
        }

        fn total_samples(&self) -> usize {
            match self {
                Self::FlushMode(inner) => inner.total_samples(),
                Self::DecayMode(inner) => inner.total_samples(),
            }
        }

        fn last_flush(&self) -> Result<u64, DriftError> {
            match self {
                Self::FlushMode(inner) => Ok(inner.last_flush()),
                Self::DecayMode(_) => Err(DriftError::UnsupportedOperation),
            }
        }

        fn n_bins(&self) -> usize {
            match self {
                Self::FlushMode(inner) => inner.n_bins(),
                Self::DecayMode(inner) => inner.n_bins(),
            }
        }

        fn export_snapshot(&self) -> Result<ahash::HashMap<String, Vec<f64>>, DriftError> {
            match self {
                Self::FlushMode(inner) => inner.export_snapshot(),
                Self::DecayMode(inner) => inner.export_snapshot(),
            }
        }

        fn export_baseline(&self) -> Vec<f64> {
            match self {
                Self::FlushMode(inner) => inner.export_baseline(),
                Self::DecayMode(inner) => inner.export_baseline(),
            }
        }
    }
    /// Exposed Python APIs for streaming continuous PSI
    #[pyclass]
    pub(crate) struct PyStreamingContinuousDataDrift {
        inner: PyContinuousStreamingDriftInner,
    }

    #[pymethods]
    impl PyStreamingContinuousDataDrift {
        #[new]
        fn new<'py>(
            baseline_data: PyReadonlyArray1<'py, f64>,
            quantile_type_str: Option<String>,
            decay_opt: Option<u64>,
            flush_rate_opt: Option<u64>,
            flush_cadence_opt: Option<u64>,
        ) -> PyResult<PyStreamingContinuousDataDrift> {
            let baseline_slice = baseline_data.as_slice()?;
            let inner = PyContinuousStreamingDriftInner::serialize_from_args(
                baseline_slice,
                quantile_type_str,
                decay_opt,
                flush_rate_opt,
                flush_cadence_opt,
            )?;

            Ok(Self { inner })
        }

        fn compute_drift(&mut self, drift_type: String) -> PyResult<f64> {
            let drift_t = DataDriftType::try_from(drift_type.as_str())?;
            Ok(self.inner.compute_drift(drift_t)?)
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

        fn flush(&mut self) -> PyResult<()> {
            Ok(self.inner.flush()?)
        }

        #[getter]
        fn total_samples(&self) -> usize {
            self.inner.total_samples()
        }

        #[getter]
        fn last_flush(&self) -> PyResult<u64> {
            Ok(self.inner.last_flush()?)
        }

        #[getter]
        fn n_bins(&self) -> usize {
            self.inner.n_bins()
        }

        fn export_snapshot(&self) -> PyResult<HashMap<String, Vec<f64>>> {
            // determine snapshot shape
            Ok(self.inner.export_snapshot()?)
        }

        fn export_baseline(&self) -> Vec<f64> {
            self.inner.export_baseline()
        }
    }

    /// Python exposed api for discrete categorical PSI
    #[pyclass]
    pub(crate) struct PyCategoricalDataDrift {
        inner: CategoricalDataDrift<String>,
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

        fn compute_drift(
            &mut self,
            runtime_data: Vec<String>,
            drift_type: String,
        ) -> PyResult<f64> {
            let drift_t = DataDriftType::try_from(drift_type.as_str())?;
            Ok(self.inner.compute_drift(&runtime_data, drift_t)?)
        }

        #[pyo3(signature = (new_baseline))]
        fn reset_baseline(&mut self, new_baseline: Vec<String>) -> PyResult<()> {
            self.inner.reset_baseline(&new_baseline)?;
            Ok(())
        }

        fn export_baseline(&self) -> HashMap<String, f64> {
            self.inner.baseline.export_baseline()
        }
    }

    enum PyCategoricalStreamingDriftInner {
        FlushMode(StreamingCategoricalDataDrift<String, FlushModeMark>),
        DecayMode(StreamingCategoricalDataDrift<String, DecayModeMark>),
    }

    impl PyCategoricalStreamingDriftInner {
        fn serialize_from_args(
            baseline_data: &[String],
            decay_opt: Option<u64>,
            flush_rate_opt: Option<u64>,
            flush_cadence_opt: Option<u64>,
        ) -> PyResult<Self> {
            let mode = StreamingInitMode::get_mode(
                decay_opt.is_some(),
                flush_rate_opt.is_some(),
                flush_cadence_opt.is_some(),
            )?;

            match mode {
                StreamingInitMode::Flush => {
                    let inner = match (flush_rate_opt, flush_cadence_opt) {
                        (Some(flush_rate), Some(cadence)) => {
                            StreamingCategoricalDataDrift::new_flush(
                                baseline_data,
                                flush_rate,
                                Duration::from_secs(cadence),
                            )?
                        }
                        _ => StreamingCategoricalDataDrift::default_flush(baseline_data)?,
                    };
                    Ok(Self::FlushMode(inner))
                }
                StreamingInitMode::Decay => {
                    let decay_raw = decay_opt.ok_or(DriftError::UnsupportedConfig)?;
                    let decay = NonZeroU64::new(decay_raw).ok_or(DriftError::UnsupportedConfig)?;
                    let inner = StreamingCategoricalDataDrift::new_decay(baseline_data, decay)?;
                    Ok(Self::DecayMode(inner))
                }
            }
        }

        fn compute_drift(&mut self, drift_type: DataDriftType) -> Result<f64, DriftError> {
            match self {
                Self::FlushMode(inner) => inner.compute_drift(drift_type),
                Self::DecayMode(inner) => inner.compute_drift(drift_type),
            }
        }

        fn compute_drift_multiple_criteria(
            &mut self,
            drift_types: &[DataDriftType],
        ) -> Result<Vec<f64>, DriftError> {
            match self {
                Self::FlushMode(inner) => inner.compute_drift_multiple_criteria(drift_types),
                Self::DecayMode(inner) => inner.compute_drift_multiple_criteria(drift_types),
            }
        }

        fn reset_baseline(&mut self, new_baseline: &[String]) -> Result<(), DriftError> {
            match self {
                Self::FlushMode(inner) => inner.reset_baseline(new_baseline),
                Self::DecayMode(inner) => inner.reset_baseline(new_baseline),
            }
        }

        fn update_stream(&mut self, item: &String) {
            match self {
                Self::FlushMode(inner) => inner.update_stream(item),
                Self::DecayMode(inner) => inner.update_stream(item),
            }
        }

        fn update_stream_batch(&mut self, runtime_data: &[String]) -> Result<(), DriftError> {
            match self {
                Self::FlushMode(inner) => inner.update_stream_batch(runtime_data),
                Self::DecayMode(inner) => inner.update_stream_batch(runtime_data),
            }
        }

        fn flush(&mut self) -> Result<(), DriftError> {
            match self {
                Self::FlushMode(inner) => {
                    inner.flush();
                    Ok(())
                }
                Self::DecayMode(_) => Err(DriftError::UnsupportedOperation),
            }
        }

        fn total_samples(&self) -> usize {
            match self {
                Self::FlushMode(inner) => inner.total_samples(),
                Self::DecayMode(inner) => inner.total_samples(),
            }
        }

        fn last_flush(&self) -> Result<u64, DriftError> {
            match self {
                Self::FlushMode(inner) => Ok(inner.last_flush()),
                Self::DecayMode(_) => Err(DriftError::UnsupportedOperation),
            }
        }

        fn export_snapshot(&self) -> ahash::HashMap<String, f64> {
            match self {
                Self::FlushMode(inner) => inner.export_snapshot(),
                Self::DecayMode(inner) => inner.export_snapshot(),
            }
        }

        fn export_baseline(&self) -> ahash::HashMap<String, f64> {
            match self {
                Self::FlushMode(inner) => inner.export_baseline(),
                Self::DecayMode(inner) => inner.export_baseline(),
            }
        }
    }
    /// Exposed Python APIs for streaming categorical PSI
    #[pyclass]
    pub(crate) struct PyStreamingCategoricalDataDrift {
        inner: PyCategoricalStreamingDriftInner,
    }

    #[pymethods]
    impl PyStreamingCategoricalDataDrift {
        #[new]
        fn new(
            baseline_data: Vec<String>,
            decay: Option<u64>,
            flush_rate: Option<u64>,
            flush_cadence: Option<u64>,
        ) -> PyResult<PyStreamingCategoricalDataDrift> {
            let inner = PyCategoricalStreamingDriftInner::serialize_from_args(
                &baseline_data,
                decay,
                flush_rate,
                flush_cadence,
            )?;

            Ok(Self { inner })
        }

        #[pyo3(signature = (new_baseline))]
        fn reset_baseline(&mut self, new_baseline: Vec<String>) -> PyResult<()> {
            self.inner.reset_baseline(&new_baseline)?;
            Ok(())
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

        fn flush(&mut self) -> PyResult<()> {
            Ok(self.inner.flush()?)
        }

        #[getter]
        fn total_samples(&self) -> usize {
            self.inner.total_samples()
        }

        #[getter]
        fn last_flush(&self) -> PyResult<u64> {
            Ok(self.inner.last_flush()?)
        }

        fn export_snapshot(&self) -> HashMap<String, f64> {
            self.inner.export_snapshot()
        }

        fn export_baseline(&self) -> HashMap<String, f64> {
            self.inner.export_baseline()
        }
    }
}
