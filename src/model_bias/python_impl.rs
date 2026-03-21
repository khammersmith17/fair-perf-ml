#[cfg(feature = "python")]
pub(crate) mod py_api {
    use crate::data_handler::py_types_handler::{apply_label, report_to_py_dict};
    use crate::metrics::{ModelBiasMetric, ModelBiasMetricVec, FULL_MODEL_BIAS_METRICS};
    use crate::model_bias::core::post_training_bias;
    use crate::model_bias::DiscretePostTraining;
    use crate::reporting::DriftReport;
    use crate::runtime::ModelBiasRuntime;
    use numpy::PyUntypedArray;
    use pyo3::{
        prelude::*,
        types::{IntoPyDict, PyDict},
        Bound, PyResult, Python,
    };
    use std::collections::HashMap;

    #[pyfunction]
    #[pyo3(signature = (baseline, latest, metrics, threshold=0.10))]
    pub fn py_model_bias_partial_check<'py>(
        py: Python<'py>,
        baseline: HashMap<String, f32>,
        latest: HashMap<String, f32>,
        metrics: Vec<String>,
        threshold: f32,
    ) -> PyResult<Bound<'py, PyDict>> {
        let metrics = ModelBiasMetricVec::try_from(metrics.as_slice())?;
        let current = ModelBiasRuntime::try_from(latest)?;
        let baseline = ModelBiasRuntime::try_from(baseline)?;

        let failure_report: HashMap<ModelBiasMetric, f32> =
            current.runtime_check(baseline, threshold, metrics.as_ref());

        let drift_report: DriftReport<ModelBiasMetric> = DriftReport::from_runtime(failure_report);

        let py_dict = drift_report.into_py_dict(py)?;

        Ok(py_dict)
    }

    #[pyfunction]
    #[pyo3(signature = (baseline, latest, threshold=0.10))]
    pub fn py_model_bias_runtime_check<'py>(
        py: Python<'py>,
        baseline: HashMap<String, f32>,
        latest: HashMap<String, f32>,
        threshold: f32,
    ) -> PyResult<Bound<'py, PyDict>> {
        let current = ModelBiasRuntime::try_from(latest)?;
        let baseline = ModelBiasRuntime::try_from(baseline)?;

        let failure_report: HashMap<ModelBiasMetric, f32> =
            current.runtime_check(baseline, threshold, &FULL_MODEL_BIAS_METRICS);

        let drift_report: DriftReport<ModelBiasMetric> = DriftReport::from_runtime(failure_report);
        let py_dict = drift_report.into_py_dict(py)?;
        Ok(py_dict)
    }

    #[pyfunction]
    #[pyo3(signature = (feature_array, ground_truth_array, prediction_array, feature_label_or_threshold,
        ground_truth_label_or_threshold, prediction_label_or_threshold))]
    pub fn py_model_bias_analyzer<'py>(
        py: Python<'py>,
        feature_array: &Bound<'py, PyUntypedArray>,
        ground_truth_array: &Bound<'py, PyUntypedArray>,
        prediction_array: &Bound<'py, PyUntypedArray>,
        feature_label_or_threshold: Bound<'py, PyAny>,
        ground_truth_label_or_threshold: Bound<'py, PyAny>,
        prediction_label_or_threshold: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let preds: Vec<i16> = apply_label(py, prediction_array, prediction_label_or_threshold)?;
        let gt: Vec<i16> = apply_label(py, ground_truth_array, ground_truth_label_or_threshold)?;
        let feats: Vec<i16> = apply_label(py, feature_array, feature_label_or_threshold)?;

        let post_training_data = DiscretePostTraining::new(&feats, &preds, &gt)?;
        let analysis_res = post_training_bias(&post_training_data);

        let py_dict = report_to_py_dict(py, analysis_res?);
        Ok(py_dict)
    }
}

#[cfg(feature = "python")]
pub(crate) mod py_streaming_api {
    use crate::data_handler::{
        py_types_handler::{report_to_py_dict, PyDictResult},
        BiasSegmentationCriteria, BiasSegmentationType,
    };
    use crate::metrics::ModelBiasMetricVec;
    use crate::model_bias::streaming::StreamingModelBias;
    use pyo3::prelude::*;
    use pyo3::types::IntoPyDict;

    #[pyclass]
    pub(crate) struct PyModelBiasStreaming {
        inner: StreamingModelBias<i8, i8, i8>,
    }

    // class segmentation will happen in python layer
    // and as such, not exposing the method to update seg criteria
    #[pymethods]
    impl PyModelBiasStreaming {
        #[new]
        fn new(
            features: Vec<i8>,
            predictions: Vec<i8>,
            ground_truth: Vec<i8>,
        ) -> PyResult<PyModelBiasStreaming> {
            let inner = StreamingModelBias::new(
                &features,
                BiasSegmentationCriteria::new(1_i8, BiasSegmentationType::Label),
                &predictions,
                BiasSegmentationCriteria::new(1_i8, BiasSegmentationType::Label),
                &ground_truth,
                BiasSegmentationCriteria::new(1_i8, BiasSegmentationType::Label),
            )?;

            Ok(PyModelBiasStreaming { inner })
        }

        fn push(&mut self, feature: i8, pred: i8, gt: i8) {
            self.inner.push(&feature, &pred, &gt);
        }

        fn push_batch(&mut self, feature: Vec<i8>, pred: Vec<i8>, gt: Vec<i8>) -> PyResult<()> {
            self.inner.push_batch(&feature, &pred, &gt)?;
            Ok(())
        }

        fn flush(&mut self) {
            self.inner.flush()
        }

        fn reset_baseline(&mut self, feature: Vec<i8>, pred: Vec<i8>, gt: Vec<i8>) -> PyResult<()> {
            self.inner.reset_baseline(&feature, &pred, &gt)?;
            Ok(())
        }
        fn drift_snapshot<'py>(&self, py: Python<'py>) -> PyDictResult<'py> {
            let report = self.inner.drift_snapshot()?;
            Ok(report_to_py_dict(py, report))
        }

        fn drift_report<'py>(&self, py: Python<'py>, drift_threshold: f32) -> PyDictResult<'py> {
            let report = self.inner.drift_report(Some(drift_threshold))?;
            report.into_py_dict(py)
        }

        fn drift_report_partial_metrics<'py>(
            &self,
            py: Python<'py>,
            metrics: Vec<String>,
            drift_threshold: f32,
        ) -> PyDictResult<'py> {
            let m_vec = ModelBiasMetricVec::try_from(metrics.as_ref())?;
            let report = self
                .inner
                .drift_report_partial_metrics(m_vec.as_ref(), Some(drift_threshold))?;
            report.into_py_dict(py)
        }

        fn performance_snapshot<'py>(&self, py: Python<'py>) -> PyDictResult<'py> {
            let report = self.inner.performance_snapshot()?;

            Ok(report_to_py_dict(py, report))
        }
    }
}
