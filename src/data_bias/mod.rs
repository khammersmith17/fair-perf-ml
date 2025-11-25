use crate::data_handler::{apply_label_continuous, apply_label_discrete};
use crate::errors::DataBiasError;
use crate::metrics::{DataBiasMetric, FULL_DATA_BIAS_METRICS};
use crate::runtime::DataBiasRuntime;
use std::collections::HashMap;
pub(crate) mod core;
pub(crate) mod statistics;

pub type DataBiasAnalysisReport = HashMap<DataBiasMetric, f32>;

#[cfg(feature = "python")]
pub(crate) mod py_api {
    use super::core::data_bias_analysis_core;
    use super::data_bias_runtime_check;
    use crate::data_handler::py_types_handler::{apply_label, report_to_py_dict};
    use crate::metrics::{DataBiasMetric, DataBiasMetricVec};
    use crate::reporting::DriftReport;
    use crate::runtime::DataBiasRuntime;
    use numpy::PyUntypedArray;
    use pyo3::{
        exceptions::{PyTypeError, PyValueError},
        prelude::*,
        types::{IntoPyDict, PyDict},
        Bound, PyErr, PyResult, Python,
    };
    use std::collections::HashMap;

    impl Into<PyErr> for super::DataBiasError {
        fn into(self) -> PyErr {
            let err_msg = self.to_string();
            PyValueError::new_err(err_msg)
        }
    }

    #[pyfunction]
    #[pyo3(signature = (feature_array, ground_truth_array, feature_label_or_threshold, ground_truth_label_or_threshold))]
    pub fn py_data_bias_analyzer<'py>(
        py: Python<'py>,
        feature_array: &Bound<'py, PyUntypedArray>,
        ground_truth_array: &Bound<'py, PyUntypedArray>,
        feature_label_or_threshold: Bound<'py, PyAny>, //fix
        ground_truth_label_or_threshold: Bound<'py, PyAny>, //fix
    ) -> PyResult<Bound<'py, PyDict>> {
        let gt = match apply_label(py, ground_truth_array, ground_truth_label_or_threshold) {
            Ok(array) => array,
            Err(err) => return Err(PyTypeError::new_err(err.to_string())),
        };

        let feats = match apply_label(py, feature_array, feature_label_or_threshold) {
            Ok(array) => array,
            Err(err) => return Err(PyTypeError::new_err(err.to_string())),
        };

        let res = match data_bias_analysis_core(gt, feats) {
            Ok(r) => r,
            Err(e) => return Err(e.into()),
        };

        let py_dict = report_to_py_dict(py, res);
        Ok(py_dict)
    }

    #[pyfunction]
    #[pyo3(signature = (baseline, latest, threshold=0.10))]
    pub fn py_data_bias_runtime_check<'py>(
        py: Python<'py>,
        baseline: HashMap<String, f32>,
        latest: HashMap<String, f32>,
        threshold: f32,
    ) -> PyResult<Bound<'py, PyDict>> {
        let current = match DataBiasRuntime::try_from(latest) {
            Ok(obj) => obj,
            Err(e) => return Err(e.into()),
        };

        let baseline = match DataBiasRuntime::try_from(baseline) {
            Ok(obj) => obj,
            Err(e) => return Err(e.into()),
        };
        let failure_report = data_bias_runtime_check(baseline, current, threshold);
        let drift_report: DriftReport<DataBiasMetric> = DriftReport::from_runtime(failure_report);

        Ok(drift_report.into_py_dict(py)?)
    }

    #[pyfunction]
    #[pyo3(signature = (baseline, latest, metrics, threshold=0.10))]
    pub fn py_data_bias_partial_check<'py>(
        py: Python<'py>,
        baseline: HashMap<String, f32>,
        latest: HashMap<String, f32>,
        metrics: Vec<String>,
        threshold: f32,
    ) -> PyResult<Bound<'py, PyDict>> {
        let metrics = match DataBiasMetricVec::try_from(metrics.as_slice()) {
            Ok(m) => m,
            Err(e) => return Err(e.into()),
        };
        let current = match DataBiasRuntime::try_from(latest) {
            Ok(obj) => obj,
            Err(e) => return Err(e.into()),
        };

        let baseline = match DataBiasRuntime::try_from(baseline) {
            Ok(obj) => obj,
            Err(e) => return Err(e.into()),
        };
        let failure_report: HashMap<DataBiasMetric, f32> =
            current.runtime_check(baseline, threshold, metrics.as_ref());

        let drift_report: DriftReport<DataBiasMetric> = DriftReport::from_runtime(failure_report);
        Ok(drift_report.into_py_dict(py)?)
    }
}

// allow for different types between gt and feature value
pub fn data_bias_analyzer<F, G>(
    feature_array: &[F],
    ground_truth_array: &[G],
    feature_label: Option<F>,
    feature_threshold: Option<F>,
    ground_truth_label: Option<G>,
    ground_truth_threshold: Option<G>,
) -> Result<DataBiasAnalysisReport, DataBiasError>
where
    G: PartialEq + PartialOrd,
    F: PartialEq + PartialOrd,
{
    let labeled_feats = if let Some(feat_l) = feature_label {
        apply_label_discrete(feature_array, feat_l)
    } else if let Some(feat_t) = feature_threshold {
        apply_label_continuous(feature_array, feat_t)
    } else {
        return Err(DataBiasError::NoSegmentationParameters);
    };

    let labeled_gt = if let Some(gt_l) = ground_truth_label {
        apply_label_discrete(ground_truth_array, gt_l)
    } else if let Some(gt_t) = ground_truth_threshold {
        apply_label_continuous(ground_truth_array, gt_t)
    } else {
        return Err(DataBiasError::NoSegmentationParameters);
    };

    core::data_bias_analysis_core(labeled_feats, labeled_gt)
}

pub fn data_bias_runtime_check(
    baseline: DataBiasRuntime,
    current: DataBiasRuntime,
    threshold: f32,
) -> DataBiasAnalysisReport {
    current.runtime_check(baseline, threshold, &FULL_DATA_BIAS_METRICS)
}

pub fn data_bias_partial_check(
    baseline: DataBiasRuntime,
    latest: DataBiasRuntime,
    metrics: Vec<DataBiasMetric>,
    threshold: f32,
) -> DataBiasAnalysisReport {
    latest.runtime_check(baseline, threshold, &metrics)
}

pub struct PreTraining {
    pub facet_a: Vec<i16>,
    pub facet_d: Vec<i16>,
}

impl PreTraining {
    pub fn generate(&self) -> PreTrainingComputations {
        let a_acceptance: f32 = self.facet_a.iter().sum::<i16>() as f32 / self.facet_a.len() as f32;
        let d_acceptance: f32 = self.facet_d.iter().sum::<i16>() as f32 / self.facet_d.len() as f32;
        PreTrainingComputations {
            a_acceptance,
            d_acceptance,
        }
    }
}

pub struct PreTrainingComputations {
    pub a_acceptance: f32,
    pub d_acceptance: f32,
}
