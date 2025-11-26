use crate::data_handler::BiasDataPayload;
use crate::errors::{BiasError, DataBiasRuntimeError};
use crate::metrics::{DataBiasMetric, FULL_DATA_BIAS_METRICS};
use crate::reporting::{DataBiasAnalysisReport, DriftReport};
use crate::runtime::DataBiasRuntime;
pub(crate) mod core;
pub(crate) mod statistics;

#[cfg(feature = "python")]
pub(crate) mod py_api {
    use super::core::data_bias_analysis_core;
    use super::data_bias_runtime_check;
    use crate::data_handler::py_types_handler::{apply_label, report_to_py_dict};
    use crate::errors::InvalidMetricError;
    use crate::metrics::{DataBiasMetric, DataBiasMetricVec};
    use crate::reporting::DriftReport;
    use crate::runtime::DataBiasRuntime;
    use numpy::PyUntypedArray;
    use pyo3::{
        exceptions::PyTypeError,
        prelude::*,
        types::{IntoPyDict, PyDict},
        Bound, PyResult, Python,
    };
    use std::collections::HashMap;

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
        let bl = match convert_db_analysis(baseline) {
            Ok(b) => b,
            Err(e) => return Err(e.into()),
        };
        let rt = match convert_db_analysis(latest) {
            Ok(b) => b,
            Err(e) => return Err(e.into()),
        };

        let drift_report = match data_bias_runtime_check(bl, rt, threshold) {
            Ok(r) => r,
            Err(e) => return Err(e.into()),
        };

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

    fn convert_db_analysis(
        report: HashMap<String, f32>,
    ) -> Result<super::DataBiasAnalysisReport, InvalidMetricError> {
        let mut invalid_metrics: Vec<String> = Vec::new();
        let mut res_map: super::DataBiasAnalysisReport = HashMap::with_capacity(7);

        for (metric_str, value) in report.into_iter() {
            if let Ok(m) = DataBiasMetric::try_from(metric_str.as_ref()) {
                res_map.insert(m, value);
            } else {
                invalid_metrics.push(metric_str);
            }
        }

        if !invalid_metrics.is_empty() {
            return Err(InvalidMetricError::DataBiasMetricError(invalid_metrics));
        }
        Ok(res_map)
    }
}

// allow for different types between gt and feature value
pub fn data_bias_analyzer<'a, F, G>(
    feature: BiasDataPayload<'a, F>,
    ground_truth: BiasDataPayload<'a, G>,
) -> Result<DataBiasAnalysisReport, BiasError>
where
    G: PartialEq + PartialOrd,
    F: PartialEq + PartialOrd,
{
    let labeled_feats = feature.generate_labeled_data();

    let labeled_gt = ground_truth.generate_labeled_data();

    Ok(core::data_bias_analysis_core(labeled_feats, labeled_gt)?)
}

pub fn data_bias_runtime_check(
    baseline_report: DataBiasAnalysisReport,
    current_report: DataBiasAnalysisReport,
    threshold: f32,
) -> Result<DriftReport<DataBiasMetric>, DataBiasRuntimeError> {
    let baseline = DataBiasRuntime::try_from(baseline_report)?;
    let current = DataBiasRuntime::try_from(current_report)?;
    let check_res = current.runtime_check(baseline, threshold, &FULL_DATA_BIAS_METRICS);

    Ok(DriftReport::from_runtime(check_res))
}

pub fn data_bias_partial_check(
    baseline_report: DataBiasAnalysisReport,
    current_report: DataBiasAnalysisReport,
    metrics: Vec<DataBiasMetric>,
    threshold: f32,
) -> Result<DriftReport<DataBiasMetric>, DataBiasRuntimeError> {
    let baseline = DataBiasRuntime::try_from(baseline_report)?;
    let current = DataBiasRuntime::try_from(current_report)?;
    let check_res = current.runtime_check(baseline, threshold, &metrics);

    Ok(DriftReport::from_runtime(check_res))
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
