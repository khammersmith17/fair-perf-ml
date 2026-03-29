#[cfg(feature = "python")]
pub(crate) mod py_api {
    use crate::data_bias::{core::pre_training_bias, data_bias_runtime_check, PreTraining};
    use crate::data_handler::py_types_handler::{
        apply_label, label_python_bias_explicit_seg, report_to_py_dict, PyDictResult,
    };
    use crate::errors::{BiasError, InvalidMetricError};
    use crate::metrics::{DataBiasMetric, DataBiasMetricVec};
    use crate::reporting::{DataBiasAnalysisReport, DriftReport};
    use crate::runtime::DataBiasRuntime;
    use numpy::PyUntypedArray;
    use pyo3::{prelude::*, types::IntoPyDict, Bound, Python};
    use std::collections::HashMap;

    fn data_bias_analysis_core(
        labeled_features: Vec<i16>,
        labeled_ground_truth: Vec<i16>,
    ) -> Result<DataBiasAnalysisReport, BiasError> {
        let pre_training = PreTraining::new_from_labeled(&labeled_features, &labeled_ground_truth)?;
        Ok(pre_training_bias(pre_training)?)
    }

    /// Method to perform data bias analysis
    #[pyfunction]
    #[pyo3(signature = (feature_array, ground_truth_array, feature_label_or_threshold, ground_truth_label_or_threshold))]
    pub fn py_data_bias_analyzer<'py>(
        py: Python<'py>,
        feature_array: &Bound<'py, PyUntypedArray>,
        ground_truth_array: &Bound<'py, PyUntypedArray>,
        feature_label_or_threshold: Bound<'py, PyAny>,
        ground_truth_label_or_threshold: Bound<'py, PyAny>,
    ) -> PyDictResult<'py> {
        let gt = apply_label(py, ground_truth_array, ground_truth_label_or_threshold)?;
        let feats = apply_label(py, feature_array, feature_label_or_threshold)?;
        let res = data_bias_analysis_core(feats, gt)?;

        Ok(report_to_py_dict(py, res))
    }

    #[pyfunction]
    #[pyo3(signature = (
        feature_array,
        feat_segmentation_threshold,
        feat_segmentation_label,
        feat_threshold_type,
        ground_truth_array,
        gt_segmentation_threshold,
        gt_segmentation_label,
        gt_threshold_type
    ))]
    pub fn py_data_bias_analyzer_explicit_seg<'py>(
        py: Python<'py>,
        feature_array: &Bound<'py, PyUntypedArray>,
        feat_segmentation_threshold: Option<f64>,
        feat_segmentation_label: Option<String>,
        feat_threshold_type: Option<String>,
        ground_truth_array: &Bound<'py, PyUntypedArray>,
        gt_segmentation_threshold: Option<f64>,
        gt_segmentation_label: Option<String>,
        gt_threshold_type: Option<String>,
    ) -> PyDictResult<'py> {
        let labeled_feat = label_python_bias_explicit_seg(
            feature_array,
            feat_segmentation_threshold,
            feat_segmentation_label,
            feat_threshold_type,
        )?;

        let labeled_gt = label_python_bias_explicit_seg(
            ground_truth_array,
            gt_segmentation_threshold,
            gt_segmentation_label,
            gt_threshold_type,
        )?;

        let res = data_bias_analysis_core(labeled_feat, labeled_gt)?;
        Ok(report_to_py_dict(py, res))
    }

    #[pyfunction]
    #[pyo3(signature = (baseline, latest, threshold=0.10))]
    pub fn py_data_bias_runtime_check<'py>(
        py: Python<'py>,
        baseline: HashMap<String, f32>,
        latest: HashMap<String, f32>,
        threshold: f32,
    ) -> PyDictResult<'py> {
        let bl = convert_db_analysis(baseline)?;
        let rt = convert_db_analysis(latest)?;

        let drift_report = data_bias_runtime_check(bl, rt, Some(threshold))?;

        drift_report.into_py_dict(py)
    }

    #[pyfunction]
    #[pyo3(signature = (baseline, latest, metrics, threshold=0.10))]
    pub fn py_data_bias_partial_check<'py>(
        py: Python<'py>,
        baseline: HashMap<String, f32>,
        latest: HashMap<String, f32>,
        metrics: Vec<String>,
        threshold: f32,
    ) -> PyDictResult<'py> {
        let metrics = DataBiasMetricVec::try_from(metrics.as_slice())?;
        let current = DataBiasRuntime::try_from(latest)?;
        let baseline = DataBiasRuntime::try_from(baseline)?;

        let failure_report: HashMap<DataBiasMetric, f32> =
            current.runtime_check(baseline, threshold, metrics.as_ref());

        let drift_report: DriftReport<DataBiasMetric> = DriftReport::from_runtime(failure_report);
        drift_report.into_py_dict(py)
    }

    // Internal method to take analysis report from python, limited to a string for the metric
    // labels, into enum metric labels to be used here internally.
    fn convert_db_analysis(
        report: HashMap<String, f32>,
    ) -> Result<DataBiasAnalysisReport, InvalidMetricError> {
        let mut invalid_metrics: Vec<String> = Vec::new();
        let mut res_map: DataBiasAnalysisReport = HashMap::with_capacity(7);

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
