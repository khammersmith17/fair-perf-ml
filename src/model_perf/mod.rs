use crate::{
    errors::ModelPerformanceError,
    metrics::{ClassificationEvaluationMetric, LinearRegressionEvaluationMetric},
    reporting::{
        BinaryClassificationAnalysisReport, DriftReport, LinearRegressionAnalysisReport,
        LogisticRegressionAnalysisReport,
    },
    runtime::{BinaryClassificationRuntime, LinearRegressionRuntime, LogisticRegressionRuntime},
};
use std::collections::HashMap;
pub mod statistics;
pub mod streaming;

#[cfg(feature = "python")]
pub(crate) mod py_api {
    /// Python interface into the core logic.
    use super::{
        classification_performance_runtime, logistic_performance_runtime,
        model_perf_binary_classification_analysis, model_perf_logistic_regression_analysis,
        model_perf_regression_analysis, regression_performance_runtime, ModelPerformanceError,
    };
    use crate::data_handler::py_types_handler::{determine_type, report_to_py_dict, PassedType};
    use crate::metrics::{
        ClassificationMetricVec, LinearRegressionMetricVec, LogisticRegressionMetricVec,
        FULL_BINARY_CLASSIFICATION_METRICS, FULL_LOGISTIC_REGRESSION_METRICS,
        FULL_REGRESSION_METRICS,
    };
    use numpy::PyUntypedArray;
    use pyo3::{
        exceptions::PyValueError,
        prelude::*,
        types::{IntoPyDict, PyDict},
        Bound,
    };
    use std::collections::HashMap;

    #[pyfunction]
    #[pyo3(signature = (baseline, latest, threshold=0.10))]
    pub fn py_model_perf_lin_reg_rt_full<'py>(
        py: Python<'py>,
        baseline: HashMap<String, f32>,
        latest: HashMap<String, f32>,
        threshold: f32,
    ) -> PyResult<Bound<'py, PyDict>> {
        let drift_report = regression_performance_runtime(
            baseline,
            latest,
            FULL_REGRESSION_METRICS.as_slice(),
            Some(threshold),
        )
        .unwrap();
        drift_report.into_py_dict(py)
    }

    #[pyfunction]
    #[pyo3(signature = (baseline, latest, evaluation_metrics, threshold=0.10))]
    pub fn py_model_perf_lin_reg_rt_partial<'py>(
        py: Python<'py>,
        baseline: HashMap<String, f32>,
        latest: HashMap<String, f32>,
        evaluation_metrics: Vec<String>,
        threshold: f32,
    ) -> PyResult<Bound<'py, PyDict>> {
        let metrics_to_eval: LinearRegressionMetricVec =
            match LinearRegressionMetricVec::try_from(evaluation_metrics.as_slice()) {
                Ok(m) => m,
                Err(e) => return Err(e.into()),
            };
        let drift_report = regression_performance_runtime(
            baseline,
            latest,
            metrics_to_eval.as_ref(),
            Some(threshold),
        )
        .unwrap();
        drift_report.into_py_dict(py)
    }

    #[pyfunction]
    #[pyo3(signature = (baseline, latest, evaluation_metrics, threshold=0.10))]
    pub fn py_model_perf_log_reg_rt_partial<'py>(
        py: Python<'py>,
        baseline: HashMap<String, f32>,
        latest: HashMap<String, f32>,
        evaluation_metrics: Vec<String>,
        threshold: f32,
    ) -> PyResult<Bound<'py, PyDict>> {
        let metrics_to_eval: LogisticRegressionMetricVec =
            match LogisticRegressionMetricVec::try_from(evaluation_metrics.as_slice()) {
                Ok(m) => m,
                Err(e) => return Err(e.into()),
            };
        let drift_report = logistic_performance_runtime(
            baseline,
            latest,
            metrics_to_eval.as_ref(),
            Some(threshold),
        )
        .unwrap();
        drift_report.into_py_dict(py)
    }

    #[pyfunction]
    #[pyo3(signature = (baseline, latest,  threshold=0.10))]
    pub fn py_model_perf_log_reg_rt_full<'py>(
        py: Python<'py>,
        baseline: HashMap<String, f32>,
        latest: HashMap<String, f32>,
        threshold: f32,
    ) -> PyResult<Bound<'py, PyDict>> {
        let drift_report = logistic_performance_runtime(
            baseline,
            latest,
            FULL_LOGISTIC_REGRESSION_METRICS.as_slice(),
            Some(threshold),
        )
        .unwrap();
        drift_report.into_py_dict(py)
    }

    #[pyfunction]
    #[pyo3(signature = (baseline, latest, evaluation_metrics, threshold=0.10))]
    pub fn py_model_perf_class_rt_partial<'py>(
        py: Python<'py>,
        baseline: HashMap<String, f32>,
        latest: HashMap<String, f32>,
        evaluation_metrics: Vec<String>,
        threshold: f32,
    ) -> PyResult<Bound<'py, PyDict>> {
        let metrics_to_eval: ClassificationMetricVec =
            match ClassificationMetricVec::try_from(evaluation_metrics.as_slice()) {
                Ok(m) => m,
                Err(e) => return Err(e.into()),
            };
        let drift_report = classification_performance_runtime(
            baseline,
            latest,
            metrics_to_eval.as_ref(),
            Some(threshold),
        )
        .unwrap();
        drift_report.into_py_dict(py)
    }

    #[pyfunction]
    #[pyo3(signature = (baseline, latest,  threshold=0.10))]
    pub fn py_model_perf_class_rt_full<'py>(
        py: Python<'py>,
        baseline: HashMap<String, f32>,
        latest: HashMap<String, f32>,
        threshold: f32,
    ) -> PyResult<Bound<'py, PyDict>> {
        let drift_report = classification_performance_runtime(
            baseline,
            latest,
            FULL_BINARY_CLASSIFICATION_METRICS.as_slice(),
            Some(threshold),
        )
        .unwrap();
        drift_report.into_py_dict(py)
    }

    pub fn validate_and_cast_regression(
        py: Python<'_>,
        y_true_src: &Bound<'_, PyUntypedArray>,
        y_pred_src: &Bound<'_, PyUntypedArray>,
    ) -> Result<(Vec<f32>, Vec<f32>), PyErr> {
        let y_true: Vec<f32> = convert_f32(py, y_pred_src, determine_type(py, y_true_src))?;
        let y_pred: Vec<f32> = convert_f32(py, y_true_src, determine_type(py, y_pred_src))?;
        Ok((y_true, y_pred))
    }

    pub fn convert_f32(
        _py: Python<'_>,
        arr: &Bound<'_, PyUntypedArray>,
        passed_type: PassedType,
    ) -> Result<Vec<f32>, PyErr> {
        let mut data_container: Vec<f32> = Vec::with_capacity(arr.len()?);
        // pulls the py data type out
        // applying labels as usize

        match passed_type {
            PassedType::Float => {
                for item in arr.try_iter()? {
                    let data = item?.extract::<f64>()? as f32;
                    data_container.push(data);
                }
            }
            PassedType::Integer => {
                for item in arr.try_iter()? {
                    let data = item?.extract::<f32>()?;
                    data_container.push(data)
                }
            }
            _ => return Err(ModelPerformanceError::UnsupportedTypeError.into()),
        };
        Ok(data_container)
    }

    #[pyfunction]
    #[pyo3(signature = (y_pred_src, y_true_src))]
    pub fn py_model_perf_regression<'py>(
        py: Python<'py>,
        y_pred_src: &Bound<'_, PyUntypedArray>,
        y_true_src: &Bound<'_, PyUntypedArray>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let Ok((y_true, y_pred)) = validate_and_cast_regression(py, y_true_src, y_pred_src) else {
            return Err(PyValueError::new_err("Invalid types passed"));
        };
        let report = match model_perf_regression_analysis(&y_true, &y_pred) {
            Ok(r) => r,
            Err(e) => return Err(e.into()),
        };
        Ok(report_to_py_dict(py, report))
    }

    #[pyfunction]
    #[pyo3(signature = (y_pred_src, y_true_src))]
    pub fn py_model_perf_classification<'py>(
        py: Python<'py>,
        y_pred_src: &Bound<'py, PyUntypedArray>,
        y_true_src: &Bound<'py, PyUntypedArray>,
    ) -> PyResult<Bound<'py, PyDict>> {
        // coerce py types
        let true_type: PassedType = determine_type(py, y_true_src);
        let y_true = convert_f32(py, y_true_src, true_type)?;

        let pred_type = determine_type(py, y_pred_src);
        let y_pred = convert_f32(py, y_pred_src, pred_type)?;

        let report = model_perf_binary_classification_analysis(&y_true, &y_pred, 1_f32)?;

        Ok(report_to_py_dict(py, report))
    }

    #[pyfunction]
    #[pyo3(signature = (y_pred_src, y_true_src, threshold))]
    pub fn py_model_perf_logistic_regression<'py>(
        py: Python<'py>,
        y_pred_src: &Bound<'py, PyUntypedArray>,
        y_true_src: &Bound<'py, PyUntypedArray>,
        threshold: f32,
    ) -> PyResult<Bound<'py, PyDict>> {
        // coerce py types
        let true_type: PassedType = determine_type(py, y_true_src);
        let y_true = convert_f32(py, y_true_src, true_type)?;
        let pred_type = determine_type(py, y_pred_src);
        let y_proba = convert_f32(py, y_pred_src, pred_type)?;
        let report = model_perf_logistic_regression_analysis(&y_true, &y_proba, threshold)?;
        Ok(report_to_py_dict(py, report))
    }
}

/// Method to perform the runtime analysis against the baseline set for logisitc regression models.
/// Takes in the baseline report, the runtime report, the metrics you are interested in evaluating on,
/// and an acceptable drift percentage threshold. When a threshold is not passed, it defaults to 0.10,
/// or a 10% drift threshold.
pub fn classification_performance_runtime(
    baseline: HashMap<String, f32>,
    latest: HashMap<String, f32>,
    metrics: &[ClassificationEvaluationMetric],
    threshold_perc_opt: Option<f32>,
) -> Result<DriftReport<ClassificationEvaluationMetric>, ModelPerformanceError> {
    let threshold = threshold_perc_opt.unwrap_or(0.10_f32);
    let baseline = BinaryClassificationRuntime::try_from(baseline)?;
    let latest = BinaryClassificationRuntime::try_from(latest)?;
    let res = latest.compare_to_baseline(metrics, &baseline, threshold);
    Ok(DriftReport::<ClassificationEvaluationMetric>::from_runtime(
        res,
    ))
}

/// Method to perform the runtime analysis against the baseline set for logisitc regression models.
/// Takes in the baseline report, the runtime report, the metrics you are interested in evaluating on,
/// and an acceptable drift percentage threshold. When a threshold is not passed, it defaults to 0.10,
/// or a 10% drift threshold.
pub fn logistic_performance_runtime(
    baseline: HashMap<String, f32>,
    latest: HashMap<String, f32>,
    metrics: &[ClassificationEvaluationMetric],
    threshold_perc_opt: Option<f32>,
) -> Result<DriftReport<ClassificationEvaluationMetric>, ModelPerformanceError> {
    let threshold = threshold_perc_opt.unwrap_or(0.10_f32);
    let baseline = LogisticRegressionRuntime::try_from(baseline)?;
    let latest = LogisticRegressionRuntime::try_from(latest)?;
    let res = latest.compare_to_baseline(metrics, &baseline, threshold);
    Ok(DriftReport::from_runtime(res))
}

/// Method to perform the runtime analysis against the baseline set for linear regression models.
/// Takes in the baseline report, the runtime report, the metrics you are interested in evaluating on,
/// and an acceptable drift percentage threshold. When a threshold is not passed, it defaults to 0.10,
/// or a 10% drift threshold.
pub fn regression_performance_runtime(
    baseline: HashMap<String, f32>,
    latest: HashMap<String, f32>,
    evaluation_metrics: &[LinearRegressionEvaluationMetric],
    threshold_perc_opt: Option<f32>,
) -> Result<DriftReport<LinearRegressionEvaluationMetric>, ModelPerformanceError> {
    let threshold = threshold_perc_opt.unwrap_or(0.10_f32);
    let baseline: LinearRegressionRuntime = LinearRegressionRuntime::try_from(baseline)?;
    let latest: LinearRegressionRuntime = LinearRegressionRuntime::try_from(latest)?;
    let results = latest.compare_to_baseline(evaluation_metrics, &baseline, threshold);
    Ok(DriftReport::from_runtime(results))
}

/// Method to perform binary classification analysis on arbitrary type and label. Traditionally a
/// positive label would equate to 1 if the result is numeric or otherwise. This method allows for
/// arbitrary label type, to account for situations where a model might produce a String label, an
/// enum, and so on. Given the arbitrary labeling, the type must implement 'PartialOrd'
pub fn model_perf_binary_classification_analysis<T>(
    y_true: &[T],
    y_pred: &[T],
    positive_label: T,
) -> Result<BinaryClassificationAnalysisReport, ModelPerformanceError>
where
    T: PartialOrd,
{
    let report = BinaryClassificationRuntime::new::<T>(y_true, y_pred, &positive_label)?;
    Ok(report.generate_report())
}

pub fn model_perf_regression_analysis<T>(
    y_true: &[T],
    y_pred: &[T],
) -> Result<LinearRegressionAnalysisReport, ModelPerformanceError>
where
    T: Into<f64> + Copy,
{
    let report: LinearRegressionRuntime = LinearRegressionRuntime::new(y_true, y_pred)?;
    Ok(report.generate_report())
}

pub fn model_perf_logistic_regression_analysis(
    y_true: &[f32],
    y_proba: &[f32],
    threshold: f32,
) -> Result<LogisticRegressionAnalysisReport, ModelPerformanceError> {
    let lr_report: LogisticRegressionRuntime =
        LogisticRegressionRuntime::new(y_true, y_proba, threshold)?;
    Ok(lr_report.generate_report())
}

#[cfg(test)]
mod test_discrete_model_perf_utilities {
    use super::*;
    use crate::metrics::{ClassificationEvaluationMetric as CM, LinearRegressionEvaluationMetric as LRM};

    // Shared binary classification data
    fn bin_true() -> Vec<i32> { vec![1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1] }
    fn bin_pred() -> Vec<i32> { vec![1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1] }

    // Shared regression data
    fn reg_true() -> Vec<f32> { vec![11.0, 12.5, 14.0, 11.7, 15.1, 15.4, 13.2, 11.5, 11.6] }
    fn reg_pred() -> Vec<f32> { vec![11.1, 12.2, 13.4, 10.7, 15.8, 16.3, 14.5, 12.3, 11.0] }

    // Logistic regression data
    fn log_true() -> Vec<f32> { vec![0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0] }
    fn log_pred() -> Vec<f32> { vec![0.7, 0.3, 0.65, 0.55, 0.1, 0.2, 0.25, 0.66, 0.12, 0.98, 0.23, 0.34, 0.67, 0.77, 0.45, 0.88] }

    // --- model_perf_binary_classification_analysis ---

    #[test]
    fn binary_analysis_returns_seven_metrics() {
        let report = model_perf_binary_classification_analysis(&bin_true(), &bin_pred(), 1).unwrap();
        assert_eq!(report.len(), 7);
    }

    #[test]
    fn binary_analysis_length_mismatch_errors() {
        assert!(model_perf_binary_classification_analysis(&[1, 0], &[1], 1).is_err());
    }

    // --- model_perf_regression_analysis ---

    #[test]
    fn regression_analysis_returns_eight_metrics() {
        let report = model_perf_regression_analysis(&reg_true(), &reg_pred()).unwrap();
        assert_eq!(report.len(), 8);
    }

    #[test]
    fn regression_analysis_length_mismatch_errors() {
        assert!(model_perf_regression_analysis(&[1.0_f32, 2.0], &[1.0_f32]).is_err());
    }

    #[test]
    fn regression_analysis_empty_errors() {
        let empty: &[f32] = &[];
        assert!(model_perf_regression_analysis(empty, empty).is_err());
    }

    // --- model_perf_logistic_regression_analysis ---

    #[test]
    fn logistic_analysis_returns_eight_metrics() {
        let report = model_perf_logistic_regression_analysis(&log_true(), &log_pred(), 0.5).unwrap();
        assert_eq!(report.len(), 8);
        assert!(report.contains_key(&CM::LogLoss));
    }

    #[test]
    fn logistic_analysis_length_mismatch_errors() {
        assert!(model_perf_logistic_regression_analysis(&[1.0_f32, 0.0], &[0.7_f32], 0.5).is_err());
    }

    // --- classification_performance_runtime ---

    #[test]
    fn classification_runtime_same_data_passes() {
        let report = model_perf_binary_classification_analysis(&bin_true(), &bin_pred(), 1).unwrap();
        let string_map: HashMap<String, f32> =
            report.into_iter().map(|(k, v)| (k.to_string(), v)).collect();
        let result = classification_performance_runtime(
            string_map.clone(),
            string_map,
            &[CM::Accuracy, CM::F1Score],
            Some(1e6),
        ).unwrap();
        assert!(result.passed);
    }

    #[test]
    fn classification_runtime_missing_key_errors() {
        let mut map = HashMap::new();
        map.insert("NotAMetric".to_string(), 0.5_f32);
        assert!(classification_performance_runtime(map.clone(), map, &[CM::Accuracy], None).is_err());
    }

    // --- logistic_performance_runtime ---

    #[test]
    fn logistic_runtime_same_data_passes() {
        let report = model_perf_logistic_regression_analysis(&log_true(), &log_pred(), 0.5).unwrap();
        let string_map: HashMap<String, f32> =
            report.into_iter().map(|(k, v)| (k.to_string(), v)).collect();
        let result = logistic_performance_runtime(
            string_map.clone(),
            string_map,
            &[CM::Accuracy, CM::LogLoss],
            Some(1e6),
        ).unwrap();
        assert!(result.passed);
    }

    #[test]
    fn logistic_runtime_missing_key_errors() {
        let mut map = HashMap::new();
        map.insert("NotAMetric".to_string(), 0.5_f32);
        assert!(logistic_performance_runtime(map.clone(), map, &[CM::Accuracy], None).is_err());
    }

    // --- regression_performance_runtime ---

    #[test]
    fn regression_runtime_same_data_passes() {
        let report = model_perf_regression_analysis(&reg_true(), &reg_pred()).unwrap();
        let string_map: HashMap<String, f32> =
            report.into_iter().map(|(k, v)| (k.to_string(), v)).collect();
        let result = regression_performance_runtime(
            string_map.clone(),
            string_map,
            &[LRM::RootMeanSquaredError, LRM::MeanAbsoluteError],
            Some(1e6),
        ).unwrap();
        assert!(result.passed);
    }

    #[test]
    fn regression_runtime_missing_key_errors() {
        let mut map = HashMap::new();
        map.insert("NotAMetric".to_string(), 0.5_f32);
        assert!(regression_performance_runtime(map.clone(), map, &[LRM::RootMeanSquaredError], None).is_err());
    }
}
