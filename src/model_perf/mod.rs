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
        model_perf_binary_classification, model_perf_logistic_regression, model_perf_regression,
        regression_performance_runtime, ModelPerformanceError,
    };
    use crate::data_handler::py_types_handler::{determine_type, report_to_py_dict, PassedType};
    use crate::metrics::{
        ClassificationMetricVec, LinearRegressionMetricVec, LogisticRegressionMetricVec,
        FULL_BINARY_CLASSIFICATION_METRICS, FULL_LOGISTIC_REGRESSION_METRICS,
        FULL_REGRESSION_METRICS,
    };
    use numpy::PyUntypedArray;
    use pyo3::{
        exceptions::{PyTypeError, PyValueError},
        prelude::*,
        types::{IntoPyDict, PyDict},
        Bound,
    };
    use std::collections::HashMap;

    impl Into<PyErr> for ModelPerformanceError {
        fn into(self) -> PyErr {
            let err_msg = self.to_string();
            PyValueError::new_err(err_msg)
        }
    }

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
            threshold,
        )
        .unwrap();
        Ok(drift_report.into_py_dict(py)?)
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
        let drift_report =
            regression_performance_runtime(baseline, latest, metrics_to_eval.as_ref(), threshold)
                .unwrap();
        Ok(drift_report.into_py_dict(py)?)
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
        let drift_report =
            logistic_performance_runtime(baseline, latest, metrics_to_eval.as_ref(), threshold)
                .unwrap();
        Ok(drift_report.into_py_dict(py)?)
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
            threshold,
        )
        .unwrap();
        Ok(drift_report.into_py_dict(py)?)
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
            threshold,
        )
        .unwrap();
        Ok(drift_report.into_py_dict(py)?)
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
            threshold,
        )
        .unwrap();
        Ok(drift_report.into_py_dict(py)?)
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
                    let data = item?.extract::<f32>()? as f32;
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
        let report = match model_perf_regression(&y_true, &y_pred) {
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
        let Ok(y_true) = convert_f32(py, y_true_src, true_type) else {
            return Err(PyValueError::new_err("Invalid types passed"));
        };
        let pred_type = determine_type(py, y_pred_src);
        let Ok(y_pred) = convert_f32(py, y_pred_src, pred_type) else {
            return Err(PyValueError::new_err("Invalid types passed"));
        };

        let report = match model_perf_binary_classification(&y_true, &y_pred, 1_f32) {
            Ok(r) => r,
            Err(e) => return Err(e.into()),
        };

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
        let Ok(y_true) = convert_f32(py, y_true_src, true_type) else {
            return Err(PyTypeError::new_err("Invalid types passed"));
        };
        let pred_type = determine_type(py, y_pred_src);
        let Ok(y_proba) = convert_f32(py, y_pred_src, pred_type) else {
            return Err(PyTypeError::new_err("Invalid type passed"));
        };

        let report = match model_perf_logistic_regression(&y_true, &y_proba, threshold) {
            Ok(r) => r,
            Err(e) => return Err(e.into()),
        };
        Ok(report_to_py_dict(py, report))
    }
}

/// Perform the full suite of Post Training Bias analysis on a discrete dataset. This method can be
/// used to get a point in time snapshot into model performance across a discrete dataset. This
/// method
pub fn classification_performance_runtime(
    baseline: HashMap<String, f32>,
    latest: HashMap<String, f32>,
    metrics: &[ClassificationEvaluationMetric],
    threshold: f32,
) -> Result<DriftReport<ClassificationEvaluationMetric>, String> {
    let baseline = match BinaryClassificationRuntime::try_from(baseline) {
        Ok(v) => v,
        Err(e) => return Err(format!("Invalid baseline report: {}", e)),
    };
    let latest = match BinaryClassificationRuntime::try_from(latest) {
        Ok(v) => v,
        Err(e) => return Err(format!("Invalid baseline report: {}", e)),
    };
    let res = latest.compare_to_baseline(metrics, &baseline, threshold);

    Ok(DriftReport::<ClassificationEvaluationMetric>::from_runtime(
        res,
    ))
}

pub fn logistic_performance_runtime(
    baseline: HashMap<String, f32>,
    latest: HashMap<String, f32>,
    metrics: &[ClassificationEvaluationMetric],
    threshold: f32,
) -> Result<DriftReport<ClassificationEvaluationMetric>, String> {
    let baseline = match LogisticRegressionRuntime::try_from(baseline) {
        Ok(v) => v,
        Err(e) => return Err(format!("Invalid baseline report: {}", e)),
    };
    let latest = match LogisticRegressionRuntime::try_from(latest) {
        Ok(v) => v,
        Err(e) => return Err(format!("Invalid baseline report: {}", e)),
    };
    let res = latest.compare_to_baseline(metrics, &baseline, threshold);
    Ok(DriftReport::from_runtime(res))
}

pub fn regression_performance_runtime(
    baseline: HashMap<String, f32>,
    latest: HashMap<String, f32>,
    evaluation_metrics: &[LinearRegressionEvaluationMetric],
    threshold: f32,
) -> Result<DriftReport<LinearRegressionEvaluationMetric>, String> {
    let baseline: LinearRegressionRuntime = match LinearRegressionRuntime::try_from(baseline) {
        Ok(val) => val,
        Err(e) => return Err(format!("Invalid baseline report: {}", e)),
    };
    let latest: LinearRegressionRuntime = match LinearRegressionRuntime::try_from(latest) {
        Ok(val) => val,
        Err(e) => return Err(format!("Invalid latest report: {}", e)),
    };

    let results = latest.compare_to_baseline(&evaluation_metrics, &baseline, threshold);
    Ok(DriftReport::from_runtime(results))
}

/// Method to perform binary classification analysis on arbitrary type and label. Traditionally a
/// positive label would equate to 1 if the result is numeric or otherwise. This method allows for
/// arbitrary label type, to account for situations where a model might produce a String label, an
/// enum, and so on. Given the arbitrary labeling, the type must implement 'PartialOrd'
pub fn model_perf_binary_classification<T>(
    y_true: &[T],
    y_pred: &[T],
    positive_label: T,
) -> Result<BinaryClassificationAnalysisReport, ModelPerformanceError>
where
    T: PartialOrd,
{
    let report = BinaryClassificationRuntime::new::<T>(y_true, y_pred, positive_label)?;
    Ok(report.generate_report())
}

pub fn model_perf_regression<T>(
    y_true: &[T],
    y_pred: &[T],
) -> Result<LinearRegressionAnalysisReport, ModelPerformanceError>
where
    T: Into<f64> + Copy,
{
    let report: LinearRegressionRuntime = LinearRegressionRuntime::new(y_true, y_pred)?;
    Ok(report.generate_report())
}

pub fn model_perf_logistic_regression(
    y_true: &[f32],
    y_proba: &[f32],
    threshold: f32,
) -> Result<LogisticRegressionAnalysisReport, ModelPerformanceError> {
    let lr_report: LogisticRegressionRuntime =
        LogisticRegressionRuntime::new(y_true, y_proba, threshold)?;
    Ok(lr_report.generate_report())
}
