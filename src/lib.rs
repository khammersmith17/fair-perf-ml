use numpy::PyUntypedArray;
use pyo3::{
    exceptions::{PySystemError, PyTypeError, PyValueError},
    prelude::*,
    types::{IntoPyDict, PyDict},
};
use std::collections::HashMap;
mod data_bias;
use data_bias::core::data_bias_analysis_core;
mod model_bias;
use model_bias::core::model_bias_analysis_core;
pub(crate) mod data_handler;
use data_handler::apply_label;
mod runtime;
use runtime::{DataBiasRuntime, ModelBiasRuntime};
mod models;
use models::ModelType;
mod macros;
mod model_perf;
use model_perf::{
    model_perf_classification, model_perf_logistic_regression, model_perf_regression,
    BinaryClassificationReport, LinearRegressionReport, LogisticRegressionReport,
};
pub mod drift;
pub mod metrics;
use metrics::{
    ClassificationEvaluationMetric, ClassificationMetricVec, DataBiasMetric, DataBiasMetricVec,
    LinearRegressionEvaluationMetric, LinearRegressionMetricVec, LogisticRegressionMetricVec,
    ModelBiasMetricVec, FULL_BINARY_CLASSIFICATION_METRICS, FULL_DATA_BIAS_METRICS,
    FULL_LOGISTIC_REGRESSION_METRICS, FULL_MODEL_BIAS_METRICS, FULL_REGRESSION_METRICS,
};

pub mod reporting;
use reporting::DriftReport;

/*
* TODO:
* for runtime checks
*   1. write python wrapper functions around all core logic
*       the python wrapper will perform type serialization and coerce to correct types
*   2. Refactor analysis logic to also have python wrappers
*       type corecion is performed in the python wrapper
*       core logic uses native rust types
*   3. All python specific logic needs to be wrapped around python feature
*   4. For things that accept strings, accept Vec<String>/String in python wrapper
*   5. In core rust, accpet Into<Type> where Type is the enum that represents the string value
*      being passed
*   6. The goal is to extend the idea of having the crate useable in Rust and Python contexts
* */

pub fn data_bias_runtime_check(
    baseline: DataBiasRuntime,
    current: DataBiasRuntime,
    threshold: f32,
) -> HashMap<DataBiasMetric, f32> {
    current.runtime_check(baseline, threshold, &FULL_DATA_BIAS_METRICS)
}

#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(signature = (
    baseline,
    latest,
    threshold=0.10
)
)]
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

pub fn data_bias_partial_check(
    baseline: DataBiasRuntime,
    latest: DataBiasRuntime,
    metrics: Vec<DataBiasMetric>,
    threshold: f32,
) -> HashMap<DataBiasMetric, f32> {
    latest.runtime_check(baseline, threshold, &metrics)
}

#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(signature = (
    baseline,
    latest,
    metrics,
    threshold=0.10
)
)]
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

#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(signature = (
    baseline,
    latest,
    metrics,
    threshold=0.10
)
)]
fn model_bias_partial_check<'py>(
    py: Python<'py>,
    baseline: HashMap<String, f32>,
    latest: HashMap<String, f32>,
    metrics: Vec<String>,
    threshold: f32,
) -> PyResult<Bound<'py, PyDict>> {
    let metrics = match ModelBiasMetricVec::try_from(metrics.as_slice()) {
        Ok(m) => m,
        Err(_) => return Err(PyValueError::new_err("Invalid ModelBias metric passed")),
    };

    let current = match ModelBiasRuntime::try_from(latest) {
        Ok(obj) => obj,
        Err(_) => return Err(PyValueError::new_err("Invalid metrics body passed")),
    };
    let baseline = match ModelBiasRuntime::try_from(baseline) {
        Ok(obj) => obj,
        Err(_) => return Err(PyValueError::new_err("Invalid baseline body passed")),
    };
    let failure_report: HashMap<metrics::ModelBiasMetric, f32> =
        current.runtime_check(baseline, threshold, metrics.as_ref());

    let drift_report: DriftReport<metrics::ModelBiasMetric> =
        DriftReport::from_runtime(failure_report);

    let py_dict = drift_report.into_py_dict(py)?;

    Ok(py_dict)
}

#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(signature = (
    baseline,
    latest,
    threshold=0.10
)
)]
pub fn model_bias_runtime_check<'py>(
    py: Python<'py>,
    baseline: HashMap<String, f32>,
    latest: HashMap<String, f32>,
    threshold: f32,
) -> PyResult<Bound<'py, PyDict>> {
    let current = match ModelBiasRuntime::try_from(latest) {
        Ok(obj) => obj,
        Err(_) => return Err(PyValueError::new_err("Invalid metrics body passed")),
    };
    let baseline = match ModelBiasRuntime::try_from(baseline) {
        Ok(obj) => obj,
        Err(_) => return Err(PyValueError::new_err("Invalid baseline body passed")),
    };
    let failure_report: HashMap<metrics::ModelBiasMetric, f32> =
        current.runtime_check(baseline, threshold, &FULL_MODEL_BIAS_METRICS);

    let drift_report: DriftReport<metrics::ModelBiasMetric> =
        DriftReport::from_runtime(failure_report);

    let py_dict = drift_report.into_py_dict(py)?;

    Ok(py_dict)
}

#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(signature = (
    feature_array,
    ground_truth_array,
    prediction_array,
    feature_label_or_threshold,
    ground_truth_label_or_threshold,
    prediction_label_or_threshold)
)]
pub fn model_bias_analyzer<'py>(
    py: Python<'py>,
    feature_array: &Bound<'py, PyUntypedArray>,
    ground_truth_array: &Bound<'py, PyUntypedArray>,
    prediction_array: &Bound<'py, PyUntypedArray>,
    feature_label_or_threshold: Bound<'py, PyAny>,
    ground_truth_label_or_threshold: Bound<'py, PyAny>,
    prediction_label_or_threshold: Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyDict>> {
    let preds: Vec<i16> = match apply_label(py, prediction_array, prediction_label_or_threshold) {
        Ok(array) => array,
        Err(err) => return Err(PyTypeError::new_err(err.to_string())),
    };
    let gt: Vec<i16> = match apply_label(py, ground_truth_array, ground_truth_label_or_threshold) {
        Ok(array) => array,
        Err(err) => return Err(PyTypeError::new_err(err.to_string())),
    };
    let feats: Vec<i16> = match apply_label(py, feature_array, feature_label_or_threshold) {
        Ok(array) => array,
        Err(err) => return Err(PyTypeError::new_err(err.to_string())),
    };

    let analysis_res = match model_bias_analysis_core(feats, preds, gt) {
        Ok(res) => res,
        Err(e) => return Err(PyValueError::new_err(e)),
    };

    let py_dict = analysis_res.into_py_dict(py)?;
    Ok(py_dict)
}

#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(signature = (
    feature_array,
    ground_truth_array,
    feature_label_or_threshold,
    ground_truth_label_or_threshold)
)]
fn py_data_bias_analyzer<'py>(
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
        Err(e) => return Err(PyValueError::new_err(e)),
    };

    let py_dict = res.into_py_dict(py)?;
    Ok(py_dict)
}

#[pyfunction]
#[pyo3(signature = (
    y_pred,
    y_true)
)]
fn model_performance_regression<'py>(
    py: Python<'_>,
    y_pred: &Bound<'_, PyUntypedArray>,
    y_true: &Bound<'_, PyUntypedArray>,
) -> PyResult<HashMap<String, f32>> {
    match model_perf_regression(py, y_pred, y_true) {
        Ok(res) => Ok(res),
        Err(e) => Err(PyValueError::new_err(format!(
            "Invalid arrays for y_pred and y_true: {}",
            e
        ))),
    }
}

#[pyfunction]
#[pyo3(signature = (
    y_pred,
    y_true)
)]
fn model_performance_classification<'py>(
    py: Python<'_>,
    y_pred: &Bound<'_, PyUntypedArray>,
    y_true: &Bound<'_, PyUntypedArray>,
) -> PyResult<HashMap<String, f32>> {
    match model_perf_classification(py, y_pred, y_true) {
        Ok(res) => Ok(res),
        Err(e) => Err(PyValueError::new_err(format!(
            "Invalid arrays for y_pred and y_true: {}",
            e
        ))),
    }
}

#[pyfunction]
#[pyo3(signature = (
    y_pred,
    y_true,
    decision_threshold=0.5
)
)]
fn model_performance_logisitic_regression<'py>(
    py: Python<'_>,
    y_pred: &Bound<'_, PyUntypedArray>,
    y_true: &Bound<'_, PyUntypedArray>,
    decision_threshold: f32,
) -> PyResult<HashMap<String, f32>> {
    match model_perf_logistic_regression(py, y_pred, y_true, decision_threshold) {
        Ok(res) => Ok(res),
        Err(e) => Err(PyValueError::new_err(format!(
            "Invalid arrays for y_pred and y_true: {}",
            e
        ))),
    }
}

#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(signature = (
    model_type,
    baseline,
    latest,
    evaluation_metrics,
    threshold=0.10
)
)]
fn model_performance_runtime_entry_partial<'py>(
    py: Python<'py>,
    model_type: String,
    baseline: HashMap<String, f32>,
    latest: HashMap<String, f32>,
    evaluation_metrics: Vec<String>,
    threshold: f32,
) -> PyResult<Bound<'py, PyDict>> {
    let model_type: ModelType = match ModelType::try_from(model_type.as_str()) {
        Ok(t) => t,
        Err(_) => return Err(PyValueError::new_err("Invalid model type")),
    };

    match model_type {
        ModelType::LinearRegression => {
            let metrics_to_eval: LinearRegressionMetricVec =
                match LinearRegressionMetricVec::try_from(evaluation_metrics.as_slice()) {
                    Ok(m) => m,
                    Err(e) => return Err(e.into()),
                };
            let drift_report = regression_performance_runtime(
                baseline,
                latest,
                metrics_to_eval.as_ref(),
                threshold,
            )
            .unwrap();
            Ok(drift_report.into_py_dict(py)?)
        }
        ModelType::LogisticRegression => {
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
        ModelType::BinaryClassification => {
            let metrics_to_eval: ClassificationMetricVec =
                match ClassificationMetricVec::try_from(evaluation_metrics.as_slice()) {
                    Ok(m) => m,
                    Err(_) => return Err(PyValueError::new_err("Invalid metric name passed")),
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
    }
}

//TODO: fix unwraps here, create erorr type
#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(signature = (
    model_type,
    baseline,
    latest,
    threshold=0.10
)
)]
fn model_performance_runtime_entry_full<'py>(
    py: Python<'py>,
    model_type: String,
    baseline: HashMap<String, f32>,
    latest: HashMap<String, f32>,
    threshold: f32,
) -> PyResult<Bound<'py, PyDict>> {
    let model_type: ModelType = match ModelType::try_from(model_type.as_str()) {
        Ok(t) => t,
        Err(_) => return Err(PyValueError::new_err("Invalid model type")),
    };

    match model_type {
        ModelType::LinearRegression => {
            let drift_report = regression_performance_runtime(
                baseline,
                latest,
                &FULL_REGRESSION_METRICS,
                threshold,
            )
            .unwrap();
            Ok(drift_report.into_py_dict(py)?)
        }
        ModelType::LogisticRegression => {
            let drift_report = logistic_performance_runtime(
                baseline,
                latest,
                &FULL_LOGISTIC_REGRESSION_METRICS,
                threshold,
            )
            .unwrap();
            Ok(drift_report.into_py_dict(py)?)
        }
        ModelType::BinaryClassification => {
            let report = classification_performance_runtime(
                baseline,
                latest,
                &FULL_BINARY_CLASSIFICATION_METRICS,
                threshold,
            )
            .unwrap();

            Ok(report.into_py_dict(py)?)
        }
    }
}

fn classification_performance_runtime(
    baseline: HashMap<String, f32>,
    latest: HashMap<String, f32>,
    metrics: &[ClassificationEvaluationMetric],
    threshold: f32,
) -> Result<DriftReport<metrics::ClassificationEvaluationMetric>, String> {
    let baseline = match BinaryClassificationReport::try_from(baseline) {
        Ok(v) => v,
        Err(e) => return Err(format!("Invalid baseline report: {}", e)),
    };
    let latest = match BinaryClassificationReport::try_from(latest) {
        Ok(v) => v,
        Err(e) => return Err(format!("Invalid baseline report: {}", e)),
    };
    let res = latest.compare_to_baseline(metrics, &baseline, threshold);

    Ok(DriftReport::from_runtime(res))
}

fn logistic_performance_runtime(
    baseline: HashMap<String, f32>,
    latest: HashMap<String, f32>,
    metrics: &[ClassificationEvaluationMetric],
    threshold: f32,
) -> Result<DriftReport<ClassificationEvaluationMetric>, String> {
    let baseline = match LogisticRegressionReport::try_from(baseline) {
        Ok(v) => v,
        Err(e) => return Err(format!("Invalid baseline report: {}", e)),
    };
    let latest = match LogisticRegressionReport::try_from(latest) {
        Ok(v) => v,
        Err(e) => return Err(format!("Invalid baseline report: {}", e)),
    };
    let res = latest.compare_to_baseline(metrics, &baseline, threshold);
    Ok(DriftReport::from_runtime(res))
}

fn regression_performance_runtime(
    baseline: HashMap<String, f32>,
    latest: HashMap<String, f32>,
    evaluation_metrics: &[LinearRegressionEvaluationMetric],
    threshold: f32,
) -> Result<DriftReport<metrics::LinearRegressionEvaluationMetric>, String> {
    let baseline: LinearRegressionReport = match LinearRegressionReport::try_from(baseline) {
        Ok(val) => val,
        Err(e) => return Err(format!("Invalid baseline report: {}", e)),
    };
    let latest: LinearRegressionReport = match LinearRegressionReport::try_from(latest) {
        Ok(val) => val,
        Err(e) => return Err(format!("Invalid latest report: {}", e)),
    };

    let results = latest.compare_to_baseline(&evaluation_metrics, &baseline, threshold);
    Ok(DriftReport::from_runtime(results))
}

#[cfg(feature = "python")]
#[pymodule]
#[pyo3(name = "_fair_perf_ml")]
fn fair_perf_ml(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(model_bias_analyzer, m)?)?;
    m.add_function(wrap_pyfunction!(py_data_bias_analyzer, m)?)?;
    m.add_function(wrap_pyfunction!(py_data_bias_runtime_check, m)?)?;
    m.add_function(wrap_pyfunction!(py_data_bias_partial_check, m)?)?;
    m.add_function(wrap_pyfunction!(model_bias_runtime_check, m)?)?;
    m.add_function(wrap_pyfunction!(model_bias_partial_check, m)?)?;
    m.add_function(wrap_pyfunction!(model_performance_regression, m)?)?;
    m.add_function(wrap_pyfunction!(model_performance_classification, m)?)?;
    m.add_function(wrap_pyfunction!(model_performance_logisitic_regression, m)?)?;
    m.add_function(wrap_pyfunction!(model_performance_runtime_entry_full, m)?)?;
    m.add_function(wrap_pyfunction!(
        model_performance_runtime_entry_partial,
        m
    )?)?;
    Ok(())
}
