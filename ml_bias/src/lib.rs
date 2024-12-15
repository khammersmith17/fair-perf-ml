use numpy::PyUntypedArray;
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use std::collections::HashMap;

mod data_bias;
use data_bias::pre_training_bias;

mod model_bias;
use model_bias::post_training_bias;

mod data_handler;
use data_handler::{apply_label, perform_segmentation_data_bias, perform_segmentation_model_bias};

mod runtime;
use runtime::{DataBiasRuntime, ModelBiasRuntime};

#[pyfunction]
#[pyo3(signature = (
    baseline,
    latest)
)]
pub fn data_bias_runtime_check<'py>(
    baseline: HashMap<String, f32>,
    latest: HashMap<String, f32>,
) -> PyResult<HashMap<String, String>> {
    let current = match DataBiasRuntime::new(latest) {
        Ok(obj) => obj,
        Err(_) => return Err(PyValueError::new_err("Invalid metrics body passed")),
    };

    let baseline = match DataBiasRuntime::new(baseline) {
        Ok(obj) => obj,
        Err(_) => return Err(PyValueError::new_err("Invalid baseline body passed")),
    };
    let mut runtime_result: HashMap<String, String> = current.runtime_check(baseline);
    if runtime_result.len() > 0 {
        runtime_result.insert("status".to_string(), "failed".to_string());
    } else {
        runtime_result.insert("status".to_string(), "passed".to_string());
    }
    Ok(runtime_result)
}

#[pyfunction]
#[pyo3(signature = (
    baseline,
    latest)
)]
pub fn model_bias_runtime_check<'py>(
    baseline: HashMap<String, f32>,
    latest: HashMap<String, f32>,
) -> PyResult<HashMap<String, String>> {
    let current = match ModelBiasRuntime::new(latest) {
        Ok(obj) => obj,
        Err(_) => return Err(PyValueError::new_err("Invalid metrics body passed")),
    };
    let baseline = match ModelBiasRuntime::new(baseline) {
        Ok(obj) => obj,
        Err(_) => return Err(PyValueError::new_err("Invalid baseline body passed")),
    };
    let mut runtime_result: HashMap<String, String> = current.runtime_check(baseline);
    if runtime_result.len() > 0 {
        runtime_result.insert("status".to_string(), "failed".to_string());
    } else {
        runtime_result.insert("status".to_string(), "passed".to_string());
    }
    Ok(runtime_result)
}

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
    py: Python<'_>,
    feature_array: &Bound<'_, PyUntypedArray>,
    ground_truth_array: &Bound<'_, PyUntypedArray>,
    prediction_array: &Bound<'_, PyUntypedArray>,
    feature_label_or_threshold: Bound<'py, PyAny>, //fix
    ground_truth_label_or_threshold: Bound<'py, PyAny>, //fix
    prediction_label_or_threshold: Bound<'py, PyAny>, // fix
) -> PyResult<HashMap<String, f32>> {
    let labeled_predictions: Vec<i16> =
        match apply_label(py, prediction_array, prediction_label_or_threshold) {
            Ok(array) => array,
            Err(err) => return Err(PyTypeError::new_err(err)),
        };
    let labeled_ground_truth: Vec<i16> =
        match apply_label(py, ground_truth_array, ground_truth_label_or_threshold) {
            Ok(array) => array,
            Err(err) => return Err(PyTypeError::new_err(err)),
        };
    let labeled_features: Vec<i16> =
        match apply_label(py, feature_array, feature_label_or_threshold) {
            Ok(array) => array,
            Err(err) => return Err(PyTypeError::new_err(err)),
        };
    let (facet_a_scores, facet_d_scores, facet_a_trues, facet_d_trues) =
        match perform_segmentation_model_bias(
            labeled_features,
            labeled_predictions,
            labeled_ground_truth,
        ) {
            Ok(res) => res,
            Err(err) => return Err(PyTypeError::new_err(err)),
        };
    match post_training_bias(facet_a_scores, facet_d_scores, facet_a_trues, facet_d_trues) {
        Ok(value) => Ok(value),
        Err(err) => Err(PyTypeError::new_err(err)),
    }
}

#[pyfunction]
#[pyo3(signature = (
    feature_array,
    ground_truth_array,
    feature_label_or_threshold,
    ground_truth_label_or_threshold)
)]
fn data_bias_analyzer<'py>(
    py: Python<'_>,
    feature_array: &Bound<'_, PyUntypedArray>,
    ground_truth_array: &Bound<'_, PyUntypedArray>,
    feature_label_or_threshold: Bound<'py, PyAny>, //fix
    ground_truth_label_or_threshold: Bound<'py, PyAny>, //fix
) -> PyResult<HashMap<String, f32>> {
    let labeled_ground_truth =
        match apply_label(py, ground_truth_array, ground_truth_label_or_threshold) {
            Ok(array) => array,
            Err(err) => return Err(PyTypeError::new_err(err)),
        };

    let labeled_feature = match apply_label(py, feature_array, feature_label_or_threshold) {
        Ok(array) => array,
        Err(err) => return Err(PyTypeError::new_err(err)),
    };

    let (facet_a_trues, facet_d_trues) =
        match perform_segmentation_data_bias(labeled_feature, labeled_ground_truth) {
            Ok(values) => values,
            Err(err) => return Err(PyTypeError::new_err(err)),
        };

    match pre_training_bias(facet_a_trues, facet_d_trues) {
        Ok(result) => Ok(result),
        Err(err) => Err(PyTypeError::new_err(err)),
    }
}

/// A Python module implemented in Rust.
#[pymodule]
fn ml_bias(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(model_bias_analyzer, m)?)?;
    m.add_function(wrap_pyfunction!(data_bias_analyzer, m)?)?;
    m.add_function(wrap_pyfunction!(data_bias_runtime_check, m)?)?;
    m.add_function(wrap_pyfunction!(model_bias_runtime_check, m)?)?;

    Ok(())
}
