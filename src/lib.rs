pub mod data_bias;
pub(crate) mod data_handler;
pub mod drift;
pub mod errors;
mod macros;
pub mod metrics;
pub mod model_bias;
pub mod model_perf;
mod models;
pub mod reporting;
mod runtime;

/// Exposed Python APIs for the fair perf ml rust crate
#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
#[pymodule]
#[pyo3(name = "_fair_perf_ml")]
fn fair_perf_ml(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    use data_bias::py_api::{
        py_data_bias_analyzer, py_data_bias_partial_check, py_data_bias_runtime_check,
    };

    use drift::python_impl::py_api::{
        PyCategoricalDataDrift, PyContinuousDataDrift, PyStreamingCategoricalDataDriftDecay,
        PyStreamingCategoricalDataDriftFlush, PyStreamingContinuousDataDriftDecay,
        PyStreamingContinuousDataDriftFlush,
    };

    use model_bias::python_impl::py_api::{
        py_model_bias_analyzer, py_model_bias_partial_check, py_model_bias_runtime_check,
    };

    use model_perf::py_api::{
        py_model_perf_class_rt_full, py_model_perf_class_rt_partial, py_model_perf_classification,
        py_model_perf_lin_reg_rt_full, py_model_perf_lin_reg_rt_partial,
        py_model_perf_log_reg_rt_full, py_model_perf_log_reg_rt_partial,
        py_model_perf_logistic_regression, py_model_perf_regression,
    };

    use data_bias::streaming::py_api::PyDataBiasStreaming;
    use model_bias::python_impl::py_streaming_api::PyModelBiasStreaming;
    use model_perf::streaming::py_api::{
        PyBinaryClassificationStreaming, PyLinearRegressionStreaming, PyLogisticRegressionStreaming,
    };

    m.add_function(wrap_pyfunction!(py_model_bias_analyzer, m)?)?;
    m.add_function(wrap_pyfunction!(py_data_bias_analyzer, m)?)?;
    m.add_function(wrap_pyfunction!(py_data_bias_runtime_check, m)?)?;
    m.add_function(wrap_pyfunction!(py_data_bias_partial_check, m)?)?;
    m.add_function(wrap_pyfunction!(py_model_bias_runtime_check, m)?)?;
    m.add_function(wrap_pyfunction!(py_model_bias_partial_check, m)?)?;
    m.add_function(wrap_pyfunction!(py_model_perf_regression, m)?)?;
    m.add_function(wrap_pyfunction!(py_model_perf_class_rt_full, m)?)?;
    m.add_function(wrap_pyfunction!(py_model_perf_class_rt_partial, m)?)?;
    m.add_function(wrap_pyfunction!(py_model_perf_classification, m)?)?;
    m.add_function(wrap_pyfunction!(py_model_perf_log_reg_rt_partial, m)?)?;
    m.add_function(wrap_pyfunction!(py_model_perf_logistic_regression, m)?)?;
    m.add_function(wrap_pyfunction!(py_model_perf_lin_reg_rt_full, m)?)?;
    m.add_function(wrap_pyfunction!(py_model_perf_lin_reg_rt_partial, m)?)?;
    m.add_function(wrap_pyfunction!(py_model_perf_log_reg_rt_full, m)?)?;

    m.add_class::<PyCategoricalDataDrift>()?;
    m.add_class::<PyContinuousDataDrift>()?;
    m.add_class::<PyStreamingContinuousDataDriftFlush>()?;
    m.add_class::<PyStreamingContinuousDataDriftDecay>()?;
    m.add_class::<PyStreamingCategoricalDataDriftFlush>()?;
    m.add_class::<PyStreamingCategoricalDataDriftDecay>()?;
    m.add_class::<PyDataBiasStreaming>()?;
    m.add_class::<PyModelBiasStreaming>()?;
    m.add_class::<PyBinaryClassificationStreaming>()?;
    m.add_class::<PyLinearRegressionStreaming>()?;
    m.add_class::<PyLogisticRegressionStreaming>()?;

    Ok(())
}
