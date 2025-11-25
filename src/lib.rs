pub mod data_bias;
pub(crate) mod data_handler;
pub mod drift;
mod macros;
pub mod metrics;
pub mod model_bias;
pub mod model_perf;
mod models;
pub mod reporting;
mod runtime;

/*
* TODO:
* for runtime checks
*   1. write python wrapper functions around all core logic
*       the python wrapper will perform type serialization and coerce to correct types
*   2. Refactor analysis logic to also have python wrappers
*       type corecion is performed in the python wrapper
*       core logic uses native rust types
*   3. Determine cleaner python/ rust api
*   4. The goal is to extend the idea of having the crate useable in Rust and Python contexts
*   5. clean up lib.rs to only expose functions via ffi
* */

#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
#[pymodule]
#[pyo3(name = "_fair_perf_ml")]
fn fair_perf_ml(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    use data_bias::py_api::{
        py_data_bias_analyzer, py_data_bias_partial_check, py_data_bias_runtime_check,
    };
    use drift::psi::{
        PyCategoricalPSI, PyContinuousPSI, PyStreamingCategoricalPSI, PyStreamingContinuousPSI,
    };
    use model_bias::py_api::{
        model_bias_analyzer, model_bias_partial_check, model_bias_runtime_check,
    };
    use model_perf::py_api::{
        py_model_perf_class_rt_full, py_model_perf_class_rt_partial, py_model_perf_classification,
        py_model_perf_lin_reg_rt_full, py_model_perf_lin_reg_rt_partial,
        py_model_perf_log_reg_rt_full, py_model_perf_log_reg_rt_partial,
        py_model_perf_logistic_regression, py_model_perf_regression,
    };

    m.add_function(wrap_pyfunction!(model_bias_analyzer, m)?)?;
    m.add_function(wrap_pyfunction!(py_data_bias_analyzer, m)?)?;
    m.add_function(wrap_pyfunction!(py_data_bias_runtime_check, m)?)?;
    m.add_function(wrap_pyfunction!(py_data_bias_partial_check, m)?)?;
    m.add_function(wrap_pyfunction!(model_bias_runtime_check, m)?)?;
    m.add_function(wrap_pyfunction!(model_bias_partial_check, m)?)?;
    m.add_function(wrap_pyfunction!(py_model_perf_regression, m)?)?;
    m.add_function(wrap_pyfunction!(py_model_perf_class_rt_full, m)?)?;
    m.add_function(wrap_pyfunction!(py_model_perf_class_rt_partial, m)?)?;
    m.add_function(wrap_pyfunction!(py_model_perf_classification, m)?)?;
    m.add_function(wrap_pyfunction!(py_model_perf_log_reg_rt_full, m)?)?;
    m.add_function(wrap_pyfunction!(py_model_perf_log_reg_rt_partial, m)?)?;
    m.add_function(wrap_pyfunction!(py_model_perf_logistic_regression, m)?)?;
    m.add_function(wrap_pyfunction!(py_model_perf_lin_reg_rt_full, m)?)?;
    m.add_function(wrap_pyfunction!(py_model_perf_lin_reg_rt_partial, m)?)?;

    m.add_class::<PyCategoricalPSI>()?;
    m.add_class::<PyContinuousPSI>()?;
    m.add_class::<PyStreamingContinuousPSI>()?;
    m.add_class::<PyStreamingCategoricalPSI>()?;
    Ok(())
}
