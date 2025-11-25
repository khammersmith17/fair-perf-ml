use crate::metrics::{DataBiasMetric, FULL_DATA_BIAS_METRICS};
use crate::runtime::DataBiasRuntime;
use std::collections::HashMap;
pub(crate) mod core;

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
            Err(e) => return Err(PyValueError::new_err(e)),
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

pub fn data_bias_runtime_check(
    baseline: DataBiasRuntime,
    current: DataBiasRuntime,
    threshold: f32,
) -> HashMap<DataBiasMetric, f32> {
    current.runtime_check(baseline, threshold, &FULL_DATA_BIAS_METRICS)
}

pub fn data_bias_partial_check(
    baseline: DataBiasRuntime,
    latest: DataBiasRuntime,
    metrics: Vec<DataBiasMetric>,
    threshold: f32,
) -> HashMap<DataBiasMetric, f32> {
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

pub fn class_imbalance(data: &PreTraining) -> f32 {
    return (data.facet_a.len() as f32 - data.facet_d.len() as f32).abs() as f32
        / (data.facet_a.len() + data.facet_d.len()) as f32;
}

pub fn diff_in_proportion_of_labels(data: &PreTraining) -> f32 {
    let q_a: f32 = data.facet_a.iter().sum::<i16>() as f32 / data.facet_a.len() as f32;
    let q_d: f32 = data.facet_d.iter().sum::<i16>() as f32 / data.facet_d.len() as f32;

    return q_a - q_d;
}

pub fn kl_divergence(data: &PreTrainingComputations) -> f32 {
    return data.a_acceptance * (data.a_acceptance / data.d_acceptance).ln()
        + (1.0_f32 - data.a_acceptance)
            * ((1.0_f32 - data.a_acceptance) / (1.0_f32 - data.d_acceptance)).ln();
}

fn ks_kl_div(p_facet: f32, p: f32) -> f32 {
    return p_facet * (p_facet / p).ln()
        + (1.0_f32 - p_facet) * ((1.0_f32 - p_facet) / (1.0_f32 - p)).ln();
}

pub fn jensen_shannon(data: &PreTraining, pre_comp: &PreTrainingComputations) -> f32 {
    let p: f32 = 0.5_f32
        * (data.facet_a.iter().sum::<i16>() as f32 / data.facet_d.len() as f32
            + data.facet_d.iter().sum::<i16>() as f32 / data.facet_a.len() as f32);

    return 0.5 * (ks_kl_div(pre_comp.a_acceptance, p) + ks_kl_div(pre_comp.d_acceptance, p));
}

pub fn lp_norm(data: &PreTrainingComputations) -> f32 {
    return ((data.a_acceptance - data.d_acceptance).powf(2.0)
        + (1.0_f32 - data.a_acceptance - 1.0_f32 - data.d_acceptance).powf(2.0))
    .sqrt();
}

pub fn total_variation_distance(data: &PreTrainingComputations) -> f32 {
    return (data.a_acceptance - data.d_acceptance).abs()
        + ((1.0_f32 - data.a_acceptance) - (1.0_f32 - data.a_acceptance)).abs();
}

pub fn kolmorogv_smirnov(data: &PreTraining) -> f32 {
    let a_0_dist: f32 = data
        .facet_a
        .iter()
        .map(|value| if *value == 0 { 1.0_f32 } else { 0.0_f32 })
        .sum::<f32>()
        / data.facet_a.len() as f32;

    let a_1_dist = data
        .facet_a
        .iter()
        .map(|value| if *value == 1 { 1.0_f32 } else { 0.0_f32 })
        .sum::<f32>()
        / data.facet_a.len() as f32;

    let d_0_dist = data
        .facet_d
        .iter()
        .map(|value| if *value == 0 { 1.0_f32 } else { 0.0_f32 })
        .sum::<f32>()
        / data.facet_d.len() as f32;

    let d_1_dist = data
        .facet_d
        .iter()
        .map(|value| if *value == 1 { 1.0_f32 } else { 0.0_f32 })
        .sum::<f32>()
        / data.facet_d.len() as f32;

    let neg_outcome_diff = (a_0_dist - d_0_dist).abs();
    let pos_outcome_diff = (a_1_dist - d_1_dist).abs();

    if neg_outcome_diff > pos_outcome_diff {
        return pos_outcome_diff;
    } else {
        return neg_outcome_diff;
    }
}

pub fn pre_training_bias(data: PreTraining) -> DataBiasAnalysisReport {
    use DataBiasMetric as M;
    let computed_data: PreTrainingComputations = data.generate();
    let mut result: HashMap<DataBiasMetric, f32> = HashMap::with_capacity(7);
    result.insert(M::ClassImbalance, class_imbalance(&data));
    result.insert(
        M::DifferenceInProportionOfLabels,
        diff_in_proportion_of_labels(&data),
    );
    result.insert(M::KlDivergence, kl_divergence(&computed_data));
    result.insert(M::JsDivergence, jensen_shannon(&data, &computed_data));
    result.insert(M::LpNorm, lp_norm(&computed_data));
    result.insert(
        M::TotalVariationDistance,
        total_variation_distance(&computed_data),
    );
    result.insert(M::KolmorogvSmirnov, kolmorogv_smirnov(&data));

    result
}
