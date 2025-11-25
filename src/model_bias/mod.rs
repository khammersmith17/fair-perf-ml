use crate::metrics::ModelBiasMetric;
use std::collections::HashMap;
pub (crate) mod statistics;

pub(crate) mod core;

pub type ModelBiasAnalysisReport = HashMap<ModelBiasMetric, f32>;


//TODO: expose rust apis, see data bias for details

#[cfg(feature = "python")]
pub(crate) mod py_api {
    use super::core::model_bias_analysis_core;
    use crate::data_handler::py_types_handler::{apply_label, report_to_py_dict};
    use crate::metrics::{ModelBiasMetric, ModelBiasMetricVec, FULL_MODEL_BIAS_METRICS};
    use crate::reporting::DriftReport;
    use crate::runtime::ModelBiasRuntime;
    use numpy::PyUntypedArray;
    use pyo3::{
        exceptions::{PyTypeError, PyValueError},
        prelude::*,
        types::{IntoPyDict, PyDict},
        Bound, PyResult, Python,
    };
    use std::collections::HashMap;

    #[pyfunction]
    #[pyo3(signature = (baseline, latest, metrics, threshold=0.10))]
    pub fn model_bias_partial_check<'py>(
        py: Python<'py>,
        baseline: HashMap<String, f32>,
        latest: HashMap<String, f32>,
        metrics: Vec<String>,
        threshold: f32,
    ) -> PyResult<Bound<'py, PyDict>> {
        let metrics = match ModelBiasMetricVec::try_from(metrics.as_slice()) {
            Ok(m) => m,
            Err(e) => return Err(e.into()),
        };

        let current = match ModelBiasRuntime::try_from(latest) {
            Ok(obj) => obj,
            Err(e) => return Err(e.into()),
        };
        let baseline = match ModelBiasRuntime::try_from(baseline) {
            Ok(obj) => obj,
            Err(e) => return Err(e.into()),
        };
        let failure_report: HashMap<ModelBiasMetric, f32> =
            current.runtime_check(baseline, threshold, metrics.as_ref());

        let drift_report: DriftReport<ModelBiasMetric> = DriftReport::from_runtime(failure_report);

        let py_dict = drift_report.into_py_dict(py)?;

        Ok(py_dict)
    }

    #[pyfunction]
    #[pyo3(signature = (baseline, latest, threshold=0.10))]
    pub fn model_bias_runtime_check<'py>(
        py: Python<'py>,
        baseline: HashMap<String, f32>,
        latest: HashMap<String, f32>,
        threshold: f32,
    ) -> PyResult<Bound<'py, PyDict>> {
        let current = match ModelBiasRuntime::try_from(latest) {
            Ok(obj) => obj,
            Err(e) => return Err(e.into()),
        };
        let baseline = match ModelBiasRuntime::try_from(baseline) {
            Ok(obj) => obj,
            Err(e) => return Err(e.into()),
        };
        let failure_report: HashMap<ModelBiasMetric, f32> =
            current.runtime_check(baseline, threshold, &FULL_MODEL_BIAS_METRICS);

        let drift_report: DriftReport<ModelBiasMetric> = DriftReport::from_runtime(failure_report);

        let py_dict = drift_report.into_py_dict(py)?;

        Ok(py_dict)
    }

    #[pyfunction]
    #[pyo3(signature = (feature_array, ground_truth_array, prediction_array, feature_label_or_threshold, 
        ground_truth_label_or_threshold, prediction_label_or_threshold))]
    pub fn model_bias_analyzer<'py>(
        py: Python<'py>,
        feature_array: &Bound<'py, PyUntypedArray>,
        ground_truth_array: &Bound<'py, PyUntypedArray>,
        prediction_array: &Bound<'py, PyUntypedArray>,
        feature_label_or_threshold: Bound<'py, PyAny>,
        ground_truth_label_or_threshold: Bound<'py, PyAny>,
        prediction_label_or_threshold: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let preds: Vec<i16> = match apply_label(py, prediction_array, prediction_label_or_threshold)
        {
            Ok(array) => array,
            Err(err) => return Err(PyTypeError::new_err(err.to_string())),
        };
        let gt: Vec<i16> =
            match apply_label(py, ground_truth_array, ground_truth_label_or_threshold) {
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

        let py_dict = report_to_py_dict(py, analysis_res);
        Ok(py_dict)
    }
}

pub struct PostTrainingData {
    pub facet_a_scores: Vec<i16>,
    pub facet_d_scores: Vec<i16>,
    pub facet_a_trues: Vec<i16>,
    pub facet_d_trues: Vec<i16>,
}

pub struct PostTrainingComputations {
    pub true_positives_a: f32,
    pub true_positives_d: f32,
    pub false_positives_a: f32,
    pub false_positives_d: f32,
    pub false_negatives_a: f32,
    pub false_negatives_d: f32,
    pub true_negatives_a: f32,
    pub true_negatives_d: f32,
}

impl PostTrainingData {
    pub fn general_data_computations(&self) -> PostTrainingComputations {
        PostTrainingComputations {
            true_positives_a: self.true_positives_a(),
            true_positives_d: self.true_positives_d(),
            false_positives_a: self.false_positives_a(),
            false_positives_d: self.false_positives_d(),
            false_negatives_a: self.false_negatives_a(),
            false_negatives_d: self.false_negatives_d(),
            true_negatives_a: self.true_negatives_a(),
            true_negatives_d: self.true_negatives_d(),
        }
    }

    fn true_positives_a(&self) -> f32 {
        self.facet_a_scores
            .iter()
            .zip(self.facet_a_trues.iter())
            .map(|(y_pred, y_true)| {
                if *y_pred == 1_i16 && *y_true == 1_i16 {
                    1_f32
                } else {
                    0_f32
                }
            })
            .sum::<f32>()
            .into()
    }

    fn true_positives_d(&self) -> f32 {
        self.facet_d_scores
            .iter()
            .zip(self.facet_d_trues.iter())
            .map(|(y_pred, y_true)| {
                if *y_pred == 1_i16 && *y_true == 1_i16 {
                    1_f32
                } else {
                    0_f32
                }
            })
            .sum::<f32>()
    }

    fn false_positives_a(&self) -> f32 {
        self.facet_a_scores
            .iter()
            .zip(self.facet_a_trues.iter())
            .map(|(y_pred, y_true)| {
                if *y_pred == 1_i16 && *y_true == 0_i16 {
                    1_f32
                } else {
                    0_f32
                }
            })
            .sum::<f32>()
    }

    fn false_positives_d(&self) -> f32 {
        self.facet_d_scores
            .iter()
            .zip(self.facet_d_trues.iter())
            .map(|(y_pred, y_true)| {
                if *y_pred == 1 && *y_true == 0 {
                    1_f32
                } else {
                    0_f32
                }
            })
            .sum::<f32>()
    }

    fn false_negatives_a(&self) -> f32 {
        self.facet_a_scores
            .iter()
            .zip(self.facet_a_trues.iter())
            .map(|(y_pred, y_true)| {
                if *y_pred == 0 && *y_true == 1 {
                    1_f32
                } else {
                    0_f32
                }
            })
            .sum::<f32>()
    }

    fn false_negatives_d(&self) -> f32 {
        self.facet_d_scores
            .iter()
            .zip(self.facet_d_trues.iter())
            .map(|(y_pred, y_true)| {
                if *y_pred == 0 && *y_true == 1 {
                    1_f32
                } else {
                    0_f32
                }
            })
            .sum::<f32>()
    }

    fn true_negatives_a(&self) -> f32 {
        self.facet_a_scores
            .iter()
            .zip(self.facet_a_trues.iter())
            .map(|(y_pred, y_true)| {
                if *y_pred == 0 && *y_true == 0 {
                    1_f32
                } else {
                    0_f32
                }
            })
            .sum::<f32>()
    }

    fn true_negatives_d(&self) -> f32 {
        self.facet_d_scores
            .iter()
            .zip(self.facet_d_trues.iter())
            .map(|(y_pred, y_true)| {
                if *y_pred == 0 && *y_true == 0 {
                    1_f32
                } else {
                    0_f32
                }
            })
            .sum::<f32>()
    }
}


