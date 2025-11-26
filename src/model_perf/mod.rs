use crate::{
    errors::ModelPerformanceError,
    metrics::{ClassificationEvaluationMetric, LinearRegressionEvaluationMetric},
    reporting::{
        BinaryClassificationReport, DriftReport, LinearRegressionReport, LogisticRegressionReport,
    },
    zip,
};

use std::collections::HashMap;

fn classification_performance_runtime(
    baseline: HashMap<String, f32>,
    latest: HashMap<String, f32>,
    metrics: &[ClassificationEvaluationMetric],
    threshold: f32,
) -> Result<DriftReport<ClassificationEvaluationMetric>, String> {
    let baseline = match BinaryClassificationAnalysisResult::try_from(baseline) {
        Ok(v) => v,
        Err(e) => return Err(format!("Invalid baseline report: {}", e)),
    };
    let latest = match BinaryClassificationAnalysisResult::try_from(latest) {
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
    let baseline = match LogisticRegressionAnalysisResult::try_from(baseline) {
        Ok(v) => v,
        Err(e) => return Err(format!("Invalid baseline report: {}", e)),
    };
    let latest = match LogisticRegressionAnalysisResult::try_from(latest) {
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
) -> Result<DriftReport<LinearRegressionEvaluationMetric>, String> {
    let baseline: LinearRegressionAnalysisResult =
        match LinearRegressionAnalysisResult::try_from(baseline) {
            Ok(val) => val,
            Err(e) => return Err(format!("Invalid baseline report: {}", e)),
        };
    let latest: LinearRegressionAnalysisResult =
        match LinearRegressionAnalysisResult::try_from(latest) {
            Ok(val) => val,
            Err(e) => return Err(format!("Invalid latest report: {}", e)),
        };

    let results = latest.compare_to_baseline(&evaluation_metrics, &baseline, threshold);
    Ok(DriftReport::from_runtime(results))
}

#[cfg(feature = "python")]
pub(crate) mod py_api {
    use super::model_perf_classification;
    use super::{
        classification_performance_runtime, logistic_performance_runtime,
        model_perf_logistic_regression, model_perf_regression, regression_performance_runtime,
        ModelPerformanceError,
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
        //

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
        let report = match model_perf_regression(y_true, y_pred) {
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

        let report = match model_perf_classification(y_true, y_pred) {
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

        let report = match model_perf_logistic_regression(y_true, y_proba, threshold) {
            Ok(r) => r,
            Err(e) => return Err(e.into()),
        };
        Ok(report_to_py_dict(py, report))
    }
}

pub fn model_perf_classification(
    y_true: Vec<f32>,
    y_pred: Vec<f32>,
) -> Result<BinaryClassificationReport, ModelPerformanceError> {
    let perf: ClassificationPerf = ClassificationPerf::new(y_true, y_pred)?;
    let report: BinaryClassificationAnalysisResult = perf.into();
    Ok(report.generate_report())
}

pub fn model_perf_regression(
    y_true: Vec<f32>,
    y_pred: Vec<f32>,
) -> Result<LinearRegressionReport, ModelPerformanceError> {
    let perf: LinearRegressionPerf = LinearRegressionPerf::new(y_true, y_pred)?;
    let report: LinearRegressionAnalysisResult = perf.into();
    Ok(report.generate_report())
}

pub fn model_perf_logistic_regression(
    y_true: Vec<f32>,
    y_proba: Vec<f32>,
    threshold: f32,
) -> Result<LogisticRegressionReport, ModelPerformanceError> {
    let perf: LogisticRegressionPerf = LogisticRegressionPerf::new(y_true, y_proba, threshold)?;
    let lr_report: LogisticRegressionAnalysisResult = perf.into();
    Ok(lr_report.generate_report())
}

struct GeneralClassificationMetrics;

impl GeneralClassificationMetrics {
    fn balanced_accuracy(rp: f32, rn: f32) -> f32 {
        rp * rn * 0.5_f32
    }

    fn precision_positive(y_pred: &[f32], y_true: &[f32]) -> f32 {
        let total_pred_positives: f32 = y_pred.iter().sum::<f32>();
        let mut true_positives: f32 = 0_f32;
        for (t, p) in zip!(y_true, y_pred) {
            if (*t - 1_f32).abs() <= f32::EPSILON && (t - p).abs() <= f32::EPSILON {
                true_positives += 1_f32;
            }
        }
        true_positives / total_pred_positives
    }

    fn precision_negative(y_pred: &[f32], y_true: &[f32], len: f32) -> f32 {
        let total_pred_negatives: f32 = len - y_pred.iter().sum::<f32>();
        let mut true_negatives: f32 = 0_f32;
        for (t, p) in zip!(y_true, y_pred) {
            if (*t - 0_f32).abs() <= f32::EPSILON && (t - p).abs() <= f32::EPSILON {
                true_negatives += 1_f32;
            }
        }
        true_negatives / total_pred_negatives
    }

    fn recall_positive(y_pred: &[f32], y_true: &[f32]) -> f32 {
        let total_true_positives: f32 = y_true.iter().sum::<f32>();
        let mut true_positives: f32 = 0_f32;
        for (t, p) in zip!(y_true, y_pred) {
            if (*t - 1_f32).abs() <= f32::EPSILON && (t - p).abs() <= f32::EPSILON {
                true_positives += 1_f32;
            }
        }
        true_positives / total_true_positives
    }

    fn recall_negative(y_pred: &[f32], y_true: &[f32], len: f32) -> f32 {
        let total_true_negatives: f32 = len - y_true.iter().sum::<f32>();
        let mut true_negatives: f32 = 0_f32;
        for (t, p) in zip!(y_true, y_pred) {
            if (*t - 0_f32).abs() <= f32::EPSILON && (t - p).abs() <= f32::EPSILON {
                true_negatives += 1_f32;
            }
        }
        true_negatives / total_true_negatives
    }

    fn accuracy(y_pred: &Vec<f32>, y_true: &Vec<f32>, mean_f: f32) -> f32 {
        let mut correct: f32 = 0_f32;
        for (t, p) in zip!(y_true, y_pred) {
            if t == p {
                correct += 1_f32;
            }
        }
        correct * mean_f
    }

    fn f1_score(rp: f32, pp: f32) -> f32 {
        2_f32 * rp * pp / (rp + pp)
    }

    fn log_loss_score(y_proba: &[f32], y_true: &[f32], mean_f: f32) -> f32 {
        let mut penalties = 0_f32;
        for (t, p) in zip!(y_true, y_proba) {
            penalties += t * f32::log10(*p) + (1_f32 - t) * f32::log10(1_f32 - p);
        }
        let res = -1_f32 * mean_f * penalties;

        if res.is_nan() {
            0_f32
        } else {
            res
        }
    }
}

//pub struct PerfEntry;

//impl PerfEntry {
#[cfg(feature = "python")]
mod py_perf_entry {}

// TODO: change to follow the pattern in bias monitors
pub struct BinaryClassificationAnalysisResult {
    balanced_accuracy: f32,
    precision_positive: f32,
    precision_negative: f32,
    recall_positive: f32,
    recall_negative: f32,
    accuracy: f32,
    f1_score: f32,
}

impl BinaryClassificationAnalysisResult {
    pub fn generate_report(&self) -> BinaryClassificationReport {
        use ClassificationEvaluationMetric as C;
        let mut map: HashMap<C, f32> = HashMap::with_capacity(7);
        map.insert(C::BalancedAccuracy, self.balanced_accuracy);
        map.insert(C::PrecisionPositive, self.precision_positive);
        map.insert(C::PrecisionNegative, self.precision_negative);
        map.insert(C::RecallPositive, self.recall_positive);
        map.insert(C::RecallNegative, self.recall_negative);
        map.insert(C::Accuracy, self.accuracy);
        map.insert(C::F1Score, self.f1_score);
        map
    }
}

impl TryFrom<HashMap<String, f32>> for BinaryClassificationAnalysisResult {
    type Error = String;
    fn try_from(map: HashMap<String, f32>) -> Result<Self, Self::Error> {
        let Some(balanced_accuracy) = map.get("BalancedAccuracy") else {
            return Err("Invalid regression report".into());
        };
        let Some(precision_positive) = map.get("PrecisionPositive") else {
            return Err("Invalid regression report".into());
        };
        let Some(precision_negative) = map.get("PrecisionNegative") else {
            return Err("Invalid regression report".into());
        };
        let Some(recall_positive) = map.get("RecallPositive") else {
            return Err("Invalid regression report".into());
        };
        let Some(recall_negative) = map.get("RecallNegative") else {
            return Err("Invalid regression report".into());
        };
        let Some(accuracy) = map.get("Accuracy") else {
            return Err("Invalid regression report".into());
        };
        let Some(f1_score) = map.get("F1Score") else {
            return Err("Invalid regression report".into());
        };

        Ok(BinaryClassificationAnalysisResult {
            balanced_accuracy: *balanced_accuracy,
            precision_positive: *precision_positive,
            precision_negative: *precision_negative,
            recall_positive: *recall_positive,
            recall_negative: *recall_negative,
            accuracy: *accuracy,
            f1_score: *f1_score,
        })
    }
}

impl BinaryClassificationAnalysisResult {
    pub fn compare_to_baseline(
        &self,
        metrics: &[ClassificationEvaluationMetric],
        baseline: &Self,
        drift_threshold: f32,
    ) -> HashMap<ClassificationEvaluationMetric, f32> {
        use ClassificationEvaluationMetric as C;
        let mut res: HashMap<C, f32> = HashMap::with_capacity(7);
        let drift_factor = 1_f32 - drift_threshold;
        // log loss should not be present here
        // so when log loss comes up, we return Err
        for m in metrics.iter() {
            match *m {
                C::BalancedAccuracy => {
                    if self.balanced_accuracy < baseline.balanced_accuracy * drift_factor {
                        res.insert(
                            C::BalancedAccuracy,
                            baseline.balanced_accuracy - self.balanced_accuracy,
                        );
                    }
                }
                C::PrecisionPositive => {
                    if self.precision_positive < baseline.precision_positive * drift_factor {
                        res.insert(
                            C::PrecisionPositive,
                            baseline.precision_positive - self.precision_positive,
                        );
                    }
                }
                C::PrecisionNegative => {
                    if self.precision_negative < baseline.precision_negative * drift_factor {
                        res.insert(
                            C::PrecisionNegative,
                            baseline.precision_negative - self.precision_negative,
                        );
                    }
                }
                C::RecallPositive => {
                    if self.recall_positive < baseline.recall_positive * drift_factor {
                        res.insert(
                            C::RecallPositive,
                            baseline.recall_positive - self.recall_positive,
                        );
                    }
                }
                C::RecallNegative => {
                    if self.recall_negative < baseline.recall_negative * drift_factor {
                        res.insert(
                            C::RecallNegative,
                            baseline.recall_negative - self.recall_negative,
                        );
                    }
                }
                C::Accuracy => {
                    if self.accuracy < baseline.accuracy * drift_factor {
                        res.insert(C::Accuracy, baseline.accuracy - self.accuracy);
                    }
                }
                C::F1Score => {
                    if self.f1_score < baseline.f1_score * drift_factor {
                        res.insert(C::F1Score, baseline.f1_score - self.f1_score);
                    }
                }
                _ => continue,
            }
        }

        res
    }
}

pub struct LogisticRegressionAnalysisResult {
    balanced_accuracy: f32,
    precision_positive: f32,
    precision_negative: f32,
    recall_positive: f32,
    recall_negative: f32,
    accuracy: f32,
    f1_score: f32,
    log_loss: f32,
}

impl LogisticRegressionAnalysisResult {
    pub fn generate_report(&self) -> LogisticRegressionReport {
        use ClassificationEvaluationMetric as M;
        let mut map: HashMap<M, f32> = HashMap::with_capacity(8);
        map.insert(M::BalancedAccuracy, self.balanced_accuracy);
        map.insert(M::PrecisionPositive, self.precision_positive);
        map.insert(M::PrecisionNegative, self.precision_negative);
        map.insert(M::RecallPositive, self.recall_positive);
        map.insert(M::RecallNegative, self.recall_negative);
        map.insert(M::Accuracy, self.accuracy);
        map.insert(M::F1Score, self.f1_score);
        map.insert(M::LogLoss, self.log_loss);
        map
    }
}

impl TryFrom<HashMap<String, f32>> for LogisticRegressionAnalysisResult {
    type Error = String;
    fn try_from(map: HashMap<String, f32>) -> Result<Self, Self::Error> {
        let Some(balanced_accuracy) = map.get("BalancedAccuracy") else {
            return Err("Invalid regression report".into());
        };
        let Some(precision_positive) = map.get("PrecisionPositive") else {
            return Err("Invalid regression report".into());
        };
        let Some(precision_negative) = map.get("PrecisionNegative") else {
            return Err("Invalid regression report".into());
        };
        let Some(recall_positive) = map.get("RecallPositive") else {
            return Err("Invalid regression report".into());
        };
        let Some(recall_negative) = map.get("RecallNegative") else {
            return Err("Invalid regression report".into());
        };
        let Some(accuracy) = map.get("Accuracy") else {
            return Err("Invalid regression report".into());
        };
        let Some(f1_score) = map.get("F1Score") else {
            return Err("Invalid regression report".into());
        };
        let Some(log_loss) = map.get("LogLoss") else {
            return Err("Invalid regression report".into());
        };

        Ok(LogisticRegressionAnalysisResult {
            balanced_accuracy: *balanced_accuracy,
            precision_positive: *precision_positive,
            precision_negative: *precision_negative,
            recall_positive: *recall_positive,
            recall_negative: *recall_negative,
            accuracy: *accuracy,
            f1_score: *f1_score,
            log_loss: *log_loss,
        })
    }
}

impl LogisticRegressionAnalysisResult {
    pub fn compare_to_baseline(
        &self,
        metrics: &[ClassificationEvaluationMetric],
        baseline: &Self,
        drift_threshold: f32,
    ) -> HashMap<ClassificationEvaluationMetric, f32> {
        // all the metrics here are used, at this point we have
        // everything correct, thus no Result<T,E>
        use ClassificationEvaluationMetric as C;
        let mut res: HashMap<C, f32> = HashMap::with_capacity(7);
        let drift_factor = 1_f32 - drift_threshold;
        for m in metrics.iter() {
            match *m {
                C::BalancedAccuracy => {
                    if self.balanced_accuracy < baseline.balanced_accuracy * drift_factor {
                        res.insert(
                            C::BalancedAccuracy,
                            baseline.balanced_accuracy - self.balanced_accuracy,
                        );
                    }
                }
                C::PrecisionPositive => {
                    if self.precision_positive < baseline.precision_positive * drift_factor {
                        res.insert(
                            C::PrecisionPositive,
                            baseline.precision_positive - self.precision_positive,
                        );
                    }
                }
                C::PrecisionNegative => {
                    if self.precision_negative < baseline.precision_negative * drift_factor {
                        res.insert(
                            C::PrecisionNegative,
                            baseline.precision_negative - self.precision_negative,
                        );
                    }
                }
                C::RecallPositive => {
                    if self.recall_positive < baseline.recall_positive * drift_factor {
                        res.insert(
                            C::RecallPositive,
                            baseline.recall_positive - self.recall_positive,
                        );
                    }
                }
                C::RecallNegative => {
                    if self.recall_negative < baseline.recall_negative * drift_factor {
                        res.insert(
                            C::RecallNegative,
                            baseline.recall_negative - self.recall_negative,
                        );
                    }
                }
                C::Accuracy => {
                    if self.accuracy < baseline.accuracy * drift_factor {
                        res.insert(C::Accuracy, baseline.accuracy - self.accuracy);
                    }
                }
                C::F1Score => {
                    if self.f1_score < baseline.f1_score * drift_factor {
                        res.insert(C::F1Score, baseline.f1_score - self.f1_score);
                    }
                }
                C::LogLoss => {
                    if self.log_loss < baseline.log_loss * drift_factor {
                        res.insert(C::F1Score, baseline.log_loss - self.log_loss);
                    }
                }
            }
        }
        res
    }
}

pub struct ClassificationPerf {
    len: f32,
    mean_f: f32,
    y_pred: Vec<f32>,
    y_true: Vec<f32>,
}

impl Into<BinaryClassificationAnalysisResult> for ClassificationPerf {
    fn into(self) -> BinaryClassificationAnalysisResult {
        let recall_positive =
            GeneralClassificationMetrics::recall_positive(&self.y_pred, &self.y_true);
        let precision_positive =
            GeneralClassificationMetrics::precision_positive(&self.y_pred, &self.y_true);
        let recall_negative =
            GeneralClassificationMetrics::recall_negative(&self.y_pred, &self.y_true, self.len);
        BinaryClassificationAnalysisResult {
            balanced_accuracy: GeneralClassificationMetrics::balanced_accuracy(
                recall_positive,
                recall_negative,
            ),
            precision_positive,
            precision_negative: GeneralClassificationMetrics::precision_negative(
                &self.y_pred,
                &self.y_true,
                self.len,
            ),
            recall_positive,
            recall_negative,
            accuracy: GeneralClassificationMetrics::accuracy(
                &self.y_pred,
                &self.y_true,
                self.mean_f,
            ),
            f1_score: GeneralClassificationMetrics::f1_score(recall_positive, precision_positive),
        }
    }
}

impl ClassificationPerf {
    pub fn new(
        y_true: Vec<f32>,
        y_pred: Vec<f32>,
    ) -> Result<ClassificationPerf, ModelPerformanceError> {
        if y_true.len() != y_pred.len() {
            return Err(ModelPerformanceError::DataVectorLengthMismatch);
        }
        if y_pred.len() == 0 {
            return Err(ModelPerformanceError::EmptyDataVector);
        }
        let len: f32 = y_pred.len() as f32;
        let mean_f: f32 = 1_f32 / len;
        Ok(ClassificationPerf {
            y_true,
            y_pred,
            mean_f,
            len,
        })
    }
}

//TODO: update to only hold 1 vec for proba
//compute label inline
struct LogisticRegressionPerf {
    y_true: Vec<f32>,
    y_pred: Vec<f32>,
    y_proba: Vec<f32>,
    mean_f: f32,
    len: f32,
}

impl LogisticRegressionPerf {
    pub fn new(
        y_true: Vec<f32>,
        y_proba: Vec<f32>,
        threshold: f32,
    ) -> Result<LogisticRegressionPerf, ModelPerformanceError> {
        if y_true.len() != y_proba.len() {
            return Err(ModelPerformanceError::DataVectorLengthMismatch);
        }

        if y_proba.len() == 0 {
            return Err(ModelPerformanceError::EmptyDataVector);
        }
        let y_pred = y_proba
            .clone()
            .iter()
            .map(|x| if *x >= threshold { 1_f32 } else { 0_f32 })
            .collect::<Vec<f32>>();

        let len: f32 = y_true.len() as f32;

        Ok(LogisticRegressionPerf {
            y_true,
            y_pred,
            y_proba,
            mean_f: 1_f32 / len,
            len,
        })
    }
}

impl Into<LogisticRegressionAnalysisResult> for LogisticRegressionPerf {
    fn into(self) -> LogisticRegressionAnalysisResult {
        let recall_positive =
            GeneralClassificationMetrics::recall_positive(&self.y_pred, &self.y_true);
        let precision_positive =
            GeneralClassificationMetrics::precision_positive(&self.y_pred, &self.y_true);
        let recall_negative =
            GeneralClassificationMetrics::recall_negative(&self.y_pred, &self.y_true, self.len);
        LogisticRegressionAnalysisResult {
            balanced_accuracy: GeneralClassificationMetrics::balanced_accuracy(
                recall_positive,
                recall_negative,
            ),
            precision_positive,
            precision_negative: GeneralClassificationMetrics::precision_negative(
                &self.y_pred,
                &self.y_true,
                self.len,
            ),
            recall_positive,
            recall_negative,
            accuracy: GeneralClassificationMetrics::accuracy(
                &self.y_pred,
                &self.y_true,
                self.mean_f,
            ),
            f1_score: GeneralClassificationMetrics::f1_score(recall_positive, precision_positive),
            log_loss: GeneralClassificationMetrics::log_loss_score(
                &self.y_proba,
                &self.y_true,
                self.mean_f,
            ),
        }
    }
}

pub struct LinearRegressionAnalysisResult {
    rmse: f32,
    mse: f32,
    mae: f32,
    r_squared: f32,
    max_error: f32,
    msle: f32,
    rmsle: f32,
    mape: f32,
}

impl TryFrom<HashMap<String, f32>> for LinearRegressionAnalysisResult {
    type Error = String;
    fn try_from(map: HashMap<String, f32>) -> Result<Self, Self::Error> {
        let Some(rmse) = map.get("RootMeanSquaredError") else {
            return Err("Invalid regression report".into());
        };
        let Some(mse) = map.get("MeanSquaredError") else {
            return Err("Invalid regression report".into());
        };
        let Some(mae) = map.get("MeanAbsoluteError") else {
            return Err("Invalid regression report".into());
        };
        let Some(r_squared) = map.get("RSquared") else {
            return Err("Invalid regression report".into());
        };
        let Some(max_error) = map.get("MaxError") else {
            return Err("Invalid regression report".into());
        };
        let Some(msle) = map.get("MeanSquaredLogError") else {
            return Err("Invalid regression report".into());
        };
        let Some(rmsle) = map.get("RootMeanSquaredLogError") else {
            return Err("Invalid regression report".into());
        };
        let Some(mape) = map.get("MeanAbsolutePercentageError") else {
            return Err("Invalid regression report".into());
        };
        Ok(LinearRegressionAnalysisResult {
            rmse: *rmse,
            mse: *mse,
            mae: *mae,
            r_squared: *r_squared,
            max_error: *max_error,
            msle: *msle,
            rmsle: *rmsle,
            mape: *mape,
        })
    }
}

impl LinearRegressionAnalysisResult {
    pub fn generate_report(&self) -> LinearRegressionReport {
        use LinearRegressionEvaluationMetric as L;
        let mut map: HashMap<L, f32> = HashMap::with_capacity(8);
        map.insert(L::RootMeanSquaredError, self.rmse);
        map.insert(L::MeanSquaredError, self.mse);
        map.insert(L::MeanAbsoluteError, self.mae);
        map.insert(L::RSquared, self.r_squared);
        map.insert(L::MaxError, self.max_error);
        map.insert(L::MeanSquaredLogError, self.msle);
        map.insert(L::RootMeanSquaredLogError, self.rmsle);
        map.insert(L::MeanAbsolutePercentageError, self.mape);
        map
    }
}

impl LinearRegressionAnalysisResult {
    pub fn compare_to_baseline(
        &self,
        metrics: &[LinearRegressionEvaluationMetric],
        baseline: &LinearRegressionAnalysisResult,
        drift_threshold: f32,
    ) -> HashMap<LinearRegressionEvaluationMetric, f32> {
        use LinearRegressionEvaluationMetric as L;
        let mut res: HashMap<L, f32> = HashMap::with_capacity(8);
        for m in metrics.iter() {
            match *m {
                L::RootMeanSquaredError => {
                    if self.rmse > baseline.rmse * (1_f32 + drift_threshold) {
                        res.insert(L::RootMeanSquaredError, self.rmse - baseline.rmse);
                    }
                }
                L::MeanSquaredError => {
                    if self.mse > baseline.mse * (1_f32 + drift_threshold) {
                        res.insert(L::MeanSquaredError, self.mse - baseline.mse);
                    }
                }
                L::MeanAbsoluteError => {
                    if self.mae > baseline.mae * (1_f32 + drift_threshold) {
                        res.insert(L::MeanAbsoluteError, self.mae - baseline.mae);
                    }
                }
                L::RSquared => {
                    if self.r_squared > baseline.r_squared * (1_f32 + drift_threshold) {
                        res.insert(L::RSquared, self.r_squared - baseline.r_squared);
                    }
                }
                L::MaxError => {
                    if self.max_error > baseline.max_error * (1_f32 + drift_threshold) {
                        res.insert(L::MaxError, self.max_error - baseline.max_error);
                    }
                }
                L::MeanSquaredLogError => {
                    if self.msle > baseline.msle * (1_f32 + drift_threshold) {
                        res.insert(L::MeanSquaredLogError, self.msle - baseline.msle);
                    }
                }
                L::RootMeanSquaredLogError => {
                    if self.rmsle > baseline.rmsle * (1_f32 + drift_threshold) {
                        res.insert(L::RootMeanSquaredLogError, self.rmsle - baseline.rmsle);
                    }
                }
                L::MeanAbsolutePercentageError => {
                    if self.mape > baseline.mape * (1_f32 + drift_threshold) {
                        res.insert(L::MeanAbsolutePercentageError, self.mape - baseline.mape);
                    }
                }
            }
        }
        res
    }
}

pub struct LinearRegressionPerf {
    y_pred: Vec<f32>,
    y_true: Vec<f32>,
    mean_f: f32,
}

impl Into<LinearRegressionAnalysisResult> for LinearRegressionPerf {
    fn into(self) -> LinearRegressionAnalysisResult {
        LinearRegressionAnalysisResult {
            rmse: self.root_mean_squared_error(),
            mse: self.mean_squared_error(),
            mae: self.mean_absolute_error(),
            r_squared: self.r_squared(),
            max_error: self.max_error(),
            msle: self.mean_squared_log_error(),
            rmsle: self.root_mean_squared_log_error(),
            mape: self.mean_absolute_percentage_error(),
        }
    }
}

impl LinearRegressionPerf {
    pub fn new(
        y_true: Vec<f32>,
        y_pred: Vec<f32>,
    ) -> Result<LinearRegressionPerf, ModelPerformanceError> {
        if y_true.len() != y_pred.len() {
            return Err(ModelPerformanceError::DataVectorLengthMismatch);
        }
        if y_true.len() == 0 {
            return Err(ModelPerformanceError::EmptyDataVector);
        }
        let mean_f: f32 = 1_f32 / y_pred.len() as f32;
        Ok(LinearRegressionPerf {
            y_true,
            y_pred,
            mean_f,
        })
    }

    fn root_mean_squared_error(&self) -> f32 {
        let mut errors = 0_f32;
        for (t, p) in zip!(self.y_true, &self.y_pred) {
            errors += (t - p).powi(2);
        }
        (errors * self.mean_f).powf(0.5_f32)
    }

    fn mean_squared_error(&self) -> f32 {
        let mut errors = 0_f32;
        for (t, p) in zip!(&self.y_true, &self.y_pred) {
            errors += (t - p).powi(2);
        }
        errors * self.mean_f
    }

    fn mean_absolute_error(&self) -> f32 {
        let mut errors = 0_f32;
        for (t, p) in zip!(&self.y_true, &self.y_pred) {
            errors += (t - p).abs();
        }
        errors * self.mean_f
    }

    fn r_squared(&self) -> f32 {
        let y_mean: f32 = self.y_true.iter().sum::<f32>() * self.mean_f;
        let mut ss_regression: f32 = 0_f32;
        for (t, p) in zip!(self.y_true, &self.y_pred) {
            ss_regression += (t - p).powi(2);
        }
        let ss_total: f32 = self
            .y_true
            .iter()
            .map(|y| (y - y_mean).powi(2))
            .sum::<f32>();
        ss_regression / ss_total
    }

    fn max_error(&self) -> f32 {
        let mut res = 0_f32;
        for (t, p) in zip!(&self.y_true, &self.y_pred) {
            res = f32::max(t - p, res);
        }
        res
    }

    fn mean_squared_log_error(&self) -> f32 {
        let mut sum = 0_f32;
        for (t, p) in zip!(&self.y_true, &self.y_pred) {
            sum += (1_f32 + t).log10() - (1_f32 + p).log10();
        }
        sum.powi(2) / self.mean_f
    }

    fn root_mean_squared_log_error(&self) -> f32 {
        let mut sum = 0_f32;
        for (t, p) in zip!(&self.y_true, &self.y_pred) {
            sum += (1_f32 + t).log10() - (1_f32 + p).log10();
        }
        sum.powi(2).sqrt() / self.mean_f
    }

    fn mean_absolute_percentage_error(&self) -> f32 {
        let mut sum = 0_f32;
        for (t, p) in zip!(&self.y_true, &self.y_pred) {
            sum += (t - p).abs() / t;
        }
        sum * self.mean_f * 100_f32
    }
}
