use crate::{
    data_handler::{ApplyThreshold, ConfusionMatrix},
    errors::ModelPerformanceError,
    metrics::{ClassificationEvaluationMetric, LinearRegressionEvaluationMetric},
    reporting::{
        BinaryClassificationAnalysisReport, BinaryClassificationRuntimeReport, DriftReport,
        LinearRegressionAnalysisReport, LinearRegressionRuntimeReport,
        LogisticRegressionAnalysisReport, LogisticRegressionRuntimeReport,
    },
    zip_iters,
};
use std::collections::HashMap;
pub mod statistics;

#[cfg(feature = "python")]
pub(crate) mod py_api {
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
    let baseline = match BinaryClassificationAnalysisResult::try_from(baseline) {
        Ok(v) => v,
        Err(e) => return Err(format!("Invalid baseline report: {}", e)),
    };
    let latest = match BinaryClassificationAnalysisResult::try_from(latest) {
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

pub fn regression_performance_runtime(
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
    let report = BinaryClassificationAnalysisResult::new::<T>(y_true, y_pred, positive_label)?;
    Ok(report.generate_report())
}

pub fn model_perf_regression<T>(
    y_true: &[T],
    y_pred: &[T],
) -> Result<LinearRegressionAnalysisReport, ModelPerformanceError>
where
    T: Into<f64> + Copy,
{
    let report: LinearRegressionAnalysisResult =
        LinearRegressionAnalysisResult::new(y_true, y_pred)?;
    Ok(report.generate_report())
}

pub fn model_perf_logistic_regression(
    y_true: &[f32],
    y_proba: &[f32],
    threshold: f32,
) -> Result<LogisticRegressionAnalysisReport, ModelPerformanceError> {
    let lr_report: LogisticRegressionAnalysisResult =
        LogisticRegressionAnalysisResult::new(y_true, y_proba, threshold)?;
    Ok(lr_report.generate_report())
}

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
    pub fn new<T>(
        y_true: &[T],
        y_pred: &[T],
        label: T,
    ) -> Result<BinaryClassificationAnalysisResult, ModelPerformanceError>
    where
        T: PartialOrd,
    {
        use statistics::classification_metrics as metrics;
        let mut c_matrix = ConfusionMatrix::default();

        for (t, p) in zip_iters!(y_true, y_pred) {
            let is_positive = *p == label;
            let is_true = *p == *t;
            c_matrix.true_p += (is_true && is_positive) as i32 as f32;
            c_matrix.false_p += (!is_true && is_positive) as i32 as f32;
            c_matrix.true_n += (is_true && !is_positive) as i32 as f32;
            c_matrix.false_n += (!is_true && !is_positive) as i32 as f32;
        }

        let accuracy = metrics::accuracy(y_true, y_pred)?;
        let balanced_accuracy = metrics::balanced_accuracy(&c_matrix);
        let precision_positive = metrics::precision_positive(&c_matrix);
        let precision_negative = metrics::precision_negative(&c_matrix);
        let recall_positive = metrics::recall_positive(&c_matrix);
        let recall_negative = metrics::recall_negative(&c_matrix);
        let f1_score = metrics::f1_score(&c_matrix);

        Ok(BinaryClassificationAnalysisResult {
            balanced_accuracy,
            precision_positive,
            precision_negative,
            recall_positive,
            recall_negative,
            accuracy,
            f1_score,
        })
    }
}

impl BinaryClassificationAnalysisResult {
    pub fn generate_report(&self) -> BinaryClassificationAnalysisReport {
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

impl TryFrom<&BinaryClassificationAnalysisReport> for BinaryClassificationAnalysisResult {
    type Error = ModelPerformanceError;
    fn try_from(payload: &BinaryClassificationAnalysisReport) -> Result<Self, Self::Error> {
        use ClassificationEvaluationMetric as C;
        let value_fetcher = |p: &BinaryClassificationAnalysisReport, key: C| {
            let Some(v) = p.get(&key) else {
                return Err(ModelPerformanceError::InvalidAnalysisReport);
            };

            Ok(*v)
        };
        Ok(BinaryClassificationAnalysisResult {
            balanced_accuracy: value_fetcher(&payload, C::BalancedAccuracy)?,
            precision_positive: value_fetcher(&payload, C::PrecisionPositive)?,
            precision_negative: value_fetcher(&payload, C::PrecisionNegative)?,
            recall_positive: value_fetcher(&payload, C::RecallPositive)?,
            recall_negative: value_fetcher(&payload, C::RecallNegative)?,
            accuracy: value_fetcher(&payload, C::Accuracy)?,
            f1_score: value_fetcher(&payload, C::F1Score)?,
        })
    }
}

impl TryFrom<HashMap<String, f32>> for BinaryClassificationAnalysisResult {
    type Error = ModelPerformanceError;
    fn try_from(mut payload: HashMap<String, f32>) -> Result<Self, Self::Error> {
        let value_fetcher = |p: &mut HashMap<String, f32>, key: &str| {
            let Some(v) = p.remove(key) else {
                return Err(ModelPerformanceError::InvalidAnalysisReport);
            };

            Ok(v)
        };

        Ok(BinaryClassificationAnalysisResult {
            balanced_accuracy: value_fetcher(&mut payload, "BalancedAccuracy")?,
            precision_positive: value_fetcher(&mut payload, "PrecisionPositive")?,
            precision_negative: value_fetcher(&mut payload, "PrecisionNegative")?,
            recall_positive: value_fetcher(&mut payload, "RecallPositive")?,
            recall_negative: value_fetcher(&mut payload, "RecallNegative")?,
            accuracy: value_fetcher(&mut payload, "Accuracy")?,
            f1_score: value_fetcher(&mut payload, "F1Score")?,
        })
    }
}

impl BinaryClassificationAnalysisResult {
    pub fn compare_to_baseline(
        &self,
        metrics: &[ClassificationEvaluationMetric],
        baseline: &Self,
        drift_threshold: f32,
    ) -> BinaryClassificationRuntimeReport {
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

// assume that positive label is 1
impl LogisticRegressionAnalysisResult {
    fn new(
        y_true: &[f32],
        y_pred: &[f32],
        threshold: f32,
    ) -> Result<LogisticRegressionAnalysisResult, ModelPerformanceError> {
        if y_true.len() != y_pred.len() {
            return Err(ModelPerformanceError::DataVectorLengthMismatch);
        }

        use statistics::classification_metrics as metrics;
        let mut c_matrix = ConfusionMatrix::default();

        for (t, p) in zip_iters!(y_true, y_pred) {
            let label = p.apply_threshold(threshold);
            let is_positive = label == 1_f32;
            let is_true = label == *t;

            c_matrix.true_p += (is_true && is_positive) as i32 as f32;
            c_matrix.false_p += (!is_true && is_positive) as i32 as f32;
            c_matrix.true_n += (is_true && !is_positive) as i32 as f32;
            c_matrix.false_n += (!is_true && !is_positive) as i32 as f32;
        }

        let accuracy = metrics::accuracy(y_true, y_pred)?;
        let balanced_accuracy = metrics::balanced_accuracy(&c_matrix);
        let precision_positive = metrics::precision_positive(&c_matrix);
        let precision_negative = metrics::precision_negative(&c_matrix);
        let recall_positive = metrics::recall_positive(&c_matrix);
        let recall_negative = metrics::recall_negative(&c_matrix);
        let f1_score = metrics::f1_score(&c_matrix);
        let log_loss = metrics::log_loss_score(y_true, y_pred)?;

        Ok(LogisticRegressionAnalysisResult {
            balanced_accuracy,
            precision_positive,
            precision_negative,
            recall_positive,
            recall_negative,
            accuracy,
            f1_score,
            log_loss,
        })
    }
}

impl LogisticRegressionAnalysisResult {
    pub fn generate_report(&self) -> LogisticRegressionAnalysisReport {
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

impl TryFrom<&LogisticRegressionAnalysisReport> for LogisticRegressionAnalysisResult {
    type Error = ModelPerformanceError;
    fn try_from(payload: &LogisticRegressionAnalysisReport) -> Result<Self, Self::Error> {
        use ClassificationEvaluationMetric as L;
        let value_fetcher = |p: &LogisticRegressionAnalysisReport, key: L| {
            let Some(v) = p.get(&key) else {
                return Err(ModelPerformanceError::InvalidAnalysisReport);
            };
            Ok(*v)
        };
        Ok(LogisticRegressionAnalysisResult {
            balanced_accuracy: value_fetcher(&payload, L::BalancedAccuracy)?,
            precision_positive: value_fetcher(&payload, L::PrecisionPositive)?,
            precision_negative: value_fetcher(&payload, L::PrecisionNegative)?,
            recall_positive: value_fetcher(&payload, L::RecallPositive)?,
            recall_negative: value_fetcher(&payload, L::RecallNegative)?,
            accuracy: value_fetcher(&payload, L::Accuracy)?,
            f1_score: value_fetcher(&payload, L::F1Score)?,
            log_loss: value_fetcher(&payload, L::LogLoss)?,
        })
    }
}

impl TryFrom<HashMap<String, f32>> for LogisticRegressionAnalysisResult {
    type Error = ModelPerformanceError;
    fn try_from(mut payload: HashMap<String, f32>) -> Result<Self, Self::Error> {
        let value_fetcher = |p: &mut HashMap<String, f32>, key: &str| {
            let Some(v) = p.remove(key) else {
                return Err(ModelPerformanceError::InvalidAnalysisReport);
            };
            Ok(v)
        };
        Ok(LogisticRegressionAnalysisResult {
            balanced_accuracy: value_fetcher(&mut payload, "BalancedAccuracy")?,
            precision_positive: value_fetcher(&mut payload, "PrecisionPositive")?,
            precision_negative: value_fetcher(&mut payload, "PrecisionNegative")?,
            recall_positive: value_fetcher(&mut payload, "RecallPositive")?,
            recall_negative: value_fetcher(&mut payload, "RecallNegative")?,
            accuracy: value_fetcher(&mut payload, "Accuracy")?,
            f1_score: value_fetcher(&mut payload, "F1Score")?,
            log_loss: value_fetcher(&mut payload, "LogLoss")?,
        })
    }
}

impl LogisticRegressionAnalysisResult {
    pub fn compare_to_baseline(
        &self,
        metrics: &[ClassificationEvaluationMetric],
        baseline: &Self,
        drift_threshold: f32,
    ) -> LogisticRegressionRuntimeReport {
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

impl LinearRegressionAnalysisResult {
    pub fn new<T>(
        y_true: &[T],
        y_pred: &[T],
    ) -> Result<LinearRegressionAnalysisResult, ModelPerformanceError>
    where
        T: Into<f64> + Copy,
    {
        if y_true.len() != y_pred.len() {
            return Err(ModelPerformanceError::DataVectorLengthMismatch);
        }
        if y_true.len() == 0 {
            return Err(ModelPerformanceError::EmptyDataVector);
        }

        let n = y_true.len() as f64;

        let mut squared_error_sum = 0_f64;
        let mut abs_error_sum = 0_f64;
        let mut max_error = 0_f64;
        let mut log_error_sum = 0_f64;
        let mut abs_percent_error_sum = 0_f64;
        let mut y_true_sum = 0_f64;

        for (t_ref, p_ref) in zip_iters!(y_true, y_pred) {
            let t: f64 = (*t_ref).into();
            let p: f64 = (*p_ref).into();

            y_true_sum += t;
            squared_error_sum += (t - p).powi(2);
            abs_error_sum += (t - p).abs();
            max_error = max_error.max((t - p).abs());
            log_error_sum += (1_f64 + t).log10() - (1_f64 + p).log10();
            abs_percent_error_sum += (t - p).abs() / t;
        }

        let mut ss_total = 0_f64;
        let y_true_mean = y_true_sum / n;
        for t_ref in y_true.iter() {
            let t: f64 = (*t_ref).into();
            ss_total += (t - y_true_mean).powi(2);
        }

        let mse = squared_error_sum / n;
        let msle = log_error_sum / n;

        Ok(LinearRegressionAnalysisResult {
            r_squared: (1_f64 - (squared_error_sum / ss_total)) as f32,
            rmse: (mse).powf(0_f64) as f32,
            mse: mse as f32,
            mae: (abs_error_sum / n) as f32,
            max_error: max_error as f32,
            msle: msle as f32,
            rmsle: (msle.powf(0.5_f64)) as f32,
            mape: (abs_percent_error_sum / n) as f32,
        })
    }
}

impl TryFrom<&LinearRegressionAnalysisReport> for LinearRegressionAnalysisResult {
    type Error = ModelPerformanceError;
    fn try_from(payload: &LinearRegressionAnalysisReport) -> Result<Self, Self::Error> {
        use LinearRegressionEvaluationMetric as L;
        let value_fetcher = |p: &LinearRegressionAnalysisReport, key: L| {
            let Some(v) = p.get(&key) else {
                return Err(ModelPerformanceError::InvalidAnalysisReport);
            };
            Ok(*v)
        };

        Ok(LinearRegressionAnalysisResult {
            rmse: value_fetcher(&payload, L::RootMeanSquaredError)?,
            mse: value_fetcher(&payload, L::MeanSquaredError)?,
            mae: value_fetcher(&payload, L::MeanAbsoluteError)?,
            r_squared: value_fetcher(&payload, L::RSquared)?,
            max_error: value_fetcher(&payload, L::MaxError)?,
            msle: value_fetcher(&payload, L::MeanSquaredLogError)?,
            rmsle: value_fetcher(&payload, L::RootMeanSquaredLogError)?,
            mape: value_fetcher(&payload, L::MeanAbsolutePercentageError)?,
        })
    }
}

impl TryFrom<HashMap<String, f32>> for LinearRegressionAnalysisResult {
    type Error = ModelPerformanceError;
    fn try_from(mut payload: HashMap<String, f32>) -> Result<Self, Self::Error> {
        let value_fetcher = |p: &mut HashMap<String, f32>, key: &str| {
            let Some(v) = p.remove(key) else {
                return Err(ModelPerformanceError::InvalidAnalysisReport);
            };
            Ok(v)
        };

        Ok(LinearRegressionAnalysisResult {
            rmse: value_fetcher(&mut payload, "RootMeanSquaredError")?,
            mse: value_fetcher(&mut payload, "MeanSquaredError")?,
            mae: value_fetcher(&mut payload, "MeanAbsoluteError")?,
            r_squared: value_fetcher(&mut payload, "RSquared")?,
            max_error: value_fetcher(&mut payload, "MaxError")?,
            msle: value_fetcher(&mut payload, "MeanSquaredLogError")?,
            rmsle: value_fetcher(&mut payload, "RootMeanSquaredLogError")?,
            mape: value_fetcher(&mut payload, "MeanAbsolutePercentageError")?,
        })
    }
}

impl LinearRegressionAnalysisResult {
    pub fn generate_report(&self) -> LinearRegressionAnalysisReport {
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
    ) -> LinearRegressionRuntimeReport {
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
