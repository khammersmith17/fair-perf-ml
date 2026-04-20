use super::EQUALITY_ERROR_ALLOWANCE;
use crate::{
    data_handler::{ApplyThreshold, ConfusionMatrix},
    errors::ModelPerformanceError,
    metrics::{ClassificationEvaluationMetric, LinearRegressionEvaluationMetric},
    model_perf::streaming::LinearRegressionErrorBuckets,
    reporting::{
        BinaryClassificationAnalysisReport, BinaryClassificationRuntimeReport,
        LinearRegressionAnalysisReport, LinearRegressionRuntimeReport,
        LogisticRegressionAnalysisReport, LogisticRegressionRuntimeReport,
    },
};
use std::collections::HashMap;
#[derive(Debug, Clone)]
pub struct BinaryClassificationRuntime {
    pub(crate) balanced_accuracy: f32,
    pub(crate) precision_positive: f32,
    pub(crate) precision_negative: f32,
    pub(crate) recall_positive: f32,
    pub(crate) recall_negative: f32,
    pub(crate) accuracy: f32,
    pub(crate) f1_score: f32,
}

impl PartialEq for BinaryClassificationRuntime {
    fn eq(&self, other: &Self) -> bool {
        if (self.balanced_accuracy - other.balanced_accuracy).abs() > EQUALITY_ERROR_ALLOWANCE {
            return false;
        }
        if (self.precision_positive - other.precision_positive).abs() > EQUALITY_ERROR_ALLOWANCE {
            return false;
        }
        if (self.precision_negative - other.precision_negative).abs() > EQUALITY_ERROR_ALLOWANCE {
            return false;
        }
        if (self.recall_positive - other.recall_positive).abs() > EQUALITY_ERROR_ALLOWANCE {
            return false;
        }
        if (self.recall_negative - other.recall_negative).abs() > EQUALITY_ERROR_ALLOWANCE {
            return false;
        }
        if (self.accuracy - other.accuracy).abs() > EQUALITY_ERROR_ALLOWANCE {
            return false;
        }
        if (self.f1_score - other.f1_score).abs() > EQUALITY_ERROR_ALLOWANCE {
            return false;
        }
        true
    }
}

impl BinaryClassificationRuntime {
    pub fn new<T>(
        y_true: &[T],
        y_pred: &[T],
        label: &T,
    ) -> Result<BinaryClassificationRuntime, ModelPerformanceError>
    where
        T: PartialOrd,
    {
        use crate::model_perf::statistics::classification_metrics_from_parts as metrics;
        let mut c_matrix = ConfusionMatrix::default();

        c_matrix.push_dataset(y_true, y_pred, |v: &T| v.eq(label))?;

        Ok(BinaryClassificationRuntime {
            balanced_accuracy: metrics::balanced_accuracy(&c_matrix),
            precision_positive: metrics::precision_positive(&c_matrix),
            precision_negative: metrics::precision_negative(&c_matrix),
            recall_positive: metrics::recall_positive(&c_matrix),
            recall_negative: metrics::recall_negative(&c_matrix),
            accuracy: metrics::accuracy(&c_matrix),
            f1_score: metrics::f1_score(&c_matrix),
        })
    }

    // Utlity to easliy compute the current model performance runtime state from the bucketing
    // style containers used in the stream variants
    pub(crate) fn runtime_from_parts(c_matrix: &ConfusionMatrix) -> BinaryClassificationRuntime {
        use crate::model_perf::statistics::classification_metrics_from_parts as metrics;

        BinaryClassificationRuntime {
            balanced_accuracy: metrics::balanced_accuracy(c_matrix),
            precision_positive: metrics::precision_positive(c_matrix),
            precision_negative: metrics::precision_negative(c_matrix),
            recall_positive: metrics::recall_positive(c_matrix),
            recall_negative: metrics::recall_negative(c_matrix),
            accuracy: c_matrix.accuracy(),
            f1_score: metrics::f1_score(c_matrix),
        }
    }

    pub fn compare_to_baseline(
        &self,
        metrics: &[ClassificationEvaluationMetric],
        baseline: &Self,
        drift_threshold: f32,
    ) -> BinaryClassificationRuntimeReport {
        use ClassificationEvaluationMetric as C;
        let mut res: HashMap<C, f32> = HashMap::with_capacity(7);
        let drift_factor = 1_f32 + drift_threshold;
        // log loss should not be present here
        // so when log loss comes up, we return Err
        for m in metrics.iter() {
            match *m {
                C::BalancedAccuracy => {
                    if (self.balanced_accuracy * drift_factor) < baseline.balanced_accuracy {
                        res.insert(
                            C::BalancedAccuracy,
                            baseline.balanced_accuracy - self.balanced_accuracy,
                        );
                    }
                }
                C::PrecisionPositive => {
                    if (self.precision_positive * drift_factor) < baseline.precision_positive {
                        res.insert(
                            C::PrecisionPositive,
                            baseline.precision_positive - self.precision_positive,
                        );
                    }
                }
                C::PrecisionNegative => {
                    if (self.precision_negative * drift_factor) < baseline.precision_negative {
                        res.insert(
                            C::PrecisionNegative,
                            baseline.precision_negative - self.precision_negative,
                        );
                    }
                }
                C::RecallPositive => {
                    if (self.recall_positive * drift_factor) < baseline.recall_positive {
                        res.insert(
                            C::RecallPositive,
                            baseline.recall_positive - self.recall_positive,
                        );
                    }
                }
                C::RecallNegative => {
                    if (self.recall_negative * drift_factor) < baseline.recall_negative {
                        res.insert(
                            C::RecallNegative,
                            baseline.recall_negative - self.recall_negative,
                        );
                    }
                }
                C::Accuracy => {
                    if (self.accuracy * drift_factor) < baseline.accuracy {
                        res.insert(C::Accuracy, baseline.accuracy - self.accuracy);
                    }
                }
                C::F1Score => {
                    if (self.f1_score * drift_factor) < baseline.f1_score {
                        res.insert(C::F1Score, baseline.f1_score - self.f1_score);
                    }
                }
                _ => continue,
            }
        }

        res
    }

    pub(crate) fn runtime_drift_report(
        &self,
        baseline: &BinaryClassificationRuntime,
    ) -> BinaryClassificationRuntimeReport {
        use crate::metrics::ClassificationEvaluationMetric as C;
        let mut report = BinaryClassificationRuntimeReport::with_capacity(7);
        report.insert(
            C::BalancedAccuracy,
            baseline.balanced_accuracy - self.balanced_accuracy,
        );
        report.insert(
            C::PrecisionPositive,
            baseline.precision_positive - self.precision_positive,
        );
        report.insert(
            C::PrecisionNegative,
            baseline.precision_negative - self.precision_negative,
        );

        report.insert(
            C::RecallPositive,
            baseline.recall_positive - self.recall_positive,
        );
        report.insert(
            C::RecallNegative,
            baseline.recall_negative - self.recall_negative,
        );
        report.insert(C::Accuracy, baseline.accuracy - self.accuracy);
        report.insert(C::F1Score, baseline.f1_score - self.f1_score);

        report
    }
}

impl BinaryClassificationRuntime {
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

impl TryFrom<&BinaryClassificationAnalysisReport> for BinaryClassificationRuntime {
    type Error = ModelPerformanceError;
    fn try_from(payload: &BinaryClassificationAnalysisReport) -> Result<Self, Self::Error> {
        use ClassificationEvaluationMetric as C;
        let value_fetcher = |p: &BinaryClassificationAnalysisReport, key: C| {
            let Some(v) = p.get(&key) else {
                return Err(ModelPerformanceError::InvalidAnalysisReport);
            };

            Ok(*v)
        };
        Ok(BinaryClassificationRuntime {
            balanced_accuracy: value_fetcher(payload, C::BalancedAccuracy)?,
            precision_positive: value_fetcher(payload, C::PrecisionPositive)?,
            precision_negative: value_fetcher(payload, C::PrecisionNegative)?,
            recall_positive: value_fetcher(payload, C::RecallPositive)?,
            recall_negative: value_fetcher(payload, C::RecallNegative)?,
            accuracy: value_fetcher(payload, C::Accuracy)?,
            f1_score: value_fetcher(payload, C::F1Score)?,
        })
    }
}

impl TryFrom<HashMap<String, f32>> for BinaryClassificationRuntime {
    type Error = ModelPerformanceError;
    fn try_from(mut payload: HashMap<String, f32>) -> Result<Self, Self::Error> {
        let value_fetcher = |p: &mut HashMap<String, f32>, key: &str| {
            let Some(v) = p.remove(key) else {
                return Err(ModelPerformanceError::InvalidAnalysisReport);
            };

            Ok(v)
        };

        Ok(BinaryClassificationRuntime {
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

#[derive(Debug, Clone)]
pub struct LogisticRegressionRuntime {
    pub(crate) balanced_accuracy: f32,
    pub(crate) precision_positive: f32,
    pub(crate) precision_negative: f32,
    pub(crate) recall_positive: f32,
    pub(crate) recall_negative: f32,
    pub(crate) accuracy: f32,
    pub(crate) f1_score: f32,
    pub(crate) log_loss: f32,
}

impl PartialEq for LogisticRegressionRuntime {
    fn eq(&self, other: &Self) -> bool {
        if (self.balanced_accuracy - other.balanced_accuracy).abs() > EQUALITY_ERROR_ALLOWANCE {
            return false;
        }
        if (self.precision_positive - other.precision_positive).abs() > EQUALITY_ERROR_ALLOWANCE {
            return false;
        }
        if (self.precision_negative - other.precision_negative).abs() > EQUALITY_ERROR_ALLOWANCE {
            return false;
        }
        if (self.recall_positive - other.recall_positive).abs() > EQUALITY_ERROR_ALLOWANCE {
            return false;
        }
        if (self.recall_negative - other.recall_negative).abs() > EQUALITY_ERROR_ALLOWANCE {
            return false;
        }
        if (self.accuracy - other.accuracy).abs() > EQUALITY_ERROR_ALLOWANCE {
            return false;
        }
        if (self.f1_score - other.f1_score).abs() > EQUALITY_ERROR_ALLOWANCE {
            return false;
        }
        if (self.log_loss - other.log_loss).abs() > EQUALITY_ERROR_ALLOWANCE {
            return false;
        }
        true
    }
}

// assume that positive label is 1
impl LogisticRegressionRuntime {
    pub(crate) fn new(
        y_true: &[f32],
        y_pred: &[f32],
        threshold: f32,
    ) -> Result<LogisticRegressionRuntime, ModelPerformanceError> {
        if y_true.len() != y_pred.len() {
            return Err(ModelPerformanceError::DataVectorLengthMismatch);
        }

        use crate::model_perf::statistics::classification_metrics_from_parts as metrics;
        let mut c_matrix = ConfusionMatrix::default();

        let true_label = 1_f32;
        c_matrix.push_dataset(y_true, y_pred, |v: &f32| {
            v.apply_threshold(&threshold).eq(&true_label)
        })?;

        /*
                for (t, p) in zip_iters!(y_true, y_pred) {
                    let label = p.apply_threshold(&threshold);

                    c_matrix.push(ConfusionPushPayload {
                        true_gt: t.eq(&true_label),
                        true_pred: label.eq(&true_label),
                    });
                }
        */

        let accuracy = c_matrix.accuracy();
        let balanced_accuracy = metrics::balanced_accuracy(&c_matrix);
        let precision_positive = metrics::precision_positive(&c_matrix);
        let precision_negative = metrics::precision_negative(&c_matrix);
        let recall_positive = metrics::recall_positive(&c_matrix);
        let recall_negative = metrics::recall_negative(&c_matrix);
        let f1_score = metrics::f1_score(&c_matrix);
        let log_loss = metrics::log_loss_score(y_true, y_pred)?;

        Ok(LogisticRegressionRuntime {
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
    // Utlity to easliy compute the current model performance runtime state from the bucketing
    // style containers used in the stream variants
    pub(crate) fn runtime_from_parts(
        c_matrix: &ConfusionMatrix,
        log_loss: f32,
    ) -> Result<LogisticRegressionRuntime, ModelPerformanceError> {
        if c_matrix.len() == 0_f32 {
            return Err(ModelPerformanceError::EmptyDataVector);
        };
        use crate::model_perf::statistics::classification_metrics_from_parts as metrics;
        Ok(LogisticRegressionRuntime {
            balanced_accuracy: metrics::balanced_accuracy(c_matrix),
            precision_positive: metrics::precision_positive(c_matrix),
            precision_negative: metrics::precision_negative(c_matrix),
            recall_positive: metrics::recall_positive(c_matrix),
            recall_negative: metrics::recall_negative(c_matrix),
            f1_score: metrics::f1_score(c_matrix),
            log_loss,
            accuracy: c_matrix.accuracy(),
        })
    }

    pub(crate) fn runtime_drift_report(&self, bl: &Self) -> LogisticRegressionRuntimeReport {
        let mut report = LogisticRegressionAnalysisReport::with_capacity(8);
        report.insert(
            ClassificationEvaluationMetric::Accuracy,
            bl.accuracy - self.accuracy,
        );
        report.insert(
            ClassificationEvaluationMetric::BalancedAccuracy,
            bl.balanced_accuracy - self.balanced_accuracy,
        );
        report.insert(
            ClassificationEvaluationMetric::PrecisionPositive,
            bl.precision_positive - self.precision_positive,
        );
        report.insert(
            ClassificationEvaluationMetric::PrecisionNegative,
            bl.precision_negative - self.precision_negative,
        );
        report.insert(
            ClassificationEvaluationMetric::RecallPositive,
            bl.recall_positive - self.recall_positive,
        );
        report.insert(
            ClassificationEvaluationMetric::RecallNegative,
            bl.recall_negative - self.recall_negative,
        );
        report.insert(
            ClassificationEvaluationMetric::F1Score,
            bl.f1_score - self.f1_score,
        );
        report.insert(
            ClassificationEvaluationMetric::LogLoss,
            bl.log_loss - self.log_loss,
        );
        report
    }
}

impl LogisticRegressionRuntime {
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

impl TryFrom<&LogisticRegressionAnalysisReport> for LogisticRegressionRuntime {
    type Error = ModelPerformanceError;
    fn try_from(payload: &LogisticRegressionAnalysisReport) -> Result<Self, Self::Error> {
        use ClassificationEvaluationMetric as L;
        let value_fetcher = |p: &LogisticRegressionAnalysisReport, key: L| {
            let Some(v) = p.get(&key) else {
                return Err(ModelPerformanceError::InvalidAnalysisReport);
            };
            Ok(*v)
        };
        Ok(LogisticRegressionRuntime {
            balanced_accuracy: value_fetcher(payload, L::BalancedAccuracy)?,
            precision_positive: value_fetcher(payload, L::PrecisionPositive)?,
            precision_negative: value_fetcher(payload, L::PrecisionNegative)?,
            recall_positive: value_fetcher(payload, L::RecallPositive)?,
            recall_negative: value_fetcher(payload, L::RecallNegative)?,
            accuracy: value_fetcher(payload, L::Accuracy)?,
            f1_score: value_fetcher(payload, L::F1Score)?,
            log_loss: value_fetcher(payload, L::LogLoss)?,
        })
    }
}

impl TryFrom<HashMap<String, f32>> for LogisticRegressionRuntime {
    type Error = ModelPerformanceError;
    fn try_from(mut payload: HashMap<String, f32>) -> Result<Self, Self::Error> {
        let value_fetcher = |p: &mut HashMap<String, f32>, key: &str| {
            let Some(v) = p.remove(key) else {
                return Err(ModelPerformanceError::InvalidAnalysisReport);
            };
            Ok(v)
        };
        Ok(LogisticRegressionRuntime {
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

impl LogisticRegressionRuntime {
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
                    if self.log_loss > baseline.log_loss * (1_f32 + drift_threshold) {
                        res.insert(C::LogLoss, self.log_loss - baseline.log_loss);
                    }
                }
            }
        }
        res
    }
}

#[derive(Debug, Clone)]
pub struct LinearRegressionRuntime {
    pub(crate) rmse: f32,
    pub(crate) mse: f32,
    pub(crate) mae: f32,
    pub(crate) r_squared: f32,
    pub(crate) max_error: f32,
    pub(crate) msle: f32,
    pub(crate) rmsle: f32,
    pub(crate) mape: f32,
}

/// Implementing this by hand to allow for some small error.
impl PartialEq for LinearRegressionRuntime {
    fn eq(&self, other: &Self) -> bool {
        if (self.rmse - other.rmse).abs() > EQUALITY_ERROR_ALLOWANCE {
            return false;
        }
        if (self.mse - other.mse).abs() > EQUALITY_ERROR_ALLOWANCE {
            return false;
        }
        if (self.mae - other.mae).abs() > EQUALITY_ERROR_ALLOWANCE {
            return false;
        }
        if (self.r_squared - other.r_squared).abs() > EQUALITY_ERROR_ALLOWANCE {
            return false;
        }
        if (self.max_error - other.max_error).abs() > EQUALITY_ERROR_ALLOWANCE {
            return false;
        }
        if (self.msle - other.msle).abs() > EQUALITY_ERROR_ALLOWANCE {
            return false;
        }
        if (self.rmsle - other.rmsle).abs() > EQUALITY_ERROR_ALLOWANCE {
            return false;
        }
        if (self.mape - other.mape).abs() > EQUALITY_ERROR_ALLOWANCE {
            return false;
        }
        true
    }
}

impl LinearRegressionRuntime {
    pub fn new<T>(
        y_true: &[T],
        y_pred: &[T],
    ) -> Result<LinearRegressionRuntime, ModelPerformanceError>
    where
        T: Into<f64> + Copy,
    {
        let n = y_true.len() as f64;
        let error_buckets = LinearRegressionErrorBuckets::from_dataset(y_true, y_pred)?;

        let LinearRegressionErrorBuckets {
            squared_error_sum,
            abs_error_sum,
            max_error,
            squared_log_error_sum,
            abs_percent_error_sum,
            ..
        } = error_buckets;
        let r_squared = error_buckets.r2_snapshot() as f32;

        let mse = squared_error_sum / n;
        let msle = squared_log_error_sum / n;

        Ok(LinearRegressionRuntime {
            r_squared,
            rmse: (mse).powf(0.5_f64) as f32,
            mse: mse as f32,
            mae: (abs_error_sum / n) as f32,
            max_error: max_error as f32,
            msle: msle as f32,
            rmsle: (msle.powf(0.5_f64)) as f32,
            mape: (abs_percent_error_sum / n) as f32,
        })
    }

    /// Utlity to easliy compute the current model performance runtime state from the bucketing
    /// style containers used in the stream variants. Acknowledging here the explicit cast from f64
    /// to f32 which may forgoe some precision here.
    pub(crate) fn runtime_from_parts(
        parts: &LinearRegressionErrorBuckets,
    ) -> Result<LinearRegressionRuntime, ModelPerformanceError> {
        let n = parts.len;
        if n == 0_f64 {
            return Err(ModelPerformanceError::EmptyDataVector);
        }

        let mse = (parts.squared_error_sum / n) as f32;
        let msle = (parts.squared_log_error_sum / n) as f32;

        Ok(LinearRegressionRuntime {
            r_squared: parts.r2_snapshot() as f32,
            mse,
            rmse: mse.powf(0.5_f32),
            max_error: parts.max_error as f32,
            mae: (parts.abs_error_sum / n) as f32,
            msle,
            rmsle: msle.sqrt(),
            mape: (parts.abs_percent_error_sum / n) as f32,
        })
    }

    pub fn compare_to_baseline(
        &self,
        metrics: &[LinearRegressionEvaluationMetric],
        baseline: &LinearRegressionRuntime,
        drift_threshold: f32,
    ) -> LinearRegressionRuntimeReport {
        use LinearRegressionEvaluationMetric as L;
        let mut res: HashMap<L, f32> = HashMap::with_capacity(8);
        for m in metrics.iter() {
            // All values should be positive here, so all comparisons are greater than allowable
            // drift threshold define by the user.
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
                    if self.r_squared < baseline.r_squared * (1_f32 - drift_threshold) {
                        res.insert(L::RSquared, baseline.r_squared - self.r_squared);
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

    pub fn runtime_drift_report(&self, bl: &Self) -> LinearRegressionRuntimeReport {
        use crate::metrics::LinearRegressionEvaluationMetric as L;
        let mut result = LinearRegressionRuntimeReport::with_capacity(8);
        result.insert(L::RootMeanSquaredError, (bl.rmse - self.rmse).abs());
        result.insert(L::MeanSquaredError, (bl.mse - self.mse).abs());
        result.insert(L::MeanAbsoluteError, (bl.mae - self.mae).abs());
        result.insert(L::RSquared, (bl.r_squared - self.r_squared).abs());
        result.insert(L::MaxError, (bl.max_error - self.max_error).abs());
        result.insert(L::MeanSquaredLogError, (bl.msle - self.msle).abs());
        result.insert(L::RootMeanSquaredLogError, (bl.rmsle - self.rmsle).abs());
        result.insert(L::MeanAbsolutePercentageError, (bl.mape - self.mape).abs());
        result
    }

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

impl TryFrom<&LinearRegressionAnalysisReport> for LinearRegressionRuntime {
    type Error = ModelPerformanceError;
    fn try_from(payload: &LinearRegressionAnalysisReport) -> Result<Self, Self::Error> {
        use LinearRegressionEvaluationMetric as L;
        let value_fetcher = |p: &LinearRegressionAnalysisReport, key: L| {
            let Some(v) = p.get(&key) else {
                return Err(ModelPerformanceError::InvalidAnalysisReport);
            };
            Ok(*v)
        };

        Ok(LinearRegressionRuntime {
            rmse: value_fetcher(payload, L::RootMeanSquaredError)?,
            mse: value_fetcher(payload, L::MeanSquaredError)?,
            mae: value_fetcher(payload, L::MeanAbsoluteError)?,
            r_squared: value_fetcher(payload, L::RSquared)?,
            max_error: value_fetcher(payload, L::MaxError)?,
            msle: value_fetcher(payload, L::MeanSquaredLogError)?,
            rmsle: value_fetcher(payload, L::RootMeanSquaredLogError)?,
            mape: value_fetcher(payload, L::MeanAbsolutePercentageError)?,
        })
    }
}

impl TryFrom<HashMap<String, f32>> for LinearRegressionRuntime {
    type Error = ModelPerformanceError;
    fn try_from(mut payload: HashMap<String, f32>) -> Result<Self, Self::Error> {
        let value_fetcher = |p: &mut HashMap<String, f32>, key: &str| {
            let Some(v) = p.remove(key) else {
                return Err(ModelPerformanceError::InvalidAnalysisReport);
            };
            Ok(v)
        };

        Ok(LinearRegressionRuntime {
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
