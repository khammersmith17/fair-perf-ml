use crate::data_handler::{determine_type, PassedType};
use crate::zip;
use numpy::PyUntypedArray;
use pyo3::prelude::*;
use std::collections::HashMap;
use std::error::Error;

pub const FULL_REGRESSION_METRICS: [LinearRegressionEvaluationMetrics; 8] = [
    LinearRegressionEvaluationMetrics::RootMeanSquaredError,
    LinearRegressionEvaluationMetrics::MeanSquaredError,
    LinearRegressionEvaluationMetrics::MeanAbsoluteError,
    LinearRegressionEvaluationMetrics::RSquared,
    LinearRegressionEvaluationMetrics::MaxError,
    LinearRegressionEvaluationMetrics::MeanSquaredLogError,
    LinearRegressionEvaluationMetrics::RootMeanSquaredLogError,
    LinearRegressionEvaluationMetrics::MeanAbsolutePercentageError,
];

pub const FULL_LOGISTIC_REGRESSION_METRICS: [ClassificationEvaluationMetrics; 8] = [
    ClassificationEvaluationMetrics::BalancedAccuracy,
    ClassificationEvaluationMetrics::PrecisionPositive,
    ClassificationEvaluationMetrics::PrecisionNegative,
    ClassificationEvaluationMetrics::RecallPositive,
    ClassificationEvaluationMetrics::RecallNegative,
    ClassificationEvaluationMetrics::Accuracy,
    ClassificationEvaluationMetrics::F1Score,
    ClassificationEvaluationMetrics::LogLoss,
];

pub const FULL_BINARY_CLASSIFICATION_METRICS: [ClassificationEvaluationMetrics; 7] = [
    ClassificationEvaluationMetrics::BalancedAccuracy,
    ClassificationEvaluationMetrics::PrecisionPositive,
    ClassificationEvaluationMetrics::PrecisionNegative,
    ClassificationEvaluationMetrics::RecallPositive,
    ClassificationEvaluationMetrics::RecallNegative,
    ClassificationEvaluationMetrics::Accuracy,
    ClassificationEvaluationMetrics::F1Score,
];

pub enum ClassificationEvaluationMetrics {
    BalancedAccuracy,
    PrecisionPositive,
    PrecisionNegative,
    RecallPositive,
    RecallNegative,
    Accuracy,
    F1Score,
    LogLoss,
}

pub enum LinearRegressionEvaluationMetrics {
    RootMeanSquaredError,
    MeanSquaredError,
    MeanAbsoluteError,
    RSquared,
    MaxError,
    MeanSquaredLogError,
    RootMeanSquaredLogError,
    MeanAbsolutePercentageError,
}

impl TryFrom<&str> for ClassificationEvaluationMetrics {
    type Error = String;
    fn try_from(val: &str) -> Result<Self, Self::Error> {
        match val {
            "BalancedAccuracy" => Ok(Self::BalancedAccuracy),
            "PrecisionPositive" => Ok(Self::PrecisionPositive),
            "PrecisionNegative" => Ok(Self::PrecisionNegative),
            "RecallPositive" => Ok(Self::RecallPositive),
            "RecallNegative" => Ok(Self::RecallNegative),
            "Accuracy" => Ok(Self::Accuracy),
            "F1Score" => Ok(Self::F1Score),
            "LogLoss" => Ok(Self::LogLoss),
            _ => Err("Invalid metric type".into()),
        }
    }
}

impl TryFrom<&str> for LinearRegressionEvaluationMetrics {
    type Error = String;
    fn try_from(val: &str) -> Result<Self, Self::Error> {
        match val {
            "RootMeanSquaredError" => Ok(Self::RootMeanSquaredError),
            "MeanSquaredError" => Ok(Self::MeanSquaredError),
            "MeanAbsoluteError" => Ok(Self::MeanAbsoluteError),
            "RSquared" => Ok(Self::RSquared),
            "MaxError" => Ok(Self::MaxError),
            "MeanSquaredLogError" => Ok(Self::MeanSquaredLogError),
            "RootMeanSquaredLogError" => Ok(Self::RootMeanSquaredLogError),
            "MeanAbsolutePercentageError" => Ok(Self::MeanAbsolutePercentageError),
            _ => Err("Invalid metric name passed".into()),
        }
    }
}

pub fn map_string_to_linear_metric(
    metrics_string: Vec<String>,
) -> Result<Vec<LinearRegressionEvaluationMetrics>, Box<dyn Error>> {
    let mut v: Vec<LinearRegressionEvaluationMetrics> = Vec::with_capacity(metrics_string.len());
    for m_str in metrics_string.iter() {
        let new = match LinearRegressionEvaluationMetrics::try_from(m_str.as_str()) {
            Ok(val) => val,
            Err(_) => return Err("Invalid metric name".into()),
        };
        v.push(new);
    }
    Ok(v)
}

pub fn map_string_to_bin_metric(
    metrics_string: Vec<String>,
) -> Result<Vec<ClassificationEvaluationMetrics>, Box<dyn Error>> {
    let mut v: Vec<ClassificationEvaluationMetrics> = Vec::with_capacity(metrics_string.len());
    for m_str in metrics_string.iter() {
        let new = match ClassificationEvaluationMetrics::try_from(m_str.as_str()) {
            Ok(val) => val,
            Err(_) => return Err("Invalid metric name".into()),
        };
        v.push(new);
    }
    Ok(v)
}

fn update_failure_report_above(map: &mut HashMap<String, String>, metric: String, diff: f32) {
    map.insert(metric, format!("Exceeded threshold by {diff}"));
}

fn update_failure_report_below(map: &mut HashMap<String, String>, metric: String, diff: f32) {
    map.insert(metric, format!("Below threshold by {diff}"));
}

pub fn model_perf_regression<'py>(
    py: Python<'_>,
    y_pred_src: &Bound<'_, PyUntypedArray>,
    y_true_src: &Bound<'_, PyUntypedArray>,
) -> Result<HashMap<String, f32>, Box<dyn Error>> {
    let (y_true, y_pred) = PerfEntry::validate_and_cast_regression(py, y_true_src, y_pred_src)?;
    let perf: LinearRegressionPerf = LinearRegressionPerf::new(y_true, y_pred);
    let report: LinearRegressionReport = perf.into();
    Ok(report.generate_report())
}

pub fn model_perf_classification<'py>(
    py: Python<'_>,
    y_pred_src: &Bound<'_, PyUntypedArray>,
    y_true_src: &Bound<'_, PyUntypedArray>,
) -> Result<HashMap<String, f32>, Box<dyn Error>> {
    let perf: ClassificationPerf = ClassificationPerf::new(py, y_true_src, y_pred_src)?;
    let report: BinaryClassificationReport = perf.into();
    Ok(report.generate_report())
}

pub fn model_perf_logistic_regression<'py>(
    py: Python<'_>,
    y_pred_src: &Bound<'_, PyUntypedArray>,
    y_true_src: &Bound<'_, PyUntypedArray>,
    threshold: f32,
) -> Result<HashMap<String, f32>, Box<dyn Error>> {
    let perf: LogisticRegressionPerf =
        LogisticRegressionPerf::new(py, y_pred_src, y_true_src, threshold)?;
    let lr_report: LogisticRegressionReport = perf.into();
    let map = lr_report.report();
    Ok(map)
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
            if *t == 1_f32 && t == p {
                true_positives += 1_f32;
            }
        }
        true_positives / total_pred_positives
    }

    fn precision_negative(y_pred: &[f32], y_true: &[f32], len: f32) -> f32 {
        let total_pred_negatives: f32 = len - y_pred.iter().sum::<f32>();
        let mut true_negatives: f32 = 0_f32;
        for (t, p) in zip!(y_true, y_pred) {
            if *t == 0_f32 && t == p {
                true_negatives += 1_f32;
            }
        }
        true_negatives / total_pred_negatives
    }

    fn recall_positive(y_pred: &[f32], y_true: &[f32]) -> f32 {
        let total_true_positives: f32 = y_true.iter().sum::<f32>();
        let mut true_positives: f32 = 0_f32;
        for (t, p) in zip!(y_true, y_pred) {
            if *t == 1_f32 && t == p {
                true_positives += 1_f32;
            }
        }
        true_positives / total_true_positives
    }

    fn recall_negative(y_pred: &[f32], y_true: &[f32], len: f32) -> f32 {
        let total_true_negatives: f32 = len - y_true.iter().sum::<f32>();
        let mut true_negatives: f32 = 0_f32;
        for (t, p) in zip!(y_true, y_pred) {
            if *t == 0_f32 && t == p {
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
        -1_f32 * mean_f * penalties
    }
}
pub struct PerfEntry;

impl PerfEntry {
    fn validate_and_cast_classification(
        py: Python<'_>,
        y_true_src: &Bound<'_, PyUntypedArray>,
        y_pred_src: &Bound<'_, PyUntypedArray>,
        needs_decision: bool,
        threshold: Option<f32>,
    ) -> Result<(Vec<f32>, Vec<f32>), Box<dyn Error>> {
        let pred_type: PassedType = determine_type(py, y_pred_src);
        let gt_type: PassedType = determine_type(py, y_true_src);

        if pred_type != gt_type {
            return Err("Type between y_true and y_pred do not match".into());
        }

        if needs_decision {
            let Some(thres) = threshold else {
                return Err("Threshold must be set for logisitc model type".into());
            };
            Self::convert_w_label_application(py, y_true_src, y_pred_src, thres, gt_type, pred_type)
        } else {
            let y_pred = Self::convert_f32(py, y_pred_src, pred_type)?;
            let y_true = Self::convert_f32(py, y_true_src, gt_type)?;
            Ok((y_pred, y_true))
        }
    }

    pub fn validate_and_cast_regression(
        py: Python<'_>,
        y_true_src: &Bound<'_, PyUntypedArray>,
        y_pred_src: &Bound<'_, PyUntypedArray>,
    ) -> Result<(Vec<f32>, Vec<f32>), Box<dyn Error>> {
        let y_true: Vec<f32> = Self::convert_f32(py, y_pred_src, determine_type(py, y_true_src))?;
        let y_pred: Vec<f32> = Self::convert_f32(py, y_true_src, determine_type(py, y_pred_src))?;
        Ok((y_true, y_pred))
    }

    fn convert_f32(
        _py: Python<'_>,
        arr: &Bound<'_, PyUntypedArray>,
        passed_type: PassedType,
    ) -> Result<Vec<f32>, Box<dyn Error>> {
        // pulls the py data type out
        // applying labels as usize
        let res: Vec<f32> = match passed_type {
            PassedType::Float => arr
                .iter()?
                .map(|item| item.unwrap().extract::<f64>().unwrap() as f32)
                .collect::<Vec<f32>>(),
            PassedType::Integer => arr
                .iter()?
                .clone()
                .map(|item| item.unwrap().extract::<f32>().unwrap() as f32)
                .collect::<Vec<f32>>(),
            _ => panic!("Wrong PassedType made it through"),
        };
        Ok(res)
    }

    fn convert_w_label_application(
        py: Python<'_>,
        y_true_src: &Bound<'_, PyUntypedArray>,
        y_pred_src: &Bound<'_, PyUntypedArray>,
        threshold: f32,
        true_passed_type: PassedType,
        pred_passed_type: PassedType,
    ) -> Result<(Vec<f32>, Vec<f32>), Box<dyn Error>> {
        // if string
        // applying the threshold label

        let y_pred: Vec<f32> = Self::convert_f32(py, y_pred_src, pred_passed_type)?
            .iter()
            .map(|x| if x >= &threshold { 1_f32 } else { 0_f32 })
            .collect::<Vec<f32>>();
        let y_true: Vec<f32> = Self::convert_f32(py, y_true_src, true_passed_type)?
            .iter()
            .map(|x| if x >= &threshold { 1_f32 } else { 0_f32 })
            .collect::<Vec<f32>>();
        Ok((y_pred, y_true))
    }
}

pub struct BinaryClassificationReport {
    balanced_accuracy: f32,
    precision_positive: f32,
    precision_negative: f32,
    recall_positive: f32,
    recall_negative: f32,
    accuracy: f32,
    f1_score: f32,
}

impl BinaryClassificationReport {
    pub fn generate_report(&self) -> HashMap<String, f32> {
        let mut map: HashMap<String, f32> = HashMap::with_capacity(7);
        map.insert("BalancedAccuracy".into(), self.balanced_accuracy);
        map.insert("PrecisionPositive".into(), self.precision_positive);
        map.insert("PrecisionNegative".into(), self.precision_negative);
        map.insert("RecallPositive".into(), self.recall_positive);
        map.insert("RecallNegative".into(), self.recall_negative);
        map.insert("Accuracy".into(), self.accuracy);
        map.insert("F1Score".into(), self.f1_score);
        map
    }
}

impl TryFrom<HashMap<String, f32>> for BinaryClassificationReport {
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

        Ok(BinaryClassificationReport {
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

impl BinaryClassificationReport {
    pub fn compare_to_baseline(
        &self,
        metrics: &[ClassificationEvaluationMetrics],
        baseline: &Self,
        drift_threshold: f32,
    ) -> Result<HashMap<String, String>, Box<dyn Error>> {
        use ClassificationEvaluationMetrics as C;
        let mut res: HashMap<String, String> = HashMap::with_capacity(7);
        let drift_factor = 1_f32 - drift_threshold;
        // log loss should not be present here
        // so when log loss comes up, we return Err
        for m in metrics.iter() {
            match *m {
                C::BalancedAccuracy => {
                    if self.balanced_accuracy < baseline.balanced_accuracy * drift_factor {
                        update_failure_report_below(
                            &mut res,
                            "BalancedAccuracy".into(),
                            baseline.balanced_accuracy - self.balanced_accuracy,
                        );
                    }
                }
                C::PrecisionPositive => {
                    if self.precision_positive < baseline.precision_positive * drift_factor {
                        update_failure_report_below(
                            &mut res,
                            "PrecisionPositive".into(),
                            baseline.precision_positive - self.precision_positive,
                        );
                    }
                }
                C::PrecisionNegative => {
                    if self.precision_negative < baseline.precision_negative * drift_factor {
                        update_failure_report_below(
                            &mut res,
                            "PrecisionNegative".into(),
                            baseline.precision_negative - self.precision_negative,
                        );
                    }
                }
                C::RecallPositive => {
                    if self.recall_positive < baseline.recall_positive * drift_factor {
                        update_failure_report_below(
                            &mut res,
                            "RecallPositive".into(),
                            baseline.recall_positive - self.recall_positive,
                        );
                    }
                }
                C::RecallNegative => {
                    if self.recall_negative < baseline.recall_negative * drift_factor {
                        update_failure_report_below(
                            &mut res,
                            "RecallNegative".into(),
                            baseline.recall_negative - self.recall_negative,
                        );
                    }
                }
                C::Accuracy => {
                    if self.accuracy < baseline.accuracy * drift_factor {
                        update_failure_report_below(
                            &mut res,
                            "Accuracy".into(),
                            baseline.accuracy - self.accuracy,
                        );
                    }
                }
                C::F1Score => {
                    if self.f1_score < baseline.f1_score * drift_factor {
                        update_failure_report_below(
                            &mut res,
                            "F1Score".into(),
                            baseline.f1_score - self.f1_score,
                        );
                    }
                }
                C::LogLoss => return Err("LogLoss is not valid for Binary Classification".into()),
            }
        }

        Ok(res)
    }
}

pub struct LogisticRegressionReport {
    balanced_accuracy: f32,
    precision_positive: f32,
    precision_negative: f32,
    recall_positive: f32,
    recall_negative: f32,
    accuracy: f32,
    f1_score: f32,
    log_loss: f32,
}

impl LogisticRegressionReport {
    pub fn report(&self) -> HashMap<String, f32> {
        let mut map: HashMap<String, f32> = HashMap::with_capacity(8);
        map.insert("BalancedAccuracy".into(), self.balanced_accuracy);
        map.insert("PrecisionPositive".into(), self.precision_positive);
        map.insert("PrecisionNegative".into(), self.precision_negative);
        map.insert("RecallPositive".into(), self.recall_positive);
        map.insert("RecallNegative".into(), self.recall_negative);
        map.insert("Accuracy".into(), self.accuracy);
        map.insert("F1Score".into(), self.f1_score);
        map.insert("LogLoss".into(), self.log_loss);
        map
    }
}

impl TryFrom<HashMap<String, f32>> for LogisticRegressionReport {
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

        Ok(LogisticRegressionReport {
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

impl LogisticRegressionReport {
    pub fn compare_to_baseline(
        &self,
        metrics: &[ClassificationEvaluationMetrics],
        baseline: &Self,
        drift_threshold: f32,
    ) -> HashMap<String, String> {
        // all the metrics here are used, at this point we have
        // everything correct, thus no Result<T,E>
        use ClassificationEvaluationMetrics as C;
        let mut res: HashMap<String, String> = HashMap::with_capacity(7);
        let drift_factor = 1_f32 - drift_threshold;
        for m in metrics.iter() {
            match *m {
                C::BalancedAccuracy => {
                    if self.balanced_accuracy < baseline.balanced_accuracy * drift_factor {
                        update_failure_report_below(
                            &mut res,
                            "BalancedAccuracy".into(),
                            baseline.balanced_accuracy - self.balanced_accuracy,
                        );
                    }
                }
                C::PrecisionPositive => {
                    if self.precision_positive < baseline.precision_positive * drift_factor {
                        update_failure_report_below(
                            &mut res,
                            "PrecisionPositive".into(),
                            baseline.precision_positive - self.precision_positive,
                        );
                    }
                }
                C::PrecisionNegative => {
                    if self.precision_negative < baseline.precision_negative * drift_factor {
                        update_failure_report_below(
                            &mut res,
                            "PrecisionNegative".into(),
                            baseline.precision_negative - self.precision_negative,
                        );
                    }
                }
                C::RecallPositive => {
                    if self.recall_positive < baseline.recall_positive * drift_factor {
                        update_failure_report_below(
                            &mut res,
                            "RecallPositive".into(),
                            baseline.recall_positive - self.recall_positive,
                        );
                    }
                }
                C::RecallNegative => {
                    if self.recall_negative < baseline.recall_negative * drift_factor {
                        update_failure_report_below(
                            &mut res,
                            "RecallNegative".into(),
                            baseline.recall_negative - self.recall_negative,
                        );
                    }
                }
                C::Accuracy => {
                    if self.accuracy < baseline.accuracy * drift_factor {
                        update_failure_report_below(
                            &mut res,
                            "Accuracy".into(),
                            baseline.accuracy - self.accuracy,
                        );
                    }
                }
                C::F1Score => {
                    if self.f1_score < baseline.f1_score * drift_factor {
                        update_failure_report_below(
                            &mut res,
                            "F1Score".into(),
                            baseline.f1_score - self.f1_score,
                        );
                    }
                }
                C::LogLoss => {
                    if self.log_loss < baseline.log_loss * drift_factor {
                        update_failure_report_below(
                            &mut res,
                            "F1Score".into(),
                            baseline.log_loss - self.log_loss,
                        );
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

impl Into<BinaryClassificationReport> for ClassificationPerf {
    fn into(self) -> BinaryClassificationReport {
        let recall_positive =
            GeneralClassificationMetrics::recall_positive(&self.y_pred, &self.y_true);
        let precision_positive =
            GeneralClassificationMetrics::precision_positive(&self.y_pred, &self.y_true);
        let recall_negative =
            GeneralClassificationMetrics::recall_negative(&self.y_pred, &self.y_true, self.len);
        BinaryClassificationReport {
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
        py: Python<'_>,
        y_true_src: &Bound<'_, PyUntypedArray>,
        y_pred_src: &Bound<'_, PyUntypedArray>,
    ) -> Result<ClassificationPerf, Box<dyn Error>> {
        let (y_pred, y_true) =
            PerfEntry::validate_and_cast_classification(py, y_true_src, y_pred_src, false, None)?;
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

struct LogisticRegressionPerf {
    y_true: Vec<f32>,
    y_pred: Vec<f32>,
    y_proba: Vec<f32>,
    mean_f: f32,
    len: f32,
}

impl LogisticRegressionPerf {
    fn new(
        py: Python<'_>,
        y_pred_src: &Bound<'_, PyUntypedArray>,
        y_true_src: &Bound<'_, PyUntypedArray>,
        threshold: f32,
    ) -> Result<LogisticRegressionPerf, Box<dyn Error>> {
        let true_type: PassedType = determine_type(py, y_true_src);
        let y_true: Vec<f32> = PerfEntry::convert_f32(py, y_true_src, true_type)?;
        let pred_type = determine_type(py, y_pred_src);
        let y_proba: Vec<f32> = PerfEntry::convert_f32(py, y_true_src, pred_type)?;
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

impl Into<LogisticRegressionReport> for LogisticRegressionPerf {
    fn into(self) -> LogisticRegressionReport {
        let recall_positive =
            GeneralClassificationMetrics::recall_positive(&self.y_pred, &self.y_true);
        let precision_positive =
            GeneralClassificationMetrics::precision_positive(&self.y_pred, &self.y_true);
        let recall_negative =
            GeneralClassificationMetrics::recall_negative(&self.y_pred, &self.y_true, self.len);
        LogisticRegressionReport {
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

pub struct LinearRegressionReport {
    rmse: f32,
    mse: f32,
    mae: f32,
    r_squared: f32,
    max_error: f32,
    msle: f32,
    rmsle: f32,
    mape: f32,
}

impl TryFrom<HashMap<String, f32>> for LinearRegressionReport {
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
        Ok(LinearRegressionReport {
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

impl LinearRegressionReport {
    pub fn generate_report(&self) -> HashMap<String, f32> {
        let mut map: HashMap<String, f32> = HashMap::with_capacity(8);
        map.insert("RootMeanSquaredError".into(), self.rmse);
        map.insert("MeanSquaredError".into(), self.mse);
        map.insert("MeanAbsoluteError".into(), self.mae);
        map.insert("RSquared".into(), self.r_squared);
        map.insert("MaxError".into(), self.max_error);
        map.insert("MeanSquaredLogError".into(), self.msle);
        map.insert("RootMeanSquaredLogError".into(), self.rmsle);
        map.insert("MeanAbsolutePercentageError".into(), self.mape);
        map
    }
}

impl LinearRegressionReport {
    pub fn compare_to_baseline(
        &self,
        metrics: &[LinearRegressionEvaluationMetrics],
        baseline: &LinearRegressionReport,
        drift_threshold: f32,
    ) -> HashMap<String, String> {
        use LinearRegressionEvaluationMetrics as L;
        let mut res: HashMap<String, String> = HashMap::with_capacity(8);
        for m in metrics.iter() {
            match *m {
                L::RootMeanSquaredError => {
                    if self.rmse > baseline.rmse * (1_f32 + drift_threshold) {
                        update_failure_report_above(
                            &mut res,
                            "RootMeanSquaredError".into(),
                            self.rmse - baseline.rmse,
                        );
                    }
                }
                L::MeanSquaredError => {
                    if self.mse > baseline.mse * (1_f32 + drift_threshold) {
                        update_failure_report_above(
                            &mut res,
                            "MeanSquaredError".into(),
                            self.mse - baseline.mse,
                        );
                    }
                }
                L::MeanAbsoluteError => {
                    if self.mae > baseline.mae * (1_f32 + drift_threshold) {
                        update_failure_report_above(
                            &mut res,
                            "MeanAbsoluteError".into(),
                            self.mae - baseline.mae,
                        );
                    }
                }
                L::RSquared => {
                    if self.r_squared > baseline.r_squared * (1_f32 + drift_threshold) {
                        update_failure_report_above(
                            &mut res,
                            "RSquared".into(),
                            self.r_squared - baseline.r_squared,
                        );
                    }
                }
                L::MaxError => {
                    if self.max_error > baseline.max_error * (1_f32 + drift_threshold) {
                        update_failure_report_above(
                            &mut res,
                            "MaxError".into(),
                            self.max_error - baseline.max_error,
                        );
                    }
                }
                L::MeanSquaredLogError => {
                    if self.msle > baseline.msle * (1_f32 + drift_threshold) {
                        update_failure_report_above(
                            &mut res,
                            "MeanSquaredLogError".into(),
                            self.msle - baseline.msle,
                        );
                    }
                }
                L::RootMeanSquaredLogError => {
                    if self.rmsle > baseline.rmsle * (1_f32 + drift_threshold) {
                        update_failure_report_above(
                            &mut res,
                            "RootMeanSquaredLogError".into(),
                            self.rmsle - baseline.rmsle,
                        );
                    }
                }
                L::MeanAbsolutePercentageError => {
                    if self.mape > baseline.mape * (1_f32 + drift_threshold) {
                        update_failure_report_above(
                            &mut res,
                            "MeanAbsolutePercentageError".into(),
                            self.mape - baseline.mape,
                        );
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

impl Into<LinearRegressionReport> for LinearRegressionPerf {
    fn into(self) -> LinearRegressionReport {
        LinearRegressionReport {
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
    pub fn new(y_true: Vec<f32>, y_pred: Vec<f32>) -> LinearRegressionPerf {
        let mean_f: f32 = 1_f32 / y_pred.len() as f32;
        LinearRegressionPerf {
            y_true,
            y_pred,
            mean_f,
        }
    }

    fn root_mean_squared_error(&self) -> f32 {
        let mut errors = 0_f32;
        for (t, p) in self.y_true.iter().zip(&self.y_pred) {
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
        for (t, p) in self.y_true.iter().zip(&self.y_pred) {
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
