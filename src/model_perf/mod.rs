use crate::data_handler::{determine_type, PassedType};
use crate::zip;
use numpy::PyUntypedArray;
use pyo3::prelude::*;
use std::collections::HashMap;
use std::error::Error;

fn update_failure_report_above(map: &HashMap<String, String>, metric: String, diff: f32) {
    map.insert(metric, format!("Exceeded threshold by {diff}"));
}

fn update_failure_report_below(map: &HashMap<String, String>, metric: String, diff: f32) {
    map.insert(metric, format!("Below threshold by {diff}"));
}

pub trait ModelPerformanceType {
    fn compare_to_baseline(
        &self,
        _baseline: &Self,
        _drift_threshold: f32,
    ) -> HashMap<String, String> {
        HashMap::new()
    }
}
struct GeneralClassificationMetrics;

impl GeneralClassificationMetrics {
    fn balanced_accuracy(rp: f32, rn: f32) -> f32 {
        rp * rn * 0.5_f32
    }

    fn precision_positive(y_pred: &Vec<f32>, y_true: &Vec<f32>) -> f32 {
        let total_pred_positives: f32 = y_pred.iter().sum::<f32>();
        let mut true_positives: f32 = 0_f32;
        for (t, p) in zip!(y_true, y_pred) {
            if *t == 1_f32 && t == p {
                true_positives += 1_f32;
            }
        }
        true_positives / total_pred_positives
    }

    fn precision_negative(y_pred: &Vec<f32>, y_true: &Vec<f32>, len: f32) -> f32 {
        let total_pred_negatives: f32 = len - y_pred.iter().sum::<f32>();
        let mut true_negatives: f32 = 0_f32;
        for (t, p) in zip!(y_true, y_pred) {
            if *t == 0_f32 && t == p {
                true_negatives += 1_f32;
            }
        }
        true_negatives / total_pred_negatives
    }

    fn recall_positive(y_pred: &Vec<f32>, y_true: &Vec<f32>) -> f32 {
        let total_true_positives: f32 = y_true.iter().sum::<f32>();
        let mut true_positives: f32 = 0_f32;
        for (t, p) in zip!(y_true, y_pred) {
            if *t == 1_f32 && t == p {
                true_positives += 1_f32;
            }
        }
        true_positives / total_true_positives
    }

    fn recall_negative(y_pred: &Vec<f32>, y_true: &Vec<f32>, len: f32) -> f32 {
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

    fn log_loss_score(y_proba: &Vec<f32>, y_true: &Vec<f32>, mean_f: f32) -> f32 {
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

    fn convert_usize(
        _py: Python<'_>,
        arr: &Bound<'_, PyUntypedArray>,
        passed_type: PassedType,
    ) -> Result<Vec<usize>, Box<dyn Error>> {
        // pulls the py data type out
        // applying labels as usize
        let res: Vec<usize> = match passed_type {
            PassedType::Float => arr
                .iter()?
                .map(|item| item.unwrap().extract::<f64>().unwrap() as usize)
                .collect::<Vec<usize>>(),
            PassedType::Integer => arr
                .iter()?
                .clone()
                .map(|item| item.unwrap().extract::<f32>().unwrap() as usize)
                .collect::<Vec<usize>>(),
            _ => panic!("Wrong PassedType made it through"),
        };
        Ok(res)
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

impl ModelPerformanceType for BinaryClassificationReport {
    fn compare_to_baseline(
        &self,
        baseline: &Self,
        drift_threshold: f32,
    ) -> HashMap<String, String> {
        let mut res: HashMap<String, String> = HashMap::with_capacity(7);
        let drift_factor = 1_f32 - drift_threshold;
        if self.balanced_accuracy < baseline.balanced_accuracy * drift_factor {
            update_failure_report_below(
                &mut res,
                "BalancedAccuracy".into(),
                baseline.balanced_accuracy - self.balanced_accuracy,
            );
        }
        if self.precision_positive < baseline.precision_positive * drift_factor {
            update_failure_report_below(
                &mut res,
                "PrecisionPositive".into(),
                baseline.precision_positive - self.precision_positive,
            );
        }
        if self.precision_negative < baseline.precision_negative * drift_factor {
            update_failure_report_below(
                &mut res,
                "PrecisionNegative".into(),
                baseline.precision_negative - self.precision_negative,
            );
        }
        if self.recall_positive < baseline.recall_positive * drift_factor {
            update_failure_report_below(
                &mut res,
                "RecallPositive".into(),
                baseline.recall_positive - self.recall_positive,
            );
        }
        if self.recall_negative < baseline.recall_negative * drift_factor {
            update_failure_report_below(
                &mut res,
                "RecallNegative".into(),
                baseline.recall_negative - self.recall_negative,
            );
        }
        if self.accuracy < baseline.accuracy * drift_factor {
            update_failure_report_below(
                &mut res,
                "Accuracy".into(),
                baseline.accuracy - self.accuracy,
            );
        }
        if self.f1_score < baseline.f1_score * drift_factor {
            update_failure_report_below(
                &mut res,
                "F1Score".into(),
                baseline.f1_score - self.f1_score,
            );
        }
        res
    }
}

pub struct LogisiticRegressionReport {
    balanced_accuracy: f32,
    precision_positive: f32,
    precision_negative: f32,
    recall_positive: f32,
    recall_negative: f32,
    accuracy: f32,
    f1_score: f32,
    log_loss: f32,
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
        };
        todo!()
    }
}

impl ClassificationPerf {
    pub fn new_from_label(
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

    pub fn new_from_proba(
        py: Python<'_>,
        y_true_src: &Bound<'_, PyUntypedArray>,
        y_pred_src: &Bound<'_, PyUntypedArray>,
        threshold: f32,
    ) -> Result<ClassificationPerf, Box<dyn Error>> {
        let (y_pred, y_true) = PerfEntry::validate_and_cast_classification(
            py,
            y_true_src,
            y_pred_src,
            true,
            Some(threshold),
        )?;
        let len: f32 = y_pred.len() as f32;
        let mean_f: f32 = 1_f32 / len;
        Ok(ClassificationPerf {
            y_true,
            y_pred,
            mean_f,
            len,
        })
    }

    pub fn report(&self) -> HashMap<String, f32> {
        let res: HashMap<String, f32> = HashMap::new();
        res
    }
}

struct LogisiticRegressionPerf {
    y_proba: Vec<f32>,
    label_data: ClassificationPerf,
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
        let Some(rmse) = map.get("RootMeanSqauredError") else {
            return Err("Invalid regression report".into());
        };
        let Some(mse) = map.get("MeanSquaredError") else {
            return Err("Invalid regression report".into());
        };
        let Some(mae) = map.get("MeanAbsoluteError") else {
            return Err("Invalid regression report".into());
        };
        let Some(r_squared) = map.get("R-Squared") else {
            return Err("Invalid regression report".into());
        };
        let Some(max_error) = map.get("MaxError") else {
            return Err("Invalid regression report".into());
        };
        let Some(msle) = map.get("MeanSqauredLogError") else {
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
        map.insert("RootMeanSqauredError".into(), self.rmse);
        map.insert("MeanSqauredError".into(), self.mse);
        map.insert("MeanAbsoluteError".into(), self.mae);
        map.insert("R-Squared".into(), self.r_squared);
        map.insert("MaxError".into(), self.max_error);
        map.insert("MeanSqauredLogError".into(), self.msle);
        map.insert("RootMeanSquaredLogError".into(), self.rmsle);
        map.insert("MeanAbsolutePercentageError".into(), self.mape);
        map
    }
}

impl ModelPerformanceType for LinearRegressionReport {
    fn compare_to_baseline(
        &self,
        baseline: &LinearRegressionReport,
        drift_threshold: f32,
    ) -> HashMap<String, String> {
        let mut res: HashMap<String, String> = HashMap::with_capacity(8);
        if self.rmse > baseline.rmse * (1_f32 + drift_threshold) {
            res.insert(
                "RootMeanSqauredError".into(),
                format!("Exceeded threshold by {}", self.rmse - baseline.rmse),
            );
        }
        if self.mse > baseline.mse * (1_f32 + drift_threshold) {
            res.insert(
                "MeanSqauredError".into(),
                format!("Exceeded threshold by {}", self.mse - baseline.mse),
            );
        }
        if self.mae > baseline.mae * (1_f32 + drift_threshold) {
            res.insert(
                "MeanAbsoluteError".into(),
                format!("Exceeded threshold by {}", self.mae - baseline.mae),
            );
        }
        if self.r_squared > baseline.r_squared * (1_f32 + drift_threshold) {
            res.insert(
                "R-Sqaured".into(),
                format!(
                    "Exceeded threshold by {}",
                    self.r_squared - baseline.r_squared
                ),
            );
        }
        if self.max_error > baseline.max_error * (1_f32 + drift_threshold) {
            res.insert(
                "MaxError".into(),
                format!(
                    "Exceeded threshold by {}",
                    self.max_error - baseline.max_error
                ),
            );
        }
        if self.msle > baseline.msle * (1_f32 + drift_threshold) {
            res.insert(
                "MeanSqauredLogError".into(),
                format!("Exceeded threshold by {}", self.msle - baseline.msle),
            );
        }
        if self.rmsle > baseline.rmsle * (1_f32 + drift_threshold) {
            res.insert(
                "RootMeanSqauredLogError".into(),
                format!("Exceeded threshold by {}", self.rmsle - baseline.rmsle),
            );
        }
        if self.mape > baseline.mape * (1_f32 + drift_threshold) {
            res.insert(
                "MeanAbsolutePercentageError".into(),
                format!("Exceeded threshold by {}", self.rmsle - baseline.rmsle),
            );
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
        for (t, p) in zip!(&self.y_true, &self.y_pred) {
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
        let ss_regression = self
            .y_pred
            .iter()
            .map(|y| (y - y_mean).powi(2))
            .sum::<f32>();
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
            sum += (1_f32 - t).log10() - (1_f32 - p).log10();
        }
        sum.powi(2) / self.mean_f
    }

    fn root_mean_squared_log_error(&self) -> f32 {
        let mut sum = 0_f32;
        for (t, p) in zip!(&self.y_true, &self.y_pred) {
            sum += (1_f32 - t).log10() - (1_f32 - p).log10();
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
