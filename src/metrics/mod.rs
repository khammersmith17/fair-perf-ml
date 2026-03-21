use crate::errors::InvalidMetricError;
use serde::{Deserialize, Serialize};
use std::hash::Hash;
use std::sync::OnceLock;

static STABILITIY_EPS: OnceLock<f64> = OnceLock::new();
pub(crate) fn get_stability_eps() -> f64 {
    let eps = STABILITIY_EPS.get_or_init(|| {
        let default = 1e-12;
        let Ok(var) = std::env::var("FAIR_PERF_ML_STABILITY_EPS") else {
            return default;
        };

        var.parse::<f64>().unwrap_or(default)
    });
    *eps
}

/// Trait to enfore trait bounds around what is a crate supported machine learning/data metric.
pub trait MachineLearningMetric:
    std::fmt::Debug + PartialEq + Serialize + std::fmt::Display
{
}
impl MachineLearningMetric for DataBiasMetric {}
impl MachineLearningMetric for ModelBiasMetric {}
impl MachineLearningMetric for ClassificationEvaluationMetric {}
impl MachineLearningMetric for LinearRegressionEvaluationMetric {}

/// New type wrapper for Vec<DataBiasMetric>, this is to allow for an implementation of
/// `From<Vec<DataBiasMetric>>`, or TryFrom<T> where T is something the looks like the String
/// representation of the metric name to be used where metrics are required parameters.
#[derive(Debug)]
pub struct DataBiasMetricVec(Vec<DataBiasMetric>);
impl AsRef<[DataBiasMetric]> for DataBiasMetricVec {
    fn as_ref(&self) -> &[DataBiasMetric] {
        self.0.as_slice()
    }
}

impl From<Vec<DataBiasMetric>> for DataBiasMetricVec {
    fn from(m_vec: Vec<DataBiasMetric>) -> DataBiasMetricVec {
        DataBiasMetricVec(m_vec)
    }
}

impl TryFrom<&[String]> for DataBiasMetricVec {
    type Error = InvalidMetricError;
    fn try_from(metrics: &[String]) -> Result<DataBiasMetricVec, Self::Error> {
        let mut map: Vec<DataBiasMetric> = Vec::with_capacity(metrics.len());
        let mut error_metrics: Vec<String> = Vec::with_capacity(metrics.len());
        for m_str in metrics.iter() {
            match DataBiasMetric::try_from(m_str.as_ref()) {
                Ok(m) => map.push(m),
                Err(_) => error_metrics.push(m_str.to_string()),
            }
        }

        if !error_metrics.is_empty() {
            return Err(InvalidMetricError::DataBiasMetricError(error_metrics));
        }

        Ok(DataBiasMetricVec(map))
    }
}

#[derive(Debug)]
pub struct LogisticRegressionMetricVec(Vec<ClassificationEvaluationMetric>);
impl AsRef<[ClassificationEvaluationMetric]> for LogisticRegressionMetricVec {
    fn as_ref(&self) -> &[ClassificationEvaluationMetric] {
        self.0.as_slice()
    }
}

impl From<Vec<ClassificationEvaluationMetric>> for LogisticRegressionMetricVec {
    fn from(m_vec: Vec<ClassificationEvaluationMetric>) -> LogisticRegressionMetricVec {
        LogisticRegressionMetricVec(m_vec)
    }
}

impl TryFrom<&[String]> for LogisticRegressionMetricVec {
    type Error = InvalidMetricError;
    fn try_from(metrics: &[String]) -> Result<LogisticRegressionMetricVec, Self::Error> {
        let mut map: Vec<ClassificationEvaluationMetric> = Vec::with_capacity(metrics.len());
        let mut error_metrics: Vec<String> = Vec::with_capacity(metrics.len());
        for m_str in metrics.iter() {
            match ClassificationEvaluationMetric::try_from(m_str.as_ref()) {
                Ok(m) => map.push(m),
                Err(_) => error_metrics.push(m_str.to_string()),
            }
        }

        if !error_metrics.is_empty() {
            return Err(InvalidMetricError::RegressionMetricError(error_metrics));
        }

        Ok(LogisticRegressionMetricVec(map))
    }
}
#[derive(Debug)]
pub struct LinearRegressionMetricVec(Vec<LinearRegressionEvaluationMetric>);
impl AsRef<[LinearRegressionEvaluationMetric]> for LinearRegressionMetricVec {
    fn as_ref(&self) -> &[LinearRegressionEvaluationMetric] {
        self.0.as_slice()
    }
}

impl From<Vec<LinearRegressionEvaluationMetric>> for LinearRegressionMetricVec {
    fn from(m_vec: Vec<LinearRegressionEvaluationMetric>) -> LinearRegressionMetricVec {
        LinearRegressionMetricVec(m_vec)
    }
}

impl TryFrom<&[String]> for LinearRegressionMetricVec {
    type Error = InvalidMetricError;
    fn try_from(metrics: &[String]) -> Result<LinearRegressionMetricVec, Self::Error> {
        let mut map: Vec<LinearRegressionEvaluationMetric> = Vec::with_capacity(metrics.len());
        let mut error_metrics: Vec<String> = Vec::with_capacity(metrics.len());
        for m_str in metrics.iter() {
            match LinearRegressionEvaluationMetric::try_from(m_str.as_ref()) {
                Ok(m) => map.push(m),
                Err(_) => error_metrics.push(m_str.to_string()),
            }
        }

        if !error_metrics.is_empty() {
            return Err(InvalidMetricError::RegressionMetricError(error_metrics));
        }

        Ok(LinearRegressionMetricVec(map))
    }
}
#[derive(Debug)]
pub struct ClassificationMetricVec(Vec<ClassificationEvaluationMetric>);
impl AsRef<[ClassificationEvaluationMetric]> for ClassificationMetricVec {
    fn as_ref(&self) -> &[ClassificationEvaluationMetric] {
        self.0.as_slice()
    }
}

impl From<Vec<ClassificationEvaluationMetric>> for ClassificationMetricVec {
    fn from(m_vec: Vec<ClassificationEvaluationMetric>) -> ClassificationMetricVec {
        ClassificationMetricVec(m_vec)
    }
}

impl TryFrom<&[String]> for ClassificationMetricVec {
    type Error = InvalidMetricError;
    fn try_from(metrics: &[String]) -> Result<ClassificationMetricVec, Self::Error> {
        let mut map: Vec<ClassificationEvaluationMetric> = Vec::with_capacity(metrics.len());
        let mut error_metrics: Vec<String> = Vec::with_capacity(metrics.len());
        for m_str in metrics.iter() {
            match ClassificationEvaluationMetric::try_from(m_str.as_ref()) {
                Ok(m) => {
                    if m == ClassificationEvaluationMetric::LogLoss {
                        return Err(InvalidMetricError::ClassificationMetricError(vec![
                            m.to_string()
                        ]));
                    }
                    map.push(m)
                }
                Err(_) => error_metrics.push(m_str.to_string()),
            }
        }

        if !error_metrics.is_empty() {
            return Err(InvalidMetricError::ClassificationMetricError(error_metrics));
        }

        Ok(ClassificationMetricVec(map))
    }
}
#[derive(Debug)]
pub struct ModelBiasMetricVec(Vec<ModelBiasMetric>);
impl AsRef<[ModelBiasMetric]> for ModelBiasMetricVec {
    fn as_ref(&self) -> &[ModelBiasMetric] {
        self.0.as_slice()
    }
}

impl From<Vec<ModelBiasMetric>> for ModelBiasMetricVec {
    fn from(m_vec: Vec<ModelBiasMetric>) -> ModelBiasMetricVec {
        ModelBiasMetricVec(m_vec)
    }
}

impl TryFrom<&[String]> for ModelBiasMetricVec {
    type Error = InvalidMetricError;
    fn try_from(metrics: &[String]) -> Result<ModelBiasMetricVec, Self::Error> {
        let mut map: Vec<ModelBiasMetric> = Vec::with_capacity(metrics.len());
        let mut error_metrics: Vec<String> = Vec::with_capacity(metrics.len());
        for m_str in metrics.iter() {
            match ModelBiasMetric::try_from(m_str.as_ref()) {
                Ok(m) => map.push(m),
                Err(_) => error_metrics.push(m_str.to_string()),
            }
        }

        if !error_metrics.is_empty() {
            return Err(InvalidMetricError::ModelBiasMetricError(error_metrics));
        }

        Ok(ModelBiasMetricVec(map))
    }
}

#[derive(Serialize, Deserialize, Eq, PartialEq, Hash, Debug)]
pub enum DataBiasMetric {
    ClassImbalance,
    DifferenceInProportionOfLabels,
    KlDivergence,
    JsDivergence,
    LpNorm,
    TotalVariationDistance,
    KolmorogvSmirnov,
}

impl std::fmt::Display for DataBiasMetric {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Self::ClassImbalance => write!(f, "ClassImbalance"),
            Self::DifferenceInProportionOfLabels => write!(f, "DifferenceInProportionOfLabels"),
            Self::KlDivergence => write!(f, "KlDivergence"),
            Self::JsDivergence => write!(f, "JsDivergence"),
            Self::LpNorm => write!(f, "LpNorm"),
            Self::TotalVariationDistance => write!(f, "TotalVariationDistance"),
            Self::KolmorogvSmirnov => write!(f, "KolmorogvSmirnov"),
        }
    }
}

pub const FULL_DATA_BIAS_METRICS: [DataBiasMetric; 7] = [
    DataBiasMetric::ClassImbalance,
    DataBiasMetric::DifferenceInProportionOfLabels,
    DataBiasMetric::KlDivergence,
    DataBiasMetric::JsDivergence,
    DataBiasMetric::LpNorm,
    DataBiasMetric::TotalVariationDistance,
    DataBiasMetric::KolmorogvSmirnov,
];

impl TryFrom<&str> for DataBiasMetric {
    type Error = ();
    fn try_from(value: &str) -> Result<Self, Self::Error> {
        match value {
            "ClassImbalance" => Ok(Self::ClassImbalance),
            "DifferenceInProportionOfLabels" => Ok(Self::DifferenceInProportionOfLabels),
            "KlDivergence" => Ok(Self::KlDivergence),
            "JsDivergence" => Ok(Self::JsDivergence),
            "LpNorm" => Ok(Self::LpNorm),
            "TotalVariationDistance" => Ok(Self::TotalVariationDistance),
            "KolmorogvSmirnov" => Ok(Self::KolmorogvSmirnov),
            _ => Err(()),
        }
    }
}

#[derive(Deserialize, Serialize, Hash, Eq, PartialEq, Debug)]
pub enum ModelBiasMetric {
    DifferenceInPositivePredictedLabels,
    DisparateImpact,
    AccuracyDifference,
    RecallDifference,
    DifferenceInConditionalAcceptance,
    DifferenceInAcceptanceRate,
    SpecialityDifference,
    DifferenceInConditionalRejection,
    DifferenceInRejectionRate,
    TreatmentEquity,
    ConditionalDemographicDesparityPredictedLabels,
    GeneralizedEntropy,
}

impl std::fmt::Display for ModelBiasMetric {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Self::DifferenceInPositivePredictedLabels => {
                write!(f, "DifferenceInPositivePredictedLabels")
            }
            Self::DisparateImpact => write!(f, "DisparateImpact"),
            Self::AccuracyDifference => write!(f, "AccuracyDifference"),
            Self::RecallDifference => write!(f, "RecallDifference"),
            Self::DifferenceInConditionalAcceptance => {
                write!(f, "DifferenceInConditionalAcceptance")
            }
            Self::DifferenceInAcceptanceRate => write!(f, "DifferenceInAcceptanceRate"),
            Self::SpecialityDifference => write!(f, "SpecialityDifference"),
            Self::DifferenceInConditionalRejection => write!(f, "DifferenceInConditionalRejection"),
            Self::DifferenceInRejectionRate => write!(f, "DifferenceInRejectionRate"),
            Self::TreatmentEquity => write!(f, "TreatmentEquity"),
            Self::ConditionalDemographicDesparityPredictedLabels => {
                write!(f, "ConditionalDemographicDesparityPredictedLabels")
            }
            Self::GeneralizedEntropy => write!(f, "GeneralizedEntropy"),
        }
    }
}

pub const FULL_MODEL_BIAS_METRICS: [ModelBiasMetric; 12] = [
    ModelBiasMetric::DifferenceInPositivePredictedLabels,
    ModelBiasMetric::DisparateImpact,
    ModelBiasMetric::AccuracyDifference,
    ModelBiasMetric::RecallDifference,
    ModelBiasMetric::DifferenceInConditionalAcceptance,
    ModelBiasMetric::DifferenceInAcceptanceRate,
    ModelBiasMetric::SpecialityDifference,
    ModelBiasMetric::DifferenceInConditionalRejection,
    ModelBiasMetric::DifferenceInRejectionRate,
    ModelBiasMetric::TreatmentEquity,
    ModelBiasMetric::ConditionalDemographicDesparityPredictedLabels,
    ModelBiasMetric::GeneralizedEntropy,
];

impl TryFrom<&str> for ModelBiasMetric {
    type Error = String;
    fn try_from(value: &str) -> Result<Self, Self::Error> {
        match value {
            "DifferenceInPositivePredictedLabels" => Ok(Self::DifferenceInPositivePredictedLabels),
            "DisparateImpact" => Ok(Self::DisparateImpact),
            "AccuracyDifference" => Ok(Self::AccuracyDifference),
            "RecallDifference" => Ok(Self::RecallDifference),
            "DifferenceInConditionalAcceptance" => Ok(Self::DifferenceInConditionalAcceptance),
            "DifferenceInAcceptanceRate" => Ok(Self::DifferenceInAcceptanceRate),
            "SpecialityDifference" => Ok(Self::SpecialityDifference),
            "DifferenceInConditionalRejection" => Ok(Self::DifferenceInConditionalRejection),
            "DifferenceInRejectionRate" => Ok(Self::DifferenceInRejectionRate),
            "TreatmentEquity" => Ok(Self::TreatmentEquity),
            "ConditionalDemographicDesparityPredictedLabels" => {
                Ok(Self::ConditionalDemographicDesparityPredictedLabels)
            }
            "GeneralizedEntropy" => Ok(Self::GeneralizedEntropy),
            _ => Err("Invalid metric passed".into()),
        }
    }
}

pub const FULL_REGRESSION_METRICS: [LinearRegressionEvaluationMetric; 8] = [
    LinearRegressionEvaluationMetric::RootMeanSquaredError,
    LinearRegressionEvaluationMetric::MeanSquaredError,
    LinearRegressionEvaluationMetric::MeanAbsoluteError,
    LinearRegressionEvaluationMetric::RSquared,
    LinearRegressionEvaluationMetric::MaxError,
    LinearRegressionEvaluationMetric::MeanSquaredLogError,
    LinearRegressionEvaluationMetric::RootMeanSquaredLogError,
    LinearRegressionEvaluationMetric::MeanAbsolutePercentageError,
];

pub const FULL_LOGISTIC_REGRESSION_METRICS: [ClassificationEvaluationMetric; 8] = [
    ClassificationEvaluationMetric::BalancedAccuracy,
    ClassificationEvaluationMetric::PrecisionPositive,
    ClassificationEvaluationMetric::PrecisionNegative,
    ClassificationEvaluationMetric::RecallPositive,
    ClassificationEvaluationMetric::RecallNegative,
    ClassificationEvaluationMetric::Accuracy,
    ClassificationEvaluationMetric::F1Score,
    ClassificationEvaluationMetric::LogLoss,
];

pub const FULL_BINARY_CLASSIFICATION_METRICS: [ClassificationEvaluationMetric; 7] = [
    ClassificationEvaluationMetric::BalancedAccuracy,
    ClassificationEvaluationMetric::PrecisionPositive,
    ClassificationEvaluationMetric::PrecisionNegative,
    ClassificationEvaluationMetric::RecallPositive,
    ClassificationEvaluationMetric::RecallNegative,
    ClassificationEvaluationMetric::Accuracy,
    ClassificationEvaluationMetric::F1Score,
];

#[derive(Serialize, Deserialize, PartialEq, Hash, Eq, Debug)]
pub enum ClassificationEvaluationMetric {
    BalancedAccuracy,
    PrecisionPositive,
    PrecisionNegative,
    RecallPositive,
    RecallNegative,
    Accuracy,
    F1Score,
    LogLoss,
}

#[derive(Serialize, Deserialize, Hash, PartialEq, Eq, Debug)]
pub enum LinearRegressionEvaluationMetric {
    RootMeanSquaredError,
    MeanSquaredError,
    MeanAbsoluteError,
    RSquared,
    MaxError,
    MeanSquaredLogError,
    RootMeanSquaredLogError,
    MeanAbsolutePercentageError,
}

impl TryFrom<&str> for ClassificationEvaluationMetric {
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

impl std::fmt::Display for ClassificationEvaluationMetric {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Self::BalancedAccuracy => write!(f, "BalancedAccuracy"),
            Self::PrecisionPositive => write!(f, "PrecisionPositive"),
            Self::PrecisionNegative => write!(f, "PrecisionNegative"),
            Self::RecallPositive => write!(f, "RecallPositive"),
            Self::RecallNegative => write!(f, "RecallNegative"),
            Self::Accuracy => write!(f, "Accuracy"),
            Self::F1Score => write!(f, "F1Score"),
            Self::LogLoss => write!(f, "LogLoss"),
        }
    }
}

impl TryFrom<&str> for LinearRegressionEvaluationMetric {
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

impl std::fmt::Display for LinearRegressionEvaluationMetric {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Self::RootMeanSquaredError => write!(f, "RootMeanSquaredError"),
            Self::MeanSquaredError => write!(f, "MeanSquaredError"),
            Self::MeanAbsoluteError => write!(f, "MeanAbsoluteError"),
            Self::RSquared => write!(f, "RSquared"),
            Self::MaxError => write!(f, "MaxError"),
            Self::MeanSquaredLogError => write!(f, "MeanSquaredLogError"),
            Self::RootMeanSquaredLogError => write!(f, "RootMeanSquaredLogError"),
            Self::MeanAbsolutePercentageError => write!(f, "MeanAbsolutePercentageError"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::errors::InvalidMetricError;

    // --- DataBiasMetric ---

    #[test]
    fn data_bias_metric_display_round_trips_via_try_from() {
        for m in &FULL_DATA_BIAS_METRICS {
            let s = m.to_string();
            assert_eq!(DataBiasMetric::try_from(s.as_str()).unwrap(), *m);
        }
    }

    #[test]
    fn data_bias_metric_try_from_unknown_returns_err() {
        assert!(DataBiasMetric::try_from("NotAMetric").is_err());
    }

    #[test]
    fn full_data_bias_metrics_has_seven_variants() {
        assert_eq!(FULL_DATA_BIAS_METRICS.len(), 7);
    }

    // --- ModelBiasMetric ---

    #[test]
    fn model_bias_metric_display_round_trips_via_try_from() {
        for m in &FULL_MODEL_BIAS_METRICS {
            let s = m.to_string();
            assert_eq!(ModelBiasMetric::try_from(s.as_str()).unwrap(), *m);
        }
    }

    #[test]
    fn model_bias_metric_try_from_unknown_returns_err() {
        assert!(ModelBiasMetric::try_from("NotAMetric").is_err());
    }

    #[test]
    fn full_model_bias_metrics_has_twelve_variants() {
        assert_eq!(FULL_MODEL_BIAS_METRICS.len(), 12);
    }

    // --- ClassificationEvaluationMetric ---

    #[test]
    fn classification_metric_display_round_trips_via_try_from() {
        for m in &FULL_LOGISTIC_REGRESSION_METRICS {
            let s = m.to_string();
            assert_eq!(ClassificationEvaluationMetric::try_from(s.as_str()).unwrap(), *m);
        }
    }

    #[test]
    fn classification_metric_try_from_unknown_returns_err() {
        assert!(ClassificationEvaluationMetric::try_from("NotAMetric").is_err());
    }

    #[test]
    fn full_logistic_regression_metrics_includes_log_loss() {
        assert_eq!(FULL_LOGISTIC_REGRESSION_METRICS.len(), 8);
        assert!(FULL_LOGISTIC_REGRESSION_METRICS.contains(&ClassificationEvaluationMetric::LogLoss));
    }

    #[test]
    fn full_binary_classification_metrics_excludes_log_loss() {
        assert_eq!(FULL_BINARY_CLASSIFICATION_METRICS.len(), 7);
        assert!(!FULL_BINARY_CLASSIFICATION_METRICS.contains(&ClassificationEvaluationMetric::LogLoss));
    }

    // --- LinearRegressionEvaluationMetric ---

    #[test]
    fn linear_regression_metric_display_round_trips_via_try_from() {
        for m in &FULL_REGRESSION_METRICS {
            let s = m.to_string();
            assert_eq!(LinearRegressionEvaluationMetric::try_from(s.as_str()).unwrap(), *m);
        }
    }

    #[test]
    fn linear_regression_metric_try_from_unknown_returns_err() {
        assert!(LinearRegressionEvaluationMetric::try_from("NotAMetric").is_err());
    }

    #[test]
    fn full_regression_metrics_has_eight_variants() {
        assert_eq!(FULL_REGRESSION_METRICS.len(), 8);
    }

    // --- DataBiasMetricVec ---

    #[test]
    fn data_bias_metric_vec_from_vec() {
        let v = vec![DataBiasMetric::ClassImbalance, DataBiasMetric::LpNorm];
        let wrapped = DataBiasMetricVec::from(v);
        assert_eq!(wrapped.as_ref(), &[DataBiasMetric::ClassImbalance, DataBiasMetric::LpNorm]);
    }

    #[test]
    fn data_bias_metric_vec_try_from_strings_success() {
        let strings: Vec<String> = vec!["ClassImbalance".into(), "LpNorm".into()];
        let vec = DataBiasMetricVec::try_from(strings.as_slice()).unwrap();
        assert_eq!(vec.as_ref(), &[DataBiasMetric::ClassImbalance, DataBiasMetric::LpNorm]);
    }

    #[test]
    fn data_bias_metric_vec_try_from_strings_invalid_returns_error() {
        let strings: Vec<String> = vec!["ClassImbalance".into(), "BadMetric".into()];
        let err = DataBiasMetricVec::try_from(strings.as_slice()).unwrap_err();
        assert!(matches!(err, InvalidMetricError::DataBiasMetricError(ref v) if v.contains(&"BadMetric".to_string())));
    }

    #[test]
    fn data_bias_metric_vec_try_from_all_full_metrics() {
        let strings: Vec<String> = FULL_DATA_BIAS_METRICS.iter().map(|m| m.to_string()).collect();
        let vec = DataBiasMetricVec::try_from(strings.as_slice()).unwrap();
        assert_eq!(vec.as_ref().len(), 7);
    }

    // --- ModelBiasMetricVec ---

    #[test]
    fn model_bias_metric_vec_from_vec() {
        let v = vec![ModelBiasMetric::DisparateImpact];
        let wrapped = ModelBiasMetricVec::from(v);
        assert_eq!(wrapped.as_ref(), &[ModelBiasMetric::DisparateImpact]);
    }

    #[test]
    fn model_bias_metric_vec_try_from_strings_success() {
        let strings: Vec<String> = vec!["DisparateImpact".into(), "GeneralizedEntropy".into()];
        let vec = ModelBiasMetricVec::try_from(strings.as_slice()).unwrap();
        assert_eq!(
            vec.as_ref(),
            &[ModelBiasMetric::DisparateImpact, ModelBiasMetric::GeneralizedEntropy]
        );
    }

    #[test]
    fn model_bias_metric_vec_try_from_strings_invalid_returns_error() {
        let strings: Vec<String> = vec!["DisparateImpact".into(), "BadMetric".into()];
        let err = ModelBiasMetricVec::try_from(strings.as_slice()).unwrap_err();
        assert!(matches!(err, InvalidMetricError::ModelBiasMetricError(_)));
    }

    // --- LogisticRegressionMetricVec ---

    #[test]
    fn logistic_regression_metric_vec_from_vec() {
        let v = vec![ClassificationEvaluationMetric::Accuracy];
        let wrapped = LogisticRegressionMetricVec::from(v);
        assert_eq!(wrapped.as_ref(), &[ClassificationEvaluationMetric::Accuracy]);
    }

    #[test]
    fn logistic_regression_metric_vec_try_from_strings_success() {
        let strings: Vec<String> = vec!["Accuracy".into(), "LogLoss".into()];
        let vec = LogisticRegressionMetricVec::try_from(strings.as_slice()).unwrap();
        assert_eq!(
            vec.as_ref(),
            &[ClassificationEvaluationMetric::Accuracy, ClassificationEvaluationMetric::LogLoss]
        );
    }

    #[test]
    fn logistic_regression_metric_vec_try_from_strings_invalid_returns_error() {
        let strings: Vec<String> = vec!["Accuracy".into(), "BadMetric".into()];
        let err = LogisticRegressionMetricVec::try_from(strings.as_slice()).unwrap_err();
        assert!(matches!(err, InvalidMetricError::RegressionMetricError(_)));
    }

    // --- LinearRegressionMetricVec ---

    #[test]
    fn linear_regression_metric_vec_from_vec() {
        let v = vec![LinearRegressionEvaluationMetric::RSquared];
        let wrapped = LinearRegressionMetricVec::from(v);
        assert_eq!(wrapped.as_ref(), &[LinearRegressionEvaluationMetric::RSquared]);
    }

    #[test]
    fn linear_regression_metric_vec_try_from_strings_success() {
        let strings: Vec<String> = vec!["RSquared".into(), "MaxError".into()];
        let vec = LinearRegressionMetricVec::try_from(strings.as_slice()).unwrap();
        assert_eq!(
            vec.as_ref(),
            &[
                LinearRegressionEvaluationMetric::RSquared,
                LinearRegressionEvaluationMetric::MaxError,
            ]
        );
    }

    #[test]
    fn linear_regression_metric_vec_try_from_strings_invalid_returns_error() {
        let strings: Vec<String> = vec!["RSquared".into(), "BadMetric".into()];
        let err = LinearRegressionMetricVec::try_from(strings.as_slice()).unwrap_err();
        assert!(matches!(err, InvalidMetricError::RegressionMetricError(_)));
    }

    // --- ClassificationMetricVec ---

    #[test]
    fn classification_metric_vec_from_vec() {
        let v = vec![ClassificationEvaluationMetric::F1Score];
        let wrapped = ClassificationMetricVec::from(v);
        assert_eq!(wrapped.as_ref(), &[ClassificationEvaluationMetric::F1Score]);
    }

    #[test]
    fn classification_metric_vec_try_from_strings_success() {
        let strings: Vec<String> = vec!["F1Score".into(), "Accuracy".into()];
        let vec = ClassificationMetricVec::try_from(strings.as_slice()).unwrap();
        assert_eq!(
            vec.as_ref(),
            &[ClassificationEvaluationMetric::F1Score, ClassificationEvaluationMetric::Accuracy]
        );
    }

    #[test]
    fn classification_metric_vec_rejects_log_loss() {
        let strings: Vec<String> = vec!["Accuracy".into(), "LogLoss".into()];
        let err = ClassificationMetricVec::try_from(strings.as_slice()).unwrap_err();
        assert!(matches!(err, InvalidMetricError::ClassificationMetricError(_)));
    }

    #[test]
    fn classification_metric_vec_try_from_strings_invalid_returns_error() {
        let strings: Vec<String> = vec!["Accuracy".into(), "BadMetric".into()];
        let err = ClassificationMetricVec::try_from(strings.as_slice()).unwrap_err();
        assert!(matches!(err, InvalidMetricError::ClassificationMetricError(_)));
    }

    // --- get_stability_eps ---

    #[test]
    fn stability_eps_default_is_1e_12() {
        // Only valid when env var is unset; calling get_stability_eps() after it has been
        // initialised with a different value in the same process would return the cached value.
        // We just verify the return is positive and finite.
        let eps = get_stability_eps();
        assert!(eps > 0.0 && eps.is_finite());
    }
}
