use crate::drift::psi::StringLike;
use serde::{Deserialize, Serialize};
use std::hash::Hash;
use thiserror::Error;

#[cfg(feature = "python")]
use pyo3::{exceptions::PyValueError, PyErr};

pub(crate) trait MachineLearningMetric {}

impl MachineLearningMetric for DataBiasMetric {}
impl MachineLearningMetric for ModelBiasMetric {}
impl MachineLearningMetric for ClassificationEvaluationMetric {}
impl MachineLearningMetric for LinearRegressionEvaluationMetric {}

#[derive(Debug, Error)]
pub enum InvalidMetricError {
    #[error("Metrics: {0:?} are not supported")]
    DataBiasMetricError(Vec<String>),
    #[error("Metrics: {0:?} are not supported")]
    ModelBiasMetricError(Vec<String>),
    #[error("Metrics: {0:?} are not supported")]
    ClassificationMetricError(Vec<String>),
    #[error("Metrics: {0:?} are not supported")]
    RegressionMetricError(Vec<String>),
}

#[cfg(feature = "python")]
impl Into<PyErr> for InvalidMetricError {
    fn into(self) -> PyErr {
        let err_msg = self.to_string();
        PyValueError::new_err(err_msg)
    }
}

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

impl<S> TryFrom<&[S]> for DataBiasMetricVec
where
    S: StringLike,
{
    type Error = InvalidMetricError;
    fn try_from(metrics: &[S]) -> Result<DataBiasMetricVec, Self::Error> {
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

impl<S> TryFrom<&[S]> for LogisticRegressionMetricVec
where
    S: StringLike,
{
    type Error = InvalidMetricError;
    fn try_from(metrics: &[S]) -> Result<LogisticRegressionMetricVec, Self::Error> {
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

impl<S> TryFrom<&[S]> for LinearRegressionMetricVec
where
    S: StringLike,
{
    type Error = InvalidMetricError;
    fn try_from(metrics: &[S]) -> Result<LinearRegressionMetricVec, Self::Error> {
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

impl<S> TryFrom<&[S]> for ClassificationMetricVec
where
    S: StringLike,
{
    type Error = InvalidMetricError;
    fn try_from(metrics: &[S]) -> Result<ClassificationMetricVec, Self::Error> {
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

impl<S> TryFrom<&[S]> for ModelBiasMetricVec
where
    S: StringLike,
{
    type Error = InvalidMetricError;
    fn try_from(metrics: &[S]) -> Result<ModelBiasMetricVec, Self::Error> {
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

#[derive(Serialize, Deserialize, Eq, PartialEq, Hash)]
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

#[derive(Deserialize, Serialize, Hash, Eq, PartialEq)]
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

#[derive(Serialize, Deserialize, PartialEq, Hash, Eq)]
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

#[derive(Serialize, Deserialize, Hash, PartialEq, Eq)]
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

pub(crate) trait ModelPerfMetricBase {}
impl ModelPerfMetricBase for ClassificationEvaluationMetric {}
impl ModelPerfMetricBase for LinearRegressionEvaluationMetric {}

pub(crate) trait ModelPerformanceMetric: Eq + Hash + ModelPerfMetricBase {}
impl<T> ModelPerformanceMetric for T where T: Eq + Hash + ModelPerfMetricBase {}
