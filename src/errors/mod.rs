use thiserror::Error;

#[derive(Debug, Error)]
pub enum BiasError {
    #[error("No deviation in behavior between facets")]
    NoFacetDeviation,
    #[error("No segmentation parameters provided")]
    NoSegmentationParameters,
    #[error("Ambiguous segmentation parameters")]
    AmbiguousSegmentationParameters,
    #[error("All data arrays must have equal length")]
    DataLengthError,
}

#[derive(Debug, Error)]
pub enum DriftError {
    #[error("Data used for runtime drift analysis must be non empty")]
    EmptyRuntimeData,
    #[error("Unable to convert internal timestamp into DateTime object")]
    DateTimeError,
    #[error("Internal runtime bins are malformed")]
    MalformedRuntimeData,
    #[error("Baseline data must be non empty")]
    EmptyBaselineData,
    #[error("NaN values are not supported")]
    NaNValueError,
}

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

#[derive(Debug, Error)]
pub enum DataBiasRuntimeError {
    #[error("ClassImbalance not present")]
    ClassImbalance,
    #[error("DifferenceInProportionOfLabels not present")]
    DifferenceInProportionOfLabels,
    #[error("KlDivergence not present")]
    KlDivergence,
    #[error("JsDivergence not present")]
    JsDivergence,
    #[error("TotalVariationDistance not present")]
    TotalVariationDistance,
    #[error("KolmorogvSmirnov not present")]
    KolmorogvSmirnov,
    #[error("LpNorm not present")]
    LpNorm,
}

#[derive(Debug, Error)]
pub enum ModelBiasRuntimeError {
    #[error("DifferenceInPositivePredictedLabels not present")]
    DifferenceInPositivePredictedLabels,
    #[error("DisparateImpact not present")]
    DisparateImpact,
    #[error("AccuracyDifference not present")]
    AccuracyDifference,
    #[error("RecallDifference not present")]
    RecallDifference,
    #[error("DifferenceInConditionalAcceptance not present")]
    DifferenceInConditionalAcceptance,
    #[error("DifferenceInAcceptanceRate not present")]
    DifferenceInAcceptanceRate,
    #[error("SpecialityDifference not present")]
    SpecialityDifference,
    #[error("DifferenceInConditionalRejection not present")]
    DifferenceInConditionalRejection,
    #[error("TreatmentEquity not present")]
    TreatmentEquity,
    #[error("ConditionalDemographicDesparityPredictedLabels not present")]
    ConditionalDemographicDesparityPredictedLabels,
    #[error("DifferenceInRejectionRate not present")]
    DifferenceInRejectionRate,
    #[error("GeneralizedEntropy not present")]
    GeneralizedEntropy,
}

#[derive(Debug, Error)]
pub enum ModelPerformanceError {
    #[error("Empty data vectors")]
    EmptyDataVector,
    #[error("Data vectors must be equal length")]
    DataVectorLengthMismatch,
    #[error("Invalid analysis report")]
    InvalidAnalysisReport,
    // py only errors
    #[allow(unused)]
    #[error("Data Vector type mistmatch")]
    DataVectorTypeMismatch,
    #[allow(unused)]
    #[error("UnSupportedType")]
    UnsupportedTypeError,
}

#[cfg(feature = "python")]
pub(crate) mod py_errors {
    use super::*;
    use pyo3::{exceptions, PyErr};

    impl Into<PyErr> for DataBiasRuntimeError {
        fn into(self) -> PyErr {
            let err_msg = self.to_string();
            exceptions::PyValueError::new_err(err_msg)
        }
    }

    impl Into<PyErr> for ModelBiasRuntimeError {
        fn into(self) -> PyErr {
            let err_msg = self.to_string();
            exceptions::PyValueError::new_err(err_msg)
        }
    }

    impl Into<PyErr> for DriftError {
        fn into(self) -> PyErr {
            let err_message = self.to_string();
            match self {
                Self::EmptyRuntimeData | Self::EmptyBaselineData | Self::NaNValueError => {
                    exceptions::PyValueError::new_err(err_message)
                }
                Self::DateTimeError | Self::MalformedRuntimeData => {
                    exceptions::PySystemError::new_err(err_message)
                }
            }
        }
    }

    impl Into<PyErr> for InvalidMetricError {
        fn into(self) -> PyErr {
            let err_msg = self.to_string();
            exceptions::PyValueError::new_err(err_msg)
        }
    }

    impl Into<PyErr> for BiasError {
        fn into(self) -> PyErr {
            let err_msg = self.to_string();
            exceptions::PyValueError::new_err(err_msg)
        }
    }
}
