use thiserror::Error;

/// Result type for any operation that may return a `ModelPerformanceError`.
pub type ModelPerfResult<T> = Result<T, ModelPerformanceError>;

#[derive(Debug, Error)]
pub enum BiasError {
    #[error("No deviation in behavior between facets")]
    NoFacetDeviation,
    #[error("No segmentation parameters provided")]
    NoSegmentationParameters,
    #[error("Ambiguous segmentation parameters")]
    AmbiguousSegmentationParameters,
    #[error("All data arrays must have equal length and be at least lenght 1")]
    DataLengthError,
}

impl From<ModelPerformanceError> for BiasError {
    fn from(err: ModelPerformanceError) -> BiasError {
        match err {
            ModelPerformanceError::InvalidData => BiasError::NoFacetDeviation,
            ModelPerformanceError::EmptyDataVector => BiasError::DataLengthError,
            _ => panic!("Invalid error conversion"),
        }
    }
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
    #[error("Metric cannot be computed with given data")]
    InvalidData,
    #[error("Empty data vectors")]
    EmptyDataVector,
    #[error("Data vectors must be equal length")]
    DataVectorLengthMismatch,
    #[error("Invalid analysis report")]
    InvalidAnalysisReport,
    #[error("Bias Error: {0:?}")]
    BiasError(BiasError),
    // py only errors
    #[allow(unused)]
    #[error("Data Vector type mistmatch")]
    DataVectorTypeMismatch,
    #[allow(unused)]
    #[error("UnSupportedType")]
    UnsupportedTypeError,
}

impl From<BiasError> for ModelPerformanceError {
    fn from(err: BiasError) -> ModelPerformanceError {
        ModelPerformanceError::BiasError(err)
    }
}

#[cfg(feature = "python")]
pub(crate) mod py_errors {
    use super::*;
    use pyo3::{exceptions, PyErr};

    impl From<DataBiasRuntimeError> for PyErr {
        fn from(err: DataBiasRuntimeError) -> PyErr {
            let err_msg = err.to_string();
            exceptions::PyValueError::new_err(err_msg)
        }
    }

    impl From<ModelBiasRuntimeError> for PyErr {
        fn from(err: ModelBiasRuntimeError) -> PyErr {
            let err_msg = err.to_string();
            exceptions::PyValueError::new_err(err_msg)
        }
    }

    impl From<DriftError> for PyErr {
        fn from(err: DriftError) -> PyErr {
            let err_message = err.to_string();
            match err {
                DriftError::EmptyRuntimeData
                | DriftError::EmptyBaselineData
                | DriftError::NaNValueError => exceptions::PyValueError::new_err(err_message),
                DriftError::DateTimeError | DriftError::MalformedRuntimeData => {
                    exceptions::PySystemError::new_err(err_message)
                }
            }
        }
    }

    impl From<InvalidMetricError> for PyErr {
        fn from(err: InvalidMetricError) -> PyErr {
            let err_msg = err.to_string();
            exceptions::PyValueError::new_err(err_msg)
        }
    }

    impl From<BiasError> for PyErr {
        fn from(err: BiasError) -> PyErr {
            let err_msg = err.to_string();
            exceptions::PyValueError::new_err(err_msg)
        }
    }

    impl From<ModelPerformanceError> for PyErr {
        fn from(err: ModelPerformanceError) -> PyErr {
            use ModelPerformanceError as E;
            let err_message = err.to_string();

            match err {
                E::EmptyDataVector | E::DataVectorLengthMismatch | E::InvalidAnalysisReport => {
                    exceptions::PyValueError::new_err(err_message)
                }
                _ => exceptions::PyTypeError::new_err(err_message),
            }
        }
    }
}
