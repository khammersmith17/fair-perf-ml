use thiserror::Error;

/// Result type for any operation that may return a `ModelPerformanceError`.
pub type ModelPerfResult<T> = Result<T, ModelPerformanceError>;

#[derive(Debug, Error, PartialEq)]
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
    #[error("Unsupported drift type")]
    UnsupportedDriftType,
    #[error("Operation not supported in current drift mode")]
    UnsupportedOperation,
    #[error("Configuration not supported in current drift mode")]
    UnsupportedConfig,
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
    #[error("KolmogorovSmirnov not present")]
    KolmogorovSmirnov,
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

#[cfg(test)]
mod tests {
    use super::*;

    // --- BiasError display ---

    #[test]
    fn bias_error_display_no_facet_deviation() {
        assert_eq!(
            BiasError::NoFacetDeviation.to_string(),
            "No deviation in behavior between facets"
        );
    }

    #[test]
    fn bias_error_display_no_segmentation_parameters() {
        assert_eq!(
            BiasError::NoSegmentationParameters.to_string(),
            "No segmentation parameters provided"
        );
    }

    #[test]
    fn bias_error_display_ambiguous_segmentation_parameters() {
        assert_eq!(
            BiasError::AmbiguousSegmentationParameters.to_string(),
            "Ambiguous segmentation parameters"
        );
    }

    #[test]
    fn bias_error_display_data_length_error() {
        assert_eq!(
            BiasError::DataLengthError.to_string(),
            "All data arrays must have equal length and be at least lenght 1"
        );
    }

    // --- BiasError PartialEq ---

    #[test]
    fn bias_error_eq() {
        assert_eq!(BiasError::NoFacetDeviation, BiasError::NoFacetDeviation);
        assert_ne!(BiasError::NoFacetDeviation, BiasError::DataLengthError);
    }

    // --- From<ModelPerformanceError> for BiasError ---

    #[test]
    fn model_perf_invalid_data_converts_to_no_facet_deviation() {
        let bias_err = BiasError::from(ModelPerformanceError::InvalidData);
        assert_eq!(bias_err, BiasError::NoFacetDeviation);
    }

    #[test]
    fn model_perf_empty_data_converts_to_data_length_error() {
        let bias_err = BiasError::from(ModelPerformanceError::EmptyDataVector);
        assert_eq!(bias_err, BiasError::DataLengthError);
    }

    #[test]
    #[should_panic]
    fn model_perf_other_variant_panics_on_bias_conversion() {
        let _ = BiasError::from(ModelPerformanceError::DataVectorLengthMismatch);
    }

    // --- From<BiasError> for ModelPerformanceError ---

    #[test]
    fn bias_error_converts_to_model_perf_bias_error_variant() {
        let model_err = ModelPerformanceError::from(BiasError::NoFacetDeviation);
        assert!(matches!(
            model_err,
            ModelPerformanceError::BiasError(BiasError::NoFacetDeviation)
        ));
    }

    #[test]
    fn bias_error_display_survives_round_trip_via_model_perf() {
        let original = BiasError::DataLengthError;
        let wrapped = ModelPerformanceError::from(original);
        assert!(wrapped.to_string().contains("DataLengthError"));
    }

    // --- DriftError display ---

    #[test]
    fn drift_error_display_empty_runtime() {
        assert_eq!(
            DriftError::EmptyRuntimeData.to_string(),
            "Data used for runtime drift analysis must be non empty"
        );
    }

    #[test]
    fn drift_error_display_empty_baseline() {
        assert_eq!(
            DriftError::EmptyBaselineData.to_string(),
            "Baseline data must be non empty"
        );
    }

    #[test]
    fn drift_error_display_nan_value() {
        assert_eq!(
            DriftError::NaNValueError.to_string(),
            "NaN values are not supported"
        );
    }

    #[test]
    fn drift_error_display_unsupported_drift_type() {
        assert_eq!(
            DriftError::UnsupportedDriftType.to_string(),
            "Unsupported drift type"
        );
    }

    #[test]
    fn drift_error_display_unsupported_config() {
        assert_eq!(
            DriftError::UnsupportedConfig.to_string(),
            "Configuration not supported in current drift mode"
        );
    }

    // --- InvalidMetricError display ---

    #[test]
    fn invalid_metric_error_display_includes_metric_names() {
        let err = InvalidMetricError::DataBiasMetricError(vec![
            "bad_metric".to_string(),
            "another".to_string(),
        ]);
        let msg = err.to_string();
        assert!(msg.contains("bad_metric"));
        assert!(msg.contains("another"));
    }

    #[test]
    fn invalid_metric_error_model_bias_display_includes_metric_names() {
        let err = InvalidMetricError::ModelBiasMetricError(vec!["unknown_metric".to_string()]);
        assert!(err.to_string().contains("unknown_metric"));
    }

    // --- ModelPerformanceError display ---

    #[test]
    fn model_perf_error_display_empty_data_vector() {
        assert_eq!(
            ModelPerformanceError::EmptyDataVector.to_string(),
            "Empty data vectors"
        );
    }

    #[test]
    fn model_perf_error_display_length_mismatch() {
        assert_eq!(
            ModelPerformanceError::DataVectorLengthMismatch.to_string(),
            "Data vectors must be equal length"
        );
    }

    #[test]
    fn model_perf_error_display_bias_error_wraps_inner_message() {
        let err = ModelPerformanceError::BiasError(BiasError::NoFacetDeviation);
        assert!(err.to_string().contains("NoFacetDeviation"));
    }

    // --- DataBiasRuntimeError display ---

    #[test]
    fn data_bias_runtime_error_display_class_imbalance() {
        assert_eq!(
            DataBiasRuntimeError::ClassImbalance.to_string(),
            "ClassImbalance not present"
        );
    }

    #[test]
    fn data_bias_runtime_error_display_lp_norm() {
        assert_eq!(
            DataBiasRuntimeError::LpNorm.to_string(),
            "LpNorm not present"
        );
    }

    // --- ModelBiasRuntimeError display ---

    #[test]
    fn model_bias_runtime_error_display_disparate_impact() {
        assert_eq!(
            ModelBiasRuntimeError::DisparateImpact.to_string(),
            "DisparateImpact not present"
        );
    }

    #[test]
    fn model_bias_runtime_error_display_generalized_entropy() {
        assert_eq!(
            ModelBiasRuntimeError::GeneralizedEntropy.to_string(),
            "GeneralizedEntropy not present"
        );
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
                | DriftError::NaNValueError
                | DriftError::UnsupportedDriftType
                | DriftError::UnsupportedConfig
                | DriftError::UnsupportedOperation => {
                    exceptions::PyValueError::new_err(err_message)
                }
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
