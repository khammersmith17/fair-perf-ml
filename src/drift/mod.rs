pub mod baseline;
/// The data drift module is intended to provide types that can serve to identify drift in the
/// distribution of some dataset, for example  a feature dataset, or composed to track drift across
/// and entire feature set. This module can also serve to provide proxies for model drift. By
/// leveraging the same techinques used to identify data drfit, model drift can also be
/// approximated. When the distribution of inference scores drifts significantly, that is probably
/// a decent sign for a deeper investigation.
pub mod data_drift;
pub mod distribution;
pub mod drift_metrics;
#[cfg(feature = "python")]
pub(crate) mod python_impl;
use crate::errors::DriftError;
use data_drift::{CategoricalDataDrift, ContinuousDataDrift};
use distribution::QuantileType;
use drift_metrics::DataDriftType;
use std::hash::Hash;

const DEFAULT_STREAM_FLUSH_CADENCE: u64 = 3600 * 24;
const DEFAULT_MAX_STREAM_SIZE: u64 = 1_000_000_u64;
const DEFAULT_DECAY_HALF_LIFE: u64 = 86400; // Defaul half life 1 day

pub fn compute_drift_continuous_distribution(
    baseline_distribution: &[f64],
    candidate_distribution: &[f64],
    drift_metrics: &[DataDriftType],
    quantile_type: Option<QuantileType>,
) -> Result<Vec<f64>, DriftError> {
    let mut drift_container =
        ContinuousDataDrift::new_from_baseline(quantile_type, baseline_distribution)?;
    let drift_res =
        drift_container.compute_drift_multiple_criteria(candidate_distribution, drift_metrics)?;
    Ok(drift_res)
}

pub fn compute_drift_categorical_distribution<T: Hash + Ord + Clone>(
    baseline_distribution: &[T],
    candidate_distribution: &[T],
    drift_metrics: &[DataDriftType],
) -> Result<Vec<f64>, DriftError> {
    let mut drift_container = CategoricalDataDrift::new(baseline_distribution)?;
    let drift_res =
        drift_container.compute_drift_multiple_criteria(candidate_distribution, drift_metrics)?;
    Ok(drift_res)
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- compute_drift_continuous_distribution ---

    #[test]
    fn continuous_no_drift_returns_near_zero() {
        let baseline = [1.0, 2.0, 3.0, 4.0, 5.0];
        let candidate = [1.0, 2.0, 3.0, 4.0, 5.0];
        let metrics = [DataDriftType::PopulationStabilityIndex];

        let result =
            compute_drift_continuous_distribution(&baseline, &candidate, &metrics, None).unwrap();

        assert_eq!(result.len(), 1);
        assert!(result[0].abs() < 1e-9);
    }

    #[test]
    fn continuous_shifted_distribution_detects_drift() {
        let baseline = [1.0, 2.0, 3.0, 4.0, 5.0];
        let candidate = [20.0, 21.0, 22.0, 23.0, 24.0];
        let metrics = [DataDriftType::PopulationStabilityIndex];

        let result =
            compute_drift_continuous_distribution(&baseline, &candidate, &metrics, None).unwrap();

        assert_eq!(result.len(), 1);
        assert!(result[0] > 0.5);
    }

    #[test]
    fn continuous_multiple_metrics_returns_one_value_per_metric() {
        let baseline = [1.0, 2.0, 3.0, 4.0, 5.0];
        let candidate = [1.0, 2.0, 3.0, 4.0, 5.0];
        let metrics = [
            DataDriftType::PopulationStabilityIndex,
            DataDriftType::JensenShannon,
            DataDriftType::KullbackLeibler,
            DataDriftType::WassersteinDistance,
        ];

        let result =
            compute_drift_continuous_distribution(&baseline, &candidate, &metrics, None).unwrap();

        assert_eq!(result.len(), 4);
        for score in &result {
            assert!(score.abs() < 1e-9);
        }
    }

    #[test]
    fn continuous_explicit_quantile_type_is_accepted() {
        let baseline = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let candidate = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let metrics = [DataDriftType::JensenShannon];

        let result = compute_drift_continuous_distribution(
            &baseline,
            &candidate,
            &metrics,
            Some(QuantileType::Sturges),
        )
        .unwrap();

        assert_eq!(result.len(), 1);
        assert!(result[0].abs() < 1e-9);
    }

    #[test]
    fn continuous_empty_baseline_returns_error() {
        let baseline: &[f64] = &[];
        let candidate = [1.0, 2.0];
        let metrics = [DataDriftType::PopulationStabilityIndex];

        let result =
            compute_drift_continuous_distribution(baseline, &candidate, &metrics, None);

        assert!(result.is_err());
    }

    // --- compute_drift_categorical_distribution ---

    #[test]
    fn categorical_no_drift_returns_near_zero() {
        let baseline = ["a", "b", "a", "c", "b", "a"];
        let candidate = ["a", "b", "a", "c", "b", "a"];
        let metrics = [DataDriftType::PopulationStabilityIndex];

        let result =
            compute_drift_categorical_distribution(&baseline, &candidate, &metrics).unwrap();

        assert_eq!(result.len(), 1);
        assert!(result[0].abs() < 1e-9);
    }

    #[test]
    fn categorical_shifted_distribution_detects_drift() {
        let baseline = ["a", "a", "a", "a", "b"];
        let candidate = ["b", "b", "b", "b", "a"];
        let metrics = [DataDriftType::PopulationStabilityIndex];

        let result =
            compute_drift_categorical_distribution(&baseline, &candidate, &metrics).unwrap();

        assert_eq!(result.len(), 1);
        assert!(result[0] > 0.1);
    }

    #[test]
    fn categorical_multiple_metrics_returns_one_value_per_metric() {
        let baseline = ["x", "y", "x", "z", "y"];
        let candidate = ["x", "y", "x", "z", "y"];
        let metrics = [
            DataDriftType::PopulationStabilityIndex,
            DataDriftType::JensenShannon,
            DataDriftType::KullbackLeibler,
            DataDriftType::WassersteinDistance,
        ];

        let result =
            compute_drift_categorical_distribution(&baseline, &candidate, &metrics).unwrap();

        assert_eq!(result.len(), 4);
        for score in &result {
            assert!(score.abs() < 1e-9);
        }
    }

    #[test]
    fn categorical_unseen_label_in_candidate_is_handled() {
        let baseline = ["a", "b", "a", "b"];
        let candidate = ["a", "b", "c", "d"];
        let metrics = [DataDriftType::JensenShannon];

        // unseen labels should be bucketed into the overflow bin, not panic
        let result =
            compute_drift_categorical_distribution(&baseline, &candidate, &metrics).unwrap();

        assert_eq!(result.len(), 1);
        assert!(result[0] >= 0.0);
    }

    #[test]
    fn categorical_integer_labels_are_supported() {
        let baseline = [1i32, 2, 1, 3, 2, 1];
        let candidate = [1i32, 2, 1, 3, 2, 1];
        let metrics = [DataDriftType::PopulationStabilityIndex];

        let result =
            compute_drift_categorical_distribution(&baseline, &candidate, &metrics).unwrap();

        assert_eq!(result.len(), 1);
        assert!(result[0].abs() < 1e-9);
    }

    #[test]
    fn categorical_empty_baseline_returns_error() {
        let baseline: &[&str] = &[];
        let candidate = ["a", "b"];
        let metrics = [DataDriftType::PopulationStabilityIndex];

        let result = compute_drift_categorical_distribution(baseline, &candidate, &metrics);

        assert!(result.is_err());
    }
}
