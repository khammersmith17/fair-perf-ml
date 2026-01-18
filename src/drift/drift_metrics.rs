use super::StringLike;
use crate::errors::DriftError;
use crate::metrics::get_stability_eps;

//define traits to use the continuous and discrete drift bins, where getting the implementation of
//a particular metric is declared via trait methods. This will only implement the logic on the
//class when the user it in scope. Traits will be implemented where the bin types are implemented

// Compute psi on runtime and baseline bins. Element wise distance for each bucket with a
// sum reduction.
#[inline]
pub(crate) fn compute_psi(baseline_hist: &[f64], runtime_bins: &[f64], n: f64) -> f64 {
    // validate that rt and baseline bins are of same length
    debug_assert_eq!(runtime_bins.len(), baseline_hist.len());
    let eps = get_stability_eps();

    baseline_hist
        .iter()
        .zip(runtime_bins.iter())
        .map(|(bl, rt)| {
            let b = (bl + eps).max(eps);
            let r = ((rt + eps) / n).max(eps);
            (b - r) * (b / r).ln()
        })
        .sum()
}

#[inline]
pub(crate) fn compute_kl_divergence_drift(
    baseline_hist: &[f64],
    runtime_bins: &[f64],
    n: f64,
) -> f64 {
    // validate that rt and baseline bins are of same length
    debug_assert_eq!(runtime_bins.len(), baseline_hist.len());
    let eps = get_stability_eps();

    baseline_hist
        .iter()
        .zip(runtime_bins.iter())
        .map(|(bl, rt)| {
            let dist_rt = (*rt + eps) / n;
            let dist_bl = (*bl + eps).max(eps);
            dist_bl * (dist_bl / dist_rt).max(eps).ln()
        })
        .sum()
}

#[inline]
pub(crate) fn compute_jensen_shannon_divergence_drift(
    baseline_hist: &[f64],
    runtime_bins: &[f64],
    n: f64,
) -> f64 {
    // validate that rt and baseline bins are of same length
    debug_assert_eq!(runtime_bins.len(), baseline_hist.len());
    let eps = get_stability_eps();

    let mut js = 0_f64;
    let half_fac = 0.5_f64;

    for (bl, rt) in baseline_hist.iter().zip(runtime_bins.iter()) {
        let p = (bl + eps).max(eps);
        let q = ((rt / n) + eps).max(eps);
        let m = (p + q) * half_fac;

        js += (half_fac * p * (p / m).ln()) + (half_fac * q * (q / m).ln());
    }

    js / 2_f64.ln()
}

#[inline]
pub(crate) fn continuous_wasserstein_distance(
    baseline_hist: &[f64],
    runtime_bins: &[f64],
    bin_edges: &[f64],
    n: f64,
) -> f64 {
    debug_assert_eq!(runtime_bins.len(), baseline_hist.len());
    let n_bin_edges = bin_edges.len();
    debug_assert_eq!(n_bin_edges, runtime_bins.len() + 1);

    let eps = get_stability_eps();
    let mut w_dist = 0_f64;

    for (i, (bl, rt)) in baseline_hist.iter().zip(runtime_bins.iter()).enumerate() {
        let bin_width = bin_edges[i + 1] - bin_edges[i];
        let p = (bl + eps).max(eps);
        let q = ((rt / n) + eps).max(eps);

        w_dist += (p - q).abs() * bin_width;
    }

    w_dist / (bin_edges[n_bin_edges - 1] - bin_edges[0])
}

#[inline]
pub(crate) fn categorical_wasserstein_distance(
    baseline_hist: &[f64],
    runtime_bins: &[f64],
    n: f64,
) -> f64 {
    // bins are effectively unit width for categorical distributions
    // this effectively turns into total variation distance
    debug_assert_eq!(runtime_bins.len(), baseline_hist.len());

    let eps = get_stability_eps();
    let mut w_dist = 0_f64;

    for (bl, rt) in baseline_hist.iter().zip(runtime_bins.iter()) {
        let p = (bl + eps).max(eps);
        let q = ((rt / n) + eps).max(eps);

        w_dist += (p - q).abs();
    }

    w_dist * 0.5_f64
}

#[allow(unused)]
use super::data_drift::CategoricalDataDrift;
#[allow(unused)]
use super::data_drift::ContinuousDataDrift;

/// KL Divergence implementation for streaming types. Bringing this trait into scope will provided
/// the implementation for the associated streaming type.
pub trait StreamingKlDivergenceDrift {
    fn kl_divergence_drift(&self) -> Result<f64, DriftError>;
}

/// Provides KL Divergence implementation for [`ContinuousDataDrift`].
pub trait ContinuousKlDivergenceDrift {
    fn kl_divergence_drift(&mut self, runtime_slice: &[f64]) -> Result<f64, DriftError>;
}

/// Provides KL Divergence implementation for [`CategoricalDataDrift`].
pub trait CategoricalKlDivergenceDrift {
    fn kl_divergence_drift<S: StringLike>(
        &mut self,
        runtime_slice: &[S],
    ) -> Result<f64, DriftError>;
}

/// Population Stability Index implementation for streaming types. Bringing this trait into scope will provided
/// the implementation for the associated streaming type.
pub trait StreamingPopulationStabilityIndexDrift {
    fn psi_drift(&self) -> Result<f64, DriftError>;
}

/// Provides Population Stability Index implementation for [`ContinuousDataDrift`].
pub trait ContinuousPSIDrift {
    fn psi_drift(&mut self, runtime_slice: &[f64]) -> Result<f64, DriftError>;
}

/// Provides Population Stability Index implementation for [`CategoricalDataDrift`].
pub trait CategoricalPSIDrift {
    fn psi_drift<S: StringLike>(&mut self, runtime_slice: &[S]) -> Result<f64, DriftError>;
}

/// Provides the implementation of Jensen-Divergence drift for streaming drift types.
pub trait StreamingJensenShannonDivergenceDrift {
    fn js_drift(&self) -> Result<f64, DriftError>;
}

/// Provides the implementation of Jensen-Divergence drift for [`ContinuousDataDrift`].
pub trait ContinuousJensenShannonDivergenceDrift {
    fn js_drift(&mut self, runtime_slice: &[f64]) -> Result<f64, DriftError>;
}

/// Provides the implementation of Jensen-Divergence drift for [`CategoricalDataDrift`].
pub trait CategoricalJensenShannonDivergenceDrift {
    fn js_drift<S: StringLike>(&mut self, runtime_slice: &[S]) -> Result<f64, DriftError>;
}

/// Provides the implementation of Wasserstein distance drift for streaming drift types.
pub trait StreamingWassersteinDistance {
    fn wasserstein_distance(&self) -> Result<f64, DriftError>;
}

/// Provides the implementation of Wasserstein distance drift for [`ContinuousDataDrift`].
pub trait ContinuousWassersteinDistance {
    fn wasserstein_distance(&mut self, runtime_data: &[f64]) -> Result<f64, DriftError>;
}

/// Provides the implementation of Wasserstein distance drift for [`CategoricalDataDrift`].
pub trait CategoricalWassersteinDistance {
    fn wasserstein_distance<S: StringLike>(
        &mut self,
        runtime_data: &[S],
    ) -> Result<f64, DriftError>;
}
