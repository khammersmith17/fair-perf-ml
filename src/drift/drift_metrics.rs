use super::StringLike;
use crate::errors::DriftError;
use crate::metrics::STABILITY_EPS;

//define traits to use the continuous and discrete drift bins, where getting the implementation of
//a particular metric is declared via trait methods. This will only implement the logic on the
//class when the user it in scope. Traits will be implemented where the bin types are implemented

// Compute psi on runtime and baseline bins. Element wise distance for each bucket with a
// sum reduction.
#[inline]
pub(crate) fn compute_psi(baseline_hist: &[f64], runtime_bins: &[f64], n: f64) -> f64 {
    // validate that rt and baseline bins are of same length
    debug_assert_eq!(runtime_bins.len(), baseline_hist.len());
    let eps = *STABILITY_EPS;

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
    let eps = *STABILITY_EPS;

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
    let eps = *STABILITY_EPS;

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
