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
pub(crate) fn kl_divergence_drift(baseline_hist: &[f64], runtime_bins: &[f64], n: f64) -> f64 {
    debug_assert_eq!(runtime_bins.len(), baseline_hist.len());
    let eps = *STABILITY_EPS;

    baseline_hist
        .iter()
        .zip(runtime_bins.iter())
        .map(|(bl, rt)| {
            let dist_rt = (rt + eps) / n;
            let dist_bl = (bl + eps).max(eps);
            dist_bl * (dist_bl / dist_rt).max(eps).ln()
        })
        .sum()
}

pub trait StreamingKlDivergenceDrift {
    fn kl_divergence_drift(&self) -> Result<f64, DriftError>;
}

pub trait StreamingPopulationStabilityIndexDrift {
    fn psi_drift(&self) -> Result<f64, DriftError>;
}

pub trait ContinuousPSIDrift {
    fn psi_drift(&mut self, runtime_slice: &[f64]) -> Result<f64, DriftError>;
}

pub trait ContinuousKLDivergenceDrift {
    fn psi_drift(&self, runtime_slice: &[f64]) -> Result<f64, DriftError>;
}

pub trait CategoricalPSIDrift {
    fn psi_drift<S>(&mut self, runtime_slice: &[S]) -> Result<f64, DriftError>
    where
        S: StringLike;
}

pub trait CategoricalKLDivergenceDrift {
    fn psi_drift<S>(&self, runtime_slice: &[S]) -> Result<f64, DriftError>;
}
