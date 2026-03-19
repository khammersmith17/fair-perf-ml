use crate::metrics::get_stability_eps;

// Enum to be used in DriftContainer trait when drift metric computation args differ between the
// Continuous and Categorical DriftContainers.
#[derive(Debug, PartialEq)]
pub(crate) enum DriftContainerType {
    Continuous,
    Categorical,
}

#[derive(Debug, PartialEq, Copy, Clone)]
#[non_exhaustive]
pub enum DataDriftType {
    JensenShannon,
    PopulationStabilityIndex,
    WassersteinDistance,
    KullbackLeibler,
}

impl TryFrom<&str> for DataDriftType {
    type Error = crate::errors::DriftError;
    fn try_from(val: &str) -> Result<Self, Self::Error> {
        match val {
            "JensenShannon" => Ok(Self::JensenShannon),
            "PopulationStabilityIndex" => Ok(Self::PopulationStabilityIndex),
            "WassersteinDistance" => Ok(Self::WassersteinDistance),
            "KullbackLeibler" => Ok(Self::KullbackLeibler),
            _ => Err(Self::Error::UnsupportedDriftType),
        }
    }
}

/// Trait that defines common behavior required to compute drift metrics.
/// When additional drift criteria is added, this may be built upon. Behavior should not be removed
/// for backward compatibility later.
pub(crate) trait DriftContainer {
    fn get_baseline_hist(&self) -> &[f64];

    fn get_runtime_bins(&self) -> &[f64];

    fn num_examples(&self) -> f64;

    fn get_bin_edges(&self) -> Option<&[f64]>;

    fn container_type(&self) -> DriftContainerType;
}

/// Global drift computation that allows a shared interface between all drift types.
/// Additional drift criteria added later should all dispatch drift through this method.
pub(crate) fn global_compute_drift<T: DriftContainer>(
    drift_container: &T,
    drift_type: DataDriftType,
) -> f64 {
    match drift_type {
        DataDriftType::PopulationStabilityIndex => compute_psi(
            drift_container.get_baseline_hist(),
            drift_container.get_runtime_bins(),
            drift_container.num_examples(),
        ),
        DataDriftType::KullbackLeibler => compute_kl_divergence_drift(
            drift_container.get_baseline_hist(),
            drift_container.get_runtime_bins(),
            drift_container.num_examples(),
        ),
        DataDriftType::JensenShannon => compute_jensen_shannon_divergence_drift(
            drift_container.get_baseline_hist(),
            drift_container.get_runtime_bins(),
            drift_container.num_examples(),
        ),
        DataDriftType::WassersteinDistance => match drift_container.container_type() {
            DriftContainerType::Continuous => continuous_wasserstein_distance(
                drift_container.get_baseline_hist(),
                drift_container.get_runtime_bins(),
                drift_container.get_bin_edges().unwrap(),
                drift_container.num_examples(),
            ),
            DriftContainerType::Categorical => categorical_wasserstein_distance(
                drift_container.get_baseline_hist(),
                drift_container.get_runtime_bins(),
                drift_container.num_examples(),
            ),
        },
    }
}

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
