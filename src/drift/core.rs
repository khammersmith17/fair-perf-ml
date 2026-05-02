use super::baseline::ContinuousBinEdges;
use super::opt;
use crate::errors::DriftError;
use ahash::{HashMap, HashMapExt};
use std::collections::BTreeMap;
use std::hash::Hash;

pub(crate) fn compute_dataset_from_bins_continuous(
    dataset: &[f64],
    edges: &ContinuousBinEdges,
) -> Vec<f64> {
    opt::continuous::parallel_approx_dataset(dataset, edges)
}

// Take the baseline bin counts and compute the proportional bin sizes based on total population
// size.
#[inline]
pub(crate) fn compute_new_hist_prob(
    num_items: usize,
    hist: &[f64],
) -> Result<Vec<f64>, DriftError> {
    let total_n = num_items as f64;
    if total_n == 0_f64 {
        return Err(DriftError::EmptyRuntimeData);
    }
    let bl_hist = hist.iter().map(|n| *n / total_n).collect::<Vec<f64>>();
    Ok(bl_hist)
}

/// Defines the lookup map for categorical fields, and constructs the baseline histogram for drift
/// at "runtime".
pub(crate) fn categorical_derive_baseline_state<T: Hash + Ord + Clone>(
    baseline_dataset: &[T],
) -> Result<(HashMap<T, usize>, Vec<f64>), DriftError> {
    if baseline_dataset.is_empty() {
        return Err(DriftError::EmptyBaselineData);
    }
    let n = baseline_dataset.len() as f64;

    let mut initial_bins: BTreeMap<T, f64> = BTreeMap::new();
    for cat in baseline_dataset.iter() {
        if let Some(count) = initial_bins.get_mut(cat) {
            *count += 1_f64;
        } else {
            initial_bins.insert(cat.clone(), 1_f64);
        }
    }

    // Preallocate space for cardinatity of the dataset + 1
    // The additional bin is reserved for data values not observed in the baseline dataset
    let mut baseline_bins = vec![0_f64; initial_bins.len() + 1_usize];
    let mut idx_map: HashMap<T, usize> = HashMap::with_capacity(initial_bins.len());
    for (i, (key, count)) in initial_bins.into_iter().enumerate() {
        idx_map.insert(key, i);
        baseline_bins[i] = count / n;
    }
    Ok((idx_map, baseline_bins))
}

#[cfg(test)]
mod core_drift_test {
    use super::*;
    #[test]
    fn test_new_hist_prob() {
        let bl_hist = vec![10.0, 20.0, 30.0, 40.0];
        let base: Vec<f64> = vec![0.10, 0.20, 0.30, 0.40];
        let test_bins = compute_new_hist_prob(100, &bl_hist).unwrap();
        assert_eq!(base, test_bins);
    }
}
