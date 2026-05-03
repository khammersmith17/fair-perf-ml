use super::{distribution::QuantileType, opt};
use crate::errors::DriftError;
use ahash::{HashMap, HashMapExt};
use std::collections::BTreeMap;
use std::hash::Hash;

#[derive(Debug, Default, PartialEq, Clone)]
pub struct ContinuousBinEdges {
    bin_edges: Vec<f64>,
    n_bins: usize,
}

impl ContinuousBinEdges {
    pub fn new_from_parts(bin_edges: Vec<f64>) -> ContinuousBinEdges {
        let n_bins = bin_edges.len() + 2;
        ContinuousBinEdges { bin_edges, n_bins }
    }
    /// Assumes data is sorted.
    pub fn new_from_dataset_with_quantile_type(
        dataset: &[f64],
        quantile_type: QuantileType,
    ) -> ContinuousBinEdges {
        let n_bins = quantile_type.compute_num_bins(dataset);
        ContinuousBinEdges::new_from_dataset_with_bin_count(dataset, n_bins)
    }

    /// Assumes data is sorted.
    pub fn new_from_dataset_with_bin_count(dataset: &[f64], n_bins: usize) -> ContinuousBinEdges {
        /*
         * - Bin edges will be of size num_bins - 2.
         * - The outer bins, or tail bins in the distribution will be reserved for values observed in the
         *  distribution that fall outsde the bounds of the baseline distribution.
         *  - Bin/quantile size will have its "step" size determined by evenly diving the difference
         *  between the max and min of the distribution and dividing by the number of bins - 2.
         *  - A value is assigned to a particular quantile if left <= value < right, otherwise it will
         *  be assigned to one of the tail quantile bins.
         *  - Each bin has a constant step size.
         * */
        let mut bin_edges = vec![0_f64; n_bins - 2];
        let n = dataset.len();
        let n_0 = dataset[0];
        let bin_step = (dataset[n - 1] - n_0) / n as f64;
        let mut edge_value = n_0;

        for edge in bin_edges.iter_mut() {
            *edge = edge_value;
            edge_value += bin_step;
        }

        ContinuousBinEdges { bin_edges, n_bins }
    }

    pub(crate) fn n_bins(&self) -> usize {
        self.n_bins
    }

    #[inline]
    fn left_bin_edge(&self) -> f64 {
        self.bin_edges[0]
    }

    #[inline]
    fn right_bin_edge(&self) -> f64 {
        // bin_edges.len == n_bins - 2
        self.bin_edges[self.len() - 1]
    }

    pub(crate) fn len(&self) -> usize {
        self.bin_edges.len()
    }

    pub(crate) fn export_edges(&self) -> Vec<f64> {
        self.bin_edges.clone()
    }

    pub(crate) fn take_edges(self) -> Vec<f64> {
        let Self { bin_edges, .. } = self;
        bin_edges
    }

    #[inline]
    pub fn resolve_bin(&self, sample: f64) -> usize {
        if sample < self.left_bin_edge() {
            return 0_usize;
        }

        if sample > self.right_bin_edge() {
            return self.n_bins - 1;
        }
        // find "pivot" point
        // ie the bin where value >= left and < right
        // this incorrectly misses the left and right edge currently
        // as these values would not created a parition within the edges
        let i = self.bin_edges.partition_point(|edge| sample >= *edge);
        i.clamp(0, self.n_bins - 1)
    }
}

/// Utility wrapper type to encapsulate bin resolution when approximating an entire dataset.
pub struct CategoricalBinEdges<'a, T: Hash + Ord + Clone>(pub &'a ahash::HashMap<T, usize>);

impl<T: Hash + Ord + Clone> CategoricalBinEdges<'_, T> {
    pub fn resolve_bin<Q>(&self, key: &Q) -> usize
    where
        T: std::borrow::Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        if let Some(idx) = self.0.get(key) {
            *idx
        } else {
            self.n_bins() - 1
        }
    }

    pub(crate) fn n_bins(&self) -> usize {
        self.0.len() + 1
    }
}

pub(crate) fn compute_dataset_from_bins_continuous(
    dataset: &[f64],
    edges: &ContinuousBinEdges,
) -> Vec<f64> {
    let thread_count = opt::get_thread_count(dataset.len());
    if thread_count > 1 {
        opt::continuous::parallel_approx_dataset(dataset, edges, thread_count)
    } else {
        compute_dataset_from_bins_continuous_seq(dataset, edges)
    }
}

fn compute_dataset_from_bins_continuous_seq(
    dataset: &[f64],
    edges: &ContinuousBinEdges,
) -> Vec<f64> {
    let mut hist = vec![0_f64; edges.n_bins()];
    dataset
        .iter()
        .for_each(|e| hist[edges.resolve_bin(*e)] += 1_f64);
    hist
}

pub(crate) fn compute_dataset_from_bins_categorical_parallel<'a, T: Hash + Ord + Clone + Sync>(
    dataset: &'a [T],
    edges: &'a CategoricalBinEdges<T>,
) -> Vec<f64> {
    opt::categorical::parallel_approx_dataset(dataset, edges)
}

pub(crate) fn compute_dataset_from_bins_categorical<'a, T: Hash + Ord + Clone>(
    dataset: &'a [T],
    edges: &'a CategoricalBinEdges<T>,
) -> Vec<f64> {
    let mut hist = vec![0_f64; edges.n_bins()];
    dataset
        .iter()
        .for_each(|e| hist[edges.resolve_bin(e)] += 1_f64);
    hist
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
