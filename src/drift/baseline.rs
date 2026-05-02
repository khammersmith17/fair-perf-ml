use super::{
    core,
    distribution::{QuantileType, MIN_BIN_CLAMP},
    export::{CategoricalDriftBaselineExport, ContinuousDriftBaselineExport},
};
use crate::errors::{DriftError, DriftExportError};
use ahash::{HashMap, HashMapExt};
use std::borrow::Borrow;
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};
use std::hash::Hash;

// Take the baseline bin counts and compute the proportional bin sizes based on total population
// size.
#[inline]
fn compute_new_hist_prob(num_items: usize, hist: &[f64]) -> Result<Vec<f64>, DriftError> {
    let total_n = num_items as f64;
    if total_n == 0_f64 {
        return Err(DriftError::EmptyRuntimeData);
    }
    let bl_hist = hist.iter().map(|n| *n / total_n).collect::<Vec<f64>>();
    Ok(bl_hist)
}

#[derive(Debug, Default, PartialEq, Clone)]
pub(crate) struct ContinuousBinEdges {
    bin_edges: Vec<f64>,
    n_bins: usize,
}

impl ContinuousBinEdges {
    /// Assumes data is sorted.
    fn new_from_dataset_with_quantile_type(
        dataset: &[f64],
        quantile_type: QuantileType,
    ) -> ContinuousBinEdges {
        let n_bins = quantile_type.compute_num_bins(dataset);
        ContinuousBinEdges::new_from_dataset_with_bin_count(dataset, n_bins)
    }

    /// Assumes data is sorted.
    fn new_from_dataset_with_bin_count(dataset: &[f64], n_bins: usize) -> ContinuousBinEdges {
        /*
         * - Bin edges will be of size num_bins - 2.
         * - The outer bins, or tail bins in the distribution will be reserved for values observed in the
         *  distribution that fall outsde the bounds of the baseline distribution.
         *  - Bin/quantile size will have its "step" size determined by evenly diving the difference
         *  between the max and min of the distribution and dividing by the number of bins - 2.
         *  - A value is assigned to a particular quantile if left <= value < right, otherwise it will
         *  be assigned to one of the tail quantile bins.
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

    fn export_edges(&self) -> Vec<f64> {
        self.bin_edges.clone()
    }

    #[inline]
    pub(crate) fn resolve_bin(&self, sample: f64) -> usize {
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

// Break out baseline to have shared logic between the discrete and the streaming variants of drift
// utilities.
// Also allows for more elegant composition of different usage
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct BaselineContinuousBins {
    pub bin_edges: ContinuousBinEdges,
    pub baseline_hist: Vec<f64>,
    quantile_type: QuantileType,
}

impl TryFrom<ContinuousDriftBaselineExport> for BaselineContinuousBins {
    type Error = DriftExportError;
    fn try_from(export: ContinuousDriftBaselineExport) -> Result<Self, Self::Error> {
        let ContinuousDriftBaselineExport {
            bin_edges: raw_bin_edges,
            baseline_hist,
            quantile_type,
        } = export;
        let n_bins = baseline_hist.len();
        if raw_bin_edges.len() != n_bins - 2 || n_bins < MIN_BIN_CLAMP {
            return Err(DriftExportError::InvalidDataShape);
        }

        let bin_edges = ContinuousBinEdges {
            bin_edges: raw_bin_edges,
            n_bins,
        };

        Ok(BaselineContinuousBins {
            bin_edges,
            baseline_hist,
            quantile_type,
        })
    }
}

impl From<BaselineContinuousBins> for ContinuousDriftBaselineExport {
    fn from(baseline: BaselineContinuousBins) -> ContinuousDriftBaselineExport {
        let BaselineContinuousBins {
            bin_edges: bin_edges_outer,
            baseline_hist,
            quantile_type,
            ..
        } = baseline;

        ContinuousDriftBaselineExport {
            bin_edges: bin_edges_outer.bin_edges,
            baseline_hist,
            quantile_type,
        }
    }
}

impl BaselineContinuousBins {
    pub(crate) fn new_from_export(
        export: ContinuousDriftBaselineExport,
    ) -> Result<BaselineContinuousBins, DriftExportError> {
        Self::try_from(export)
    }

    // Constructor on a baseline dataset. Allocates then hyrdates with the provided baseline
    // dataset.
    pub(crate) fn new(
        baseline_data: &[f64],
        quantile_resolution: QuantileType,
    ) -> Result<BaselineContinuousBins, DriftError> {
        let sorted_baseline = Self::sort_baseline_data(baseline_data)?;
        let bin_edges = ContinuousBinEdges::new_from_dataset_with_quantile_type(
            &sorted_baseline,
            quantile_resolution,
        );

        let baseline_hist = compute_new_hist_prob(
            baseline_data.len(),
            &core::compute_dataset_from_bins_continuous(baseline_data, &bin_edges),
        )?;

        Ok(BaselineContinuousBins {
            bin_edges,
            baseline_hist,
            quantile_type: quantile_resolution,
        })
    }

    fn sort_baseline_data(data: &[f64]) -> Result<Vec<f64>, DriftError> {
        if data.len() <= 1 {
            return Err(DriftError::EmptyBaselineData);
        }

        // do not accept NaNs
        if data.iter().any(|value| value.is_nan()) {
            return Err(DriftError::NaNValueError);
        }

        let mut sorted_baseline = data.to_vec();
        sorted_baseline.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
        Ok(sorted_baseline)
    }

    pub(crate) fn n_bins(&self) -> usize {
        self.bin_edges.n_bins
    }

    pub fn export_bin_edges(&self) -> Vec<f64> {
        self.bin_edges.export_edges()
    }

    // Resolve the bin a particular data example falls into.
    #[inline]
    pub(crate) fn resolve_bin(&self, sample: f64) -> usize {
        self.bin_edges.resolve_bin(sample)
    }

    pub(crate) fn export_baseline(&self) -> Vec<f64> {
        self.baseline_hist.clone()
    }

    // call into init method
    pub(crate) fn reset(&mut self, baseline_data: &[f64]) -> Result<(), DriftError> {
        let sorted_baseline = Self::sort_baseline_data(baseline_data)?;
        self.bin_edges = ContinuousBinEdges::new_from_dataset_with_quantile_type(
            &sorted_baseline,
            self.quantile_type,
        );

        self.baseline_hist = compute_new_hist_prob(
            baseline_data.len(),
            &core::compute_dataset_from_bins_continuous(baseline_data, &self.bin_edges),
        )?;
        Ok(())
    }
}

/*
* Trait bounds here enforce that the categorical values must be hashable to be stored as keys in
* the lookup app, comparable, and
* */

// idx_map holds the bin for a particular data value.
// Baseline bins are the histogram generated on baseline data, and other label represents the
// "other" bucket for when a discrete value not seen in the baseline set is observed.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct BaselineCategoricalBins<T: Hash + Ord + Clone> {
    pub(crate) idx_map: HashMap<T, usize>,
    pub(crate) baseline_bins: Vec<f64>,
}

impl<T: Hash + Ord + Clone + serde::Serialize> TryInto<CategoricalDriftBaselineExport>
    for BaselineCategoricalBins<T>
{
    type Error = serde_json::Error;
    fn try_into(self) -> Result<CategoricalDriftBaselineExport, Self::Error> {
        let BaselineCategoricalBins {
            idx_map,
            baseline_bins: baseline_hist,
        } = self;

        let value_set: BTreeSet<T> = idx_map.into_iter().map(|(key, _)| key).collect();
        let mut baseline_values: Vec<serde_json::Value> = Vec::with_capacity(value_set.len());
        for value in value_set.into_iter() {
            baseline_values.push(serde_json::to_value(value)?);
        }

        Ok(CategoricalDriftBaselineExport {
            baseline_hist,
            baseline_values,
        })
    }
}

impl<T: Hash + Ord + Clone + serde::de::DeserializeOwned> TryFrom<CategoricalDriftBaselineExport>
    for BaselineCategoricalBins<T>
{
    type Error = DriftExportError;
    fn try_from(export: CategoricalDriftBaselineExport) -> Result<Self, Self::Error> {
        let CategoricalDriftBaselineExport {
            baseline_hist,
            baseline_values,
        } = export;

        if baseline_hist.len() - 1 != baseline_values.len() {
            return Err(DriftExportError::InvalidDataShape);
        }
        let mut labels: BTreeSet<T> = BTreeSet::new();

        for v in baseline_values.into_iter() {
            labels.insert(serde_json::from_value(v)?);
        }

        let idx_map: HashMap<T, usize> = labels
            .into_iter()
            .enumerate()
            .map(|(i, label)| (label, i))
            .collect();

        Ok(BaselineCategoricalBins {
            baseline_bins: baseline_hist,
            idx_map,
        })
    }
}

impl<T: Hash + Ord + Clone + serde::de::DeserializeOwned> BaselineCategoricalBins<T> {
    pub(crate) fn new_from_export(
        export: CategoricalDriftBaselineExport,
    ) -> Result<BaselineCategoricalBins<T>, DriftExportError> {
        Self::try_from(export)
    }
}

/// Defines the lookup map for categorical fields, and constructs the baseline histogram for drift
/// at "runtime".
fn categorical_derive_baseline_state<T: Hash + Ord + Clone>(
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

/*
* Each value present in the baseline dataset is mapped to a bin in the histogram Vec.
* The furthest right, ie len(set(baseline data)) index in the histogram Vec is reserved for
* observed values that were not part of the baseline set
* */

impl<T: Hash + Ord + Clone> BaselineCategoricalBins<T> {
    // bins and index map, allocated bins, fill histogram with counts.
    pub(crate) fn new(baseline_data: &[T]) -> Result<BaselineCategoricalBins<T>, DriftError> {
        let (idx_map, baseline_bins) = categorical_derive_baseline_state(baseline_data)?;
        Ok(BaselineCategoricalBins {
            idx_map,
            baseline_bins,
        })
    }

    /// Resolve the bin idx for a particular key, otherwise return out the bin reserved for the
    /// "other" bucket.
    pub(crate) fn resolve_bin<Q>(&self, key: &Q) -> usize
    where
        T: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        if let Some(idx) = self.idx_map.get(key) {
            *idx
        } else {
            self.baseline_bins.len() - 1
        }
    }

    /// Export the baseline histogram.
    pub(crate) fn export_baseline(&self) -> HashMap<T, f64> {
        self.idx_map
            .iter()
            .map(|(feat_name, i)| (feat_name.clone(), self.baseline_bins[*i]))
            .collect()
    }

    pub(crate) fn n_bins(&self) -> usize {
        self.baseline_bins.len()
    }

    /// Redefine the baseline.
    pub(crate) fn reset(&mut self, baseline_data: &[T]) -> Result<(), DriftError> {
        *self = Self::new(baseline_data)?;
        Ok(())
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_new_hist_prob() {
        let bl_hist = vec![10.0, 20.0, 30.0, 40.0];
        let base: Vec<f64> = vec![0.10, 0.20, 0.30, 0.40];
        let test_bins = compute_new_hist_prob(100, &bl_hist).unwrap();
        assert_eq!(base, test_bins);
    }

    #[test]
    fn continuous_baseline_reset() {
        let test_dataset: Vec<f64> = (0..1000).map(|_| rand::random::<f64>() * 100_f64).collect();
        let mut bl =
            BaselineContinuousBins::new(&test_dataset, super::QuantileType::default()).unwrap();
        let test = bl.clone();
        bl.reset(&test_dataset).unwrap();
        assert_eq!(test, bl)
    }

    #[test]
    fn categorical_baseline_reset() {
        let candidates = vec!["a", "b", "c", "d"];
        let test_dataset: Vec<&'static str> = (0..1000)
            .map(|_| candidates[(rand::random::<f32>() * 4_f32).floor() as usize])
            .collect();

        let mut bl = BaselineCategoricalBins::new(&test_dataset).unwrap();
        dbg!(&bl);
        let base = bl.clone();
        bl.reset(&test_dataset).unwrap();
        assert_eq!(base, bl);
    }
}
