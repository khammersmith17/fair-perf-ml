use super::distribution::QuantileType;
use crate::errors::DriftError;
use ahash::{HashMap, HashMapExt};
use std::cmp::Ordering;
use std::collections::BTreeMap;
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

// Break out baseline to have shared logic between the discrete and the streaming variants of drift
// utilities.
// Also allows for more elegant composition of different usage
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct BaselineContinuousBins {
    pub n_bins: usize,
    pub bin_edges: Vec<f64>,
    pub baseline_hist: Vec<f64>,
    q_type: QuantileType,
}

impl BaselineContinuousBins {
    // Constructor on a baseline dataset. Allocates then hyrdates with the provided baseline
    // dataset.
    pub(crate) fn new(
        baseline_data: &[f64],
        quantile_resolution: QuantileType,
    ) -> Result<BaselineContinuousBins, DriftError> {
        let sorted_baseline = Self::sort_baseline_data(baseline_data)?;
        let n_bins = quantile_resolution.compute_num_bins(&sorted_baseline);
        let mut obj = BaselineContinuousBins {
            n_bins,
            bin_edges: Vec::new(),
            baseline_hist: Vec::new(),
            q_type: quantile_resolution,
        };

        obj.init_baseline_hist(&sorted_baseline)?;
        Ok(obj)
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

    // init method moved out of constructor to be reusable across new initialization as well as
    // when reseting the baseline
    fn init_baseline_hist(&mut self, baseline_data: &[f64]) -> Result<(), DriftError> {
        self.define_bins(baseline_data)?;
        let bl_hist = self.build_bl_dist(baseline_data);
        match compute_new_hist_prob(baseline_data.len(), &bl_hist) {
            Ok(processed_bl_hist) => {
                self.baseline_hist = processed_bl_hist;
                Ok(())
            }
            Err(_) => Err(DriftError::EmptyBaselineData),
        }
    }

    // Build the baseline histogram.
    fn build_bl_dist(&self, data_slice: &[f64]) -> Vec<f64> {
        let mut hist = vec![0_f64; self.n_bins];
        for item in data_slice {
            let idx = self.resolve_bin(*item);
            hist[idx] += 1_f64;
        }

        hist
    }

    /*
     * - Bin edges will be of size num_bins - 2.
     * - The outer bins, or tail bins in the distribution will be reserved for values observed in the
     *  distribution that fall outsde the bounds of the baseline distribution.
     *  - Bin/quantile size will have its "step" size determined by evenly diving the difference
     *  between the max and min of the distribution and dividing by the number of bins - 2.
     *  - A value is assigned to a particular quantile if left <= value < right, otherwise it will
     *  be assigned to one of the tail quantile bins.
     * */
    fn define_bins(&mut self, sorted_baseline: &[f64]) -> Result<(), DriftError> {
        self.bin_edges.clear();

        // if all sorted items in the baseline sample are equal, one logical bin
        // safe unwrap because we know there are items in the array
        if sorted_baseline.first().unwrap() == sorted_baseline.last().unwrap() {
            self.bin_edges = vec![sorted_baseline[0]];
            return Ok(());
        }

        // Safely can unwrap here as we know the baseline data is non empty.
        let max = sorted_baseline.last().unwrap();
        let min = sorted_baseline.first().unwrap();
        let quantile_step_size = (max - min) / (self.n_bins - 2) as f64;
        self.bin_edges = vec![0_f64; self.n_bins - 2];

        let mut edge = sorted_baseline[0];

        for i in 0..self.n_bins - 2 {
            self.bin_edges[i] = edge;
            edge += quantile_step_size;
        }

        Ok(())
    }

    #[inline]
    fn left_bin_edge(&self) -> f64 {
        self.bin_edges[0]
    }

    #[inline]
    fn right_bin_edge(&self) -> f64 {
        // bin_edges.len == n_bins - 2
        // bin_edges[-1] == bin_edges[n_bins - 3]
        self.bin_edges[self.n_bins - 3]
    }

    // Resolve the bin a particular data example falls into.
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

    pub(crate) fn export_baseline(&self) -> Vec<f64> {
        self.baseline_hist.clone()
    }

    // call into init method
    pub(crate) fn reset(&mut self, baseline_data: &[f64]) -> Result<(), DriftError> {
        let sorted_baseline = Self::sort_baseline_data(baseline_data)?;
        self.n_bins = self.q_type.compute_num_bins(&sorted_baseline);
        self.init_baseline_hist(&sorted_baseline)?;
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

/*
* Each value present in the baseline dataset is mapped to a bin in the histogram Vec.
* The furthest right, ie len(set(baseline data)) index in the histogram Vec is reserved for
* observed values that were not part of the baseline set
* */

impl<T: Hash + Ord + Clone> BaselineCategoricalBins<T> {
    // bins and index map, allocated bins, fill histogram with counts.
    pub(crate) fn new(baseline_data: &[T]) -> Result<BaselineCategoricalBins<T>, DriftError> {
        let idx_map: HashMap<T, usize> = HashMap::new();
        let mut bins = BaselineCategoricalBins {
            idx_map,
            baseline_bins: Vec::new(),
        };

        bins.define_bins(baseline_data)?;

        Ok(bins)
    }

    /// Resolve the bin idx for a particular key, otherwise return out the bin reserved for the
    /// "other" bucket.
    pub(crate) fn get_bin(&self, key: &T) -> usize {
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

    /// Common methods to define bins. Used both at construction and when baseline bins are
    /// redefined.
    fn define_bins(&mut self, baseline_data: &[T]) -> Result<(), DriftError> {
        if baseline_data.is_empty() {
            return Err(DriftError::EmptyBaselineData);
        }
        let n = baseline_data.len() as f64;
        self.idx_map.clear();

        let mut initial_bins: BTreeMap<T, f64> = BTreeMap::new();
        for cat in baseline_data.iter() {
            if let Some(count) = initial_bins.get_mut(cat) {
                *count += 1_f64;
            } else {
                initial_bins.insert(cat.clone(), 1_f64);
            }
        }

        // Preallocate space for cardinatity of the dataset + 1
        // The additional bin is reserved for data values not observed in the baseline dataset
        self.baseline_bins = vec![0_f64; initial_bins.len() + 1_usize];
        for (i, (key, count)) in initial_bins.into_iter().enumerate() {
            self.idx_map.insert(key, i);
            self.baseline_bins[i] = count / n;
        }
        Ok(())
    }

    /// Redefine the baseline.
    pub(crate) fn reset(&mut self, baseline_data: &[T]) -> Result<(), DriftError> {
        self.define_bins(baseline_data)
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
