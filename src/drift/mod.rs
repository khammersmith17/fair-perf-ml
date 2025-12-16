pub mod psi;
use crate::errors::DriftError;
use crate::metrics::STABILITY_EPS;
use ahash::{HashMap, HashMapExt};
use std::cmp::Ordering;
use std::hash::Hash;
use uuid::Uuid;

// A trait to define what can be treated as string like in this context, what we need here is that
// is can be represented as &str, can be hashed, can be compared for eqaulity and can be
// transformed into an owned String.
pub trait StringLike: AsRef<str> + Eq + Hash + ToString {}
impl<T> StringLike for T where T: AsRef<str> + Eq + Hash + ToString {}

const DEFAULT_STREAM_FLUSH: i64 = 3600 * 24;
const MAX_STREAM_SIZE: usize = 1_000_000;
// read in from user defined env var or set to default epsilon
// optional user config

// Break out baseline to have shared logic between the discrete and the streaming variants of drift
// utilities.
// Also allows for more elegant composition of different usage
pub(crate) struct BaselineContinuousBins {
    n_bins: usize,
    bin_edges: Vec<f64>,
    baseline_hist: Vec<f64>,
}

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

impl BaselineContinuousBins {
    // Constructor on a baseline dataset.
    fn new(n_bins: usize, baseline_data: &[f64]) -> Result<BaselineContinuousBins, DriftError> {
        let mut obj = BaselineContinuousBins {
            n_bins,
            bin_edges: Vec::new(),
            baseline_hist: Vec::new(),
        };

        obj.init_baseline_hist(baseline_data)?;
        Ok(obj)
    }

    // init method moved out of constructor to be reusable across new initialization as well as
    // when reseting the baseline
    fn init_baseline_hist(&mut self, baseline_data: &[f64]) -> Result<(), DriftError> {
        self.define_bins(baseline_data)?;
        let (bl_count, bl_hist) = self.build_bl_hist(baseline_data);
        match compute_new_hist_prob(bl_count, &bl_hist) {
            Ok(processed_bl_hist) => {
                self.baseline_hist = processed_bl_hist;
                Ok(())
            }
            Err(_) => Err(DriftError::EmptyBaselineData.into()),
        }
    }

    // Build the baseline histogram.
    fn build_bl_hist(&self, data_slice: &[f64]) -> (usize, Vec<f64>) {
        let bl_count = data_slice.len();
        let mut hist = vec![0_f64; self.bin_edges.len() - 1];
        let n_bins = self.bin_edges.len() - 1;
        for item in data_slice {
            let i = self.bin_edges.partition_point(|edge| *item >= *edge);
            let idx = i.saturating_sub(1).min(n_bins - 1);
            hist[idx] += 1_f64;
        }

        (bl_count, hist)
    }

    // Define the bin edges. Makes best effort to accomodate user requested number of bins, if the
    // user define number of bins is less than the size of the dataset, or the number of bins
    // cannot reasonably satify the distribtution. In a happy case, the number of bin edges will be
    // equal to the number of requested bins + 1.
    fn define_bins(&mut self, data: &[f64]) -> Result<(), DriftError> {
        // sort baseline data for more efficient bin edge computation
        let mut sorted_baseline: Vec<f64> = data.to_vec();

        // baselining requires > 1 baseline sample
        if sorted_baseline.len() <= 1 {
            return Err(DriftError::EmptyBaselineData);
        }

        // do not accept NaNs
        if sorted_baseline.iter().any(|value| value.is_nan()) {
            return Err(DriftError::NaNValueError);
        }

        sorted_baseline.sort_by(|a, b| a.partial_cmp(&b).unwrap_or(Ordering::Equal));

        self.bin_edges.clear();

        // if all sorted items in the baseline sample are equal, one logical bin
        // safe unwrap because we know there are items in the array
        if sorted_baseline.first().unwrap() == sorted_baseline.last().unwrap() {
            self.bin_edges
                .extend(vec![sorted_baseline[0], sorted_baseline[0]]);
            return Ok(());
        }

        let n_bl_samples = sorted_baseline.len();
        let n_bins = self.n_bins.min(n_bl_samples - 1).max(1);
        let bin_size = sorted_baseline.len() / n_bins;

        // set population left edge
        self.bin_edges.push(sorted_baseline[0]);

        // set bins using an O(number of bins) loop
        // index in bin_size * 1
        // creates bins of [sorted_baseline[(i - 1) * bin_size], sorted_baseline[i * bin_size]]
        for i in 1..(n_bins) {
            let idx = i * bin_size;
            if idx < n_bl_samples {
                self.bin_edges.push(sorted_baseline[idx - 1]);
            }
        }

        // set population right edge
        self.bin_edges.push(sorted_baseline[n_bl_samples - 1]);
        self.n_bins = n_bins;
        Ok(())
    }

    // Resolve the bin a particular data example falls into.
    #[inline]
    fn resolve_bin(&self, sample: f64) -> usize {
        // find "pivot" point
        // ie the bin where value >= left and < right
        let i = self.bin_edges.partition_point(|edge| sample >= *edge);
        i.saturating_sub(1).clamp(0, self.n_bins - 1)
    }

    // call into init method
    fn reset(&mut self, baseline_data: &[f64]) -> Result<(), DriftError> {
        self.init_baseline_hist(baseline_data)?;
        Ok(())
    }
}

// Estimate the number of classes in a categorical dataset, used to estimate a preallocated
// container for the bins.
fn compute_expected_categorical_bins(n: f64) -> usize {
    n.powf(0.7).ceil() as usize
}

// let user specify other label
// otherwise use preset label with uuid to be sure
fn construct_other_label() -> String {
    if let Ok(var) = std::env::var("FAIR_PERF_OTHER_LABEL") {
        return var;
    } else {
        let uuid = Uuid::new_v4();
        format!("__fairperf_othercat__{}", uuid)
    }
}

// Compute psi on runtime and baseline bins. Element wise distance for each bucket with a
// sum reduction.
#[inline]
fn compute_psi(baseline_hist: &[f64], runtime_bins: &[f64], n: f64) -> f64 {
    // validate that rt and baseline bins are of same length
    debug_assert_eq!(runtime_bins.len(), baseline_hist.len());
    let mut psi: f64 = 0_f64;
    let eps = *STABILITY_EPS;
    for i in 0..baseline_hist.len() {
        // cheap clone, unsafe to skip bounds checks
        let mut b = unsafe { *baseline_hist.get_unchecked(i) };
        b = (b + eps).max(eps);
        let mut r = unsafe { *runtime_bins.get_unchecked(i) };
        r = r / n;
        r = (r + eps).max(eps);
        psi += (b - r) * (b / r).ln();
    }

    psi
}

// idx_map holds the bin for a particular data value.
// Baseline bins are the histogram generated on baseline data, and other label represents the
// "other" bucket for when a discrete value not seen in the baseline set is observed.
struct BaselineCategoricalBins {
    idx_map: HashMap<String, usize>,
    baseline_bins: Vec<f64>,
    other_label: String,
}

impl BaselineCategoricalBins {
    // Dataset must be implement StringLike. Preallocated with estimated capacity needed, define
    // bins and index map, allocated bins, fill histogram with counts.
    fn new<S: StringLike>(baseline_data: &[S]) -> BaselineCategoricalBins {
        let n = baseline_data.len() as f64;
        let other_label = construct_other_label();

        let predicted_capacity = compute_expected_categorical_bins(n);
        let mut initial_bins: HashMap<String, f64> = HashMap::with_capacity(predicted_capacity);

        for cat in baseline_data.iter() {
            if let Some(count) = initial_bins.get_mut(cat.as_ref()) {
                *count += 1_f64;
            } else {
                initial_bins.insert(cat.to_string(), 1_f64);
            }
        }
        initial_bins.insert(other_label.clone(), 0_f64);

        let mut idx_map: HashMap<String, usize> = HashMap::with_capacity(initial_bins.len());
        let mut baseline_bins: Vec<f64> = Vec::with_capacity(initial_bins.len());

        let mut i = 0_usize;
        for (key, count) in initial_bins.into_iter() {
            idx_map.insert(key, i);
            baseline_bins.push(count / n);
            i += 1;
        }

        BaselineCategoricalBins {
            idx_map,
            baseline_bins,
            other_label,
        }
    }

    // Export the baseline histogram.
    fn export_baseline(&self) -> HashMap<String, f64> {
        self.idx_map
            .iter()
            .map(|(feat_name, i)| (feat_name.clone(), self.baseline_bins[*i]))
            .collect()
    }

    // Redefine the baseline.
    fn reset<S: StringLike>(&mut self, baseline_data: &[S]) {
        let n = baseline_data.len() as f64;
        let new_predicted_capacity = compute_expected_categorical_bins(n);

        self.idx_map.clear();

        let mut initial_bins: HashMap<String, f64> = HashMap::with_capacity(new_predicted_capacity);
        for cat in baseline_data.iter() {
            if let Some(count) = initial_bins.get_mut(cat.as_ref()) {
                *count += 1_f64;
            } else {
                initial_bins.insert(cat.to_string(), 1_f64);
            }
        }
        initial_bins.insert(self.other_label.clone(), 0_f64);

        self.baseline_bins.clear();
        let mut i = 0_usize;
        for (key, count) in initial_bins.into_iter() {
            self.idx_map.insert(key, i);
            self.baseline_bins.push(count / n);
            i += 1;
        }
    }

    // Accept runtime bins from "parent" then compute the snapshot PSI.
    #[inline]
    fn normalize(&self, rt_bins: &[f64], n: f64) -> f64 {
        compute_psi(&self.baseline_bins, rt_bins, n)
    }
}
