use super::{
    baseline::{BaselineCategoricalBins, BaselineContinuousBins},
    distribution::QuantileType,
    drift_metrics::{global_compute_drift, DataDriftType, DriftContainer, DriftContainerType},
    DEFAULT_MAX_STREAM_SIZE, DEFAULT_STREAM_FLUSH_CADENCE,
};
use crate::errors::DriftError;
use ahash::{HashMap, HashMapExt};
use std::hash::Hash;
use std::marker::PhantomData;
use std::num::NonZeroU64;
use std::time::{Duration, Instant};

#[non_exhaustive]
#[derive(Debug)]
pub enum StreamingDriftMode {
    Flush { size: u64, cadence: u64 },
    ExponentialDecay(NonZeroU64),
}

impl Default for StreamingDriftMode {
    fn default() -> StreamingDriftMode {
        StreamingDriftMode::Flush {
            size: DEFAULT_MAX_STREAM_SIZE,
            cadence: DEFAULT_STREAM_FLUSH_CADENCE,
        }
    }
}

#[derive(Default, Debug)]
pub struct ContinuousStreamingDriftConfig {
    mode: StreamingDriftMode,
    quantile_type: QuantileType,
}

#[derive(Debug)]
enum StreamModeInner {
    Flush {
        size: f64,
        cadence: Duration,
        last_flush_ts: Instant,
    },
    ExponentialDecay(f64),
}

impl From<StreamingDriftMode> for StreamModeInner {
    fn from(mode: StreamingDriftMode) -> StreamModeInner {
        match mode {
            StreamingDriftMode::Flush { size, cadence } => StreamModeInner::Flush {
                size: size as f64,
                cadence: Duration::new(cadence, 0),
                last_flush_ts: Instant::now(),
            },
            StreamingDriftMode::ExponentialDecay(half_life) => {
                let hl = half_life.get();
                StreamModeInner::ExponentialDecay(0.5_f64.powf(1_f64 / hl as f64))
            }
        }
    }
}

impl StreamModeInner {
    /// Resets state on flush.
    fn perform_flush(&mut self) {
        match self {
            StreamModeInner::Flush { last_flush_ts, .. } => {
                *last_flush_ts = Instant::now();
            }
            _ => {}
        }
    }

    /// Determine if a flush is needed. When mode is using ExponentialDecay, this should be
    /// compiled out in release mode.
    fn needs_flush(&self, total_stream_size: f64) -> bool {
        match self {
            StreamModeInner::Flush {
                size,
                cadence,
                last_flush_ts,
            } => {
                // First check size.
                // If size is valid, mask the Instant check to only check every 255
                // This introduces small error and amortizes the Instant check. Instant check is
                // non trivial expensive.
                total_stream_size >= *size
                    || (total_stream_size as usize & 255 == 0
                        && Instant::now().duration_since(*last_flush_ts) >= *cadence)
            }
            StreamModeInner::ExponentialDecay(_) => false,
        }
    }

    /// Fetch the number of seconds since last flush.
    fn last_flush(&self) -> u64 {
        match self {
            StreamModeInner::Flush { last_flush_ts, .. } => {
                Instant::now().duration_since(*last_flush_ts).as_secs()
            }
            StreamModeInner::ExponentialDecay(_) => u64::default(),
        }
    }
}

// TODO: apply these markers to statically limit API behavior.
pub struct FlushModeMark;
pub struct DecayModeMark;

trait StreamingDataDriftMark {}
impl StreamingDataDriftMark for FlushModeMark {}
impl StreamingDataDriftMark for DecayModeMark {}

/// Data drift type for continuous data. This type keeps a baseline state, and will compute the
/// drift on a discrete dataset.
pub struct ContinuousDataDrift {
    baseline: BaselineContinuousBins,
    rt_bins: Vec<f64>,
}

impl DriftContainer for ContinuousDataDrift {
    fn get_baseline_hist(&self) -> &[f64] {
        &self.baseline.baseline_hist
    }

    fn get_runtime_bins(&self) -> &[f64] {
        &self.rt_bins
    }

    fn get_bin_edges(&self) -> Option<&[f64]> {
        Some(&self.baseline.bin_edges)
    }

    fn num_examples(&self) -> f64 {
        self.rt_bins.iter().sum()
    }

    fn container_type(&self) -> DriftContainerType {
        DriftContainerType::Continuous
    }
}

impl ContinuousDataDrift {
    /// Construct a new instance with the provided baseline set. The number of bins will be
    /// determined using the [`QuantileType`] method provided. If none is provided it will use the
    /// default method.
    pub fn new_from_baseline(
        quantile_type: Option<QuantileType>,
        bl_slice: &[f64],
    ) -> Result<ContinuousDataDrift, DriftError> {
        let baseline = BaselineContinuousBins::new(bl_slice, quantile_type.unwrap_or_default())?;
        let mut obj = ContinuousDataDrift {
            baseline,
            rt_bins: Vec::new(),
        };
        obj.init_runtime_containers();
        Ok(obj)
    }

    fn clear_rt(&mut self) {
        self.rt_bins.fill(0.);
    }

    pub fn compute_drift(
        &mut self,
        runtime_data: &[f64],
        drift_type: DataDriftType,
    ) -> Result<f64, DriftError> {
        self.build_rt_hist(runtime_data)?;
        let drift = global_compute_drift(self, drift_type);
        self.clear_rt();
        Ok(drift)
    }

    #[inline]
    fn build_rt_hist(&mut self, runtime_data: &[f64]) -> Result<(), DriftError> {
        if runtime_data.is_empty() {
            return Err(DriftError::EmptyRuntimeData);
        }
        for item in runtime_data {
            let idx = self.baseline.resolve_bin(*item);
            self.rt_bins[idx] += 1_f64;
        }
        Ok(())
    }

    fn init_runtime_containers(&mut self) {
        let len = self.baseline.baseline_hist.len();
        self.rt_bins = vec![0_f64; len];
    }

    /// Reset the baseline state with a new baseline dataset. Same rules apply to the bin count at
    /// construction, but in this instance, a best effort attempt will be made to use the current
    /// number of bins. Errors when the dataset passed in empty.
    pub fn reset_baseline(&mut self, baseline_slice: &[f64]) -> Result<(), DriftError> {
        self.baseline.reset(baseline_slice)?;
        self.init_runtime_containers();
        Ok(())
    }

    pub fn n_bins(&self) -> usize {
        self.baseline.n_bins
    }

    pub fn export_baseline(&self) -> Vec<f64> {
        self.baseline.export_baseline()
    }
}

/// A streaming variant of the [`ContinuousDataDrift`] type. This is a stateful "stream" for long
/// running drift monitoring. Every new example pushed into the stream will update state, and a
/// drift snapshot can be computed at any point in time, granted that there is data present in the
/// stream.
pub struct StreamingContinuousDataDrift<M> {
    baseline: BaselineContinuousBins,
    stream_bins: Vec<f64>,
    total_stream_size: f64,
    mode: StreamModeInner,
    _mark: PhantomData<M>,
}

impl<M> DriftContainer for StreamingContinuousDataDrift<M> {
    fn get_baseline_hist(&self) -> &[f64] {
        &self.baseline.baseline_hist
    }

    fn get_runtime_bins(&self) -> &[f64] {
        &self.stream_bins
    }

    fn get_bin_edges(&self) -> Option<&[f64]> {
        Some(&self.baseline.bin_edges)
    }

    fn num_examples(&self) -> f64 {
        self.total_stream_size
    }

    fn container_type(&self) -> DriftContainerType {
        DriftContainerType::Continuous
    }
}

impl StreamingContinuousDataDrift<DecayModeMark> {
    pub fn new_decay(
        baseline_data: &[f64],
        quantile_type: QuantileType,
        half_life: NonZeroU64,
    ) -> Result<StreamingContinuousDataDrift<DecayModeMark>, DriftError> {
        let baseline = BaselineContinuousBins::new(baseline_data, quantile_type)?;
        let bl_hist_len = baseline.baseline_hist.len();
        let stream_bins: Vec<f64> = vec![0_f64; bl_hist_len];
        let mode = StreamModeInner::ExponentialDecay(0.5_f64.powf(1_f64 / half_life.get() as f64));

        Ok(StreamingContinuousDataDrift {
            stream_bins,
            baseline,
            total_stream_size: 0_f64,
            mode,
            _mark: PhantomData,
        })
    }
    /// Computes the drift using the specified drift type.
    pub fn compute_drift(&mut self, drift_type: DataDriftType) -> Result<f64, DriftError> {
        if self.is_empty() {
            return Err(DriftError::EmptyRuntimeData);
        }
        // decay only gets applied in decay mode.
        // This should be compiled out in release mode.
        self.apply_decay();
        Ok(global_compute_drift(self, drift_type))
    }

    pub fn compute_drift_multiple_criteria(
        &mut self,
        drift_types: &[DataDriftType],
    ) -> Result<Vec<f64>, DriftError> {
        if self.is_empty() {
            return Err(DriftError::EmptyRuntimeData);
        }
        self.apply_decay();
        Ok(drift_types
            .iter()
            .map(|drift_t| global_compute_drift(self, *drift_t))
            .collect())
    }

    /// Apply the decay factor to lower priority on older records.
    fn apply_decay(&mut self) {
        let StreamModeInner::ExponentialDecay(decay_factor) = self.mode else {
            unreachable!()
        };
        for bin in self.stream_bins.iter_mut() {
            *bin = (*bin * decay_factor).floor();
        }
        self.total_stream_size = (self.total_stream_size * decay_factor).floor();
    }

    /// Push a single example into the stream.
    #[inline]
    pub fn update_stream(&mut self, runtime_example: f64) {
        let idx = self.baseline.resolve_bin(runtime_example);

        self.stream_bins[idx] += 1_f64;
        self.total_stream_size += 1_f64;
    }

    /// Push a batch dataset to the stream.
    pub fn update_stream_batch(&mut self, runtime_slice: &[f64]) -> Result<(), DriftError> {
        if runtime_slice.is_empty() {
            return Err(DriftError::EmptyRuntimeData);
        }

        for item in runtime_slice {
            self.update_stream(*item)
        }

        Ok(())
    }
}

impl StreamingContinuousDataDrift<FlushModeMark> {
    pub fn default_flush(
        baseline_data: &[f64],
    ) -> Result<StreamingContinuousDataDrift<FlushModeMark>, DriftError> {
        Self::new_flush(
            baseline_data,
            QuantileType::default(),
            DEFAULT_MAX_STREAM_SIZE,
            Duration::from_secs(DEFAULT_STREAM_FLUSH_CADENCE / 1000),
        )
    }
    pub fn new_flush(
        baseline_data: &[f64],
        quantile_type: QuantileType,
        flush_size: u64,
        flush_cadence: Duration,
    ) -> Result<StreamingContinuousDataDrift<FlushModeMark>, DriftError> {
        let baseline = BaselineContinuousBins::new(baseline_data, quantile_type)?;
        let bl_hist_len = baseline.baseline_hist.len();
        let stream_bins: Vec<f64> = vec![0_f64; bl_hist_len];
        let mode = StreamModeInner::Flush {
            size: flush_size as f64,
            cadence: flush_cadence,
            last_flush_ts: Instant::now(),
        };

        Ok(StreamingContinuousDataDrift {
            stream_bins,
            baseline,
            total_stream_size: 0_f64,
            mode,
            _mark: PhantomData,
        })
    }
    /// Computes the drift using the specified drift type.
    pub fn compute_drift(&mut self, drift_type: DataDriftType) -> Result<f64, DriftError> {
        if self.is_empty() {
            return Err(DriftError::EmptyRuntimeData);
        }
        Ok(global_compute_drift(self, drift_type))
    }

    pub fn compute_drift_multiple_criteria(
        &mut self,
        drift_types: &[DataDriftType],
    ) -> Result<Vec<f64>, DriftError> {
        if self.is_empty() {
            return Err(DriftError::EmptyRuntimeData);
        }

        Ok(drift_types
            .iter()
            .map(|drift_t| global_compute_drift(self, *drift_t))
            .collect())
    }

    /// Push a single example into the stream.
    #[inline]
    pub fn update_stream(&mut self, runtime_example: f64) {
        let idx = self.baseline.resolve_bin(runtime_example);
        if self.mode.needs_flush(self.total_stream_size) {
            self.flush()
        }
        self.stream_bins[idx] += 1_f64;
        self.total_stream_size += 1_f64;
    }

    /// Push a batch dataset to the stream.
    pub fn update_stream_batch(&mut self, runtime_slice: &[f64]) -> Result<(), DriftError> {
        if runtime_slice.is_empty() {
            return Err(DriftError::EmptyRuntimeData);
        }

        for item in runtime_slice {
            self.update_stream(*item)
        }

        Ok(())
    }

    /// FLush stream. This will clear all runtime bins, this is reccomended every so often to clear
    /// old state and get a more recent view of drift. Baseline state is not altered.
    pub fn flush(&mut self) {
        self.flush_runtime_stream();
        self.mode.perform_flush();
    }

    pub fn last_flush(&self) -> u64 {
        self.mode.last_flush()
    }
}

impl<M: StreamingDataDriftMark> StreamingContinuousDataDrift<M> {
    pub fn is_empty(&self) -> bool {
        self.stream_bins.iter().sum::<f64>() == 0_f64
    }

    /// Reset the baseline with a new baseline dataset. A best effort is made to maintain the same
    /// number of bins, but is subject to the same bin size restrictions as the initial baseline
    /// construction.
    pub fn reset_baseline(&mut self, baseline_slice: &[f64]) -> Result<(), DriftError> {
        self.baseline.reset(baseline_slice)?;
        self.stream_bins = vec![0_f64; self.baseline.baseline_hist.len()];
        self.total_stream_size = 0_f64;
        self.mode.perform_flush();
        Ok(())
    }

    /// The number of total samples accumulated in the stream.
    pub fn total_samples(&self) -> usize {
        self.total_stream_size as usize
    }

    /// The number of histogram bins.
    pub fn n_bins(&self) -> usize {
        self.baseline.n_bins
    }

    /// Export a snapshot of the stream state. This includes, the baseline bins, the current bin
    /// distribution of the runtime data, and the bin edges that determine the internal histogram binning.
    pub fn export_snapshot(&self) -> Result<HashMap<String, Vec<f64>>, DriftError> {
        if self.total_stream_size == 0_f64 {
            return Err(DriftError::EmptyRuntimeData);
        }
        // determine snapshot shape
        let mut table: HashMap<String, Vec<f64>> = HashMap::with_capacity(3);
        table.insert("binEdges".into(), self.baseline.bin_edges.clone());
        table.insert("baselineBins".into(), self.export_baseline());
        let bin_ratio_snapshot = self
            .stream_bins
            .iter()
            .map(|v| *v / self.total_stream_size)
            .collect();
        table.insert("streamBins".into(), bin_ratio_snapshot);
        Ok(table)
    }

    /// Export a the baseline bin proportions. Returns an owned `Vec<f64>`, which contains the
    /// proportional bin distribution present in the baseline set, and thus what all drift metrics
    /// are computed with respect to.
    pub fn export_baseline(&self) -> Vec<f64> {
        self.baseline.export_baseline()
    }

    // zero out all bins
    fn flush_runtime_stream(&mut self) {
        self.stream_bins.fill(0_f64);
        self.total_stream_size = 0_f64;
    }
}

// store bins as vec for better performance on psi computation and bin accumulation
// store cat label to index in map
pub struct CategoricalDataDrift<T: Hash + Ord + Clone> {
    pub(crate) baseline: BaselineCategoricalBins<T>,
    rt_bins: Vec<f64>,
}

impl<T: Hash + Ord + Clone> DriftContainer for CategoricalDataDrift<T> {
    fn get_baseline_hist(&self) -> &[f64] {
        &self.baseline.baseline_bins
    }

    fn get_runtime_bins(&self) -> &[f64] {
        &self.rt_bins
    }

    fn get_bin_edges(&self) -> Option<&[f64]> {
        None
    }

    fn num_examples(&self) -> f64 {
        self.rt_bins.iter().sum()
    }

    fn container_type(&self) -> DriftContainerType {
        DriftContainerType::Categorical
    }
}

impl<T: Hash + Ord + Clone> CategoricalDataDrift<T> {
    /// Construct a new instance with the provided baseline dataset. [`Hash`] indicates
    /// something that can be used as a reference to key into a `HashMap<String, f64>`, these
    /// bounds are to allow some other type of label value, such as an enum. The number of bins
    /// will be equal to the number of unique values present in the baseline data set, with an
    /// additional bin for values that occur in the runtime dataset that do not occur in the
    /// baseline dataset.
    pub fn new(baseline_data: &[T]) -> Result<CategoricalDataDrift<T>, DriftError> {
        if baseline_data.is_empty() {
            return Err(DriftError::EmptyBaselineData);
        }

        let baseline = BaselineCategoricalBins::new(baseline_data)?;
        let num_bins = baseline.baseline_bins.len();
        let rt_bins: Vec<f64> = vec![0_f64; num_bins];

        Ok(CategoricalDataDrift { baseline, rt_bins })
    }

    pub fn compute_drift(
        &mut self,
        runtime_data: &[T],
        drift_type: DataDriftType,
    ) -> Result<f64, DriftError> {
        self.build_rt_hist(runtime_data)?;
        let drift = global_compute_drift(self, drift_type);
        self.clear_rt();
        Ok(drift)
    }

    fn build_rt_hist(&mut self, runtime_data: &[T]) -> Result<(), DriftError> {
        if runtime_data.is_empty() {
            return Err(DriftError::EmptyRuntimeData);
        }
        for cat in runtime_data.iter() {
            let i = self.baseline.get_bin(cat);

            self.rt_bins[i] += 1_f64;
        }
        Ok(())
    }

    fn clear_rt(&mut self) {
        self.rt_bins.fill(0_f64);
    }

    // Reset the baseline state with a new baseline dataset. This will adjust the number of bins to
    // n + 1 where n is the number of observed unique examples.
    pub fn reset_baseline(&mut self, new_baseline: &[T]) -> Result<(), DriftError> {
        self.baseline.reset(new_baseline)?;
        let num_bins = self.baseline.baseline_bins.len();

        // pay the cost to reallocate bins in order to have correct size
        // not common path
        self.rt_bins = vec![0_f64; num_bins];
        Ok(())
    }

    pub fn export_baseline(&self) -> HashMap<T, f64> {
        self.baseline.export_baseline()
    }
}

/// Streaming implementation of '[CategoricalDataDrift]' type. This is intended for long running
/// services to give an indication of the data drift over a longer contiguous window. A stateful
/// stream, where point in time snapshots can be generated.
pub struct StreamingCategoricalDataDrift<T: Hash + Ord + Clone, M> {
    baseline: BaselineCategoricalBins<T>,
    stream_bins: Vec<f64>,
    total_stream_size: f64,
    mode: StreamModeInner,
    _mark: PhantomData<M>,
}

impl<T: Hash + Ord + Clone, M> DriftContainer for StreamingCategoricalDataDrift<T, M> {
    fn get_baseline_hist(&self) -> &[f64] {
        &self.baseline.baseline_bins
    }

    fn get_runtime_bins(&self) -> &[f64] {
        &self.stream_bins
    }

    fn get_bin_edges(&self) -> Option<&[f64]> {
        None
    }

    fn num_examples(&self) -> f64 {
        self.total_stream_size
    }

    fn container_type(&self) -> DriftContainerType {
        DriftContainerType::Categorical
    }
}

impl<T: Hash + Ord + Clone> StreamingCategoricalDataDrift<T, FlushModeMark> {
    pub fn default_flush(
        baseline_data: &[T],
    ) -> Result<StreamingCategoricalDataDrift<T, FlushModeMark>, DriftError> {
        Self::new_flush(
            baseline_data,
            DEFAULT_MAX_STREAM_SIZE,
            Duration::from_secs(DEFAULT_STREAM_FLUSH_CADENCE / 1000),
        )
    }
    pub fn new_flush(
        baseline_data: &[T],
        flush_size: u64,
        flush_cadence: Duration,
    ) -> Result<StreamingCategoricalDataDrift<T, FlushModeMark>, DriftError> {
        let baseline = BaselineCategoricalBins::new(baseline_data)?;
        let bl_hist_len = baseline.baseline_bins.len();
        let stream_bins: Vec<f64> = vec![0_f64; bl_hist_len];
        let mode = StreamModeInner::Flush {
            size: flush_size as f64,
            cadence: flush_cadence,
            last_flush_ts: Instant::now(),
        };

        Ok(StreamingCategoricalDataDrift {
            stream_bins,
            baseline,
            total_stream_size: 0_f64,
            mode,
            _mark: PhantomData,
        })
    }

    pub fn compute_drift(&mut self, drift_type: DataDriftType) -> Result<f64, DriftError> {
        if self.is_empty() {
            return Err(DriftError::EmptyRuntimeData);
        }
        Ok(global_compute_drift(self, drift_type))
    }

    pub fn compute_drift_multiple_criteria(
        &mut self,
        drift_types: &[DataDriftType],
    ) -> Result<Vec<f64>, DriftError> {
        if self.is_empty() {
            return Err(DriftError::EmptyRuntimeData);
        }
        Ok(drift_types
            .iter()
            .map(|drift_t| global_compute_drift(self, *drift_t))
            .collect())
    }

    #[inline]
    pub fn update_stream(&mut self, item: &T) {
        let idx = self.baseline.get_bin(item);
        if self.mode.needs_flush(self.total_stream_size) {
            self.flush();
        }
        self.stream_bins[idx] += 1_f64;
        self.total_stream_size += 1_f64;
    }

    pub fn update_stream_batch(&mut self, runtime_data: &[T]) -> Result<(), DriftError> {
        if runtime_data.is_empty() {
            return Err(DriftError::EmptyRuntimeData);
        }

        for cat in runtime_data.iter() {
            self.update_stream(cat)
        }

        Ok(())
    }

    pub fn flush(&mut self) {
        self.flush_runtime_stream();
        self.mode.perform_flush();
    }
}
impl<T: Hash + Ord + Clone> StreamingCategoricalDataDrift<T, DecayModeMark> {
    pub fn new_decay(
        baseline_data: &[T],
        half_life: NonZeroU64,
    ) -> Result<StreamingCategoricalDataDrift<T, DecayModeMark>, DriftError> {
        let baseline = BaselineCategoricalBins::new(baseline_data)?;
        let bl_hist_len = baseline.baseline_bins.len();
        let stream_bins: Vec<f64> = vec![0_f64; bl_hist_len];
        let mode = StreamModeInner::ExponentialDecay(0.5_f64.powf(1_f64 / half_life.get() as f64));

        Ok(StreamingCategoricalDataDrift {
            stream_bins,
            baseline,
            total_stream_size: 0_f64,
            mode,
            _mark: PhantomData,
        })
    }

    pub fn compute_drift(&mut self, drift_type: DataDriftType) -> Result<f64, DriftError> {
        if self.is_empty() {
            return Err(DriftError::EmptyRuntimeData);
        }
        self.apply_decay();
        Ok(global_compute_drift(self, drift_type))
    }

    pub fn compute_drift_multiple_criteria(
        &mut self,
        drift_types: &[DataDriftType],
    ) -> Result<Vec<f64>, DriftError> {
        if self.is_empty() {
            return Err(DriftError::EmptyRuntimeData);
        }
        self.apply_decay();
        Ok(drift_types
            .iter()
            .map(|drift_t| global_compute_drift(self, *drift_t))
            .collect())
    }

    fn apply_decay(&mut self) {
        let StreamModeInner::ExponentialDecay(decay_factor) = self.mode else {
            unreachable!()
        };
        for bin in self.stream_bins.iter_mut() {
            *bin = (*bin * decay_factor).floor();
        }
        self.total_stream_size = (self.total_stream_size * decay_factor).floor();
    }

    #[inline]
    pub fn update_stream(&mut self, item: &T) {
        let idx = self.baseline.get_bin(item);
        self.stream_bins[idx] += 1_f64;
        self.total_stream_size += 1_f64;
    }

    pub fn update_stream_batch(&mut self, runtime_data: &[T]) -> Result<(), DriftError> {
        if runtime_data.is_empty() {
            return Err(DriftError::EmptyRuntimeData);
        }

        for cat in runtime_data.iter() {
            self.update_stream(cat)
        }

        Ok(())
    }
}

impl<T: Hash + Ord + Clone, M: StreamingDataDriftMark> StreamingCategoricalDataDrift<T, M> {
    pub fn is_empty(&self) -> bool {
        self.stream_bins.iter().sum::<f64>() == 0_f64
    }

    pub fn reset_baseline(&mut self, new_baseline: &[T]) -> Result<(), DriftError> {
        self.baseline.reset(new_baseline)?;
        self.mode.perform_flush();
        self.init_stream_bins();
        Ok(())
    }

    pub fn total_samples(&self) -> usize {
        self.total_stream_size.floor() as usize
    }

    pub fn last_flush(&self) -> u64 {
        self.mode.last_flush()
    }

    pub fn export_snapshot(&self) -> HashMap<T, f64> {
        self.baseline
            .idx_map
            .iter()
            .map(|(feat_name, i)| (feat_name.clone(), self.stream_bins[*i]))
            .collect()
    }

    pub fn export_baseline(&self) -> HashMap<T, f64> {
        self.baseline.export_baseline()
    }

    fn init_stream_bins(&mut self) {
        self.stream_bins = vec![0_f64; self.baseline.baseline_bins.len()]
    }

    fn flush_runtime_stream(&mut self) {
        self.stream_bins.fill(0_f64);
        self.total_stream_size = 0_f64;
    }
}

#[cfg(test)]
mod continuous_tests {
    use super::*;

    #[test]
    fn test_continuous_baseline_builds_expected_bins() {
        let baseline = [1.0, 2.0, 3.0, 4.0];
        let psi = ContinuousDataDrift::new_from_baseline(None, &baseline).unwrap();

        // 3 bins → 4 edges
        assert_eq!(psi.baseline.bin_edges.len(), 4);
        assert_eq!(psi.rt_bins.len(), 3);
    }

    #[test]
    fn test_continuous_psi_zero_when_no_drift() {
        let baseline = [1.0, 2.0, 3.0, 4.0];
        let mut psi = ContinuousDataDrift::new_from_baseline(None, &baseline).unwrap();
        let runtime = [1.0, 2.0, 3.0, 4.0];

        let drift = psi
            .compute_drift(&runtime, DataDriftType::PopulationStabilityIndex)
            .unwrap();
        assert!(drift.abs() < 1e-9);
    }

    #[test]
    fn test_continuous_psi_detects_shift() {
        let baseline = [1.0, 2.0, 3.0, 4.0];
        let mut psi = ContinuousDataDrift::new_from_baseline(None, &baseline).unwrap();
        let runtime = [10.0, 11.0, 12.0, 13.0];
        let drift = psi
            .compute_drift(&runtime, DataDriftType::PopulationStabilityIndex)
            .unwrap();
        assert!(drift > 0.5);
    }

    #[test]
    fn test_streaming_continuous_accumulation() {
        let baseline = [1_f64, 2_f64, 3_f64, 3_f64, 4_f64];
        let mut streaming = StreamingContinuousDataDrift::default_flush(&baseline).unwrap();

        streaming
            .update_stream_batch(&[1.0, 2.0, 2.0, 3.0, 4.0])
            .unwrap();

        let d1 = streaming
            .compute_drift(DataDriftType::PopulationStabilityIndex)
            .unwrap();
        streaming
            .update_stream_batch(&[3.0, 4.0, 2.0, 2.0, 1.0, 3.0])
            .unwrap();

        let d2 = streaming
            .compute_drift(DataDriftType::PopulationStabilityIndex)
            .unwrap();

        assert!(d1.abs() < 1e-9);
        assert!(d2.abs() < 1e-2);
        assert_eq!(streaming.total_samples(), 11);
    }

    #[test]
    fn test_streaming_flush() {
        let baseline = [1.0, 2.0, 3.0, 4.0];
        let mut streaming = StreamingContinuousDataDrift::default_flush(&baseline).unwrap();

        streaming.update_stream_batch(&[1.0, 2.0, 3.0]).unwrap();
        streaming.flush();

        assert_eq!(streaming.total_samples(), 0);
    }
}

#[cfg(test)]
mod categorical_tests {
    use super::*;

    #[test]
    fn test_categorical_baseline_builds_expected_size() {
        let baseline = ["a", "b", "a", "c"];
        let psi = CategoricalDataDrift::new(&baseline).unwrap();

        // baseline has 3 real labels + OTHER bucket
        assert_eq!(psi.baseline.baseline_bins.len(), 4);
    }

    #[test]
    fn test_categorical_psi_zero_when_no_drift() {
        let baseline = ["a", "b", "a", "c"];
        let mut psi = CategoricalDataDrift::new(&baseline).unwrap();
        let runtime = ["a", "b", "a", "c"];
        let drift = psi
            .compute_drift(&runtime, DataDriftType::PopulationStabilityIndex)
            .unwrap();
        assert!(drift.abs() < 1e-9);
    }

    #[test]
    fn test_categorical_psi_detects_shift() {
        let baseline = ["a", "b", "a", "c"];
        let mut psi = CategoricalDataDrift::new(&baseline).unwrap();
        let runtime = ["x", "x", "x", "x"]; // go to other bucket
        let drift = psi
            .compute_drift(&runtime, DataDriftType::PopulationStabilityIndex)
            .unwrap();
        assert!(drift > 0.5);
    }

    #[test]
    fn test_streaming_categorical_accumulation() {
        let baseline = ["a", "b"];
        let mut streaming = StreamingCategoricalDataDrift::default_flush(&baseline).unwrap();

        streaming.update_stream_batch(&["a", "b"]).unwrap();
        let d1 = streaming
            .compute_drift(DataDriftType::PopulationStabilityIndex)
            .unwrap();
        let mut stream = Vec::new();

        for _ in 0..500 {
            stream.push("a")
        }

        for _ in 0..490 {
            stream.push("b")
        }
        streaming.update_stream_batch(&stream).unwrap();
        let d2 = streaming
            .compute_drift(DataDriftType::PopulationStabilityIndex)
            .unwrap();

        assert_eq!(streaming.total_samples(), 992);
        assert!(d1 < 1e-9);
        assert!(d2 < 1e-2);
    }
}
