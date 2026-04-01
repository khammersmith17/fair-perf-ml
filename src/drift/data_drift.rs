use super::{
    baseline::{BaselineCategoricalBins, BaselineContinuousBins},
    distribution::QuantileType,
    drift_metrics::{global_compute_drift, DataDriftType, DriftContainer, DriftContainerType},
    DEFAULT_DECAY_HALF_LIFE, DEFAULT_MAX_STREAM_SIZE, DEFAULT_STREAM_FLUSH_CADENCE,
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

/// Mode marker for streaming drift types that operate in flush mode. When parameterized with
/// this marker, the stream accumulates data until either a sample size threshold or a time
/// cadence is reached, at which point all accumulated data is cleared and monitoring begins
/// fresh. This mode exposes [`flush`], [`last_flush`], and automatic flush on push.
///
/// [`flush`]: StreamingContinuousDataDrift::flush
/// [`last_flush`]: StreamingContinuousDataDrift::last_flush
pub struct FlushModeMark;

/// Mode marker for streaming drift types that operate in exponential decay mode. When
/// parameterized with this marker, older data is continuously down-weighted on each call to
/// [`compute_drift`] or [`compute_drift_multiple_criteria`] by a decay factor
/// α = 0.5^(1/half_life), where `half_life` is expressed in seconds. Data is never hard-cleared,
/// giving a recency-weighted view of the distribution with no discontinuities. This mode does
/// not expose [`flush`] or [`last_flush`].
///
/// [`compute_drift`]: StreamingContinuousDataDrift::compute_drift
/// [`compute_drift_multiple_criteria`]: StreamingContinuousDataDrift::compute_drift_multiple_criteria
/// [`flush`]: StreamingContinuousDataDrift::flush
/// [`last_flush`]: StreamingContinuousDataDrift::last_flush
pub struct DecayModeMark;

// Marker trait to allow shared behavior across the 2 modes.
// Requires #[allow(private_bounds)] at call sites.
pub(crate) trait StreamingDataDriftMark {}
impl StreamingDataDriftMark for FlushModeMark {}
impl StreamingDataDriftMark for DecayModeMark {}

/// Batch drift detector for continuous (floating-point) features. Compares a provided runtime
/// dataset against a fixed baseline distribution using histogram binning.
///
/// The baseline histogram is built once at construction and held until [`reset_baseline`] is
/// called. Runtime data is binned on each call to [`compute_drift`] and discarded immediately
/// after — no state is accumulated between calls.
///
/// # Bin count
///
/// The number of histogram bins is derived automatically from the baseline data using one of
/// three heuristics, selected via [`QuantileType`]:
///
/// - **[`FreedmanDiaconis`]** *(default)*: `width = 2 * IQR * n^(-1/3)`, `k = ceil((max - min) / width)`.
///   Robust to outliers. Preferred for most use cases.
/// - **[`Scott`]**: `width = 3.49 * σ * n^(-1/3)`. Assumes approximately normal data. Sensitive
///   to outliers in the tails.
/// - **[`Sturges`]**: `k = floor(ln(n)) + 1`. Simple log-based rule. Works best for small,
///   roughly normal datasets.
///
/// [`reset_baseline`]: ContinuousDataDrift::reset_baseline
/// [`compute_drift`]: ContinuousDataDrift::compute_drift
/// [`FreedmanDiaconis`]: QuantileType::FreedmanDiaconis
/// [`Scott`]: QuantileType::Scott
/// [`Sturges`]: QuantileType::Sturges
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

    fn num_examples(&self) -> f64 {
        self.rt_bins.iter().sum()
    }

    fn container_type(&self) -> DriftContainerType {
        DriftContainerType::Continuous
    }
}

impl ContinuousDataDrift {
    /// Construct a new instance from a baseline dataset. The baseline is sorted and used to
    /// define histogram bin edges.
    ///
    /// `quantile_type` controls how many bins are derived from the baseline. The bin count
    /// determines the resolution of the drift signal — more bins capture finer distributional
    /// shifts but require more runtime data per bin to be statistically meaningful. Pass `None`
    /// to use the default. Options:
    ///
    /// - [`FreedmanDiaconis`] *(default)*: `width = 2 * IQR * n^(-1/3)`. Robust to outliers.
    ///   Preferred for most use cases.
    /// - [`Scott`]: `width = 3.49 * σ * n^(-1/3)`. Assumes approximately normal data; sensitive
    ///   to outliers in the tails.
    /// - [`Sturges`]: `k = floor(log2(n)) + 1`. Simple rule that works best for small, roughly
    ///   normal datasets.
    ///
    /// Returns [`DriftError::EmptyBaselineData`] if the baseline slice has fewer than 2 elements,
    /// or [`DriftError::NaNValueError`] if any value is NaN.
    ///
    /// [`FreedmanDiaconis`]: QuantileType::FreedmanDiaconis
    /// [`Scott`]: QuantileType::Scott
    /// [`Sturges`]: QuantileType::Sturges
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

    /// Compute drift between the baseline and the provided runtime dataset. The runtime data is
    /// binned against the baseline histogram edges, drift is computed, and the runtime bins are
    /// cleared. Each call is stateless with respect to prior runtime data.
    ///
    /// To compute drift across multiple criteria, use [`ContinuousDataDrift::compute_drift_multiple_criteria`]
    ///
    /// Returns [`DriftError::EmptyRuntimeData`] if `runtime_data` is empty.
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

    /// Compute drift between the baseline and the provided runtime dataset for multiple data drift
    /// types. The runtime data is binned against the baseline histogram edges, drift is computed,
    /// and the runtime bins are cleared. Each call is stateless with respect to prior runtime data.
    ///
    /// This method is much more efficient for computing drift across multiple criteria as it only
    /// requires a single build of the runtime data distribution representation.
    ///
    /// Returns [`DriftError::EmptyRuntimeData`] if `runtime_data` is empty.
    pub fn compute_drift_multiple_criteria(
        &mut self,
        runtime_data: &[f64],
        drift_types: &[DataDriftType],
    ) -> Result<Vec<f64>, DriftError> {
        self.build_rt_hist(runtime_data)?;
        let drift = drift_types
            .iter()
            .map(|drift_type| global_compute_drift(self, *drift_type))
            .collect();
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

    /// Replace the baseline with a new dataset. The bin count is recomputed from the new data
    /// using the same [`QuantileType`] as construction. Any previously accumulated runtime bins
    /// are reset to zero.
    ///
    /// Returns [`DriftError::EmptyBaselineData`] if the slice has fewer than 2 elements.
    pub fn reset_baseline(&mut self, baseline_slice: &[f64]) -> Result<(), DriftError> {
        self.baseline.reset(baseline_slice)?;
        self.init_runtime_containers();
        Ok(())
    }

    /// The number of histogram bins derived from the baseline dataset.
    pub fn n_bins(&self) -> usize {
        self.baseline.n_bins
    }

    /// Export the baseline bin proportions. Each value represents the proportion of baseline
    /// samples that fell into the corresponding bin.
    pub fn export_baseline(&self) -> Vec<f64> {
        self.baseline.export_baseline()
    }
}

/// Streaming drift detector for continuous (floating-point) features. Maintains a running
/// histogram of observed runtime data that is compared against a fixed baseline distribution.
///
/// The type parameter `M` controls the window management strategy and is set at construction:
///
/// - [`FlushModeMark`]: accumulates data until a sample count or time cadence threshold is
///   reached, then hard-resets the stream window. Exposes [`flush`] and [`last_flush`].
/// - [`DecayModeMark`]: applies exponential decay α = 0.5^(1/half_life) to all bin counts on
///   each [`compute_drift`] call, giving a recency-weighted distribution with no hard cutoff.
///   Does not expose [`flush`] or [`last_flush`].
///
/// # Bin count
///
/// The number of histogram bins is derived from the baseline data using one of three heuristics,
/// selected via [`QuantileType`]:
///
/// - **[`FreedmanDiaconis`]** *(default)*: `width = 2 * IQR * n^(-1/3)`, `k = ceil((max - min) / width)`.
///   Robust to outliers. Preferred for most use cases.
/// - **[`Scott`]**: `width = 3.49 * σ * n^(-1/3)`. Assumes approximately normal data.
/// - **[`Sturges`]**: `k = floor(ln(n)) + 1`. Log-based rule, best for small datasets.
///
/// [`flush`]: StreamingContinuousDataDrift::flush
/// [`last_flush`]: StreamingContinuousDataDrift::last_flush
/// [`compute_drift`]: StreamingContinuousDataDrift::compute_drift
/// [`FreedmanDiaconis`]: QuantileType::FreedmanDiaconis
/// [`Scott`]: QuantileType::Scott
/// [`Sturges`]: QuantileType::Sturges
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

    fn num_examples(&self) -> f64 {
        self.total_stream_size
    }

    fn container_type(&self) -> DriftContainerType {
        DriftContainerType::Continuous
    }
}

impl StreamingContinuousDataDrift<DecayModeMark> {
    /// Construct a decay-mode stream. On each [`compute_drift`] or
    /// [`compute_drift_multiple_criteria`] call, all bin counts and `total_stream_size` are
    /// multiplied by α = 0.5^(1/`half_life`), where `half_life` is the number of seconds after
    /// which a sample's weight is halved. Older data is continuously down-weighted rather than
    /// discarded, giving a recency-weighted view of the distribution with no hard resets.
    ///
    /// `quantile_type` controls histogram bin count. See [`ContinuousDataDrift::new_from_baseline`]
    /// for a full description of each option. Pass `None` to use [`FreedmanDiaconis`] (default).
    ///
    /// `half_life_opt`: decay half-life in seconds. A shorter half-life makes the signal more
    /// sensitive to recent shifts at the cost of higher variance — older data loses influence
    /// quickly. A longer half-life produces a smoother, more stable signal that responds slowly
    /// to new patterns. Defaults to 86,400 (24 hours), meaning a sample's contribution is halved
    /// after 24 hours worth of [`compute_drift`] calls.
    ///
    /// When computing multiple drift metrics on the same accumulated state, use
    /// [`compute_drift_multiple_criteria`] — decay is applied once before all metrics are
    /// evaluated. Calling [`compute_drift`] in a loop applies decay on each call.
    ///
    /// Returns [`DriftError::EmptyBaselineData`] if the baseline has fewer than 2 elements.
    ///
    /// [`FreedmanDiaconis`]: QuantileType::FreedmanDiaconis
    /// [`compute_drift`]: StreamingContinuousDataDrift::compute_drift
    /// [`compute_drift_multiple_criteria`]: StreamingContinuousDataDrift::compute_drift_multiple_criteria
    pub fn new_decay(
        baseline_data: &[f64],
        quantile_type: Option<QuantileType>,
        half_life_opt: Option<NonZeroU64>,
    ) -> Result<StreamingContinuousDataDrift<DecayModeMark>, DriftError> {
        let baseline =
            BaselineContinuousBins::new(baseline_data, quantile_type.unwrap_or_default())?;
        let bl_hist_len = baseline.baseline_hist.len();
        let stream_bins: Vec<f64> = vec![0_f64; bl_hist_len];
        let half_life = half_life_opt.unwrap_or(NonZeroU64::new(DEFAULT_DECAY_HALF_LIFE).unwrap());
        let mode = StreamModeInner::ExponentialDecay(0.5_f64.powf(1_f64 / half_life.get() as f64));

        Ok(StreamingContinuousDataDrift {
            stream_bins,
            baseline,
            total_stream_size: 0_f64,
            mode,
            _mark: PhantomData,
        })
    }

    /// Compute drift between the accumulated stream and the baseline. Applies exponential decay
    /// to all bin counts before computing, down-weighting older data by α = 0.5^(1/half_life).
    ///
    /// To compute multiple metrics on the same decayed state, use
    /// [`compute_drift_multiple_criteria`] instead. Each call to this method applies decay once,
    /// so calling it repeatedly for different metrics will compound the decay.
    ///
    /// Returns [`DriftError::EmptyRuntimeData`] if no data has been accumulated.
    ///
    /// [`compute_drift_multiple_criteria`]: StreamingContinuousDataDrift::compute_drift_multiple_criteria
    pub fn compute_drift(&mut self, drift_type: DataDriftType) -> Result<f64, DriftError> {
        if self.is_empty() {
            return Err(DriftError::EmptyRuntimeData);
        }
        self.apply_decay();
        Ok(global_compute_drift(self, drift_type))
    }

    /// Compute multiple drift metrics against the accumulated stream in a single call. Decay is
    /// applied once before all metrics are evaluated, ensuring all results reflect the same
    /// decayed state. Prefer this over calling [`compute_drift`] in a loop when multiple metrics
    /// are needed simultaneously.
    ///
    /// Returns [`DriftError::EmptyRuntimeData`] if no data has been accumulated.
    ///
    /// [`compute_drift`]: StreamingContinuousDataDrift::compute_drift
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
            *bin = *bin * decay_factor;
        }
        self.total_stream_size = self.total_stream_size * decay_factor;
    }

    /// Push a single example into the stream.
    #[inline]
    pub fn update_stream(&mut self, runtime_example: f64) {
        let idx = self.baseline.resolve_bin(runtime_example);

        self.stream_bins[idx] += 1_f64;
        self.total_stream_size += 1_f64;
    }

    /// Push a batch of examples into the stream.
    ///
    /// Returns [`DriftError::EmptyRuntimeData`] if the slice is empty.
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
    /// Construct a flush-mode stream. The stream accumulates data until either `flush_size_opt`
    /// samples have been observed or `flush_cadence_opt` has elapsed since the last flush —
    /// whichever is reached first — at which point all accumulated runtime data is cleared and
    /// the window restarts fresh.
    ///
    /// `quantile_type` controls histogram bin count. See [`ContinuousDataDrift::new_from_baseline`]
    /// for a full description of each option. Pass `None` to use [`FreedmanDiaconis`] (default).
    ///
    /// `flush_size_opt`: number of accumulated samples that triggers an automatic flush. A lower
    /// value means more frequent resets and a more responsive signal, but each window contains
    /// fewer samples making the drift estimate noisier. Defaults to 1,000,000.
    ///
    /// `flush_cadence_opt`: time elapsed since the last flush that triggers an automatic flush,
    /// regardless of sample count. The time check is amortized over batches of 256 pushes to
    /// avoid reading the clock on every sample. Defaults to 86,400 seconds (24 hours).
    ///
    /// Returns [`DriftError::EmptyBaselineData`] if the baseline has fewer than 2 elements.
    ///
    /// [`FreedmanDiaconis`]: QuantileType::FreedmanDiaconis
    pub fn new_flush(
        baseline_data: &[f64],
        quantile_type: Option<QuantileType>,
        flush_size_opt: Option<u64>,
        flush_cadence_opt: Option<Duration>,
    ) -> Result<StreamingContinuousDataDrift<FlushModeMark>, DriftError> {
        let baseline =
            BaselineContinuousBins::new(baseline_data, quantile_type.unwrap_or_default())?;
        let bl_hist_len = baseline.baseline_hist.len();
        let stream_bins: Vec<f64> = vec![0_f64; bl_hist_len];
        let flush_size = flush_size_opt.unwrap_or(DEFAULT_MAX_STREAM_SIZE);
        let cadence = flush_cadence_opt.unwrap_or(Duration::new(DEFAULT_STREAM_FLUSH_CADENCE, 0));
        let mode = StreamModeInner::Flush {
            size: flush_size as f64,
            cadence,
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
    /// Compute drift between the accumulated stream and the baseline.
    ///
    /// To compute multiple metrics on the same accumulated state, use
    /// [`compute_drift_multiple_criteria`] instead.
    ///
    /// Returns [`DriftError::EmptyRuntimeData`] if no data has been accumulated since the last
    /// flush.
    ///
    /// [`compute_drift_multiple_criteria`]: StreamingContinuousDataDrift::compute_drift_multiple_criteria
    pub fn compute_drift(&mut self, drift_type: DataDriftType) -> Result<f64, DriftError> {
        if self.is_empty() {
            return Err(DriftError::EmptyRuntimeData);
        }
        Ok(global_compute_drift(self, drift_type))
    }

    /// Compute multiple drift metrics against the accumulated stream in a single call. Prefer
    /// this over calling [`compute_drift`] in a loop when multiple metrics are needed
    /// simultaneously.
    ///
    /// Returns [`DriftError::EmptyRuntimeData`] if no data has been accumulated since the last
    /// flush.
    ///
    /// [`compute_drift`]: StreamingContinuousDataDrift::compute_drift
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

    /// Push a single example into the stream. A flush is triggered before the item is recorded
    /// if the flush size or cadence threshold has been reached, starting a fresh window.
    #[inline]
    pub fn update_stream(&mut self, runtime_example: f64) {
        let idx = self.baseline.resolve_bin(runtime_example);
        if self.mode.needs_flush(self.total_stream_size) {
            self.flush()
        }
        self.stream_bins[idx] += 1_f64;
        self.total_stream_size += 1_f64;
    }

    /// Push a batch of examples into the stream. Each item is checked against flush thresholds
    /// individually, so a flush may occur mid-batch if the size threshold is crossed.
    ///
    /// Returns [`DriftError::EmptyRuntimeData`] if the slice is empty.
    pub fn update_stream_batch(&mut self, runtime_slice: &[f64]) -> Result<(), DriftError> {
        if runtime_slice.is_empty() {
            return Err(DriftError::EmptyRuntimeData);
        }

        for item in runtime_slice {
            self.update_stream(*item)
        }

        Ok(())
    }

    /// Manually flush the stream, clearing all accumulated runtime data. The baseline is not
    /// affected. The flush timestamp is reset so the cadence timer restarts from this point.
    pub fn flush(&mut self) {
        self.flush_runtime_stream();
        self.mode.perform_flush();
    }

    /// Returns the number of seconds elapsed since the last flush.
    pub fn last_flush(&self) -> u64 {
        self.mode.last_flush()
    }
}

#[allow(private_bounds)]
impl<M: StreamingDataDriftMark> StreamingContinuousDataDrift<M> {
    /// Returns `true` if no data has been accumulated since construction or the last flush.
    pub fn is_empty(&self) -> bool {
        self.stream_bins.iter().sum::<f64>() == 0_f64
    }

    /// Replace the baseline with a new dataset. The bin count is recomputed from the new data
    /// using the same [`QuantileType`] as construction. All accumulated stream data is cleared
    /// and the flush timestamp is reset.
    ///
    /// Returns [`DriftError::EmptyBaselineData`] if the slice has fewer than 2 elements.
    pub fn reset_baseline(&mut self, baseline_slice: &[f64]) -> Result<(), DriftError> {
        self.baseline.reset(baseline_slice)?;
        self.stream_bins = vec![0_f64; self.baseline.baseline_hist.len()];
        self.total_stream_size = 0_f64;
        self.mode.perform_flush();
        Ok(())
    }

    /// The number of samples accumulated in the stream since the last flush. In decay mode this
    /// reflects the effective (decayed) sample count rather than the raw push count.
    pub fn total_samples(&self) -> usize {
        self.total_stream_size as usize
    }

    /// The number of histogram bins derived from the baseline dataset.
    pub fn n_bins(&self) -> usize {
        self.baseline.n_bins
    }

    /// Export a point-in-time snapshot of the stream state as a map with three keys:
    ///
    /// - `"binEdges"`: the histogram bin edge values defining the boundaries between bins.
    /// - `"baselineBins"`: proportional bin distribution of the baseline dataset.
    /// - `"streamBins"`: proportional bin distribution of the currently accumulated stream data.
    ///
    /// All bin values are normalized to proportions in `[0, 1]`.
    ///
    /// Returns [`DriftError::EmptyRuntimeData`] if no data has been accumulated.
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

/// Batch drift detector for categorical (label) features. Compares a provided runtime dataset
/// against a fixed baseline distribution using a label-frequency histogram.
///
/// The baseline histogram is built once at construction and held until [`reset_baseline`] is
/// called. Runtime data is binned on each call to [`compute_drift`] and discarded immediately
/// after — no state is accumulated between calls.
///
/// # Bin count
///
/// The number of bins equals the number of unique values observed in the baseline dataset plus
/// one additional "other" bucket. Any runtime value not present in the baseline is routed to
/// this "other" bin rather than being silently ignored, so novel categories are always reflected
/// in the drift signal.
///
/// # Type bounds
///
/// `T` must implement [`Hash`], [`Ord`], and [`Clone`]. In practice `T` is typically `String` or
/// `&str`, but any hashable, ordered label type works (e.g. an enum that derives those traits).
///
/// [`reset_baseline`]: CategoricalDataDrift::reset_baseline
/// [`compute_drift`]: CategoricalDataDrift::compute_drift
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

    fn num_examples(&self) -> f64 {
        self.rt_bins.iter().sum()
    }

    fn container_type(&self) -> DriftContainerType {
        DriftContainerType::Categorical
    }
}

impl<T: Hash + Ord + Clone> CategoricalDataDrift<T> {
    /// Construct a new instance from a baseline dataset. The baseline is used to build a
    /// label-frequency histogram with one bin per unique value, plus one reserved "other" bin
    /// for values not present in the baseline.
    ///
    /// Returns [`DriftError::EmptyBaselineData`] if `baseline_data` is empty.
    pub fn new(baseline_data: &[T]) -> Result<CategoricalDataDrift<T>, DriftError> {
        if baseline_data.is_empty() {
            return Err(DriftError::EmptyBaselineData);
        }

        let baseline = BaselineCategoricalBins::new(baseline_data)?;
        let num_bins = baseline.baseline_bins.len();
        let rt_bins: Vec<f64> = vec![0_f64; num_bins];

        Ok(CategoricalDataDrift { baseline, rt_bins })
    }

    /// Compute drift between the baseline and the provided runtime dataset. The runtime data is
    /// binned against the baseline label map, drift is computed, and the runtime bins are
    /// cleared. Each call is stateless with respect to prior runtime data.
    ///
    /// To compute drift across multiple criteria, use [`CategoricalDataDrift::compute_drift_multiple_criteria`]
    ///
    /// Runtime labels not seen in the baseline are accumulated in the "other" bin.
    ///
    /// Returns [`DriftError::EmptyRuntimeData`] if `runtime_data` is empty.
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

    /// Compute drift between the baseline and the provided runtime dataset for multiple drift
    /// metric types. The runtime data is binned against the baseline label map, drift is computed,
    /// and the runtime bins are cleared. Each call is stateless with respect to prior runtime data.
    ///
    /// This method is much more efficient for computing drift across multiple criteria as it only
    /// requires a single build of the runtime data distribution representation.
    ///
    /// Runtime labels not seen in the baseline are accumulated in the "other" bin.
    ///
    /// Returns [`DriftError::EmptyRuntimeData`] if `runtime_data` is empty.
    pub fn compute_drift_multiple_criteria(
        &mut self,
        runtime_data: &[T],
        drift_types: &[DataDriftType],
    ) -> Result<Vec<f64>, DriftError> {
        self.build_rt_hist(runtime_data)?;
        let drift = drift_types
            .iter()
            .map(|drift_type| global_compute_drift(self, *drift_type))
            .collect();
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

    /// Replace the baseline with a new dataset. The bin count is recomputed from the new data —
    /// the number of bins becomes the new cardinality plus one "other" bin. Any previously
    /// accumulated runtime bins are reset to zero.
    ///
    /// Returns [`DriftError::EmptyBaselineData`] if `new_baseline` is empty.
    pub fn reset_baseline(&mut self, new_baseline: &[T]) -> Result<(), DriftError> {
        self.baseline.reset(new_baseline)?;
        let num_bins = self.baseline.baseline_bins.len();

        // pay the cost to reallocate bins in order to have correct size
        // not common path
        self.rt_bins = vec![0_f64; num_bins];
        Ok(())
    }

    /// Export the baseline label proportions as a map from label to proportion. Each value
    /// represents the fraction of baseline samples with that label.
    pub fn export_baseline(&self) -> HashMap<T, f64> {
        self.baseline.export_baseline()
    }

    pub fn num_bins(&self) -> usize {
        self.rt_bins.len()
    }
}

/// Streaming drift detector for categorical (label) features. Maintains a running histogram of
/// observed runtime labels that is compared against a fixed baseline distribution.
///
/// The type parameter `M` controls the window management strategy and is set at construction:
///
/// - [`FlushModeMark`]: accumulates data until a sample count or time cadence threshold is
///   reached, then hard-resets the stream window. Exposes [`flush`] and [`last_flush`].
/// - [`DecayModeMark`]: applies exponential decay α = 0.5^(1/half_life) to all bin counts on
///   each [`compute_drift`] call, giving a recency-weighted distribution with no hard cutoff.
///   Does not expose [`flush`] or [`last_flush`].
///
/// # Bin count
///
/// The number of bins equals the cardinality of the baseline dataset plus one "other" bin for
/// labels not observed at baseline time. Novel runtime labels are always captured in the signal.
///
/// # Type bounds
///
/// `T` must implement [`Hash`], [`Ord`], and [`Clone`]. Typically `String` or `&str`, but any
/// hashable, ordered label type works.
///
/// [`flush`]: StreamingCategoricalDataDrift::flush
/// [`last_flush`]: StreamingCategoricalDataDrift::last_flush
/// [`compute_drift`]: StreamingCategoricalDataDrift::compute_drift
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

    fn num_examples(&self) -> f64 {
        self.total_stream_size
    }

    fn container_type(&self) -> DriftContainerType {
        DriftContainerType::Categorical
    }
}

impl<T: Hash + Ord + Clone> StreamingCategoricalDataDrift<T, FlushModeMark> {
    /// Construct a flush-mode stream. The stream accumulates data until either `flush_size_opt`
    /// samples have been observed or `flush_cadence_opt` has elapsed since the last flush —
    /// whichever is reached first — at which point all accumulated runtime data is cleared and
    /// the window restarts fresh.
    ///
    /// `flush_size_opt`: number of accumulated samples that triggers an automatic flush. A lower
    /// value means more frequent resets and a more responsive signal, but each window contains
    /// fewer samples making the drift estimate noisier. Defaults to 1,000,000.
    ///
    /// `flush_cadence_opt`: time elapsed since the last flush that triggers an automatic flush,
    /// regardless of sample count. The time check is amortized over batches of 256 pushes to
    /// avoid reading the clock on every sample. Defaults to 86,400 seconds (24 hours).
    ///
    /// Returns [`DriftError::EmptyBaselineData`] if `baseline_data` is empty.
    pub fn new_flush(
        baseline_data: &[T],
        flush_size_opt: Option<u64>,
        flush_cadence_opt: Option<Duration>,
    ) -> Result<StreamingCategoricalDataDrift<T, FlushModeMark>, DriftError> {
        let baseline = BaselineCategoricalBins::new(baseline_data)?;
        let bl_hist_len = baseline.baseline_bins.len();
        let stream_bins: Vec<f64> = vec![0_f64; bl_hist_len];
        let size = flush_size_opt.unwrap_or(DEFAULT_MAX_STREAM_SIZE);
        let cadence = flush_cadence_opt.unwrap_or(Duration::new(DEFAULT_STREAM_FLUSH_CADENCE, 0));
        let mode = StreamModeInner::Flush {
            size: size as f64,
            cadence,
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

    /// Compute drift between the accumulated stream and the baseline.
    ///
    /// To compute multiple metrics on the same accumulated state, use
    /// [`compute_drift_multiple_criteria`] instead.
    ///
    /// Returns [`DriftError::EmptyRuntimeData`] if no data has been accumulated since the last
    /// flush.
    ///
    /// [`compute_drift_multiple_criteria`]: StreamingCategoricalDataDrift::compute_drift_multiple_criteria
    pub fn compute_drift(&mut self, drift_type: DataDriftType) -> Result<f64, DriftError> {
        if self.is_empty() {
            return Err(DriftError::EmptyRuntimeData);
        }
        Ok(global_compute_drift(self, drift_type))
    }

    /// Compute multiple drift metrics against the accumulated stream in a single call. Prefer
    /// this over calling [`compute_drift`] in a loop when multiple metrics are needed
    /// simultaneously.
    ///
    /// Returns [`DriftError::EmptyRuntimeData`] if no data has been accumulated since the last
    /// flush.
    ///
    /// [`compute_drift`]: StreamingCategoricalDataDrift::compute_drift
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

    /// Push a single label into the stream. A flush is triggered before the item is recorded
    /// if the flush size or cadence threshold has been reached, starting a fresh window.
    #[inline]
    pub fn update_stream(&mut self, item: &T) {
        let idx = self.baseline.get_bin(item);
        if self.mode.needs_flush(self.total_stream_size) {
            self.flush();
        }
        self.stream_bins[idx] += 1_f64;
        self.total_stream_size += 1_f64;
    }

    /// Push a batch of labels into the stream. Each item is checked against flush thresholds
    /// individually, so a flush may occur mid-batch if the size threshold is crossed.
    ///
    /// Returns [`DriftError::EmptyRuntimeData`] if the slice is empty.
    pub fn update_stream_batch(&mut self, runtime_data: &[T]) -> Result<(), DriftError> {
        if runtime_data.is_empty() {
            return Err(DriftError::EmptyRuntimeData);
        }

        for cat in runtime_data.iter() {
            self.update_stream(cat)
        }

        Ok(())
    }

    /// Manually flush the stream, clearing all accumulated runtime data. The baseline is not
    /// affected. The flush timestamp is reset so the cadence timer restarts from this point.
    pub fn flush(&mut self) {
        self.flush_runtime_stream();
        self.mode.perform_flush();
    }
}

impl<T: Hash + Ord + Clone> StreamingCategoricalDataDrift<T, DecayModeMark> {
    /// Construct a decay-mode stream. On each [`compute_drift`] or
    /// [`compute_drift_multiple_criteria`] call, all bin counts and `total_stream_size` are
    /// multiplied by α = 0.5^(1/`half_life`), where `half_life` is the number of seconds after
    /// which a sample's weight is halved. Older data is continuously down-weighted rather than
    /// discarded, giving a recency-weighted view of the distribution with no hard resets.
    ///
    /// `half_life_opt`: decay half-life in seconds. A shorter half-life makes the signal more
    /// sensitive to recent shifts at the cost of higher variance — older data loses influence
    /// quickly. A longer half-life produces a smoother, more stable signal that responds slowly
    /// to new patterns. Defaults to 86,400 (24 hours), meaning a sample's contribution is halved
    /// after 24 hours worth of [`compute_drift`] calls.
    ///
    /// When computing multiple drift metrics on the same accumulated state, use
    /// [`compute_drift_multiple_criteria`] — decay is applied once before all metrics are
    /// evaluated. Calling [`compute_drift`] in a loop applies decay on each call.
    ///
    /// Returns [`DriftError::EmptyBaselineData`] if `baseline_data` is empty.
    ///
    /// [`compute_drift`]: StreamingCategoricalDataDrift::compute_drift
    /// [`compute_drift_multiple_criteria`]: StreamingCategoricalDataDrift::compute_drift_multiple_criteria
    pub fn new_decay(
        baseline_data: &[T],
        half_life_opt: Option<NonZeroU64>,
    ) -> Result<StreamingCategoricalDataDrift<T, DecayModeMark>, DriftError> {
        let baseline = BaselineCategoricalBins::new(baseline_data)?;
        let bl_hist_len = baseline.baseline_bins.len();
        let stream_bins: Vec<f64> = vec![0_f64; bl_hist_len];
        let half_life = half_life_opt.unwrap_or(NonZeroU64::new(DEFAULT_DECAY_HALF_LIFE).unwrap());
        let mode = StreamModeInner::ExponentialDecay(0.5_f64.powf(1_f64 / half_life.get() as f64));

        Ok(StreamingCategoricalDataDrift {
            stream_bins,
            baseline,
            total_stream_size: 0_f64,
            mode,
            _mark: PhantomData,
        })
    }

    /// Compute drift between the accumulated stream and the baseline. Applies exponential decay
    /// to all bin counts before computing, down-weighting older data by α = 0.5^(1/half_life).
    ///
    /// To compute multiple metrics on the same decayed state, use
    /// [`compute_drift_multiple_criteria`] instead. Each call to this method applies decay once,
    /// so calling it repeatedly for different metrics will compound the decay.
    ///
    /// Returns [`DriftError::EmptyRuntimeData`] if no data has been accumulated.
    ///
    /// [`compute_drift_multiple_criteria`]: StreamingCategoricalDataDrift::compute_drift_multiple_criteria
    pub fn compute_drift(&mut self, drift_type: DataDriftType) -> Result<f64, DriftError> {
        if self.is_empty() {
            return Err(DriftError::EmptyRuntimeData);
        }
        self.apply_decay();
        Ok(global_compute_drift(self, drift_type))
    }

    /// Compute multiple drift metrics against the accumulated stream in a single call. Decay is
    /// applied once before all metrics are evaluated, ensuring all results reflect the same
    /// decayed state. Prefer this over calling [`compute_drift`] in a loop when multiple metrics
    /// are needed simultaneously.
    ///
    /// Returns [`DriftError::EmptyRuntimeData`] if no data has been accumulated.
    ///
    /// [`compute_drift`]: StreamingCategoricalDataDrift::compute_drift
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
            *bin = *bin * decay_factor;
        }
        self.total_stream_size = self.total_stream_size * decay_factor;
    }

    /// Push a single label into the stream.
    #[inline]
    pub fn update_stream(&mut self, item: &T) {
        let idx = self.baseline.get_bin(item);
        self.stream_bins[idx] += 1_f64;
        self.total_stream_size += 1_f64;
    }

    /// Push a batch of labels into the stream.
    ///
    /// Returns [`DriftError::EmptyRuntimeData`] if the slice is empty.
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

#[allow(private_bounds)]
impl<T: Hash + Ord + Clone, M: StreamingDataDriftMark> StreamingCategoricalDataDrift<T, M> {
    /// Returns `true` if no data has been accumulated since construction or the last flush.
    pub fn is_empty(&self) -> bool {
        self.stream_bins.iter().sum::<f64>() == 0_f64
    }

    /// Replace the baseline with a new dataset. The bin count is recomputed from the new data —
    /// the number of bins becomes the new cardinality plus one "other" bin. All accumulated
    /// stream data is cleared and the flush timestamp is reset.
    ///
    /// Returns [`DriftError::EmptyBaselineData`] if `new_baseline` is empty.
    pub fn reset_baseline(&mut self, new_baseline: &[T]) -> Result<(), DriftError> {
        self.baseline.reset(new_baseline)?;
        self.mode.perform_flush();
        self.init_stream_bins();
        self.total_stream_size = 0_f64;
        Ok(())
    }

    /// The number of samples accumulated in the stream since the last flush. In decay mode this
    /// reflects the effective (decayed) sample count rather than the raw push count.
    pub fn total_samples(&self) -> usize {
        self.total_stream_size.floor() as usize
    }

    /// Returns the number of seconds elapsed since the last flush. Returns `0` in decay mode
    /// as flush semantics do not apply.
    pub fn last_flush(&self) -> u64 {
        self.mode.last_flush()
    }

    /// Export a point-in-time snapshot of the stream as a map from label to raw (un-normalized)
    /// bin count. The "other" bin for unseen labels is not included in the returned map.
    ///
    /// To compute proportional distributions, divide each value by [`total_samples`].
    ///
    /// [`total_samples`]: StreamingCategoricalDataDrift::total_samples
    pub fn export_snapshot(&self) -> HashMap<T, f64> {
        self.baseline
            .idx_map
            .iter()
            .map(|(feat_name, i)| (feat_name.clone(), self.stream_bins[*i]))
            .collect()
    }

    /// Export the baseline label proportions as a map from label to proportion. Each value
    /// represents the fraction of baseline samples with that label.
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

    pub fn num_bins(&self) -> usize {
        self.stream_bins.len()
    }
}

#[cfg(test)]
mod continuous_tests {
    use super::*;

    #[test]
    fn test_continuous_baseline_builds_expected_bins() {
        let baseline = [1.0, 2.0, 3.0, 4.0];
        let psi = ContinuousDataDrift::new_from_baseline(None, &baseline).unwrap();

        let expected_bins =
            super::super::distribution::QuantileType::FreedmanDiaconis.compute_num_bins(&baseline);

        assert_eq!(psi.baseline.bin_edges.len(), expected_bins - 2);
        assert_eq!(psi.rt_bins.len(), expected_bins);
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
        let mut streaming =
            StreamingContinuousDataDrift::new_flush(&baseline, None, None, None).unwrap();

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
        let mut streaming =
            StreamingContinuousDataDrift::new_flush(&baseline, None, None, None).unwrap();

        streaming.update_stream_batch(&[1.0, 2.0, 3.0]).unwrap();
        streaming.flush();

        assert_eq!(streaming.total_samples(), 0);
    }

    // --- error paths ---

    #[test]
    fn continuous_batch_empty_baseline_returns_err() {
        assert!(ContinuousDataDrift::new_from_baseline(None, &[1.0]).is_err());
    }

    #[test]
    fn continuous_batch_nan_baseline_returns_err() {
        assert!(ContinuousDataDrift::new_from_baseline(None, &[1.0, f64::NAN, 3.0]).is_err());
    }

    #[test]
    fn continuous_batch_empty_runtime_returns_err() {
        let mut det =
            ContinuousDataDrift::new_from_baseline(None, &[1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        assert!(det
            .compute_drift(&[], DataDriftType::PopulationStabilityIndex)
            .is_err());
    }

    #[test]
    fn continuous_streaming_empty_batch_returns_err() {
        let mut s =
            StreamingContinuousDataDrift::new_flush(&[1.0, 2.0, 3.0, 4.0, 5.0], None, None, None)
                .unwrap();
        assert!(s.update_stream_batch(&[]).is_err());
    }

    #[test]
    fn continuous_streaming_compute_on_empty_returns_err() {
        let mut s =
            StreamingContinuousDataDrift::new_flush(&[1.0, 2.0, 3.0, 4.0, 5.0], None, None, None)
                .unwrap();
        assert!(s
            .compute_drift(DataDriftType::PopulationStabilityIndex)
            .is_err());
    }

    // --- statelessness ---

    #[test]
    fn continuous_batch_compute_drift_is_stateless() {
        let baseline: Vec<f64> = (0..50).map(|i| i as f64).collect();
        let runtime: Vec<f64> = (10..40).map(|i| i as f64).collect();
        let mut det = ContinuousDataDrift::new_from_baseline(None, &baseline).unwrap();

        let d1 = det
            .compute_drift(&runtime, DataDriftType::PopulationStabilityIndex)
            .unwrap();
        let d2 = det
            .compute_drift(&runtime, DataDriftType::PopulationStabilityIndex)
            .unwrap();
        assert!((d1 - d2).abs() < 1e-12);
    }

    // --- all drift types ---

    #[test]
    fn continuous_batch_all_drift_types_finite_nonnegative() {
        let baseline: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let runtime: Vec<f64> = (20..80).map(|i| i as f64).collect();
        let mut det = ContinuousDataDrift::new_from_baseline(None, &baseline).unwrap();

        for drift_type in [
            DataDriftType::PopulationStabilityIndex,
            DataDriftType::KullbackLeibler,
            DataDriftType::JensenShannon,
            DataDriftType::WassersteinDistance,
        ] {
            let v = det.compute_drift(&runtime, drift_type).unwrap();
            assert!(v.is_finite(), "{drift_type:?} produced non-finite value");
            assert!(v >= 0.0, "{drift_type:?} produced negative value");
        }
    }

    // --- reset_baseline ---

    #[test]
    fn continuous_batch_reset_baseline_changes_n_bins() {
        let mut det =
            ContinuousDataDrift::new_from_baseline(None, &[1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        let old_bins = det.n_bins();

        let large_baseline: Vec<f64> = (0..200).map(|i| i as f64).collect();
        det.reset_baseline(&large_baseline).unwrap();

        assert_ne!(det.n_bins(), old_bins);
        assert_eq!(det.rt_bins.len(), det.n_bins());
    }

    // --- export_baseline ---

    #[test]
    fn continuous_batch_export_baseline_sums_to_one() {
        let baseline: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let det = ContinuousDataDrift::new_from_baseline(None, &baseline).unwrap();
        let sum: f64 = det.export_baseline().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);
    }

    // --- streaming flush-mode extras ---

    #[test]
    fn continuous_streaming_flush_is_empty_after() {
        let baseline: Vec<f64> = (0..50).map(|i| i as f64).collect();
        let mut s = StreamingContinuousDataDrift::new_flush(&baseline, None, None, None).unwrap();
        s.update_stream_batch(&[1.0, 2.0, 3.0]).unwrap();
        assert!(!s.is_empty());
        s.flush();
        assert!(s.is_empty());
    }

    #[test]
    fn continuous_streaming_reset_baseline_clears_stream() {
        let baseline: Vec<f64> = (0..50).map(|i| i as f64).collect();
        let mut s = StreamingContinuousDataDrift::new_flush(&baseline, None, None, None).unwrap();
        s.update_stream_batch(&[10.0, 20.0, 30.0]).unwrap();

        let new_baseline: Vec<f64> = (0..100).map(|i| i as f64).collect();
        s.reset_baseline(&new_baseline).unwrap();

        assert!(s.is_empty());
        assert_eq!(s.total_samples(), 0);
        assert_eq!(s.stream_bins.len(), s.baseline.baseline_hist.len());
    }

    #[test]
    fn continuous_streaming_export_snapshot_empty_returns_err() {
        let baseline: Vec<f64> = (0..50).map(|i| i as f64).collect();
        let s = StreamingContinuousDataDrift::new_flush(&baseline, None, None, None).unwrap();
        assert!(s.export_snapshot().is_err());
    }

    #[test]
    fn continuous_streaming_export_snapshot_has_expected_keys() {
        let baseline: Vec<f64> = (0..50).map(|i| i as f64).collect();
        let mut s = StreamingContinuousDataDrift::new_flush(&baseline, None, None, None).unwrap();
        s.update_stream_batch(&[10.0, 20.0, 30.0]).unwrap();

        let snap = s.export_snapshot().unwrap();
        assert!(snap.contains_key("binEdges"));
        assert!(snap.contains_key("baselineBins"));
        assert!(snap.contains_key("streamBins"));
    }

    // --- decay mode ---

    #[test]
    fn continuous_decay_empty_compute_returns_err() {
        let baseline: Vec<f64> = (0..50).map(|i| i as f64).collect();
        let mut s = StreamingContinuousDataDrift::new_decay(
            &baseline,
            Some(super::super::distribution::QuantileType::default()),
            Some(std::num::NonZeroU64::new(1).unwrap()),
        )
        .unwrap();
        assert!(s
            .compute_drift(DataDriftType::PopulationStabilityIndex)
            .is_err());
    }

    #[test]
    fn continuous_decay_reduces_total_samples() {
        // half_life=1 → α=0.5, so after one compute_drift call total_samples halves
        let baseline: Vec<f64> = (0..50).map(|i| i as f64).collect();
        let mut s = StreamingContinuousDataDrift::new_decay(
            &baseline,
            Some(super::super::distribution::QuantileType::default()),
            Some(std::num::NonZeroU64::new(1).unwrap()),
        )
        .unwrap();

        let data: Vec<f64> = (0..100).map(|i| (i % 50) as f64).collect();
        s.update_stream_batch(&data).unwrap();
        assert_eq!(s.total_samples(), 100);

        s.compute_drift(DataDriftType::PopulationStabilityIndex)
            .unwrap();
        assert!(s.total_samples() < 100);
    }

    #[test]
    fn continuous_decay_multiple_criteria_applies_decay_once() {
        // half_life=1 → α=0.5. After compute_drift_multiple_criteria with N metrics,
        // total_samples should be floor(100*0.5)=50, same as a single compute_drift call.
        // Calling compute_drift three times separately would give floor(floor(floor(100*0.5)*0.5)*0.5)=12.
        let baseline: Vec<f64> = (0..50).map(|i| i as f64).collect();
        let half_life = std::num::NonZeroU64::new(1).unwrap();
        let qt = super::super::distribution::QuantileType::default();

        let mut s_multi =
            StreamingContinuousDataDrift::new_decay(&baseline, Some(qt), Some(half_life)).unwrap();
        let data: Vec<f64> = (0..100).map(|i| (i % 50) as f64).collect();
        s_multi.update_stream_batch(&data).unwrap();
        s_multi
            .compute_drift_multiple_criteria(&[
                DataDriftType::PopulationStabilityIndex,
                DataDriftType::KullbackLeibler,
                DataDriftType::JensenShannon,
            ])
            .unwrap();
        let samples_multi = s_multi.total_samples();

        let qt2 = super::super::distribution::QuantileType::default();
        let mut s_single =
            StreamingContinuousDataDrift::new_decay(&baseline, Some(qt2), Some(half_life)).unwrap();
        s_single.update_stream_batch(&data).unwrap();
        s_single
            .compute_drift(DataDriftType::PopulationStabilityIndex)
            .unwrap();
        let samples_single = s_single.total_samples();

        assert_eq!(samples_multi, samples_single);
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
        let mut streaming =
            StreamingCategoricalDataDrift::new_flush(&baseline, None, None).unwrap();

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

    // --- error paths ---

    #[test]
    fn categorical_batch_empty_baseline_returns_err() {
        let empty: &[&str] = &[];
        assert!(CategoricalDataDrift::new(empty).is_err());
    }

    #[test]
    fn categorical_batch_empty_runtime_returns_err() {
        let mut det = CategoricalDataDrift::new(&["a", "b", "c"]).unwrap();
        let empty: &[&str] = &[];
        assert!(det
            .compute_drift(empty, DataDriftType::PopulationStabilityIndex)
            .is_err());
    }

    #[test]
    fn categorical_streaming_empty_batch_returns_err() {
        let mut s = StreamingCategoricalDataDrift::new_flush(&["a", "b"], None, None).unwrap();
        let empty: &[&str] = &[];
        assert!(s.update_stream_batch(empty).is_err());
    }

    #[test]
    fn categorical_streaming_compute_on_empty_returns_err() {
        let mut s = StreamingCategoricalDataDrift::new_flush(&["a", "b"], None, None).unwrap();
        assert!(s
            .compute_drift(DataDriftType::PopulationStabilityIndex)
            .is_err());
    }

    // --- novel label routing ---

    #[test]
    fn categorical_batch_novel_label_routes_to_other_bin() {
        // runtime with only unseen labels should produce higher drift than runtime matching baseline
        let baseline = ["a", "b", "a", "b", "a"];
        let mut det = CategoricalDataDrift::new(&baseline).unwrap();

        let matching_drift = det
            .compute_drift(
                &["a", "b", "a", "b", "a"],
                DataDriftType::PopulationStabilityIndex,
            )
            .unwrap();
        let novel_drift = det
            .compute_drift(
                &["x", "y", "z", "x", "y"],
                DataDriftType::PopulationStabilityIndex,
            )
            .unwrap();

        assert!(novel_drift > matching_drift);
    }

    // --- all drift types ---

    #[test]
    fn categorical_batch_all_drift_types_finite_nonnegative() {
        let baseline = ["a", "a", "b", "b", "c"];
        let runtime = ["a", "b", "b", "c", "c"];
        let mut det = CategoricalDataDrift::new(&baseline).unwrap();

        for drift_type in [
            DataDriftType::PopulationStabilityIndex,
            DataDriftType::KullbackLeibler,
            DataDriftType::JensenShannon,
            DataDriftType::WassersteinDistance,
        ] {
            let v = det.compute_drift(&runtime, drift_type).unwrap();
            assert!(v.is_finite(), "{drift_type:?} produced non-finite value");
            assert!(v >= 0.0, "{drift_type:?} produced negative value");
        }
    }

    // --- reset_baseline ---

    #[test]
    fn categorical_batch_reset_baseline_changes_bin_count() {
        let mut det = CategoricalDataDrift::new(&["a", "b"]).unwrap();
        assert_eq!(det.rt_bins.len(), 3); // 2 labels + other

        det.reset_baseline(&["a", "b", "c", "d"]).unwrap();
        assert_eq!(det.rt_bins.len(), 5); // 4 labels + other
    }

    // --- export_baseline ---

    #[test]
    fn categorical_batch_export_baseline_contains_all_labels() {
        let baseline = ["a", "b", "c", "a", "b"];
        let det = CategoricalDataDrift::new(&baseline).unwrap();
        let exported = det.export_baseline();

        assert!(exported.contains_key("a"));
        assert!(exported.contains_key("b"));
        assert!(exported.contains_key("c"));
        let sum: f64 = exported.values().sum();
        assert!((sum - 1.0).abs() < 1e-9);
    }

    // --- streaming flush-mode extras ---

    #[test]
    fn categorical_streaming_flush_resets_stream() {
        let mut s = StreamingCategoricalDataDrift::new_flush(&["a", "b"], None, None).unwrap();
        s.update_stream_batch(&["a", "b", "a"]).unwrap();
        assert!(!s.is_empty());
        s.flush();
        assert!(s.is_empty());
        assert_eq!(s.total_samples(), 0);
    }

    #[test]
    fn categorical_streaming_novel_label_accumulates_in_other_bin() {
        let mut s = StreamingCategoricalDataDrift::new_flush(&["a", "b"], None, None).unwrap();
        // push only labels not in the baseline
        s.update_stream_batch(&["x", "y", "x", "z"]).unwrap();

        // drift should be elevated since all traffic is in the other bin
        let drift = s
            .compute_drift(DataDriftType::PopulationStabilityIndex)
            .unwrap();
        assert!(drift > 0.5);
    }

    #[test]
    fn categorical_streaming_reset_baseline_clears_stream() {
        let mut s = StreamingCategoricalDataDrift::new_flush(&["a", "b"], None, None).unwrap();
        s.update_stream_batch(&["a", "b"]).unwrap();

        s.reset_baseline(&["x", "y", "z"]).unwrap();
        assert!(s.is_empty());
        assert_eq!(s.total_samples(), 0);
        assert_eq!(s.stream_bins.len(), 4); // 3 labels + other
    }

    // --- decay mode ---

    #[test]
    fn categorical_decay_empty_compute_returns_err() {
        let mut s = StreamingCategoricalDataDrift::<&str, DecayModeMark>::new_decay(
            &["a", "b"],
            Some(std::num::NonZeroU64::new(1).unwrap()),
        )
        .unwrap();
        assert!(s
            .compute_drift(DataDriftType::PopulationStabilityIndex)
            .is_err());
    }

    #[test]
    fn categorical_decay_reduces_total_samples() {
        // half_life=1 → α=0.5
        let mut s = StreamingCategoricalDataDrift::<&str, DecayModeMark>::new_decay(
            &["a", "b"],
            Some(std::num::NonZeroU64::new(1).unwrap()),
        )
        .unwrap();

        let data: Vec<&str> = (0..100)
            .map(|i| if i % 2 == 0 { "a" } else { "b" })
            .collect();
        s.update_stream_batch(&data).unwrap();
        assert_eq!(s.total_samples(), 100);

        s.compute_drift(DataDriftType::PopulationStabilityIndex)
            .unwrap();
        assert!(s.total_samples() < 100);
    }

    #[test]
    fn categorical_decay_multiple_criteria_applies_decay_once() {
        let half_life = std::num::NonZeroU64::new(1).unwrap();
        let data: Vec<&str> = (0..100)
            .map(|i| if i % 2 == 0 { "a" } else { "b" })
            .collect();

        let mut s_multi = StreamingCategoricalDataDrift::<&str, DecayModeMark>::new_decay(
            &["a", "b"],
            Some(half_life),
        )
        .unwrap();
        s_multi.update_stream_batch(&data).unwrap();
        s_multi
            .compute_drift_multiple_criteria(&[
                DataDriftType::PopulationStabilityIndex,
                DataDriftType::KullbackLeibler,
                DataDriftType::JensenShannon,
            ])
            .unwrap();
        let samples_multi = s_multi.total_samples();

        let mut s_single = StreamingCategoricalDataDrift::<&str, DecayModeMark>::new_decay(
            &["a", "b"],
            Some(half_life),
        )
        .unwrap();
        s_single.update_stream_batch(&data).unwrap();
        s_single
            .compute_drift(DataDriftType::PopulationStabilityIndex)
            .unwrap();
        let samples_single = s_single.total_samples();

        assert_eq!(samples_multi, samples_single);
    }
}
