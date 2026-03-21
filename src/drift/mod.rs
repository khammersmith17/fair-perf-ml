pub mod baseline;
/// The data drift module is intended to provide types that can serve to identify drift in the
/// distribution of some dataset, for example  a feature dataset, or composed to track drift across
/// and entire feature set. This module can also serve to provide proxies for model drift. By
/// leveraging the same techinques used to identify data drfit, model drift can also be
/// approximated. When the distribution of inference scores drifts significantly, that is probably
/// a decent sign for a deeper investigation.
pub mod data_drift;
pub mod distribution;
pub mod drift_metrics;
#[cfg(feature = "python")]
pub(crate) mod python_impl;

const DEFAULT_STREAM_FLUSH_CADENCE: u64 = 3600 * 24;
const DEFAULT_MAX_STREAM_SIZE: u64 = 1_000_000_u64;
const DEFAULT_DECAY_HALF_LIFE: u64 = 86400; // Defaul half life 1 day
