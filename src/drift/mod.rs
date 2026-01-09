pub mod baseline;
/// The data drift module is intended to provide types that can serve to identify drift in the
/// distribution of some dataset, for example  a feature dataset, or composed to track drift across
/// and entire feature set. This module can also serve to provide proxies for model drift. By
/// leveraging the same techinques used to identify data drfit, model drift can also be
/// approximated. When the distribution of inference scores drifts significantly, that is probably
/// a decent sign for a deeper investigation.
pub mod data_drift;
pub mod drift_metrics;
use once_cell::sync::Lazy;
use std::hash::Hash;

/// A trait to define what can be treated as string like in this context, what we need here is that
/// is can be represented as &str, can be hashed, can be compared for eqaulity and can be
/// transformed into an owned String.
pub trait StringLike: AsRef<str> + Eq + Hash + ToString {}
impl<T> StringLike for T where T: AsRef<str> + Eq + Hash + ToString {}

const DEFAULT_STREAM_FLUSH: i64 = 3600 * 24;
const MAX_STREAM_SIZE: Lazy<usize> = Lazy::new(|| {
    let default = 1_000_000_usize;
    let Ok(str_val) = std::env::var("FAIR_PERF_MAX_STREAM_SIZE") else {
        return default;
    };

    str_val.parse().unwrap_or(default)
});
// read in from user defined env var or set to default epsilon
// optional user config
