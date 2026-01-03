pub mod baseline;
pub mod data_drift;
pub mod drift_metrics;
use std::hash::Hash;

/// A trait to define what can be treated as string like in this context, what we need here is that
/// is can be represented as &str, can be hashed, can be compared for eqaulity and can be
/// transformed into an owned String.
pub trait StringLike: AsRef<str> + Eq + Hash + ToString {}
impl<T> StringLike for T where T: AsRef<str> + Eq + Hash + ToString {}

const DEFAULT_STREAM_FLUSH: i64 = 3600 * 24;
const MAX_STREAM_SIZE: usize = 1_000_000;
// read in from user defined env var or set to default epsilon
// optional user config
