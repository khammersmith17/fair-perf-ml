use super::{data_drift::StreamingDriftMode, distribution::QuantileType};
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Deserialize, Serialize)]
pub struct ContinuousDriftBaselineExport {
    pub bin_edges: Vec<f64>,
    pub baseline_hist: Vec<f64>,
    pub quantile_type: QuantileType,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct StreamingContinuousBaseExport {
    pub baseline: ContinuousDriftBaselineExport,
    pub stream_mode: StreamingDriftMode,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct StreamingContinuousStatefulExport {
    pub stream_bins: Vec<f64>,
    pub baseline: ContinuousDriftBaselineExport,
    pub stream_mode: StreamingDriftMode,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct CategoricalDriftBaselineExport {
    pub baseline_hist: Vec<f64>,
    pub baseline_values: Vec<Value>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct StreamingCategoricalBaseExport {
    pub baseline: CategoricalDriftBaselineExport,
    pub stream_mode: StreamingDriftMode,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct StreamingCategoricalStatefulExport {
    pub stream_bins: Vec<f64>,
    pub baseline: CategoricalDriftBaselineExport,

    pub stream_mode: StreamingDriftMode,
}
