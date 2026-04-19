use super::distribution::QuantileType;
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Deserialize, Serialize)]
pub enum StreamingType {
    Decay,
    Flush,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct ContinuousDriftBaselineExport {
    pub bin_edges: Vec<f64>,
    pub baseline_hist: Vec<f64>,
    pub quantile_type: QuantileType,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct StreamingContinuousBaseExport {
    pub streaming_type: StreamingType,
    pub baseline: ContinuousDriftBaselineExport,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct StreamingContinuousStatefulExport {
    pub streaming_type: StreamingType,
    pub stream_bins: Vec<f64>,
    pub baseline: ContinuousDriftBaselineExport,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct CategoricalDriftBaselineExport {
    pub baseline_hist: Vec<f64>,
    pub baseline_values: Vec<Value>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct StreamingCategoricalBaseExport {
    pub streaming_type: StreamingType,
    pub baseline: CategoricalDriftBaselineExport,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct StreamingCategoricalStatefulExport {
    pub streaming_type: StreamingType,
    pub stream_bins: Vec<f64>,
    pub baseline: CategoricalDriftBaselineExport,
}
