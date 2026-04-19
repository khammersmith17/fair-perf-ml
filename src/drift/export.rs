use super::distribution::QuantileType;
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Deserialize, Serialize)]
pub struct ContinuousDriftBaselineExport {
    pub bin_edges: Vec<f64>,
    pub baseline_hist: Vec<f64>,
    pub quantile_type: QuantileType,
}

pub struct CategoricalDriftBaselineExport {
    pub baseline_hist: Vec<f64>,
    pub baseline_values: Vec<Value>,
}

