use super::{data_drift::StreamingDriftMode, distribution::QuantileType};
use crate::errors::DriftExportError;
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use serde_json::Value;
use std::path::PathBuf;

pub trait LoadDataDriftExport: DeserializeOwned {
    fn from_file(filepath: PathBuf) -> Result<Self, DriftExportError> {
        let file_data = std::fs::read_to_string(filepath)?;
        let export: Self = serde_json::from_str(&file_data)?;
        Ok(export)
    }

    fn from_bytes(export_bytes: &[u8]) -> Result<Self, DriftExportError> {
        let export: Self = serde_json::from_slice(export_bytes)?;
        Ok(export)
    }
}

#[derive(Debug, Deserialize, Serialize)]
pub struct ContinuousDriftBaselineExport {
    pub bin_edges: Vec<f64>,
    pub baseline_hist: Vec<f64>,
    pub quantile_type: QuantileType,
}

impl LoadDataDriftExport for ContinuousDriftBaselineExport {}

#[derive(Debug, Deserialize, Serialize)]
pub struct StreamingContinuousBaseExport {
    pub baseline: ContinuousDriftBaselineExport,
    pub stream_mode: StreamingDriftMode,
}

impl LoadDataDriftExport for StreamingContinuousBaseExport {}

#[derive(Debug, Deserialize, Serialize)]
pub struct StreamingContinuousStatefulExport {
    pub stream_bins: Vec<f64>,
    pub baseline: ContinuousDriftBaselineExport,
    pub stream_mode: StreamingDriftMode,
}

impl LoadDataDriftExport for StreamingContinuousStatefulExport {}

#[derive(Debug, Deserialize, Serialize)]
pub struct CategoricalDriftBaselineExport {
    pub baseline_hist: Vec<f64>,
    pub baseline_values: Vec<Value>,
}

impl LoadDataDriftExport for CategoricalDriftBaselineExport {}

#[derive(Debug, Deserialize, Serialize)]
pub struct StreamingCategoricalBaseExport {
    pub baseline: CategoricalDriftBaselineExport,
    pub stream_mode: StreamingDriftMode,
}

impl LoadDataDriftExport for StreamingCategoricalBaseExport {}

#[derive(Debug, Deserialize, Serialize)]
pub struct StreamingCategoricalStatefulExport {
    pub stream_bins: Vec<f64>,
    pub baseline: CategoricalDriftBaselineExport,

    pub stream_mode: StreamingDriftMode,
}

impl LoadDataDriftExport for StreamingCategoricalStatefulExport {}
