use crate::model_perf::ModelPerformanceType;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Serialize, Deserialize)]
pub struct FailureRuntimeReturn {
    pub passed: bool,
    pub fail_report: Option<HashMap<String, String>>,
}

#[derive(Serialize, Deserialize)]
pub struct PassedRuntimeReturn {
    pub passed: bool,
}

pub enum ModelType {
    LinearRegression,
    LogisiticRegression,
    BinaryClassification,
}

impl TryFrom<&str> for ModelType {
    type Error = String;
    fn try_from(value: &str) -> Result<Self, Self::Error> {
        match value {
            "LinearRegression" => Ok(Self::LinearRegression),
            "LogisiticRegression" => Ok(Self::LogisiticRegression),
            "BinaryClassification" => Ok(Self::BinaryClassification),
            _ => Err("invalid model type".into()),
        }
    }
}

pub struct ModelPerformance<T>
where
    T: ModelPerformanceType,
{
    model_type: ModelType,
    performance_data: T,
}
