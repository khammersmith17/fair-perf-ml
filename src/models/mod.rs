use crate::DataBiasMetric;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Serialize, Deserialize)]
pub enum ModelType {
    LinearRegression,
    LogisticRegression,
    BinaryClassification,
}

impl TryFrom<&str> for ModelType {
    type Error = String;
    fn try_from(value: &str) -> Result<Self, Self::Error> {
        match value {
            "LinearRegression" => Ok(Self::LinearRegression),
            "LogisticRegression" => Ok(Self::LogisticRegression),
            "BinaryClassification" => Ok(Self::BinaryClassification),
            _ => Err("invalid model type".into()),
        }
    }
}
