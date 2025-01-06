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
