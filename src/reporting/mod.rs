use crate::metrics;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Type aliases to document in code where `HashMap<T, f32>' where T is a metric differs from drift
/// computation and actual computation of the associated statistic for the dataset
pub type DataBiasRuntimeReport = HashMap<metrics::DataBiasMetric, f32>;
pub type ModelBiasRuntimeReport = HashMap<metrics::ModelBiasMetric, f32>;

/// Type aliases to document in code maps that represent result from analysis, and not drift degree
pub type ModelBiasAnalysisReport = HashMap<metrics::ModelBiasMetric, f32>;
pub type DataBiasAnalysisReport = HashMap<metrics::DataBiasMetric, f32>;
pub type BinaryClassificationReport = HashMap<metrics::ClassificationEvaluationMetric, f32>;
pub type LinearRegressionReport = HashMap<metrics::LinearRegressionEvaluationMetric, f32>;
pub type LogisticRegressionReport = HashMap<metrics::ClassificationEvaluationMetric, f32>;

#[cfg(feature = "python")]
use pyo3::{
    types::{IntoPyDict, PyDict, PyDictMethods, PyList, PyListMethods},
    Bound, PyResult, Python,
};

#[derive(Serialize, Deserialize)]
pub struct MetricDrift<T> {
    metric: T,
    drift: f32,
}

#[derive(Serialize, Deserialize)]
pub struct DriftReport<T> {
    pub passed: bool,
    pub failed_report: Option<Vec<MetricDrift<T>>>,
}

impl<T> DriftReport<T>
where
    T: metrics::MachineLearningMetric + Serialize + std::fmt::Display,
{
    pub(crate) fn from_runtime(runtime: HashMap<T, f32>) -> DriftReport<T> {
        if runtime.is_empty() {
            return DriftReport {
                passed: true,
                failed_report: None,
            };
        }

        let failed_report: Vec<MetricDrift<T>> = runtime
            .into_iter()
            .map(|(metric, drift)| MetricDrift { metric, drift })
            .collect();

        DriftReport {
            passed: false,
            failed_report: Some(failed_report),
        }
    }
}

#[cfg(feature = "python")]
impl<T> IntoPyDict<'_> for DriftReport<T>
where
    T: metrics::MachineLearningMetric + Serialize + std::fmt::Display,
{
    fn into_py_dict(self, py: Python<'_>) -> PyResult<Bound<'_, PyDict>> {
        let dict = PyDict::new(py);
        let Self {
            passed,
            failed_report,
        } = self;
        let _ = dict.set_item("passed", passed);

        if passed {
            return Ok(dict);
        }

        let fp = failed_report.unwrap();

        let failed_report_list = PyList::empty(py);

        for report in fp.into_iter() {
            let curr = PyDict::new(py);
            let MetricDrift { metric, drift } = report;
            let _ = curr.set_item("metric", metric.to_string());
            let _ = curr.set_item("drift", drift);
            failed_report_list.append(curr)?;
        }
        let _ = dict.set_item("failed_report", failed_report_list);

        Ok(dict)
    }
}
