use crate::metrics::MachineLearningMetric;
use serde::Serialize;
use std::collections::HashMap;

#[cfg(feature = "python")]
use pyo3::{
    types::{IntoPyDict, PyDict, PyDictMethods, PyList, PyListMethods},
    Bound, PyResult, Python,
};

#[derive(Serialize)]
pub struct MetricDrift<T> {
    metric: T,
    drift: f32,
}

#[derive(Serialize)]
pub struct DriftReport<T> {
    pub passed: bool,
    pub failed_report: Option<Vec<MetricDrift<T>>>,
}

impl<T> DriftReport<T>
where
    T: MachineLearningMetric + Serialize + std::fmt::Display,
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
    T: MachineLearningMetric + Serialize + std::fmt::Display,
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
