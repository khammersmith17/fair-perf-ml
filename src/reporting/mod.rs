use crate::metrics;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Type aliases to document in code where `HashMap<T, f32>' where T is a metric differs from drift
/// computation and actual computation of the associated statistic for the dataset
pub type DataBiasRuntimeReport = HashMap<metrics::DataBiasMetric, f32>;
pub type ModelBiasRuntimeReport = HashMap<metrics::ModelBiasMetric, f32>;
pub type BinaryClassificationRuntimeReport = HashMap<metrics::ClassificationEvaluationMetric, f32>;
pub type LinearRegressionRuntimeReport = HashMap<metrics::LinearRegressionEvaluationMetric, f32>;
pub type LogisticRegressionRuntimeReport = HashMap<metrics::ClassificationEvaluationMetric, f32>;

/// Type aliases to document in code where `HashMap<T, f32>' maps to results from analysis, and not drift
/// degree.
pub type ModelBiasAnalysisReport = HashMap<metrics::ModelBiasMetric, f32>;
pub type DataBiasAnalysisReport = HashMap<metrics::DataBiasMetric, f32>;
pub type BinaryClassificationAnalysisReport = HashMap<metrics::ClassificationEvaluationMetric, f32>;
pub type LinearRegressionAnalysisReport = HashMap<metrics::LinearRegressionEvaluationMetric, f32>;
pub type LogisticRegressionAnalysisReport = HashMap<metrics::ClassificationEvaluationMetric, f32>;

/// Type aliases for drift snapshot payloads — point-in-time metric values captured before or
/// after a runtime check, used as the inputs to runtime comparison functions.
pub type ModelBiasDriftSnapshot = HashMap<metrics::ModelBiasMetric, f32>;
pub type DataBiasDriftSnapshot = HashMap<metrics::DataBiasMetric, f32>;
pub type BinaryClassificationDriftSnapshot = HashMap<metrics::ClassificationEvaluationMetric, f32>;
pub type LinearRegressionDriftSnapshot = HashMap<metrics::LinearRegressionEvaluationMetric, f32>;
pub type LogisticRegressionDriftSnapshot = HashMap<metrics::ClassificationEvaluationMetric, f32>;

/// Default drift threshold used when no explicit threshold is supplied (10% relative drift).
pub const DEFAULT_DRIFT_THRESHOLD: f32 = 0.10;

/// A single metric that drifted outside the acceptable threshold, paired with the magnitude
/// of the drift observed.
#[derive(Serialize, Deserialize, Debug, PartialEq)]
pub struct MetricDrift<T: metrics::MachineLearningMetric> {
    pub metric: T,
    pub drift: f32,
}

/// Result of a runtime "check". A runtime check evaluates the runtime data passed into any of
/// the utilities against the baseline set. This type contains a boolean pass/fail flag, which is
/// `false` when any of the metrics drift outside the defined threshold, and `true` when all
/// metrics are within the allowable drift threshold from the baseline. The `failed_report` field
/// will contain the metrics that drifted outside the allowable bounds and the degree of drift.
/// Implements `serde::Serialize` and `serde::Deserialize` so drift report payloads can be sent
/// or loaded from external sources.
#[derive(Serialize, Deserialize, Debug, PartialEq)]
pub struct DriftReport<T: metrics::MachineLearningMetric> {
    pub passed: bool,
    pub failed_report: Option<Vec<MetricDrift<T>>>,
}

impl<T: metrics::MachineLearningMetric> DriftReport<T>
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
pub(crate) mod py_report_utils {
    use super::{DriftReport, MetricDrift};
    use crate::metrics;
    use pyo3::{
        types::{IntoPyDict, PyDict, PyDictMethods, PyList, PyListMethods},
        Bound, PyResult, Python,
    };
    use serde::Serialize;

    /// Utility to coerce a report into a Python dictionary.
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
}
