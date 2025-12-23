use crate::data_handler::{BiasDataPayload, BiasSegmentationCriteria};
use crate::errors::{BiasError, DataBiasRuntimeError};
use crate::metrics::{DataBiasMetric, DataBiasMetricVec, FULL_DATA_BIAS_METRICS};
use crate::reporting::{DataBiasAnalysisReport, DriftReport, DEFAULT_DRIFT_THRESHOLD};
use crate::runtime::DataBiasRuntime;
use crate::zip_iters;
pub(crate) mod core;
pub mod statistics;
pub mod streaming;

#[cfg(feature = "python")]
pub(crate) mod py_api {
    use super::core::data_bias_analysis_core;
    use super::data_bias_runtime_check;
    use crate::data_handler::py_types_handler::{apply_label, report_to_py_dict, PyDictResult};
    use crate::errors::InvalidMetricError;
    use crate::metrics::{DataBiasMetric, DataBiasMetricVec};
    use crate::reporting::DriftReport;
    use crate::runtime::DataBiasRuntime;
    use numpy::PyUntypedArray;
    use pyo3::{exceptions::PyTypeError, prelude::*, types::IntoPyDict, Bound, Python};
    use std::collections::HashMap;

    /// Method to perform data bias analysis
    #[pyfunction]
    #[pyo3(signature = (feature_array, ground_truth_array, feature_label_or_threshold, ground_truth_label_or_threshold))]
    pub fn py_data_bias_analyzer<'py>(
        py: Python<'py>,
        feature_array: &Bound<'py, PyUntypedArray>,
        ground_truth_array: &Bound<'py, PyUntypedArray>,
        feature_label_or_threshold: Bound<'py, PyAny>, //fix
        ground_truth_label_or_threshold: Bound<'py, PyAny>, //fix
    ) -> PyDictResult<'py> {
        let gt = match apply_label(py, ground_truth_array, ground_truth_label_or_threshold) {
            Ok(array) => array,
            Err(err) => return Err(PyTypeError::new_err(err.to_string())),
        };

        let feats = match apply_label(py, feature_array, feature_label_or_threshold) {
            Ok(array) => array,
            Err(err) => return Err(PyTypeError::new_err(err.to_string())),
        };

        let res = match data_bias_analysis_core(gt, feats) {
            Ok(r) => r,
            Err(e) => return Err(e.into()),
        };

        let py_dict = report_to_py_dict(py, res);
        Ok(py_dict)
    }

    #[pyfunction]
    #[pyo3(signature = (baseline, latest, threshold=0.10))]
    pub fn py_data_bias_runtime_check<'py>(
        py: Python<'py>,
        baseline: HashMap<String, f32>,
        latest: HashMap<String, f32>,
        threshold: f32,
    ) -> PyDictResult<'py> {
        let bl = match convert_db_analysis(baseline) {
            Ok(b) => b,
            Err(e) => return Err(e.into()),
        };
        let rt = match convert_db_analysis(latest) {
            Ok(b) => b,
            Err(e) => return Err(e.into()),
        };

        let drift_report = match data_bias_runtime_check(bl, rt, Some(threshold)) {
            Ok(r) => r,
            Err(e) => return Err(e.into()),
        };

        Ok(drift_report.into_py_dict(py)?)
    }

    #[pyfunction]
    #[pyo3(signature = (baseline, latest, metrics, threshold=0.10))]
    pub fn py_data_bias_partial_check<'py>(
        py: Python<'py>,
        baseline: HashMap<String, f32>,
        latest: HashMap<String, f32>,
        metrics: Vec<String>,
        threshold: f32,
    ) -> PyDictResult<'py> {
        let metrics = match DataBiasMetricVec::try_from(metrics.as_slice()) {
            Ok(m) => m,
            Err(e) => return Err(e.into()),
        };
        let current = match DataBiasRuntime::try_from(latest) {
            Ok(obj) => obj,
            Err(e) => return Err(e.into()),
        };

        let baseline = match DataBiasRuntime::try_from(baseline) {
            Ok(obj) => obj,
            Err(e) => return Err(e.into()),
        };
        let failure_report: HashMap<DataBiasMetric, f32> =
            current.runtime_check(baseline, threshold, metrics.as_ref());

        let drift_report: DriftReport<DataBiasMetric> = DriftReport::from_runtime(failure_report);
        Ok(drift_report.into_py_dict(py)?)
    }

    // Internal method to take analysis report from python, limited to a string for the metric
    // labels, into enum metric labels to be used here internally.
    fn convert_db_analysis(
        report: HashMap<String, f32>,
    ) -> Result<super::DataBiasAnalysisReport, InvalidMetricError> {
        let mut invalid_metrics: Vec<String> = Vec::new();
        let mut res_map: super::DataBiasAnalysisReport = HashMap::with_capacity(7);

        for (metric_str, value) in report.into_iter() {
            if let Ok(m) = DataBiasMetric::try_from(metric_str.as_ref()) {
                res_map.insert(m, value);
            } else {
                invalid_metrics.push(metric_str);
            }
        }

        if !invalid_metrics.is_empty() {
            return Err(InvalidMetricError::DataBiasMetricError(invalid_metrics));
        }
        Ok(res_map)
    }
}

/// Function to perform analysis on a dataset pretrainingm, or without regards to a model
/// prediction. This is to be used to indicate a bias between groups in real world outcomes in
/// dataset. The features and ground truth are passed in a BiasDataPayload type, where the
/// segmentaion criteria and segmentation type are provided. This is in turn used to segmented the
/// data into the two facets for bias analysis. This is best used for point in time analysis.
pub fn data_bias_analyzer<'a, F, G>(
    feature: BiasDataPayload<'a, F>,
    ground_truth: BiasDataPayload<'a, G>,
) -> Result<DataBiasAnalysisReport, BiasError>
where
    G: PartialEq + PartialOrd,
    F: PartialEq + PartialOrd,
{
    let labeled_feats = feature.generate_labeled_data();
    let labeled_gt = ground_truth.generate_labeled_data();

    Ok(core::data_bias_analysis_core(labeled_feats, labeled_gt)?)
}

/// Function to perform runtime check across all available DataBias metrics, see
/// `metrics::DataBiasMetric` for the full list. The threshold determines whether the metric is
/// within the bounds of a "passing" score and represents the absolute percent drift from the
/// baseline metric score. This is optional and defaults to 0.1.
pub fn data_bias_runtime_check(
    baseline_report: DataBiasAnalysisReport,
    current_report: DataBiasAnalysisReport,
    threshold: Option<f32>,
) -> Result<DriftReport<DataBiasMetric>, DataBiasRuntimeError> {
    let t = threshold.unwrap_or(DEFAULT_DRIFT_THRESHOLD);
    let baseline = DataBiasRuntime::try_from(baseline_report)?;
    let current = DataBiasRuntime::try_from(current_report)?;
    let check_res = current.runtime_check(baseline, t, &FULL_DATA_BIAS_METRICS);

    Ok(DriftReport::from_runtime(check_res))
}
/// Function to perform runtime check across a subset of available DataBias metrics, see
/// `metrics::DataBiasMetric` for the full list. The method accepts a 'metrics::DataBiasMetricVec'
/// which implements 'From<Vec<DataBiasMetric>>' and 'From<&[T]>' where T is string like.
/// The threshold determines whether the metric is within the bounds of a "passing" score and
/// represents the absolute percent drift from the baseline metric score. This is optional and defaults to 0.1.
pub fn data_bias_partial_check(
    baseline_report: DataBiasAnalysisReport,
    current_report: DataBiasAnalysisReport,
    metrics: DataBiasMetricVec,
    threshold: Option<f32>,
) -> Result<DriftReport<DataBiasMetric>, DataBiasRuntimeError> {
    let t = threshold.unwrap_or(DEFAULT_DRIFT_THRESHOLD);
    let baseline = DataBiasRuntime::try_from(baseline_report)?;
    let current = DataBiasRuntime::try_from(current_report)?;
    let check_res = current.runtime_check(baseline, t, metrics.as_ref());

    Ok(DriftReport::from_runtime(check_res))
}

// TODO:
// instead of storing entire vector, store pos class count and neg class count for each facet

#[derive(Default)]
pub(crate) struct PreTrainingDistribution {
    pub positive: u64,
    pub len: u64,
}

impl PreTrainingDistribution {
    #[inline]
    pub(crate) fn acceptance(&self) -> f32 {
        self.positive as f32 / self.len as f32
    }

    fn clear(&mut self) {
        self.positive = 0_u64;
        self.len = 0_u64;
    }
}

#[derive(Default)]
pub(crate) struct PreTraining {
    facet_a: PreTrainingDistribution,
    facet_d: PreTrainingDistribution,
}

impl PreTraining {
    pub(crate) fn new_from_labeled(
        feature_data: &[i16],
        gt_data: &[i16],
    ) -> Result<PreTraining, BiasError> {
        let mut facet_a = PreTrainingDistribution::default();
        let mut facet_d = PreTrainingDistribution::default();

        for (f, gt) in zip_iters!(feature_data, gt_data) {
            if *f == 1_i16 {
                facet_a.positive += *gt as u64;
                facet_a.len += 1
            } else {
                facet_d.positive += *gt as u64;
                facet_d.len += 1
            }
        }

        if facet_a.positive == 0 || facet_d.positive == 0 {
            return Err(BiasError::NoFacetDeviation);
        }

        Ok(PreTraining { facet_a, facet_d })
    }

    pub(crate) fn size(&self) -> u64 {
        self.facet_a.len + self.facet_d.len
    }

    pub(crate) fn clear(&mut self) {
        self.facet_a.clear();
        self.facet_d.clear();
    }

    pub(crate) fn new_from_segmentation<F, G>(
        feature: &[F],
        feat_seg: &BiasSegmentationCriteria<F>,
        gt: &[G],
        gt_seg: &BiasSegmentationCriteria<G>,
    ) -> Result<PreTraining, BiasError>
    where
        F: PartialOrd,
        G: PartialOrd,
    {
        let mut len_a = 0_u64;
        let mut positive_a = 0_u64;

        let mut len_d = 0_u64;
        let mut positive_d = 0_u64;

        if len_a != len_d {
            return Err(BiasError::DataLengthError);
        }

        if len_a == 0 {
            return Err(BiasError::DataLengthError);
        }

        for (f, g) in zip_iters!(feature, gt) {
            let is_favored = feat_seg.label(f);
            let is_positive = gt_seg.label(g);

            len_a += is_favored as u64;
            positive_a += (is_favored && is_positive) as u64;
            len_d += !is_favored as u64;
            positive_d += (!is_favored && is_positive) as u64;
        }

        Ok(PreTraining {
            facet_a: PreTrainingDistribution {
                len: len_a,
                positive: positive_a,
            },
            facet_d: PreTrainingDistribution {
                len: len_d,
                positive: positive_d,
            },
        })
    }

    #[inline]
    fn accumulate_runtime_single<F, G>(
        &mut self,
        f: &F,
        feat_seg: &BiasSegmentationCriteria<F>,
        g: &G,

        gt_seg: &BiasSegmentationCriteria<G>,
    ) where
        F: PartialOrd,
        G: PartialOrd,
    {
        let is_a = feat_seg.label(f);
        let is_positive = gt_seg.label(g);

        self.facet_a.len += is_a as u64;
        self.facet_d.len += !is_a as u64;
        self.facet_a.positive += (is_a && is_positive) as u64;
        self.facet_d.positive += (!is_a && is_positive) as u64;
    }

    fn accumulate_runtime_batch<F, G>(
        &mut self,
        features: &[F],
        feat_seg: &BiasSegmentationCriteria<F>,
        gt: &[G],
        gt_seg: &BiasSegmentationCriteria<G>,
    ) -> Result<(), BiasError>
    where
        F: PartialOrd,
        G: PartialOrd,
    {
        if features.len() != gt.len() {
            return Err(BiasError::DataLengthError);
        }

        if features.is_empty() {
            return Err(BiasError::DataLengthError);
        }

        for (f, g) in zip_iters!(features, gt) {
            self.accumulate_runtime_single(f, feat_seg, g, gt_seg);
        }

        Ok(())
    }
}
