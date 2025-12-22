use crate::data_handler::{
    BiasDataPayload, BiasSegmentationCriteria, BiasSegmentationType, ConfusionMatrix,
};
use crate::errors::{BiasError, ModelBiasRuntimeError};
use crate::metrics::{ModelBiasMetric, ModelBiasMetricVec, FULL_MODEL_BIAS_METRICS};
use crate::runtime::ModelBiasRuntime;
use crate::zip_iters;
use std::collections::HashMap;
pub(crate) mod core;
pub mod statistics;
use crate::reporting::{DriftReport, ModelBiasAnalysisReport};
use core::post_training_bias;
pub mod streaming;

#[cfg(feature = "python")]
pub(crate) mod py_api {
    use super::core::post_training_bias;
    use super::DiscretePostTraining;
    use crate::data_handler::py_types_handler::{apply_label, report_to_py_dict};
    use crate::errors::BiasError;
    use crate::metrics::{ModelBiasMetric, ModelBiasMetricVec, FULL_MODEL_BIAS_METRICS};
    use crate::reporting::DriftReport;
    use crate::runtime::ModelBiasRuntime;
    use numpy::PyUntypedArray;
    use pyo3::{
        exceptions::PyTypeError,
        prelude::*,
        types::{IntoPyDict, PyDict},
        Bound, PyResult, Python,
    };
    use std::collections::HashMap;

    #[pyfunction]
    #[pyo3(signature = (baseline, latest, metrics, threshold=0.10))]
    pub fn py_model_bias_partial_check<'py>(
        py: Python<'py>,
        baseline: HashMap<String, f32>,
        latest: HashMap<String, f32>,
        metrics: Vec<String>,
        threshold: f32,
    ) -> PyResult<Bound<'py, PyDict>> {
        let metrics = match ModelBiasMetricVec::try_from(metrics.as_slice()) {
            Ok(m) => m,
            Err(e) => return Err(e.into()),
        };

        let current = match ModelBiasRuntime::try_from(latest) {
            Ok(obj) => obj,
            Err(e) => return Err(e.into()),
        };
        let baseline = match ModelBiasRuntime::try_from(baseline) {
            Ok(obj) => obj,
            Err(e) => return Err(e.into()),
        };
        let failure_report: HashMap<ModelBiasMetric, f32> =
            current.runtime_check(baseline, threshold, metrics.as_ref());

        let drift_report: DriftReport<ModelBiasMetric> = DriftReport::from_runtime(failure_report);

        let py_dict = drift_report.into_py_dict(py)?;

        Ok(py_dict)
    }

    #[pyfunction]
    #[pyo3(signature = (baseline, latest, threshold=0.10))]
    pub fn py_model_bias_runtime_check<'py>(
        py: Python<'py>,
        baseline: HashMap<String, f32>,
        latest: HashMap<String, f32>,
        threshold: f32,
    ) -> PyResult<Bound<'py, PyDict>> {
        let current = match ModelBiasRuntime::try_from(latest) {
            Ok(obj) => obj,
            Err(e) => return Err(e.into()),
        };
        let baseline = match ModelBiasRuntime::try_from(baseline) {
            Ok(obj) => obj,
            Err(e) => return Err(e.into()),
        };
        let failure_report: HashMap<ModelBiasMetric, f32> =
            current.runtime_check(baseline, threshold, &FULL_MODEL_BIAS_METRICS);

        let drift_report: DriftReport<ModelBiasMetric> = DriftReport::from_runtime(failure_report);

        let py_dict = drift_report.into_py_dict(py)?;

        Ok(py_dict)
    }

    #[pyfunction]
    #[pyo3(signature = (feature_array, ground_truth_array, prediction_array, feature_label_or_threshold,
        ground_truth_label_or_threshold, prediction_label_or_threshold))]
    pub fn py_model_bias_analyzer<'py>(
        py: Python<'py>,
        feature_array: &Bound<'py, PyUntypedArray>,
        ground_truth_array: &Bound<'py, PyUntypedArray>,
        prediction_array: &Bound<'py, PyUntypedArray>,
        feature_label_or_threshold: Bound<'py, PyAny>,
        ground_truth_label_or_threshold: Bound<'py, PyAny>,
        prediction_label_or_threshold: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let preds: Vec<i16> = match apply_label(py, prediction_array, prediction_label_or_threshold)
        {
            Ok(array) => array,
            Err(err) => return Err(PyTypeError::new_err(err.to_string())),
        };
        let gt: Vec<i16> =
            match apply_label(py, ground_truth_array, ground_truth_label_or_threshold) {
                Ok(array) => array,
                Err(err) => return Err(PyTypeError::new_err(err.to_string())),
            };
        let feats: Vec<i16> = match apply_label(py, feature_array, feature_label_or_threshold) {
            Ok(array) => array,
            Err(err) => return Err(PyTypeError::new_err(err.to_string())),
        };

        let post_training_data = DiscretePostTraining::new(&feats, &preds, &gt)
            .map_err(|e| <BiasError as Into<PyErr>>::into(e))?;

        let analysis_res = post_training_bias(&post_training_data);

        let py_dict = report_to_py_dict(py, analysis_res);
        Ok(py_dict)
    }
}

pub fn model_bias_runtime_check(
    baseline: ModelBiasAnalysisReport,
    latest: ModelBiasAnalysisReport,
    threshold: f32,
) -> Result<DriftReport<ModelBiasMetric>, ModelBiasRuntimeError> {
    let current = match ModelBiasRuntime::try_from(latest) {
        Ok(obj) => obj,
        Err(e) => return Err(e.into()),
    };
    let baseline = match ModelBiasRuntime::try_from(baseline) {
        Ok(obj) => obj,
        Err(e) => return Err(e.into()),
    };
    let failure_report: HashMap<ModelBiasMetric, f32> =
        current.runtime_check(baseline, threshold, &FULL_MODEL_BIAS_METRICS);

    let drift_report: DriftReport<ModelBiasMetric> = DriftReport::from_runtime(failure_report);

    Ok(drift_report)
}

pub fn model_bias_partial_runtime_check(
    baseline: ModelBiasAnalysisReport,
    latest: ModelBiasAnalysisReport,
    threshold: f32,
    metrics: ModelBiasMetricVec,
) -> Result<DriftReport<ModelBiasMetric>, ModelBiasRuntimeError> {
    let current = ModelBiasRuntime::try_from(latest)?;
    let baseline = ModelBiasRuntime::try_from(baseline)?;
    let failure_report: HashMap<ModelBiasMetric, f32> =
        current.runtime_check(baseline, threshold, metrics.as_ref());

    let drift_report: DriftReport<ModelBiasMetric> = DriftReport::from_runtime(failure_report);

    Ok(drift_report)
}

pub fn model_bias_analyzer<'a, F, P, G>(
    feature: BiasDataPayload<'a, F>,
    predictions: BiasDataPayload<'a, P>,
    ground_truth: BiasDataPayload<'a, G>,
) -> Result<ModelBiasAnalysisReport, BiasError>
where
    G: PartialEq + PartialOrd,
    F: PartialEq + PartialOrd,
    P: PartialEq + PartialOrd,
{
    let labeled_features = feature.generate_labeled_data();
    let labeled_gt = ground_truth.generate_labeled_data();
    let labeled_preds = predictions.generate_labeled_data();

    let post_training_base =
        DiscretePostTraining::new(&labeled_features, &labeled_preds, &labeled_gt)?;

    let analysis_res = post_training_bias(&post_training_base);
    Ok(analysis_res)
}

#[derive(Default)]
pub(crate) struct BucketGeneralizedEntropy {
    count_0: u64,
    count_1: u64,
    count_2: u64,
}

impl BucketGeneralizedEntropy {
    pub(crate) fn accumulate<T, P>(
        &mut self,
        y_true: &[T],
        true_seg: &BiasSegmentationCriteria<T>,
        y_pred: &[P],
        pred_seg: &BiasSegmentationCriteria<P>,
    ) where
        T: PartialOrd,
        P: PartialOrd,
    {
        for (t, p) in zip_iters!(y_true, y_pred) {
            let t_label = true_seg.label(t);
            let p_label = pred_seg.label(p);
            if !p_label && t_label {
                self.count_0 += 1;
            } else if p_label && t_label {
                self.count_1 += 1;
            } else {
                self.count_2 += 1
            }
        }
    }

    pub(crate) fn clear(&mut self) {
        self.count_0 = 0;
        self.count_1 = 0;
        self.count_2 = 0;
    }

    pub(crate) fn ge_snapshot(&self) -> f32 {
        let n = (self.count_0 + self.count_1 + self.count_2) as f32;
        // total benefits sum
        let s1 = (self.count_1 + (2 * self.count_2)) as f32;
        // squared benefits sum
        let s2 = (self.count_1 + (4 * self.count_2)) as f32;
        ((n.powi(2)) / (s1.powi(2))) * s2 - n
    }

    pub fn len(&self) -> u64 {
        self.count_0 + self.count_1 + self.count_2
    }
}

#[derive(Default)]
pub(crate) struct PostTrainingDistribution {
    len: u64,
    positive_gt: u64,
    positive_pred: u64,
}

impl PostTrainingDistribution {
    pub(crate) fn clear(&mut self) {
        self.len = 0;
        self.positive_gt = 0;
        self.positive_pred = 0;
    }
}

#[derive(Default)]
pub(crate) struct PostTraining {
    pub confusion_a: ConfusionMatrix,
    pub confusion_d: ConfusionMatrix,
    pub dist_a: PostTrainingDistribution,
    pub dist_d: PostTrainingDistribution,
}

pub(crate) struct DiscretePostTraining {
    post_training: PostTraining,
    ge: f32,
}

impl DiscretePostTraining {
    pub(crate) fn new(
        labeled_feature: &[i16],
        labeled_prediction: &[i16],
        labeled_gt: &[i16],
    ) -> Result<DiscretePostTraining, BiasError> {
        let post_training =
            PostTraining::new_from_labeled_data(labeled_feature, labeled_prediction, labeled_gt)?;

        let ge = statistics::inner::generalized_entropy(labeled_gt, labeled_prediction);

        Ok(DiscretePostTraining { post_training, ge })
    }
}

impl PostTraining {
    fn new_from_labeled_data(
        labeled_feature: &[i16],
        labeled_prediction: &[i16],
        labeled_gt: &[i16],
    ) -> Result<PostTraining, BiasError> {
        let labeled_seg = BiasSegmentationCriteria::new(1_i16, BiasSegmentationType::Label);

        let mut pt = PostTraining {
            confusion_a: ConfusionMatrix::default(),
            confusion_d: ConfusionMatrix::default(),
            dist_d: PostTrainingDistribution::default(),
            dist_a: PostTrainingDistribution::default(),
        };

        pt.accumulate_batch(
            labeled_feature,
            &labeled_seg,
            labeled_prediction,
            &labeled_seg,
            labeled_gt,
            &labeled_seg,
        )?;

        Ok(pt)
    }

    pub(crate) fn clear(&mut self) {
        self.confusion_a.clear();
        self.confusion_d.clear();
        self.dist_a.clear();
        self.dist_d.clear();
    }
}

impl PostTraining {
    pub(crate) fn new_from_segmentation_criteria<F, P, G>(
        features: &[F],
        feat_seg: &BiasSegmentationCriteria<F>,
        preds: &[P],
        pred_seg: &BiasSegmentationCriteria<P>,
        gt: &[G],
        gt_seg: &BiasSegmentationCriteria<G>,
    ) -> Result<PostTraining, BiasError>
    where
        F: PartialOrd,
        P: PartialOrd,
        G: PartialOrd,
    {
        let n = features.len();

        if (preds.len() != n || gt.len() != n) || n == 0 {
            return Err(BiasError::DataLengthError);
        };

        let mut post_t = PostTraining::default();
        post_t.accumulate_batch(features, feat_seg, preds, pred_seg, gt, gt_seg)?;
        Ok(post_t)
    }

    #[inline]
    pub(crate) fn accumulate_single(
        &mut self,
        is_a: bool,
        pred_is_positive: bool,
        gt_is_positive: bool,
    ) {
        self.dist_a.len += is_a as u64;
        self.dist_a.positive_pred += (is_a as usize * pred_is_positive as usize) as u64;
        self.dist_a.positive_gt += (is_a as usize * gt_is_positive as usize) as u64;

        self.dist_d.len += !is_a as u64;
        self.dist_d.positive_pred += (!is_a as usize * pred_is_positive as usize) as u64;
        self.dist_d.positive_gt += (!is_a as usize * gt_is_positive as usize) as u64;

        let is_true = pred_is_positive == gt_is_positive;
        let tp = (pred_is_positive && is_true) as usize;
        let tn = (!pred_is_positive && is_true) as usize;
        let fp = (pred_is_positive && !is_true) as usize;
        let r#fn = (!pred_is_positive && !is_true) as usize;

        let grp = is_a as usize;

        self.confusion_a.true_p += (grp * tp) as f32;
        self.confusion_a.true_n += (grp * tn) as f32;
        self.confusion_a.false_p += (grp * fp) as f32;
        self.confusion_a.false_n += (grp * r#fn) as f32;

        self.confusion_d.true_p += (grp * tp) as f32;
        self.confusion_d.true_n += (grp * tn) as f32;
        self.confusion_d.false_p += (grp * fp) as f32;
        self.confusion_d.false_n += (grp * r#fn) as f32;
        todo!()
    }

    pub(crate) fn accumulate_batch<F, P, G>(
        &mut self,
        features: &[F],
        feat_seg: &BiasSegmentationCriteria<F>,
        preds: &[P],
        pred_seg: &BiasSegmentationCriteria<P>,
        gt: &[G],
        gt_seg: &BiasSegmentationCriteria<G>,
    ) -> Result<(), BiasError>
    where
        F: PartialOrd,
        P: PartialOrd,
        G: PartialOrd,
    {
        let n = features.len();

        if (preds.len() != n || gt.len() != n) || n == 0 {
            return Err(BiasError::DataLengthError);
        }

        for (f, (p, gt)) in zip_iters!(features, preds, gt) {
            self.accumulate_single(feat_seg.label(f), pred_seg.label(p), gt_seg.label(gt));
        }
        Ok(())
    }
}

pub struct PostTrainingComputations {
    pub true_positives_a: f32,
    pub true_positives_d: f32,
    pub false_positives_a: f32,
    pub false_positives_d: f32,
    pub false_negatives_a: f32,
    pub false_negatives_d: f32,
    pub true_negatives_a: f32,
    pub true_negatives_d: f32,
}
