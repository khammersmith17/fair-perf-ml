use crate::data_handler::{
    BiasDataPayload, BiasSegmentationCriteria, BiasSegmentationType,
    ConditionalConfusionPushPayload, ConfusionMatrix,
};
use crate::errors::{BiasError, ModelBiasRuntimeError, ModelPerfResult, ModelPerformanceError};
use crate::metrics::{ModelBiasMetric, FULL_MODEL_BIAS_METRICS};
use crate::runtime::ModelBiasRuntime;
use crate::zip_iters;
use std::collections::HashMap;
pub(crate) mod core;
pub mod statistics;
use crate::reporting::{DriftReport, ModelBiasAnalysisReport};
use core::post_training_bias;
#[cfg(feature = "python")]
pub(crate) mod python_impl;
pub mod streaming;

pub fn model_bias_runtime_check(
    baseline: ModelBiasAnalysisReport,
    latest: ModelBiasAnalysisReport,
    threshold: f32,
) -> Result<DriftReport<ModelBiasMetric>, ModelBiasRuntimeError> {
    let current = ModelBiasRuntime::try_from(latest)?;
    let baseline = ModelBiasRuntime::try_from(baseline)?;
    let failure_report: HashMap<ModelBiasMetric, f32> =
        current.runtime_check(baseline, threshold, &FULL_MODEL_BIAS_METRICS);

    let drift_report: DriftReport<ModelBiasMetric> = DriftReport::from_runtime(failure_report);

    Ok(drift_report)
}

pub fn model_bias_partial_runtime_check<V>(
    baseline: ModelBiasAnalysisReport,
    latest: ModelBiasAnalysisReport,
    threshold: f32,
    metrics: &[ModelBiasMetric],
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
    post_training_bias(&post_training_base)
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

#[derive(Default, Debug, PartialEq)]
pub(crate) struct PostTrainingDistribution {
    // total number of predictions made for the segmentation class
    pub(crate) len: u64,
    // number of positive predictions based on segmentation criteria
    pub(crate) positive_gt: u64,
    // number of positive ground truth outcomes based on segmentation criteria
    pub(crate) positive_pred: u64,
}

impl PostTrainingDistribution {
    pub(crate) fn clear(&mut self) {
        self.len = 0;
        self.positive_gt = 0;
        self.positive_pred = 0;
    }

    pub(crate) fn cond_acceptance(&self) -> ModelPerfResult<f32> {
        if self.positive_gt == 0_u64 {
            return Err(ModelPerformanceError::InvalidData);
        }
        Ok(self.positive_pred as f32 / self.positive_gt as f32)
    }

    pub(crate) fn positive_prediction_rate(&self) -> ModelPerfResult<f32> {
        if self.len == 0_u64 {
            return Err(ModelPerformanceError::EmptyDataVector);
        }
        Ok(self.positive_pred as f32 / self.len as f32)
    }

    pub(crate) fn conditional_rejection(&self) -> ModelPerfResult<f32> {
        let len = self.len as f32;
        let pos_pred = self.positive_pred as f32;
        let pos_gt = self.positive_gt as f32;

        let n = len - pos_gt;
        let d = len - pos_pred;

        if d == 0_f32 {
            return Err(ModelPerformanceError::InvalidData);
        }
        Ok(n / d)
    }
}

#[derive(Default, Debug)]
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

    /// The "is_positive" here is defined as being a positive segmentation as evaluated by the user
    /// provided BiasSegmentationCriteria. It does not necessarily refer to the accuracy of the
    /// prediction, though it may in the case this is a classification model as in that case the
    /// segmentation logic generally should follow the model inference score classification logic.
    /// This is where the updates to state happen.
    #[inline]
    pub(crate) fn accumulate_single(
        &mut self,
        is_a: bool,
        pred_is_positive: bool,
        gt_is_positive: bool,
    ) {
        self.dist_a.len += is_a as u64;
        self.dist_a.positive_pred += (is_a && pred_is_positive) as u64;
        self.dist_a.positive_gt += (is_a && gt_is_positive) as u64;

        self.dist_d.len += !is_a as u64;
        self.dist_d.positive_pred += (!is_a && pred_is_positive) as u64;
        self.dist_d.positive_gt += (!is_a && gt_is_positive) as u64;

        self.confusion_a
            .conditional_push(ConditionalConfusionPushPayload {
                cond: is_a,
                true_gt: gt_is_positive,
                true_pred: pred_is_positive,
            });
        self.confusion_d
            .conditional_push(ConditionalConfusionPushPayload {
                cond: !is_a,
                true_gt: gt_is_positive,
                true_pred: pred_is_positive,
            });
    }

    /// Requires the slices passed to be none empty. Will error in that case that the slices are
    /// not of the same length or the slices are empty.
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

#[cfg(test)]
mod model_bias_components {
    use super::*;
    use crate::data_handler::{BiasSegmentationCriteria, BiasSegmentationType};
    use crate::metrics::ModelBiasMetric as M;

    // --- shared test data ---

    // 8 samples: 4 advantaged (feat=1), 4 disadvantaged (feat=0).
    // All predictions match ground truth → all difference metrics should be zero.
    fn symmetric_data() -> (Vec<i32>, Vec<i32>, Vec<i32>) {
        let feat = vec![1, 1, 1, 1, 0, 0, 0, 0];
        let pred = vec![1, 1, 0, 0, 1, 1, 0, 0];
        let gt   = vec![1, 0, 1, 0, 1, 0, 1, 0];
        (feat, pred, gt)
    }

    fn seg() -> BiasSegmentationCriteria<i32> {
        BiasSegmentationCriteria::new(1_i32, BiasSegmentationType::Label)
    }

    fn full_model_bias_report(v: f32) -> crate::reporting::ModelBiasAnalysisReport {
        use std::collections::HashMap;
        let mut m: HashMap<M, f32> = HashMap::with_capacity(12);
        m.insert(M::DifferenceInPositivePredictedLabels, v);
        m.insert(M::DisparateImpact, v);
        m.insert(M::AccuracyDifference, v);
        m.insert(M::RecallDifference, v);
        m.insert(M::DifferenceInConditionalAcceptance, v);
        m.insert(M::DifferenceInAcceptanceRate, v);
        m.insert(M::SpecialityDifference, v);
        m.insert(M::DifferenceInConditionalRejection, v);
        m.insert(M::DifferenceInRejectionRate, v);
        m.insert(M::TreatmentEquity, v);
        m.insert(M::ConditionalDemographicDesparityPredictedLabels, v);
        m.insert(M::GeneralizedEntropy, v);
        m
    }

    // --- PostTrainingDistribution ---

    #[test]
    fn post_training_distribution_cond_acceptance_errors_when_no_gt_positives() {
        let d = PostTrainingDistribution { len: 5, positive_gt: 0, positive_pred: 3 };
        assert!(d.cond_acceptance().is_err());
    }

    #[test]
    fn post_training_distribution_positive_prediction_rate_errors_when_empty() {
        let d = PostTrainingDistribution::default();
        assert!(d.positive_prediction_rate().is_err());
    }

    #[test]
    fn post_training_distribution_conditional_rejection_errors_when_all_predicted_positive() {
        // d = len - pos_pred = 0 when all predicted positive
        let d = PostTrainingDistribution { len: 4, positive_gt: 2, positive_pred: 4 };
        assert!(d.conditional_rejection().is_err());
    }

    #[test]
    fn post_training_distribution_clear_resets() {
        let mut d = PostTrainingDistribution { len: 5, positive_gt: 3, positive_pred: 2 };
        d.clear();
        assert_eq!(d, PostTrainingDistribution::default());
    }

    // --- PostTraining::accumulate_batch ---

    #[test]
    fn post_training_accumulate_batch_increments_correctly() {
        let (feat, pred, gt) = symmetric_data();
        let feat_seg = seg();
        let pred_seg = seg();
        let gt_seg = seg();
        let mut pt = PostTraining::default();
        pt.accumulate_batch(&feat, &feat_seg, &pred, &pred_seg, &gt, &gt_seg).unwrap();

        // Each facet has 4 samples, 2 positive predictions, 2 positive ground truths.
        assert_eq!(pt.dist_a, PostTrainingDistribution { len: 4, positive_pred: 2, positive_gt: 2 });
        assert_eq!(pt.dist_d, PostTrainingDistribution { len: 4, positive_pred: 2, positive_gt: 2 });
    }

    #[test]
    fn post_training_accumulate_batch_mismatched_lengths_errors() {
        let feat_seg = seg();
        let pred_seg = seg();
        let gt_seg = seg();
        let mut pt = PostTraining::default();
        assert!(pt.accumulate_batch(&[1_i32, 0], &feat_seg, &[1_i32], &pred_seg, &[1_i32], &gt_seg).is_err());
    }

    #[test]
    fn post_training_accumulate_batch_empty_errors() {
        let feat_seg = seg();
        let pred_seg = seg();
        let gt_seg = seg();
        let mut pt = PostTraining::default();
        assert!(pt.accumulate_batch(&[], &feat_seg, &[], &pred_seg, &[], &gt_seg).is_err());
    }

    // --- PostTraining::clear ---

    #[test]
    fn post_training_clear_resets_all_fields() {
        let (feat, pred, gt) = symmetric_data();
        let feat_seg = seg();
        let pred_seg = seg();
        let gt_seg = seg();
        let mut pt = PostTraining::default();
        pt.accumulate_batch(&feat, &feat_seg, &pred, &pred_seg, &gt, &gt_seg).unwrap();
        pt.clear();
        assert_eq!(pt.dist_a, PostTrainingDistribution::default());
        assert_eq!(pt.dist_d, PostTrainingDistribution::default());
    }

    // --- PostTraining::new_from_segmentation_criteria ---

    #[test]
    fn post_training_new_from_seg_criteria_happy_path() {
        let (feat, pred, gt) = symmetric_data();
        let pt = PostTraining::new_from_segmentation_criteria(
            &feat, &seg(), &pred, &seg(), &gt, &seg(),
        ).unwrap();
        assert_eq!(pt.dist_a.len + pt.dist_d.len, 8);
    }

    #[test]
    fn post_training_new_from_seg_criteria_mismatched_lengths_errors() {
        assert!(PostTraining::new_from_segmentation_criteria(
            &[1_i32, 0], &seg(), &[1_i32], &seg(), &[1_i32], &seg(),
        ).is_err());
    }

    #[test]
    fn post_training_new_from_seg_criteria_empty_errors() {
        let s = BiasSegmentationCriteria::new(0_i32, BiasSegmentationType::Label);
        assert!(PostTraining::new_from_segmentation_criteria(
            &[], &s, &[], &s, &[], &s,
        ).is_err());
    }

    // --- BucketGeneralizedEntropy ---

    #[test]
    fn bucket_ge_len_is_sum_of_all_counts() {
        let mut ge = BucketGeneralizedEntropy::default();
        assert_eq!(ge.len(), 0);
        let gt = vec![1_i32, 0, 1, 0];
        let pred = vec![0_i32, 1, 1, 0];
        let gt_seg = BiasSegmentationCriteria::new(1_i32, BiasSegmentationType::Label);
        let pred_seg = BiasSegmentationCriteria::new(1_i32, BiasSegmentationType::Label);
        ge.accumulate(&gt, &gt_seg, &pred, &pred_seg);
        assert_eq!(ge.len(), 4);
    }

    #[test]
    fn bucket_ge_clear_resets_to_zero() {
        let mut ge = BucketGeneralizedEntropy::default();
        let gt = vec![1_i32, 0];
        let pred = vec![1_i32, 0];
        let gt_seg = BiasSegmentationCriteria::new(1_i32, BiasSegmentationType::Label);
        let pred_seg = BiasSegmentationCriteria::new(1_i32, BiasSegmentationType::Label);
        ge.accumulate(&gt, &gt_seg, &pred, &pred_seg);
        ge.clear();
        assert_eq!(ge.len(), 0);
    }

    #[test]
    fn bucket_ge_perfect_predictions_give_zero_entropy() {
        // All (gt=1,pred=1) → benefit=1 for all → mean=1 → (1/1)^2-1=0 → GE=0
        let mut ge = BucketGeneralizedEntropy::default();
        let gt = vec![1_i32, 1, 1, 1];
        let pred = vec![1_i32, 1, 1, 1];
        let gt_seg = BiasSegmentationCriteria::new(1_i32, BiasSegmentationType::Label);
        let pred_seg = BiasSegmentationCriteria::new(1_i32, BiasSegmentationType::Label);
        ge.accumulate(&gt, &gt_seg, &pred, &pred_seg);
        assert!((ge.ge_snapshot()).abs() < 1e-4);
    }

    // --- model_bias_analyzer ---

    #[test]
    fn model_bias_analyzer_returns_twelve_metrics() {
        use crate::data_handler::BiasDataPayload;
        let (feat, pred, gt) = symmetric_data();
        let report = model_bias_analyzer(
            BiasDataPayload::new_from_criteria(&feat, seg()),
            BiasDataPayload::new_from_criteria(&pred, seg()),
            BiasDataPayload::new_from_criteria(&gt, seg()),
        ).unwrap();
        assert_eq!(report.len(), 12);
    }

    #[test]
    fn model_bias_analyzer_mismatched_lengths_returns_error() {
        use crate::data_handler::BiasDataPayload;
        let feat = vec![1_i32, 0, 1];
        let pred = vec![1_i32, 0];
        let gt   = vec![1_i32, 0, 1];
        assert!(model_bias_analyzer(
            BiasDataPayload::new_from_criteria(&feat, seg()),
            BiasDataPayload::new_from_criteria(&pred, seg()),
            BiasDataPayload::new_from_criteria(&gt, seg()),
        ).is_err());
    }

    // --- model_bias_runtime_check / partial ---

    #[test]
    fn model_bias_runtime_check_identical_reports_passes() {
        let result = model_bias_runtime_check(
            full_model_bias_report(0.5),
            full_model_bias_report(0.5),
            0.1,
        ).unwrap();
        assert!(result.passed);
    }

    #[test]
    fn model_bias_runtime_check_missing_metric_returns_error() {
        let mut incomplete = full_model_bias_report(0.5);
        incomplete.remove(&M::GeneralizedEntropy);
        assert!(model_bias_runtime_check(full_model_bias_report(0.5), incomplete, 0.1).is_err());
    }

    #[test]
    fn model_bias_partial_runtime_check_filters_to_requested_subset() {
        // baseline ddpl=0.1, current ddpl=0.5 → exceeds threshold → flagged
        // baseline di=0.5, current di=0.5 → no change → not flagged
        let mut current = full_model_bias_report(0.1);
        current.insert(M::DifferenceInPositivePredictedLabels, 0.5);
        let result = model_bias_partial_runtime_check::<()>(
            full_model_bias_report(0.1),
            current,
            0.1,
            &[M::DifferenceInPositivePredictedLabels],
        ).unwrap();
        assert!(!result.passed);
        let failed = result.failed_report.unwrap();
        assert!(failed.iter().any(|f| f.metric == M::DifferenceInPositivePredictedLabels));
    }

    #[test]
    fn test_post_training_accum() {
        let mut container = PostTraining::default();

        container.accumulate_single(true, true, true);

        assert_eq!(
            container.dist_a,
            PostTrainingDistribution {
                len: 1,
                positive_gt: 1,
                positive_pred: 1
            }
        );
        assert_eq!(
            container.confusion_a,
            crate::data_handler::ConfusionMatrix {
                true_p: 1_f32,
                true_n: 0_f32,
                false_p: 0_f32,
                false_n: 0_f32
            }
        );
        assert_eq!(container.dist_d, PostTrainingDistribution::default());
        assert_eq!(
            container.confusion_d,
            crate::data_handler::ConfusionMatrix::default()
        );
        container.accumulate_single(false, true, false);

        assert_eq!(
            container.dist_d,
            PostTrainingDistribution {
                len: 1,
                positive_gt: 0,
                positive_pred: 1
            }
        );
        assert_eq!(
            container.confusion_d,
            crate::data_handler::ConfusionMatrix {
                true_p: 0_f32,
                true_n: 0_f32,
                false_p: 1_f32,
                false_n: 0_f32
            }
        );
        assert_eq!(
            container.dist_a,
            PostTrainingDistribution {
                len: 1,
                positive_gt: 1,
                positive_pred: 1
            }
        );
        assert_eq!(
            container.confusion_a,
            crate::data_handler::ConfusionMatrix {
                true_p: 1_f32,
                true_n: 0_f32,
                false_p: 0_f32,
                false_n: 0_f32
            }
        );

        container.accumulate_single(false, false, false);
        assert_eq!(
            container.confusion_d,
            crate::data_handler::ConfusionMatrix {
                true_p: 0_f32,
                true_n: 1_f32,
                false_p: 1_f32,
                false_n: 0_f32
            }
        );
        container.accumulate_single(false, false, true);
        assert_eq!(
            container.confusion_d,
            crate::data_handler::ConfusionMatrix {
                true_p: 0_f32,
                true_n: 1_f32,
                false_p: 1_f32,
                false_n: 1_f32
            }
        );
    }
}
