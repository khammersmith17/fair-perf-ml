use super::{BucketGeneralizedEntropy, PostTraining};
use crate::data_handler::BiasSegmentationCriteria;
use crate::metrics::ModelBiasMetric;
use crate::reporting::DriftReport;
use crate::runtime::ModelBiasRuntime;

/// This is a type geared toward a long running ML Observability service. Internally it leverages a
/// bucketing alogrithm to store all information necessary to compute model bias metrics in compact
/// space. This type takes in class segmentation logic for the feature, predictions, and ground
/// truth. A single instance can store a single feature segmentation for a single feature, thus
/// many instances of this type can be composed to provide full insight into a model if you wish to
/// monitor many features or many models in the same service.
pub struct StreamingModelBias<F, P, G>
where
    F: PartialOrd,
    G: PartialOrd,
    P: PartialOrd,
{
    feat_seg: BiasSegmentationCriteria<F>,
    pred_seg: BiasSegmentationCriteria<P>,
    gt_seg: BiasSegmentationCriteria<G>,
    rt: PostTraining,
    ge: BucketGeneralizedEntropy,
    bl: ModelBiasRuntime,
}

impl<F, P, G> StreamingModelBias<F, P, G>
where
    F: PartialOrd,
    P: PartialOrd,
    G: PartialOrd,
{
    /// Constructor requires segmentation criteria for the feature of interest, predictions, and
    /// ground truth, and the baseline dataset for each respective data dimension. On construction,
    /// baseline are computed on the baseline dataset provided. This type then takes ownership of
    /// the segmentation criteria to be used later for runtime segmentation and bucketing.
    pub fn new(
        features: &[F],
        feat_seg: BiasSegmentationCriteria<F>,
        preds: &[P],
        pred_seg: BiasSegmentationCriteria<P>,
        gt: &[G],
        gt_seg: BiasSegmentationCriteria<G>,
    ) -> StreamingModelBias<F, P, G> {
        let bl_pt = PostTraining::new_from_segmentation_criteria(
            features, &feat_seg, preds, &pred_seg, gt, &gt_seg,
        );

        let mut bl_ge_bucket = BucketGeneralizedEntropy::default();
        bl_ge_bucket.accumulate(gt, &gt_seg, preds, &pred_seg);
        let bl_ge = bl_ge_bucket.ge_snapshot();
        let bl = ModelBiasRuntime::new_from_post_training(&bl_pt, bl_ge);

        StreamingModelBias {
            feat_seg,
            pred_seg,
            gt_seg,
            bl,
            rt: PostTraining::default(),
            ge: BucketGeneralizedEntropy::default(),
        }
    }

    /// Push runtime feature data, prediction data, and ground truth data into the stream. This
    /// method will use the segmentation logic passed at type construction. This method returns
    /// nothing, use the drift_snapshot method to generate the report.
    pub fn push(&mut self, features: &[F], preds: &[P], gt: &[G]) {
        self.rt.accumulate(
            features,
            &self.feat_seg,
            preds,
            &self.pred_seg,
            gt,
            &self.gt_seg,
        );
        self.ge.accumulate(gt, &self.gt_seg, preds, &self.pred_seg);
    }

    /// Flush the data accumulated in the streaming container. This will periodically be helpful to
    /// reset state over time in a long running service.
    pub fn flush(&mut self) {
        self.rt.clear();
        self.ge.clear();
    }

    /// Reset the baseline data used to compute drift. This is useful for a model retraining, or to
    /// update the baseline on more recent data.
    pub fn reset_baseline(&mut self, feature: &[F], prediction: &[P], ground_truth: &[G]) {
        let bl_pt = PostTraining::new_from_segmentation_criteria(
            feature,
            &self.feat_seg,
            prediction,
            &self.pred_seg,
            ground_truth,
            &self.gt_seg,
        );

        let mut bl_ge_bucket = BucketGeneralizedEntropy::default();
        bl_ge_bucket.accumulate(ground_truth, &self.gt_seg, prediction, &self.pred_seg);
        let bl_ge = bl_ge_bucket.ge_snapshot();
        self.bl = ModelBiasRuntime::new_from_post_training(&bl_pt, bl_ge);
    }

    /// Generateas a point in time drift report, this will consider the baseline set and all the
    /// data that has been accumulated since the last flush.
    pub fn drift_snapshot(&self) -> DriftReport<ModelBiasMetric> {
        let rt_ge = self.ge.ge_snapshot();
        let rt_snapshot = ModelBiasRuntime::new_from_post_training(&self.rt, rt_ge);
        let report = rt_snapshot.runtime_drift_report(&self.bl);
        DriftReport::from_runtime(report)
    }
}
