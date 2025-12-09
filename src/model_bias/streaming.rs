use super::{BucketGeneralizedEntropy, PostTraining};
use crate::data_handler::BiasSegmentationCriteria;

pub struct StreamingModelBias<F, P, G>
where
    F: PartialOrd,
    G: PartialOrd,
    P: PartialOrd,
{
    feat_seg: BiasSegmentationCriteria<F>,
    pred_seg: BiasSegmentationCriteria<P>,
    gt_seg: BiasSegmentationCriteria<G>,
    facet_a: PostTraining,
    facet_d: PostTraining,
    ge: BucketGeneralizedEntropy,
}
