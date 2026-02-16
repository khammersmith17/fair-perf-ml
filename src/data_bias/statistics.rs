use super::PreTrainingDistribution;
use crate::data_handler::BiasSegmentationCriteria;
use crate::errors::BiasError;
use crate::zip_iters;

/// All methods in this mod take in feature and ground truth data, in addition to an assoicated
/// 'BiasSegmentationCriteria' for each, in order to segment data into positive and negative
/// classes based on the features, this effectively divides the data into two distinct demographic
/// groups. The ground truth is segmented into positive and negative outcomes. Continuoys model
/// values, thus, need to be coerce to class labels. It is up to the user to determine the
/// threshold of a "positive" outcome in this case. These methods are used in the discrete
/// monitoring approach taken in this crate, and are exposed as discrete methods

struct AdHocSegmentation {
    facet_a: PreTrainingDistribution,
    facet_d: PreTrainingDistribution,
}

impl AdHocSegmentation {
    fn new<F, G>(
        feature: &[F],
        feat_seg: BiasSegmentationCriteria<F>,
        gt: &[G],
        gt_seg: BiasSegmentationCriteria<G>,
    ) -> Result<AdHocSegmentation, BiasError>
    where
        G: PartialEq + PartialOrd,
        F: PartialEq + PartialOrd,
    {
        let mut facet_a = PreTrainingDistribution::default();
        let mut facet_d = PreTrainingDistribution::default();

        for (f, g) in zip_iters!(feature, gt) {
            let group = feat_seg.label(f);
            facet_a.len += group as u64;
            facet_a.positive += group as u64 * gt_seg.label(g) as u64;

            facet_d.len += !group as u64;
            facet_d.positive += !group as u64 * gt_seg.label(g) as u64;
        }

        if facet_a.len == 0 || facet_d.len == 0 {
            return Err(BiasError::NoFacetDeviation);
        }

        Ok(AdHocSegmentation { facet_a, facet_d })
    }
}

/// Method to compute the class imbalance. The class imbalance is the ratio between the difference
/// in class count and the total number of examples.
///
/// The result will be in the range [-1, 1]. Values further from 0 indicate higher imbalance.
pub fn class_imbalance<F, G>(
    feature: &[F],
    feat_seg: BiasSegmentationCriteria<F>,
    gt: &[G],
    gt_seg: BiasSegmentationCriteria<G>,
) -> Result<f32, BiasError>
where
    G: PartialEq + PartialOrd,
    F: PartialEq + PartialOrd,
{
    let seg = AdHocSegmentation::new(feature, feat_seg, gt, gt_seg)?;
    Ok(
        (seg.facet_a.len as i64 - seg.facet_d.len as i64).abs() as f32
            / (seg.facet_a.len as i64 + seg.facet_d.len as i64) as f32,
    )
}

/// The difference in the acceptance rate between the two classes. This can also be thought of as
/// the label imbalance between the advantaged and disadvantaged classes. The acceptance is defined as
/// the ratio of positives outcomes to the total count of examples that belong to a particular
/// class. For example for feature examples belonging to the advantaged class the acceptance would
/// be:
/// <count of positive outcomes in the advantaged class> / <total number of examples in the positive class>.
///
/// The result will be in the range [-1, 1]. Values further from 0 indicate higher imbalance.
pub fn diff_in_proportion_of_labels<F, G>(
    feature: &[F],
    feat_seg: BiasSegmentationCriteria<F>,
    gt: &[G],
    gt_seg: BiasSegmentationCriteria<G>,
) -> Result<f32, BiasError>
where
    G: PartialEq + PartialOrd,
    F: PartialEq + PartialOrd,
{
    let seg = AdHocSegmentation::new(feature, feat_seg, gt, gt_seg)?;

    Ok(seg.facet_a.acceptance()? - seg.facet_d.acceptance()?)
}

/// Kullback-Leibler Divergence (KL). Computes the divergence between the label distribution
/// between the advantaged and disadvantaged class in the population set. Review the source code
/// for the formula if interested. This values grows toward infinity as the classes divergence.
pub fn kl_divergence<F, G>(
    feature: &[F],
    feat_seg: BiasSegmentationCriteria<F>,
    gt: &[G],
    gt_seg: BiasSegmentationCriteria<G>,
) -> Result<f32, BiasError>
where
    G: PartialEq + PartialOrd,
    F: PartialEq + PartialOrd,
{
    // Sum of [P_a(Y) * log(P_a(Y) / P_d(Y))] across all distribution bins
    let seg = AdHocSegmentation::new(feature, feat_seg, gt, gt_seg)?;
    let a_acceptance = seg.facet_a.acceptance()?;
    let d_acceptance = seg.facet_d.acceptance()?;

    // kl divergence across accept and not accept distributions of the 2 classes
    Ok(a_acceptance * (a_acceptance / d_acceptance).ln()
        + (1_f32 - a_acceptance) * ((1_f32 - a_acceptance) / (1_f32 - d_acceptance)).ln())
}

fn ks_kl_div(p_facet: f32, p: f32) -> f32 {
    return p_facet * (p_facet / p).ln()
        + (1_f32 - p_facet) * ((1_f32 - p_facet) / (1_f32 - p)).ln();
}

/// Jensen Shannon Divergence. Measures divergance between the two classes entropically. Values will be in the range [0,
/// infinity), the result will grow toward infinity as behavior across the 2 classes diverges.
pub fn jensen_shannon<F, G>(
    feature: &[F],
    feat_seg: BiasSegmentationCriteria<F>,
    gt: &[G],
    gt_seg: BiasSegmentationCriteria<G>,
) -> Result<f32, BiasError>
where
    G: PartialEq + PartialOrd,
    F: PartialEq + PartialOrd,
{
    let seg = AdHocSegmentation::new(feature, feat_seg, gt, gt_seg)?;
    let a_acceptance = seg.facet_a.acceptance()?;
    let d_acceptance = seg.facet_d.acceptance()?;
    let p = 0.5_f32
        * (seg.facet_a.positive as f32 / seg.facet_d.len as f32
            + seg.facet_d.positive as f32 / seg.facet_a.len as f32);

    Ok(0.5 * (ks_kl_div(a_acceptance, p) + ks_kl_div(d_acceptance, p)))
}

/// Lp Norm. Measures the p norm distance between the distribution of labels across the 2 classes.
/// Values will be in the range [0, infinity), the result will grow toward infinity as behavior across the 2 classes diverges.
pub fn lp_norm<F, G>(
    feature: &[F],
    feat_seg: BiasSegmentationCriteria<F>,
    gt: &[G],
    gt_seg: BiasSegmentationCriteria<G>,
) -> Result<f32, BiasError>
where
    G: PartialEq + PartialOrd,
    F: PartialEq + PartialOrd,
{
    let seg = AdHocSegmentation::new(feature, feat_seg, gt, gt_seg)?;
    let a_acceptance = seg.facet_a.acceptance()?;
    let d_acceptance = seg.facet_d.acceptance()?;
    Ok(((a_acceptance - d_acceptance).powf(2.0)
        + (1_f32 - a_acceptance - 1_f32 - d_acceptance).powf(2.0))
    .sqrt())
}

/// Total Variation Distance, the l1 norm distance between the distribution across the classes.
pub fn total_variation_distance<F, G>(
    feature: &[F],
    feat_seg: BiasSegmentationCriteria<F>,
    gt: &[G],
    gt_seg: BiasSegmentationCriteria<G>,
) -> Result<f32, BiasError>
where
    G: PartialEq + PartialOrd,
    F: PartialEq + PartialOrd,
{
    let seg = AdHocSegmentation::new(feature, feat_seg, gt, gt_seg)?;
    let a_acceptance = seg.facet_a.acceptance()?;
    let d_acceptance = seg.facet_d.acceptance()?;
    Ok((a_acceptance - d_acceptance).abs()
        + ((1_f32 - a_acceptance) - (1_f32 - a_acceptance)).abs())
}

/// Measures the maximum divergence between the distributions across the two feature classes.
pub fn kolmorogv_smirnov<F, G>(
    feature: &[F],
    feat_seg: BiasSegmentationCriteria<F>,
    gt: &[G],
    gt_seg: BiasSegmentationCriteria<G>,
) -> Result<f32, BiasError>
where
    G: PartialEq + PartialOrd,
    F: PartialEq + PartialOrd,
{
    let seg = AdHocSegmentation::new(feature, feat_seg, gt, gt_seg)?;
    let a_1_dist = seg.facet_a.acceptance()?;
    let a_0_dist = 1_f32 - a_1_dist;
    let d_1_dist = seg.facet_d.acceptance()?;
    let d_0_dist = 1_f32 - d_1_dist;

    let neg_outcome_diff = (a_0_dist - d_0_dist).abs();
    let pos_outcome_diff = (a_1_dist - d_1_dist).abs();

    if neg_outcome_diff > pos_outcome_diff {
        Ok(pos_outcome_diff)
    } else {
        Ok(neg_outcome_diff)
    }
}

pub(crate) mod inner {
    use super::super::PreTraining;
    use crate::errors::ModelPerfResult;
    pub fn class_imbalance(data: &PreTraining) -> f32 {
        return (data.facet_a.len as i64 - data.facet_d.len as i64).abs() as f32
            / (data.facet_a.len + data.facet_d.len) as f32;
    }

    pub fn diff_in_proportion_of_labels(data: &PreTraining) -> ModelPerfResult<f32> {
        Ok(data.facet_a.acceptance()? - data.facet_d.acceptance()?)
    }

    pub fn kl_divergence(data: &PreTraining) -> ModelPerfResult<f32> {
        let a_acceptance = data.facet_a.acceptance()?;
        let d_acceptance = data.facet_d.acceptance()?;
        Ok(a_acceptance * (a_acceptance / d_acceptance).ln()
            + (1_f32 - a_acceptance) * ((1_f32 - a_acceptance) / (1_f32 - d_acceptance)).ln())
    }

    fn ks_kl_div(p_facet: f32, p: f32) -> f32 {
        return p_facet * (p_facet / p).ln()
            + (1_f32 - p_facet) * ((1_f32 - p_facet) / (1_f32 - p)).ln();
    }

    pub fn jensen_shannon(data: &PreTraining) -> ModelPerfResult<f32> {
        let a_acceptance = data.facet_a.acceptance()?;
        let d_acceptance = data.facet_d.acceptance()?;
        let p = 0.5_f32
            * (data.facet_a.positive as f32 / data.facet_d.len as f32
                + data.facet_d.positive as f32 / data.facet_a.len as f32);

        Ok(0.5 * (ks_kl_div(a_acceptance, p) + ks_kl_div(d_acceptance, p)))
    }

    pub fn lp_norm(data: &PreTraining) -> ModelPerfResult<f32> {
        let a_acceptance = data.facet_a.acceptance()?;
        let d_acceptance = data.facet_d.acceptance()?;
        Ok(((a_acceptance - d_acceptance).powf(2.0)
            + (1_f32 - a_acceptance - 1_f32 - d_acceptance).powf(2.0))
        .sqrt())
    }

    pub fn total_variation_distance(data: &PreTraining) -> ModelPerfResult<f32> {
        let a_acceptance = data.facet_a.acceptance()?;
        let d_acceptance = data.facet_d.acceptance()?;
        Ok((a_acceptance - d_acceptance).abs()
            + ((1_f32 - a_acceptance) - (1_f32 - a_acceptance)).abs())
    }

    pub fn kolmorogv_smirnov(data: &PreTraining) -> ModelPerfResult<f32> {
        let a_1_dist = data.facet_a.acceptance()?;
        let a_0_dist = 1_f32 - a_1_dist;
        let d_1_dist = data.facet_d.acceptance()?;
        let d_0_dist = 1_f32 - d_1_dist;

        let neg_outcome_diff = (a_0_dist - d_0_dist).abs();
        let pos_outcome_diff = (a_1_dist - d_1_dist).abs();

        if neg_outcome_diff > pos_outcome_diff {
            Ok(pos_outcome_diff)
        } else {
            Ok(neg_outcome_diff)
        }
    }
}
