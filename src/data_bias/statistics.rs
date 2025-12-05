use super::PreTrainingDistribution;
use crate::data_handler::BiasSegmentationCriteria;
use crate::errors::BiasError;
use crate::zip_iters;

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

    Ok(seg.facet_a.acceptance() - seg.facet_d.acceptance())
}

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
    let seg = AdHocSegmentation::new(feature, feat_seg, gt, gt_seg)?;
    let a_acceptance = seg.facet_a.acceptance();
    let d_acceptance = seg.facet_d.acceptance();
    Ok(a_acceptance * (a_acceptance / d_acceptance).ln()
        + (1_f32 - a_acceptance) * ((1_f32 - a_acceptance) / (1_f32 - d_acceptance)).ln())
}

fn ks_kl_div(p_facet: f32, p: f32) -> f32 {
    return p_facet * (p_facet / p).ln()
        + (1_f32 - p_facet) * ((1_f32 - p_facet) / (1_f32 - p)).ln();
}

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
    let a_acceptance = seg.facet_a.acceptance();
    let d_acceptance = seg.facet_d.acceptance();
    let p = 0.5_f32
        * (seg.facet_a.positive as f32 / seg.facet_d.len as f32
            + seg.facet_d.positive as f32 / seg.facet_a.len as f32);

    Ok(0.5 * (ks_kl_div(a_acceptance, p) + ks_kl_div(d_acceptance, p)))
}

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
    let a_acceptance = seg.facet_a.acceptance();
    let d_acceptance = seg.facet_d.acceptance();
    Ok(((a_acceptance - d_acceptance).powf(2.0)
        + (1_f32 - a_acceptance - 1_f32 - d_acceptance).powf(2.0))
    .sqrt())
}

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
    let a_acceptance = seg.facet_a.acceptance();
    let d_acceptance = seg.facet_d.acceptance();
    Ok((a_acceptance - d_acceptance).abs()
        + ((1_f32 - a_acceptance) - (1_f32 - a_acceptance)).abs())
}

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
    let a_1_dist = seg.facet_a.acceptance();
    let a_0_dist = 1_f32 - a_1_dist;
    let d_1_dist = seg.facet_d.acceptance();
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
    pub fn class_imbalance(data: &PreTraining) -> f32 {
        return (data.facet_a.len as i64 - data.facet_d.len as i64).abs() as f32
            / (data.facet_a.len + data.facet_d.len) as f32;
    }

    pub fn diff_in_proportion_of_labels(data: &PreTraining) -> f32 {
        return data.facet_a.acceptance() - data.facet_d.acceptance();
    }

    pub fn kl_divergence(data: &PreTraining) -> f32 {
        let a_acceptance = data.facet_a.acceptance();
        let d_acceptance = data.facet_d.acceptance();
        return a_acceptance * (a_acceptance / d_acceptance).ln()
            + (1_f32 - a_acceptance) * ((1_f32 - a_acceptance) / (1_f32 - d_acceptance)).ln();
    }

    fn ks_kl_div(p_facet: f32, p: f32) -> f32 {
        return p_facet * (p_facet / p).ln()
            + (1_f32 - p_facet) * ((1_f32 - p_facet) / (1_f32 - p)).ln();
    }

    pub fn jensen_shannon(data: &PreTraining) -> f32 {
        let a_acceptance = data.facet_a.acceptance();
        let d_acceptance = data.facet_d.acceptance();
        let p = 0.5_f32
            * (data.facet_a.positive as f32 / data.facet_d.len as f32
                + data.facet_d.positive as f32 / data.facet_a.len as f32);

        return 0.5 * (ks_kl_div(a_acceptance, p) + ks_kl_div(d_acceptance, p));
    }

    pub fn lp_norm(data: &PreTraining) -> f32 {
        let a_acceptance = data.facet_a.acceptance();
        let d_acceptance = data.facet_d.acceptance();
        return ((a_acceptance - d_acceptance).powf(2.0)
            + (1_f32 - a_acceptance - 1_f32 - d_acceptance).powf(2.0))
        .sqrt();
    }

    pub fn total_variation_distance(data: &PreTraining) -> f32 {
        let a_acceptance = data.facet_a.acceptance();
        let d_acceptance = data.facet_d.acceptance();
        return (a_acceptance - d_acceptance).abs()
            + ((1_f32 - a_acceptance) - (1_f32 - a_acceptance)).abs();
    }

    pub fn kolmorogv_smirnov(data: &PreTraining) -> f32 {
        let a_1_dist = data.facet_a.acceptance();
        let a_0_dist = 1_f32 - a_1_dist;
        let d_1_dist = data.facet_d.acceptance();
        let d_0_dist = 1_f32 - d_1_dist;

        let neg_outcome_diff = (a_0_dist - d_0_dist).abs();
        let pos_outcome_diff = (a_1_dist - d_1_dist).abs();

        if neg_outcome_diff > pos_outcome_diff {
            pos_outcome_diff
        } else {
            neg_outcome_diff
        }
    }
}

/*
* TESTS TO VERIFY NEW LOGIC IS STILL VALID
mod data_bias_test_new_logic {
    use super::*;

    fn construct_args() -> (
        super::super::PreTraining,
        PreTraining,
        PreTrainingComputations,
    ) {
        let mut feature_data = Vec::with_capacity(10_000);
        let mut gt_data = Vec::with_capacity(10_000);

        for _ in 0..10_000 {
            feature_data.push(rand::random_range(0_i16..=1_i16));
            gt_data.push(rand::random_range(0_i16..=1_i16));
        }

        let pre_training_v2 = super::super::PreTraining::new(&feature_data, &gt_data);
        let pre_training =
            crate::data_bias::core::perform_segmentation_data_bias(feature_data, gt_data).unwrap();
        let computations = pre_training.generate();

        (pre_training_v2, pre_training, computations)
    }

    #[test]
    fn class_imbalance() {
        let (v2, v1, _) = construct_args();
        let ci1 = super::class_imbalance(&v1);
        let ci2 = super::v2::class_imbalance(&v2);

        assert!((ci1 as f32 - ci2).abs() < 1e-5_f32)
    }

    #[test]
    fn diff_in_proportion_of_labels() {
        let (v2, v1, _) = construct_args();
        let res_v1 = super::diff_in_proportion_of_labels(&v1);
        let res_v2 = super::v2::diff_in_proportion_of_labels(&v2);

        assert!((res_v1 as f32 - res_v2).abs() < 1e-5_f32)
    }

    #[test]
    fn kl_divergence() {
        let (v2, _, comp) = construct_args();
        let res_v1 = super::kl_divergence(&comp);
        let res_v2 = super::v2::kl_divergence(&v2);

        assert!((res_v1 as f32 - res_v2).abs() < 1e-5_f32)
    }

    #[test]
    fn jensen_shannon() {
        let (v2, v1, comp) = construct_args();
        let res_v1 = super::jensen_shannon(&v1, &comp);
        let res_v2 = super::v2::jensen_shannon(&v2);

        assert!((res_v1 as f32 - res_v2).abs() < 1e-5_f32)
    }

    #[test]
    fn lp_norm() {
        let (v2, _, comp) = construct_args();
        let res_v1 = super::lp_norm(&comp);
        let res_v2 = super::v2::lp_norm(&v2);

        assert!((res_v1 as f32 - res_v2).abs() < 1e-5_f32)
    }
    #[test]
    fn total_variation_distance() {
        let (v2, _, comp) = construct_args();
        let res_v1 = super::total_variation_distance(&comp);
        let res_v2 = super::v2::total_variation_distance(&v2);

        assert!((res_v1 as f32 - res_v2).abs() < 1e-5_f32)
    }

    #[test]
    fn kolmorogv_smirnov() {
        let (v2, v1, _) = construct_args();
        let res_v1 = super::kolmorogv_smirnov(&v1);
        let res_v2 = super::v2::kolmorogv_smirnov(&v2);

        assert!((res_v1 as f32 - res_v2).abs() < 1e-5_f32)
    }
}
*/
