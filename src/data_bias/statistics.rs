use super::{PreTraining, PreTrainingDistribution};
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
pub(super) struct AdHocSegmentation {
    pub(super) facet_a: PreTrainingDistribution,
    pub(super) facet_d: PreTrainingDistribution,
}

impl AdHocSegmentation {
    fn new<F, G>(
        feature: &[F],
        feat_seg: BiasSegmentationCriteria<F>,
        gt: &[G],
        gt_seg: BiasSegmentationCriteria<G>,
    ) -> Result<AdHocSegmentation, BiasError>
    where
        G: PartialOrd,
        F: PartialOrd,
    {
        let mut facet_a = PreTrainingDistribution::default();
        let mut facet_d = PreTrainingDistribution::default();

        for (f, g) in zip_iters!(feature, gt) {
            let group = feat_seg.label(f);
            facet_a.len += group as u64;
            facet_a.positive += (group && gt_seg.label(g)) as u64;

            facet_d.len += !group as u64;
            facet_d.positive += (!group && gt_seg.label(g)) as u64;
        }

        if facet_a.len == 0 || facet_d.len == 0 {
            return Err(BiasError::NoFacetDeviation);
        }

        Ok(AdHocSegmentation { facet_a, facet_d })
    }
}

/// Method to compute the class imbalance. The class imbalance is the ratio between the difference
/// in class count and the total number of examples.
/// The result will be in the range [-1, 1]. Values further from 0 indicate higher imbalance.
pub fn class_imbalance<F, G>(
    feature: &[F],
    feat_seg: BiasSegmentationCriteria<F>,
    gt: &[G],
    gt_seg: BiasSegmentationCriteria<G>,
) -> Result<f32, BiasError>
where
    G: PartialOrd,
    F: PartialOrd,
{
    let seg = AdHocSegmentation::new(feature, feat_seg, gt, gt_seg)?;
    let pre_training: PreTraining = seg.into();
    Ok(inner::class_imbalance(&pre_training))
}

/// The difference in the acceptance rate between the two classes. This can also be thought of as
/// the label imbalance between the advantaged and disadvantaged classes. The acceptance is defined as
/// the ratio of positives outcomes to the total count of examples that belong to a particular
/// class. For example for feature examples belonging to the advantaged class the acceptance would
/// be:
/// <count of positive outcomes in the advantaged class> / <total number of examples in the positive class>.
/// The result will be in the range [-1, 1]. Values further from 0 indicate higher imbalance.
pub fn diff_in_proportion_of_labels<F, G>(
    feature: &[F],
    feat_seg: BiasSegmentationCriteria<F>,
    gt: &[G],
    gt_seg: BiasSegmentationCriteria<G>,
) -> Result<f32, BiasError>
where
    G: PartialOrd,
    F: PartialOrd,
{
    let seg = AdHocSegmentation::new(feature, feat_seg, gt, gt_seg)?;
    let pre_training: PreTraining = seg.into();
    Ok(inner::diff_in_proportion_of_labels(&pre_training)?)
}

/// Kullback-Leibler Divergence (KL). Computes the divergence between the label distribution
/// between the advantaged and disadvantaged class in the population set. Review the source code
/// for the formula if interested. This values grows toward infinity as the classes divergence.
/// This particularly is KL(P_a | P_d), where P_a is the probability of a positive outcome for
/// facet a and P_d is the probability of a positive outcome for facet d.
pub fn kl_divergence<F, G>(
    feature: &[F],
    feat_seg: BiasSegmentationCriteria<F>,
    gt: &[G],
    gt_seg: BiasSegmentationCriteria<G>,
) -> Result<f32, BiasError>
where
    G: PartialOrd,
    F: PartialOrd,
{
    // Sum of [P_a(Y) * log(P_a(Y) / P_d(Y))] across all distribution bins
    let seg = AdHocSegmentation::new(feature, feat_seg, gt, gt_seg)?;
    let pre_training: PreTraining = seg.into();
    // kl divergence across accept and not accept distributions of the 2 classes
    Ok(inner::kl_divergence(&pre_training)?)
}

/// Jensen Shannon Divergence. Measures divergance between the two classes entropically. Values will be in the range [0,
/// ln(2)), the result will grow toward infinity as behavior across the 2 classes diverges. This
/// can be described as the average of KL Divergence for each class distribution with respect to
/// the average acceptance rate across the entire population.
pub fn jensen_shannon<F, G>(
    feature: &[F],
    feat_seg: BiasSegmentationCriteria<F>,
    gt: &[G],
    gt_seg: BiasSegmentationCriteria<G>,
) -> Result<f32, BiasError>
where
    G: PartialOrd,
    F: PartialOrd,
{
    let seg = AdHocSegmentation::new(feature, feat_seg, gt, gt_seg)?;
    let pre_training: PreTraining = seg.into();
    Ok(inner::jensen_shannon(&pre_training)?)
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
    G: PartialOrd,
    F: PartialOrd,
{
    let seg = AdHocSegmentation::new(feature, feat_seg, gt, gt_seg)?;
    let pre_training: PreTraining = seg.into();
    Ok(inner::lp_norm(&pre_training)?)
}

/// Total Variation Distance, the l1 norm distance between the distribution across the classes.
pub fn total_variation_distance<F, G>(
    feature: &[F],
    feat_seg: BiasSegmentationCriteria<F>,
    gt: &[G],
    gt_seg: BiasSegmentationCriteria<G>,
) -> Result<f32, BiasError>
where
    G: PartialOrd,
    F: PartialOrd,
{
    let seg = AdHocSegmentation::new(feature, feat_seg, gt, gt_seg)?;
    let pre_training: PreTraining = seg.into();
    Ok(inner::total_variation_distance(&pre_training)?)
}

/// Measures the maximum divergence between the distributions across the two feature classes.
pub fn kolmogorov_smirnov<F, G>(
    feature: &[F],
    feat_seg: BiasSegmentationCriteria<F>,
    gt: &[G],
    gt_seg: BiasSegmentationCriteria<G>,
) -> Result<f32, BiasError>
where
    G: PartialOrd,
    F: PartialOrd,
{
    let seg = AdHocSegmentation::new(feature, feat_seg, gt, gt_seg)?;
    let pre_training: PreTraining = seg.into();
    Ok(inner::kolmogorov_smirnov(&pre_training)?)
}

pub(crate) mod inner {
    use super::super::PreTraining;
    use crate::errors::ModelPerfResult;
    pub fn class_imbalance(data: &PreTraining) -> f32 {
        (data.facet_a.len as i64 - data.facet_d.len as i64).abs() as f32
            / (data.facet_a.len + data.facet_d.len) as f32
    }

    pub fn diff_in_proportion_of_labels(data: &PreTraining) -> ModelPerfResult<f32> {
        Ok(data.facet_a.acceptance()? - data.facet_d.acceptance()?)
    }

    pub fn kl_divergence(data: &PreTraining) -> ModelPerfResult<f32> {
        Ok(ks_kl_div_bin(
            data.facet_a.acceptance()?,
            data.facet_d.acceptance()?,
        ))
    }

    // Utility to compute KL divergence for 2 bins, given there are 2 facet distributions in scope
    // within this module.
    pub(super) fn ks_kl_div_bin(p_facet: f32, p: f32) -> f32 {
        p_facet * (p_facet / p).ln() + (1_f32 - p_facet) * ((1_f32 - p_facet) / (1_f32 - p)).ln()
    }

    pub fn jensen_shannon(data: &PreTraining) -> ModelPerfResult<f32> {
        let a_acceptance = data.facet_a.acceptance()?;
        let d_acceptance = data.facet_d.acceptance()?;
        // p represents the average acceptance across the 2 classes
        let p = 0.5_f32 * (a_acceptance + d_acceptance);

        Ok(0.5 * (ks_kl_div_bin(a_acceptance, p) + ks_kl_div_bin(d_acceptance, p)))
    }

    pub fn lp_norm(data: &PreTraining) -> ModelPerfResult<f32> {
        let a_acceptance = data.facet_a.acceptance()?;
        let d_acceptance = data.facet_d.acceptance()?;
        Ok(((a_acceptance - d_acceptance).powi(2)
            + ((1_f32 - a_acceptance) - (1_f32 - d_acceptance)).powi(2))
        .sqrt())
    }

    pub fn total_variation_distance(data: &PreTraining) -> ModelPerfResult<f32> {
        let a_acc = data.facet_a.acceptance()?;
        let d_acc = data.facet_d.acceptance()?;

        Ok(0.5_f32 * ((a_acc - d_acc).abs() + ((1_f32 - a_acc) - (1_f32 - d_acc)).abs()))
    }

    pub fn kolmogorov_smirnov(data: &PreTraining) -> ModelPerfResult<f32> {
        let a_1_dist = data.facet_a.acceptance()?;
        let a_0_dist = 1_f32 - a_1_dist;
        let d_1_dist = data.facet_d.acceptance()?;
        let d_0_dist = 1_f32 - d_1_dist;

        let neg_outcome_diff = (a_0_dist - d_0_dist).abs();
        let pos_outcome_diff = (a_1_dist - d_1_dist).abs();

        Ok(f32::max(neg_outcome_diff, pos_outcome_diff))
    }
}

#[cfg(test)]
mod test_data_bias_statistics {
    use super::*;
    use crate::data_handler::BiasSegmentationCriteria;
    use crate::data_handler::BiasSegmentationType;

    #[test]
    fn test_ad_hoc_segmentation() {
        let feature_data: Vec<i32> =
            vec![1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1];
        let gt_data: Vec<i32> = vec![0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1];

        let ad_hoc_seg = AdHocSegmentation::new(
            &feature_data,
            BiasSegmentationCriteria::new(1_i32, BiasSegmentationType::Label),
            &gt_data,
            BiasSegmentationCriteria::new(1_i32, BiasSegmentationType::Label),
        )
        .unwrap();

        assert_eq!(ad_hoc_seg.facet_a.len, 10);
        assert_eq!(ad_hoc_seg.facet_d.len, 10);
        assert_eq!(
            ad_hoc_seg.facet_a.len as usize + ad_hoc_seg.facet_d.len as usize,
            feature_data.len()
        );
        assert_eq!(ad_hoc_seg.facet_a.positive, 7);
        assert_eq!(ad_hoc_seg.facet_d.positive, 2);
    }

    #[test]
    fn test_ad_hoc_class_imbalance() {
        let feature_data: Vec<i32> =
            vec![1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1];
        let gt_data: Vec<i32> = vec![0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1];
        let ci = class_imbalance(
            &feature_data,
            BiasSegmentationCriteria::new(1_i32, BiasSegmentationType::Label),
            &gt_data,
            BiasSegmentationCriteria::new(1_i32, BiasSegmentationType::Label),
        )
        .unwrap();

        assert_eq!(ci, 0_f32);
        let feature_data: Vec<i32> = vec![
            1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1,
        ];
        let gt_data: Vec<i32> = vec![
            0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1,
        ];
        let ci = class_imbalance(
            &feature_data,
            BiasSegmentationCriteria::new(1_i32, BiasSegmentationType::Label),
            &gt_data,
            BiasSegmentationCriteria::new(1_i32, BiasSegmentationType::Label),
        )
        .unwrap();

        assert_eq!(1_f32 / 21_f32, ci);
    }

    #[test]
    fn test_ad_hoc_diff_in_proportion_of_labels() {
        let feature_data: Vec<i32> =
            vec![1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1];
        let gt_data: Vec<i32> = vec![0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1];

        let dpl = diff_in_proportion_of_labels(
            &feature_data,
            BiasSegmentationCriteria::new(1_i32, BiasSegmentationType::Label),
            &gt_data,
            BiasSegmentationCriteria::new(1_i32, BiasSegmentationType::Label),
        )
        .unwrap();

        assert_eq!(0.7_f32 - 0.2_f32, dpl);
    }

    #[test]
    fn test_ad_hoc_kl_divergence() {
        let feature_data: Vec<i32> =
            vec![1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1];
        let gt_data: Vec<i32> = vec![0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1];

        let kl = kl_divergence(
            &feature_data,
            BiasSegmentationCriteria::new(1_i32, BiasSegmentationType::Label),
            &gt_data,
            BiasSegmentationCriteria::new(1_i32, BiasSegmentationType::Label),
        )
        .unwrap();
        let base = (0.7_f32 * (0.7_f32 / 0.2_f32).ln()) + (0.3_f32 * (0.3_f32 / 0.8_f32).ln());
        assert_eq!(base, kl)
    }

    #[test]
    fn ad_hoc_jensen_shannon() {
        let feature_data: Vec<i32> =
            vec![1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1];
        let gt_data: Vec<i32> = vec![0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1];
        let js = jensen_shannon(
            &feature_data,
            BiasSegmentationCriteria::new(1_i32, BiasSegmentationType::Label),
            &gt_data,
            BiasSegmentationCriteria::new(1_i32, BiasSegmentationType::Label),
        )
        .unwrap();
        let p = 0.5_f32 * (0.7_f32 + 0.2_f32);
        let f =
            |a: f32, d: f32| (a * (a / d).ln()) + ((1_f32 - a) * ((1_f32 - a) / (1_f32 - d)).ln());

        let base = 0.5 * (f(0.7_f32, p) + f(0.2_f32, p));
        assert_eq!(base, js)
    }

    #[test]
    fn ad_hoc_lp_norm() {
        let feature_data: Vec<i32> =
            vec![1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1];
        let gt_data: Vec<i32> = vec![0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1];
        let lp_norm = lp_norm(
            &feature_data,
            BiasSegmentationCriteria::new(1_i32, BiasSegmentationType::Label),
            &gt_data,
            BiasSegmentationCriteria::new(1_i32, BiasSegmentationType::Label),
        )
        .unwrap();

        let base = ((0.7_f32 - 0.2_f32).powi(2) + (0.3_f32 - 0.8_f32).powi(2)).sqrt();
        assert_eq!(base, lp_norm)
    }

    #[test]
    fn ad_hoc_total_variation_distance() {
        let feature_data: Vec<i32> =
            vec![1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1];
        let gt_data: Vec<i32> = vec![0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1];
        let tvd = total_variation_distance(
            &feature_data,
            BiasSegmentationCriteria::new(1_i32, BiasSegmentationType::Label),
            &gt_data,
            BiasSegmentationCriteria::new(1_i32, BiasSegmentationType::Label),
        )
        .unwrap();
        let base = 0.5_f32 * ((0.7_f32 - 0.2_f32).abs() + (0.3_f32 - 0.8_f32).abs());
        dbg!(base);
        assert_eq!(tvd, base);
        assert!(tvd < 1_f32 && tvd > 0_f32)
    }

    #[test]
    fn ad_hoc_kolmorgov_smirnov() {
        let feature_data: Vec<i32> =
            vec![1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1];
        let gt_data: Vec<i32> = vec![0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1];
        let ks = kolmogorov_smirnov(
            &feature_data,
            BiasSegmentationCriteria::new(1_i32, BiasSegmentationType::Label),
            &gt_data,
            BiasSegmentationCriteria::new(1_i32, BiasSegmentationType::Label),
        )
        .unwrap();
        assert_eq!(ks, 0.5_f32);
    }
}
