use super::{ConfusionMatrix, PostTraining};
use crate::data_handler::BiasSegmentationCriteria;
use crate::errors::BiasError;
use crate::zip_iters;

struct AdHocSegmentation {
    facet_a_pred: Vec<i16>,
    facet_a_true: Vec<i16>,
    facet_d_pred: Vec<i16>,
    facet_d_true: Vec<i16>,
}

impl AdHocSegmentation {
    fn new<F, P, G>(
        feature: &[F],
        feature_seg: BiasSegmentationCriteria<F>,
        pred: &[P],
        pred_seg: BiasSegmentationCriteria<P>,
        gt: &[G],
        gt_seg: BiasSegmentationCriteria<G>,
    ) -> Result<AdHocSegmentation, BiasError>
    where
        F: PartialOrd + PartialEq,
        P: PartialOrd + PartialEq,
        G: PartialOrd + PartialEq,
    {
        let n_feature = feature.len();
        let n_pred = pred.len();
        let n_gt = gt.len();

        if !(n_feature == n_pred && n_pred == n_gt) {
            return Err(BiasError::DataLengthError);
        }

        let mut facet_a_pred: Vec<i16> = Vec::with_capacity(n_feature / 2 as usize);
        let mut facet_a_true: Vec<i16> = Vec::with_capacity(n_feature / 2 as usize);
        let mut facet_d_pred: Vec<i16> = Vec::with_capacity(n_feature / 2 as usize);
        let mut facet_d_true: Vec<i16> = Vec::with_capacity(n_feature / 2 as usize);
        for (f, (p, g)) in zip_iters!(feature, pred, gt) {
            let pred_label = pred_seg.label(p);
            let gt_label = gt_seg.label(g);
            let group = feature_seg.label(f);

            if group {
                facet_a_pred.push(pred_label as i16);
                facet_a_true.push(gt_label as i16);
            } else {
                facet_d_pred.push(pred_label as i16);
                facet_d_true.push(gt_label as i16);
            }
        }

        if facet_a_pred.is_empty() || facet_d_pred.is_empty() {
            return Err(BiasError::NoFacetDeviation);
        }

        Ok(AdHocSegmentation {
            facet_a_pred,
            facet_a_true,
            facet_d_pred,
            facet_d_true,
        })
    }

    fn get_confusion(&self) -> (ConfusionMatrix, ConfusionMatrix) {
        let mut confusion_a = ConfusionMatrix::default();
        let mut confusion_d = ConfusionMatrix::default();

        for (at, (ap, (dt, dp))) in zip_iters!(
            self.facet_a_true,
            self.facet_a_pred,
            self.facet_d_true,
            self.facet_d_pred
        ) {
            confusion_a.true_p += (*at == 1 && *ap == 1) as usize as f32;
            confusion_a.true_n += (*at == 0 && *ap == 0) as usize as f32;
            confusion_a.false_p += (*at == 0 && *ap == 1) as usize as f32;
            confusion_a.false_n += (*at == 1 && *ap == 0) as usize as f32;

            confusion_d.true_p += (*dt == 1 && *dp == 1) as usize as f32;
            confusion_d.true_n += (*dt == 0 && *dp == 0) as usize as f32;
            confusion_d.false_p += (*dt == 0 && *dp == 1) as usize as f32;
            confusion_d.false_n += (*dt == 1 && *dp == 0) as usize as f32;
        }

        (confusion_a, confusion_d)
    }
}

// Publicly exposed methods for ad hoc use
pub fn diff_in_pos_proportion_in_pred_labels<F, P, G>(
    feature: &[F],
    feature_seg: BiasSegmentationCriteria<F>,
    pred: &[P],
    pred_seg: BiasSegmentationCriteria<P>,
    gt: &[G],
    gt_seg: BiasSegmentationCriteria<G>,
) -> Result<f32, BiasError>
where
    F: PartialOrd + PartialEq,
    P: PartialOrd + PartialEq,
    G: PartialOrd + PartialEq,
{
    let seg = AdHocSegmentation::new(feature, feature_seg, pred, pred_seg, gt, gt_seg)?;
    let q_prime_a: f32 =
        seg.facet_a_pred.iter().sum::<i16>() as f32 / seg.facet_a_true.iter().sum::<i16>() as f32;
    let q_prime_d: f32 =
        seg.facet_d_pred.iter().sum::<i16>() as f32 / seg.facet_d_true.iter().sum::<i16>() as f32;
    Ok(q_prime_a - q_prime_d)
}

pub fn disparate_impact<F, P, G>(
    feature: &[F],
    feature_seg: BiasSegmentationCriteria<F>,
    pred: &[P],
    pred_seg: BiasSegmentationCriteria<P>,
    gt: &[G],
    gt_seg: BiasSegmentationCriteria<G>,
) -> Result<f32, BiasError>
where
    F: PartialOrd + PartialEq,
    P: PartialOrd + PartialEq,
    G: PartialOrd + PartialEq,
{
    let seg = AdHocSegmentation::new(feature, feature_seg, pred, pred_seg, gt, gt_seg)?;
    let q_prime_a: f32 =
        seg.facet_a_pred.iter().sum::<i16>() as f32 / seg.facet_d_true.iter().sum::<i16>() as f32;
    let q_prime_d: f32 =
        seg.facet_d_pred.iter().sum::<i16>() as f32 / seg.facet_d_true.iter().sum::<i16>() as f32;

    if q_prime_d == 0.0 {
        return Ok(0.0);
    }
    Ok(q_prime_a / q_prime_d)
}

pub fn diff_in_cond_acceptance<F, P, G>(
    feature: &[F],
    feature_seg: BiasSegmentationCriteria<F>,
    pred: &[P],
    pred_seg: BiasSegmentationCriteria<P>,
    gt: &[G],
    gt_seg: BiasSegmentationCriteria<G>,
) -> Result<f32, BiasError>
where
    F: PartialOrd + PartialEq,
    P: PartialOrd + PartialEq,
    G: PartialOrd + PartialEq,
{
    let seg = AdHocSegmentation::new(feature, feature_seg, pred, pred_seg, gt, gt_seg)?;
    let sum_true_facet_a: f32 = seg.facet_a_true.iter().sum::<i16>().into();
    let sum_scores_facet_a: f32 = seg.facet_a_pred.iter().sum::<i16>().into();
    let c_facet_a: f32 = sum_true_facet_a / sum_scores_facet_a;

    let sum_true_facet_d: f32 = seg.facet_d_true.iter().sum::<i16>().into();
    let sum_scores_facet_d: f32 = seg.facet_d_pred.iter().sum::<i16>().into();
    let c_facet_d: f32 = sum_true_facet_d / sum_scores_facet_d;

    Ok(c_facet_a - c_facet_d)
}

pub fn accuracy_difference<F, P, G>(
    feature: &[F],
    feature_seg: BiasSegmentationCriteria<F>,
    pred: &[P],
    pred_seg: BiasSegmentationCriteria<P>,
    gt: &[G],
    gt_seg: BiasSegmentationCriteria<G>,
) -> Result<f32, BiasError>
where
    F: PartialOrd + PartialEq,
    P: PartialOrd + PartialEq,
    G: PartialOrd + PartialEq,
{
    let seg = AdHocSegmentation::new(feature, feature_seg, pred, pred_seg, gt, gt_seg)?;
    let (confusion_a, confusion_d) = seg.get_confusion();
    let acc_a: f32 = (confusion_a.true_p + confusion_a.true_n) / seg.facet_a_pred.len() as f32;

    let acc_d: f32 = (confusion_d.true_p + confusion_d.true_n) / seg.facet_d_pred.len() as f32;

    Ok(acc_a - acc_d)
}

pub fn diff_in_acceptance_rate<F, P, G>(
    feature: &[F],
    feature_seg: BiasSegmentationCriteria<F>,
    pred: &[P],
    pred_seg: BiasSegmentationCriteria<P>,
    gt: &[G],
    gt_seg: BiasSegmentationCriteria<G>,
) -> Result<f32, BiasError>
where
    F: PartialOrd + PartialEq,
    P: PartialOrd + PartialEq,
    G: PartialOrd + PartialEq,
{
    let seg = AdHocSegmentation::new(feature, feature_seg, pred, pred_seg, gt, gt_seg)?;
    let (confusion_a, confusion_d) = seg.get_confusion();
    let precision_a: f32 = confusion_a.true_p / (confusion_a.true_p + confusion_d.false_p);
    let precision_d: f32 = confusion_d.true_p / (confusion_d.true_p + confusion_d.false_p);

    Ok(precision_a - precision_d)
}

pub fn recall_difference<F, P, G>(
    feature: &[F],
    feature_seg: BiasSegmentationCriteria<F>,
    pred: &[P],
    pred_seg: BiasSegmentationCriteria<P>,
    gt: &[G],
    gt_seg: BiasSegmentationCriteria<G>,
) -> Result<f32, BiasError>
where
    F: PartialOrd + PartialEq,
    P: PartialOrd + PartialEq,
    G: PartialOrd + PartialEq,
{
    let seg = AdHocSegmentation::new(feature, feature_seg, pred, pred_seg, gt, gt_seg)?;
    let (confusion_a, confusion_d) = seg.get_confusion();

    let recall_a: f32 = confusion_a.true_p / (confusion_a.true_p + confusion_a.false_n);
    let recall_d: f32 = confusion_d.true_p / (confusion_d.true_p + confusion_d.false_n);

    Ok(recall_a - recall_d)
}

pub fn diff_in_cond_rejection<F, P, G>(
    feature: &[F],
    feature_seg: BiasSegmentationCriteria<F>,
    pred: &[P],
    pred_seg: BiasSegmentationCriteria<P>,
    gt: &[G],
    gt_seg: BiasSegmentationCriteria<G>,
) -> Result<f32, BiasError>
where
    F: PartialOrd + PartialEq,
    P: PartialOrd + PartialEq,
    G: PartialOrd + PartialEq,
{
    let seg = AdHocSegmentation::new(feature, feature_seg, pred, pred_seg, gt, gt_seg)?;
    let n_prime_d: f32 = seg
        .facet_d_pred
        .iter()
        .map(|value| if *value == 0 { 1_f32 } else { 0_f32 })
        .sum::<f32>();
    let n_d: f32 = seg
        .facet_d_true
        .iter()
        .map(|value| if *value == 0 { 1_f32 } else { 0_f32 })
        .sum::<f32>();
    let r_d: f32 = n_d / n_prime_d;

    let n_prime_a: f32 = seg
        .facet_a_pred
        .iter()
        .map(|value| if *value == 0 { 1_f32 } else { 0_f32 })
        .sum::<f32>();
    let n_a: f32 = seg
        .facet_a_true
        .iter()
        .map(|value| if *value == 0 { 1_f32 } else { 0_f32 })
        .sum::<f32>();
    let r_a: f32 = n_a / n_prime_a;

    Ok(r_d - r_a)
}

pub fn specailty_difference<F, P, G>(
    feature: &[F],
    feature_seg: BiasSegmentationCriteria<F>,
    pred: &[P],
    pred_seg: BiasSegmentationCriteria<P>,
    gt: &[G],
    gt_seg: BiasSegmentationCriteria<G>,
) -> Result<f32, BiasError>
where
    F: PartialOrd + PartialEq,
    P: PartialOrd + PartialEq,
    G: PartialOrd + PartialEq,
{
    let seg = AdHocSegmentation::new(feature, feature_seg, pred, pred_seg, gt, gt_seg)?;
    let (confusion_a, confusion_d) = seg.get_confusion();
    let true_negative_rate_d: f32 = confusion_d.true_n / (confusion_d.true_n + confusion_d.false_p);
    let true_negative_rate_a: f32 = confusion_a.true_n / (confusion_a.true_n + confusion_a.false_p);

    Ok(true_negative_rate_d - true_negative_rate_a)
}

pub fn diff_in_rejection_rate<F, P, G>(
    feature: &[F],
    feature_seg: BiasSegmentationCriteria<F>,
    pred: &[P],
    pred_seg: BiasSegmentationCriteria<P>,
    gt: &[G],
    gt_seg: BiasSegmentationCriteria<G>,
) -> Result<f32, BiasError>
where
    F: PartialOrd + PartialEq,
    P: PartialOrd + PartialEq,
    G: PartialOrd + PartialEq,
{
    let seg = AdHocSegmentation::new(feature, feature_seg, pred, pred_seg, gt, gt_seg)?;
    let (confusion_a, confusion_d) = seg.get_confusion();
    let value_d: f32 = confusion_d.true_n / (confusion_d.true_n + confusion_d.false_n);
    let value_a: f32 = confusion_a.true_n / (confusion_a.true_n + confusion_a.false_n);

    Ok(value_d - value_a)
}

pub fn cond_dem_desp_in_pred_labels<F, P, G>(
    feature: &[F],
    feature_seg: BiasSegmentationCriteria<F>,
    pred: &[P],
    pred_seg: BiasSegmentationCriteria<P>,
    gt: &[G],
    gt_seg: BiasSegmentationCriteria<G>,
) -> Result<f32, BiasError>
where
    F: PartialOrd + PartialEq,
    P: PartialOrd + PartialEq,
    G: PartialOrd + PartialEq,
{
    let seg = AdHocSegmentation::new(feature, feature_seg, pred, pred_seg, gt, gt_seg)?;
    let n_prime_0: f32 = seg
        .facet_a_pred
        .iter()
        .map(|value| if *value == 0 { 1_f32 } else { 0_f32 })
        .sum::<f32>()
        + seg
            .facet_d_pred
            .iter()
            .map(|value| if *value == 0 { 1_f32 } else { 0_f32 })
            .sum::<f32>();

    let n_prime_1: f32 = seg
        .facet_a_pred
        .iter()
        .map(|value| if *value == 1 { 1_f32 } else { 0_f32 })
        .sum::<f32>()
        + seg
            .facet_d_pred
            .iter()
            .map(|value| if *value == 1 { 1_f32 } else { 0_f32 })
            .sum::<f32>();

    let n_prime_d_0: f32 = seg
        .facet_d_pred
        .iter()
        .map(|value| if *value == 0 { 1_f32 } else { 0_f32 })
        .sum::<f32>();
    let n_prime_d_1: f32 = seg
        .facet_d_pred
        .iter()
        .map(|value| if *value == 1 { 1_f32 } else { 0_f32 })
        .sum::<f32>();

    Ok(n_prime_d_0 / n_prime_0 - n_prime_d_1 / n_prime_1)
}

pub fn treatment_equity<F, P, G>(
    feature: &[F],
    feature_seg: BiasSegmentationCriteria<F>,
    pred: &[P],
    pred_seg: BiasSegmentationCriteria<P>,
    gt: &[G],
    gt_seg: BiasSegmentationCriteria<G>,
) -> Result<f32, BiasError>
where
    F: PartialOrd + PartialEq,
    P: PartialOrd + PartialEq,
    G: PartialOrd + PartialEq,
{
    let seg = AdHocSegmentation::new(feature, feature_seg, pred, pred_seg, gt, gt_seg)?;
    let (confusion_a, confusion_d) = seg.get_confusion();
    let value_d: f32 = confusion_d.false_n / confusion_d.false_p;
    let value_a: f32 = confusion_a.false_n / confusion_a.false_p;

    Ok(value_d - value_a)
}

pub fn generalized_entropy_v1<F, P, G>(
    feature: &[F],
    feature_seg: BiasSegmentationCriteria<F>,
    pred: &[P],
    pred_seg: BiasSegmentationCriteria<P>,
    gt: &[G],
    gt_seg: BiasSegmentationCriteria<G>,
) -> Result<f32, BiasError>
where
    F: PartialOrd + PartialEq,
    P: PartialOrd + PartialEq,
    G: PartialOrd + PartialEq,
{
    let seg = AdHocSegmentation::new(feature, feature_seg, pred, pred_seg, gt, gt_seg)?;
    let AdHocSegmentation {
        facet_a_true,
        facet_d_true,
        facet_a_pred,
        facet_d_pred,
    } = seg;

    let y_true = [facet_a_true, facet_d_true].concat();
    let y_pred = [facet_a_pred, facet_d_pred].concat();

    let benefits: Vec<f32> = y_true
        .iter()
        .zip(y_pred.iter())
        .map(|(y_true, y_pred)| {
            if *y_pred == 0 && *y_true == 1 {
                0_f32
            } else if *y_pred == 1 && *y_true == 1 {
                1_f32
            } else {
                2_f32
            }
        })
        .collect();

    let sum: f32 = benefits.iter().sum::<f32>();
    let n = benefits.len() as f32;
    let mean: f32 = sum / n;
    let transformed_benefits: Vec<f32> = benefits
        .iter()
        .map(|value| ((*value / mean).powf(2.0)) - 1.0)
        .collect();
    let result: f32 = transformed_benefits.iter().sum::<f32>();
    Ok(result * (0.5 * n))
}

pub(crate) mod inner {
    use super::PostTraining;
    use crate::zip_iters;

    pub(crate) fn diff_in_pos_proportion_in_pred_labels(data: &PostTraining) -> f32 {
        let q_prime_a: f32 = data.dist_a.positive_pred as f32 / data.dist_a.positive_gt as f32;
        let q_prime_d: f32 = data.dist_d.positive_pred as f32 / data.dist_d.positive_gt as f32;

        q_prime_a - q_prime_d
    }

    pub(crate) fn disparate_impact(data: &PostTraining) -> f32 {
        let q_prime_a: f32 = data.dist_a.positive_pred as f32 / data.dist_d.positive_gt as f32;
        let q_prime_d: f32 = data.dist_d.positive_pred as f32 / data.dist_a.positive_gt as f32;

        if q_prime_d == 0.0 {
            return 0.0;
        }
        q_prime_a / q_prime_d
    }

    pub(crate) fn accuracy_difference(data: &PostTraining) -> f32 {
        let acc_a =
            (data.confusion_a.true_p + data.confusion_a.true_n) as f32 / data.dist_a.len as f32;

        let acc_d =
            (data.confusion_d.true_p + data.confusion_d.true_n) as f32 / data.dist_d.len as f32;

        acc_a - acc_d
    }

    pub(crate) fn recall_difference(data: &PostTraining) -> f32 {
        let recall_a: f32 =
            data.confusion_a.true_p / data.confusion_a.true_p + data.confusion_a.false_n;

        let recall_d: f32 =
            data.confusion_d.true_p / data.confusion_d.true_p + data.confusion_d.false_n;

        recall_a - recall_d
    }

    pub(crate) fn diff_in_cond_acceptance(data: &PostTraining) -> f32 {
        let c_facet_a: f32 = data.dist_a.positive_pred as f32 / data.dist_a.positive_gt as f32;
        let c_facet_d: f32 = data.dist_d.positive_pred as f32 / data.dist_d.positive_gt as f32;

        c_facet_a - c_facet_d
    }

    pub(crate) fn diff_in_acceptance_rate(data: &PostTraining) -> f32 {
        // difference in precision across the 2 facets
        let pa: f32 =
            data.confusion_a.true_p / (data.confusion_a.true_p + data.confusion_a.false_p);
        let pd: f32 =
            data.confusion_d.true_p / (data.confusion_d.true_p + data.confusion_d.false_p);

        pa - pd
    }

    pub(crate) fn specailty_difference(data: &PostTraining) -> f32 {
        // difference in true negative rate
        let tnr_a: f32 =
            data.confusion_a.true_n / (data.confusion_a.true_n + data.confusion_a.false_p);
        let tnr_d: f32 =
            data.confusion_d.true_n / (data.confusion_d.true_n + data.confusion_d.false_p);

        tnr_d - tnr_a
    }

    pub(crate) fn diff_in_cond_rejection(data: &PostTraining) -> f32 {
        // difference in ratio of observed negatives to predicted negatives
        let r_a: f32 = (data.dist_a.len - data.dist_a.positive_gt) as f32
            / (data.dist_a.len - data.dist_a.positive_pred) as f32;

        let r_d: f32 = (data.dist_d.len - data.dist_d.positive_gt) as f32
            / (data.dist_d.len - data.dist_d.positive_pred) as f32;

        r_d - r_a
    }

    pub(crate) fn diff_in_rejection_rate(data: &PostTraining) -> f32 {
        // difference in correct rejection rate

        let rr_a: f32 =
            data.confusion_a.true_n / (data.confusion_a.true_n + data.confusion_a.false_n);
        let rr_d: f32 =
            data.confusion_d.true_n / (data.confusion_d.true_n + data.confusion_d.false_n);

        rr_d - rr_a
    }

    pub(crate) fn treatment_equity(data: &PostTraining) -> f32 {
        // difference in ratio of fn/fp
        let ta: f32 = data.confusion_a.false_n / data.confusion_a.false_p;
        let td: f32 = data.confusion_d.false_n / data.confusion_d.false_p;

        td - ta
    }

    pub(crate) fn cond_dem_desp_in_pred_labels(data: &PostTraining) -> f32 {
        // difference in population negative p

        let total_n = data.dist_a.len + data.dist_d.len;

        // DDPL per class
        // n'(0) = predicted negatives for the class / total predicted negatives
        // n'(1) = predicted positives for the class / total predicted positives
        // DDPL_n = (class predicted negatives / n'(0)) - (class predicted positives / n'(1))
        //
        //
        // CDDPL = (1 / n) * sum of (n_i * CDDPL_i) where n_i is the size of the class

        let total_predicted_positives = data.dist_a.positive_pred + data.dist_d.positive_pred;
        let total_predicted_negatives = 1_u64 - total_predicted_positives;

        let d_n_prime_0 =
            (data.dist_d.len - data.dist_d.positive_pred) as f32 / total_predicted_negatives as f32;
        let d_n_prime_1 = data.dist_d.positive_pred as f32 / total_predicted_negatives as f32;

        let ddpl_d = ((data.dist_d.len - data.dist_d.positive_pred) as f32 / d_n_prime_0)
            - (data.dist_d.positive_pred as f32 / d_n_prime_1);

        let a_n_prime_0 =
            (data.dist_a.len - data.dist_a.positive_pred) as f32 / total_predicted_negatives as f32;
        let a_n_prime_1 = data.dist_a.positive_pred as f32 / total_predicted_negatives as f32;

        let ddpl_a = ((data.dist_a.len - data.dist_a.positive_pred) as f32 / a_n_prime_0)
            - (data.dist_a.positive_pred as f32 / a_n_prime_1);

        ((data.dist_a.len as f32 * ddpl_a) + (data.dist_d.len as f32 * ddpl_d)) / total_n as f32
    }

    pub(crate) fn generalized_entropy(y_true: &[i16], y_pred: &[i16]) -> f32 {
        let mut benefits_sum = 0_f32;
        let n = y_true.len();
        let mut benefits_vec: Vec<f32> = Vec::with_capacity(n);

        for (t, p) in zip_iters!(y_true, y_pred) {
            let b = if *p == 0_i16 && *t == 1_i16 {
                0_f32
            } else if *p == 1_i16 && *t == 1_i16 {
                1_f32
            } else {
                2_f32
            };
            benefits_sum += b;
            benefits_vec.push(b);
        }

        // inverse of mean for repeated multiplication
        let m = 1_f32 / (benefits_sum / n as f32);

        benefits_vec
            .into_iter()
            .map(|v| (v * m).powf(2_f32) - 1_f32)
            .sum()
    }
}
