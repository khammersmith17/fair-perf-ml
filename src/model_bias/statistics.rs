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

        // Using the feature data length as a preallocation heuristic.
        let mut facet_a_pred: Vec<i16> = Vec::with_capacity(n_feature / 2_usize);
        let mut facet_a_true: Vec<i16> = Vec::with_capacity(n_feature / 2_usize);
        let mut facet_d_pred: Vec<i16> = Vec::with_capacity(n_feature / 2_usize);
        let mut facet_d_true: Vec<i16> = Vec::with_capacity(n_feature / 2_usize);
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

/// Computes the difference in the proportion of positive predicted labels between the
/// advantaged and disadvantaged facets. A value near zero indicates parity.
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

/// Computes the ratio of positive prediction rates between the advantaged and disadvantaged
/// facets. A value of 1.0 indicates no disparate impact; values below 0.8 are commonly flagged.
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

/// Computes the difference in conditional acceptance rates between the advantaged and
/// disadvantaged facets (true positive rate given a positive ground truth).
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

/// Computes the difference in accuracy between the advantaged and disadvantaged facets.
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

/// Computes the difference in precision (positive predictive value) between the advantaged and
/// disadvantaged facets.
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

/// Computes the difference in recall (true positive rate) between the advantaged and
/// disadvantaged facets.
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

/// Computes the difference in conditional rejection rates between the disadvantaged and
/// advantaged facets (ratio of observed negatives to predicted negatives).
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

/// Computes the difference in true negative rate (specificity) between the disadvantaged and
/// advantaged facets.
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

/// Computes the difference in negative predictive value (rejection accuracy) between the
/// disadvantaged and advantaged facets.
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

/// Computes the conditional demographic disparity in predicted labels across the two facets.
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

/// Computes the difference in the ratio of false negatives to false positives (treatment equity)
/// between the disadvantaged and advantaged facets.
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

/// Computes the generalized entropy index across both facets combined, measuring overall
/// inequality in the distribution of benefits assigned by the model.
pub fn generalized_entropy<F, P, G>(
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
    use crate::errors::{ModelPerfResult, ModelPerformanceError};
    use crate::zip_iters;

    pub(crate) fn diff_in_pos_proportion_in_pred_labels(
        data: &PostTraining,
    ) -> Result<f32, ModelPerformanceError> {
        Ok(data.dist_a.positive_prediction_rate()? - data.dist_d.positive_prediction_rate()?)
    }

    pub(crate) fn disparate_impact(data: &PostTraining) -> ModelPerfResult<f32> {
        let q_prime_a: f32 = data.dist_a.positive_prediction_rate()?;
        let q_prime_d: f32 = data.dist_d.positive_prediction_rate()?;

        if q_prime_d == 0.0 {
            return Err(ModelPerformanceError::InvalidData);
        }
        Ok(q_prime_a / q_prime_d)
    }

    pub(crate) fn accuracy_difference(data: &PostTraining) -> f32 {
        data.confusion_a.accuracy() - data.confusion_d.accuracy()
    }

    pub(crate) fn recall_difference(data: &PostTraining) -> f32 {
        use crate::model_perf::statistics::classification_metrics_from_parts::recall_positive as recall;
        recall(&data.confusion_a) - recall(&data.confusion_d)
    }

    pub(crate) fn diff_in_cond_acceptance(
        data: &PostTraining,
    ) -> Result<f32, ModelPerformanceError> {
        Ok(data.dist_a.cond_acceptance()? - data.dist_d.cond_acceptance()?)
    }

    /// difference in precision across the 2 facets
    pub(crate) fn diff_in_acceptance_rate(data: &PostTraining) -> f32 {
        use crate::model_perf::statistics::classification_metrics_from_parts::precision_positive as precision;
        precision(&data.confusion_a) - precision(&data.confusion_d)
    }

    /// difference in true negative rate
    pub(crate) fn specailty_difference(data: &PostTraining) -> f32 {
        use crate::model_perf::statistics::classification_metrics_from_parts::recall_negative as tnr;
        tnr(&data.confusion_a) - tnr(&data.confusion_d)
    }

    /// difference in ratio of observed negatives to predicted negatives
    pub(crate) fn diff_in_cond_rejection(data: &PostTraining) -> ModelPerfResult<f32> {
        Ok(data.dist_d.conditional_rejection()? - data.dist_a.conditional_rejection()?)
    }

    /// difference in accuracy of rejection rate
    pub(crate) fn diff_in_rejection_rate(data: &PostTraining) -> ModelPerfResult<f32> {
        Ok(data.confusion_d.rejection_rate_acc()? - data.confusion_a.rejection_rate_acc()?)
    }

    /// difference in ratio of fn/fp
    pub(crate) fn treatment_equity(data: &PostTraining) -> ModelPerfResult<f32> {
        Ok(data.confusion_d.fn_fp_ratio()? - data.confusion_a.fn_fp_ratio()?)
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
        let total_predicted_negatives = total_n - total_predicted_positives;

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

#[cfg(test)]
mod statistics_tests {
    use super::*;
    use crate::data_handler::{BiasSegmentationCriteria, BiasSegmentationType};

    // Symmetric dataset: 4 advantaged (feat=1), 4 disadvantaged (feat=0).
    // Each facet has identical pred=[1,1,0,0] and gt=[1,0,1,0].
    // All difference metrics should be 0; disparate_impact should be 1.
    fn sym_feat() -> Vec<i32> {
        vec![1, 1, 1, 1, 0, 0, 0, 0]
    }
    fn sym_pred() -> Vec<i32> {
        vec![1, 1, 0, 0, 1, 1, 0, 0]
    }
    fn sym_gt() -> Vec<i32> {
        vec![1, 0, 1, 0, 1, 0, 1, 0]
    }

    fn feat_seg() -> BiasSegmentationCriteria<i32> {
        BiasSegmentationCriteria::new(1_i32, BiasSegmentationType::Label)
    }

    // --- AdHocSegmentation error paths ---

    #[test]
    fn mismatched_lengths_returns_data_length_error() {
        // pred is shorter than feature/gt
        let result = accuracy_difference(
            &[1_i32, 0, 1],
            feat_seg(),
            &[1_i32, 0],
            feat_seg(),
            &[1_i32, 0, 1],
            feat_seg(),
        );
        assert!(result.is_err());
    }

    #[test]
    fn no_facet_deviation_returns_error_when_one_group_empty() {
        // all features == 1 → facet_d is empty → NoFacetDeviation
        let feat = vec![1_i32, 1, 1, 1];
        let pred = vec![1_i32, 0, 1, 0];
        let gt = vec![1_i32, 0, 1, 0];
        let result = accuracy_difference(&feat, feat_seg(), &pred, feat_seg(), &gt, feat_seg());
        assert!(result.is_err());
    }

    // --- diff_in_pos_proportion_in_pred_labels ---

    #[test]
    fn diff_in_pos_proportion_symmetric_is_zero() {
        let v = diff_in_pos_proportion_in_pred_labels(
            &sym_feat(),
            feat_seg(),
            &sym_pred(),
            feat_seg(),
            &sym_gt(),
            feat_seg(),
        )
        .unwrap();
        assert!((v).abs() < 1e-4);
    }

    #[test]
    fn diff_in_pos_proportion_mismatch_errors() {
        assert!(diff_in_pos_proportion_in_pred_labels(
            &[1_i32, 0],
            feat_seg(),
            &[1_i32],
            feat_seg(),
            &[1_i32, 0],
            feat_seg(),
        )
        .is_err());
    }

    // --- disparate_impact ---

    #[test]
    fn disparate_impact_symmetric_is_one() {
        let v = disparate_impact(
            &sym_feat(),
            feat_seg(),
            &sym_pred(),
            feat_seg(),
            &sym_gt(),
            feat_seg(),
        )
        .unwrap();
        assert!((v - 1.0).abs() < 1e-4);
    }

    #[test]
    fn disparate_impact_mismatch_errors() {
        assert!(disparate_impact(
            &[1_i32, 0],
            feat_seg(),
            &[1_i32],
            feat_seg(),
            &[1_i32, 0],
            feat_seg(),
        )
        .is_err());
    }

    // --- diff_in_cond_acceptance ---

    #[test]
    fn diff_in_cond_acceptance_symmetric_is_zero() {
        let v = diff_in_cond_acceptance(
            &sym_feat(),
            feat_seg(),
            &sym_pred(),
            feat_seg(),
            &sym_gt(),
            feat_seg(),
        )
        .unwrap();
        assert!((v).abs() < 1e-4);
    }

    // --- accuracy_difference ---

    #[test]
    fn accuracy_difference_symmetric_is_zero() {
        let v = accuracy_difference(
            &sym_feat(),
            feat_seg(),
            &sym_pred(),
            feat_seg(),
            &sym_gt(),
            feat_seg(),
        )
        .unwrap();
        assert!((v).abs() < 1e-4);
    }

    #[test]
    fn accuracy_difference_nonzero_when_groups_differ() {
        // facet_a (feat=1): pred=[1,0], gt=[1,0] → perfect → acc=1.0
        // facet_d (feat=0): pred=[1,0], gt=[0,1] → all wrong → acc=0.0
        let feat = vec![1_i32, 1, 0, 0];
        let pred = vec![1_i32, 0, 1, 0];
        let gt = vec![1_i32, 0, 0, 1];
        let v = accuracy_difference(&feat, feat_seg(), &pred, feat_seg(), &gt, feat_seg()).unwrap();
        assert!((v - 1.0).abs() < 1e-4);
    }

    // --- diff_in_acceptance_rate ---

    #[test]
    fn diff_in_acceptance_rate_symmetric_is_zero() {
        let v = diff_in_acceptance_rate(
            &sym_feat(),
            feat_seg(),
            &sym_pred(),
            feat_seg(),
            &sym_gt(),
            feat_seg(),
        )
        .unwrap();
        assert!((v).abs() < 1e-4);
    }

    // --- recall_difference ---

    #[test]
    fn recall_difference_symmetric_is_zero() {
        let v = recall_difference(
            &sym_feat(),
            feat_seg(),
            &sym_pred(),
            feat_seg(),
            &sym_gt(),
            feat_seg(),
        )
        .unwrap();
        assert!((v).abs() < 1e-4);
    }

    #[test]
    fn recall_difference_nonzero_when_groups_differ() {
        // facet_a (feat=1): pred=[1,0], gt=[1,0] → TP=1,FN=0 → recall=1.0
        // facet_d (feat=0): pred=[1,0], gt=[0,1] → TP=0,FN=1 → recall=0.0
        let feat = vec![1_i32, 1, 0, 0];
        let pred = vec![1_i32, 0, 1, 0];
        let gt = vec![1_i32, 0, 0, 1];
        let v = recall_difference(&feat, feat_seg(), &pred, feat_seg(), &gt, feat_seg()).unwrap();
        assert!((v - 1.0).abs() < 1e-4);
    }

    // --- diff_in_cond_rejection ---

    #[test]
    fn diff_in_cond_rejection_symmetric_is_zero() {
        let v = diff_in_cond_rejection(
            &sym_feat(),
            feat_seg(),
            &sym_pred(),
            feat_seg(),
            &sym_gt(),
            feat_seg(),
        )
        .unwrap();
        assert!((v).abs() < 1e-4);
    }

    // --- specailty_difference ---

    #[test]
    fn specailty_difference_symmetric_is_zero() {
        let v = specailty_difference(
            &sym_feat(),
            feat_seg(),
            &sym_pred(),
            feat_seg(),
            &sym_gt(),
            feat_seg(),
        )
        .unwrap();
        assert!((v).abs() < 1e-4);
    }

    // --- diff_in_rejection_rate ---

    #[test]
    fn diff_in_rejection_rate_symmetric_is_zero() {
        let v = diff_in_rejection_rate(
            &sym_feat(),
            feat_seg(),
            &sym_pred(),
            feat_seg(),
            &sym_gt(),
            feat_seg(),
        )
        .unwrap();
        assert!((v).abs() < 1e-4);
    }

    // --- cond_dem_desp_in_pred_labels ---

    #[test]
    fn cond_dem_desp_in_pred_labels_returns_ok_for_symmetric_data() {
        let v = cond_dem_desp_in_pred_labels(
            &sym_feat(),
            feat_seg(),
            &sym_pred(),
            feat_seg(),
            &sym_gt(),
            feat_seg(),
        )
        .unwrap();
        // With symmetric data both groups have identical distributions → result is 0
        assert!((v).abs() < 1e-4);
    }

    // --- treatment_equity ---

    #[test]
    fn treatment_equity_symmetric_is_zero() {
        let v = treatment_equity(
            &sym_feat(),
            feat_seg(),
            &sym_pred(),
            feat_seg(),
            &sym_gt(),
            feat_seg(),
        )
        .unwrap();
        assert!((v).abs() < 1e-4);
    }

    // --- generalized_entropy ---

    #[test]
    fn generalized_entropy_returns_ok_for_symmetric_data() {
        assert!(generalized_entropy(
            &sym_feat(),
            feat_seg(),
            &sym_pred(),
            feat_seg(),
            &sym_gt(),
            feat_seg(),
        )
        .is_ok());
    }

    #[test]
    fn generalized_entropy_mismatch_errors() {
        assert!(generalized_entropy(
            &[1_i32, 0],
            feat_seg(),
            &[1_i32],
            feat_seg(),
            &[1_i32, 0],
            feat_seg(),
        )
        .is_err());
    }
}
