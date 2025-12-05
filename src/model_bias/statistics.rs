use super::zip_iters;
use super::{PostTrainingComputations, PostTrainingData, PostTrainingDataV2};

pub fn diff_in_pos_proportion_in_pred_labels_v1(data: &PostTrainingData) -> f32 {
    let q_prime_a: f32 = data.facet_a_scores.iter().sum::<i16>() as f32
        / data.facet_a_trues.iter().sum::<i16>() as f32;
    let q_prime_d: f32 = data.facet_d_scores.iter().sum::<i16>() as f32
        / data.facet_d_trues.iter().sum::<i16>() as f32;

    return q_prime_a - q_prime_d;
}

pub(crate) fn diff_in_pos_proportion_in_pred_labels(data: &PostTrainingDataV2) -> f32 {
    let q_prime_a: f32 = data.dist_a.positive_pred as f32 / data.dist_a.positive_gt as f32;
    let q_prime_d: f32 = data.dist_d.positive_pred as f32 / data.dist_d.positive_gt as f32;

    q_prime_a - q_prime_d
}

pub fn disparate_impact_v1(data: &PostTrainingData) -> f32 {
    let q_prime_a: f32 = data.facet_a_scores.iter().sum::<i16>() as f32
        / data.facet_d_trues.iter().sum::<i16>() as f32;
    let q_prime_d: f32 = data.facet_d_scores.iter().sum::<i16>() as f32
        / data.facet_d_trues.iter().sum::<i16>() as f32;

    if q_prime_d == 0.0 {
        return 0.0;
    }
    q_prime_a / q_prime_d
}

pub(crate) fn disparate_impact(data: &PostTrainingDataV2) -> f32 {
    let q_prime_a: f32 = data.dist_a.positive_pred as f32 / data.dist_d.positive_gt as f32;
    let q_prime_d: f32 = data.dist_d.positive_pred as f32 / data.dist_a.positive_gt as f32;

    if q_prime_d == 0.0 {
        return 0.0;
    }
    q_prime_a / q_prime_d
}

pub fn accuracy_difference_v1(
    pre_computed_data: &PostTrainingComputations,
    data: &PostTrainingData,
) -> f32 {
    let acc_a: f32 = (pre_computed_data.true_positives_a + pre_computed_data.true_negatives_a)
        / data.facet_a_scores.len() as f32;

    let acc_d: f32 = (pre_computed_data.true_positives_d + pre_computed_data.true_negatives_d)
        / data.facet_d_scores.len() as f32;

    return acc_a - acc_d;
}

pub(crate) fn accuracy_difference(data: &PostTrainingDataV2) -> f32 {
    let acc_a = (data.confusion_a.true_p + data.confusion_a.true_n) as f32 / data.dist_a.len as f32;

    let acc_d = (data.confusion_d.true_p + data.confusion_d.true_n) as f32 / data.dist_d.len as f32;

    acc_a - acc_d
}

pub fn recall_difference_v1(pre_computed_data: &PostTrainingComputations) -> f32 {
    let recall_a: f32 = pre_computed_data.true_positives_a
        / (pre_computed_data.true_positives_a + pre_computed_data.false_negatives_a);
    let recall_d: f32 = pre_computed_data.true_positives_d
        / (pre_computed_data.true_positives_d + pre_computed_data.false_negatives_d);

    recall_a - recall_d
}

pub(crate) fn recall_difference(data: &PostTrainingDataV2) -> f32 {
    let recall_a: f32 =
        data.confusion_a.true_p / data.confusion_a.true_p + data.confusion_a.false_n;

    let recall_d: f32 =
        data.confusion_d.true_p / data.confusion_d.true_p + data.confusion_d.false_n;

    recall_a - recall_d
}

pub fn diff_in_cond_acceptance_v1(data: &PostTrainingData) -> f32 {
    let sum_true_facet_a: f32 = data.facet_a_trues.iter().sum::<i16>().into();
    let sum_scores_facet_a: f32 = data.facet_a_scores.iter().sum::<i16>().into();
    let c_facet_a: f32 = sum_true_facet_a / sum_scores_facet_a;

    let sum_true_facet_d: f32 = data.facet_d_trues.iter().sum::<i16>().into();
    let sum_scores_facet_d: f32 = data.facet_d_scores.iter().sum::<i16>().into();
    let c_facet_d: f32 = sum_true_facet_d / sum_scores_facet_d;

    c_facet_a - c_facet_d
}

pub(crate) fn diff_in_cond_acceptance(data: &PostTrainingDataV2) -> f32 {
    let c_facet_a: f32 = data.dist_a.positive_pred as f32 / data.dist_a.positive_gt as f32;
    let c_facet_d: f32 = data.dist_d.positive_pred as f32 / data.dist_d.positive_gt as f32;

    c_facet_a - c_facet_d
}

pub fn diff_in_acceptance_rate_v1(pre_computed_data: &PostTrainingComputations) -> f32 {
    let precision_a: f32 = pre_computed_data.true_positives_a
        / (pre_computed_data.true_positives_a + pre_computed_data.false_positives_a);
    let precision_d: f32 = pre_computed_data.true_positives_d
        / (pre_computed_data.true_positives_d + pre_computed_data.false_positives_d);

    precision_a - precision_d
}

pub(crate) fn diff_in_acceptance_rate(data: &PostTrainingDataV2) -> f32 {
    // difference in precision across the 2 facets
    let pa: f32 = data.confusion_a.true_p / (data.confusion_a.true_p + data.confusion_a.false_p);
    let pd: f32 = data.confusion_d.true_p / (data.confusion_d.true_p + data.confusion_d.false_p);

    pa - pd
}

pub fn specailty_difference_v1(pre_computed_data: &PostTrainingComputations) -> f32 {
    let true_negative_rate_d: f32 = pre_computed_data.true_negatives_d
        / (pre_computed_data.true_negatives_d + pre_computed_data.false_positives_d);
    let true_negative_rate_a: f32 = pre_computed_data.true_negatives_a
        / (pre_computed_data.true_negatives_a + pre_computed_data.false_positives_a);

    true_negative_rate_d - true_negative_rate_a
}

pub(crate) fn specailty_difference(data: &PostTrainingDataV2) -> f32 {
    // difference in true negative rate
    let tnr_a: f32 = data.confusion_a.true_n / (data.confusion_a.true_n + data.confusion_a.false_p);
    let tnr_d: f32 = data.confusion_d.true_n / (data.confusion_d.true_n + data.confusion_d.false_p);

    tnr_d - tnr_a
}

pub fn diff_in_cond_rejection_v1(data: &PostTrainingData) -> f32 {
    let n_prime_d: f32 = data
        .facet_d_scores
        .iter()
        .map(|value| if *value == 0 { 1_f32 } else { 0_f32 })
        .sum::<f32>();
    let n_d: f32 = data
        .facet_d_trues
        .iter()
        .map(|value| if *value == 0 { 1_f32 } else { 0_f32 })
        .sum::<f32>();
    let r_d: f32 = n_d / n_prime_d;

    let n_prime_a: f32 = data
        .facet_a_scores
        .iter()
        .map(|value| if *value == 0 { 1_f32 } else { 0_f32 })
        .sum::<f32>();
    let n_a: f32 = data
        .facet_a_trues
        .iter()
        .map(|value| if *value == 0 { 1_f32 } else { 0_f32 })
        .sum::<f32>();
    let r_a: f32 = n_a / n_prime_a;

    r_d - r_a
}

pub(crate) fn diff_in_cond_rejection(data: &PostTrainingDataV2) -> f32 {
    // difference in ratio of observed negatives to predicted negatives
    let r_a: f32 = (data.dist_a.len - data.dist_a.positive_gt) as f32
        / (data.dist_a.len - data.dist_a.positive_pred) as f32;

    let r_d: f32 = (data.dist_d.len - data.dist_d.positive_gt) as f32
        / (data.dist_d.len - data.dist_d.positive_pred) as f32;

    r_d - r_a
}

pub fn diff_in_rejection_rate_v1(pre_computed_data: &PostTrainingComputations) -> f32 {
    let value_d: f32 = pre_computed_data.true_negatives_d
        / (pre_computed_data.true_negatives_d + pre_computed_data.false_negatives_d);
    let value_a: f32 = pre_computed_data.true_negatives_a
        / (pre_computed_data.true_negatives_a + pre_computed_data.false_negatives_a);

    value_d - value_a
}

pub(crate) fn diff_in_rejection_rate(data: &PostTrainingDataV2) -> f32 {
    // difference in correct rejection rate

    let rr_a: f32 = data.confusion_a.true_n / (data.confusion_a.true_n + data.confusion_a.false_n);
    let rr_d: f32 = data.confusion_d.true_n / (data.confusion_d.true_n + data.confusion_d.false_n);

    rr_d - rr_a
}

pub fn treatment_equity_v1(pre_computed_data: &PostTrainingComputations) -> f32 {
    let value_d: f32 = pre_computed_data.false_negatives_d / pre_computed_data.false_positives_d;
    let value_a: f32 = pre_computed_data.false_negatives_a / pre_computed_data.false_positives_a;

    value_d - value_a
}

pub(crate) fn treatment_equity(data: &PostTrainingDataV2) -> f32 {
    // difference in ratio of fn/fp
    let ta: f32 = data.confusion_a.false_n / data.confusion_a.false_p;
    let td: f32 = data.confusion_d.false_n / data.confusion_d.false_p;

    td - ta
}

pub fn cond_dem_desp_in_pred_labels_v1(data: &PostTrainingData) -> f32 {
    let n_prime_0: f32 = data
        .facet_a_scores
        .iter()
        .map(|value| if *value == 0 { 1_f32 } else { 0_f32 })
        .sum::<f32>()
        + data
            .facet_d_scores
            .iter()
            .map(|value| if *value == 0 { 1_f32 } else { 0_f32 })
            .sum::<f32>();

    let n_prime_1: f32 = data
        .facet_a_scores
        .iter()
        .map(|value| if *value == 1 { 1_f32 } else { 0_f32 })
        .sum::<f32>()
        + data
            .facet_d_scores
            .iter()
            .map(|value| if *value == 1 { 1_f32 } else { 0_f32 })
            .sum::<f32>();

    let n_prime_d_0: f32 = data
        .facet_d_scores
        .iter()
        .map(|value| if *value == 0 { 1_f32 } else { 0_f32 })
        .sum::<f32>();
    let n_prime_d_1: f32 = data
        .facet_d_scores
        .iter()
        .map(|value| if *value == 1 { 1_f32 } else { 0_f32 })
        .sum::<f32>();

    n_prime_d_0 / n_prime_0 - n_prime_d_1 / n_prime_1
}

pub(crate) fn cond_dem_desp_in_pred_labels(data: &PostTrainingDataV2) -> f32 {
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

/*
pub fn generalized_entropy(data: &PostTrainingData) -> f32 {
    let y_true = [data.facet_a_trues.clone(), data.facet_d_trues.clone()].concat();
    let y_pred = [data.facet_a_scores.clone(), data.facet_d_scores.clone()].concat();

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
    result * (0.5 * n)
}
*/

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

    let m = 1_f32 / (benefits_sum / n as f32);

    benefits_vec
        .into_iter()
        .map(|v| (v * m).powf(2_f32) - 1_f32)
        .sum()
}
