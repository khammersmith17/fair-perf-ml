use std::collections::HashMap;

pub struct PostTrainingData {
    pub facet_a_scores: Vec<i16>,
    pub facet_d_scores: Vec<i16>,
    pub facet_a_trues: Vec<i16>,
    pub facet_d_trues: Vec<i16>,
}

pub struct PostTrainingComputations {
    pub true_positives_a: i16,
    pub true_positives_d: i16,
    pub false_positives_a: i16,
    pub false_positives_d: i16,
    pub false_negatives_a: i16,
    pub false_negatives_d: i16,
    pub true_negatives_a: i16,
    pub true_negatives_d: i16,
}

impl PostTrainingData {
    pub fn general_data_computations(&self) -> PostTrainingComputations {
        PostTrainingComputations {
            true_positives_a: self.true_positives_a(),
            true_positives_d: self.true_positives_d(),
            false_positives_a: self.false_positives_a(),
            false_positives_d: self.false_positives_d(),
            false_negatives_a: self.false_negatives_a(),
            false_negatives_d: self.false_negatives_d(),
            true_negatives_a: self.true_negatives_a(),
            true_negatives_d: self.true_negatives_d(),
        }
    }

    fn true_positives_a(&self) -> i16 {
        self.facet_a_scores
            .iter()
            .zip(self.facet_a_trues.iter())
            .map(|(y_pred, y_true)| {
                if *y_pred == 1 as i16 && *y_true == 1 as i16 {
                    1
                } else {
                    0
                }
            })
            .sum::<i16>()
    }

    fn true_positives_d(&self) -> i16 {
        self.facet_d_scores
            .iter()
            .zip(self.facet_d_trues.iter())
            .map(
                |(y_pred, y_true)| {
                    if *y_pred == 1 && *y_true == 1 {
                        1
                    } else {
                        0
                    }
                },
            )
            .sum::<i16>()
    }

    fn false_positives_a(&self) -> i16 {
        self.facet_a_scores
            .iter()
            .zip(self.facet_a_trues.iter())
            .map(
                |(y_pred, y_true)| {
                    if *y_pred == 1 && *y_true == 0 {
                        1
                    } else {
                        0
                    }
                },
            )
            .sum::<i16>()
    }

    fn false_positives_d(&self) -> i16 {
        self.facet_d_scores
            .iter()
            .zip(self.facet_d_trues.iter())
            .map(
                |(y_pred, y_true)| {
                    if *y_pred == 1 && *y_true == 0 {
                        1
                    } else {
                        0
                    }
                },
            )
            .sum::<i16>()
    }

    fn false_negatives_a(&self) -> i16 {
        self.facet_a_scores
            .iter()
            .zip(self.facet_a_trues.iter())
            .map(
                |(y_pred, y_true)| {
                    if *y_pred == 0 && *y_true == 1 {
                        1
                    } else {
                        0
                    }
                },
            )
            .sum::<i16>()
    }

    fn false_negatives_d(&self) -> i16 {
        self.facet_d_scores
            .iter()
            .zip(self.facet_d_trues.iter())
            .map(
                |(y_pred, y_true)| {
                    if *y_pred == 0 && *y_true == 1 {
                        1
                    } else {
                        0
                    }
                },
            )
            .sum::<i16>()
    }

    fn true_negatives_a(&self) -> i16 {
        self.facet_a_scores
            .iter()
            .zip(self.facet_a_trues.iter())
            .map(
                |(y_pred, y_true)| {
                    if *y_pred == 0 && *y_true == 0 {
                        1
                    } else {
                        0
                    }
                },
            )
            .sum::<i16>()
    }

    fn true_negatives_d(&self) -> i16 {
        self.facet_d_scores
            .iter()
            .zip(self.facet_d_trues.iter())
            .map(
                |(y_pred, y_true)| {
                    if *y_pred == 0 && *y_true == 0 {
                        1
                    } else {
                        0
                    }
                },
            )
            .sum::<i16>()
    }
}

pub fn diff_in_pos_proportion_in_pred_labels(data: &PostTrainingData) -> f32 {
    let q_prime_a: f32 =
        (data.facet_a_scores.iter().sum::<i16>() / data.facet_a_trues.iter().sum::<i16>()).into();
    let q_prime_d: f32 =
        (data.facet_d_scores.iter().sum::<i16>() / data.facet_d_trues.iter().sum::<i16>()).into();

    return q_prime_a - q_prime_d;
}

pub fn disparate_impact(data: &PostTrainingData) -> f32 {
    let q_prime_a: f32 =
        (data.facet_a_scores.iter().sum::<i16>() / data.facet_d_trues.iter().sum::<i16>()).into();
    let q_prime_d: f32 =
        (data.facet_d_scores.iter().sum::<i16>() / data.facet_d_trues.iter().sum::<i16>()).into();

    if q_prime_d == 0.0 {
        return 0.0;
    }
    return (q_prime_a / q_prime_d) as f32;
}

pub fn accuracy_difference(
    pre_computed_data: &PostTrainingComputations,
    data: &PostTrainingData,
) -> f32 {
    let acc_a: f32 = ((pre_computed_data.true_positives_a + pre_computed_data.true_negatives_a)
        / data.facet_a_scores.len() as i16)
        .into();

    let acc_d: f32 = ((pre_computed_data.true_positives_d + pre_computed_data.true_negatives_d)
        / data.facet_d_scores.len() as i16)
        .into();

    return acc_a - acc_d;
}

pub fn recall_difference(pre_computed_data: &PostTrainingComputations) -> f32 {
    let recall_a: f32 = (pre_computed_data.true_positives_a
        / (pre_computed_data.true_positives_a + pre_computed_data.false_negatives_a))
        .into();
    let recall_d: f32 = (pre_computed_data.true_positives_d
        / (pre_computed_data.true_positives_d + pre_computed_data.false_negatives_d))
        .into();

    let result: f32 = recall_a - recall_d;

    result
}

pub fn diff_in_cond_acceptance(data: &PostTrainingData) -> f32 {
    let sum_true_facet_a: f32 = data.facet_a_trues.iter().sum::<i16>().into();
    let sum_scores_facet_a: f32 = data.facet_a_scores.iter().sum::<i16>().into();
    let c_facet_a: f32 = sum_true_facet_a / sum_scores_facet_a;

    let sum_true_facet_d: f32 = data.facet_d_trues.iter().sum::<i16>().into();
    let sum_scores_facet_d: f32 = data.facet_d_scores.iter().sum::<i16>().into();
    let c_facet_d: f32 = sum_true_facet_d / sum_scores_facet_d;

    let result = c_facet_a - c_facet_d;

    result
}

pub fn diff_in_acceptance_rate(pre_computed_data: &PostTrainingComputations) -> f32 {
    let precision_a: f32 = (pre_computed_data.true_positives_a
        / (pre_computed_data.true_positives_a + pre_computed_data.false_positives_a))
        .into();
    let precision_d: f32 = (pre_computed_data.true_positives_d
        / (pre_computed_data.true_positives_d + pre_computed_data.false_positives_d))
        .into();

    return precision_a - precision_d;
}

pub fn specailty_difference(pre_computed_data: &PostTrainingComputations) -> f32 {
    let true_negative_rate_d: f32 = (pre_computed_data.true_negatives_d
        / (pre_computed_data.true_negatives_d + pre_computed_data.false_positives_d))
        .into();
    let true_negative_rate_a: f32 = (pre_computed_data.true_negatives_a
        / (pre_computed_data.true_negatives_a + pre_computed_data.false_positives_a))
        .into();

    return true_negative_rate_d - true_negative_rate_a;
}

pub fn diff_in_cond_rejection(data: &PostTrainingData) -> f32 {
    let n_prime_d = data
        .facet_d_scores
        .iter()
        .map(|value| if *value == 0 { 1 } else { 0 })
        .sum::<i16>();
    let n_d = data
        .facet_d_trues
        .iter()
        .map(|value| if *value == 0 { 1 } else { 0 })
        .sum::<i16>();
    let r_d: f32 = (n_d / n_prime_d).into();

    let n_prime_a = data
        .facet_a_scores
        .iter()
        .map(|value| if *value == 0 { 1 } else { 0 })
        .sum::<i16>();
    let n_a = data
        .facet_a_trues
        .iter()
        .map(|value| if *value == 0 { 1 } else { 0 })
        .sum::<i16>();
    let r_a: f32 = (n_a / n_prime_a).into();

    return r_d - r_a;
}

pub fn diff_in_rejection_rate(pre_computed_data: &PostTrainingComputations) -> f32 {
    let value_d: f32 = pre_computed_data.true_negatives_d as f32
        / (pre_computed_data.true_negatives_d + pre_computed_data.false_negatives_d) as f32;
    let value_a: f32 = pre_computed_data.true_negatives_a as f32
        / (pre_computed_data.true_negatives_a + pre_computed_data.false_negatives_a) as f32;

    return value_d - value_a;
}

pub fn treatment_equity(pre_computed_data: &PostTrainingComputations) -> f32 {
    let value_d: f32 =
        pre_computed_data.false_negatives_d as f32 / pre_computed_data.false_positives_d as f32;
    let value_a: f32 =
        pre_computed_data.false_negatives_a as f32 / pre_computed_data.false_positives_a as f32;

    return value_d - value_a;
}

pub fn cond_dem_desp_in_pred_labels(data: &PostTrainingData) -> f32 {
    let n_prime_0: f32 = (data
        .facet_a_scores
        .iter()
        .map(|value| if *value == 0 { 1 } else { 0 })
        .sum::<i16>()
        + data
            .facet_d_scores
            .iter()
            .map(|value| if *value == 0 { 1 } else { 0 })
            .sum::<i16>())
    .into();

    let n_prime_1: f32 = (data
        .facet_a_scores
        .iter()
        .map(|value| if *value == 1 { 1 } else { 0 })
        .sum::<i16>()
        + data
            .facet_d_scores
            .iter()
            .map(|value| if *value == 1 { 1 } else { 0 })
            .sum::<i16>())
    .into();

    let n_prime_d_0: f32 = data
        .facet_d_scores
        .iter()
        .map(|value| if *value == 0 { 1 } else { 0 })
        .sum::<i16>()
        .into();
    let n_prime_d_1: f32 = data
        .facet_d_scores
        .iter()
        .map(|value| if *value == 1 { 1 } else { 0 })
        .sum::<i16>()
        .into();

    let result: f32 = (n_prime_d_0 / n_prime_0) as f32 - (n_prime_d_1 / n_prime_1) as f32;
    result
}

pub fn generalized_entropy(data: &PostTrainingData) -> f32 {
    let y_true = [data.facet_a_trues.clone(), data.facet_d_trues.clone()].concat();
    let y_pred = [data.facet_a_scores.clone(), data.facet_d_scores.clone()].concat();

    let benefits: Vec<i8> = y_true
        .iter()
        .zip(y_pred.iter())
        .map(|(y_true, y_pred)| {
            if *y_pred == 0 && *y_true == 1 {
                0
            } else if *y_pred == 1 && *y_true == 1 {
                1
            } else {
                2
            }
        })
        .collect();

    let sum: f32 = benefits.iter().sum::<i8>() as f32;
    let n = benefits.len() as f32;
    let mean: f32 = sum / n;
    let transformed_benefits: Vec<f32> = benefits
        .iter()
        .map(|value| ((*value as f32 / mean).powf(2.0)) - 1.0)
        .collect();
    let result: f32 = transformed_benefits.iter().sum::<f32>();
    return result * (0.5 * n);
}

pub fn post_training_bias(
    facet_a_scores: Vec<i16>,
    facet_d_scores: Vec<i16>,
    facet_a_trues: Vec<i16>,
    facet_d_trues: Vec<i16>,
) -> Result<HashMap<String, f32>, String> {
    let data = PostTrainingData {
        facet_a_scores,
        facet_d_scores,
        facet_a_trues,
        facet_d_trues,
    };
    let pre_computed_data: PostTrainingComputations = data.general_data_computations();
    let mut result = HashMap::new();
    result.insert("DDPL".into(), diff_in_pos_proportion_in_pred_labels(&data));
    result.insert("DI".into(), disparate_impact(&data));
    result.insert("AD".into(), accuracy_difference(&pre_computed_data, &data));
    result.insert("RD".into(), recall_difference(&pre_computed_data));
    result.insert("CDACC".into(), diff_in_cond_acceptance(&data));
    result.insert("DAR".into(), diff_in_acceptance_rate(&pre_computed_data));
    result.insert("SD".into(), specailty_difference(&pre_computed_data));
    result.insert("DCR".into(), diff_in_cond_rejection(&data));
    result.insert("DRR".into(), diff_in_rejection_rate(&pre_computed_data));
    result.insert("TE".into(), treatment_equity(&pre_computed_data));
    result.insert("CCDPL".into(), cond_dem_desp_in_pred_labels(&data));
    result.insert("GE".into(), generalized_entropy(&data));

    Ok(result)
}
