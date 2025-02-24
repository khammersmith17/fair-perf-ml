use std::collections::HashMap;

pub struct PreTraining {
    pub facet_a: Vec<i16>,
    pub facet_d: Vec<i16>,
}

impl PreTraining {
    pub fn generate(&self) -> PreTrainingComputations {
        let a_acceptance: f32 = self.facet_a.iter().sum::<i16>() as f32 / self.facet_a.len() as f32;
        let d_acceptance: f32 = self.facet_d.iter().sum::<i16>() as f32 / self.facet_d.len() as f32;
        PreTrainingComputations {
            a_acceptance,
            d_acceptance,
        }
    }
}

pub struct PreTrainingComputations {
    pub a_acceptance: f32,
    pub d_acceptance: f32,
}

pub fn class_imbalance(data: &PreTraining) -> f32 {
    return (data.facet_a.len() as i32 - data.facet_d.len() as i32).abs() as f32
        / (data.facet_a.len() + data.facet_d.len()) as f32;
}

pub fn diff_in_proportion_of_labels(data: &PreTraining) -> f32 {
    let q_a: f32 = data.facet_a.iter().sum::<i16>() as f32 / data.facet_a.len() as f32;
    let q_d: f32 = data.facet_d.iter().sum::<i16>() as f32 / data.facet_d.len() as f32;

    return q_a - q_d;
}

pub fn kl_divergence(data: &PreTrainingComputations) -> f32 {
    return data.a_acceptance * (data.a_acceptance / data.d_acceptance).ln()
        + (1.0_f32 - data.a_acceptance)
            * ((1.0_f32 - data.a_acceptance) / (1.0_f32 - data.d_acceptance)).ln();
}

fn ks_kl_div(p_facet: f32, p: f32) -> f32 {
    return p_facet * (p_facet / p).ln()
        + (1.0_f32 - p_facet) * ((1.0_f32 - p_facet) / (1.0_f32 - p)).ln();
}

pub fn jensen_shannon(data: &PreTraining, pre_comp: &PreTrainingComputations) -> f32 {
    let p: f32 = 0.5_f32
        * (data.facet_a.iter().sum::<i16>() as f32 / data.facet_d.len() as f32
            + data.facet_d.iter().sum::<i16>() as f32 / data.facet_a.len() as f32);

    return 0.5 * (ks_kl_div(pre_comp.a_acceptance, p) + ks_kl_div(pre_comp.d_acceptance, p));
}

pub fn lp_norm(data: &PreTrainingComputations) -> f32 {
    return ((data.a_acceptance - data.d_acceptance).powf(2.0)
        + (1.0_f32 - data.a_acceptance - 1.0_f32 - data.d_acceptance).powf(2.0))
    .sqrt();
}

pub fn total_variation_distance(data: &PreTrainingComputations) -> f32 {
    return (data.a_acceptance - data.d_acceptance).abs()
        + ((1.0_f32 - data.a_acceptance) - (1.0_f32 - data.a_acceptance)).abs();
}

pub fn kolmorogv_smirnov(data: &PreTraining) -> f32 {
    let a_0_dist: f32 = data
        .facet_a
        .iter()
        .map(|value| if *value == 0 { 1.0_f32 } else { 0.0_f32 })
        .sum::<f32>()
        / data.facet_a.len() as f32;

    let a_1_dist = data
        .facet_a
        .iter()
        .map(|value| if *value == 1 { 1.0_f32 } else { 0.0_f32 })
        .sum::<f32>()
        / data.facet_a.len() as f32;

    let d_0_dist = data
        .facet_d
        .iter()
        .map(|value| if *value == 0 { 1.0_f32 } else { 0.0_f32 })
        .sum::<f32>()
        / data.facet_d.len() as f32;

    let d_1_dist = data
        .facet_d
        .iter()
        .map(|value| if *value == 1 { 1.0_f32 } else { 0.0_f32 })
        .sum::<f32>()
        / data.facet_d.len() as f32;

    let neg_outcome_diff = (a_0_dist - d_0_dist).abs();
    let pos_outcome_diff = (a_1_dist - d_1_dist).abs();

    if neg_outcome_diff > pos_outcome_diff {
        return pos_outcome_diff;
    } else {
        return neg_outcome_diff;
    }
}

pub fn pre_training_bias(data: PreTraining) -> Result<HashMap<String, f32>, String> {
    let computed_data: PreTrainingComputations = data.generate();
    let mut result: HashMap<String, f32> = HashMap::with_capacity(7);
    result.insert("ClassImbalance".into(), class_imbalance(&data));
    result.insert(
        "DifferenceInProportionOfLabels".into(),
        diff_in_proportion_of_labels(&data),
    );
    result.insert("KlDivergence".into(), kl_divergence(&computed_data));
    result.insert("JsDivergence".into(), jensen_shannon(&data, &computed_data));
    result.insert("LpNorm".into(), lp_norm(&computed_data));
    result.insert(
        "TotalVarationDistance".into(),
        total_variation_distance(&computed_data),
    );
    result.insert("KolmorogvSmirnov".into(), kolmorogv_smirnov(&data));

    Ok(result)
}
