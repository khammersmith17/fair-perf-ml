use super::{DataBiasAnalysisReport, PreTraining};
use crate::errors::BiasError;
use std::collections::HashMap;

pub fn data_bias_analysis_core(
    labeled_features: Vec<i16>,
    labeled_ground_truth: Vec<i16>,
) -> Result<DataBiasAnalysisReport, BiasError> {
    let pre_training = PreTraining::new(&labeled_features, &labeled_ground_truth)?;
    Ok(pre_training_bias(pre_training))
}

pub fn pre_training_bias(data: PreTraining) -> DataBiasAnalysisReport {
    use super::statistics as stats;
    use crate::metrics::DataBiasMetric as M;
    let mut result: HashMap<M, f32> = HashMap::with_capacity(7);
    result.insert(M::ClassImbalance, stats::class_imbalance(&data));
    result.insert(
        M::DifferenceInProportionOfLabels,
        stats::diff_in_proportion_of_labels(&data),
    );
    result.insert(M::KlDivergence, stats::kl_divergence(&data));
    result.insert(M::JsDivergence, stats::jensen_shannon(&data));
    result.insert(M::LpNorm, stats::lp_norm(&data));
    result.insert(
        M::TotalVariationDistance,
        stats::total_variation_distance(&data),
    );
    result.insert(M::KolmorogvSmirnov, stats::kolmorogv_smirnov(&data));

    result
}
