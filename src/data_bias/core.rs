use super::{DataBiasAnalysisReport, PreTraining, PreTrainingComputations};
use crate::{errors::BiasError, zip};
use std::collections::HashMap;

pub fn data_bias_analysis_core(
    labeled_features: Vec<i16>,
    labeled_ground_truth: Vec<i16>,
) -> Result<DataBiasAnalysisReport, BiasError> {
    let pre_training = perform_segmentation_data_bias(labeled_features, labeled_ground_truth)?;
    Ok(pre_training_bias(pre_training))
}

pub fn perform_segmentation_data_bias(
    feature_values: Vec<i16>,
    ground_truth_values: Vec<i16>,
) -> Result<PreTraining, BiasError> {
    let mut facet_a: Vec<i16> = Vec::new();
    let mut facet_d: Vec<i16> = Vec::new();

    for (feature, ground_truth) in zip!(feature_values, ground_truth_values) {
        match *feature {
            1_i16 => {
                facet_a.push(ground_truth);
            }
            _ => facet_d.push(ground_truth),
        }
    }

    if facet_a.is_empty() | facet_d.is_empty() {
        return Err(BiasError::NoFacetDeviation);
    }

    Ok(PreTraining { facet_a, facet_d })
}

pub fn pre_training_bias(data: PreTraining) -> DataBiasAnalysisReport {
    use super::statistics as stats;
    use crate::metrics::DataBiasMetric as M;
    let computed_data: PreTrainingComputations = data.generate();
    let mut result: HashMap<M, f32> = HashMap::with_capacity(7);
    result.insert(M::ClassImbalance, stats::class_imbalance(&data));
    result.insert(
        M::DifferenceInProportionOfLabels,
        stats::diff_in_proportion_of_labels(&data),
    );
    result.insert(M::KlDivergence, stats::kl_divergence(&computed_data));
    result.insert(
        M::JsDivergence,
        stats::jensen_shannon(&data, &computed_data),
    );
    result.insert(M::LpNorm, stats::lp_norm(&computed_data));
    result.insert(
        M::TotalVariationDistance,
        stats::total_variation_distance(&computed_data),
    );
    result.insert(M::KolmorogvSmirnov, stats::kolmorogv_smirnov(&data));

    result
}
