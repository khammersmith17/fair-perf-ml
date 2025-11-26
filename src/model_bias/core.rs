use super::{ModelBiasAnalysisReport, PostTrainingComputations, PostTrainingData};
use crate::errors::BiasError;
use crate::metrics::ModelBiasMetric;
use crate::zip;
use std::collections::HashMap;

pub fn perform_segmentation_model_bias(
    feature_values: Vec<i16>,
    prediction_values: Vec<i16>,
    ground_truth_values: Vec<i16>,
) -> Result<PostTrainingData, BiasError> {
    let mut facet_a_trues: Vec<i16> = Vec::new();
    let mut facet_a_scores: Vec<i16> = Vec::new();
    let mut facet_d_scores: Vec<i16> = Vec::new();
    let mut facet_d_trues: Vec<i16> = Vec::new();

    for (feature, (prediction, ground_truth)) in
        zip!(feature_values, prediction_values, ground_truth_values)
    {
        match *feature {
            1_i16 => {
                facet_a_trues.push(ground_truth);
                facet_a_scores.push(*prediction);
            }
            _ => {
                facet_d_trues.push(ground_truth);
                facet_d_scores.push(*prediction);
            }
        }
    }
    if facet_a_trues.is_empty() | facet_d_trues.is_empty() {
        return Err(BiasError::NoFacetDeviation);
    }
    Ok(PostTrainingData {
        facet_a_trues,
        facet_a_scores,
        facet_d_trues,
        facet_d_scores,
    })
}

pub fn model_bias_analysis_core(
    labeled_features: Vec<i16>,
    labeled_predictions: Vec<i16>,
    labeled_ground_truth: Vec<i16>,
) -> Result<ModelBiasAnalysisReport, BiasError> {
    let post_training_data = perform_segmentation_model_bias(
        labeled_features,
        labeled_predictions,
        labeled_ground_truth,
    )?;
    Ok(post_training_bias(post_training_data))
}

pub fn post_training_bias(data: PostTrainingData) -> ModelBiasAnalysisReport {
    use super::statistics as stats;
    use ModelBiasMetric as M;
    let pre_computed_data: PostTrainingComputations = data.general_data_computations();
    let mut result: HashMap<ModelBiasMetric, f32> = HashMap::with_capacity(12);
    result.insert(
        M::DifferenceInPositivePredictedLabels,
        stats::diff_in_pos_proportion_in_pred_labels(&data),
    );
    result.insert(M::DisparateImpact, stats::disparate_impact(&data));
    result.insert(
        M::AccuracyDifference,
        stats::accuracy_difference(&pre_computed_data, &data),
    );
    result.insert(
        M::RecallDifference,
        stats::recall_difference(&pre_computed_data),
    );
    result.insert(
        M::DifferenceInConditionalAcceptance,
        stats::diff_in_cond_acceptance(&data),
    );
    result.insert(
        M::DifferenceInAcceptanceRate,
        stats::diff_in_acceptance_rate(&pre_computed_data),
    );
    result.insert(
        M::SpecialityDifference,
        stats::specailty_difference(&pre_computed_data),
    );
    result.insert(
        M::DifferenceInConditionalRejection,
        stats::diff_in_cond_rejection(&data),
    );
    result.insert(
        M::DifferenceInRejectionRate,
        stats::diff_in_rejection_rate(&pre_computed_data),
    );
    result.insert(
        M::TreatmentEquity,
        stats::treatment_equity(&pre_computed_data),
    );
    result.insert(
        M::ConditionalDemographicDesparityPredictedLabels,
        stats::cond_dem_desp_in_pred_labels(&data),
    );
    result.insert(M::GeneralizedEntropy, stats::generalized_entropy(&data));

    result
}
