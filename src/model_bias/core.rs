use super::{DiscretePostTraining, ModelBiasAnalysisReport};
use crate::errors::BiasError;
use crate::metrics::ModelBiasMetric;
use std::collections::HashMap;

pub fn post_training_bias(
    data: &DiscretePostTraining,
) -> Result<ModelBiasAnalysisReport, BiasError> {
    use super::statistics::inner as stats;
    use ModelBiasMetric as M;
    let mut result: HashMap<ModelBiasMetric, f32> = HashMap::with_capacity(12);
    result.insert(
        M::DifferenceInPositivePredictedLabels,
        stats::diff_in_pos_proportion_in_pred_labels(&data.post_training)?,
    );
    result.insert(
        M::DisparateImpact,
        stats::disparate_impact(&data.post_training)?,
    );
    result.insert(
        M::AccuracyDifference,
        stats::accuracy_difference(&data.post_training),
    );
    result.insert(
        M::RecallDifference,
        stats::recall_difference(&data.post_training),
    );
    result.insert(
        M::DifferenceInConditionalAcceptance,
        stats::diff_in_cond_acceptance(&data.post_training)?,
    );
    result.insert(
        M::DifferenceInAcceptanceRate,
        stats::diff_in_acceptance_rate(&data.post_training),
    );
    result.insert(
        M::SpecialityDifference,
        stats::specailty_difference(&data.post_training),
    );
    result.insert(
        M::DifferenceInConditionalRejection,
        stats::diff_in_cond_rejection(&data.post_training),
    );
    result.insert(
        M::DifferenceInRejectionRate,
        stats::diff_in_rejection_rate(&data.post_training),
    );
    result.insert(
        M::TreatmentEquity,
        stats::treatment_equity(&data.post_training),
    );
    result.insert(
        M::ConditionalDemographicDesparityPredictedLabels,
        stats::cond_dem_desp_in_pred_labels(&data.post_training),
    );
    result.insert(M::GeneralizedEntropy, data.ge);

    Ok(result)
}
