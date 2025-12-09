use super::{ModelBiasAnalysisReport, PostTraining};
use crate::metrics::ModelBiasMetric;
use std::collections::HashMap;

pub fn post_training_bias(data: &PostTraining) -> ModelBiasAnalysisReport {
    use super::statistics::inner as stats;
    use ModelBiasMetric as M;
    let mut result: HashMap<ModelBiasMetric, f32> = HashMap::with_capacity(12);
    result.insert(
        M::DifferenceInPositivePredictedLabels,
        stats::diff_in_pos_proportion_in_pred_labels(&data),
    );
    result.insert(M::DisparateImpact, stats::disparate_impact(&data));
    result.insert(M::AccuracyDifference, stats::accuracy_difference(data));
    result.insert(M::RecallDifference, stats::recall_difference(&data));
    result.insert(
        M::DifferenceInConditionalAcceptance,
        stats::diff_in_cond_acceptance(&data),
    );
    result.insert(
        M::DifferenceInAcceptanceRate,
        stats::diff_in_acceptance_rate(&data),
    );
    result.insert(M::SpecialityDifference, stats::specailty_difference(&data));
    result.insert(
        M::DifferenceInConditionalRejection,
        stats::diff_in_cond_rejection(&data),
    );
    result.insert(
        M::DifferenceInRejectionRate,
        stats::diff_in_rejection_rate(&data),
    );
    result.insert(M::TreatmentEquity, stats::treatment_equity(&data));
    result.insert(
        M::ConditionalDemographicDesparityPredictedLabels,
        stats::cond_dem_desp_in_pred_labels(&data),
    );
    result.insert(M::GeneralizedEntropy, data.ge);

    result
}
