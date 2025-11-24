use super::{post_training_bias, ModelBiasAnalysisReport};
use crate::data_handler::perform_segmentation_model_bias;

pub fn model_bias_analysis_core(
    labeled_features: Vec<i16>,
    labeled_predictions: Vec<i16>,
    labeled_ground_truth: Vec<i16>,
) -> Result<ModelBiasAnalysisReport, String> {
    let post_training_data = perform_segmentation_model_bias(
        labeled_features,
        labeled_predictions,
        labeled_ground_truth,
    )?;
    Ok(post_training_bias(post_training_data))
}
