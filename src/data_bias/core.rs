use super::{pre_training_bias, DataBiasAnalysisReport};
use crate::data_handler::perform_segmentation_data_bias;

pub fn data_bias_analysis_core(
    labeled_features: Vec<i16>,
    labeled_ground_truth: Vec<i16>,
) -> Result<DataBiasAnalysisReport, String> {
    let pre_training = perform_segmentation_data_bias(labeled_features, labeled_ground_truth)?;
    Ok(pre_training_bias(pre_training))
}
